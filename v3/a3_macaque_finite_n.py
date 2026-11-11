"""The macaque arrays at finite samples (Section 7 and Appendix H of the extended version): the two terms evaluated at the arrays'
own sample counts, beside the mean field the NeurReps version printed.

Three Utah arrays in anesthetized macaque V1 (CRCNS pvc-11, Kohn and Smith 2016; raw/pvc11/spikes_gratings, loaded by
scripts_canonical/macaque_pvc11.py), 12 directions x 200 trials, raw counts. The measured ladder (declared rungs 1, 2, 3, 4, 6, 8, 12
classes, ten-draw floor, seed 42) is asserted equal to data_canonical/macaque_pvc11.json. Nothing is fitted:
  within: W0 mean field (stored), W1 the population overlap identity on the declared rungs, W2 the finite-sample identity (each class
          centred on its own sample mean: q1 = q2 = m - 1, q3 = (m - 1)^2 / m), J the curvature term from Gaussian surrogates through
          the same ladder; within_200 / cm_200 / delta_200 = the measured parts against a 200-draw floor;
  class-mean: mean field (full-set traces carried through the rungs by u = Tr B_k / Tr B, floor at c_full), the finite-sample floor
          term ck(1/(1 + eps_s)), and the exact rung side.
total_fin = W2 + J + cm_fin is the two-term value at the arrays' sample counts (Section 7: -0.346 against the measured -0.357 on the
negative array, same array, 200-draw floor).
Needs raw/pvc11/spikes_gratings/data_monkey{1,2,3}_gratings.mat. Out: v3/a3_macaque_finite_n.json.
Run: VECLIB_MAXIMUM_THREADS=1 python v3/a3_macaque_finite_n.py"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SC = HERE.parent / "scripts_canonical"
K = 12
RUNGS = (1, 2, 3, 4, 6, 8, 12)
N_FLOOR, N_SIM, N_SPLIT = 200, 40, 10


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, SC / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


M = _load("mpvc", "macaque_pvc11.py")
ow = _load("ow", "cr_order_and_withinbin_controls.py")


def moments(X, lab, rng):
    t, m4, E, dof = np.zeros(K), np.zeros(K), [], np.zeros(K)
    for c in range(K):
        Xc = X[lab == c]; R = Xc - Xc.mean(0); d = len(R) - 1; E.append(R); dof[c] = d
        t[c] = (R * R).sum() / d
        m4[c] = (((R * R).sum(1)) ** 2).mean() * (len(R) / d) ** 2
    T = np.zeros((K, K))
    for c in range(K):
        for d in range(c + 1, K):
            T[c, d] = T[d, c] = ((E[c] @ E[d].T) ** 2).sum() / (dof[c] * dof[d])
    for c in range(K):
        Xc = X[lab == c]; acc = 0.0
        for _ in range(N_SPLIT):
            h = np.zeros(len(Xc), int); h[rng.permutation(len(Xc))[::2]] = 1
            Ra = Xc[h == 0] - Xc[h == 0].mean(0); Rb = Xc[h == 1] - Xc[h == 1].mean(0)
            acc += ((Ra @ Rb.T) ** 2).sum() / ((len(Ra) - 1) * (len(Rb) - 1))
        T[c, c] = acc / N_SPLIT
    return {"t": t, "T": T, "m4": m4, "E": E, "dof": dof}


def q_of(m):
    return m - 1.0, m - 1.0, (m - 1.0) ** 2 / m


def lpr_fin(counts, mom, gauss=False):
    cl = [(c, *q_of(m)) for c, m in enumerate(counts) if m >= 2]
    num = sum(q1 * mom["t"][c] for c, q1, _, _ in cl) ** 2
    den = sum(a[1] * b[1] * mom["T"][a[0], b[0]] for a in cl for b in cl)
    for c, q1, q2, q3 in cl:
        t, Tc = mom["t"][c], mom["T"][c, c]
        m4 = t * t + 2 * Tc if gauss else mom["m4"][c]
        den += q2 * (t * t + Tc) + q3 * (m4 - t * t - 2 * Tc)
    return float(np.log(num / den))


def lpr_pop(counts, mom):
    w = np.asarray(counts, float)
    return float(np.log((w @ mom["t"]) ** 2 / (w @ mom["T"] @ w)))


def full_terms(X, lab):
    X = X - X.mean(0)
    cm = np.stack([X[lab == c].mean(0) for c in range(K)]); p = np.bincount(lab, minlength=K) / len(lab)
    n = len(X); Wh = (X - cm[lab]) / np.sqrt(n); Bh = np.sqrt(p)[:, None] * cm
    return dict(w1=float((Wh * Wh).sum()), b1=float((Bh * Bh).sum()), w2=float(((Wh @ Wh.T) ** 2).sum()),
                b2=float(((Bh @ Bh.T) ** 2).sum()), x=float(((Wh @ Bh.T) ** 2).sum()), cm=cm, counts=np.bincount(lab, minlength=K))


def ck(u, T):
    beta = T["b1"] / T["w1"]
    return float(np.log((1 + u * beta) ** 2 / (1 + (u ** 2 * T["b2"] + 2 * u * T["x"]) / T["w2"])))


def us_of(T):
    out = []
    for k in RUNGS:
        cnt = T["counts"][:k]; p = cnt / cnt.sum(); mu = (p[:, None] * T["cm"][:k]).sum(0)
        out.append(float((p * ((T["cm"][:k] - mu) ** 2).sum(1)).sum()) / T["b1"])
    return out


def slope(sizes, y):
    return float(np.polyfit(np.log(sizes), y, 1)[0])


def main():
    stored = json.load(open(HERE.parent / "data_canonical" / "macaque_pvc11.json"))
    out = []
    for mk in (1, 2, 3):
        X, lab, info = M.load_array(mk)
        G = (X @ X.T).astype(np.float64)
        sets = M.sets_from(lab, list(range(K))); sizes = [len(s) for s in sets]
        meas = ow.ladder(G, lab, sets, np.random.default_rng(42))
        arr = [a for a in stored["arrays"] if a.get("monkey") == mk][0] if isinstance(stored["arrays"], list) else stored["arrays"][str(mk)]
        st = arr["raw"]["ladder"] if "raw" in arr else arr["ladder"]
        for k in ("delta", "within", "class_means"):
            assert abs(meas[k] - st[k]) < 1e-9, (mk, k, meas[k], st[k])
        rng = np.random.default_rng(1000 + mk)
        mom = moments(X, lab, rng)
        n_c = np.bincount(lab, minlength=K).astype(float)
        # within: declared rungs (all trials of the first k classes) and the floor (random label-blind subsets)
        lad_f = [lpr_fin([n_c[c] if c < k else 0 for c in range(K)], mom) for k in RUNGS]
        lad_g = [lpr_fin([n_c[c] if c < k else 0 for c in range(K)], mom, gauss=True) for k in RUNGS]
        lad_p = [lpr_pop([n_c[c] if c < k else 0 for c in range(K)], mom) for k in RUNGS]
        flo_f, flo_g, flo_p, flo_e, flo_ck = [], [], [], [], []
        T = full_terms(X, lab); us = us_of(T)
        t_c = mom["t"]
        lad_emp = np.array([ow.parts_from_gram(G[np.ix_(s, s)], lab[s]) for s in sets])
        flo_emp = []
        for s in sizes:
            f, g, p, e, cc, emp = [], [], [], [], [], np.zeros(2)
            for _ in range(N_FLOOR):
                sub = rng.choice(len(lab), s, replace=False); m = np.bincount(lab[sub], minlength=K).astype(float)
                f.append(lpr_fin(m, mom)); g.append(lpr_fin(m, mom, gauss=True)); p.append(lpr_pop(m, mom))
                ok = m >= 1; pp = m / m.sum()
                eps = float((pp[ok] * (1 / m[ok] - 1 / n_c[ok]) * t_c[ok]).sum() / (pp[ok] * t_c[ok]).sum())
                e.append(eps); cc.append(ck(1 / (1 + eps), T))
                emp += np.array(ow.parts_from_gram(G[np.ix_(sub, sub)], lab[sub]))
            flo_f.append(np.mean(f)); flo_g.append(np.mean(g)); flo_p.append(np.mean(p)); flo_e.append(np.mean(e)); flo_ck.append(np.mean(cc))
            flo_emp.append(emp / N_FLOOR)
        flo_emp = np.array(flo_emp)
        W1 = slope(sizes, np.subtract(lad_p, flo_p)); W2 = slope(sizes, np.subtract(lad_f, flo_f)); W2g = slope(sizes, np.subtract(lad_g, flo_g))
        sims = []
        for kk in range(N_SIM):
            Xs = np.zeros_like(X)
            for c in range(K):
                mm = lab == c
                Xs[mm] = rng.normal(size=(mm.sum(), len(mom["E"][c]))) @ mom["E"][c] / np.sqrt(mom["dof"][c])
            sims.append(ow.ladder((Xs @ Xs.T).astype(np.float64), lab, sets, np.random.default_rng(2000 + kk))["within"])
        J = float(np.mean(sims) - W2g)
        lad_cm = lad_emp[:, 0] - lad_emp[:, 1]; flo_cm = flo_emp[:, 0] - flo_emp[:, 1]
        ck_r = [ck(u, T) for u in us]; c_full = ck(1.0, T)
        row = {"monkey": mk, "N": int(X.shape[1]), "per_class": n_c.tolist(), "beta": T["b1"] / T["w1"], "measured": meas,
               "W0_stored": arr.get("raw", arr).get("mean_field"), "W1": W1, "W2": W2, "J": J, "W2J": W2 + J,
               "sim_within_sd": float(np.std(sims, ddof=1)),
               "within_200": slope(sizes, lad_emp[:, 1] - flo_emp[:, 1]), "cm_200": slope(sizes, lad_cm - flo_cm),
               "delta_200": slope(sizes, lad_emp[:, 0] - flo_emp[:, 0]),
               "cm_mf": slope(sizes, np.array(ck_r) - c_full), "cm_fin": slope(sizes, np.array(ck_r) - np.array(flo_ck)),
               "cm_rung_exact_fin_floor": slope(sizes, lad_cm - np.array(flo_ck)),
               "rung_side": slope(sizes, lad_cm) - slope(sizes, ck_r), "floor_side_emp": -slope(sizes, flo_cm),
               "floor_side_fin": -slope(sizes, flo_ck), "eps": flo_e, "u": us, "c_full": c_full, "sizes": sizes}
        row["total_mf"] = (row["W0_stored"] or np.nan) + row["cm_mf"]; row["total_fin"] = row["W2J"] + row["cm_fin"]
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                          if k not in ("eps", "u", "sizes", "per_class")}), flush=True)
        out.append(row)
    json.dump(out, open(HERE / "a3_macaque_finite_n.json", "w"), indent=1, default=float)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
