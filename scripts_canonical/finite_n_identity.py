"""The overlap identity at finite samples: why the mean-field overlap law of Fig 3A misses the lattice points, and what gives every
within-class part. Every system in Fig 3A uses one estimator: rungs of 1, 2, 3, 4, 6, 8 classes in angular order, within part = least-
squares slope on log size of [log PR(W) of the rung's pooled within-class covariance - mean log PR(W) of same-size random subsets with
their own labels]. The overlap law ln[8/(1+7 sigma)]/ln 8 is a population statement. This module evaluates the overlap identity
(App K) rung by rung, in the population limit (W1) and at the rung's sample counts (W2), in both the declared ladder and its floor:

   PR_fin = (sum_c nu_c t_c)^2 / [ sum_cd nu_c nu_d T_cd + sum_c nu_c (m4_c - T_cc) ],   nu_c = m_c - 1 samples of class c,

t_c = Tr Sigma_c, T_cd = Tr(Sigma_c Sigma_d), m4_c = E||x - mu_c||^4 (unbiased estimates from one ensemble's samples); for one Gaussian
class it is Eq. 2. Nothing is fitted. W2x takes the moments from independent ensembles (a prediction); W2g uses Gaussian fourth moments.
`validate` checks PR_fin against sampled pooled PR on synthetic Gaussian and bimodal-sign classes. main() scores the arms on the outputs of
finite_n_lattice_regen.py (temperature means against 2 SE of their 8 ensembles; nematic cells against 2 SE of a cell mean, 0.026).
Run: python -X faulthandler finite_n_identity.py [validate]   Out: data_canonical/finite_n_lattice.json (finite_n_validate.json)"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
BIG = HERE.parent / "data_canonical" / "finite_n"
BIN_COUNTS = (1, 2, 3, 4, 6, 8)
N_COMP = 200          # random floor compositions per rung
N_SPLIT = 10          # random split halves for the diagonal T_cc


# ---------------------------------------------------------------- moments
def _blocks(R_list):
    """Callable (c, d) -> ||R_c R_d^T||_F^2 = Tr(R_c^T R_c R_d^T R_d), in the cheaper of sample or feature space."""
    D = R_list[0].shape[1]
    if max(len(R) for R in R_list) >= D:
        S = [R.T @ R for R in R_list]
        return lambda c, d: float((S[c] * S[d]).sum())
    return lambda c, d: float(((R_list[c] @ R_list[d].T) ** 2).sum())


def moments(F, lab, n_class=8, seed=0):
    """Unbiased per-class moments: t_c = Tr Sigma_c, T_cd = Tr(Sigma_c Sigma_d), m4_c = E||x - mu_c||^4; plus the record
    K = T / outer(t, t) and Kn = K / sqrt(diag diag). c != d from full-class covariances (independent classes); c = d from split
    halves each centred on its own mean (mean over N_SPLIT splits)."""
    rng = np.random.default_rng(seed)
    R, nu = [], np.zeros(n_class)
    for c in range(n_class):
        Fc = np.asarray(F[lab == c], np.float64)
        R.append(Fc - Fc.mean(0)); nu[c] = len(Fc) - 1
    t = np.array([(r * r).sum() for r in R]) / nu
    m4 = np.array([((r * r).sum(1) ** 2).mean() for r in R]) * ((nu + 1) / nu) ** 2
    blk = _blocks(R)
    T = np.zeros((n_class, n_class))
    for c in range(n_class):
        for d in range(c + 1, n_class):
            T[c, d] = T[d, c] = blk(c, d) / (nu[c] * nu[d])
    for c in range(n_class):
        Fc = np.asarray(F[lab == c], np.float64); acc = 0.0
        for _ in range(N_SPLIT):
            p = rng.permutation(len(Fc)); A, B = Fc[p[: len(Fc) // 2]], Fc[p[len(Fc) // 2:]]
            A = A - A.mean(0); B = B - B.mean(0)
            if A.shape[1] <= len(A):
                v = float(((A.T @ A) * (B.T @ B)).sum())
            else:
                v = float(((A @ B.T) ** 2).sum())
            acc += v / ((len(A) - 1) * (len(B) - 1))
        T[c, c] = acc / N_SPLIT
    K = T / np.outer(t, t)
    Kn = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))
    return {"t": t, "T": T, "m4": m4, "K": K, "Kn": Kn, "n": nu + 1}


def pool_moments(ms):
    """Moments of one population from independent ensembles: sample-count-weighted means (the cross-fitted inputs, W2x)."""
    w = np.array([m["n"] for m in ms], float)                       # ensembles x classes
    out = {}
    for k in ("t", "m4"):
        out[k] = (np.array([m[k] for m in ms]) * w).sum(0) / w.sum(0)
    ww = np.einsum("ec,ed->ecd", w, w)
    out["T"] = (np.array([m["T"] for m in ms]) * ww).sum(0) / ww.sum(0)
    return out


# ---------------------------------------------------------------- the identity, population and finite n
def pr_pop(w, mom):
    """App K: 1/PR(W_A) = sum alpha_c alpha_d K_cd with class weights w (sample counts)."""
    return float((w @ mom["t"]) ** 2 / (w @ mom["T"] @ w))


def pr_fin(m, mom, gauss=False):
    """Finite-n form: m = samples per class in the subset (each class centred on its own sample mean)."""
    nu = np.maximum(np.asarray(m, float) - 1, 0)
    m4 = mom["t"] ** 2 + 2 * np.diag(mom["T"]) if gauss else mom["m4"]
    return float((nu @ mom["t"]) ** 2 / (nu @ mom["T"] @ nu + nu @ (m4 - np.diag(mom["T"]))))


def rung_sizes(counts):
    return sorted({int(np.sum(counts[:k])) for k in BIN_COUNTS if np.sum(counts[:k]) >= 10})


def compositions(lab, size, rng, n=N_COMP, n_class=8):
    return np.array([np.bincount(lab[rng.choice(len(lab), size, replace=False)], minlength=n_class) for _ in range(n)])


def slope(sizes, ys):
    return float(np.polyfit(np.log(sizes), ys, 1)[0])


def arms(mom, counts, lab, sigma, seed, mom_x=None):
    """Within-part predictions for one ensemble. counts = the scored ensemble's class counts, lab its labels (floor compositions).
    mom = its own moments; mom_x = independent moments (W2x). Returns per-rung curves and slopes."""
    counts = np.asarray(counts); sizes = rung_sizes(counts)
    ks = [next(k for k in BIN_COUNTS if np.sum(counts[:k]) == s) for s in sizes]
    rng = np.random.default_rng(seed)
    comps = [compositions(lab, s, rng) for s in sizes]
    out = {"sizes": sizes, "rung_classes": ks}
    lad_w = [np.where(np.arange(8) < k, counts, 0) for k in ks]
    for name, M, g in (("W2", mom, False), ("W2g", mom, True), ("W2x", mom_x, False)):
        if M is None:
            continue
        lad = [np.log(pr_fin(m, M, g)) for m in lad_w]
        flo = [float(np.mean([np.log(pr_fin(m, M, g)) for m in C])) for C in comps]
        out[f"{name}_ladder"], out[f"{name}_floor"] = lad, flo
        out[name] = slope(sizes, np.subtract(lad, flo))
        out[f"{name}_floor_slope"] = slope(sizes, flo)
    lad1 = [np.log(pr_pop(m, mom)) for m in lad_w]
    flo1 = [float(np.mean([np.log(pr_pop(m, mom)) for m in C])) for C in comps]
    out.update(W1_ladder=lad1, W1_floor=flo1, W1=slope(sizes, np.subtract(lad1, flo1)), W1_floor_slope=slope(sizes, flo1))
    out["ladder_finite_only"] = slope(sizes, np.subtract(out["W2_ladder"], flo1))      # finite-n ladder, population floor
    out["floor_finite_only"] = slope(sizes, np.subtract(lad1, out["W2_floor"]))       # population ladder, finite-n floor
    out["MFr"] = slope(sizes, [np.log(k / (1 + (k - 1) * sigma)) for k in ks])
    out["W0"] = float(np.log(8 / (1 + 7 * sigma)) / np.log(8))
    # population-limit dimensions for the convergence rule
    out["PR_W_pop"] = pr_pop(counts.astype(float), mom)
    out["PR_class_pop"] = [float(mom["t"][c] ** 2 / mom["T"][c, c]) for c in range(8)]
    return out


# ---------------------------------------------------------------- synthetic validation of pr_fin (before any lattice data)
def validate(n_rep=400):
    """Monte Carlo check of E-ratio PR_fin against sampled pooled within-class PR for Gaussian and bimodal-sign classes."""
    rng = np.random.default_rng(1)
    D, res = 40, []
    base = rng.normal(size=(8, D, 6)) * np.linspace(1, 0.3, 6)
    for kind in ("gauss", "bimodal"):
        for m_c in (3, 8, 25):
            m = np.full(8, m_c); m[:3] = m_c + 2
            Sig = []
            for c in range(8):
                Sig.append(base[c] @ base[c].T + 0.05 * np.eye(D))
            def draw(c, k):
                z = rng.normal(size=(k, D)) @ np.linalg.cholesky(Sig[c]).T
                if kind == "bimodal":                   # a random +- sign along the class's top direction (the polar-sign analogue)
                    u = np.linalg.eigh(Sig[c])[1][:, -1] * 3.0
                    z = z + np.outer(rng.choice([-1, 1], k), u)
                return z
            # population moments by a large sample
            big = [draw(c, 20000) for c in range(8)]
            mom = {"t": np.array([((b - b.mean(0)) ** 2).sum(1).mean() for b in big]),
                   "T": np.array([[float((np.cov(big[c].T, bias=True) * np.cov(big[d].T, bias=True)).sum()) for d in range(8)] for c in range(8)]),
                   "m4": np.array([(((b - b.mean(0)) ** 2).sum(1) ** 2).mean() for b in big])}
            lp = []
            for _ in range(n_rep):
                R = np.concatenate([(lambda z: z - z.mean(0))(draw(c, m[c])) for c in range(8)])
                C = R.T @ R
                lp.append(np.log(np.trace(C) ** 2 / (C * C).sum()))
            pred = np.log(pr_fin(m, mom)); pred_g = np.log(pr_fin(m, mom, gauss=True)); pop = np.log(pr_pop(m.astype(float), mom))
            res.append({"kind": kind, "m_c": m_c, "mean_logPR": float(np.mean(lp)), "se": float(np.std(lp) / np.sqrt(n_rep)),
                        "pred_fin": float(pred), "pred_fin_gauss": float(pred_g), "pop": float(pop)})
            print(f"validate {kind:7s} m_c {m_c:3d}: sampled E log PR {np.mean(lp):.4f} (se {np.std(lp) / np.sqrt(n_rep):.4f}) | "
                  f"fin {pred:.4f} fin-gauss {pred_g:.4f} pop {pop:.4f}", flush=True)
    return res


# ---------------------------------------------------------------- scoring (every criterion printed)
def _load_group(group):
    rows = json.load(open(DATA / f"finite_n_regen_{group}.json"))
    for r in rows:
        z = np.load(BIG / f"{group}_{r['key']}.npz")
        r["T"] = float(next(p[1:] for p in r["key"].split("_") if p.startswith("T"))) if "T" in r["key"] else None   # temperature
        r["lab"] = z["lab"]; r["mom"] = {"t": z["t"], "T": z["T"], "m4": z["m4"]}
        r["mom"]["n"] = np.bincount(r["lab"], minlength=8)
    return rows


def _score_rows(rows, xfit, seed0):
    out = []
    for i, r in enumerate(rows):
        a = arms(r["mom"], r["counts"], r["lab"], r["sigma_H1"], seed0 + i, mom_x=xfit(r))
        assert abs(a["W0"] - r["pred_within_printed"]) < 1e-9, (r["key"], a["W0"], r["pred_within_printed"])
        assert a["sizes"] == r["sizes"], (r["key"], a["sizes"], r["sizes"])
        a.update({k: r[k] for k in r if k not in ("lab", "mom", "t", "T", "T_cd", "m4", "K", "Kn", "K_om")})
        a["T"] = r["T"]                                                   # the temperature (restored by _load_group)
        out.append(a)
    return out


ARMS = ("W0", "MFr", "W1", "ladder_finite_only", "floor_finite_only", "W2", "W2g", "W2x")


def score_polar():
    rows = _load_group("polar")
    def xfit(r):
        return pool_moments([q["mom"] for q in rows if q["T"] == r["T"] and q["ens"] != r["ens"]])
    S = _score_rows(rows, xfit, 11_000)
    summ = {}
    for T in (0.60, 0.70, 0.80, 0.85):
        g = [s for s in S if s["T"] == T]; m = np.array([s["measured_within"] for s in g])
        bar = 2 * m.std(ddof=1) / np.sqrt(len(m))
        summ[f"{T:.2f}"] = {"measured_mean": float(m.mean()), "bar_2se": float(bar),
                            **{f"err_{k}": float(np.mean([s[k] for s in g]) - m.mean()) for k in ARMS},
                            "floor_slope_measured": float(np.mean([s["floor_slope_noise"] for s in g])),
                            "floor_slope_W2": float(np.mean([s["W2_floor_slope"] for s in g])),
                            "PR_W_pop": float(np.mean([s["PR_W_pop"] for s in g])),
                            "PR_class_pop": float(np.mean([np.mean(s["PR_class_pop"]) for s in g])),
                            "per_class": float(np.mean([np.mean(s["counts"]) for s in g]))}
    e = lambda T, k: summ[f"{T:.2f}"][f"err_{k}"]
    b = lambda T: summ[f"{T:.2f}"]["bar_2se"]
    crit = {"P1_W1_outside_bar_T0.80": abs(e(0.8, "W1")) > b(0.8), "P1_W1_outside_bar_T0.85": abs(e(0.85, "W1")) > b(0.85),
            "P2_W2_within_bar_all4": all(abs(e(T, "W2")) <= b(T) for T in (0.6, 0.7, 0.8, 0.85)),
            "P2_W2x_within_bar_count": int(sum(abs(e(T, "W2x")) <= b(T) for T in (0.6, 0.7, 0.8, 0.85))),
            "P3_floor_carries_T0.80": abs(e(0.8, "floor_finite_only") - e(0.8, "W0")) > abs(e(0.8, "ladder_finite_only") - e(0.8, "W0")),
            "P3_floor_carries_T0.85": abs(e(0.85, "floor_finite_only") - e(0.85, "W0")) > abs(e(0.85, "ladder_finite_only") - e(0.85, "W0"))}
    crit["P1_met"] = crit["P1_W1_outside_bar_T0.80"] and crit["P1_W1_outside_bar_T0.85"]
    crit["P2_met"] = crit["P2_W2_within_bar_all4"] and crit["P2_W2x_within_bar_count"] >= 3
    crit["P3_met"] = crit["P3_floor_carries_T0.80"] and crit["P3_floor_carries_T0.85"]
    m = np.array([s["measured_within"] for s in S])
    desc = {f"pearson_all32_{k}": float(np.corrcoef([s[k] for s in S], m)[0, 1]) for k in ("W2", "W2x", "W1")}
    for k in ("W2", "W2x"):
        desc[f"pearson_within_T_{k}"] = [float(np.corrcoef([s[k] for s in S if s["T"] == T], [s["measured_within"] for s in S if s["T"] == T])[0, 1])
                                         for T in (0.60, 0.70, 0.80, 0.85)]
    return S, summ, crit, desc


def score_nem():
    rows = _load_group("nem")
    def xfit(r):
        big = max(q["per_class"] for q in rows if q["L"] == r["L"] and q["T"] == r["T"])
        return next(q["mom"] for q in rows if q["L"] == r["L"] and q["T"] == r["T"] and q["per_class"] == big and q["ens"] != r["ens"])
    S = _score_rows(rows, xfit, 12_000)
    cells = {}
    for s in S:
        cells.setdefault((s["L"], s["T"], s["per_class"]), []).append(s)
    bar = 0.026          # 2 SE of a cell mean from the paired ensembles' pooled SD
    cs = []
    for (L, T, per), g in sorted(cells.items()):
        m = np.mean([s["measured_within"] for s in g])
        cs.append({"L": L, "T": T, "per_class": per, "measured": float(m), **{f"err_{k}": float(np.mean([s[k] for s in g]) - m) for k in ARMS},
                   "floor_slope_measured": float(np.mean([s["floor_slope_noise"] for s in g])),
                   "floor_slope_W2": float(np.mean([s["W2_floor_slope"] for s in g])),
                   "PR_W_pop": float(np.mean([s["PR_W_pop"] for s in g])), "PR_class_pop": float(np.mean([np.mean(s["PR_class_pop"]) for s in g])),
                   "n_over_PRclass_pop": float(per / np.mean([np.mean(s["PR_class_pop"]) for s in g])),
                   "n_over_PRclass_printed": float(per / np.mean([s["PR_class_printed"] for s in g])) if "PR_class_printed" in g[0] else None})
    crit = {"bar": bar, "P1_W2_cells_within": int(sum(abs(c["err_W2"]) <= bar for c in cs)),
            "P1_W2x_cells_within": int(sum(abs(c["err_W2x"]) <= bar for c in cs)), "n_cells": len(cs)}
    crit["P1_met"] = crit["P1_W2_cells_within"] >= 9 and crit["P1_W2x_cells_within"] >= 8
    return S, cs, crit


def score_sw():
    rows = _load_group("sw")
    def xfit(r):
        return pool_moments([q["mom"] for q in rows if q["per_class"] == 200 and q["seed"] != r["seed"]])
    S = _score_rows(rows, xfit, 13_000)
    summ = {}
    for per in (25, 200):
        g = [s for s in S if s["per_class"] == per]; m = np.array([s["measured_within"] for s in g])
        summ[str(per)] = {"measured_mean": float(m.mean()), "bar_2se": float(2 * m.std(ddof=1) / np.sqrt(len(m))),
                          **{f"err_{k}": float(np.mean([s[k] for s in g]) - m.mean()) for k in ARMS},
                          "floor_slope_measured": float(np.mean([s["floor_slope_noise"] for s in g])),
                          "floor_slope_W2": float(np.mean([s["W2_floor_slope"] for s in g])),
                          "PR_W_pop": float(np.mean([s["PR_W_pop"] for s in g])), "PR_class_pop": float(np.mean([np.mean(s["PR_class_pop"]) for s in g]))}
    crit = {"P2_W2_within_bar_25": abs(summ["25"]["err_W2"]) <= summ["25"]["bar_2se"],
            "P2_W2_within_bar_200": abs(summ["200"]["err_W2"]) <= summ["200"]["bar_2se"]}
    crit["P2_met"] = crit["P2_W2_within_bar_25"] and crit["P2_W2_within_bar_200"]
    return S, summ, crit


def _clean(S):
    return [{k: v for k, v in s.items() if not k.endswith(("_ladder", "_floor"))} | {k: v for k, v in s.items() if k.endswith(("_ladder", "_floor"))}
            for s in S]


def main():
    out = {}
    for group, fn, n_full in (("sw", score_sw, 16), ("nem", score_nem, 24), ("polar", score_polar, 32)):
        f = DATA / f"finite_n_regen_{group}.json"
        if not f.exists() or len(json.load(open(f))) < n_full:
            print(f"{group}: regeneration incomplete, not scored", flush=True); continue
        res = fn()
        out[group] = {"rows": _clean(res[0]), **dict(zip(("summary", "criteria", "descriptive")[: len(res) - 1], res[1:]))}
        print(f"==== {group}", flush=True)
        for k, v in (res[1].items() if isinstance(res[1], dict) else enumerate(res[1])):
            print(" ", k, {q: (round(w, 4) if isinstance(w, float) else w) for q, w in v.items()}, flush=True)
        for r in res[2:]:
            print("  CRITERIA/DESC", json.dumps(r), flush=True)
    json.dump(out, open(DATA / "finite_n_lattice.json", "w"), indent=1, default=float)
    print("DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "validate":
        json.dump(validate(), open(DATA / "finite_n_validate.json", "w"), indent=1)
        print("DONE", flush=True)
    else:
        main()
