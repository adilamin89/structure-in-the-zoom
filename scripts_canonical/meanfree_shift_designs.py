"""How the mean-free shift depends on the design.
(a) Is the shift's design dependence a population property? Label-independent halves S1 (odd / even presentation in time).
    Imaging (8 recordings, raw input as Table 9) and Allen VISp (32 sessions, raw counts): the shift for 16 origin x traversal
    ladders, 20 random class orders (rng 5), the orientation ladder (c and c + 4 merged, rungs 1-4) and the coarse schedule (rungs
    2, 4, 8), on both halves; per population the Pearson r across designs of half A vs half B, the across-design SD on each half
    and the half-to-half SD of a design ((A - B) / sqrt 2).
(b) Finite-size inversion (imaging; DeepSeek Q1 form). From half A projected off its own class-mean span: per class t_c = Tr W_c,
    T_cc (independent split halves), T_cd, kappa_c (fourth moment). For a set of classes with residual dof m_c the ratio of
    expectations PR = (sum m_c t_c)^2 / {sum_c [m_c^2 T_cc + m_c (t_c^2 + T_cc + kappa_c)] + sum_{c != d} m_c m_d T_cd}; a floor
    subset of n_s trials is the equal-weight mixture with dof n_s - 1 and kappa_mix = mean_c (t_c^2 + 2 T_cc + kappa_c) - t^2 - 2 T.
    Predicted mean-free shift = slope over log rung size of (declared - floor) log PR. Measured on half B subsampled to n = 16, 32,
    64, 128 and the largest equal count per class (5 draws), its own span removed (R1c design). Baseline: population PRs without
    the finite-n terms (constant in n).
Out: data_canonical/meanfree_shift_designs.json"""
import importlib.util
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
DATA = HERE.parent / "data_canonical"
OUT = DATA / "meanfree_shift_designs.json"
RUNGS = (1, 2, 3, 4, 6, 8)
NS = (16, 32, 64, 128)
N_DRAW = 5


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def designs(rng_seed=5):
    out = [(f"origin{o}_{'fwd' if s > 0 else 'rev'}", [(o + s * j) % 8 for j in range(8)], RUNGS, False) for o in range(8) for s in (1, -1)]
    rng = np.random.default_rng(rng_seed)
    out += [(f"random{i}", rng.permutation(8).tolist(), RUNGS, False) for i in range(20)]
    out += [("orientation", [0, 1, 2, 3], (1, 2, 3, 4), True), ("coarse", list(range(8)), (2, 4, 8), False)]
    return out


def design_shift(ow, G, y, order, rungs, merge):
    yy = y % 4 if merge else y
    m = [np.where(yy == b)[0] for b in range(yy.max() + 1)]
    sets = [np.concatenate([m[j] for j in order[:c]]) for c in rungs]
    return ow.ladder(G, yy, sets, np.random.default_rng(42))["delta"]


def replication(ow, X, y, h):
    D = designs()
    res = {}
    for hv in (0, 1):
        sel = np.where(h == hv)[0]
        G = (X[sel] @ X[sel].T).astype(np.float64)
        res[hv] = [design_shift(ow, G, y[sel], o, r, m) for _, o, r, m in D]
    a, b = np.array(res[0]), np.array(res[1])
    return {"designs": [d[0] for d in D], "half0": a.tolist(), "half1": b.tolist(), "r": float(np.corrcoef(a, b)[0, 1]),
            "sd_across_designs": [float(a.std()), float(b.std())], "sd_half_to_half": float(np.std((a - b) / np.sqrt(2)))}


# --------------------------------------------------------------------------------------------- (b) finite-size inversion
def centred(B):
    B = B - B.mean(1, keepdims=True)
    return B - B.mean(0, keepdims=True)


def population_terms(G, y):
    """Per-class t_c, T_cd (T_cc from split halves) and kappa_c from the Gram matrix of projected data."""
    idx = [np.where(y == c)[0] for c in range(8)]
    t = np.zeros(8)
    kap = np.zeros(8)
    T = np.zeros((8, 8))
    for c, I in enumerate(idx):
        Gc = centred(G[np.ix_(I, I)])
        n = len(I)
        t[c] = np.trace(Gc) / (n - 1)
        a, b = I[0::2], I[1::2]
        Bab = centred(G[np.ix_(a, b)])
        T[c, c] = (Bab * Bab).sum() / ((len(a) - 1) * (len(b) - 1))
        q = np.diag(Gc) * n / (n - 1)                      # ||r_i||^2, residual shrinkage corrected
        kap[c] = (q ** 2).mean() - t[c] ** 2 - 2 * T[c, c]
    for c in range(8):
        for d in range(c + 1, 8):
            B = centred(G[np.ix_(idx[c], idx[d])])
            T[c, d] = T[d, c] = (B * B).sum() / ((len(idx[c]) - 1) * (len(idx[d]) - 1))
    return t, T, kap


def log_pr_set(t, T, kap, classes, m):
    num = (m * t[classes]).sum() ** 2
    den = sum(m ** 2 * T[c, c] + m * (t[c] ** 2 + T[c, c] + kap[c]) for c in classes)
    den += sum(m * m * T[c, d] for c in classes for d in classes if c != d)
    return np.log(num / den)


def log_pr_floor(t, T, kap, ns):
    tb, Tb = t.mean(), T.mean()
    km = np.mean(t ** 2 + 2 * np.diag(T) + kap) - tb ** 2 - 2 * Tb
    nu = ns - 1
    return np.log((nu * tb) ** 2 / (nu ** 2 * Tb + nu * (tb ** 2 + Tb + km)))


def predict(t, T, kap, n):
    sizes = [k * n for k in RUNGS]
    yd = [log_pr_set(t, T, kap, list(range(k)), n - 1) for k in RUNGS]
    yf = [log_pr_floor(t, T, kap, s) for s in sizes]
    pop_d = [np.log(t[:k].sum() ** 2 / T[:k, :k].sum()) for k in RUNGS]
    pop_f = np.log(t.sum() ** 2 / T.sum())
    x = np.log(sizes)
    return float(np.polyfit(x, np.array(yd) - np.array(yf), 1)[0]), float(np.polyfit(x, np.array(pop_d) - pop_f, 1)[0])


def run_imaging(fp):
    warnings.filterwarnings("ignore")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    R1 = _load("h4r1", "meanfree_shift.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    X /= X.std() + 1e-9
    y = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    tt = np.asarray(dat["stimtimes"], float).ravel()
    del dat
    assert np.all(np.diff(tt) > 0)
    h = np.arange(len(y)) % 2
    row = {"name": fp.stem, "replication": replication(ow, X, y, h)}
    print(f"{fp.stem[:32]} replication r {row['replication']['r']:.3f} sd designs {row['replication']['sd_across_designs']} "
          f"half-to-half {row['replication']['sd_half_to_half']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    A = np.where(h == 0)[0]
    XA, yA = X[A], y[A]
    Q = R1.mean_subspace(XA, yA)
    ZA = XA - (XA @ Q.astype(np.float32)) @ Q.T.astype(np.float32)
    t, T, kap = population_terms((ZA @ ZA.T).astype(np.float64), yA)
    row["terms"] = {"t": t.tolist(), "T": T.tolist(), "kappa": kap.tolist()}
    Bi = np.where(h == 1)[0]
    XB, yB = X[Bi], y[Bi]
    nmax = int(np.bincount(yB, minlength=8).min())
    inv = []
    for n in NS + (nmax,):
        if n > nmax:
            continue
        pred, pop = predict(t, T, kap, n)
        meas = []
        for d in range(N_DRAW if n < nmax else 1):
            rng = np.random.default_rng(31 * n + d)
            sub = np.concatenate([rng.choice(np.where(yB == c)[0], n, replace=False) for c in range(8)])
            Xs, ys = XB[sub], yB[sub]
            Qs = R1.mean_subspace(Xs, ys)
            Ps = (Xs @ Qs.astype(np.float32)).astype(np.float64)
            G = (Xs @ Xs.T).astype(np.float64) - Ps @ Ps.T
            meas.append(ow.ladder(G, ys, R1.rung_sets(ys), np.random.default_rng(42))["delta"])
        inv.append({"n": n, "pred": pred, "pop_baseline": pop, "measured": float(np.mean(meas)), "measured_draws": meas})
        print(f"{fp.stem[:32]} n {n}: pred {pred:+.3f} (population {pop:+.3f}) measured {np.mean(meas):+.3f} ({time.time() - t0:.0f}s)", flush=True)
    row["inversion"] = inv
    return row


def run_allen(s):
    E = _load("E", "allen_engine.py")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    S = E.load_session(s["session"])
    idx = E.populations(S).get("VISp")
    if idx is None:
        return None
    c = np.load(E.COUNTS / f"session_{s['session']}.npz")
    order = np.argsort(c["start"], kind="stable")
    X, y = S["X"][order][:, idx], S["lab"][order]
    rep = replication(ow, X, y, np.arange(len(y)) % 2)
    print(s["session"], f"Allen replication r {rep['r']:.3f} sd designs {np.round(rep['sd_across_designs'], 4)} half-to-half {rep['sd_half_to_half']:.4f}", flush=True)
    return {"session": s["session"], "replication": rep}


def main():
    files = sorted(RAW.glob("gratings*.npy"))
    sess = json.load(open(DATA / "allen_expansion_all_sessions.json"))["results"]
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        im = list(ex.map(run_imaging, files))
        al = [r for r in ex.map(run_allen, sess) if r is not None]
    rr_i = [r["replication"]["r"] for r in im]
    rr_a = [r["replication"]["r"] for r in al]
    ratio = lambda R: [np.mean(r["replication"]["sd_across_designs"]) / r["replication"]["sd_half_to_half"] for r in R]  # noqa: E731
    cells = [(r["name"], x) for r in im for x in r["inversion"]]
    pred = np.array([x["pred"] for _, x in cells])
    meas = np.array([x["measured"] for _, x in cells])
    rng_meas = [max(x["measured"] for x in r["inversion"]) - min(x["measured"] for x in r["inversion"]) for r in im]
    sc = {"a_imaging_r_median": float(np.median(rr_i)), "a_imaging_r": rr_i,
          "a_imaging_sd_ratio": ratio(im), "a_imaging_sd_ratio_gt2": int(sum(v > 2 for v in ratio(im))),
          "a_allen_r_median": float(np.median(rr_a)), "a_allen_sd_ratio_median": float(np.median(ratio(al))),
          "a_allen_sd_ratio_gt2": int(sum(v > 2 for v in ratio(al))), "a_allen_n": len(al),
          "b_pearson": float(np.corrcoef(pred, meas)[0, 1]), "b_median_abs_err": float(np.median(np.abs(pred - meas))),
          "b_median_across_n_range": float(np.median(rng_meas)),
          "b_baseline_pearson": float(np.corrcoef([x["pop_baseline"] for _, x in cells], meas)[0, 1]),
          "b_baseline_median_abs_err": float(np.median(np.abs(np.array([x["pop_baseline"] for _, x in cells]) - meas)))}
    print("SCORE", json.dumps(sc, indent=1), flush=True)
    json.dump({"imaging": im, "allen": al, "score": sc}, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        r = run_imaging(sorted(RAW.glob("gratings_drifting_GT3*.npy"))[0])
        print(json.dumps(r["inversion"], indent=0)[:1500])
    else:
        main()
