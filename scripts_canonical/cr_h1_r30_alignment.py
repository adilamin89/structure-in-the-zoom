"""H1-R30 (Oct 22): what sets the alignment a = xi PR_W (B1c: imaging 1.1, Allen 0.25, and it carries the class-mean sign)?
Hypothesis (run57's lambda, written before this run): fluctuations are a gain on the response m_c = mu0 + t_c (mu0 the
class-averaged mean, t_c the tuned part). The gain direction is m_c; when the untuned baseline dominates (|mu0| >> |t|, low
modulation depth, spike counts with high baseline rates) the gain fluctuations lie along mu0, a direction the class means'
covariance B (built from t) does not contain, so a is low; when the baseline is small (deconvolved calcium), the gain lies along
t, inside B, and a ~ 1.
Measured at matched N (64, 128, 256; ten draws, B1 / B1c seeds) in imaging (8 recordings) and Allen VISp (TF-residualized), raw:
  depth    rho_m = |mu0|^2 / mean_c |t_c|^2 (baseline relative to tuning)
  mode     cos^2 between W's top eigenvector and mu0-hat, and the share of Tr W along mu0-hat relative to chance (N u'Wu/Tr W)
  removal  a and the full-set class-mean term c after projecting every residual orthogonal to mu0-hat (W -> P W P, P = I - uu')
Macaque (pvc-11, 3 arrays, all units): the same, plus a and c for count windows 0.28-0.53, 0.28-0.78, 0.28-1.28 and 0-1.28 s.
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler cr_h1_r30_alignment.py   Out: data_canonical/cr_h1_r30.json"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402

RAW = HERE.parent / "raw"
PVC = HERE.parent / "raw" / "pvc11" / "spikes_gratings"
OUT = HERE.parent / "data_canonical" / "cr_h1_r30.json"
NS, DRAWS = (64, 128, 256), 10


def stats_of(X, lab, K):
    """Raw-frame alignment statistics, with and without the mu0 direction in the fluctuations."""
    M = np.stack([X[lab == c].mean(0) for c in range(K)]); mu0 = M.mean(0); T = M - mu0
    R = X - M[lab]; n = len(X); p = np.bincount(lab, minlength=K) / n
    u = mu0 / (np.linalg.norm(mu0) + 1e-30)
    def terms(Rr):
        Wh = Rr / np.sqrt(n); Bh = np.sqrt(p)[:, None] * T
        w1, b1 = (Wh * Wh).sum(), (Bh * Bh).sum()
        w2 = ((Wh @ Wh.T) ** 2).sum(); b2 = ((Bh @ Bh.T) ** 2).sum(); x = ((Wh @ Bh.T) ** 2).sum()
        beta, prw = b1 / w1, w1 ** 2 / w2
        return {"a": float(x / (w1 * b1) * prw), "beta": float(beta), "PR_W": float(prw), "PR_B": float(b1 ** 2 / b2),
                "c": float(np.log((1 + beta) ** 2 / (1 + (b2 + 2 * x) / w2)))}
    raw = terms(R); proj = terms(R - np.outer(R @ u, u))
    Wh = R / np.sqrt(n); G = Wh @ Wh.T; w, V = np.linalg.eigh(G)
    top = Wh.T @ V[:, -1]; top /= np.linalg.norm(top)
    trW = (Wh * Wh).sum(); along = float(((Wh @ u) ** 2).sum())
    return {"rho_m": float((mu0 @ mu0) / np.mean((T * T).sum(1))), "cos2_top_mu0": float((top @ u) ** 2),
            "mu0_share_over_chance": float(along / (trW / X.shape[1])), "raw": raw, "proj_mu0": proj}


def pvc_counts(m, win):
    d = loadmat(PVC / f"data_monkey{m}_gratings.mat")["data"][0, 0]
    E_, snr = d["EVENTS"], d["SNR"].ravel(); n_u, n_d, n_t = E_.shape
    C = np.zeros((n_d * n_t, n_u))
    for i in range(n_u):
        for j in range(n_d):
            for t in range(n_t):
                s = np.asarray(E_[i, j, t]).ravel(); C[j * n_t + t, i] = np.count_nonzero((s >= win[0]) & (s < win[1]))
    lab = np.repeat(np.arange(n_d), n_t)
    keep = (snr >= 1.5) & (C.mean(0) / (win[1] - win[0]) >= 1.0)
    return C[:, keep], lab


def main():
    t0 = time.time(); out = {"imaging": [], "allen": [], "pvc11": []}
    for fp in sorted(RAW.glob("gratings*.npy")):
        dat = np.load(fp, allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        lab = E.ow.bins(np.asarray(dat["istim"], float).ravel(), 8); del dat
        for N in NS:
            for d in range(DRAWS):
                idx = np.sort(np.random.default_rng(1000 * N + d).choice(X.shape[1], N, replace=False))
                out["imaging"].append({"name": fp.stem, "N": N, "draw": d, **stats_of(np.asarray(X[:, idx], np.float64), lab, 8)})
        out["imaging"].append({"name": fp.stem, "N": int(X.shape[1]), "draw": 0, "full": True, **stats_of(np.asarray(X, np.float64), lab, 8)})
        print(f"imaging {fp.stem} ({time.time() - t0:.0f}s)", flush=True)
    sids = [s["session"] for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
    for sid in sids:
        S = E.load_session(sid); idx0 = E.populations(S).get("VISp")
        if idx0 is None:
            continue
        P = E.pop_view(S, "VISp", idx0)   # H5-R1 masking path
        lab = P["lab"]; Y = E.tf_residualize(P["X"], lab, P["cell"]); n0 = P["X"].shape[1]
        for N in [n for n in NS if n <= n0] + [n0]:
            for d in range(DRAWS if N < n0 else 1):
                cols = np.sort(np.random.default_rng(1000 * N + d).choice(n0, N, replace=False)) if N < n0 else np.arange(n0)
                out["allen"].append({"session": sid, "N": int(N), "draw": d, "full": bool(N == n0), **stats_of(Y[:, cols], lab, 8)})
    print(f"allen done ({time.time() - t0:.0f}s)", flush=True)
    for m in (1, 2, 3):
        for win in ((0.28, 0.53), (0.28, 0.78), (0.28, 1.28), (0.0, 1.28)):
            X, lab = pvc_counts(m, win)
            out["pvc11"].append({"monkey": m, "window": win, "N": int(X.shape[1]), **stats_of(X, lab, 12)})
        print(f"pvc11 monkey{m} ({time.time() - t0:.0f}s)", flush=True)
    med = lambda rs, f: float(np.median([f(r) for r in rs]))
    summ = {}
    for N in NS:
        I = [r for r in out["imaging"] if r["N"] == N]; A = [r for r in out["allen"] if r["N"] == N]
        summ[f"N{N}"] = {f"{ds}_{k}": med(rs, f) for ds, rs in (("img", I), ("allen", A)) for k, f in
                         (("rho_m", lambda r: r["rho_m"]), ("cos2_top_mu0", lambda r: r["cos2_top_mu0"]),
                          ("mu0_share_over_chance", lambda r: r["mu0_share_over_chance"]), ("a_raw", lambda r: r["raw"]["a"]),
                          ("a_proj", lambda r: r["proj_mu0"]["a"]), ("c_raw", lambda r: r["raw"]["c"]), ("c_proj", lambda r: r["proj_mu0"]["c"]),
                          ("c_proj_neg_frac", lambda r: float(r["proj_mu0"]["c"] < 0)), ("c_raw_neg_frac", lambda r: float(r["raw"]["c"] < 0)))}
        print(f"N{N}", json.dumps({k: round(v, 3) for k, v in summ[f'N{N}'].items()}), flush=True)
    for r in out["pvc11"]:
        print(f"pvc11 m{r['monkey']} win {r['window']} N {r['N']} rho_m {r['rho_m']:.2f} cos2 {r['cos2_top_mu0']:.2f} a {r['raw']['a']:.3f} -> proj {r['proj_mu0']['a']:.3f} c {r['raw']['c']:+.3f} -> {r['proj_mu0']['c']:+.3f}", flush=True)
    out["summary"] = summ
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
