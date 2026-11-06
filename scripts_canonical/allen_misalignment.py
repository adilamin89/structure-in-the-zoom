"""Is Allen's remaining misalignment (a ~ 0.25 after removing the baseline direction) the alignment of independent
units? Allen VISp (full populations, TF-residualized) and imaging at N = 128 (B1 seeds 1000 N + d, 10 draws), raw frame.
a = xi PR_W with xi = Tr WB/(Tr W Tr B). Variants: W as measured; W -> diag(W) (independent units); W with its top k = 1, 2, 3, 5
eigenmodes removed. Spearman across Allen populations of a with rho_m = |mu0|^2/mean|t|^2, mean rate and the diagonal share of
Tr W^2. Out: data_canonical/allen_misalignment.json"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import allen_engine as E  # noqa: E402

RAW = HERE.parent / "raw"


def variants(X, lab, K=8):
    M = np.stack([X[lab == c].mean(0) for c in range(K)]); mu0 = M.mean(0); T = M - mu0
    p = np.bincount(lab, minlength=K) / len(lab)
    R = (X - M[lab]) / np.sqrt(len(X)); W = R.T @ R; B = (T * p[:, None]).T @ T
    def a_of(Wm):
        w1, b1 = np.trace(Wm), np.trace(B)
        return float(np.sum(Wm * B) / (w1 * b1) * w1 ** 2 / np.sum(Wm * Wm))
    out = {"a": a_of(W), "a_diag": a_of(np.diag(np.diag(W)))}
    lam, V = np.linalg.eigh(W)
    for k in (1, 2, 3, 5):
        Vk = V[:, -k:]; out[f"a_minus_top{k}"] = a_of(W - (Vk * lam[-k:]) @ Vk.T)
    out["rho_m"] = float(mu0 @ mu0 / np.mean((T * T).sum(1))); out["mean_rate"] = float(mu0.mean())
    out["diag_share_W2"] = float(np.sum(np.diag(W) ** 2) / np.sum(W * W))
    return out


res = {"allen": [], "imaging": []}
for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]:
    S = E.load_session(s["session"]); idx = E.populations(S).get("VISp")
    if idx is None:
        continue
    P = E.pop_view(S, "VISp", idx)   # masking path
    Y = E.tf_residualize(P["X"], P["lab"], P["cell"])
    res["allen"].append({"session": s["session"], "N": int(P["X"].shape[1]), **variants(Y, P["lab"])})
for fp in sorted(RAW.glob("gratings*.npy")):
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    lab = E.ow.bins(np.asarray(dat["istim"], float).ravel(), 8); del dat
    for d in range(10):
        idx = np.sort(np.random.default_rng(1000 * 128 + d).choice(X.shape[1], 128, replace=False))
        res["imaging"].append({"name": fp.stem, "draw": d, **variants(np.asarray(X[:, idx], np.float64), lab)})
    print(fp.stem, flush=True)
summ = {}
for ds, rows in res.items():
    summ[ds] = {k: float(np.median([r[k] for r in rows])) for k in ("a", "a_diag", "a_minus_top1", "a_minus_top2", "a_minus_top3", "a_minus_top5", "rho_m", "diag_share_W2")}
A = res["allen"]
summ["allen_spearman"] = {k: float(stats.spearmanr([r["a"] for r in A], [r[k] for r in A])[0]) for k in ("rho_m", "mean_rate", "diag_share_W2")}
print(json.dumps(summ, indent=1))
json.dump({"rows": res, "summary": summ}, open(HERE.parent / "data_canonical" / "allen_misalignment.json", "w"), indent=1)
print("ALL DONE")
