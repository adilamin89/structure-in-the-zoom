"""Mining P5. The mean field ln[8/(1+7 sigma)]/ln 8 assumes every pair of classes overlaps alike; Allen's overlaps
are angular (opposite directions share more than neighbours, dK < 0 in 32/32 VISp). Replace it by the separation kernel: from half A
(split halves inside A, alternate presentations of each direction x TF cell; noise residuals as sigma), K_cd -> kappa(s) = mean of
K_cd / mean diag over pairs at separation s; with equal class weights the declared rung of k classes (0..k-1) has
1/PR(W_k) proportional to k^-2 sum_{c,d < k} kappa(|c - d|) and every floor rung holds all eight classes (a constant), so the
predicted within part = least-squares slope over half B's log rung sizes of -log(k^-2 sum kappa). Target: half B's TF-residualized
ladder within part; both ways averaged. Mean field from the same half A (kappa(s) = sigma for s > 0) side by side.
Out: data_canonical/fluctuation_kernel_law.json"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import allen_engine as E  # noqa: E402

RUNGS = (1, 2, 3, 4, 6, 8)


def kernel_curve(kappa, sizes):
    sep = lambda c, d: min(abs(c - d), 8 - abs(c - d))
    y = [-np.log(np.mean([kappa[sep(c, d)] for c in range(k) for d in range(k)])) for k in RUNGS]
    return float(np.polyfit(np.log(sizes), y, 1)[0])


rows = []
sids = [s["session"] for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
for sid in sids:
    S = E.load_session(sid); idx = E.populations(S).get("VISp")
    if idx is None:
        continue
    X, lab, cell = S["X"][:, idx], S["lab"], S["cell"]
    h = E.halves(cell); ways = []
    for a, b in ((0, 1), (1, 0)):
        A, B = h == a, h == b
        hh = E.halves(cell[A])
        Ra = E.om.residuals(X[A][hh == 0], lab[A][hh == 0], cell[A][hh == 0], "noise")
        Rb = E.om.residuals(X[A][hh == 1], lab[A][hh == 1], cell[A][hh == 1], "noise")
        K = E.om.split_half_K(Ra, Rb); d0 = np.diag(K).mean()
        sepm = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
        kappa = [float(K[sepm == s].mean() / d0) for s in range(5)]
        sigma = float(K[sepm > 0].mean() / d0)
        sizes = [int((lab[B] < k).sum()) for k in RUNGS]
        target = E.ladder(E.tf_residualize(X[B], lab[B], cell[B]), lab[B])["within"]
        ways.append({"kappa": kappa, "sigma": sigma, "pred_kernel": kernel_curve(kappa, sizes),
                     "pred_meanfield": kernel_curve([1.0] + [sigma] * 4, sizes), "target": target})
    rows.append({"session": sid, "ways": ways, **{k: float(np.mean([w[k] for w in ways])) for k in ("pred_kernel", "pred_meanfield", "target")}})
    print(sid, " ".join(f"{k} {rows[-1][k]:+.3f}" for k in ("pred_kernel", "pred_meanfield", "target")), flush=True)
pk, pm, t = (np.array([r[k] for r in rows]) for k in ("pred_kernel", "pred_meanfield", "target"))
v = {"r_kernel": float(stats.pearsonr(pk, t)[0]), "mae_kernel": float(np.abs(pk - t).mean()), "bias_kernel": float((pk - t).mean()),
     "r_meanfield": float(stats.pearsonr(pm, t)[0]), "mae_meanfield": float(np.abs(pm - t).mean()), "bias_meanfield": float((pm - t).mean())}
print("VERDICT", json.dumps(v, indent=1))
json.dump({"rows": rows, "verdict": v}, open(HERE.parent / "data_canonical" / "fluctuation_kernel_law.json", "w"), indent=1)
print("ALL DONE")
