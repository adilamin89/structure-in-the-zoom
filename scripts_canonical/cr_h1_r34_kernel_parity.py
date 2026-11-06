"""H1-R34 (Oct 23): odd/even balance of the FLUCTUATION overlap kernel and the order the fluctuations favour, in imaging,
Allen VISp and macaque V1. Registration in ex0_REGISTERED.md (design there). Out: data_canonical/cr_h1_r34.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402

RAW = HERE.parent / "raw"
spec = importlib.util.spec_from_file_location("m", HERE / "cr_h1_m_pvc11.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
om = E.om


def parity(Kmat):
    K = len(Kmat); d = np.sqrt(np.outer(np.diag(Kmat), np.diag(Kmat))); Kn = Kmat / d
    a = np.arange(K); sep = (a[None, :] - a[:, None]) % K
    kappa = np.array([Kn[sep == s].mean() for s in range(K)])
    kl = np.array([np.mean(kappa * np.cos(2 * np.pi * l * np.arange(K) / K)) for l in range(K // 2 + 1)])
    po = float(sum(kl[l] ** 2 for l in range(1, K // 2 + 1) if l % 2 == 1)); pe = float(sum(kl[l] ** 2 for l in range(2, K // 2 + 1) if l % 2 == 0))
    return {"kappa": kappa.tolist(), "harmonics": kl.tolist(), "P_odd": po, "P_even": pe, "R": po / pe, "margin": float(kappa[1] - kappa[K // 2])}


def classmean_bc(X, lab, K):
    Cm = np.corrcoef(np.stack([X[lab == c].mean(0) for c in range(K)]))
    prof = np.array([np.mean([Cm[i, (i + k) % K] for i in range(K)]) for k in range(K)]); ang = 2 * np.pi * np.arange(K) / K
    return float((prof * np.cos(2 * ang)).sum() / (prof * np.cos(ang)).sum())


rows = []
b7 = {r["name"]: r["resid"]["order_within"] for r in json.load(open(HERE.parent / "data_canonical" / "cr_h1_b7.json"))["rows"]}
b9 = {r["name"]: r["ds_fraction"] for r in json.load(open(HERE.parent / "data_canonical" / "cr_h1_b9.json"))["rows"]}
for fp in sorted(RAW.glob("gratings*.npy")):
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab, sub = E.ow.bins(phi, 8), om.bins(phi, 48); h = om.halves_by_sub(sub)
    Kmat = om.split_half_K(om.residuals(X[h == 0], lab[h == 0], sub[h == 0], "raw"), om.residuals(X[h == 1], lab[h == 1], sub[h == 1], "raw"))
    rows.append({"prep": "imaging", "id": fp.stem, **parity(Kmat), "bc": classmean_bc(X, lab, 8), "ds_fraction": b9[fp.stem], "order_within": b7[fp.stem]})
    print(rows[-1]["id"][9:30], "R %.3f margin %+.3f order %+.3f" % (rows[-1]["R"], rows[-1]["margin"], rows[-1]["order_within"]), flush=True)
for fp in sorted((HERE.parent / "data_canonical" / "h1_allen").glob("session_*.json")):
    v = json.load(open(fp)); sid = v["session"]; a = v["areas"]["VISp"]
    S = E.load_session(sid); idx = E.populations(S)["VISp"]; P = E.pop_view(S, "VISp", idx); X, lab, cell = P["X"], P["lab"], P["cell"]; h = E.halves(cell)  # H5-R1
    Kmat = om.split_half_K(om.residuals(X[h == 0], lab[h == 0], cell[h == 0], "noise"), om.residuals(X[h == 1], lab[h == 1], cell[h == 1], "noise"))
    rows.append({"prep": "allen", "id": sid, **parity(Kmat), "bc": classmean_bc(E.tf_residualize(X, lab, cell), lab, 8), "ds_fraction": a["ds_fraction"],
                 "order_within": float(np.mean([o["within"] for o in a["order_tfres"]]))})
pv = {r["monkey"]: r for r in json.load(open(HERE.parent / "data_canonical" / "cr_h1_m_pvc11.json"))["arrays"]}
for m in (1, 2, 3):
    X, lab, _ = M.load_array(m); h = M.halves(lab)
    Kmat = om.split_half_K(om.residuals(X[h == 0], lab[h == 0], lab[h == 0], "raw"), om.residuals(X[h == 1], lab[h == 1], lab[h == 1], "raw"))
    rows.append({"prep": "macaque", "id": m, **parity(Kmat), "bc": classmean_bc(X, lab, 12), "ds_fraction": pv[m]["raw"]["ds_fraction"], "order_within": pv[m]["raw"]["order_mean"]["within"]})
summ = {}
for p in ("imaging", "allen", "macaque"):
    rs = [r for r in rows if r["prep"] == p]
    summ[p] = {"R_range": [min(r["R"] for r in rs), max(r["R"] for r in rs)], "R_median": float(np.median([r["R"] for r in rs])),
               "order_neg": sum(r["order_within"] < 0 for r in rs), "n": len(rs), "R_gt1": sum(r["R"] > 1 for r in rs)}
best = max(((t, sum((r["R"] > t) == (r["order_within"] < 0) for r in rows)) for t in sorted(r["R"] for r in rows)), key=lambda z: z[1])
summ["best_threshold"] = {"R": best[0], "correct": best[1], "of": len(rows)}
summ["R_gt1_vs_order_neg_agree"] = sum((r["R"] > 1) == (r["order_within"] < 0) for r in rows)
summ["spearman_R_order"] = float(stats.spearmanr([r["R"] for r in rows], [r["order_within"] for r in rows])[0])
summ["spearman_R_ds_imaging"] = float(stats.spearmanr([r["R"] for r in rows if r["prep"] == "imaging"], [r["ds_fraction"] for r in rows if r["prep"] == "imaging"])[0])
summ["spearman_bc_order"] = float(stats.spearmanr([r["bc"] for r in rows], [r["order_within"] for r in rows])[0])
print(json.dumps(summ, indent=1))
json.dump({"rows": rows, "summary": summ}, open(HERE.parent / "data_canonical" / "cr_h1_r34.json", "w"), indent=1)
print("ALL DONE")
