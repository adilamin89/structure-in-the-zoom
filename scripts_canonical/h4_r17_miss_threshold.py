"""H4-R17 (Oct 25; registration H4-R17 in data_canonical/ex0_REGISTERED.md): are the paper's 14 Allen sign misses where the split is
estimation-limited? Joins h1_allen_b2_predictions.json to h1_allen/session_*.json tfres as c12_allen_misses.py; q = trials per class
/ PR(W) (sign_tfres.PR_W). Out: data_canonical/h4_r17_miss_threshold.json"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

DATA = Path(__file__).resolve().parent.parent / "data_canonical"
pred = {(p["session"], p["area"]): p for p in json.load(open(DATA / "h1_allen_b2_predictions.json"))}
rows = []
for f in sorted(glob.glob(str(DATA / "h1_allen" / "session_*.json"))):
    d = json.load(open(f))
    for area, v in d["areas"].items():
        p = pred[(d["session"], area)]
        q = (v["n_pres"] / 8) / v["sign_tfres"]["PR_W"]
        rows.append({"session": d["session"], "area": area, "q": float(q), "delta": v["tfres"]["delta"], "delta_pred": p["delta_pred"],
                     "hit": bool(np.sign(v["tfres"]["delta"]) == np.sign(p["delta_pred"]))})
q = np.array([r["q"] for r in rows]); hit = np.array([r["hit"] for r in rows]); ad = np.abs([r["delta"] for r in rows])
n_miss = json.load(open(DATA / "c12_allen_misses.json"))["n_miss"]   # the same join (14 canonical; 15 under the H5 masks)
assert len(rows) == 167 and (~hit).sum() == n_miss, (len(rows), (~hit).sum(), n_miss)
t = np.percentile(q, [100 / 3, 200 / 3])
sc = {"n": len(rows), "misses": int((~hit).sum()), "q_range": [float(q.min()), float(q.max())],
      "acc_q_ge_2": [int(hit[q >= 2].sum()), int((q >= 2).sum())], "acc_q_lt_2": [int(hit[q < 2].sum()), int((q < 2).sum())],
      "tertile_edges": t.tolist(),
      "acc_by_tertile": [[int(hit[m].sum()), int(m.sum())] for m in (q < t[0], (q >= t[0]) & (q < t[1]), q >= t[1])],
      "spearman_q_absdelta": float(stats.spearmanr(q, ad)[0]),
      "q_median_misses": float(np.median(q[~hit])), "q_median_hits": float(np.median(q[hit])),
      "mannwhitney_p_misses_lower": float(stats.mannwhitneyu(q[~hit], q[hit], alternative="less").pvalue),
      "absdelta_median_misses": float(np.median(ad[~hit]))}
print(json.dumps(sc, indent=1))
json.dump({"registration": "H4-R17 (ex0_REGISTERED.md)", "rows": rows, "score": sc}, open(DATA / "h4_r17_miss_threshold.json", "w"), indent=1)
