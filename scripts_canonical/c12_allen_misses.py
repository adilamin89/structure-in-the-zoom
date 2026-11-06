"""Where the committed two-term prediction misses the sign of Allen's shift (C12, Oct 23; answers the cold read's 'the 14 of 167
misses are unexamined'). Joins h1_allen_b2_predictions.json (committed before scoring, 760ee7cd) to the measured TF-separated
split in h1_allen/session_*.json, exactly as make_fig_c11_twoterm.py does, and characterizes the sign misses: measured |delta|,
area, and which term misses (class-mean term vs the mean-field within-class term). Nothing is fitted.
Out: data_canonical/c12_allen_misses.json"""
import glob
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data_canonical"
pred = {(p["session"], p["area"]): p for p in json.load(open(DATA / "h1_allen_b2_predictions.json"))}
rows = []
for f in sorted(glob.glob(str(DATA / "h1_allen" / "session_*.json"))):
    d = json.load(open(f))
    for area, v in d["areas"].items():
        q, m = pred[(d["session"], area)], v["tfres"]
        rows.append({"session": d["session"], "area": area, "N": q["N"],
                     "delta_pred": q["delta_pred"], "delta": m["delta"],
                     "classmean_pred": q["classmean_pred"], "class_means": m["class_means"],
                     "within_pred": q["within_pred"], "within": m["within"]})
assert len(rows) == 167, len(rows)
sg = lambda x: np.sign(x)
miss = [r for r in rows if sg(r["delta_pred"]) != sg(r["delta"])]
hit = [r for r in rows if r not in miss]
ad = np.array([abs(r["delta"]) for r in rows])
cut = float(np.ceil(max(abs(r["delta"]) for r in miss) * 100) / 100)   # largest missed |delta|, rounded up to two decimals
out = {
    "n": len(rows), "n_miss": len(miss), "n_visp_miss": sum(r["area"] == "VISp" for r in miss),
    "miss_areas": {a: sum(r["area"] == a for r in miss) for a in sorted({r["area"] for r in miss})},
    "abs_delta_median_all": float(np.median(ad)), "abs_delta_median_miss": float(np.median([abs(r["delta"]) for r in miss])),
    "abs_delta_max_miss": float(max(abs(r["delta"]) for r in miss)),
    "n_abs_delta_ge_cut": int((ad >= cut).sum()),
    "n_abs_delta_ge_cut_sign_right": int(sum(abs(r["delta"]) >= cut for r in hit)),
    "n_abs_delta_lt_cut": int((ad < cut).sum()),
    "miss_classmean_sign_right": int(sum(sg(r["classmean_pred"]) == sg(r["class_means"]) for r in miss)),
    "miss_within_measured_negative": int(sum(r["within"] < 0 for r in miss)),
    "miss_within_pred_positive": int(sum(r["within_pred"] > 0 for r in miss)),
    "all_within_measured_negative": int(sum(r["within"] < 0 for r in rows)),
    "all_classmean_sign_right": int(sum(sg(r["classmean_pred"]) == sg(r["class_means"]) for r in rows)),
    "cut": cut, "misses": miss,
}
for k, v in out.items():
    if k != "misses":
        print(k, v)
json.dump(out, open(DATA / "c12_allen_misses.json", "w"), indent=1)
print("wrote c12_allen_misses.json DONE")
