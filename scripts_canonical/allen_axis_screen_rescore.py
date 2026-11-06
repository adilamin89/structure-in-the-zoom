"""R2's axis screen re-scored with
(a) family-wise data splitting (discover at p <= 0.05/4 on one half, replicate at p <= 0.05 on the other) and (b) an across-mouse
split (two sets of 16 sessions by alternate index in session-id order; pick by the median over sessions of the mean half z).
Reads data_canonical/allen_axis_screen.json only. Out: data_canonical/allen_axis_screen_rescore.json"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE.parent / "data_canonical" / "allen_axis_screen.json"))
AXES = ("direction", "tf", "running", "random")
rows = sorted(D["rows"], key=lambda r: r["session"])
out = {"a_family_wise": {}, "b_across_mice": {}}
for sel, conf in (("half0", "half1"), ("half1", "half0")):
    res = {}
    for a in AXES:
        disc = [r for r in rows if r[sel][a]["p_one"] <= 0.05 / 4]
        rep = [r for r in disc if r[conf][a]["p_one"] <= 0.05]
        res[a] = {"discovered": len(disc), "replicated": len(rep), "rate": len(rep) / len(disc) if disc else None}
    out["a_family_wise"][f"discover_{sel}"] = res
sign_need = min(k for k in range(17) if 1 - stats.binom.cdf(k - 1, 16, 0.5) <= 0.05)   # one-sided sign test at 0.05, n = 16
assert sign_need == 12, sign_need                                                        # the registered '>= 12 of 16'
sets = {"set0": rows[0::2], "set1": rows[1::2]}


def zmean(r, a):
    return 0.5 * (r["half0"][a]["z"] + r["half1"][a]["z"])


per = {}
for s, rs in sets.items():
    per[s] = {}
    for a in AXES:
        z = np.array([zmean(r, a) for r in rs])
        k = int((z > 0).sum())
        per[s][a] = {"median_z": float(np.median(z)), "n_pos": k, "sign_p": float(1 - stats.binom.cdf(k - 1, len(z), 0.5)),
                     "both_halves_p_le_0.05": int(sum(r["half0"][a]["p_one"] <= 0.05 and r["half1"][a]["p_one"] <= 0.05 for r in rs))}
for s, o in (("set0", "set1"), ("set1", "set0")):
    pick = max(AXES, key=lambda a: per[s][a]["median_z"])
    first = max(AXES, key=lambda a: per[o][a]["median_z"]) == pick
    out["b_across_mice"][f"pick_{s}"] = {"pick": pick, "first_on_other": bool(first), "n_pos_other": per[o][pick]["n_pos"],
                                         "sign_need": sign_need, "confirmed": bool(first and per[o][pick]["n_pos"] >= sign_need)}
out["b_across_mice"]["per_set"] = per
out["b_across_mice"]["sets"] = {s: [r["session"] for r in rs] for s, rs in sets.items()}
print(json.dumps({k: v for k, v in out.items()}, indent=1))
json.dump({**out}, open(HERE.parent / "data_canonical" / "allen_axis_screen_rescore.json", "w"), indent=1)
