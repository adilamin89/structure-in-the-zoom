"""H1-R33a (Oct 22): remove Allen's top shared fluctuation modes and re-measure. Allen VISp, TF-residualized input Y. W's top k
eigenmodes (k = 1, 3, 5) are estimated from the class residuals of one half of each direction x TF cell (alternate presentations)
and projected out of every residual (both halves; class means kept exactly), so the modes are not fitted on the trials measured
in the other half. Then the Table 9 ladder and split, the full-set term c, PR(W) and the alignment a, on the whole population.
Out: data_canonical/cr_h1_r33a.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402

rows = []
for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]:
    S = E.load_session(s["session"]); idx = E.populations(S).get("VISp")
    if idx is None:
        continue
    lab, cell = S["lab"], S["cell"]
    Y = E.tf_residualize(S["X"][:, idx], lab, cell)
    M = np.stack([Y[lab == c].mean(0) for c in range(8)]); R = Y - M[lab]
    h = E.halves(cell)
    Ra = R[h == 0]; lam, V = np.linalg.eigh(Ra.T @ Ra)
    row = {"session": s["session"], "N": int(len(idx))}
    for k in (0, 1, 3, 5):
        Vk = V[:, -k:] if k else np.zeros((R.shape[1], 0))
        Rk = R - (R @ Vk) @ Vk.T
        Z = M[lab] + Rk
        sg = E.sign_law(Z, lab); lad = E.ladder(Z, lab)
        a = sg["x_share"] * sg["trB_trW"] / (2 * (1 - sg["x_share"]) * sg["PR_B"]) * sg["PR_W"]
        row[f"k{k}"] = {**lad, "c_full": sg["c_full"], "PR_W": sg["PR_W"], "a": float(a), "law_neg": bool(sg["ratio"] > sg["threshold"])}
    rows.append(row)
    print(s["session"], " ".join(f"k{k}: cm {row[f'k{k}']['class_means']:+.3f} c {row[f'k{k}']['c_full']:+.3f} a {row[f'k{k}']['a']:.2f} d {row[f'k{k}']['delta']:+.3f}" for k in (0, 1, 3, 5)), flush=True)
summ = {f"k{k}": {"cm_negative": int(sum(r[f"k{k}"]["class_means"] < 0 for r in rows)), "c_negative": int(sum(r[f"k{k}"]["c_full"] < 0 for r in rows)),
                  "cfull_agrees": int(sum((r[f"k{k}"]["c_full"] < 0) == (r[f"k{k}"]["class_means"] < 0) for r in rows)),
                  "a_median": float(np.median([r[f"k{k}"]["a"] for r in rows])), "delta_median": float(np.median([r[f"k{k}"]["delta"] for r in rows])),
                  "delta_pos": int(sum(r[f"k{k}"]["delta"] > 0 for r in rows)), "n": len(rows)} for k in (0, 1, 3, 5)}
print("SUMMARY", json.dumps(summ, indent=1))
json.dump({"rows": rows, "summary": summ}, open(HERE.parent / "data_canonical" / "cr_h1_r33a.json", "w"), indent=1)
print("ALL DONE")
