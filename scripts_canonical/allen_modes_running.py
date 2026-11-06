"""Do Allen's top shared fluctuation modes track running? Allen VISp, TF-residualized residuals R (presentation x
unit); W's top 3 eigenmodes; per presentation the mode score R v_k; Pearson r with the presentation's mean running speed
(allen_running_modal.py; same presentations and order as the counts, start times asserted equal). Also the fraction of the
residual variance explained by speed (R^2 of regressing every unit's residual on speed). Out: data_canonical/allen_modes_running.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import allen_engine as E  # noqa: E402

RUN = HERE.parent / "raw" / "allen_running"
rows = []
for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]:
    sid = s["session"]; S = E.load_session(sid); idx = E.populations(S).get("VISp")
    if idx is None:
        continue
    r = np.load(RUN / f"session_{sid}.npz"); c = np.load(E.COUNTS / f"session_{sid}.npz")
    assert np.allclose(r["start"], c["start"]), sid
    P = E.pop_view(S, "VISp", idx, extra={"speed": r["speed"]})   # masking path
    sp = P["speed"]; lab = P["lab"]
    Y = E.tf_residualize(P["X"], lab, P["cell"]); R = Y - np.stack([Y[lab == k].mean(0) for k in range(8)])[lab]
    lam, V = np.linalg.eigh(R.T @ R); V = V[:, ::-1]; lam = lam[::-1]
    sc = R @ V[:, :3]
    rr = [float(np.corrcoef(sc[:, k], sp)[0, 1]) if sp.std() > 0 else float("nan") for k in range(3)]
    spc = sp - sp.mean(); beta = (spc @ R) / max(spc @ spc, 1e-12)
    r2 = float(((np.outer(spc, beta)) ** 2).sum() / (R ** 2).sum()) if sp.std() > 0 else float("nan")
    rows.append({"session": sid, "r_mode1_3": rr, "max_abs_r": float(np.nanmax(np.abs(rr))), "speed_mean": float(sp.mean()),
                 "speed_sd": float(sp.std()), "frac_running": float((sp > 1).mean()), "var_explained_by_speed": r2,
                 "top3_var_share": float(lam[:3].sum() / lam.sum())})
    print(sid, "r", np.round(rr, 2), "speed %.1f+-%.1f run %.2f | R2 %.3f | top3 share %.2f" % (sp.mean(), sp.std(), rows[-1]["frac_running"], r2, rows[-1]["top3_var_share"]), flush=True)
ok = [x for x in rows if np.isfinite(x["max_abs_r"])]
summ = {"n": len(rows), "n_with_speed_variance": len(ok), "max_abs_r_ge_0.3": int(sum(x["max_abs_r"] >= 0.3 for x in ok)),
        "median_max_abs_r": float(np.median([x["max_abs_r"] for x in ok])), "median_var_explained_by_speed": float(np.median([x["var_explained_by_speed"] for x in ok])),
        "median_top3_share": float(np.median([x["top3_var_share"] for x in rows]))}
print("SUMMARY", json.dumps(summ, indent=1))
json.dump({"rows": rows, "summary": summ}, open(HERE.parent / "data_canonical" / "allen_modes_running.json", "w"), indent=1)
print("ALL DONE")
