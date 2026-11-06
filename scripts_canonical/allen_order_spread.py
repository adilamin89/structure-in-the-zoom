"""Does Allen's declared-order direction shift survive averaging over class orders?
The paper's Allen numbers (Section 6, Appendix H) are shifts along the declared angular order. On the Allen two-photon experiments
(heldout_allen2p_*), at the same design of about 75 presentations per class and a few hundred cells, the shift's SD over random class
orders was as large as the shift, set by how much the classes' own fluctuation dimensions differ; the imaging recordings' order SD is
0.04-0.09 (Appendix C). This script reads the Neuropixels populations the same way.
For every population (the release's masking path, variant B) and both arms (pooled and tfres, Appendix H's split arm): the declared
shift with allen_run.py's floor (ten draws, a fresh seed 42 per population, rung by rung), equal to data_canonical/allen_sessions/ to 1e-15,
the order-averaged shift over 50 random orders of the eight direction classes (each order with its own floor) and its SD over orders.
The direction shifts of Figure 1C and the population table (allen_expansion_all_sessions.json, 158 of 167 positive) draw their ten-draw
floor from one seed-42 stream per session shared across its areas; with this script's floor 156 of 167 pooled shifts are positive. The two
populations that change sign lie within 0.04 of zero (Appendix H: the floor draw moves a few such shifts).
Usage (raw/allen_counts from allen_counts_local.py): VECLIB_MAXIMUM_THREADS=1 python scripts_canonical/allen_order_spread.py [visp|all]
Out: data_canonical/allen_order_spread.json (rows + summary)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "allen_order_spread.json"
REPO = Path(os.environ.get("RUNG_REPO", HERE.parent))
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts_canonical"))
import rung  # noqa: E402
import allen_engine as E  # noqa: E402

K_ORDERS = 50


def one(X, lab):
    r = rung.zoom(X, lab, n_perm=0, k_orders=K_ORDERS, n_floor_draws=10, floor_seed=42, floor_by_rung=True, seed=0)
    return {"delta": float(r["delta"]), "delta_orderavg": float(r["delta_orderavg"]), "delta_orderavg_sd": float(r["delta_orderavg_sd"])}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "visp"
    sids = sorted(int(p.stem.split("_")[1]) for p in (REPO / "raw" / "allen_counts").glob("session_*.npz"))
    rows, t0 = [], time.time()
    for i, sid in enumerate(sids):
        S = E.load_session(sid)
        for area, idx in E.populations(S).items():
            if which == "visp" and area != "VISp":
                continue
            V = E.pop_view(S, area, idx)
            row = {"session": sid, "area": area, "n_units": int(V["X"].shape[1]), "n_pres": int(V["X"].shape[0])}
            row["pooled"] = one(V["X"], V["lab"])
            row["tfres"] = one(E.tf_residualize(V["X"], V["lab"], V["cell"]), V["lab"])
            rows.append(row)
            print(f"{sid} {area:6s} units {row['n_units']:3d} | pooled declared {row['pooled']['delta']:+.3f} orderavg "
                  f"{row['pooled']['delta_orderavg']:+.3f} (sd {row['pooled']['delta_orderavg_sd']:.3f}) | tfres declared "
                  f"{row['tfres']['delta']:+.3f} orderavg {row['tfres']['delta_orderavg']:+.3f} (sd {row['tfres']['delta_orderavg_sd']:.3f}) "
                  f"| {time.time() - t0:.0f}s", flush=True)
    summ = {}
    for arm in ("pooled", "tfres"):
        for scope, sel in (("VISp", [r for r in rows if r["area"] == "VISp"]), ("all", rows)):
            if not sel:
                continue
            d = np.array([r[arm]["delta"] for r in sel]); o = np.array([r[arm]["delta_orderavg"] for r in sel])
            sd = np.array([r[arm]["delta_orderavg_sd"] for r in sel])
            summ[f"{arm}_{scope}"] = {"n": len(sel), "declared_positive": int((d > 0).sum()), "orderavg_positive": int((o > 0).sum()),
                                      "declared_mean": float(d.mean()), "orderavg_mean": float(o.mean()),
                                      "order_sd_median": float(np.median(sd)), "abs_declared_median": float(np.median(np.abs(d))),
                                      "r_declared_orderavg": float(np.corrcoef(d, o)[0, 1]) if len(sel) > 2 else None}
    for k, v in summ.items():
        print(k, json.dumps(v), flush=True)
    json.dump({"k_orders": K_ORDERS, "floor": "declared: ten draws, seed 42, rung by rung (the paper's); orders: rung 1.8.0 per-order floors",
               "rows": rows, "summary": summ}, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
