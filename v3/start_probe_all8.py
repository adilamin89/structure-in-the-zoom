"""Where a cyclic axis starts, in all eight grating recordings (Section 3 and Appendix C's table of starts and order averages in
the extended version).

Per recording (Stringer et al. 2021 gratings in raw/: sresp neurons x presentations, istim radians), eight 45-degree direction
classes, rung.zoom with its default 20-draw floor, seed 0, no permutation null (only shifts are read):
  walls_k    the bin walls turned by k x 45/8 degrees, k = 0..7 (k = 0 = the paper's binning, asserted equal to the np.digitize
             binning REFERENCE.md uses): the declared shift (classes added in angular order from the first bin) and the
             order-averaged shift over 50 random class orders with its SD over orders;
  start_s    the paper's walls (k = 0), the angular order started at class s = 0..7 and run forwards or backwards (16 declared
             shifts).
Computed with rung 1.13.0. rung 1.14 returns the same declared shifts; its default order average for eight classes is the exact
mean over all 40,320 orders, so this script passes order_average='random' to replay the random-order average (--exact gives the
exact one instead; rung data --continuous reads the wall positions with the exact average).
Out: v3/start_probe_all8.json (rows per recording), progress on stdout; about an hour of CPU for all eight.
Run: python v3/start_probe_all8.py [--exact] [RECORDING ...]   (names as 'Drifting GT3' or file stems)"""
import json
import sys
import time

import numpy as np

from _common import GRATINGS, HERE, RAW, rung

EXACT = "--exact" in sys.argv
ARGS = [a for a in sys.argv[1:] if a != "--exact"]
OUT = HERE / ("start_probe_all8_exact.json" if EXACT else "start_probe_all8.json")
W = 2 * np.pi / 8


def labels_for(phi, k):
    return (np.floor(((phi - k * W / 8) % (2 * np.pi)) / W).astype(int)) % 8


def probe(name, stem):
    d = np.load(RAW / f"{stem}.npy", allow_pickle=True).item()
    X = np.asarray(d["sresp"], np.float32).T.astype(np.float64)        # presentations by neurons
    phi = np.asarray(d["istim"], float) % (2 * np.pi)
    ref = np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
    assert X.shape[0] == phi.size
    row = {"stem": stem, "shape": list(X.shape), "walls": {}, "starts": {}}
    for k in range(8):
        lab = labels_for(phi, k)
        if k == 0:
            assert (lab == ref).all(), "k = 0 must reproduce the paper's binning"
        t0 = time.time()
        z = rung.zoom(X, lab, n_perm=0, k_orders=50, workers=4, order_average="exact" if EXACT else "random")
        row["walls"][str(k)] = {"offset_deg": k * 45 / 8, "delta": z["delta"], "theta_obs": z["theta_obs"],
                                "theta_floor": z["theta_floor"], "delta_orderavg": z["delta_orderavg"],
                                "delta_orderavg_sd": z["delta_orderavg_sd"], "counts": np.bincount(lab, minlength=8).tolist()}
        print(f"{name:17s} walls+{k * 45 / 8:5.2f}  declared {z['delta']:+.3f}  order-avg {z['delta_orderavg']:+.3f}"
              f" (SD {z['delta_orderavg_sd']:.3f})  {time.time() - t0:.0f} s", flush=True)
    for s in range(8):
        for direction in (1, -1):
            order = [(s + direction * i) % 8 for i in range(8)]
            z = rung.zoom(X, ref, n_perm=0, k_orders=0, order=order, workers=4)
            row["starts"][f"{s}{'+' if direction > 0 else '-'}"] = {"order": order, "delta": z["delta"]}
        print(f"{name:17s} start {s}: forwards {row['starts'][f'{s}+']['delta']:+.3f}, backwards {row['starts'][f'{s}-']['delta']:+.3f}",
              flush=True)
    dec = np.array([v["delta"] for v in row["walls"].values()])
    oa = np.array([v["delta_orderavg"] for v in row["walls"].values()])
    st = np.array([v["delta"] for v in row["starts"].values()])
    row["summary"] = {"walls_declared_min": dec.min(), "walls_declared_max": dec.max(), "walls_declared_sd": dec.std(ddof=1),
                      "walls_orderavg_min": oa.min(), "walls_orderavg_max": oa.max(), "walls_orderavg_sd": oa.std(ddof=1),
                      "starts_min": st.min(), "starts_max": st.max(), "starts_sd": st.std(ddof=1),
                      "paper_walls_declared": dec[0], "paper_walls_orderavg": oa[0]}
    return row


def main():
    names = [n for n in GRATINGS if not ARGS or GRATINGS[n] in ARGS or n in ARGS]
    out = {"rung": rung.__version__, "order_average": "exact" if EXACT else "random", "rows": {}}
    for name in names:
        out["rows"][name] = probe(name, GRATINGS[name])
        json.dump(out, open(OUT, "w"), indent=1, default=float)        # save after every recording
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
