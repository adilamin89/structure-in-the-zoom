"""Lagged state regressors per Allen drifting-grating presentation,
aligned with allen_counts (start times asserted equal). From each session NWB with eye tracking (raw/allen_nwb):
filtered pupil_area and running speed averaged over [start + L, stop + L) for L in LAGS, NaN samples ignored, NaN where < 50% valid;
and the pupil change over the presentation (mean of the last 0.25 s minus the first 0.25 s, each NaN where < 50% valid).
Run: python allen_state_extract.py
Out: ../raw/allen_state/session_<sid>.npz (pupil_L<ms>, speed_L<ms>, dpupil, start)"""
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
NWB = HERE.parent / "raw" / "allen_nwb"
COUNTS = HERE.parent / "raw" / "allen_counts"
OUT = HERE.parent / "raw" / "allen_state"
LAGS = (0.0, 0.5, 1.0, 1.5)


def window_mean(t, v, a, b):
    ok = np.isfinite(v)
    cs = np.concatenate([[0], np.cumsum(np.where(ok, v, 0))]); cn = np.concatenate([[0], np.cumsum(ok)])
    i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
    n, nv = i1 - i0, cn[i1] - cn[i0]
    return np.where(nv >= 0.5 * np.maximum(n, 1), (cs[i1] - cs[i0]) / np.maximum(nv, 1), np.nan)


def series(g):
    v, t = g["data"][:].astype(np.float64), g["timestamps"][:]
    o = np.argsort(t, kind="stable")
    return t[o], v[o]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]:
        sid = s["session"]
        f = h5py.File(NWB / f"session_{sid}.nwb", "r")
        if "filtered_gaze_mapping" not in f["processing"]:
            continue
        dgn = [k for k in f["intervals"].keys() if "drifting_gratings" in k and "contrast" not in k]
        dg = f["intervals"][dgn[0]]
        ori = np.asarray(dg["orientation"][:], float); keep = np.isfinite(ori)
        start, stop = dg["start_time"][:][keep], dg["stop_time"][:][keep]
        c = np.load(COUNTS / f"session_{sid}.npz")
        assert np.allclose(c["start"], start), sid
        tp, vp = series(f["processing/filtered_gaze_mapping/pupil_area"])
        ts, vs = series(f["processing/running/running_speed"])
        out = {"start": start}
        for L in LAGS:
            out[f"pupil_L{int(L * 1000)}"] = window_mean(tp, vp, start + L, stop + L)
            out[f"speed_L{int(L * 1000)}"] = window_mean(ts, vs, start + L, stop + L)
        out["dpupil"] = window_mean(tp, vp, stop - 0.25, stop) - window_mean(tp, vp, start, start + 0.25)
        np.savez_compressed(OUT / f"session_{sid}.npz", **out)
        print(sid, {k: int(np.isnan(v).sum()) for k, v in out.items() if k != "start"}, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
