"""H5b-R3 (Oct 26; registration in data_canonical/ex0_REGISTERED.md): pupil area per Allen drifting-grating presentation, aligned with
h1_allen_counts (same presentations: finite orientation, NWB order; start times asserted equal). Reads each session NWB in
raw/allen_nwb (fetched by h4_allen_fetch_nwb.sh; byte-identical to the S3 ecephys-cache files): the mean of
processing/filtered_gaze_mapping/pupil_area over [start_time, stop_time), NaN samples (blinks, failed fits) ignored, and the fraction of
valid samples; pupil is NaN where fewer than half the samples are valid. Sessions whose NWB has no eye-tracking modules are listed.
Run: python h5b_allen_pupil_local.py
Out: ../raw/allen_pupil/session_<sid>.npz (pupil, valid_frac, start, n_samples); summary data_canonical/h5b_allen_pupil.json"""
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
NWB = HERE.parent / "raw" / "allen_nwb"
COUNTS = HERE.parent / "raw" / "allen_counts"
OUT = HERE.parent / "raw" / "allen_pupil"


def extract(path):
    f = h5py.File(path, "r")
    if "filtered_gaze_mapping" not in f["processing"]:
        return None
    dgn = [k for k in f["intervals"].keys() if "drifting_gratings" in k and "contrast" not in k]
    assert len(dgn) == 1, dgn
    dg = f["intervals"][dgn[0]]
    ori = np.asarray(dg["orientation"][:], float); keep = np.isfinite(ori)
    start, stop = dg["start_time"][:][keep], dg["stop_time"][:][keep]
    pa = f["processing/filtered_gaze_mapping/pupil_area"]
    v, t = pa["data"][:].astype(np.float64), pa["timestamps"][:]
    o = np.argsort(t, kind="stable"); v, t = v[o], t[o]
    i0, i1 = np.searchsorted(t, start), np.searchsorted(t, stop)
    ok = np.isfinite(v)
    cs = np.concatenate([[0], np.cumsum(np.where(ok, v, 0))]); cn = np.concatenate([[0], np.cumsum(ok)])
    n, nv = i1 - i0, cn[i1] - cn[i0]
    frac = nv / np.maximum(n, 1)
    pupil = np.where(frac >= 0.5, (cs[i1] - cs[i0]) / np.maximum(nv, 1), np.nan)
    return dict(pupil=pupil, valid_frac=frac, start=start, n_samples=n)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sids = [s["session"] for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
    summ = {"with_pupil": [], "without_pupil": [], "rows": []}
    for sid in sids:
        d = extract(NWB / f"session_{sid}.nwb")
        if d is None:
            summ["without_pupil"].append(sid); print(sid, "no eye tracking", flush=True); continue
        c = np.load(COUNTS / f"session_{sid}.npz")
        assert len(c["start"]) == len(d["start"]) and np.allclose(c["start"], d["start"]), sid
        np.savez_compressed(OUT / f"session_{sid}.npz", **d)
        summ["with_pupil"].append(sid)
        row = {"session": sid, "n": int(len(d["pupil"])), "n_nan": int(np.isnan(d["pupil"]).sum()),
               "samples_median": float(np.median(d["n_samples"])), "valid_frac_median": float(np.median(d["valid_frac"])),
               "pupil_median": float(np.nanmedian(d["pupil"])), "pupil_cv": float(np.nanstd(d["pupil"]) / np.nanmean(d["pupil"]))}
        summ["rows"].append(row); print(row, flush=True)
    json.dump(summ, open(HERE.parent / "data_canonical" / "h5b_allen_pupil.json", "w"), indent=1)
    print(f"DONE: {len(summ['with_pupil'])} with pupil, {len(summ['without_pupil'])} without", flush=True)


if __name__ == "__main__":
    main()
