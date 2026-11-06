"""H1-R33b (Oct 22): running speed per Allen drifting-grating presentation, aligned with h1_allen_counts (same presentations,
finite orientation, NWB order): mean of processing/running/running_speed/data over [start_time, stop_time).
Run: modal run h1_allen_running_modal.py
Out: ../raw/allen_running/session_<sid>.npz (speed, start)"""
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "raw" / "allen_running"   # written by h1_allen_running_modal.py from the Allen Brain Observatory NWBs
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "h5py")
app = modal.App("h1-allen-running")
vol = modal.Volume.from_name("allen-neuropixels-data", create_if_missing=False)


@app.function(image=image, volumes={"/data": vol}, timeout=1200, cpu=1, memory=8192)
def extract(sid: int) -> bytes:
    import io
    import h5py
    import numpy as np
    f = h5py.File(f"/data/nwb/session_{sid}.nwb", "r")
    dgn = [k for k in f["intervals"].keys() if "drifting_gratings" in k and "contrast" not in k]
    dg = f["intervals"][dgn[0]]
    ori = np.asarray(dg["orientation"][:], float); keep = np.isfinite(ori)
    start, stop = dg["start_time"][:][keep], dg["stop_time"][:][keep]
    rs = f["processing/running/running_speed"]; v, t = rs["data"][:], rs["timestamps"][:]
    o = np.argsort(t); v, t = v[o], t[o]
    i0, i1 = np.searchsorted(t, start), np.searchsorted(t, stop)
    cs = np.concatenate([[0], np.cumsum(np.nan_to_num(v))])
    speed = (cs[i1] - cs[i0]) / np.maximum(i1 - i0, 1)
    buf = io.BytesIO(); np.savez_compressed(buf, speed=speed, start=start, n_samples=i1 - i0)
    print(f"[{sid}] {len(speed)} presentations, mean speed {speed.mean():.2f} DONE", flush=True)
    return buf.getvalue()


@app.local_entrypoint()
def main():
    import json
    LOCAL.mkdir(parents=True, exist_ok=True)
    sids = [s["session"] for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
    for sid, blob in zip(sids, extract.map(sids)):
        (LOCAL / f"session_{sid}.npz").write_bytes(blob)
    print("ALL DONE", flush=True)
