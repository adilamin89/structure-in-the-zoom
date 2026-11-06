"""Allen Neuropixels drifting-grating spike counts per presentation, every unit, with temporal frequency and
unit quality metrics, so that every Allen test runs locally on one saved input.

Same extraction as allen_expansion.py / run66 (counts in [start_time, stop_time) of each drifting-grating presentation with
a finite orientation; unit area = location of its peak channel; units in NWB order), plus the presentation's temporal
frequency and each unit's isi_violations, amplitude_cutoff, presence_ratio. No unit filter here (the paper's populations
use all units); the quality arm filters downstream.

Run (from this directory): modal run allen_counts_modal.py [--sids 715093703]
Out: Volume allen-neuropixels-data /h1_counts/session_<sid>.npz, copied to
     ../raw/allen_counts/session_<sid>.npz (raw intermediate; not in git, not in the iCloud backup)."""
import os
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "raw" / "allen_counts"   # written by allen_counts_modal.py from the Allen Brain Observatory NWBs
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "h5py")
app = modal.App("h1-allen-counts")
vol = modal.Volume.from_name("allen-neuropixels-data", create_if_missing=False)


@app.function(image=image, volumes={"/data": vol}, timeout=1800, cpu=2, memory=16384)
def extract(sid: int) -> bytes:
    import io
    import h5py
    import numpy as np

    f = h5py.File(f"/data/nwb/session_{sid}.nwb", "r")
    u = f["units"]
    st, sti, peak = u["spike_times"][:], u["spike_times_index"][:], u["peak_channel_id"][:]
    el = f["general/extracellular_ephys/electrodes"]
    ch_to_loc = dict(zip(el["id"][:], [x.decode() if isinstance(x, bytes) else str(x) for x in el["location"][:]]))
    area = np.array([ch_to_loc.get(c, "") for c in peak])
    dgn = [k for k in f["intervals"].keys() if "drifting_gratings" in k and "contrast" not in k]
    assert len(dgn) == 1, dgn
    dg = f["intervals"][dgn[0]]
    ori = np.asarray(dg["orientation"][:], float); tf = np.asarray(dg["temporal_frequency"][:], float)
    keep = np.isfinite(ori)
    start, stop = dg["start_time"][:][keep], dg["stop_time"][:][keep]
    ori, tf = ori[keep], tf[keep]
    assert np.isfinite(tf).all()
    bounds = np.concatenate([[0], sti])
    counts = np.zeros((len(ori), len(sti)), dtype=np.uint16)
    for j in range(len(sti)):
        sp = np.sort(st[bounds[j]:bounds[j + 1]])
        counts[:, j] = np.searchsorted(sp, stop) - np.searchsorted(sp, start)
    buf = io.BytesIO()
    np.savez_compressed(buf, counts=counts, orientation=ori, temporal_frequency=tf, start=start, area=area,
                        isi_violations=u["isi_violations"][:], amplitude_cutoff=u["amplitude_cutoff"][:],
                        presence_ratio=u["presence_ratio"][:], unit_id=u["id"][:])
    os.makedirs("/data/h1_counts", exist_ok=True)
    open(f"/data/h1_counts/session_{sid}.npz", "wb").write(buf.getvalue())
    vol.commit()
    print(f"[{sid}] {counts.shape} presentations x units, max count {counts.max()} DONE", flush=True)
    return buf.getvalue()


@app.local_entrypoint()
def main(sids: str = ""):
    import json
    LOCAL.mkdir(parents=True, exist_ok=True)
    exp = json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]
    todo = [int(x) for x in sids.split(",") if x] if sids else [s["session"] for s in exp]
    todo = [s for s in todo if not (LOCAL / f"session_{s}.npz").exists()]
    print(f"{len(todo)} sessions to extract", flush=True)
    for sid, blob in zip(todo, extract.map(todo)):
        (LOCAL / f"session_{sid}.npz").write_bytes(blob)
        print(f"saved session_{sid}.npz ({len(blob) / 1e6:.1f} MB)", flush=True)
    print("ALL DONE", flush=True)
