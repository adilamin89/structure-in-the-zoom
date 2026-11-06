"""Fresh-clone fallback for allen_counts_modal.py (no Modal): fetch each Allen Brain Observatory Neuropixels session NWB from
Allen's public S3 bucket, extract the same per-presentation spike counts, temporal frequencies and unit quality metrics, and
write the same session_<sid>.npz (identical arrays; checked on session 715093703). Session ids are read from
data_canonical/allen_expansion_all_sessions.json. Each NWB is ~2-3 GB; it is deleted after extraction unless --keep.
Run: python allen_counts_local.py [sid ...] [--keep]
Out: ../raw/allen_counts/session_<sid>.npz"""
import json
import sys
import urllib.request
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "raw" / "allen_counts"
NWB = HERE.parent / "raw" / "allen_nwb"
S3 = "https://allen-brain-observatory.s3.us-west-2.amazonaws.com/visual-coding-neuropixels/ecephys-cache"


def extract(path):
    f = h5py.File(path, "r")
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
    return dict(counts=counts, orientation=ori, temporal_frequency=tf, start=start, area=area,
                isi_violations=u["isi_violations"][:], amplitude_cutoff=u["amplitude_cutoff"][:],
                presence_ratio=u["presence_ratio"][:], unit_id=u["id"][:])


def main():
    keep_nwb = "--keep" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--keep"]
    sids = [int(a) for a in args] or [s["session"] for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
    OUT.mkdir(parents=True, exist_ok=True); NWB.mkdir(parents=True, exist_ok=True)
    for sid in sids:
        out = OUT / f"session_{sid}.npz"
        if out.exists():
            print(f"[{sid}] exists, skipped", flush=True); continue
        path = NWB / f"session_{sid}.nwb"
        if not path.exists():
            print(f"[{sid}] downloading", flush=True)
            urllib.request.urlretrieve(f"{S3}/session_{sid}/session_{sid}.nwb", str(path) + ".part")
            Path(str(path) + ".part").rename(path)
        np.savez_compressed(out, **extract(path))
        if not keep_nwb:
            path.unlink()
        print(f"[{sid}] DONE", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
