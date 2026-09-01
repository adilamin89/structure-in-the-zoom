"""Modal run - per-population within-class PR for the Allen partial-correlation
control (the one control run1 could not do from stored JSONs).

For every session-area population (>=50 units, same filter as the paper), on
the same spike-count matrix as allen_expansion.py: the centered PR of each
single direction class, the first ladder class (rung 1), and the within-class
trial correlation.

REGISTERED PREDICTION (before run, the two-knobs test): the delta-quadrupole
association across the 167 populations WEAKENS substantially when rung-1
within-class PR is partialled out (predict partial r < +0.20, vs pooled
+0.41), because the fitted quadrupole amplitude proxies within-class geometry
(tuned-gain co-rotation), while it SURVIVES partialling the dipole (+0.47
measured in run1). If instead the partial survives unchanged, the co-rotation
account of Section 7 loses its Allen support and the paper's proxy sentence
must be revised.

Run:  modal run modal_allen_withinclass_pr.py
Out:  /data/results_wcpr/session_<sid>.json on Volume allen-neuropixels-data,
      aggregated locally to feedback_runs/allen_withinclass_pr.json
"""
import json
from pathlib import Path

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy", "h5py", "requests")
app = modal.App("allen-withinclass-pr")
vol = modal.Volume.from_name("allen-neuropixels-data", create_if_missing=False)

S3 = ("https://allen-brain-observatory.s3.us-west-2.amazonaws.com/"
      "visual-coding-neuropixels/ecephys-cache")
MIN_UNITS = 50


@app.function(image=image, volumes={"/data": vol}, timeout=3600, cpu=4,
              memory=32768)
def process_session(sid: int):
    import os
    import time

    import h5py
    import numpy as np
    import requests

    t0 = time.time()
    os.makedirs("/data/results_wcpr", exist_ok=True)
    out_path = f"/data/results_wcpr/session_{sid}.json"
    if os.path.exists(out_path):
        return json.load(open(out_path))

    nwb_path = f"/data/nwb/session_{sid}.nwb"
    if not os.path.exists(nwb_path):
        url = f"{S3}/session_{sid}/session_{sid}.nwb"
        print(f"[{sid}] downloading (was not on volume)", flush=True)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(nwb_path + ".part", "wb") as fo:
                for chunk in r.iter_content(chunk_size=1 << 22):
                    fo.write(chunk)
        os.rename(nwb_path + ".part", nwb_path)
        vol.commit()

    f = h5py.File(nwb_path, "r")
    units = f["units"]
    st = units["spike_times"][:]
    sti = units["spike_times_index"][:]
    peak_ch = units["peak_channel_id"][:]
    el = f["general/extracellular_ephys/electrodes"]
    ch_to_loc = dict(zip(el["id"][:],
                         [x.decode() if isinstance(x, bytes) else str(x)
                          for x in el["location"][:]]))
    unit_loc = np.array([ch_to_loc.get(c, "") for c in peak_ch])

    dg_name = [k for k in f["intervals"].keys() if "drifting_gratings" in k
               and "contrast" not in k]
    if not dg_name:
        out = {"session": sid, "status": "no_drifting_gratings"}
        json.dump(out, open(out_path, "w")); vol.commit()
        return out
    dg = f["intervals"][dg_name[0]]
    start, stop = dg["start_time"][:], dg["stop_time"][:]
    ori_raw = dg["orientation"][:]
    ori = np.array([float(x) if not isinstance(x, bytes) else
                    (np.nan if x in (b"null", b"") else float(x))
                    for x in ori_raw])
    keep = np.isfinite(ori)
    start, stop, ori = start[keep], stop[keep], ori[keep]
    dirs = np.sort(np.unique(ori))

    n_units = len(sti)
    bounds = np.concatenate([[0], sti])
    counts = np.zeros((len(ori), n_units), dtype=np.float32)
    for j in range(n_units):
        sp = np.sort(st[bounds[j]:bounds[j + 1]])
        counts[:, j] = np.searchsorted(sp, stop) - np.searchsorted(sp, start)

    def pr_trace(X):
        Xc = X - X.mean(axis=0)
        n = Xc.shape[0]
        if n < 3:
            return 1.0
        G = (Xc @ Xc.T).astype(np.float64) / (n - 1)
        tr, tr2 = float(np.trace(G)), float((G * G).sum())
        return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0

    out = {"session": sid, "status": "ok", "areas": {}}
    for area in sorted(set(unit_loc)):
        if not area.startswith("VIS"):
            continue
        ucols = np.where(unit_loc == area)[0]
        if len(ucols) < MIN_UNITS:
            continue
        X = counts[:, ucols]
        cls_prs, cls_cors = [], []
        for d in dirs:
            idx = np.where(ori == d)[0]
            V = X[idx]
            cls_prs.append(pr_trace(V))
            Vn = V - V.mean(axis=1, keepdims=True)
            Vn /= np.linalg.norm(Vn, axis=1, keepdims=True) + 1e-9
            Cm = Vn @ Vn.T
            m = len(idx)
            cls_cors.append(float((Cm.sum() - m) / (m * (m - 1))))
        out["areas"][area] = {
            "n_units": int(len(ucols)),
            "pr_within_rung1": cls_prs[0],
            "pr_within_mean": float(np.mean(cls_prs)),
            "pr_within_per_class": cls_prs,
            "corr_within_mean": float(np.mean(cls_cors)),
        }
    json.dump(out, open(out_path, "w"))
    vol.commit()
    print(f"[{sid}] done {len(out['areas'])} areas ({time.time()-t0:.0f}s)",
          flush=True)
    return out


@app.local_entrypoint()
def main():
    here = Path(__file__).resolve().parent
    exp = json.load(open(here.parent / "data_canonical" /
                         "allen_expansion_all_sessions.json"))["results"]
    sids = [s["session"] for s in exp]
    print(f"{len(sids)} sessions", flush=True)
    results = list(process_session.map(sids))
    json.dump({"results": results},
              open(here / "allen_withinclass_pr.json", "w"), indent=1)
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"DONE modal: {ok}/{len(results)} sessions ok", flush=True)
