"""Runs on data the paper never saw: trials x cells arrays from Allen Brain Observatory Visual Coding two-photon NWBs
(session A, drifting gratings). The release is NWB 1.0.5, which pynwb does not read; h5py does.

Fetch each experiment's NWB from the public bucket (the 61 ids, lines and depths are in data_canonical/heldout_allen2p_experiments.json):
  curl -o raw/allen2p_nwb/<id>.nwb https://allen-brain-observatory.s3.amazonaws.com/visual-coding-2p/ophys_experiment_data/<id>.nwb
(about 0.4-0.6 GB each). Reads each NWB with h5py (no allensdk) and writes, per experiment, the inputs `rung data` takes:
  X.npy        presentations x ROIs, the release's own response (AllenSDK StimulusAnalysis.get_sweep_response):
               corrected F = demixed trace - r * neuropil (pipeline >= 2.0), per presentation
               100 * (F / F0 - 1) with F0 the mean over the 30 frames before the start, averaged over the 60
               frames from the start (sweeplength 60, interlength 30, extralength 0 in drifting_gratings.py)
  X_dff.npy    the same window on the release's DfOverF trace, minus its 30-frame pre-stimulus mean (a variant)
  X_nobase.npy the release's DfOverF averaged over the window, no per-presentation baseline (--nobase-only adds it)
  labels.npy   direction class 0..7 (orientation / 45 deg), angular order
  strata.npy   temporal-frequency index 0..4 (1, 2, 4, 8, 15 Hz)
  times.npy    presentation [start, stop] in seconds (acquisition order: rows are sorted by start)
  running.npy  mean running speed over each presentation (cm/s)
  pairs.json   {c: (c + 4) % 8}
  meta.json    counts, line, depth, donor, and every check below
Blank sweeps are dropped. Checks recorded in meta.json (scanned before any axis is read): NaN or non-finite values per
ROI, ROIs whose F0 <= 0 in any presentation (the ratio is undefined there; those ROIs are dropped and counted), and dead
stretches (presentations where the population's summed |response| is below 1% of its median).

On 704298735 the response matches the release's own precomputed mean_sweep_response_dg (its analysis file) to 5e-5.

Usage: python scripts_canonical/heldout_allen2p_extract.py raw/allen2p_nwb raw/allen2p data_canonical/heldout_allen2p_experiments.json [--nobase-only]
Then, per experiment E:  rung data raw/allen2p/E/X.npy raw/allen2p/E/labels.npy --antipode raw/allen2p/E/pairs.json --sectors
  --decompose --cov-floors --strata raw/allen2p/E/strata.npy --time-blocks 48 --out data_canonical/heldout_allen2p/E.json
"""
import json
import os
import sys

import h5py
import numpy as np

SWEEP, INTER = 60, 30
PIPE = "processing/brain_observatory_pipeline"


def extract(nwb_path):
    with h5py.File(nwb_path, "r") as f:
        g = f["stimulus/presentation/drifting_gratings_stimulus"]
        feats = [x.decode() for x in g["features"][()]]
        assert feats == ["temporal_frequency", "orientation", "blank_sweep"], feats
        d = g["data"][()]
        fd = g["frame_duration"][()].astype(int)
        version = [x.decode() for x in f["general/generated_by"][()]]
        vnum = float(version[version.index("version") + 1])
        fl = f[f"{PIPE}/Fluorescence/imaging_plane_1"]
        if vnum >= 2.0:
            raw = f[f"{PIPE}/Fluorescence/imaging_plane_1_demixed_signal/data"][()]
        else:
            raw = fl["data"][()]
        neu = fl["neuropil_traces"][()]
        r = fl["r"][()]
        ts = fl["timestamps"][()]
        dff = f[f"{PIPE}/DfOverF/imaging_plane_1/data"][()]
        cell_ids = f[f"{PIPE}/DfOverF/imaging_plane_1/segmentation_interface/cell_specimen_ids"][()]
        run = f[f"{PIPE}/BehavioralTimeSeries/running_speed/data"][()]
        run_ts = f[f"{PIPE}/BehavioralTimeSeries/running_speed/timestamps"][()]
    assert raw.shape == neu.shape == dff.shape and raw.shape[1] == ts.size, (raw.shape, neu.shape, dff.shape, ts.size)
    F = raw - neu * r[:, None]
    keep = d[:, 2] == 0
    keep &= np.isfinite(d[:, 1])
    d, fd = d[keep], fd[keep]
    order = np.argsort(fd[:, 0], kind="stable")
    d, fd = d[order], fd[order]
    ok = (fd[:, 0] - INTER >= 0) & (fd[:, 0] + SWEEP <= F.shape[1])
    d, fd = d[ok], fd[ok]
    n = len(d)
    starts = fd[:, 0]
    pre = np.stack([F[:, s - INTER:s].mean(1) for s in starts])           # n x cells
    win = np.stack([F[:, s:s + SWEEP].mean(1) for s in starts])
    pre_d = np.stack([dff[:, s - INTER:s].mean(1) for s in starts])
    win_d = np.stack([dff[:, s:s + SWEEP].mean(1) for s in starts])
    bad_f0 = (pre <= 0).any(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        X = 100.0 * (win / pre - 1.0)
    X_dff = win_d - pre_d
    X_nobase = win_d                      # the release's DfOverF over the window, no per-presentation baseline
    nonfinite = ~np.isfinite(X).all(0) | ~np.isfinite(X_dff).all(0)
    drop = bad_f0 | nonfinite
    X, X_dff, X_nobase = X[:, ~drop], X_dff[:, ~drop], X_nobase[:, ~drop]
    labels = np.rint(d[:, 1] / 45.0).astype(int)
    assert set(np.unique(labels)) <= set(range(8)), np.unique(d[:, 1])
    tf_vals = np.array([1.0, 2.0, 4.0, 8.0, 15.0])
    strata = np.array([int(np.argmin(np.abs(tf_vals - t))) for t in d[:, 0]])
    assert np.allclose(tf_vals[strata], d[:, 0])
    times = np.stack([ts[fd[:, 0]], ts[np.minimum(fd[:, 0] + SWEEP, ts.size - 1)]], 1)
    running = np.array([np.nanmean(run[(run_ts >= a) & (run_ts < b)]) if ((run_ts >= a) & (run_ts < b)).any() else np.nan
                        for a, b in times])
    tot = np.abs(X).sum(1)
    dead = tot < 0.01 * np.median(tot)
    counts = np.bincount(labels, minlength=8)
    meta = dict(pipeline_version=vnum, n_presentations=int(n), n_rois_total=int(F.shape[0]), n_rois_kept=int(X.shape[1]),
                n_rois_dropped_f0=int(bad_f0.sum()), n_rois_dropped_nonfinite=int((nonfinite & ~bad_f0).sum()),
                per_class=counts.tolist(), per_tf=np.bincount(strata, minlength=5).tolist(),
                n_dead_presentations=int(dead.sum()), running_nan=int(np.isnan(running).sum()),
                kept_cell_specimen_ids=[int(c) for c in cell_ids[~drop]])
    return X, X_dff, X_nobase, labels, strata, times, running, meta


def main():
    nwb_dir, out_dir, meta_json = sys.argv[1:4]
    info = {str(x["id"]): x for x in json.load(open(meta_json))}
    nobase_only = "--nobase-only" in sys.argv     # add X_nobase.npy to experiments already extracted
    for fn in sorted(os.listdir(nwb_dir)):
        if not fn.endswith(".nwb"):
            continue
        eid = fn[:-4]
        od = os.path.join(out_dir, eid)
        if nobase_only:
            if os.path.exists(os.path.join(od, "X_nobase.npy")):
                continue
            X, X_dff, X_nobase, labels, *_ = extract(os.path.join(nwb_dir, fn))
            assert np.array_equal(labels, np.load(os.path.join(od, "labels.npy")))
            assert np.allclose(X, np.load(os.path.join(od, "X.npy")))
            np.save(os.path.join(od, "X_nobase.npy"), X_nobase)
            print(f"NOBASE {eid} {X_nobase.shape}", flush=True)
            continue
        if os.path.exists(os.path.join(od, "meta.json")):
            continue
        X, X_dff, X_nobase, labels, strata, times, running, meta = extract(os.path.join(nwb_dir, fn))
        os.makedirs(od, exist_ok=True)
        for name, arr in [("X", X), ("X_dff", X_dff), ("X_nobase", X_nobase), ("labels", labels), ("strata", strata), ("times", times),
                          ("running", running)]:
            np.save(os.path.join(od, name + ".npy"), arr)
        json.dump({str(c): (c + 4) % 8 for c in range(8)}, open(os.path.join(od, "pairs.json"), "w"))
        meta.update(info.get(eid, {}))
        json.dump(meta, open(os.path.join(od, "meta.json"), "w"), indent=1)
        print(f"EXTRACTED {eid} rois {meta['n_rois_kept']}/{meta['n_rois_total']} per_class {meta['per_class']} "
              f"dead {meta['n_dead_presentations']} f0drop {meta['n_rois_dropped_f0']}", flush=True)


if __name__ == "__main__":
    main()
