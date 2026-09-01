"""Allen split-half cross-fit of the quadrupole-delta correlation
+ 10-permutation shuffle with a reused floor.

Motivation: b_quadrupole and delta_dir8 computed from the same trials share
measurement noise and realized-tuning reliability across 167 populations;
and the 3-permutation shuffle recomputed a fresh floor per permutation.

Design, per session x area (>=50 units), NWBs already on the Volume:
  - split presentations into halves A/B, stratified by direction class
    (interleaved even/odd within class -> balanced across time);
  - multipoles (a, c_dipole, b_quadrupole, b4) from class means of half A,
    delta_dir8 from half B (own 10-draw floor), and the swap (B->multipoles,
    A->delta);
  - shuffle control on the full session: 10 label permutations, floor computed
    ONCE and reused (sizes identical under permutation).

REGISTERED EXPECTATIONS (before run, 2026-08-23):
  R4-E1: cross-fitted pooled corr(b_half1, delta_half2) averaged over the two
         directions is POSITIVE and > +0.25 across the ~167 populations
         (attenuated from the same-trial +0.41 by split noise, not erased).
  R4-E2: VISp remains overwhelmingly positive in each half (>= 28/32).
  R4-E3: 10-permutation shuffle means stay near zero (|area mean| < 0.03).

Run: modal run modal_allen_crossfit.py
Output: /data/crossfit/session_<sid>.json on the Volume (kill-safe) +
        data/allen_crossfit.json locally.
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "h5py", "requests")
app = modal.App("allen-crossfit-theta")
vol = modal.Volume.from_name("allen-neuropixels-data", create_if_missing=False)

BIN_COUNTS_DIR = [1, 2, 3, 4, 6, 8]
N_NULL = 10
N_SHUFFLE = 10
MIN_UNITS = 50


@app.function(image=image, volumes={"/data": vol}, timeout=7200, cpu=4, memory=32768)
def process_session(sid: int):
    import json
    import os
    import time

    import h5py
    import numpy as np

    t0 = time.time()
    os.makedirs("/data/crossfit", exist_ok=True)
    out_path = f"/data/crossfit/session_{sid}.json"
    if os.path.exists(out_path):
        vol.reload()
        return json.load(open(out_path))
    nwb_path = f"/data/nwb/session_{sid}.nwb"
    if not os.path.exists(nwb_path):
        return {"session": sid, "status": "nwb_missing"}

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
        return {"session": sid, "status": "no_drifting_gratings"}
    dg = f["intervals"][dg_name[0]]
    start, stop = dg["start_time"][:], dg["stop_time"][:]
    ori_raw = dg["orientation"][:]
    ori = np.array([float(x) if not isinstance(x, bytes) else
                    (np.nan if x in (b"null", b"") else float(x)) for x in ori_raw])
    keep = np.isfinite(ori)
    start, stop, ori = start[keep], stop[keep], ori[keep]
    dirs = np.sort(np.unique(ori))
    n_units = len(sti)
    bounds = np.concatenate([[0], sti])
    counts = np.zeros((len(ori), n_units), dtype=np.float32)
    for j in range(n_units):
        sp = np.sort(st[bounds[j]:bounds[j + 1]])
        counts[:, j] = (np.searchsorted(sp, stop) - np.searchsorted(sp, start))
    f.close()

    def pr_trace(X):
        Xc = X - X.mean(axis=0)
        n = Xc.shape[0]
        if n < 3:
            return 1.0
        G = (Xc @ Xc.T).astype(np.float64) / (n - 1)
        tr, tr2 = float(np.trace(G)), float((G * G).sum())
        return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0

    def slope(sizes, prs):
        x = np.log(np.asarray(sizes, dtype=float))
        y = np.log(np.maximum(np.asarray(prs, dtype=float), 1e-9))
        A = np.vstack([np.ones_like(x), x]).T
        return float(np.linalg.lstsq(A, y, rcond=None)[0][1])

    def ladder_delta(X, labels, rng, floor_logs=None):
        members = [np.where(labels == c)[0] for c in dirs]
        sizes, prs = [], []
        for c in BIN_COUNTS_DIR:
            sel = np.concatenate(members[:c])
            if len(sel) < 10:
                continue
            sizes.append(len(sel))
            prs.append(pr_trace(X[sel]))
        if len(sizes) < 3:
            return None, None, None
        th_o = slope(sizes, prs)
        if floor_logs is None:
            floor_logs = np.zeros((N_NULL, len(sizes)))
            for d in range(N_NULL):
                for k, s in enumerate(sizes):
                    floor_logs[d, k] = np.log(max(pr_trace(
                        X[rng.choice(len(X), s, replace=False)]), 1e-9))
        th_f = slope(sizes, np.exp(floor_logs.mean(axis=0)))
        return th_o - th_f, floor_logs, sizes

    def multipoles(X, labels):
        M = np.stack([X[labels == c].mean(axis=0) for c in dirs])
        Mc = M - M.mean(axis=1, keepdims=True)
        Mc /= np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-9
        C = Mc @ Mc.T
        ang = np.deg2rad(dirs)
        dphi = np.abs(ang[:, None] - ang[None, :])
        dphi = np.minimum(dphi, 2 * np.pi - dphi)
        iu = np.triu_indices(len(dirs), k=1)
        d, c = dphi[iu], C[iu]
        A = np.vstack([np.ones_like(d), np.cos(d), np.cos(2 * d), np.cos(4 * d)]).T
        coef, *_ = np.linalg.lstsq(A, c, rcond=None)
        return {"a": float(coef[0]), "c_dipole": float(coef[1]),
                "b_quadrupole": float(coef[2]), "b4": float(coef[3])}

    rng = np.random.default_rng(42)
    result = {"session": sid, "status": "ok", "areas": {}}
    for area in sorted(set(unit_loc)):
        if not area.startswith("VIS"):
            continue
        idx = np.where(unit_loc == area)[0]
        if len(idx) < MIN_UNITS:
            continue
        X = counts[:, idx]
        # stratified interleaved split
        half = np.zeros(len(ori), dtype=bool)
        for c in dirs:
            m = np.where(ori == c)[0]
            half[m[::2]] = True
        XA, oA = X[half], ori[half]
        XB, oB = X[~half], ori[~half]
        dA, _, _ = ladder_delta(XA, oA, np.random.default_rng(101))
        dB, _, _ = ladder_delta(XB, oB, np.random.default_rng(102))
        mpA, mpB = multipoles(XA, oA), multipoles(XB, oB)
        # full-session shuffle with reused floor
        d_full, floor_logs, _ = ladder_delta(X, ori, np.random.default_rng(103))
        shuf = []
        for s in range(N_SHUFFLE):
            srng = np.random.default_rng(500 + s)
            ds, _, _ = ladder_delta(X, ori[srng.permutation(len(ori))],
                                    None, floor_logs=floor_logs)
            if ds is not None:
                shuf.append(ds)
        result["areas"][area] = {
            "n_units": int(len(idx)),
            "delta_A": dA, "delta_B": dB,
            "b_quad_A": mpA["b_quadrupole"], "b_quad_B": mpB["b_quadrupole"],
            "c_dip_A": mpA["c_dipole"], "c_dip_B": mpB["c_dipole"],
            "delta_full": d_full,
            "shuffled_mean_10": float(np.mean(shuf)) if shuf else None,
            "shuffled_sd_10": float(np.std(shuf)) if shuf else None,
        }
        print(f"[{sid}] {area}: n={len(idx)} dA={dA} dB={dB} "
              f"bA={mpA['b_quadrupole']:.3f} bB={mpB['b_quadrupole']:.3f} "
              f"shuf10={result['areas'][area]['shuffled_mean_10']}", flush=True)

    with open(out_path, "w") as fo:
        json.dump(result, fo, indent=1)
    vol.commit()
    print(f"[{sid}] DONE ({time.time()-t0:.0f}s)", flush=True)
    return result


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def list_downloaded():
    import os
    return sorted(int(p.split("_")[1].split(".")[0])
                  for p in os.listdir("/data/nwb") if p.endswith(".nwb"))


@app.local_entrypoint()
def main():
    import json

    import numpy as np
    sids = list_downloaded.remote()
    print(f"{len(sids)} NWBs on volume")
    results = list(process_session.map(sids, return_exceptions=True))
    ok = [r for r in results if isinstance(r, dict) and r.get("status") == "ok"]
    rows = []
    for r in ok:
        for area, a in r["areas"].items():
            if a["delta_A"] is None or a["delta_B"] is None:
                continue
            rows.append(dict(session=r["session"], area=area, **a))

    def corr(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).sum() / (np.linalg.norm(x) * np.linalg.norm(y)))

    r1 = corr([r["b_quad_A"] for r in rows], [r["delta_B"] for r in rows])
    r2 = corr([r["b_quad_B"] for r in rows], [r["delta_A"] for r in rows])
    summary = {"registration": "modal_allen_crossfit.py docstring (pre-run)",
               "n_populations": len(rows),
               "crossfit_r_AtoB": r1, "crossfit_r_BtoA": r2,
               "crossfit_r_mean": (r1 + r2) / 2,
               "results": results}
    with open("data/allen_crossfit.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=1))
    print("Saved data/allen_crossfit.json")
