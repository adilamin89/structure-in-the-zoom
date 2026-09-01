"""Allen Neuropixels EXPANSION: orientation-zoom delta + full physics, all 32
brain_observatory_1.1 sessions, per visual area. NO allensdk (direct S3 + h5py).

Lessons applied (2026-08-21/22): raw NWBs persist on the Volume (never lose raw
data again) · per-session result JSONs written to the Volume as they finish
(kill-safe) · PR via traces · progress prints · parallel .map.

Per session, per area (>=50 units), on the 8-direction drifting-gratings
presentations (spike counts per presentation window):
  - dir8 ladder (accumulate 1,2,3,4,6,8 direction classes; matches the Stringer
    dir8 construction and the published VISp +0.244) + 10-draw floor
  - ori4 ladder (4 orientations, direction-collapsed; accumulate 1,2,3,4)
  - shuffled-label control (3 seeds, reuses the floor - sizes identical)
  - random-trial zoom delta
  - physics: eigenspectrum -> f_c (MP edge), n_struct, alpha fit,
    gap ratios, level-spacing ratio <r> of structural modes (Paper B feed)

Run: modal run modal_allen_expansion.py
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "h5py", "requests")
app = modal.App("allen-expansion-theta")
vol = modal.Volume.from_name("allen-neuropixels-data", create_if_missing=True)

S3 = "https://allen-brain-observatory.s3.us-west-2.amazonaws.com/visual-coding-neuropixels/ecephys-cache"
BIN_COUNTS_DIR = [1, 2, 3, 4, 6, 8]
BIN_COUNTS_ORI = [1, 2, 3, 4]
N_NULL = 10
N_SHUFFLE = 3
MIN_UNITS = 50


@app.function(image=image, volumes={"/data": vol}, timeout=7200, cpu=4, memory=32768)
def process_session(sid: int):
    import json
    import os
    import time

    import h5py
    import numpy as np
    import requests

    t0 = time.time()
    out_path = f"/data/results/session_{sid}.json"
    os.makedirs("/data/results", exist_ok=True)
    os.makedirs("/data/nwb", exist_ok=True)
    if os.path.exists(out_path):
        vol.reload()
        return json.load(open(out_path))

    # ---- download raw NWB to the Volume (persistent) ----
    nwb_path = f"/data/nwb/session_{sid}.nwb"
    if not os.path.exists(nwb_path):
        url = f"{S3}/session_{sid}/session_{sid}.nwb"
        print(f"[{sid}] downloading {url}", flush=True)
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(nwb_path + ".part", "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 22):
                    f.write(chunk)
        os.rename(nwb_path + ".part", nwb_path)
        vol.commit()
        print(f"[{sid}] downloaded {os.path.getsize(nwb_path)/1e9:.2f} GB "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- extract units, areas, drifting-gratings presentations ----
    f = h5py.File(nwb_path, "r")
    units = f["units"]
    st = units["spike_times"][:]
    sti = units["spike_times_index"][:]
    peak_ch = units["peak_channel_id"][:]
    el = f["general/extracellular_ephys/electrodes"]
    el_id = el["id"][:]
    el_loc = np.array([x.decode() if isinstance(x, bytes) else str(x)
                       for x in el["location"][:]])
    ch_to_loc = dict(zip(el_id, el_loc))
    unit_loc = np.array([ch_to_loc.get(c, "") for c in peak_ch])

    dg_name = [k for k in f["intervals"].keys() if "drifting_gratings" in k
               and "contrast" not in k]
    if not dg_name:
        return {"session": sid, "status": "no_drifting_gratings",
                "intervals": list(f["intervals"].keys())}
    dg = f["intervals"][dg_name[0]]
    start = dg["start_time"][:]
    stop = dg["stop_time"][:]
    ori_raw = dg["orientation"][:]
    ori = np.array([float(x) if not isinstance(x, bytes) else
                    (np.nan if x in (b"null", b"") else float(x)) for x in ori_raw])
    keep = np.isfinite(ori)
    start, stop, ori = start[keep], stop[keep], ori[keep]
    dirs = np.sort(np.unique(ori))
    print(f"[{sid}] dg={dg_name[0]} n_pres={len(ori)} dirs={dirs}", flush=True)

    # spike counts per presentation per unit (searchsorted on sorted spike times)
    n_units = len(sti)
    bounds = np.concatenate([[0], sti])
    counts = np.zeros((len(ori), n_units), dtype=np.float32)
    for j in range(n_units):
        sp = np.sort(st[bounds[j]:bounds[j + 1]])
        counts[:, j] = (np.searchsorted(sp, stop) - np.searchsorted(sp, start))

    def pr_trace(X):
        Xc = X - X.mean(axis=0)
        n = Xc.shape[0]
        if n < 3:
            return 1.0
        G = (Xc @ Xc.T).astype(np.float64) / (n - 1)
        tr = float(np.trace(G))
        tr2 = float((G * G).sum())
        if not np.isfinite(tr) or not np.isfinite(tr2) or tr2 <= 0:
            return 1.0
        return tr * tr / tr2

    def slope(sizes, prs):
        x = np.log(np.asarray(sizes, dtype=float))
        y = np.log(np.maximum(np.asarray(prs, dtype=float), 1e-9))
        A = np.vstack([np.ones_like(x), x]).T
        return float(np.linalg.lstsq(A, y, rcond=None)[0][1])

    def ladder(X, labels, classes, bin_counts, rng):
        members = [np.where(labels == c)[0] for c in classes]
        sizes, prs = [], []
        for c in bin_counts:
            sel = np.concatenate(members[:c])
            if len(sel) < 10:
                continue
            sizes.append(len(sel))
            prs.append(pr_trace(X[sel]))
        if len(sizes) < 3:
            return None, None
        th_o = slope(sizes, prs)
        null_logs = np.zeros((N_NULL, len(sizes)))
        for d in range(N_NULL):
            for k, s in enumerate(sizes):
                null_logs[d, k] = np.log(max(pr_trace(
                    X[rng.choice(len(X), s, replace=False)]), 1e-9))
        return th_o - slope(sizes, np.exp(null_logs.mean(axis=0))), (sizes, th_o)

    def spacing_r(vals):
        vals = np.sort(np.asarray(vals))
        gaps = np.diff(vals)
        gaps = gaps[gaps > 0]
        if len(gaps) < 5:
            return None
        rr = np.minimum(gaps[1:], gaps[:-1]) / np.maximum(gaps[1:], gaps[:-1])
        return float(np.mean(rr))

    def physics(X):
        """Full suite; persists the top eigenvalues so ALL downstream physics
        (GOE/Anderson, FSS, condensation) recomputes from the JSON alone."""
        Xc = X - X.mean(axis=0)
        n, d = Xc.shape
        G = (Xc @ Xc.T).astype(np.float64) / (n - 1)
        lam = np.linalg.eigvalsh(G)[::-1]
        lam = lam[lam > 1e-12]
        q = n / d
        edge = np.median(lam) * (1 + np.sqrt(q)) ** 2 / ((1 - np.sqrt(q)) ** 2
                                                         if q < 1 else 1.0)
        struct = lam[lam > edge]
        fc = float(struct.sum() / lam.sum()) if len(lam) else None
        k = np.arange(1, min(len(lam), 50) + 1)
        alpha = -slope(k, lam[:len(k)]) if len(lam) >= 10 else None
        gap = float(lam[3] / lam[4]) if len(lam) > 4 else None
        return {"f_c": fc, "n_struct": int(len(struct)), "alpha_top50": alpha,
                "mp_edge": float(edge),
                "r_struct": spacing_r(struct),
                "r_2x_edge": spacing_r(lam[lam > 2 * edge]),
                "r_top30": spacing_r(lam[:30]),
                "gap_l4_l5": gap, "pr_full": pr_trace(X),
                "eigenvalues_top500": lam[:500].tolist()}

    rng = np.random.default_rng(42)
    ori_mod = ori % 180.0
    oris = np.sort(np.unique(ori_mod))
    result = {"session": sid, "status": "ok", "n_presentations": int(len(ori)),
              "directions": dirs.tolist(), "areas": {}}

    for area in sorted(set(unit_loc)):
        if not area.startswith("VIS"):
            continue
        idx = np.where(unit_loc == area)[0]
        if len(idx) < MIN_UNITS:
            continue
        X = counts[:, idx]
        d_dir, meta_dir = ladder(X, ori, dirs, BIN_COUNTS_DIR, rng)
        d_ori, _ = ladder(X, ori_mod, oris, BIN_COUNTS_ORI, rng)
        shuf = []
        for s in range(N_SHUFFLE):
            srng = np.random.default_rng(500 + s)
            d_s, _ = ladder(X, ori[srng.permutation(len(ori))], dirs,
                            BIN_COUNTS_DIR, np.random.default_rng(900 + s))
            if d_s is not None:
                shuf.append(d_s)
        rand_sizes = [50, 100, 200, min(400, len(X)), len(X)]
        rl_obs = [np.mean([pr_trace(X[rng.choice(len(X), s, replace=False)])
                           for _ in range(3)]) for s in rand_sizes]
        rl_nul = [np.mean([pr_trace(X[rng.choice(len(X), s, replace=False)])
                           for _ in range(N_NULL)]) for s in rand_sizes]
        result["areas"][area] = {
            "n_units": int(len(idx)),
            "delta_dir8": d_dir, "delta_ori4": d_ori,
            "shuffled_mean": float(np.mean(shuf)) if shuf else None,
            "shuffled_sd": float(np.std(shuf)) if shuf else None,
            "delta_rand": slope(rand_sizes, rl_obs) - slope(rand_sizes, rl_nul),
            "physics": physics(X),
        }
        print(f"[{sid}] {area}: n={len(idx)} d_dir={d_dir} d_ori={d_ori} "
              f"shuf={result['areas'][area]['shuffled_mean']}", flush=True)

    f.close()
    with open(out_path, "w") as fo:
        json.dump(result, fo, indent=1)
    vol.commit()
    print(f"[{sid}] DONE ({time.time()-t0:.0f}s)", flush=True)
    return result


@app.function(image=image, timeout=600)
def list_sessions():
    import csv
    import io

    import requests
    r = requests.get(f"{S3}/sessions.csv", timeout=120)
    rows = list(csv.DictReader(io.StringIO(r.text)))
    return [int(x["id"]) for x in rows if x["session_type"] == "brain_observatory_1.1"]


@app.local_entrypoint()
def main():
    import json
    sids = list_sessions.remote()
    print(f"{len(sids)} brain_observatory_1.1 sessions")
    results = list(process_session.map(sids, return_exceptions=True))
    ok = [r for r in results if isinstance(r, dict) and r.get("status") == "ok"]
    print(f"ok: {len(ok)} / {len(results)}")
    with open("data/allen_expansion_all_sessions.json", "w") as f:
        json.dump({"n_sessions": len(sids),
                   "results": [r if isinstance(r, dict) else {"error": str(r)}
                               for r in results]}, f, indent=1)
    deltas = [a["delta_dir8"] for r in ok for a in r["areas"].values()
              if a["delta_dir8"] is not None]
    visp = [r["areas"]["VISp"]["delta_dir8"] for r in ok if "VISp" in r["areas"]
            and r["areas"]["VISp"]["delta_dir8"] is not None]
    import statistics
    if deltas:
        print(f"ALL VIS areas: n={len(deltas)} mean d_dir8={statistics.mean(deltas):+.4f}")
    if visp:
        print(f"VISp: n={len(visp)} mean={statistics.mean(visp):+.4f} "
              f"n_positive={sum(d > 0 for d in visp)}")
    print("Saved data/allen_expansion_all_sessions.json")
