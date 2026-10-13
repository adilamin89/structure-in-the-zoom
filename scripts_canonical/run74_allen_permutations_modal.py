"""Run 74 (2026-10-05): 200 label permutations per Allen population for the direction-aligned shift (reviewer tYDC).
Registration: ../data_canonical/run74_REGISTERED.md. Spike counts built exactly as run66 / allen_expansion.py.
Run (profile adila-75793): MODAL_PROFILE=adila-75793 modal run run74_allen_permutations_modal.py
Out: /data/results_perm200/session_<sid>.json on Volume allen-neuropixels-data, aggregated to
../data_canonical/run74_allen_permutations.json"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "h5py")
app = modal.App("allen-perm200")
vol = modal.Volume.from_name("allen-neuropixels-data", create_if_missing=False)
MIN_UNITS, RUNGS, N_NULL, N_PERM = 50, (1, 2, 3, 4, 6, 8), 50, 200


@app.function(image=image, volumes={"/data": vol}, timeout=3600, cpu=2, memory=16384)
def process_session(sid: int):
    import os
    import time

    import h5py
    import numpy as np

    t0 = time.time()
    os.makedirs("/data/results_perm200", exist_ok=True)
    out_path = f"/data/results_perm200/session_{sid}.json"
    if os.path.exists(out_path):
        return json.load(open(out_path))
    nwb = f"/data/nwb/session_{sid}.nwb"
    if not os.path.exists(nwb):
        return {"session": sid, "status": "nwb_missing"}
    f = h5py.File(nwb, "r")
    units = f["units"]; st = units["spike_times"][:]; sti = units["spike_times_index"][:]; peak_ch = units["peak_channel_id"][:]
    el = f["general/extracellular_ephys/electrodes"]
    ch_to_loc = dict(zip(el["id"][:], [x.decode() if isinstance(x, bytes) else str(x) for x in el["location"][:]]))
    unit_loc = np.array([ch_to_loc.get(c, "") for c in peak_ch])
    dg_name = [k for k in f["intervals"].keys() if "drifting_gratings" in k and "contrast" not in k]
    if not dg_name:
        return {"session": sid, "status": "no_drifting_gratings"}
    dg = f["intervals"][dg_name[0]]
    start, stop = dg["start_time"][:], dg["stop_time"][:]
    ori = np.array([float(x) if not isinstance(x, bytes) else (np.nan if x in (b"null", b"") else float(x)) for x in dg["orientation"][:]])
    keep = np.isfinite(ori); start, stop, ori = start[keep], stop[keep], ori[keep]
    dirs = np.sort(np.unique(ori)); labels = np.searchsorted(dirs, ori)
    bounds = np.concatenate([[0], sti]); counts = np.zeros((len(ori), len(sti)), dtype=np.float32)
    for j in range(len(sti)):
        sp = np.sort(st[bounds[j]:bounds[j + 1]]); counts[:, j] = np.searchsorted(sp, stop) - np.searchsorted(sp, start)

    def pr_sub(G, idx):
        S = G[np.ix_(idx, idx)]; r = S.mean(0)
        C = S - r[None, :] - r[:, None] + r.mean()
        return float(np.trace(C) ** 2 / (C * C).sum())

    def ladder_slope(G, lab, logn):
        sets = [np.concatenate([np.where(lab == b)[0] for b in range(c)]) for c in RUNGS]
        return float(np.polyfit(logn, [np.log(pr_sub(G, s)) for s in sets], 1)[0])

    out = {"session": sid, "status": "ok", "areas": {}}
    for area in sorted(set(unit_loc)):
        if not area.startswith("VIS"):
            continue
        cols = np.where(unit_loc == area)[0]
        if len(cols) < MIN_UNITS:
            continue
        X = counts[:, cols].astype(np.float64); G = X @ X.T
        sizes = [int(sum((labels == b).sum() for b in range(c))) for c in RUNGS]; logn = np.log(sizes)
        rng = np.random.default_rng(74)
        floor = float(np.polyfit(logn, [np.mean([np.log(pr_sub(G, rng.choice(len(X), s, replace=False))) for _ in range(N_NULL)]) for s in sizes], 1)[0])
        obs = ladder_slope(G, labels, logn) - floor
        perm = [ladder_slope(G, rng.permutation(labels), logn) - floor for _ in range(N_PERM)]
        p = (1 + sum(x >= obs for x in perm)) / (N_PERM + 1)
        out["areas"][area] = {"n_units": int(len(cols)), "delta": obs, "p": float(p), "perm_mean": float(np.mean(perm)), "perm_sd": float(np.std(perm)),
                              "z": float((obs - np.mean(perm)) / (np.std(perm) + 1e-12))}
        print(f"[{sid}] {area} N={len(cols)} delta {obs:+.3f} perm {np.mean(perm):+.3f}±{np.std(perm):.3f} p={p:.4f} | {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(out_path, "w")); vol.commit()
    return out


@app.local_entrypoint()
def main():
    import numpy as np
    data = HERE.parent / "data_canonical"
    exp = json.load(open(data / "allen_expansion_all_sessions.json"))["results"]
    stored = {(s["session"], a): v["delta_dir8"] for s in exp for a, v in s.get("areas", {}).items()}
    results = list(process_session.map([s["session"] for s in exp]))
    pops = [(r["session"], a, v) for r in results if r.get("status") == "ok" for a, v in r["areas"].items()]
    diff = [abs(v["delta"] - stored[(s, a)]) for s, a, v in pops if (s, a) in stored]
    visp = [v for _, a, v in pops if a == "VISp"]
    verdict = {"n_populations": len(pops), "R1_median_absdiff": float(np.median(diff)) if diff else None, "R1": bool(diff and np.median(diff) < 0.04),
               "P1_VISp_sig": int(sum(v["p"] <= 0.05 for v in visp)), "P1_VISp_n": len(visp), "P1": bool(sum(v["p"] <= 0.05 for v in visp) >= 30),
               "P2_all_sig": int(sum(v["p"] <= 0.05 for _, _, v in pops)), "P2": bool(sum(v["p"] <= 0.05 for _, _, v in pops) >= 140),
               "count_p_min": int(sum(v["p"] <= 1 / 200 for _, _, v in pops)),
               "statuses": {r["session"]: r["status"] for r in results if r.get("status") != "ok"}}
    out = {"design": {"rungs": RUNGS, "n_null": N_NULL, "n_perm": N_PERM, "min_units": MIN_UNITS}, "results": results, "verdict": verdict}
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(data / "run74_allen_permutations.json", "w"), indent=1)
