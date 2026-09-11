"""Run 66 (S79, 2026-09-06) - The four-term split of the deficit on the 167
Allen Neuropixels populations (Modal), and where the within-class modes live
at the assembly scale.

WHY: on the eight two-photon recordings (10,000-21,000 neurons) the
direction-aligned shift is the pooling of class-private within-class modes
(run 63), and those modes are localized on 48-121 neurons (run 64a). The
Allen populations are spiking, 51-421 units each, at or below that assembly
scale. The same exact bookkeeping on them asks whether the deficit is still
the pooling term when the population is a few hundred units, and whether the
private modes can still look localized when N is of the order of the
assembly size (they should not: a mode on ~100 neurons out of 100 is
extended), which is the scale prediction of STEP_BACK_PHYSICS section 4.

DESIGN: the same spike-count matrix as allen_expansion.py (counts per
drifting-grating presentation, all temporal frequencies, eight directions
sorted ascending; every VIS area with >= 50 units), the eight-class direction
ladder [1, 2, 3, 4, 6, 8] in the paper's order with ten random-subset floors,
the four terms per rung and the four slopes of the shift (deficit_split.py),
three label permutations as the sampling baseline (75 trials per class at
N up to 421 make the baseline material), the private slope, and per class the
within-class participation ratio, the diagonal participation ratio and the
neuron-space participation ratio of the leading within-class eigenvector
(run 64a's statistics). The stored delta_dir8 of allen_expansion is carried
along as a cross-check on the recomputed shift.

REGISTERED EXPECTATIONS (written before the run; the population is a
session-area pair, n = 167):
A1 (the deficit is the pooling term at the population scale too): the
    pooling term is the largest-magnitude term of the excess delta-split and
    positive in at least two thirds of the populations, and the raw pooling
    term at four classes is negative in at least two thirds.
A2 (the scale prediction): the neuron-space participation ratio of the
    leading within-class mode, averaged over classes, is at least N/10 in at
    least two thirds of the populations (private modes are not localized at
    the assembly scale).
A3 (the within-class dimensionality enters through the pooling term): across
    populations the Spearman correlation of the excess pooling slope with
    the mean within-class participation ratio is <= -0.3, and its magnitude
    exceeds that of the between-class slope's correlation (App F's partial
    r = -0.47 of the shift with the within-class PR is carried by pooling).
A miss is reported at full volume.

POST-RUN NOTE (first pass): A1 missed at the registered bar (the pooling
term is the largest and positive in 93 of 167; the raw pooling term at four
classes is negative in 134 of 167); A2 missed (20 of 167: the leading
within-class mode sits on fewer than N/10 units almost everywhere), which on
raw spike counts may be rate heterogeneity (Poisson variance follows the
rate, so the top mode follows the most active units); A3 missed (the
within-class PR association rides on the between-class term, rho = -0.41,
not on pooling, -0.08). The rerun adds, descriptively: the scale-free
regrouping A + P = D + Tb of deficit_split.py, and the same localization
statistics on per-unit z-scored counts (the amplitude control). The
registered expectations are judged on the terms they named.

Run (from this directory; profile adila-75793):
    modal run run66_allen_deficit_decomposition_modal.py            # all 32 sessions
    modal run run66_allen_deficit_decomposition_modal.py --sids 715093703
Out: /data/results_deficit/session_<sid>.json on Volume allen-neuropixels-data,
     aggregated to ../data_canonical/run66_allen_deficit_decomposition.json (+ .log)
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "h5py", "requests")
         .add_local_file(str(HERE.parent.parent / "arxiv_supplement" / "rung.py"), "/root/rung.py")
         .add_local_python_source("deficit_split"))
app = modal.App("allen-deficit-decomposition")
vol = modal.Volume.from_name("allen-neuropixels-data", create_if_missing=False)

S3 = ("https://allen-brain-observatory.s3.us-west-2.amazonaws.com/"
      "visual-coding-neuropixels/ecephys-cache")
MIN_UNITS = 50
BIN_COUNTS = (1, 2, 3, 4, 6, 8)
N_NULL = 10
N_SHUFFLE = 3
TOP = 10
KEYS = ("A", "P", "Cb", "Cx")


@app.function(image=image, volumes={"/data": vol}, timeout=3600, cpu=4, memory=32768)
def process_session(sid: int, force: bool = False):
    import os
    import time

    import h5py
    import numpy as np
    from deficit_split import FREE, decompose, largest_term

    t0 = time.time()
    os.makedirs("/data/results_deficit", exist_ok=True)
    out_path = f"/data/results_deficit/session_{sid}.json"
    if os.path.exists(out_path) and not force:
        return json.load(open(out_path))

    nwb_path = f"/data/nwb/session_{sid}.nwb"
    if not os.path.exists(nwb_path):
        import requests
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
    ch_to_loc = dict(zip(el["id"][:], [x.decode() if isinstance(x, bytes) else str(x) for x in el["location"][:]]))
    unit_loc = np.array([ch_to_loc.get(c, "") for c in peak_ch])

    dg_name = [k for k in f["intervals"].keys() if "drifting_gratings" in k and "contrast" not in k]
    if not dg_name:
        out = {"session": sid, "status": "no_drifting_gratings"}
        json.dump(out, open(out_path, "w")); vol.commit()
        return out
    dg = f["intervals"][dg_name[0]]
    start, stop = dg["start_time"][:], dg["stop_time"][:]
    ori_raw = dg["orientation"][:]
    ori = np.array([float(x) if not isinstance(x, bytes) else (np.nan if x in (b"null", b"") else float(x)) for x in ori_raw])
    keep = np.isfinite(ori)
    start, stop, ori = start[keep], stop[keep], ori[keep]
    dirs = np.sort(np.unique(ori))
    labels = np.searchsorted(dirs, ori)

    n_units = len(sti)
    bounds = np.concatenate([[0], sti])
    counts = np.zeros((len(ori), n_units), dtype=np.float32)
    for j in range(n_units):
        sp = np.sort(st[bounds[j]:bounds[j + 1]])
        counts[:, j] = np.searchsorted(sp, stop) - np.searchsorted(sp, start)

    def class_stats(X):
        """Per class: within-class PR, diagonal PR, the neuron-space PR of the
        leading within-class eigenvectors, the top-10 variance fraction."""
        rows = []
        for c in range(len(dirs)):
            R = X[labels == c]; R = R - R.mean(0); n = R.shape[0]
            s2 = (R ** 2).mean(0).astype(np.float64)
            pr_diag = float(s2.sum() ** 2 / max((s2 ** 2).sum(), 1e-12))
            G = (R @ R.T).astype(np.float64) / max(n - 1, 1)
            w, U = np.linalg.eigh(G); order = np.argsort(w)[::-1]; w, U = w[order], U[:, order]
            w = np.maximum(w, 0)
            pr_full = float(w.sum() ** 2 / max((w ** 2).sum(), 1e-12))
            top = min(TOP, n - 1)
            top_frac = float(w[:top].sum() / max(w.sum(), 1e-12))
            V = (R.T @ U[:, :top]) / np.sqrt(np.maximum((n - 1) * w[:top], 1e-12))
            V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
            pr_neuron = (1.0 / np.maximum((V ** 4).sum(0), 1e-12)).tolist()
            rows.append({"class": c, "n_trials": int(n), "pr_full": pr_full, "pr_diag": pr_diag, "pr_neuron_top": pr_neuron, "top10_var_frac": top_frac})
        return rows

    out = {"session": sid, "status": "ok", "directions": dirs.tolist(), "n_presentations": int(len(ori)), "areas": {}}
    for area in sorted(set(unit_loc)):
        if not area.startswith("VIS"):
            continue
        ucols = np.where(unit_loc == area)[0]
        if len(ucols) < MIN_UNITS:
            continue
        X = np.ascontiguousarray(counts[:, ucols])
        r = decompose(X, labels, list(range(len(dirs))), BIN_COUNTS, n_null=N_NULL, seed=42, n_shuffle=N_SHUFFLE)
        if r is None:
            continue
        cls = class_stats(X)
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-9)
        clz = class_stats(Xz)
        N = int(len(ucols))
        summ = {"N": N, "pr_full_mean": float(np.mean([c["pr_full"] for c in cls])), "pr_diag_mean": float(np.mean([c["pr_diag"] for c in cls])),
                "pr_neuron_top1_mean": float(np.mean([c["pr_neuron_top"][0] for c in cls])),
                "pr_neuron_top1_over_N": float(np.mean([c["pr_neuron_top"][0] for c in cls]) / N),
                "top10_var_frac_mean": float(np.mean([c["top10_var_frac"] for c in cls])),
                "z_pr_full_mean": float(np.mean([c["pr_full"] for c in clz])), "z_pr_diag_mean": float(np.mean([c["pr_diag"] for c in clz])),
                "z_pr_neuron_top1_mean": float(np.mean([c["pr_neuron_top"][0] for c in clz])),
                "z_pr_neuron_top1_over_N": float(np.mean([c["pr_neuron_top"][0] for c in clz]) / N),
                "z_top10_var_frac_mean": float(np.mean([c["top10_var_frac"] for c in clz]))}
        ex = r["excess"]["delta_split"]
        out["areas"][area] = {"n_units": N, "delta": r["delta"], "delta_shuffle": r["shuffle"]["delta"], "delta_excess": r["excess"]["delta"],
                              "delta_split": r["delta_split"], "delta_split_shuffle": r["shuffle"]["delta_split"], "delta_split_excess": ex,
                              "largest_excess_term": largest_term(ex), "largest_free_term": largest_term(ex, FREE),
                              "delta_split_free": r["delta_split_free"], "delta_scale": r["delta_scale"],
                              "private_slope": r["private_slope"], "private_slope_shuffle": r["shuffle"]["private_slope"],
                              "private_slope_excess": r["excess"]["private_slope"], "private_slope_dim": r["private_slope_dim"], "private_slope_dim_excess": r["excess"]["private_slope_dim"],
                              "at_four": r["at_four"], "at_four_excess": r["excess"]["at_four"],
                              "late_share": r["late_share"], "rungs": r["rungs"], "rungs_shuffle": r["shuffle"]["rungs"],
                              "max_sum_check": r["max_sum_check"], "max_pr_identity_rel": r["max_pr_identity_rel"],
                              "within_class": summ, "within_class_per_class": cls, "within_class_per_class_zscored": clz}
        print(f"[{sid}] {area:6s} N {N:3d}: delta {r['delta']:+.3f} (shuffle {r['shuffle']['delta']:+.3f}) | excess A {ex['A']:+.3f} P {ex['P']:+.3f} Cb {ex['Cb']:+.3f} Cx {ex['Cx']:+.3f} "
              f"-> {largest_term(ex)} | free D {ex['D']:+.3f} Tb {ex['Tb']:+.3f} S {ex['S']:+.3f} -> {largest_term(ex, FREE)} | dim slope {r['excess']['private_slope_dim']:+.2f} | "
              f"wc PR {summ['pr_full_mean']:.1f} top-mode neuron PR {summ['pr_neuron_top1_mean']:.1f} ({summ['pr_neuron_top1_over_N']:.2f} N), z-scored {summ['z_pr_neuron_top1_mean']:.1f} ({summ['z_pr_neuron_top1_over_N']:.2f} N) | {time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(out_path, "w"))
    vol.commit()
    print(f"[{sid}] done {len(out['areas'])} areas ({time.time()-t0:.0f}s)", flush=True)
    return out


@app.local_entrypoint()
def main(sids: str = "", force: bool = False):
    import time

    import numpy as np
    from scipy.stats import spearmanr

    t0 = time.time()
    here = HERE
    data = here.parent / "data_canonical"
    exp = json.load(open(data / "allen_expansion_all_sessions.json"))["results"]
    stored = {(s["session"], a): v["delta_dir8"] for s in exp for a, v in s.get("areas", {}).items()}
    all_sids = [s["session"] for s in exp]
    todo = [int(x) for x in sids.split(",") if x] if sids else all_sids
    print(f"{len(todo)} sessions", flush=True)
    results = list(process_session.map(todo, kwargs={"force": force}))
    out = {"design": {"bin_counts": BIN_COUNTS, "n_null": N_NULL, "n_shuffle": N_SHUFFLE, "min_units": MIN_UNITS, "sessions": todo}, "results": results}
    pops = [(r["session"], a, v) for r in results if r.get("status") == "ok" for a, v in r["areas"].items()]
    n = len(pops)
    a1a = sum(v["largest_excess_term"] == "P" and v["delta_split_excess"]["P"] > 0 for _, _, v in pops)
    a1b = sum(v["at_four"]["P"] < 0 for _, _, v in pops)
    a2 = sum(v["within_class"]["pr_neuron_top1_over_N"] >= 0.1 for _, _, v in pops)
    dP = [v["delta_split_excess"]["P"] for _, _, v in pops]; dCb = [v["delta_split_excess"]["Cb"] for _, _, v in pops]
    wc = [v["within_class"]["pr_full_mean"] for _, _, v in pops]
    rho_P = float(spearmanr(dP, wc)[0]) if n > 3 else None; rho_Cb = float(spearmanr(dCb, wc)[0]) if n > 3 else None
    diff = [abs(v["delta"] - stored[(s, a)]) for s, a, v in pops if (s, a) in stored]
    dD = [v["delta_split_free"]["D"] - v["delta_split_shuffle"]["D"] for _, _, v in pops]; dTb = [v["delta_split_free"]["Tb"] - v["delta_split_shuffle"]["Tb"] for _, _, v in pops]
    a2z = sum(v["within_class"]["z_pr_neuron_top1_over_N"] >= 0.1 for _, _, v in pops)
    refined = {"largest_free_term_counts": {k: int(sum(v["largest_free_term"] == k for _, _, v in pops)) for k in ("D", "Tb", "Cb", "Cx")},
               "mean_free_excess": {k: float(np.mean([v["delta_split_excess"][k] for _, _, v in pops])) for k in ("D", "Tb", "Cb", "Cx", "S")},
               "rho_deltaD_wcPR": float(spearmanr(dD, wc)[0]) if n > 3 else None, "rho_deltaTb_wcPR": float(spearmanr(dTb, wc)[0]) if n > 3 else None,
               "count_top_mode_ge_N_over_10_zscored": int(a2z), "median_top_mode_over_N": float(np.median([v["within_class"]["pr_neuron_top1_over_N"] for _, _, v in pops])),
               "median_top_mode_over_N_zscored": float(np.median([v["within_class"]["z_pr_neuron_top1_over_N"] for _, _, v in pops])),
               "count_dimslope_excess_positive": int(sum(v["private_slope_dim_excess"] > 0 for _, _, v in pops)),
               "median_dimslope_excess": float(np.median([v["private_slope_dim_excess"] for _, _, v in pops]))}
    verdict = {"n_populations": n, "A1_count_P_largest_positive": int(a1a), "A1_count_P4_negative": int(a1b),
               "A1": bool(n > 0 and a1a >= 2 * n / 3 and a1b >= 2 * n / 3),
               "A2_count_top_mode_ge_N_over_10": int(a2), "A2": bool(n > 0 and a2 >= 2 * n / 3),
               "A3_rho_deltaP_wcPR": rho_P, "A3_rho_deltaCb_wcPR": rho_Cb, "A3": bool(rho_P is not None and rho_P <= -0.3 and abs(rho_P) > abs(rho_Cb)),
               "largest_term_counts": {k: int(sum(v["largest_excess_term"] == k for _, _, v in pops)) for k in KEYS}, "refined_descriptive": refined,
               "delta_recomputed_vs_stored_median_absdiff": float(np.median(diff)) if diff else None,
               "delta_recomputed_vs_stored_max_absdiff": float(np.max(diff)) if diff else None,
               "max_sum_check": max((v["max_sum_check"] for _, _, v in pops), default=None),
               "max_pr_identity_rel": max((v["max_pr_identity_rel"] for _, _, v in pops), default=None)}
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    outp = data / "run66_allen_deficit_decomposition.json"
    if sids:
        outp = data / f"run66_allen_deficit_decomposition_{sids.replace(',', '_')}.json"
    json.dump(out, open(outp, "w"), indent=1)
    print(f"wrote {outp} in {time.time()-t0:.0f}s", flush=True)
