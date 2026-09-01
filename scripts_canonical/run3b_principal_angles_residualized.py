"""Run 3b - Leakage control for the pi-periodic principal-angle result.

Run 3 found that alignment between within-class subspaces is pi-periodic
(45deg ~ 0.13, min at 90deg ~ 0.05, recovery at 180deg ~ 0.11-0.19). Concern:
within a 45-deg direction bin, istim still varies, so the "within-class"
subspace may contain fine-grained TUNING (signal), which is itself
pi-periodic, rather than noise structure.

Control: within each class, sub-bin istim into 6 sub-bins and subtract the
sub-bin mean response (removing fine tuning to ~7.5 deg resolution) before
computing the within-class subspace from the residuals.

REGISTERED EXPECTATIONS (before run):
L1: if the 180-deg recovery is fine-tuning leakage, the residualized profile
    loses the recovery (alignment at 180 falls to ~the 90-deg level).
L2: if it is genuine noise-subspace co-rotation, the recovery survives
    residualization (180 > 90 in all three drifting recordings).
Either outcome decides whether the result can enter the paper.

Out: feedback_runs/run3b_principal_angles_residualized.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
N_BINS = 8
N_SUB = 6
RANK = 10

FILES = {
    "GT1": "gratings_drifting_GT1_2019_04_12_1.npy",
    "GT2": "gratings_drifting_GT2_2019_04_05_1.npy",
    "GT3": "gratings_drifting_GT3_2019_04_05_1.npy",
}


def class_subspace(V, r):
    Vc = V - V.mean(axis=0)
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    return Vt[:r].T


def alignment(Q1, Q2):
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(s ** 2))


def run_recording(tag, fname):
    dat = np.load(DATA / fname, allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    edges = np.linspace(0, 2 * np.pi, N_BINS + 1)
    bl = np.clip(np.digitize(istim, edges) - 1, 0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)
    rng = np.random.default_rng(0)
    res = {}
    for mode in ("raw", "residualized"):
        subs = []
        n_per = min(int(np.bincount(bl, minlength=N_BINS).min()), 120)
        for b in range(N_BINS):
            idx = rng.choice(np.where(bl == b)[0], n_per, replace=False)
            V = Xt[idx].copy()
            if mode == "residualized":
                # subtract sub-bin means: removes tuning to ~7.5 deg
                sub_edges = np.linspace(edges[b], edges[b + 1], N_SUB + 1)
                sub = np.clip(np.digitize(istim[idx], sub_edges) - 1, 0, N_SUB - 1)
                for s in range(N_SUB):
                    m = sub == s
                    if m.sum() > 1:
                        V[m] -= V[m].mean(axis=0)
            subs.append(class_subspace(V, RANK))
        by_sep = {}
        for i in range(N_BINS):
            for j in range(i + 1, N_BINS):
                sep = min((j - i) % N_BINS, (i - j) % N_BINS) * 45
                by_sep.setdefault(sep, []).append(alignment(subs[i], subs[j]))
        prof = {str(s): float(np.mean(v)) for s, v in sorted(by_sep.items())}
        res[mode] = prof
        print(f"{tag} [{mode}]: " +
              " ".join(f"{s}deg={np.mean(by_sep[s]):.3f}" for s in sorted(by_sep)) +
              f" | 180>{'' if prof['180'] > prof['90'] else '!'}90", flush=True)
    return res


out = {}
for tag, fname in FILES.items():
    out[tag] = run_recording(tag, fname)
json.dump(out, open(HERE / "run3b_principal_angles_residualized.json", "w"),
          indent=1)
print("DONE run3b", flush=True)
