"""Run 35 — Split-half reliability of the within-class subspace alignment
profile (the last open item from the round-1 review; round-3 M-adjacent).

For each drifting recording, trials within each of 16 direction classes are
split into disjoint halves. Top-10 within-class subspaces are estimated per
half. Three registered quantities:

R1 (self-reliability): cross-half alignment of the SAME class's subspaces
    should far exceed cross-class alignment at 90 degrees (the subspaces are
    estimable at n=60/half), giving a noise ceiling for the profile.
R2 (profile reliability): the angular alignment profile computed within
    half A should correlate with the profile within half B across separation
    bins (r > 0.7 expected).
R3 (the headline shape): the pi-periodic signature (22.5 max, 90 min, 180
    recovery) should appear in BOTH halves independently.

Out: ../data_canonical/run35_splithalf_subspaces.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = HERE.parent / "data_canonical" / "run35_splithalf_subspaces.json"

N_BINS = 16
RANK = 10
N_PER = 120  # per class total; 60 per half

FILES = {
    "GT1": "gratings_drifting_GT1_2019_04_12_1.npy",
    "GT2": "gratings_drifting_GT2_2019_04_05_1.npy",
    "GT3": "gratings_drifting_GT3_2019_04_05_1.npy",
}


def align(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def subspace(Xt, idx):
    V = Xt[idx]
    Vc = V - V.mean(axis=0)
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    return Vt[:RANK].T


def profile(subs):
    by_sep = {}
    for i in range(N_BINS):
        for j in range(i + 1, N_BINS):
            sep = min((j - i) % N_BINS, (i - j) % N_BINS) * 22.5
            by_sep.setdefault(sep, []).append(align(subs[i], subs[j]))
    return {s: float(np.mean(v)) for s, v in sorted(by_sep.items())}


out = {}
for tag, fname in FILES.items():
    dat = np.load(DATA / fname, allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
                 0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)
    rng = np.random.default_rng(35)
    n_min = int(np.bincount(bl, minlength=N_BINS).min())
    n_per = min(n_min, N_PER)
    half = n_per // 2

    subs_A, subs_B = [], []
    for b in range(N_BINS):
        idx = rng.choice(np.where(bl == b)[0], n_per, replace=False)
        subs_A.append(subspace(Xt, idx[:half]))
        subs_B.append(subspace(Xt, idx[half:]))

    # R1: same-class cross-half alignment vs 90-degree cross-class
    self_rel = [align(subs_A[b], subs_B[b]) for b in range(N_BINS)]
    cross90 = [align(subs_A[i], subs_B[(i + 4) % N_BINS])
               for i in range(N_BINS)]

    # R2/R3: independent within-half profiles
    prof_A = profile(subs_A)
    prof_B = profile(subs_B)
    seps = sorted(prof_A)
    a = np.array([prof_A[s] for s in seps])
    b_ = np.array([prof_B[s] for s in seps])
    r = float(np.corrcoef(a, b_)[0, 1])

    def shape_ok(p):
        return bool(p[22.5] > p[45.0] and p[180.0] > p[90.0])

    out[tag] = {
        "n_per_half": half,
        "self_reliability_mean": float(np.mean(self_rel)),
        "cross_class_90_mean": float(np.mean(cross90)),
        "profile_r_between_halves": r,
        "profile_A": {str(k): v for k, v in prof_A.items()},
        "profile_B": {str(k): v for k, v in prof_B.items()},
        "pi_shape_half_A": shape_ok(prof_A),
        "pi_shape_half_B": shape_ok(prof_B),
    }
    print(f"{tag}: self {np.mean(self_rel):.3f} vs cross-90 "
          f"{np.mean(cross90):.3f} | profile r(A,B) {r:+.3f} | "
          f"pi-shape A {shape_ok(prof_A)} B {shape_ok(prof_B)}", flush=True)

json.dump(out, open(OUT, "w"), indent=1)
print("DONE run35", flush=True)
