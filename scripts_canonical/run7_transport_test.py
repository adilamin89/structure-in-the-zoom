"""Run 7 - Is "co-rotation" earned? Test for a common group transport of
within-class subspaces (the referee's Sigma_{phi+alpha} ~ U Sigma_phi U^T).

Setup: within-class subspaces S_0..S_7 (rank 10, 120 trials/class, as run3),
projected into the union space (rank <= 80). Two tests:

T1 STATIONARITY (necessary condition): the alignment A(i, i+k) must depend on
   the separation k only, not on the base class i. Metric: for each k, the SD
   of A(i,i+k) across i, relative to the spread of the means across k.
T2 TRANSPORT TRANSFER (the group-action test): estimate the direct rotation
   R_i taking span(S_i) onto span(S_{i+1}) (closed form from principal pairs,
   exact by the CS decomposition), then apply R_i to a DIFFERENT class j.
   If a single generator underlies the structure, R_i S_j should align with
   S_{j+1} better than untransported S_j does. Null: reversed transport
   (R_i^T, which rotates away from the neighbor) applied to the same pairs.

REGISTERED EXPECTATIONS (before run):
E1: within-separation dispersion ratio < 0.5 (profile is a function of k).
E2: cross-pair transfer gain positive in the majority of (i, j) pairs,
    i != j, in all three recordings; reversed-transport null shows loss or
    no gain. Success on E1+E2 = "co-rotation" operationally demonstrated:
    a common one-step rotation moves every class's variability subspace
    toward its neighbor's.

Out: feedback_runs/run7_transport_test.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
N_BINS = 8
RANK = 10
N_PER = 120

FILES = {
    "GT1": "gratings_drifting_GT1_2019_04_12_1.npy",
    "GT2": "gratings_drifting_GT2_2019_04_05_1.npy",
    "GT3": "gratings_drifting_GT3_2019_04_05_1.npy",
}


def subspaces(Xt, bl, rng):
    subs = []
    for b in range(N_BINS):
        idx = rng.choice(np.where(bl == b)[0], N_PER, replace=False)
        V = Xt[idx]
        Vc = V - V.mean(axis=0)
        _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
        subs.append(Vt[:RANK].T)
    # union space
    Q, _ = np.linalg.qr(np.concatenate(subs, axis=1))
    return [Q.T @ s for s in subs]         # each 80 x 10 (orthonormal cols)


def align(A, B):
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return float(np.mean(np.clip(s, 0, 1) ** 2))


def direct_rotation(A, B, dim):
    """Closed-form rotation taking span(A) onto span(B): rotate each
    principal 2-plane by its principal angle (exact via CS decomposition)."""
    U, s, Vt = np.linalg.svd(A.T @ B)
    PA, PB = A @ U, B @ Vt.T               # paired principal vectors
    R = np.eye(dim)
    for k in range(PA.shape[1]):
        a, b = PA[:, k], PB[:, k]
        c = float(np.clip(a @ b, -1, 1))
        w = b - c * a
        nw = np.linalg.norm(w)
        if nw < 1e-9:
            continue
        w = w / nw
        th = np.arccos(c)
        Rk = np.eye(dim) \
            + (np.cos(th) - 1) * (np.outer(a, a) + np.outer(w, w)) \
            + np.sin(th) * (np.outer(w, a) - np.outer(a, w))
        R = Rk @ R
    return R


def run_recording(tag, fname):
    dat = np.load(DATA / fname, allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
                 0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)
    subs = subspaces(Xt, bl, np.random.default_rng(0))
    dim = subs[0].shape[0]

    # T1: stationarity of the alignment profile
    prof = {}
    for k in range(1, 5):
        vals = [align(subs[i], subs[(i + k) % N_BINS]) for i in range(N_BINS)]
        prof[k] = (float(np.mean(vals)), float(np.std(vals)))
    within_sd = np.mean([prof[k][1] for k in prof])
    between_sd = np.std([prof[k][0] for k in prof])
    ratio = float(within_sd / max(between_sd, 1e-9))

    # T2: cross-pair transport transfer
    Rs = [direct_rotation(subs[i], subs[(i + 1) % N_BINS], dim)
          for i in range(N_BINS)]
    gains, null_gains = [], []
    for i in range(N_BINS):
        for j in range(N_BINS):
            if i == j:
                continue
            base = align(subs[j], subs[(j + 1) % N_BINS])
            gains.append(align(Rs[i] @ subs[j], subs[(j + 1) % N_BINS]) - base)
            null_gains.append(align(Rs[i].T @ subs[j],
                                    subs[(j + 1) % N_BINS]) - base)
    gains, null_gains = np.array(gains), np.array(null_gains)
    res = {
        "profile_mean_sd_by_k": {str(k): prof[k] for k in prof},
        "stationarity_ratio": ratio,
        "transfer_gain_mean": float(gains.mean()),
        "transfer_gain_frac_positive": float((gains > 0).mean()),
        "reversed_null_gain_mean": float(null_gains.mean()),
        "reversed_null_frac_positive": float((null_gains > 0).mean()),
        "baseline_onestep_alignment": prof[1][0],
    }
    print(f"{tag}: stationarity ratio={ratio:.2f} | "
          f"transfer gain={gains.mean():+.3f} "
          f"({(gains>0).mean()*100:.0f}% pos) | "
          f"reversed null={null_gains.mean():+.3f} "
          f"({(null_gains>0).mean()*100:.0f}% pos)", flush=True)
    return res


out = {}
for tag, fname in FILES.items():
    out[tag] = run_recording(tag, fname)
json.dump(out, open(HERE / "run7_transport_test.json", "w"), indent=1)
print("DONE run7", flush=True)
