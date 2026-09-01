"""Run 3 - Model-free co-rotation test: principal angles between within-class
subspaces as a function of angular separation, on real GT3 data.

If delta > 0 measures co-rotation of within-class subspaces with the class
label (the run-2 mechanism), the alignment between the within-class subspaces
of two direction classes should DECREASE with their angular separation.

REGISTERED EXPECTATIONS (before run):
A1: mean subspace alignment (mean cos^2 principal angles, top-r within-class
    PCs) decreases with Delta-phi on the half circle; adjacent (45 deg) more
    aligned than antipodal (180 deg).
A2: all class-pair alignments sit far above the random-subspace chance level
    (r/n_eff) - shared gain structure exists - but the RANKING follows
    Delta-phi.
A3: replicate on all three drifting recordings (GT1-3); ordering consistent.

Out: feedback_runs/run3_principal_angles.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
N_BINS = 8
RANKS = [10, 20]

FILES = {
    "GT1": "gratings_drifting_GT1_2019_04_12_1.npy",
    "GT2": "gratings_drifting_GT2_2019_04_05_1.npy",
    "GT3": "gratings_drifting_GT3_2019_04_05_1.npy",
}


def class_subspace(V, r):
    """Top-r right singular vectors of the centered within-class matrix."""
    Vc = V - V.mean(axis=0)
    # neurons >> trials: SVD on trials x neurons is cheap via gram trick
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    return Vt[:r].T  # n_neur x r, orthonormal columns


def alignment(Q1, Q2):
    """Mean squared cosine of principal angles = mean(svd(Q1^T Q2)^2)."""
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(s ** 2))


def run_recording(tag, fname):
    dat = np.load(DATA / fname, allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    istim = np.asarray(dat["istim"], float)
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1,
                 0, N_BINS - 1)
    Xt = np.ascontiguousarray(X.T)  # trials x neurons
    rng = np.random.default_rng(0)
    res = {}
    for r in RANKS:
        subs = []
        n_per = min(int(np.bincount(bl, minlength=N_BINS).min()), 120)
        for b in range(N_BINS):
            idx = rng.choice(np.where(bl == b)[0], n_per, replace=False)
            subs.append(class_subspace(Xt[idx], r))
        # alignment vs angular separation (direction circle, 45-deg steps)
        by_sep = {}
        for i in range(N_BINS):
            for j in range(i + 1, N_BINS):
                sep = min((j - i) % N_BINS, (i - j) % N_BINS) * 45
                by_sep.setdefault(sep, []).append(alignment(subs[i], subs[j]))
        # chance level: random orthonormal r-frames in the trial-limited space
        n_eff = n_per  # centered within-class matrix has rank <= n_per - 1
        chance = []
        for _ in range(20):
            Q1 = np.linalg.qr(rng.standard_normal((Xt.shape[1], r)))[0]
            Q2 = np.linalg.qr(rng.standard_normal((Xt.shape[1], r)))[0]
            chance.append(alignment(Q1, Q2))
        prof = {str(sep): float(np.mean(v)) for sep, v in sorted(by_sep.items())}
        res[f"r{r}"] = {"alignment_vs_sep_deg": prof,
                        "chance": float(np.mean(chance)),
                        "n_per_class": n_per}
        seps = sorted(by_sep)
        vals = [np.mean(by_sep[s]) for s in seps]
        mono = all(vals[k] >= vals[k + 1] for k in range(len(vals) - 1))
        adj, antip = prof["45"], prof["180"]
        print(f"{tag} r={r}: " +
              " ".join(f"{s}deg={np.mean(by_sep[s]):.3f}" for s in seps) +
              f" | chance={np.mean(chance):.4f} | monotone={mono} "
              f"| adjacent>{'' if adj > antip else '!'}antipodal", flush=True)
        res[f"r{r}"]["monotone"] = bool(mono)
        res[f"r{r}"]["adjacent_gt_antipodal"] = bool(adj > antip)
    return res


out = {}
for tag, fname in FILES.items():
    out[tag] = run_recording(tag, fname)
json.dump(out, open(HERE / "run3_principal_angles.json", "w"), indent=1)
print("DONE run3", flush=True)
