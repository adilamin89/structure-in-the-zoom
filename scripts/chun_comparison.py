"""Compare matched-floor delta against Chun et al. 2026's bias-corrected PR.

Chun et al. correct the finite-sample bias of PR analytically using a
second-order correction. Their corrected estimator is:
  PR_corrected = PR_raw * (1 + 1/n) * (n-1)/(n-2)  (simplified; the full
  correction depends on the spectrum shape)

For the comparison we need: on the SAME GT3 data with the SAME direction
ladder, compute delta using (a) our empirical matched floor, (b) Chun's
analytic bias correction applied to each rung. If the two residuals agree,
our floor is doing the same job as their correction; if they disagree, the
difference is informative about what our floor captures beyond bias.

REGISTERED EXPECTATION: the two should be close for random-axis ladders
(both correct the same bias) but may differ for aligned ladders because our
floor also captures the population's PR at each effective subset size, not
just the n-dependent bias.

Output: data/chun_comparison.json
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "stringer_v1" / "natimg"
OUT = HERE / "data" / "chun_comparison.json"

N_BINS = 8
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_NULL = 10


def pr_raw(X):
    """Raw centered PR."""
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    if n < 3:
        return 1.0, n
    G = (Xc @ Xc.T).astype(np.float64) / (n - 1)
    tr = float(np.trace(G))
    tr2 = float((G * G).sum())
    if not np.isfinite(tr2) or tr2 <= 0:
        return 1.0, n
    return tr * tr / tr2, n


def chun_correct(pr, n, d):
    """Chun et al. 2026 bias correction (their Eq. 3, simplified form).
    The correction accounts for the upward bias of PR from finite samples.
    For isotropic spectra: E[PR] ≈ PR_true * (n-1)/(n-1+PR_true),
    so PR_true ≈ PR_raw * (n-1) / (n-1 - PR_raw) when PR_raw < n-1.
    We use their iterative debiasing: invert the MP crossover."""
    if pr >= n - 1:
        return pr  # can't correct; already at the ceiling
    # Inversion of PR_obs = d*(n-1)/(d + n-1) gives d = PR_obs*(n-1)/(n-1-PR_obs)
    d_est = pr * (n - 1) / max(n - 1 - pr, 0.1)
    return d_est


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, float))
    y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def main():
    fp = DATA / "gratings_drifting_GT3_2019_04_05_1.npy"
    dat = np.load(fp, allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32)
    X /= X.std() + 1e-9
    Xt = np.ascontiguousarray(X.T)
    istim = np.asarray(dat["istim"], float)
    n_neur = X.shape[0]
    bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
    members = [np.where(bl == b)[0] for b in range(N_BINS)]
    rng = np.random.default_rng(42)

    # Direction-aligned ladder
    dir_sizes, dir_pr_raw, dir_pr_chun = [], [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        pr, n = pr_raw(Xt[sel])
        dir_sizes.append(len(sel))
        dir_pr_raw.append(pr)
        dir_pr_chun.append(chun_correct(pr, n, n_neur))

    # Empirical floor
    null_logs = np.zeros((N_NULL, len(dir_sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(dir_sizes):
            pr, _ = pr_raw(Xt[rng.choice(len(Xt), s, replace=False)])
            null_logs[d, k] = np.log(max(pr, 1e-9))
    th_floor = slope(dir_sizes, np.exp(null_logs.mean(axis=0)))

    th_raw = slope(dir_sizes, dir_pr_raw)
    th_chun = slope(dir_sizes, dir_pr_chun)

    delta_ours = th_raw - th_floor
    delta_chun = th_chun - slope(dir_sizes,
                                 [chun_correct(np.exp(null_logs[:, k].mean()), dir_sizes[k], n_neur)
                                  for k in range(len(dir_sizes))])

    # Random ladder for comparison
    rand_sizes = [50, 100, 200, 500, 1000, 2000, len(Xt)]
    rand_pr_raw, rand_pr_chun = [], []
    for s in rand_sizes:
        prs = [pr_raw(Xt[rng.choice(len(Xt), s, replace=False)]) for _ in range(5)]
        mean_pr = np.mean([p[0] for p in prs])
        rand_pr_raw.append(mean_pr)
        rand_pr_chun.append(chun_correct(mean_pr, s, n_neur))

    rand_null_logs = np.zeros((N_NULL, len(rand_sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(rand_sizes):
            pr, _ = pr_raw(Xt[rng.choice(len(Xt), s, replace=False)])
            rand_null_logs[d, k] = np.log(max(pr, 1e-9))
    rand_floor = slope(rand_sizes, np.exp(rand_null_logs.mean(axis=0)))

    rand_delta_ours = slope(rand_sizes, rand_pr_raw) - rand_floor
    rand_delta_chun = slope(rand_sizes, rand_pr_chun) - slope(
        rand_sizes, [chun_correct(np.exp(rand_null_logs[:, k].mean()), rand_sizes[k], n_neur)
                     for k in range(len(rand_sizes))])

    out = {
        "direction_aligned": {
            "theta_raw": th_raw, "theta_floor_empirical": th_floor,
            "theta_chun_corrected": th_chun,
            "delta_matched_floor": delta_ours,
            "delta_chun_corrected": delta_chun,
            "agreement_pct": abs(delta_ours - delta_chun) / abs(delta_ours) * 100
                             if abs(delta_ours) > 0.001 else None,
        },
        "random": {
            "delta_matched_floor": rand_delta_ours,
            "delta_chun_corrected": rand_delta_chun,
        },
    }
    print(f"DIRECTION-ALIGNED:")
    print(f"  delta (matched floor): {delta_ours:+.4f}")
    print(f"  delta (Chun corrected): {delta_chun:+.4f}")
    print(f"  agreement: {out['direction_aligned']['agreement_pct']:.1f}%")
    print(f"RANDOM:")
    print(f"  delta (matched floor): {rand_delta_ours:+.4f}")
    print(f"  delta (Chun corrected): {rand_delta_chun:+.4f}")

    with OUT.open("w") as f:
        json.dump(out, f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
