"""Batch theta decomposition across ALL Stringer V1 recordings.

Runs the optimized decomposition (Gram matrix trick) on all 18 recordings:
  - 10 natural image recordings (natimg/*.npy)
  - 8 spontaneous activity recordings (spont/*.mat)

Kill-safe: writes each row to a JSONL as it completes.
Final summary to data/stringer_theta_decomposition.json.
"""
import json
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.io import loadmat

DATA_DIR = Path(__file__).resolve().parent / "data" / "stringer_v1"
NATIMG_DIR = DATA_DIR / "natimg"
SPONT_DIR = DATA_DIR / "spont"
OUT_PATH = Path(__file__).resolve().parent / "data" / "stringer_theta_decomposition.json"
ROWS_PATH = Path(__file__).resolve().parent / "data" / "stringer_theta_decomposition_rows.jsonl"

N_NULL = 10
ZOOM_SIZES = [2000, 1000, 500, 200, 100, 50]


def pr_fast(X):
    """PR via Gram matrix (n×n) instead of covariance (d×d)."""
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    if n < 3:
        return 1.0
    G = Xc @ Xc.T / (n - 1)
    lam = np.linalg.eigvalsh(G)
    lam = lam[lam > 1e-10]
    if len(lam) == 0:
        return 1.0
    return float((lam.sum()) ** 2 / (lam ** 2).sum())


def pr_null_mp(n, d):
    """Marchenko-Pastur null PR."""
    return (n - 1) * d / ((n - 1) + d)


def decompose(X, name):
    """Full decomposition on one recording."""
    t0 = time.time()
    # Ensure stimuli × neurons
    if X.shape[0] > X.shape[1]:
        X = X.T

    n_stim, n_neur = X.shape
    print(f"\n  {name}: {n_neur} neurons × {n_stim} stimuli")

    # Zoom ladder
    pr_full = pr_fast(X)
    units = [("full", n_stim, pr_full)]
    print(f"    Full PR = {pr_full:.1f}")

    for n_sub in ZOOM_SIZES:
        if n_sub >= n_stim:
            continue
        prs = []
        for seed in range(5):
            rng = np.random.default_rng(42 + seed)
            idx = rng.choice(n_stim, size=n_sub, replace=False)
            prs.append(pr_fast(X[idx]))
        units.append(("sub", n_sub, float(np.mean(prs))))

    if len(units) < 3:
        return {"name": name, "status": "too_few_units"}

    # Theta obs
    x = np.log(np.array([u[1] for u in units], dtype=float))
    y = np.log(np.array([max(u[2], 1e-9) for u in units], dtype=float))
    A = np.vstack([np.ones(len(x)), x]).T
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    theta_obs = float(beta[1])

    # Empirical floor
    null_thetas = []
    for ni in range(N_NULL):
        rng = np.random.default_rng(1000 + ni)
        null_units = []
        for uname, n_sub, _ in units:
            idx = rng.choice(n_stim, size=n_sub, replace=False)
            null_units.append((uname, n_sub, pr_fast(X[idx])))
        xn = np.log(np.array([u[1] for u in null_units], dtype=float))
        yn = np.log(np.array([max(u[2], 1e-9) for u in null_units], dtype=float))
        An = np.vstack([np.ones(len(xn)), xn]).T
        bn, *_ = np.linalg.lstsq(An, yn, rcond=None)
        null_thetas.append(float(bn[1]))

    theta_null = float(np.mean(null_thetas))
    theta_null_sd = float(np.std(null_thetas))

    # Analytic floor
    d_eff = pr_full
    af_units = [(uname, n_sub, pr_null_mp(n_sub, d_eff))
                for uname, n_sub, _ in units]
    xfa = np.log(np.array([u[1] for u in af_units], dtype=float))
    yfa = np.log(np.array([max(u[2], 1e-9) for u in af_units], dtype=float))
    Afa = np.vstack([np.ones(len(xfa)), xfa]).T
    bfa, *_ = np.linalg.lstsq(Afa, yfa, rcond=None)
    theta_floor_analytic = float(bfa[1])

    delta = theta_obs - theta_null
    floor_frac = theta_null / theta_obs if abs(theta_obs) > 1e-6 else float("nan")

    elapsed = time.time() - t0
    print(f"    θ_obs={theta_obs:.4f}  floor={theta_null:.4f} "
          f"({floor_frac * 100:.1f}%)  δ={delta:.4f}  ({elapsed:.0f}s)")

    return {
        "name": name,
        "status": "ok",
        "n_neurons": n_neur,
        "n_stimuli": n_stim,
        "d_eff_pop": float(d_eff),
        "n_units": len(units),
        "theta_obs": theta_obs,
        "theta_floor_empirical": theta_null,
        "theta_floor_analytic": theta_floor_analytic,
        "theta_floor_sd": theta_null_sd,
        "shift_obs": delta,
        "floor_fraction": floor_frac,
        "elapsed_s": elapsed,
        "zoom_ladder": [(u[0], u[1], u[2]) for u in units],
    }


def load_natimg(path):
    dat = np.load(path, allow_pickle=True).item()
    return dat["sresp"].T  # stimuli × neurons


def load_spont(path):
    dat = loadmat(path)
    for key in ["Fsp", "spks", "F"]:
        if key in dat:
            X = dat[key]
            return X.T if X.shape[0] < X.shape[1] else X
    return None


def main():
    print("=" * 70)
    print("  STRINGER V1 THETA DECOMPOSITION - FULL BATCH")
    print("=" * 70)

    # Clear old rows
    if ROWS_PATH.exists():
        ROWS_PATH.unlink()

    rows = []

    # Natural images
    npy_files = sorted(NATIMG_DIR.glob("*.npy")) if NATIMG_DIR.exists() else []
    print(f"\nNatural image recordings: {len(npy_files)}")
    for fp in npy_files:
        try:
            X = load_natimg(fp)
            r = decompose(X, fp.stem)
            rows.append(r)
            with ROWS_PATH.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
        except Exception as e:
            print(f"  ERROR {fp.stem}: {e}")
            traceback.print_exc()
            rows.append({"name": fp.stem, "status": "error", "error": str(e)})

    # Spontaneous
    mat_files = sorted(SPONT_DIR.glob("*.mat")) if SPONT_DIR.exists() else []
    print(f"\nSpontaneous recordings: {len(mat_files)}")
    for fp in mat_files:
        if fp.stem == "dbspont":
            continue
        try:
            X = load_spont(fp)
            if X is None:
                print(f"  {fp.stem}: no usable array found")
                continue
            # Subsample time for computation
            if X.shape[0] > 5000:
                rng = np.random.default_rng(42)
                idx = rng.choice(X.shape[0], size=5000, replace=False)
                X = X[np.sort(idx)]
            r = decompose(X, f"spont_{fp.stem}")
            rows.append(r)
            with ROWS_PATH.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
        except Exception as e:
            print(f"  ERROR spont_{fp.stem}: {e}")
            traceback.print_exc()
            rows.append({"name": f"spont_{fp.stem}", "status": "error",
                         "error": str(e)})

    # Summary
    ok = [r for r in rows if r.get("status") == "ok"]
    if ok:
        ffs = [r["floor_fraction"] for r in ok
               if not np.isnan(r.get("floor_fraction", float("nan")))]
        summary = {
            "n_recordings": len(ok),
            "n_errors": len(rows) - len(ok),
            "mean_theta_obs": float(np.mean([r["theta_obs"] for r in ok])),
            "mean_floor_fraction": float(np.mean(ffs)),
            "median_floor_fraction": float(np.median(ffs)),
            "std_floor_fraction": float(np.std(ffs)),
            "mean_delta": float(np.mean([r["shift_obs"] for r in ok])),
            "rows": rows,
        }
        print(f"\n{'=' * 70}")
        print(f"  SUMMARY: {len(ok)} recordings")
        print(f"  Mean θ_obs: {summary['mean_theta_obs']:.4f}")
        print(f"  Mean floor fraction: {summary['mean_floor_fraction'] * 100:.1f}%")
        print(f"  Median floor fraction: {summary['median_floor_fraction'] * 100:.1f}%")
        print(f"  Std floor fraction: {summary['std_floor_fraction'] * 100:.1f}%")
        print(f"  Mean δ: {summary['mean_delta']:.4f}")
        print(f"{'=' * 70}")
    else:
        summary = {"n_recordings": 0, "rows": rows}

    with OUT_PATH.open("w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\n  Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
