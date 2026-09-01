"""Modal job: bootstrap all 10 Stringer orientation-zoom recordings at FULL neuron count.
Parallelized across recordings. Downloads data directly inside the container.

Run: modal run scripts/modal_bootstrap_all10.py
"""
import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install("numpy", "scipy")
app = modal.App("bootstrap-all-10")
vol = modal.Volume.from_name("stringer-v1-data", create_if_missing=True)

@app.function(image=image, volumes={"/data": vol}, timeout=3600, cpu=4, memory=8192)
def download_stringer():
    """Download Stringer data to the volume (run once)."""
    import urllib.request, os
    base = "https://osf.io/download/"
    # Stringer 2019 figshare IDs for the natimg recordings
    # These are the direct download links from the Stringer lab's data release
    out_dir = "/data/natimg"
    os.makedirs(out_dir, exist_ok=True)

    # Check if already downloaded
    existing = os.listdir(out_dir)
    if len(existing) >= 10:
        print(f"Already have {len(existing)} files")
        return True

    # The Stringer data is on figshare, but the direct URLs need the figshare API
    # Alternative: upload from local
    print("Data not on volume. Upload from local machine first:")
    print("  modal volume put stringer-v1-data data/stringer_v1/natimg/ /natimg/")
    return False

@app.function(image=image, volumes={"/data": vol}, timeout=7200, cpu=8, memory=16384)
def bootstrap_recording(filepath: str, n_boot: int = 200):
    """Bootstrap one recording at full neuron count."""
    import numpy as np
    import time

    t0 = time.time()
    dat = np.load(filepath, allow_pickle=True).item()
    X = dat['sresp'].T  # stimuli x neurons
    istim = dat['istim']
    n_stim, n_neur = X.shape

    n_bins = 8
    bl = np.clip(np.digitize(istim, np.linspace(0, 2*np.pi, n_bins + 1)) - 1, 0, n_bins - 1)

    def pr_fast(M):
        Mc = M - M.mean(axis=0)
        n = Mc.shape[0]
        if n < 3:
            return 1.0
        G = Mc @ Mc.T / (n - 1)
        lam = np.linalg.eigvalsh(G)
        lam = lam[lam > 1e-10]
        return float((lam.sum())**2 / (lam**2).sum()) if len(lam) else 1.0

    deltas = []
    for bi in range(n_boot):
        rng = np.random.default_rng(7000 + bi)
        # Resample stimuli within bins
        Xb = np.zeros_like(X)
        for b in range(n_bins):
            idx = np.where(bl == b)[0]
            Xb[bl == b] = X[rng.choice(idx, size=len(idx), replace=True)]

        # Structured ladder
        su, ru = [], []
        for no in [1, 2, 3, 4, 6, 8]:
            m = bl < no
            ns = int(m.sum())
            if ns < 10:
                continue
            su.append((ns, pr_fast(Xb[m])))
            ru.append((ns, pr_fast(Xb[rng.choice(n_stim, ns, replace=False)])))

        if len(su) >= 3:
            def fit(data):
                x = np.log([d[0] for d in data])
                y = np.log([max(d[1], 1e-9) for d in data])
                A = np.vstack([np.ones(len(x)), x]).T
                return float(np.linalg.lstsq(A, y, rcond=None)[0][1])
            deltas.append(fit(su) - fit(ru))

    d = np.array(deltas)
    elapsed = time.time() - t0
    name = filepath.split("/")[-1].replace(".npy", "")

    return {
        "name": name,
        "n_neurons": n_neur,
        "n_stim": n_stim,
        "n_boot": n_boot,
        "mean": round(float(d.mean()), 4),
        "ci_lo": round(float(np.percentile(d, 2.5)), 4),
        "ci_hi": round(float(np.percentile(d, 97.5)), 4),
        "excludes_zero": bool(np.percentile(d, 2.5) > 0),
        "elapsed_s": round(elapsed, 1),
    }

@app.local_entrypoint()
def main():
    import json, os, glob

    # Check if data is on volume
    # If not, upload it first
    data_dir = "data/stringer_v1/natimg"
    files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

    if not files:
        print("No local Stringer data found!")
        return

    print(f"Found {len(files)} recordings locally")
    print("Launching {len(files)} parallel bootstrap jobs on Modal...")

    # Launch all in parallel using Modal's .map()
    results = list(bootstrap_recording.map(files, kwargs={"n_boot": 200}))

    # Sort and display
    results.sort(key=lambda r: r["name"])
    print(f"\n{'Recording':35s} {'Neurons':>7s} {'δ mean':>8s} {'CI':>20s} {'Excl 0?':>8s} {'Time':>6s}")
    print("-" * 90)
    for r in results:
        ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
        print(f"{r['name'][:35]:35s} {r['n_neurons']:7d} {r['mean']:+8.4f} {ci:>20s} {'YES' if r['excludes_zero'] else 'NO':>8s} {r['elapsed_s']:5.0f}s")

    n_pos = sum(1 for r in results if r["mean"] > 0)
    n_excl = sum(1 for r in results if r["excludes_zero"])
    print(f"\nPositive: {n_pos}/{len(results)}")
    print(f"CI excludes zero: {n_excl}/{len(results)}")
    print(f"Sign test p = {2**(-n_pos) * 2:.6f}")

    # Save
    outpath = "data/bootstrap_all_10_orient_fullneuron.json"
    with open(outpath, "w") as f:
        json.dump({"recordings": results, "n_positive": n_pos,
                    "n_ci_excludes": n_excl, "method": "full neuron count, 200 bootstrap replicates"}, f, indent=1)
    print(f"\nSaved: {outpath}")
