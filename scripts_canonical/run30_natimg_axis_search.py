"""Run 30 — Declared-axis search on the Stringer natural-image sessions
(natimg2800): the axes the static-grating sessions could not provide.

Each of 2800 natural images gets two stimulus-computable coordinates from
its 2D Fourier spectrum: spatial frequency (radial power centroid) and
dominant orientation (pi-periodic circular mean of angular energy). Trials
are binned into 8 classes along each axis and the standard ladder runs
(accumulate [1,2,3,4,6,8] classes, 10-draw floors, 5 label shuffles).

REGISTERED EXPECTATIONS (written before the run):
N1: delta_SF > 0 — mouse V1 is spatial-frequency tuned; an SF-aligned
    ladder should expose organized covariance.
N2: delta_domori > 0 but smaller than the grating-session delta_dir
    (+0.20..+0.46): each natural image carries broadband orientation
    energy, so the per-image dominant orientation is a weak label.
N3: random 8-class relabeling ~ 0.
Either miss is reportable; N2's magnitude ordering is the interesting cell.

Data: basin_memory/data/stringer_v1/natimg2800/ (3 sessions + images,
figshare janelia 6845348). Out: ../data_canonical/run30_natimg_axis_search.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
# Stringer natimg2800 release (figshare janelia 6845348).
DATA = Path(__file__).resolve().parent.parent / "raw" / "natimg2800"
OUT = HERE.parent / "data_canonical" / "run30_natimg_axis_search.json"

spec = importlib.util.spec_from_file_location(
    "r27", HERE / "run27_static_axis_search.py")
r27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r27)

SESSIONS = ["natimg2800_M170717_MP034_2017-09-11.mat",
            "natimg2800_M160825_MP027_2016-12-14.mat",
            "natimg2800_M170604_MP031_2017-06-28.mat"]


def image_axes():
    """Per-image spatial frequency and dominant orientation from the FFT."""
    imgs = sio.loadmat(DATA / "images_natimg2800_all.mat")["imgs"]
    h, w, n = imgs.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    rad = np.sqrt(fy ** 2 + fx ** 2)
    ang = np.arctan2(fy, fx)  # orientation of the frequency component
    mask = rad > 0
    sf = np.zeros(n)
    domori = np.zeros(n)
    for i in range(n):
        im = imgs[:, :, i].astype(np.float64)
        im -= im.mean()
        P = np.abs(np.fft.fft2(im)) ** 2
        p = P[mask]
        sf[i] = (p * rad[mask]).sum() / p.sum()
        # pi-periodic circular mean: double the angle
        z = (p * np.exp(2j * ang[mask])).sum() / p.sum()
        domori[i] = np.angle(z) / 2  # in (-pi/2, pi/2]
        if i and i % 500 == 0:
            print(f"    images {i}/{n}", flush=True)
    return sf, domori


def octile_labels(vals):
    edges = np.quantile(vals, np.linspace(0, 1, 9))
    edges[-1] += 1e-9
    return np.clip(np.digitize(vals, edges) - 1, 0, 7)


def main():
    print("computing image axes (FFT over 2800 images)...", flush=True)
    sf, domori = image_axes()
    lab_sf_img = octile_labels(sf)
    # orientation: 8 equal pi-periodic bins (not octiles: the axis is angular)
    lab_ori_img = np.clip(np.digitize(domori,
                                      np.linspace(-np.pi / 2, np.pi / 2, 9))
                          - 1, 0, 7)
    print(f"  SF range {sf.min():.3f}-{sf.max():.3f} cyc/px; "
          f"ori class counts {np.bincount(lab_ori_img, minlength=8)}",
          flush=True)

    out = {"design": "declared-axis search on natimg2800",
           "axes": ["spatial_frequency", "dominant_orientation", "random"],
           "sessions": {}}
    for fname in SESSIONS:
        name = fname.replace(".mat", "")
        print(f"\n=== {name}", flush=True)
        st = sio.loadmat(DATA / fname)["stim"][0, 0]
        X = st["resp"].astype(np.float32)  # trials x neurons
        istim = st["istim"].ravel().astype(int)
        keep = istim <= 2800  # drop blank-screen trials
        X, istim = X[keep], istim[keep] - 1  # to 0-based image index
        n_trials = X.shape[0]
        print(f"  {X.shape[1]} neurons, {n_trials} image trials", flush=True)

        rng = np.random.default_rng(999)
        labelings = [("spatial_frequency", lab_sf_img[istim]),
                     ("dominant_orientation", lab_ori_img[istim]),
                     ("random", rng.integers(0, 8, n_trials))]
        sess = {"n_neurons": int(X.shape[1]), "n_trials": int(n_trials),
                "axes": {}}
        for axis_name, labels in labelings:
            d, sh = r27.ladder_delta(X, labels, 8, np.random.default_rng(11))
            counts = np.bincount(labels, minlength=8)
            sess["axes"][axis_name] = {
                "delta": d, "shuffle_mean": sh,
                "class_counts": counts.tolist()}
            print(f"  [{axis_name}] delta {d:+.4f} | shuffle {sh:+.4f} | "
                  f"counts {counts.min()}-{counts.max()}", flush=True)
        out["sessions"][name] = sess
        json.dump(out, open(OUT, "w"), indent=1)

    print("\nDONE run30", flush=True)


if __name__ == "__main__":
    main()
