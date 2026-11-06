"""Signal against the largest noise mode, beside the alignment a. Written and committed before it was run.

Question (literature check): Rumyantsev et al. 2020 (Nature 580:100) found in mouse V1 that "visual signals were perpendicular
to the largest noise mode". Sec 5 says the class means lie along the fluctuations (a = (Tr WB/Tr B)/(Tr W^2/Tr W) = 1.0 to 2.1
in the eight grating recordings). The two statements weigh different things: a weighs the class means against the whole
fluctuation spectrum, variance-weighted; Rumyantsev weigh the signal (the difference between the mean responses to two nearby
stimuli) against the single largest mode. This script measures both on the same data.

Per recording (the paper's input: global SD scaling, eight 45-degree direction classes, within-class residuals around the class
means, all neurons): the largest within-class mode u1 and its share of the fluctuation variance lambda1/Tr W; for each pair of
adjacent classes the signal d = mu_{c+1} - mu_c, its cos^2 with u1 against chance 1/N, and its own alignment
a_d = (d'Wd/|d|^2)/(Tr W^2/Tr W); the same cos^2 for the centred class means; and a (as allen_neuron_count_control.sign_law).

Registered reading (before the run): Rumyantsev's statement holds here if the mean cos^2(d, u1) over the eight adjacent pairs is
below lambda1/Tr W (the signal holds less of the top mode than a random direction drawn from the fluctuations would) in at least
6 of 8 recordings. Either outcome is reported; a >= 1 together with a small cos^2(d, u1) means the class means lie in the
fluctuations' high-variance subspace without lying along its single largest mode.
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler signal_noise_alignment.py
Out: data_canonical/signal_noise_alignment.json"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
OUT = HERE.parent / "data_canonical" / "signal_noise_alignment.json"


def bins(phi, k):
    """Eight equal-width direction bins from 0 (as cr_order_and_withinbin_controls.bins)."""
    return np.clip(np.digitize(np.mod(phi, 2 * np.pi), np.linspace(0, 2 * np.pi, k + 1)) - 1, 0, k - 1)


def one(fp):
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float64).T; X /= X.std() + 1e-9
    lab = bins(np.asarray(dat["istim"], float).ravel(), 8); del dat
    n, N = X.shape
    X -= X.mean(0)
    mu = np.stack([X[lab == c].mean(0) for c in range(8)]); p = np.bincount(lab, minlength=8) / n
    R = X - mu[lab]
    G = R @ R.T / n                                     # n x n; same nonzero spectrum as W = R'R/n
    w1, w2 = float(np.trace(G)), float((G * G).sum())
    ev, V = np.linalg.eigh(G); lam1 = float(ev[-1]); u1 = R.T @ V[:, -1]; u1 /= np.linalg.norm(u1)
    Bh = np.sqrt(p)[:, None] * mu
    b1 = float((Bh * Bh).sum()); x = float(((R @ Bh.T) ** 2).sum() / n)
    a = (x / b1) / (w2 / w1)
    pairs = []
    for c in range(8):
        d = mu[(c + 1) % 8] - mu[c]; dn = d / np.linalg.norm(d)
        pairs.append({"pair": [c, (c + 1) % 8], "cos2_u1": float((dn @ u1) ** 2),
                      "a_d": float((np.sum((R @ dn) ** 2) / n) / (w2 / w1))})
    cm = [float(((m / np.linalg.norm(m)) @ u1) ** 2) for m in mu]
    out = {"name": fp.stem, "N": int(N), "n": int(n), "lambda1_share": lam1 / w1, "chance": 1.0 / N, "a": a,
           "cos2_d_u1_mean": float(np.mean([q["cos2_u1"] for q in pairs])), "a_d_mean": float(np.mean([q["a_d"] for q in pairs])),
           "cos2_classmean_u1_mean": float(np.mean(cm)), "pairs": pairs}
    print(f"{fp.stem[9:32]:24s} N {N:6d} a {a:5.2f} | top mode {lam1 / w1:.3f} of Tr W | cos2(d,u1) "
          f"{out['cos2_d_u1_mean']:.4f} (chance {1 / N:.1e}) a_d {out['a_d_mean']:.2f} | cos2(mu,u1) {np.mean(cm):.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    rows = [one(fp) for fp in sorted(RAW.glob("gratings*.npy"))]
    held = sum(r["cos2_d_u1_mean"] < r["lambda1_share"] for r in rows)
    verdict = {"rumyantsev_reading_holds": f"{held}/{len(rows)}", "registered_bar": ">= 6/8",
               "a_range": [min(r["a"] for r in rows), max(r["a"] for r in rows)]}
    print(verdict, flush=True)
    json.dump({"rows": rows, "verdict": verdict}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
