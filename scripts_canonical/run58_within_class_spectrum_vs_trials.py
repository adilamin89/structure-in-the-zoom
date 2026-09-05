"""Run 58 (S77, 2026-09-05) - Is the within-class spectrum the calibration the
model gets wrong? A diagnostic on the raw data, no model.

The calibrated model (runs 2, 11b, 48, 56, 57) represents within-class
variability by K shared low-rank modes with a FLAT spectrum, K set to the
within-class participation ratio measured at 120 trials per class (18.5,
23.3, 20.1 on GT1, GT2, GT3). A flat K-mode spectrum saturates: its PR is K
at any trial count above K. A heavy-tailed within-class spectrum does not:
its PR keeps growing with the number of trials as the tail is resolved. The
direction ladder's first rung is one class at ~550 trials and its last is
eight classes at ~4400, so if the real within-class PR grows with trials the
model's rung-1 PR is wrong by the growth factor, and delta = slope difference
is mis-set in a way that can differ between recordings.

MEASUREMENTS: for GT1, GT2, GT3 (drifting gratings), per class, the centered
participation ratio of the within-class trial matrix at n = 60, 120, 250, 500
trials (10 random draws each, mean), and the full-population random-subset PR
at the same sizes (the floor's own curve); the growth factor PR(500)/PR(120)
per recording; and the ratio of within-class PR to random-subset PR at 500.

REGISTERED EXPECTATIONS (before the run):
W1: within-class PR grows with trials by more than 30% from 120 to 500 on all
    three recordings (heavy tail; the flat-K calibration is wrong).
W2: the growth factor orders GT1 >= GT2 > GT3 (the recordings the model
    under-predicts have the heavier within-class tails).
W3 (null alternative, stated): if growth is under 15% everywhere, the
    within-class spectrum is not the missing calibration and the search moves
    to neuron heterogeneity (a direction-selective subpopulation).

Out: ../data_canonical/run58_within_class_spectrum_vs_trials.json (+ .log)
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = DATA / "run58_within_class_spectrum_vs_trials.json"
N_BINS = 8
SIZES = [60, 120, 250, 500]
RECS = {"GT1": "gratings_drifting_GT1_2019_04_12_1", "GT2": "gratings_drifting_GT2_2019_04_05_1", "GT3": "gratings_drifting_GT3_2019_04_05_1"}


def pr_c(X):
    Xc = X - X.mean(axis=0)
    G = (Xc @ Xc.T).astype(np.float64) / max(X.shape[0] - 1, 1)
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if tr2 > 0 else 1.0


def main():
    out = {"sizes": SIZES, "rows": {}}
    for tag, name in RECS.items():
        dat = np.load(RAW / f"{name}.npy", allow_pickle=True).item()
        X = np.asarray(dat["sresp"], np.float32); X /= X.std() + 1e-9
        istim = np.asarray(dat["istim"], float)
        bl = np.clip(np.digitize(istim, np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
        Xt = np.ascontiguousarray(X.T); del dat, X
        rng = np.random.default_rng(0)
        within = {}; floor = {}
        for n in SIZES:
            vals = []
            for b in range(N_BINS):
                idx_all = np.where(bl == b)[0]
                if len(idx_all) < n: continue
                for d in range(10):
                    idx = rng.choice(idx_all, n, replace=False); vals.append(pr_c(Xt[idx]))
            within[n] = float(np.mean(vals))
            floor[n] = float(np.mean([pr_c(Xt[rng.choice(len(Xt), n, replace=False)]) for d in range(10)]))
            print(f"[{tag}] n={n}: within-class PR {within[n]:.1f} | random-subset PR {floor[n]:.1f}", flush=True)
        growth = within[500] / within[120]; fgrowth = floor[500] / floor[120]
        out["rows"][tag] = {"within_pr": within, "floor_pr": floor, "growth_120_to_500": growth, "floor_growth_120_to_500": fgrowth,
                            "within_over_floor_at_500": within[500] / floor[500]}
        print(f"  => {tag}: within growth 120->500 = {growth:.2f}x (floor {fgrowth:.2f}x); within/floor at 500 = {within[500]/floor[500]:.2f}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    g = {t: out["rows"][t]["growth_120_to_500"] for t in RECS}
    out["verdict"] = {"W1_growth_gt_1.30_all": bool(all(v > 1.30 for v in g.values())), "growth": g,
                      "W2_orders_GT1_GT2_GT3": bool(g["GT1"] >= g["GT2"] > g["GT3"]),
                      "W3_growth_lt_1.15_all": bool(all(v < 1.15 for v in g.values()))}
    print("VERDICT", json.dumps(out["verdict"], indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
