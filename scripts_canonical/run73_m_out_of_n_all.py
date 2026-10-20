"""Run 73 (2026-10-05): m-out-of-n (without replacement) bootstrap of the direction shift on all eight grating
recordings. Registration: ../data_canonical/run73_REGISTERED.md. PR = Tr(G)^2 / Tr(G^2) on the trial Gram matrix."""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent / "raw"
OUT = DATA / "run73_m_out_of_n_all.json"
RUNGS = (1, 2, 3, 4, 6, 8)
N_REP, N_NULL, FRAC = 100, 10, 0.8


def pr(Y):
    Y = Y - Y.mean(0); G = Y @ Y.T
    return float(np.trace(G) ** 2 / (G * G).sum())


def delta(Xt, lab, rng):
    m = [np.where(lab == b)[0] for b in range(8)]
    sets = [np.concatenate(m[:c]) for c in RUNGS]; sizes = np.log([len(s) for s in sets])
    lad = [np.log(pr(Xt[s])) for s in sets]
    flo = [np.mean([np.log(pr(Xt[rng.choice(len(Xt), len(s), replace=False)])) for _ in range(N_NULL)]) for s in sets]
    return float(np.polyfit(sizes, lad, 1)[0] - np.polyfit(sizes, flo, 1)[0])


def main():
    t0 = time.time(); out = {"registration": "run73_REGISTERED.md", "rows": {}}
    for fp in sorted(RAW.glob("gratings*.npy")):
        dat = np.load(fp, allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        phi = np.asarray(dat["istim"], float).ravel() % (2 * np.pi); del dat
        lab = np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
        rng = np.random.default_rng(73)
        point = delta(X, lab, rng)
        reps = []
        for r in range(N_REP):
            keep = np.concatenate([rng.choice(np.where(lab == b)[0], int(FRAC * (lab == b).sum()), replace=False) for b in range(8)])
            reps.append(delta(X[keep], lab[keep], rng))
            if (r + 1) % 10 == 0:
                print(f"{fp.stem[:28]} rep {r+1}/{N_REP} | {time.time()-t0:.0f}s", flush=True)
        lo, hi = np.percentile(reps, [2.5, 97.5])
        out["rows"][fp.stem] = {"point": point, "centre": float(np.median(reps)), "ci95": [float(lo), float(hi)], "reps": reps}
        print(f"{fp.stem}: point {point:+.3f} centre {np.median(reps):+.3f} [{lo:+.3f}, {hi:+.3f}]", flush=True)
        json.dump(out, open(OUT, "w"), indent=1); del X
    R = out["rows"]
    p1 = sum(r["ci95"][0] <= r["point"] <= r["ci95"][1] for r in R.values())
    p2 = sum(abs(r["centre"] - r["point"]) < 0.03 for r in R.values())
    out["verdict"] = {"P1_count": int(p1), "P1": bool(p1 == len(R)), "P2_count": int(p2), "P2": bool(p2 >= len(R) - 1)}
    print("VERDICT", json.dumps(out["verdict"]), flush=True); json.dump(out, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
