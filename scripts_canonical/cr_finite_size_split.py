"""Why the direction shift needs ~200 stimuli on GT3: the Sec 7 covariance split along the sampling axis (C5, Oct 20).

For each stimulus budget n, DRAWS random subsets of n presentations (labels kept) run the paper's direction ladder
(centred PR, rungs 1,2,3,4,6,8 of 8 bins, rungs < 10 samples skipped, floor = 10 random ladders) split exactly into
the within-class part log(w1^2/w2) and the class-mean part (run72 / App K identity). Also recorded per budget: the
within-class PR of single classes (each class's presentations centred on their own mean), averaged over classes, and the
same at the full set, inverted through Eq. 2 to an effective within-class dimension d_w.

EXPECTATION (written before the run): the shift is carried by the within-class part (Sec 7), and a class's fluctuation
geometry is resolved only once its presentations exceed its own effective dimension d_w (the Eq. 2 crossover applied to
one class). So (i) the within-class part, and with it delta, should stay near zero while m = n/8 presentations per class
is below ~d_w and rise across m ~ d_w; (ii) d_w should be a few tens (Sec 7: within-class PR ~ 20 in the calibrated
model), putting the crossover at n ~ 8 d_w ~ 150-300 stimuli; (iii) the single-class PR at budget n should follow
Eq. 2's d_w (m - 1)/(d_w + m).
Input: raw/gratings_drifting_GT3.npy. Out: data_canonical/cr_finite_size_split.json"""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw" / "gratings_drifting_GT3.npy"   # Stringer et al. 2021 grating recording (Figshare 8279387), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_finite_size_split.json"
spec = importlib.util.spec_from_file_location("run63", HERE / "run63_deficit_decomposition.py")
r63 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r63)
N_BINS, BIN_COUNTS, N_NULL, MIN_RUNG = 8, [1, 2, 3, 4, 6, 8], 10, 10
BUDGETS = {50: 10, 100: 10, 200: 10, 300: 10, 500: 10, 1000: 6, 2000: 4, 4442: 3}   # budget: draws


def parts(Y, lab):
    b1, w1, b2, w2, x = r63.traces(Y, lab)
    return np.array([np.log((b1 + w1) ** 2 / (b2 + w2 + 2 * x)), np.log(w1 ** 2 / w2),
                     np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2))])          # total, within, class-mean


def pr_centred(Y):
    Y = Y - Y.mean(0)
    G = (Y @ Y.T).astype(np.float64)
    return float(np.trace(G) ** 2 / (G * G).sum())


def split_ladder(Xt, bins, rng):
    members = [np.where(bins == b)[0] for b in range(N_BINS)]
    sel = [s for s in (np.concatenate(members[:c]) for c in BIN_COUNTS) if len(s) >= MIN_RUNG]
    if len(sel) < 2:
        return None
    sizes = np.log([len(s) for s in sel])
    lad = np.array([parts(Xt[s], bins[s]) for s in sel])
    flo = np.array([np.mean([parts(Xt[i], bins[i]) for i in
                             (rng.choice(len(Xt), len(s), replace=False) for _ in range(N_NULL))], 0) for s in sel])
    sl = [float(np.polyfit(sizes, lad[:, j] - flo[:, j], 1)[0]) for j in range(3)]
    wpr = np.mean([pr_centred(Xt[m]) for m in members if len(m) >= 2])
    return dict(delta=sl[0], within=sl[1], class_means=sl[2], within_pr_single=float(wpr),
                per_class=float(np.mean([len(m) for m in members])))


def main():
    d = np.load(RAW, allow_pickle=True).item()
    X = np.asarray(d["sresp"], dtype=np.float32).T
    X /= X.std() + 1e-9
    bins = np.clip(np.digitize(np.asarray(d["istim"], float), np.linspace(0, 2 * np.pi, N_BINS + 1)) - 1, 0, N_BINS - 1)
    full = [pr_centred(X[bins == b]) for b in range(N_BINS)]
    m_full = np.array([(bins == b).sum() for b in range(N_BINS)])
    d_w = [p * m / (m - 1 - p) for p, m in zip(full, m_full)]                     # Eq. 2 inverted per class
    print("single-class within PR at full set %.1f-%.1f; Eq.2-inverted d_w %.1f-%.1f (mean %.1f)"
          % (min(full), max(full), min(d_w), max(d_w), np.mean(d_w)), flush=True)
    rows = []
    for n, draws in BUDGETS.items():
        reps = []
        for k in range(draws):
            rng = np.random.default_rng(20_000 * n + k)
            idx = np.arange(len(X)) if n >= len(X) else np.sort(rng.choice(len(X), n, replace=False))
            r = split_ladder(X[idx], bins[idx], rng)
            if r is not None:
                reps.append(r)
        agg = {k: (float(np.mean([r[k] for r in reps])), float(np.std([r[k] for r in reps], ddof=1)))
               for k in ("delta", "within", "class_means", "within_pr_single", "per_class")}
        m = agg["per_class"][0]
        pred = float(np.mean([dw * (m - 1) / (dw + m) for dw in d_w]))
        rows.append(dict(n=n, draws=len(reps), eq2_single_class_pr=pred, **agg))
        print(f"n={n:5d} m={m:6.1f}  delta {agg['delta'][0]:+.3f}+-{agg['delta'][1]:.3f}  within {agg['within'][0]:+.3f}"
              f"+-{agg['within'][1]:.3f}  class-means {agg['class_means'][0]:+.3f}  single-class PR "
              f"{agg['within_pr_single'][0]:5.1f} (Eq.2 {pred:5.1f})", flush=True)
    json.dump(dict(full_single_class_pr=full, d_w=d_w, rows=rows), open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
