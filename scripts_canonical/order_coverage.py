"""Coverage of the shift's permutation inference under pseudo-labels.
On two grating recordings, balanced pseudo-labels with no stimulus information are assigned to the presentations (iid, or
constant within contiguous temporal blocks of 50 presentations), the declared 8-class ladder (rungs 1,2,3,4,6,8; centred PR
via the double-centred Gram, = run37/paper estimator) is measured against a fresh 10-draw floor, and tested against 200
label permutations (one-sided p = (1 + #null >= obs)/201). Rejection rate at alpha 0.05 with its Clopper-Pearson interval.
Usage: python order_coverage.py n_experiments -> data_canonical/order_coverage.json"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_battery as hb  # noqa: E402

RAW = HERE.parent / "raw"
FILES = ["gratings_drifting_GT1", "gratings_local_GT3"]
COUNTS = [1, 2, 3, 4, 6, 8]


def pseudo(n, kind, rng, block=50):
    if kind == "iid":
        lab = np.repeat(np.arange(8), int(np.ceil(n / 8)))[:n]
        return rng.permutation(lab)
    nb = int(np.ceil(n / block))
    blab = np.repeat(np.arange(8), int(np.ceil(nb / 8)))[:nb]
    blab = rng.permutation(blab)
    return np.repeat(blab, block)[:n]


def experiment(G, lab, rng):
    mem = hb.members_of(lab, 8)
    th, sizes = hb.ladder(G, mem, list(range(8)), COUNTS)
    th_f = hb.floor(G, len(lab), sizes, int(rng.integers(1 << 30)))
    obs = th - th_f
    null = np.array([hb.ladder(G, hb.members_of(lab[rng.permutation(len(lab))], 8), list(range(8)), COUNTS)[0] - th_f
                     for _ in range(200)])
    return obs, float((1 + (null >= obs).sum()) / 201)


def main():
    n_exp = int(sys.argv[1])
    out = {}
    for stem in FILES:
        fp = sorted(RAW.glob(stem + "*.npy"))[0]
        d = np.load(fp, allow_pickle=True).item()
        X = np.asarray(d["sresp"], np.float32).T.astype(np.float64)
        X /= X.std()
        G = X @ X.T
        n = len(X)
        out[stem] = {}
        for kind in ("iid", "block50"):
            rng = np.random.default_rng(2000 + (kind == "block50"))
            ps = []
            for e in range(n_exp):
                _, p = experiment(G, pseudo(n, kind, rng), rng)
                ps.append(p)
                if e % 10 == 9:
                    print(f"{stem} {kind} {e + 1}/{n_exp}: rejection so far {np.mean(np.array(ps) <= 0.05):.3f}", flush=True)
            k = int((np.array(ps) <= 0.05).sum())
            ci = stats.binomtest(k, n_exp).proportion_ci(method="exact")
            out[stem][kind] = {"n": n_exp, "rejections": k, "rate": k / n_exp, "ci95": [ci.low, ci.high], "p_values": ps}
            print(stem, kind, out[stem][kind]["rate"], [round(ci.low, 3), round(ci.high, 3)], flush=True)
            json.dump(out, open(hb.OUTDIR / "order_coverage.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
