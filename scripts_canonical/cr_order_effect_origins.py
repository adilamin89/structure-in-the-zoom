"""Camera-ready check (Oct 20): is the order effect of Sec 4 / App E ("adjacent classes climb faster in all eight") a property of the
recordings or of the declared origin? cr_overlap_mining_followup.py (d5) found the within-class order effect varies with
the origin (SD 0.03-0.10 over 16 cyclic origins) and that the stationary overlap kernel predicts its origin average within
0.01 in 8/8. Here the TOTAL order effect (paired - sequential, sixteen 22.5-degree classes, each arm its own ten-draw floor,
run72 design B; engine of cr_order_and_withinbin_controls), with its within-class and class-mean parts, at all sixteen
cyclic origins. Origin 0 reproduces App E / Table 9.

Expectations (written before the run): E1 origin-averaged total order effect negative in >= 6 of 8 (stationary theory:
within part negative in 6, positive in GT3, ~0 in low-contrast GT2); E2 negative at >= 12 of 16 origins in the three
localized recordings (their stationary within part is the largest, -0.14 to -0.19); E3 the origin-averaged within part
matches the stationary prediction of the follow-up within 0.02 in >= 7 of 8 (floors cancel on average).
Out: data_canonical/cr_order_effect_origins.json"""
import importlib.util
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("c", HERE / "cr_order_and_withinbin_controls.py"); CO = importlib.util.module_from_spec(spec); spec.loader.exec_module(CO)
OUT = HERE.parent / "data_canonical" / "cr_order_effect_origins.json"


def run_one(fp):
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab16 = CO.bins(phi, 16); m16 = [np.where(lab16 == b)[0] for b in range(16)]
    G = (X @ X.T).astype(np.float64)
    res = []
    for o in range(16):
        binsP = [[(j + o) % 16 for k in range(r + 1) for j in (k, k + 8)] for r in range(8)]
        binsS = [[(j + o) % 16 for k in range(r + 1) for j in (2 * k, 2 * k + 1)] for r in range(8)]
        P = [np.concatenate([m16[j] for j in s]) for s in binsP]; S = [np.concatenate([m16[j] for j in s]) for s in binsS]
        po = CO.ladder(G, lab16, P, np.random.default_rng(42)); so = CO.ladder(G, lab16, S, np.random.default_rng(43))
        res.append({k: po[k] - so[k] for k in po})
    tot = np.array([r["delta"] for r in res]); w = np.array([r["within"] for r in res])
    print(f"{fp.stem[9:30]:22s} total OE origin0 {tot[0]:+.3f} mean {tot.mean():+.3f} sd {tot.std():.3f} neg {int((tot < 0).sum())}/16"
          f" | within mean {w.mean():+.3f} | class-means mean {np.mean([r['class_means'] for r in res]):+.3f} ({time.time() - t0:.0f}s)", flush=True)
    return {"name": fp.stem, "origins": res}


def main():
    files = sorted(CO.RAW.glob("gratings*.npy"))
    with mp.get_context("spawn").Pool(4) as pool:
        rows = pool.map(run_one, files)
    json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
