"""Hot and ordered starts must agree before the lattice ensembles are read. For every (L, T) of Fig 3A: 64 chains from a random
(hot) start and 64 from an ordered start at the production n_equil (polar-ordered side: L 32, 1500 hybrid sweeps; nematic side: 300),
5 samples spaced 10 sweeps. Per chain: energy E, nematic order S2 = |<e^{2i phi}>|, polar order M1 = |<e^{i phi}>| over its samples.
Agreement: |mean_hot - mean_ordered| <= 3 combined SE for E, S2 and M1; otherwise a 4x-longer run at that (L, T).
Run: python -X faulthandler lattice_start_check.py [n_workers]   Out: data_canonical/lattice_start_check.json"""
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data_canonical"
POINTS = [(32, 0.60, 1500), (32, 0.70, 1500), (32, 0.80, 1500), (32, 0.85, 1500), (16, 1.0, 300), (16, 1.1, 300), (32, 1.0, 300)]
N_CH = 64


def chain(args):
    import lattice_ground_truth as h
    L, T, neq, hot, seed = args
    x, en = h.xy_chain_w(L, T, 0.42, 1.0, seed, hot, neq, 5, 10)
    return float(en.mean() / (L * L)), float(np.abs(np.exp(2j * x).mean(1)).mean()), float(np.abs(np.exp(1j * x).mean(1)).mean())


def point(ex, L, T, neq):
    res = {}
    for hot in (True, False):
        seeds = [91_000_000 + 7_919 * c + 1000 * L + int(round(T * 1000)) + (0 if hot else 500_000) for c in range(N_CH)]
        v = np.array(list(ex.map(chain, [(L, T, neq, hot, s) for s in seeds])))
        res["hot" if hot else "ordered"] = {"mean": v.mean(0).tolist(), "se": (v.std(0, ddof=1) / np.sqrt(N_CH)).tolist()}
    h, o = res["hot"], res["ordered"]
    z = [(h["mean"][i] - o["mean"][i]) / np.hypot(h["se"][i], o["se"][i]) for i in range(3)]
    res.update(L=L, T=T, n_equil=neq, z=dict(zip(("E", "S2", "M1"), map(float, z))), agree=bool(all(abs(q) <= 3 for q in z)))
    return res


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = []
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        for L, T, neq in POINTS:
            r = point(ex, L, T, neq)
            if not r["agree"]:
                r["longer"] = point(ex, L, T, 4 * neq)
            out.append(r)
            print(f"L{L} T{T} n_equil {neq}: hot E {r['hot']['mean'][0]:.4f} S2 {r['hot']['mean'][1]:.4f} M1 {r['hot']['mean'][2]:.4f} | "
                  f"ordered E {r['ordered']['mean'][0]:.4f} S2 {r['ordered']['mean'][1]:.4f} M1 {r['ordered']['mean'][2]:.4f} | z "
                  + " ".join(f"{k} {v:+.2f}" for k, v in r["z"].items()) + f" | agree {r['agree']}"
                  + (f" | 4x: z " + " ".join(f"{k} {v:+.2f}" for k, v in r["longer"]["z"].items()) if "longer" in r else ""), flush=True)
            json.dump(out, open(DATA / "lattice_start_check.json", "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
