"""H2-R39: does the XY within-class part (and delta) depend on configurations per director class (finite-sample onset)?
(a) lattice ensembles of 25 / 100 / 400 configurations per class (L = 32, J1 = 0.42, T = 1.0); (b) spin waves + per-site
angular jitter. Pipeline = h2_xy_mining.analyse (director ladder delta + App K split). Out: data_canonical/h2_xy_onset.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h2_lattice as h  # noqa: E402
import h2_xy_mining as xm  # noqa: E402

L, T, J1 = 32, 1.0, 0.42


def lattice_ensemble(n_chains, e):
    cfgs = []
    for c in range(n_chains):
        seed = 55_000_000 + 10_000 * e + c + 1_000_000 * (n_chains // 40)
        x, _ = h.xy_chain_w(L, T, J1, 1.0, seed, True, 300, 5, 10)
        cfgs.append(x)
    return np.concatenate(cfgs)


def main():
    out = {"lattice": {}, "spinwave": {}}
    for per_class, n_ens in ((100, 2), (200, 2)):
        rows = []
        for e in range(n_ens):
            cfg = lattice_ensemble(per_class * 8 // 5, e)
            r = xm.analyse(cfg, 2000 + e)
            rows.append(r)
            print(f"lattice {per_class}/class ens {e}: d {r['delta']:+.3f} w {r['within']:+.3f} m {r['class_mean']:+.3f} "
                  f"PR_W {r['PR_W']:.0f} PR_B {r['PR_B']:.2f}", flush=True)
        out["lattice"][per_class] = rows
        json.dump(out, open(h.OUTDIR / "h2_xy_onset.json", "w"), indent=1)
    s2 = 0.625  # R34 L = 32, T = 1.0 measured S2
    var_u = -np.log(s2) / 2
    for v in (0.0, 0.05, 0.1, 0.2, 0.4):
        for per_class in (25, 200):
            rows = []
            for e in range(2):
                rng = np.random.default_rng(7000 + 10 * e + int(v * 100))
                n = per_class * 8
                sw = np.empty((n, L * L))
                for i in range(n):
                    ps = rng.uniform(0, np.pi) + np.pi * rng.integers(0, 2)
                    sw[i] = (ps + xm.gff(L, var_u, rng).ravel() + np.sqrt(v) * rng.normal(size=L * L)) % (2 * np.pi)
                rows.append(xm.analyse(sw, 3000 + e))
            out["spinwave"][f"v{v}_n{per_class}"] = rows
            print(f"spinwave v={v} {per_class}/class: d {np.mean([r['delta'] for r in rows]):+.3f} "
                  f"w {np.mean([r['within'] for r in rows]):+.3f} m {np.mean([r['class_mean'] for r in rows]):+.3f} "
                  f"PR_W {np.mean([r['PR_W'] for r in rows]):.0f}", flush=True)
            json.dump(out, open(h.OUTDIR / "h2_xy_onset.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
