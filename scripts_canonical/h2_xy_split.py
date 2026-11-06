"""H2-R37: what sets the sign of the XY director-ladder shift. Regenerates each H2-R34 ensemble exactly (same seeds,
same sampler; asserts the stored delta_director is reproduced) and applies the paper's exact split (run72 logparts:
log PR = log(w1^2/w2) [within-class] + log((1 + b1/w1)^2 / (1 + (b2 + 2x)/w2)) [class means], per subset with the
subset's own class labels, floors split the same way) to the director ladder, plus the full-set sign statistics of
App K (Tr B/Tr W, PR(W), PR(B), the threshold 1 + 2 Tr W/Tr B, the class-mean term c).
Usage: python h2_xy_split.py L T J1 [start n_equil]   -> data_canonical/h2_xy_split_L{L}_T{T}_J{J1}[tag].json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h2_lattice as h  # noqa: E402
import nematic_polar_delta as nem  # noqa: E402

spec = importlib.util.spec_from_file_location("run72", HERE / "run72_paper_designs_split.py")
r72 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r72)


def ensemble(L, T, J1, e, start, n_equil, chains=40, samples=5, spacing=10):
    cfgs = []
    for c in range(chains):
        seed = 77_000_000 + 100_000 * int(round(J1 * 100)) + 1000 * L + 40 * e + c + int(round(T * 1000)) * 7
        x, _ = h.xy_chain_w(L, T, J1, 1.0, seed, start == "hot", n_equil, samples, spacing)
        cfgs.append(x)
    return np.concatenate(cfgs)


def main():
    L, T, J1 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    start = sys.argv[4] if len(sys.argv) > 4 else "hot"
    ne = int(sys.argv[5]) if len(sys.argv) > 5 else None
    tag = "" if start == "hot" and ne is None else f"_{start}{ne}"
    n_equil = ne or (300 if L <= 32 else 500)
    art = json.load(open(h.OUTDIR / f"h2_lattice_xy_L{L}_T{T:.2f}_J{J1:.2f}{tag}.json"))
    rows = []
    for e in range(art["n_ens"]):
        cfg = ensemble(L, T, J1, e, start, n_equil)
        _, s2, psi = h.order_params(cfg)
        feat = nem.features(cfg)
        rng = np.random.default_rng(1_000 + e)
        d_dir, sizes, counts = h.director_ladder(feat, psi, rng)
        assert abs(d_dir - art["ensembles"][e]["delta_director"]) < 1e-9, (e, d_dir, art["ensembles"][e]["delta_director"])
        lab = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
        order = np.argsort(lab, kind="stable")
        sets = [order[:s] for s in sizes]
        F = feat - feat.mean(0)
        F /= F.std() + 1e-12
        sp = r72.ladder(F.astype(np.float32), lab, sets, np.random.default_rng(42 + e))
        b1, w1, b2, w2, x = r72.r63.traces(F, lab)
        rec = {"ens": e, "delta_director": d_dir, "split_delta": sp["delta"], "within_part": sp["slope_noise"],
               "class_mean_part": sp["slope_probe"], "TrB_TrW": b1 / w1, "PR_W": w1 ** 2 / w2, "PR_B": b1 ** 2 / b2,
               "threshold": 1 + 2 * w1 / b1, "c_full": float(np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2))),
               "c_without_x": float(np.log((1 + b1 / w1) ** 2 / (1 + b2 / w2))), "x_share": 2 * x / (b2 + 2 * x)}
        rec["ratio"] = rec["PR_W"] / rec["PR_B"]
        rows.append(rec)
        print(f"ens {e}: delta {d_dir:+.4f} (split est {sp['delta']:+.4f}) within {rec['within_part']:+.4f} "
              f"class-mean {rec['class_mean_part']:+.4f} | TrB/TrW {rec['TrB_TrW']:.3f} PR_W {rec['PR_W']:.1f} "
              f"PR_B {rec['PR_B']:.2f} ratio {rec['ratio']:.1f} thr {rec['threshold']:.1f} c {rec['c_full']:+.3f}",
              flush=True)
    out = {"L": L, "T": T, "J1": J1, "start": start, "n_equil": n_equil, "rows": rows}
    for k in ("within_part", "class_mean_part", "split_delta", "TrB_TrW", "PR_W", "PR_B", "threshold", "c_full"):
        v = np.array([r[k] for r in rows])
        out[k] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v)))}
    out["sign_law_hits"] = int(sum((r["ratio"] > r["threshold"]) == (r["class_mean_part"] < 0) for r in rows))
    json.dump(out, open(h.OUTDIR / f"h2_xy_split_L{L}_T{T:.2f}_J{J1:.2f}{tag}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
