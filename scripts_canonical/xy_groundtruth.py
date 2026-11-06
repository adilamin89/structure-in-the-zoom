"""A constructed system whose answer is known, beside the XY lattice (App F ground-truth pair).

Spin-wave model: phi_i = psi + u_i (+ pi with probability 1/2: a random polar sign), psi uniform on [0, pi), u a Gaussian
free field on the L x L torus with exp(-2<u^2>) = S2 (the lattice's measured S2 at L = 32, T = 1.0, 0.625). Each director
class's fluctuations are the field u multiplied, site by site, by the class's tangent direction (-sin 2psi, cos 2psi) in
the (cos 2phi, sin 2phi) features: the fluctuation subspace rotates with the director by construction. Same features,
director bins, ladder, floor and App K split as the lattice (xy_negative_shift.analyse).
Per arm (spin wave; lattice at T = 1.0 from the R39 ensembles, regenerated): 8 seeds (spin wave) at 25 and 200
configurations per class; delta, within part, class-mean part with t intervals; split-half fluctuation shape overlap
Kn(s) = K_cd / sqrt(K_cc K_dd) by director-class separation s = 0..4 (overlap_identity_checks.split_half_K); per-rung remainder
curves (declared minus floor) of the class-mean term and the full-set term c, for the S1 sign-law scoring at large classes.
Out: data_canonical/xy_groundtruth.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lattice_ground_truth as h  # noqa: E402
import xy_negative_shift as xm  # noqa: E402
import xy_onset as xo  # noqa: E402
import nematic_polar_delta as nem  # noqa: E402

spec = importlib.util.spec_from_file_location("m", HERE / "overlap_identity_checks.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

L, S2 = 32, 0.625


def spinwave(n, seed):
    rng = np.random.default_rng(seed)
    var_u = -np.log(S2) / 2
    out = np.empty((n, L * L))
    for i in range(n):
        ps = rng.uniform(0, np.pi) + np.pi * rng.integers(0, 2)
        out[i] = (ps + xm.gff(L, var_u, rng).ravel()) % (2 * np.pi)
    return out


def overlap_profile(cfg, seed):
    _, _, psi = h.order_params(cfg)
    lab = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
    F = nem.features(cfg)
    F = F - F.mean(0)
    rng = np.random.default_rng(seed)
    A, B = [], []
    for c in range(nem.N_BINS):
        R = F[lab == c] - F[lab == c].mean(0)
        p = rng.permutation(len(R))
        A.append(R[p[: len(R) // 2]])
        B.append(R[p[len(R) // 2:]])
    K = M.split_half_K(A, B)
    kd = np.diag(K)
    Kn = K / np.sqrt(np.outer(kd, kd))
    sep = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
    return [float(Kn[sep == s].mean()) for s in range(5)]


def remainder_curves(cfg, seed):
    _, _, psi = h.order_params(cfg)
    F = nem.features(cfg)
    rng = np.random.default_rng(seed)
    d, sizes, _ = h.director_ladder(F, psi, rng)
    lab = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
    order = np.argsort(lab, kind="stable")
    G = F - F.mean(0)
    G /= G.std() + 1e-12
    sp = xm.xs.r72.ladder(G.astype(np.float32), lab, [order[:s] for s in sizes], np.random.default_rng(seed + 7))
    b1, w1, b2, w2, x = xm.xs.r72.r63.traces(G, lab)
    return {"sizes": sizes, "probe_declared": [float(l["probe"]) for l in sp["ladder"]],
            "probe_floor": [float(f["probe"]) for f in sp["floor"]],
            "noise_declared": [float(l["noise"]) for l in sp["ladder"]], "noise_floor": [float(f["noise"]) for f in sp["floor"]],
            "class_mean_part": sp["slope_probe"], "within_part": sp["slope_noise"], "delta": d,
            "c_full": float(np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2))),
            "PR_W": w1 ** 2 / w2, "PR_B": b1 ** 2 / b2, "threshold": 1 + 2 * w1 / b1}


def summarize(rows, key):
    v = np.array([r[key] for r in rows])
    ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else (v[0], v[0])
    return {"mean": float(v.mean()), "ci95": [float(ci[0]), float(ci[1])], "n": int(len(v)), "n_pos": int((v > 0).sum())}


def main():
    out = {}
    for per_class in (25, 200):
        rows = []
        for s in range(8):
            cfg = spinwave(8 * per_class, 40_000 + 100 * per_class + s)
            r = remainder_curves(cfg, 4000 + s)
            r["Kn_profile"] = overlap_profile(cfg, 4100 + s)
            rows.append(r)
            print(f"spinwave {per_class}/class seed {s}: d {r['delta']:+.3f} w {r['within_part']:+.3f} m {r['class_mean_part']:+.3f}"
                  f" c {r['c_full']:+.3f} PR_W {r['PR_W']:.1f} Kn {np.round(r['Kn_profile'], 3).tolist()}", flush=True)
        out[f"spinwave_{per_class}"] = {"rows": rows, **{k: summarize(rows, k) for k in ("delta", "within_part", "class_mean_part")}}
        json.dump(out, open(h.OUTDIR / "xy_groundtruth.json", "w"), indent=1)
    for per_class, n_ens in ((25, 4), (100, 2), (200, 2)):
        rows = []
        for e in range(n_ens):
            cfg = xo.lattice_ensemble(per_class * 8 // 5, e)
            r = remainder_curves(cfg, 5000 + e)
            r["Kn_profile"] = overlap_profile(cfg, 5100 + e)
            rows.append(r)
            print(f"lattice T=1.0 {per_class}/class ens {e}: d {r['delta']:+.3f} w {r['within_part']:+.3f} "
                  f"m {r['class_mean_part']:+.3f} c {r['c_full']:+.3f} PR_W {r['PR_W']:.0f} Kn {np.round(r['Kn_profile'], 3).tolist()}",
                  flush=True)
        out[f"lattice_{per_class}"] = {"rows": rows, **{k: summarize(rows, k) for k in ("delta", "within_part", "class_mean_part")}}
        json.dump(out, open(h.OUTDIR / "xy_groundtruth.json", "w"), indent=1)
    allr = [r for k, v in out.items() for r in v["rows"]]
    out["S1_large_classes"] = {
        "law_vs_ladder_part": int(sum((r["PR_W"] / r["PR_B"] > r["threshold"]) == (r["class_mean_part"] < 0) for r in allr)),
        "c_full_vs_ladder_part": int(sum((r["c_full"] < 0) == (r["class_mean_part"] < 0) for r in allr)), "n": len(allr)}
    print("S1", out["S1_large_classes"], flush=True)
    json.dump(out, open(h.OUTDIR / "xy_groundtruth.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
