"""Why the XY director-ladder shift is negative. Arms on the regenerated XY ensembles (xy_split.ensemble):
A1 paper features; A2 nematic features only; A3 within-bin director spread removed exactly (each configuration rotated so
its director sits at its bin centre); A4 Gaussian spin-wave model at the measured S2 (phi = psi + u, psi uniform, random
polar sign, u a Gaussian free field with exp(-2<u^2>) = S2). Each arm: director ladder delta (lattice_ground_truth.director_ladder)
and the App K split (run72 ladder + traces).
Usage: python xy_negative_shift.py L T J1 [start n_equil] -> data_canonical/xy_negative_shift_L{L}_T{T}_J{J1}[tag].json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lattice_ground_truth as h  # noqa: E402
import xy_split as xs  # noqa: E402
import nematic_polar_delta as nem  # noqa: E402


def gff(L, var_target, rng):
    k = 2 * np.pi * np.fft.fftfreq(L)
    lam = 4 - 2 * np.cos(k)[:, None] - 2 * np.cos(k)[None, :]
    lam[0, 0] = np.inf
    w = (rng.normal(size=(L, L)) + 1j * rng.normal(size=(L, L))) / np.sqrt(lam)
    u = np.real(np.fft.ifft2(w))
    return u * np.sqrt(var_target / u.var())


def analyse(cfg, rng_seed, feats="all"):
    _, s2, psi = h.order_params(cfg)
    F = nem.features(cfg)
    if feats == "nematic":
        n = cfg.shape[1]
        F = F[:, 2 * n:]
    rng = np.random.default_rng(rng_seed)
    d, sizes, _ = h.director_ladder(F, psi, rng)
    lab = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
    order = np.argsort(lab, kind="stable")
    G = F - F.mean(0)
    G /= G.std() + 1e-12
    sp = xs.r72.ladder(G.astype(np.float32), lab, [order[:s] for s in sizes], np.random.default_rng(rng_seed + 7))
    b1, w1, b2, w2, x = xs.r72.r63.traces(G, lab)
    return {"delta": d, "within": sp["slope_noise"], "class_mean": sp["slope_probe"], "PR_W": w1 ** 2 / w2,
            "PR_B": b1 ** 2 / b2, "TrB_TrW": b1 / w1, "S2": float(s2.mean())}


def main():
    L, T, J1 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
    start = sys.argv[4] if len(sys.argv) > 4 else "hot"
    ne = int(sys.argv[5]) if len(sys.argv) > 5 else None
    tag = "" if start == "hot" and ne is None else f"_{start}{ne}"
    n_equil = ne or (300 if L <= 32 else 500)
    arms = {a: [] for a in ("A1", "A2", "A3", "A4")}
    for e in range(8):
        cfg = xs.ensemble(L, T, J1, e, start, n_equil)
        arms["A1"].append(analyse(cfg, 1000 + e))
        arms["A2"].append(analyse(cfg, 1000 + e, "nematic"))
        _, _, psi = h.order_params(cfg)
        b = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
        centre = (b + 0.5) * np.pi / nem.N_BINS
        rot = (cfg - (psi - centre)[:, None]) % (2 * np.pi)
        _, _, psi_r = h.order_params(rot)
        assert np.allclose(np.minimum(np.abs(psi_r - centre), np.pi - np.abs(psi_r - centre)), 0, atol=1e-6)
        arms["A3"].append(analyse(rot, 1000 + e))
        s2 = arms["A1"][-1]["S2"]
        rng = np.random.default_rng(5000 + e)
        var_u = -np.log(max(s2, 1e-3)) / 2
        n = cfg.shape[0]
        sw = np.empty_like(cfg)
        for i in range(n):
            ps = rng.uniform(0, np.pi) + np.pi * rng.integers(0, 2)
            sw[i] = (ps + gff(L, var_u, rng).ravel()) % (2 * np.pi)
        arms["A4"].append(analyse(sw, 1000 + e))
        print(f"ens {e}: " + " | ".join(f"{a} d {arms[a][-1]['delta']:+.3f} w {arms[a][-1]['within']:+.3f} "
                                         f"m {arms[a][-1]['class_mean']:+.3f} PRW {arms[a][-1]['PR_W']:.0f}" for a in arms),
              flush=True)
    out = {"L": L, "T": T, "J1": J1, "start": start, "n_equil": n_equil, "arms": arms, "summary": {}}
    for a, rows in arms.items():
        out["summary"][a] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                                 "se": float(np.std([r[k] for r in rows], ddof=1) / np.sqrt(len(rows)))}
                             for k in rows[0]}
        s = out["summary"][a]
        print(f"{a}: delta {s['delta']['mean']:+.3f}±{s['delta']['se']:.3f} within {s['within']['mean']:+.3f}"
              f"±{s['within']['se']:.3f} class-mean {s['class_mean']['mean']:+.3f}±{s['class_mean']['se']:.3f} "
              f"PR_W {s['PR_W']['mean']:.1f} PR_B {s['PR_B']['mean']:.2f} TrB/TrW {s['TrB_TrW']['mean']:.3f}", flush=True)
    json.dump(out, open(h.OUTDIR / f"xy_negative_shift_L{L}_T{T:.2f}_J{J1:.2f}{tag}.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
