"""H1-R36..R39 (part D, Oct 22): the Discussion's macaque prediction on CRCNS pvc-11 (Kohn & Smith 2016; the Utah arrays of
Cowley et al. 2016): anesthetized macaque V1, three animals, drifting gratings in 12 directions (0-330 by 30), 200 trials each.

Input, per the dataset's own documentation and script_mean_firing_rates_gratings.m: data.EVENTS (units x 12 directions x 200
trials; spike times in s from stimulus onset; orientation asserted), spike count per trial in [0.28, 1.28) s (Cowley's window:
the last 1 s of the 1.28 s grating), units kept with SNR >= 1.5 and mean rate >= 1 spike/s over that window (Cowley's
criteria). Direction index j = 30 j degrees (file order of stimuli_gratings; checked by the class-mean correlation profile).

Statistics, generalized from the imaging generators to K = 12 classes: the ladder in angular order, rungs 1,2,3,4,6,8,12
(cr_order_and_withinbin_controls.ladder: Gram trace, exact W + B split, ten-draw floor, rng 42); 200 label permutations of delta
(the floor is label-blind, so one floor serves every permutation); residual shuffle and pooled-fluctuation control
(cr_c8_identification, ten seeds); split-half sigma (cr_overlap_mining, halves alternating within each class) and the mean
field ln[12/(1+11 sigma)]/ln 12; PR(W), PR(B), x and the sign law at the full set; direction-selective fraction (tuned:
F p < 0.01; vector strength > 0.3) and cross-fitted equal thirds (>= 20 units per third); origin-averaged (12 origins x 2
traversals) paired - sequential order (6 rungs) with dK = Kn(30) - Kn(180). Frames: raw counts (the paper's) and within-SD.
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler cr_h1_m_pvc11.py [n_workers]
Out: data_canonical/cr_h1_m_pvc11.json"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw" / "pvc11" / "spikes_gratings"   # CRCNS pvc-11 (Kohn & Smith 2016, doi:10.6080/K0NC5Z4X)
OUT = HERE.parent / "data_canonical" / "cr_h1_m_pvc11.json"
K = 12
RUNGS = (1, 2, 3, 4, 6, 8, 12)
SEEDS = tuple(range(7, 17))
N_PERM = 200
WIN = (0.28, 1.28)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_array(m):
    d = loadmat(RAW / f"data_monkey{m}_gratings.mat")["data"][0, 0]
    E, snr = d["EVENTS"], d["SNR"].ravel()
    n_u, n_d, n_t = E.shape
    assert n_d == K and n_t == 200 and len(snr) == n_u, E.shape
    C = np.zeros((n_d * n_t, n_u))
    for i in range(n_u):
        for j in range(n_d):
            for t in range(n_t):
                s = np.asarray(E[i, j, t]).ravel()
                C[j * n_t + t, i] = np.count_nonzero((s >= WIN[0]) & (s < WIN[1]))
    lab = np.repeat(np.arange(n_d), n_t)
    keep = (snr >= 1.5) & (C.mean(0) / (WIN[1] - WIN[0]) >= 1.0)
    ch = d["CHANNELS"][:, 0].astype(int)[keep]; MAP = np.asarray(d["MAP"], float)
    assert MAP.shape == (10, 10) and len(d["CHANNELS"]) == n_u
    return C[:, keep], lab, {"n_units_file": int(n_u), "n_kept": int(keep.sum()), "snr_pass": int((snr >= 1.5).sum()),
                             "channels": ch, "MAP": MAP}


def sign_law(Y, lab):
    X = Y - Y.mean(0)
    cm = np.stack([X[lab == c].mean(0) for c in range(K)]); p = np.bincount(lab, minlength=K) / len(lab)
    n = len(X); Wh = (X - cm[lab]) / np.sqrt(n); Bh = np.sqrt(p)[:, None] * cm
    w1, b1 = (Wh * Wh).sum(), (Bh * Bh).sum()
    w2 = ((Wh @ Wh.T) ** 2).sum(); b2 = ((Bh @ Bh.T) ** 2).sum(); x = ((Wh @ Bh.T) ** 2).sum()
    prW, prB = w1 ** 2 / w2, b1 ** 2 / b2
    return {"trB_trW": float(b1 / w1), "PR_W": float(prW), "PR_B": float(prB), "ratio": float(prW / prB),
            "threshold": float(1 + 2 * w1 / b1), "c_full": float(np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2))),
            "c_without_x": float(np.log((1 + b1 / w1) ** 2 / (1 + b2 / w2))), "x_share": float(2 * x / (b2 + 2 * x))}


def sets_from(lab, order):
    m = [np.where(lab == b)[0] for b in range(K)]
    return [np.concatenate([m[j] for j in order[:c]]) for c in RUNGS]


def lpr(G, idx):
    Gs = G[np.ix_(idx, idx)]; r = Gs.mean(1, keepdims=True); Gc = Gs - r - r.T + Gs.mean()
    return np.log(np.trace(Gc) ** 2 / (Gc * Gc).sum())


def permutation_p(G, lab, n_perm=N_PERM, seed=0):
    """delta and its one-sided permutation p; the ten-draw floor (rng 42, as the ladder) is label-blind and shared."""
    rng = np.random.default_rng(42); sets = sets_from(lab, list(range(K)))
    x = np.log([len(s) for s in sets])
    flo = np.array([np.mean([lpr(G, rng.choice(len(G), len(s), replace=False)) for _ in range(10)]) for s in sets])
    f = lambda l: float(np.polyfit(x, np.array([lpr(G, s) for s in sets_from(l, list(range(K)))]) - flo, 1)[0])
    obs = f(lab); prng = np.random.default_rng(seed)
    null = np.array([f(lab[prng.permutation(len(lab))]) for _ in range(n_perm)])
    return {"delta": obs, "p": float((1 + (null >= obs).sum()) / (n_perm + 1)), "null_max": float(null.max()),
            "null_mean": float(null.mean()), "null_sd": float(null.std())}


def ds_select(X, lab):
    n = len(lab); counts = np.bincount(lab, minlength=K)
    M = np.stack([X[lab == k].mean(0) for k in range(K)]); g = X.mean(0)
    ssb = sum(counts[k] * (M[k] - g) ** 2 for k in range(K)); ssw = sum(((X[lab == k] - M[k]) ** 2).sum(0) for k in range(K))
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ssb / (K - 1)) / (ssw / (n - K))
    tuned = np.nan_to_num(1 - stats.f.cdf(F, K - 1, n - K), nan=1.0) < 0.01
    R = M - M.min(0, keepdims=True) + 1e-9
    dsi = np.abs((R * np.exp(1j * 2 * np.pi * np.arange(K)[:, None] / K)).sum(0) / R.sum(0))
    tidx = np.where(tuned)[0]; nsub = len(tidx) // 3; order = tidx[np.argsort(dsi[tidx])]
    frac = float((dsi[tuned] > 0.3).mean()) if tuned.any() else float("nan")
    return order[-nsub:], order[:nsub], frac, int(tuned.sum())


def halves(lab):
    h = np.zeros(len(lab), int)
    for c in range(K):
        idx = np.where(lab == c)[0]; h[idx[1::2]] = 1
    return h


def analyze(X, lab, ow, om, ci, full=True):
    G = (X @ X.T).astype(np.float64)
    r = {"N": int(X.shape[1]), "ladder": ow.ladder(G, lab, sets_from(lab, list(range(K))), np.random.default_rng(42)),
         "sign": sign_law(X, lab)}
    h = halves(lab)
    Ra = om.residuals(X[h == 0], lab[h == 0], lab[h == 0], "raw"); Rb = om.residuals(X[h == 1], lab[h == 1], lab[h == 1], "raw")
    r["sigma"] = om.sigma(om.split_half_K(Ra, Rb)); r["mean_field"] = float(np.log(K / (1 + (K - 1) * r["sigma"])) / np.log(K))
    idx = [np.where(lab == c)[0] for c in range(K)]
    tr, _, T = om.overlaps(G, idx); Kc = om.K_of(tr, tr, T); kd = np.diag(Kc); Kn = Kc / np.sqrt(np.outer(kd, kd))
    sep = np.minimum(np.abs(np.subtract.outer(np.arange(K), np.arange(K))), K - np.abs(np.subtract.outer(np.arange(K), np.arange(K))))
    r["Kn_profile"] = [float(Kn[sep == s].mean()) for s in range(K // 2 + 1)]; r["dK"] = r["Kn_profile"][1] - r["Kn_profile"][-1]
    Cm = np.corrcoef(np.stack([X[i].mean(0) for i in idx])); r["classmean_corr_profile"] = [float(Cm[sep == s].mean()) for s in range(K // 2 + 1)]
    _, _, r["ds_fraction"], r["n_tuned"] = ds_select(X, lab)
    if not full:
        return r
    r["perm"] = permutation_p(G, lab)
    r["resid_shuf"] = [ow.ladder((Z @ Z.T).astype(np.float64), lab, sets_from(lab, list(range(K))), np.random.default_rng(42))
                       for Z in (ci.shuffle_within_class(X, lab, np.random.default_rng(s)) for s in SEEDS)]
    r["resid_pool"] = [ow.ladder((Z @ Z.T).astype(np.float64), lab, sets_from(lab, list(range(K))), np.random.default_rng(42))
                       for Z in (ci.pool_residuals(X, lab, np.random.default_rng(s)) for s in SEEDS)]
    th = {}
    for sh, mh in ((0, 1), (1, 0)):
        ds, nds, _, _ = ds_select(X[h == sh], lab[h == sh])
        if len(ds) < 20:
            th = None; break
        Xm, lm = X[h == mh], lab[h == mh]
        lad = lambda cols: ow.ladder((Xm[:, cols] @ Xm[:, cols].T).astype(np.float64), lm, sets_from(lm, list(range(K))), np.random.default_rng(42))["delta"]
        th[f"sel{sh}_meas{mh}"] = {"DS": lad(ds), "nonDS": lad(nds), "n_sub": int(len(ds))}
    r["ds_thirds"] = th
    m = [np.where(lab == b)[0] for b in range(K)]; oe = []
    for o in range(K):
        for s in (1, -1):
            P = [np.concatenate([m[(o + s * k + j) % K] for k in range(q + 1) for j in (0, K // 2)]) for q in range(K // 2)]
            Q = [np.concatenate([m[(o + s * j) % K] for j in range(2 * q + 2)]) for q in range(K // 2)]
            po = ow.ladder(G, lab, P, np.random.default_rng(42)); so = ow.ladder(G, lab, Q, np.random.default_rng(43))
            oe.append({"origin": o, "traversal": s, **{k: po[k] - so[k] for k in po}})
    r["order"] = oe
    r["order_mean"] = {k: float(np.mean([e[k] for e in oe])) for k in ("delta", "within", "class_means")}
    return r


def balance(X, lab):
    """b/c of the class means' correlation profile over the 12 directions (run51.profile_ratio at K = 12): b2 the second
    harmonic (orientation), c1 the first (direction)."""
    Cm = np.corrcoef(np.stack([X[lab == c].mean(0) for c in range(K)]))
    prof = np.array([np.mean([Cm[i, (i + k) % K] for i in range(K)]) for k in range(K)])
    ang = 2 * np.pi * np.arange(K) / K
    return float((prof * np.cos(2 * ang)).sum() / (prof * np.cos(ang)).sum())


def graining(X, lab, ch, MAP, ow):
    """H1-R40 = M5: blocks of neighbouring electrodes (tiles of t x t on the 10 x 10 MAP; units on one electrode pooled by
    the tile), against random, orientation-sorted and direction-sorted blocks with the SAME block-size list (run51
    construction: each block averaged into one unit; two floor seeds averaged for delta)."""
    pos = {}
    for r in range(10):
        for c in range(10):
            if np.isfinite(MAP[r, c]) and MAP[r, c] > 0:
                pos[int(MAP[r, c])] = (r, c)
    assert all(c in pos for c in ch), "unit channel missing from MAP"
    ang = 2 * np.pi * np.arange(K) / K
    M = np.stack([X[lab == k].mean(0) for k in range(K)]); Mc = M - M.mean(0)
    ph_ori = (np.angle((Mc * np.exp(2j * ang[:, None])).sum(0)) / 2) % np.pi
    ph_dir = np.angle((Mc * np.exp(1j * ang[:, None])).sum(0)) % (2 * np.pi)
    sets = lambda l: sets_from(l, list(range(K)))
    dl = lambda Xb: float(np.mean([ow.ladder((Xb @ Xb.T).astype(np.float64), lab, sets(lab), np.random.default_rng(42 + s))["delta"] for s in (0, 1)]))
    base = {"delta": dl(X), "bc": balance(X, lab)}
    out = {"ungrained": base}
    for t in (2, 3):
        tile = np.array([(pos[c][0] // t) * 100 + pos[c][1] // t for c in ch])
        blocks = [np.where(tile == v)[0] for v in np.unique(tile)]
        sizes = sorted(len(b) for b in blocks)
        def from_order(order):
            bl, i = [], 0
            for s_ in sizes:
                bl.append(order[i:i + s_]); i += s_
            return bl
        mk = lambda bl: np.stack([X[:, b].mean(1) for b in bl], 1)
        arms = {"anatomical": [blocks], "ori_sorted": [from_order(np.argsort(ph_ori))],
                "dir_sorted": [from_order(np.argsort(ph_dir))],
                "random": [from_order(np.random.default_rng(s).permutation(X.shape[1])) for s in range(1, 11)]}
        res = {}
        for a, bls in arms.items():
            v = [(dl(mk(bl)), balance(mk(bl), lab)) for bl in bls]
            res[a] = {"delta": float(np.mean([x[0] for x in v])), "bc": float(np.mean([x[1] for x in v])),
                      "retention": float(np.mean([x[0] for x in v]) / base["delta"]), "bc_flow": float(np.mean([x[1] for x in v]) / base["bc"])}
        res["n_blocks"] = len(blocks); res["block_sizes"] = sizes
        out[f"tile{t}"] = res
    return out


def run_one(m):
    ow = _load("ow", "cr_order_and_withinbin_controls.py"); om = _load("om", "cr_overlap_mining.py")
    ci = _load("ci", "cr_c8_identification.py")
    t0 = time.time()
    X, lab, info = load_array(m)
    sd = np.sqrt(np.mean([((X[lab == c] - X[lab == c].mean(0)) ** 2).mean(0) for c in range(K)], 0))
    keep = sd > 1e-6 * np.median(sd[sd > 0]); Z = X[:, keep] / sd[keep]
    ch, MAP = info.pop("channels"), info.pop("MAP")
    out = {"monkey": m, **info, "raw": analyze(X, lab, ow, om, ci), "z": analyze(Z, lab, ow, om, ci),
           "graining": graining(X, lab, ch, MAP, ow)}
    for t in ("tile2", "tile3"):
        g = out["graining"][t]
        print(f"monkey{m} M5 {t}: retention anat {g['anatomical']['retention']:.2f} random {g['random']['retention']:.2f} ori "
              f"{g['ori_sorted']['retention']:.2f} dir {g['dir_sorted']['retention']:.2f} | b/c flow anat {g['anatomical']['bc_flow']:.2f} "
              f"random {g['random']['bc_flow']:.2f} ({g['n_blocks']} blocks)", flush=True)
    r = out["raw"]
    print(f"monkey{m} N {r['N']}: delta {r['ladder']['delta']:+.3f} (p {r['perm']['p']:.4f}) within {r['ladder']['within']:+.3f} "
          f"class-means {r['ladder']['class_means']:+.3f} | ratio {r['sign']['ratio']:.1f} thr {r['sign']['threshold']:.1f} "
          f"c_full {r['sign']['c_full']:+.3f} | sigma {r['sigma']:.2f} mf {r['mean_field']:+.3f} | dK {r['dK']:+.3f} "
          f"OE within {r['order_mean']['within']:+.3f} | z: delta {out['z']['ladder']['delta']:+.3f} cm "
          f"{out['z']['ladder']['class_means']:+.3f} ({time.time() - t0:.0f}s) DONE", flush=True)
    return out


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(run_one, (1, 2, 3)))
    v = {"M1_delta_pos_above_all_perms": int(sum(r["raw"]["ladder"]["delta"] > 0 and r["raw"]["perm"]["p"] <= 1 / (N_PERM + 1) + 1e-12 for r in res)),
         "M2_within_pos": int(sum(r["raw"]["ladder"]["within"] > 0 for r in res)),
         "M3_law_xfree_agrees": int(sum((r["raw"]["sign"]["ratio"] > r["raw"]["sign"]["threshold"]) == (r["raw"]["ladder"]["class_means"] < 0) for r in res)),
         "M3_cfull_agrees": int(sum((r["raw"]["sign"]["c_full"] < 0) == (r["raw"]["ladder"]["class_means"] < 0) for r in res)),
         "M4_order_sign": int(sum(np.sign(r["raw"]["order_mean"]["within"]) == -np.sign(r["raw"]["dK"]) for r in res)),
         "M5_anat_retention_gt_random_tile3": int(sum(r["graining"]["tile3"]["anatomical"]["retention"] > r["graining"]["tile3"]["random"]["retention"] for r in res)),
         "M5_anat_bcflow_gt_random_tile3": int(sum(r["graining"]["tile3"]["anatomical"]["bc_flow"] > r["graining"]["tile3"]["random"]["bc_flow"] for r in res)),
         "M5_anat_retention_gt_random_tile2": int(sum(r["graining"]["tile2"]["anatomical"]["retention"] > r["graining"]["tile2"]["random"]["retention"] for r in res)),
         "M5_anat_bcflow_gt_random_tile2": int(sum(r["graining"]["tile2"]["anatomical"]["bc_flow"] > r["graining"]["tile2"]["random"]["bc_flow"] for r in res))}
    print("VERDICT", v, flush=True)
    json.dump({"window_s": WIN, "rungs": RUNGS, "arrays": res, "verdict": v}, open(OUT, "w"), default=float)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
