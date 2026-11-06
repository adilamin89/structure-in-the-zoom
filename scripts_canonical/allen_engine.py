"""The imaging tests of the camera-ready, rerun on Allen Neuropixels with temporal frequency separated.
Input: the per-session count files of allen_counts_modal.py
(raw/allen_counts/session_<sid>.npz: every unit, every drifting-grating presentation with a finite
orientation, its temporal frequency, the unit quality metrics).

Every statistic reuses the generator that produced the imaging number, so the two datasets are read by one estimator:
  ladder, within-class / class-mean split   cr_order_and_withinbin_controls.ladder (Gram trace, exact W + B split, ten-draw
                                            floor, rng 42; direction rungs 1,2,3,4,6,8 in ascending direction order)
  residual shuffle, pooled-fluctuation      withinclass_shuffle_identification.shuffle_within_class / pool_residuals (seeds 7..16)
  sigma (split halves), overlap K           overlap_identity_checks.residuals / split_half_K / overlaps (halves alternate within
                                            each sub-bin; here the sub-bin is the direction x TF cell)
  sign law                                  classmean_sign_condition (b1, w1, b2, w2, x at the full stimulus set)
  DS thirds, cross-fitted                   cr_ds_thirds_crossfit.select (F test p < 0.01, first-harmonic vector strength)
  order effect                              cr_order_effect_origins (paired - sequential, each arm its own floor, rng 42/43)
  random-stimulus ladder                    withinclass_shuffle_identification.part_b, matched aggregation (mean log PR in both arms)

Arms (the input X of every statistic):
  pooled    raw counts; the paper's estimator (each direction class pools the five TFs).
  tfres     TF-residualized: each presentation's residual from the leave-one-out mean of its direction x TF cell, plus its
            direction-class mean (cr_order_and_withinbin_controls.remove_withinbin with the cell as sub-bin): class means
            kept exactly, TF tuning removed from W.
  tfstrat   the eight-class ladder inside each TF (about 15 presentations per class), averaged over the five TFs.
  tfres_z   diagnostic: tfres with each unit scaled by its pooled within-class SD (the correlation frame of W).
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
COUNTS = HERE.parent / "raw" / "allen_counts"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


ow = _load("ow", "cr_order_and_withinbin_controls.py")
om = _load("om", "overlap_identity_checks.py")
ci = _load("ci", "withinclass_shuffle_identification.py")
dsx = _load("dsx", "cr_ds_thirds_crossfit.py")
RUNGS = (1, 2, 3, 4, 6, 8)
SEEDS = tuple(range(7, 17))   # ten seeds: at ~75 trials per class one pooled-control draw moves delta by ~0.1
MIN_UNITS = 50


# ------------------------------------------------------------------------------------------------------------ input
def load_session(sid):
    d = np.load(COUNTS / f"session_{sid}.npz")
    ori, tf = d["orientation"], d["temporal_frequency"]
    dirs, tfs = np.sort(np.unique(ori)), np.sort(np.unique(tf))
    assert len(dirs) == 8 and len(tfs) == 5, (dirs, tfs)
    lab = np.searchsorted(dirs, ori); tfl = np.searchsorted(tfs, tf)
    q = (d["isi_violations"] < 0.5) & (d["amplitude_cutoff"] < 0.1) & (d["presence_ratio"] > 0.9)
    return {"sid": sid, "X": d["counts"].astype(np.float64), "lab": lab, "tf": tfl, "cell": lab * 5 + tfl,
            "area": d["area"], "quality": q, "dirs": dirs, "tfs": tfs}


def populations(S, quality=False, min_units=MIN_UNITS):
    """{area: unit columns}, every VIS area with >= min_units units (allen_expansion / run66 rule), NWB unit order."""
    out = {}
    for a in sorted(set(S["area"])):
        if not a.startswith("VIS"):
            continue
        idx = np.where((S["area"] == a) & (S["quality"] if quality else True))[0]
        if len(idx) >= min_units:
            out[a] = idx
    return out


# ------------------------------------------------------------------------------------------------------------ masking path
VARIANT = os.environ.get("ALLEN_VARIANT", "B")   # none | A | B; B = the paper (Allen invalid_times + invalid electrodes + silence rule)
MASKS = HERE.parent / "data_canonical" / "allen_masks"
_MCACHE = {}


def pop_rows_cols(S, area, idx):
    """One population's valid presentation rows and kept unit columns for the ALLEN_VARIANT setting.
    'none': all rows, idx. 'A': Allen's invalid_times (the probe of any of the population's units, all_probes, stimulus) and
    units on valid_data False electrodes dropped. 'B': A + the silence rule. Masks from allen_invalid_map.py.
    idx may be a subset of the area's units (the quality arm): the kept columns are idx's units in units_keep, idx order."""
    idx = np.asarray(idx)
    if VARIANT == "none":
        return np.arange(len(S["lab"])), idx
    if S["sid"] not in _MCACHE:
        _MCACHE[S["sid"]] = dict(np.load(MASKS / f"session_{S['sid']}.npz"))
    m = _MCACHE[S["sid"]]
    ukeep, valid = m[f"{area}__units_keep"], m[f"{area}__valid{VARIANT}"]
    assert len(valid) == len(S["lab"]) and (S["area"][ukeep] == area).all() and (S["area"][idx] == area).all(), (S["sid"], area)
    return np.where(valid)[0], idx[np.isin(idx, ukeep)]


def pop_view(S, area, idx, extra=None):
    """One population through the masking path: X (presentations x units), lab, tf, cell, and every per-presentation array in
    extra (e.g. running speed) taken through the same rows."""
    rows, cols = pop_rows_cols(S, area, idx)
    v = {"X": S["X"][np.ix_(rows, cols)], "lab": S["lab"][rows], "tf": S["tf"][rows], "cell": S["cell"][rows],
         "rows": rows, "cols": cols}
    for k, a in (extra or {}).items():
        v[k] = np.asarray(a)[rows]
    return v


def reproduce_stored(S):
    """allen_expansion.py's delta_dir8 with its exact rng sequence (one rng 42 per session, consumed by every area's
    direction ladder, orientation ladder and random-stimulus ladder in sorted-area order). Returns {area: delta}.
    Each population through pop_view (its own valid rows and kept units)."""
    rng = np.random.default_rng(42); out = {}

    def pr(Y):
        Yc = Y - Y.mean(0); G = Yc @ Yc.T
        return np.trace(G) ** 2 / (G * G).sum()

    def sl(s, p):
        return np.polyfit(np.log(s), np.log(p), 1)[0]
    for a, idx in populations(S).items():
        P = pop_view(S, a, idx)
        Xa, ori = P["X"], S["dirs"][P["lab"]]
        mem = [np.where(ori == c)[0] for c in S["dirs"]]
        sizes = [len(np.concatenate(mem[:c])) for c in RUNGS]
        prs = [pr(Xa[np.concatenate(mem[:c])]) for c in RUNGS]
        nl = np.array([[np.log(pr(Xa[rng.choice(len(Xa), s, replace=False)])) for s in sizes] for _ in range(10)])
        out[a] = float(sl(sizes, prs) - sl(sizes, np.exp(nl.mean(0))))
        om_ = ori % 180; mo = [np.where(om_ == c)[0] for c in np.sort(np.unique(om_))]
        for _ in range(10):
            for s in [len(np.concatenate(mo[:c])) for c in (1, 2, 3, 4)]:
                rng.choice(len(Xa), s, replace=False)
        rs = [50, 100, 200, min(400, len(Xa)), len(Xa)] if len(Xa) >= 200 else [r for r in (50, 100) if r < len(Xa)] + [len(Xa)]  # addendum
        for k in (3, 10):
            for s in rs:
                for _ in range(k):
                    rng.choice(len(Xa), s, replace=False)
    return out


# ------------------------------------------------------------------------------------------------------------ arms
def tf_residualize(X, lab, cell):
    """Residual from the leave-one-out mean of the direction x TF cell, plus the direction-class mean (exactly kept)."""
    Y = X.copy()
    for b in np.unique(cell):
        m = cell == b; nb = m.sum()
        if nb < 2:
            continue
        S = X[m].sum(0); Y[m] = X[m] - (S - X[m]) / (nb - 1)
    for c in np.unique(lab):
        m = lab == c; Y[m] += X[m].mean(0)
    return Y


def class_sets(lab, order):
    m = [np.where(lab == b)[0] for b in range(8)]
    return [np.concatenate([m[j] for j in order[:c]]) for c in RUNGS]


def ladder(Y, lab, seed=42, order=tuple(range(8))):
    return ow.ladder((Y @ Y.T).astype(np.float64), lab, class_sets(lab, list(order)), np.random.default_rng(seed))


def ladder_delta_only(Y, lab, seed=42):
    """Total delta only (no within split), for permutations: centred-Gram PR, ten-draw floor, rng seed."""
    G = (Y @ Y.T).astype(np.float64); sets = class_sets(lab, list(range(8))); rng = np.random.default_rng(seed)

    def lpr(idx):
        Gs = G[np.ix_(idx, idx)]; r = Gs.mean(1, keepdims=True); Gc = Gs - r - r.T + Gs.mean()
        return np.log(np.trace(Gc) ** 2 / (Gc * Gc).sum())
    x = np.log([len(s) for s in sets]); obs = [lpr(s) for s in sets]
    flo = [np.mean([lpr(rng.choice(len(G), len(s), replace=False)) for _ in range(10)]) for s in sets]
    return float(np.polyfit(x, np.array(obs) - np.array(flo), 1)[0])


# ------------------------------------------------------------------------------------------------------------ statistics
def sign_law(Y, lab):
    """classmean_sign_condition at the full stimulus set."""
    X = Y - Y.mean(0); K = lab.max() + 1
    cm = np.stack([X[lab == c].mean(0) for c in range(K)]); p = np.bincount(lab, minlength=K) / len(lab)
    n = len(X); Wh = (X - cm[lab]) / np.sqrt(n); Bh = np.sqrt(p)[:, None] * cm
    w1, b1 = (Wh * Wh).sum(), (Bh * Bh).sum()
    w2 = ((Wh @ Wh.T) ** 2).sum(); b2 = ((Bh @ Bh.T) ** 2).sum(); x = ((Wh @ Bh.T) ** 2).sum()
    prW, prB = w1 ** 2 / w2, b1 ** 2 / b2
    return {"trB_trW": float(b1 / w1), "PR_W": float(prW), "PR_B": float(prB), "ratio": float(prW / prB),
            "threshold": float(1 + 2 * w1 / b1), "c_full": float(np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2))),
            "c_without_x": float(np.log((1 + b1 / w1) ** 2 / (1 + b2 / w2))), "x_share": float(2 * x / (b2 + 2 * x))}


def halves(cell, parity=0):
    h = np.zeros(len(cell), int)
    for b in np.unique(cell):
        idx = np.where(cell == b)[0]; h[idx[1 - parity::2]] = 1
    return h


def sigma_split(X, lab, cell, mode):
    """Split-half sigma (overlap_identity_checks part a): halves alternate within each cell, residuals built inside each half.
    mode 'raw' (x - class mean; the pooled arm, Fig 3A) or 'noise' (x - leave-one-out cell mean; the tfres arm)."""
    h = halves(cell)
    Ra = om.residuals(X[h == 0], lab[h == 0], cell[h == 0], mode)
    Rb = om.residuals(X[h == 1], lab[h == 1], cell[h == 1], mode)
    K = om.split_half_K(Ra, Rb)
    return om.sigma(K), K


def mean_field(sig):
    return float(np.log(8 / (1 + 7 * sig)) / np.log(8))


def overlap_profile(Y, lab):
    """Shape overlap Kn = K_cd / sqrt(K_cc K_dd) by class separation (fluctuation_overlap_profile, full sample, 8 classes);
    dK = Kn(45) - Kn(180)."""
    G = (Y @ Y.T).astype(np.float64); idx = [np.where(lab == c)[0] for c in range(8)]
    tr, _, T = om.overlaps(G, idx); K = om.K_of(tr, tr, T); kd = np.diag(K)
    sep = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
    Kn = K / np.sqrt(np.outer(kd, kd))
    prof = [float(Kn[sep == s].mean()) for s in range(5)]
    return prof, prof[1] - prof[4]


def order_effect(Y, lab):
    """Paired (each rung adds a class and its antipode) minus sequential (adjacent pairs), 4 rungs of 2..8 classes,
    8 origins x 2 traversals; each arm its own ten-draw floor (rng 42 paired, 43 sequential, as imaging)."""
    G = (Y @ Y.T).astype(np.float64); m = [np.where(lab == b)[0] for b in range(8)]
    res = []
    for o in range(8):
        for s in (1, -1):
            P = [np.concatenate([m[(o + s * k + j) % 8] for k in range(r + 1) for j in (0, 4)]) for r in range(4)]
            Q = [np.concatenate([m[(o + s * j) % 8] for j in range(2 * r + 2)]) for r in range(4)]
            po = ow.ladder(G, lab, P, np.random.default_rng(42)); so = ow.ladder(G, lab, Q, np.random.default_rng(43))
            res.append({"origin": o, "traversal": s, **{k: po[k] - so[k] for k in po}})
    return res


def random_axis(Y):
    """Random-stimulus ladder with matched aggregation (mean log PR of five draws vs ten single-draw floor ladders)."""
    G = (Y @ Y.T).astype(np.float64); n = len(G)
    ns = [n] + [s for s in (400, 200, 100, 50) if s < n]
    full = np.log(ci.pr_sub(G, np.arange(n)))
    obs = [full] + [np.mean([np.log(ci.pr_sub(G, np.random.default_rng(42 + k).choice(n, s, replace=False))) for k in range(5)])
                    for s in ns[1:]]
    fl = np.mean([ci.slope(ns, [np.log(ci.pr_sub(G, np.random.default_rng(1000 + i).choice(n, s, replace=False))) for s in ns])
                  for i in range(10)])
    return float(ci.slope(ns, obs) - fl)


def tf_share(X, lab, cell):
    """Share of the within-direction variance carried by TF tuning: raw (cell means of all trials) and reliable
    (cross-half products of cell-mean deviations from each half's class mean)."""
    trW = sum(((X[lab == c] - X[lab == c].mean(0)) ** 2).sum() for c in range(8))
    raw = rel = 0.0
    h = halves(cell)
    for c in range(8):
        mc = lab == c; mu = X[mc].mean(0)
        muA, muB = X[mc & (h == 0)].mean(0), X[mc & (h == 1)].mean(0)
        for b in np.unique(cell[mc]):
            mb = cell == b
            raw += mb.sum() * ((X[mb].mean(0) - mu) ** 2).sum()
            rel += mb.sum() * ((X[mb & (h == 0)].mean(0) - muA) @ (X[mb & (h == 1)].mean(0) - muB))
    return float(raw / trW), float(rel / trW)


def ds_fraction(X, lab):
    """Share of tuned units (F test over 8 directions, p < 0.01) with first-harmonic vector strength > 0.3 (App: the
    direction-selective fraction), and the tuned count."""
    n = len(lab); counts = np.bincount(lab, minlength=8)
    M = np.stack([X[lab == k].mean(0) for k in range(8)]); g = X.mean(0)
    ssb = sum(counts[k] * (M[k] - g) ** 2 for k in range(8)); ssw = sum(((X[lab == k] - M[k]) ** 2).sum(0) for k in range(8))
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ssb / 7) / (ssw / (n - 8))
    tuned = np.nan_to_num(1 - stats.f.cdf(F, 7, n - 8), nan=1.0) < 0.01
    R = M - M.min(0, keepdims=True) + 1e-9
    dsi = np.abs((R * np.exp(1j * 2 * np.pi * np.arange(8)[:, None] / 8)).sum(0) / R.sum(0))
    return float((dsi[tuned] > 0.3).mean()) if tuned.any() else float("nan"), int(tuned.sum())


def ds_thirds(X, lab, cell, arm):
    """cr_ds_thirds_crossfit on Allen: select on one half (raw counts), measure the ladder on the other half's arm input;
    both ways. Returns None when a third has < 20 units."""
    h = halves(cell); out = {}
    for sh, mh in ((0, 1), (1, 0)):
        ds, nds = dsx.select(X[h == sh], lab[h == sh])
        if len(ds) < 20:
            return None
        Xm, lm, cm = X[h == mh], lab[h == mh], cell[h == mh]
        Ym = tf_residualize(Xm, lm, cm) if arm == "tfres" else Xm
        out[f"sel{sh}_meas{mh}"] = {"DS": ladder_delta_only(np.ascontiguousarray(Ym[:, ds]), lm),
                                    "nonDS": ladder_delta_only(np.ascontiguousarray(Ym[:, nds]), lm), "n_sub": int(len(ds))}
    return out


def sigma_heldout(X, lab, cell):
    """Held-out sigma law: sigma from one half (its own split halves, noise residuals), the within-class part of the
    TF-residualized ladder on the other half; both ways."""
    h = halves(cell); out = []
    for sh, mh in ((0, 1), (1, 0)):
        Xs, ls, cs = X[h == sh], lab[h == sh], cell[h == sh]
        sig, _ = sigma_split(Xs, ls, cs, "noise")
        Xm, lm, cm = X[h == mh], lab[h == mh], cell[h == mh]
        w = ladder(tf_residualize(Xm, lm, cm), lm)["within"]
        out.append({"sigma": sig, "pred": mean_field(sig), "within": w})
    return out


def permutations(X, lab, tfl, cell, arm, n_perm=200, seed=0):
    """One-sided label-permutation p of delta: pooled = labels permuted at fixed class sizes (the paper's test);
    tfres = direction labels permuted within each TF, the whole pipeline (residualization, ladder, floor) rerun."""
    rng = np.random.default_rng(seed)
    f = (lambda l: ladder_delta_only(tf_residualize(X, l, l * 5 + tfl), l)) if arm == "tfres" else (lambda l: ladder_delta_only(X, l))
    obs = f(lab); null = []
    for _ in range(n_perm):
        if arm == "tfres":
            l = lab.copy()
            for t in np.unique(tfl):
                m = np.where(tfl == t)[0]; l[m] = lab[m][rng.permutation(len(m))]
        else:
            l = lab[rng.permutation(len(lab))]
        null.append(f(l))
    null = np.array(null)
    return {"delta": obs, "p": float((1 + (null >= obs).sum()) / (n_perm + 1)), "null_mean": float(null.mean()),
            "null_sd": float(null.std())}


def analyze_population(X, lab, tfl, full=True, n_perm=200):
    """Every Allen statistic on one population (presentations x units)."""
    cell = lab * 5 + tfl
    Y = tf_residualize(X, lab, cell)
    assert np.allclose(np.stack([Y[lab == c].mean(0) for c in range(8)]), np.stack([X[lab == c].mean(0) for c in range(8)]))
    r = {"N": int(X.shape[1]), "n_pres": int(len(lab))}
    r["pooled"] = ladder(X, lab); r["tfres"] = ladder(Y, lab)
    strat = [ladder(X[tfl == t], lab[tfl == t]) for t in range(5)]
    r["tfstrat"] = {k: float(np.mean([s[k] for s in strat])) for k in strat[0]}; r["tfstrat_per_tf"] = strat
    r["tf_share_raw"], r["tf_share_reliable"] = tf_share(X, lab, cell)
    r["sign_pooled"], r["sign_tfres"] = sign_law(X, lab), sign_law(Y, lab)
    r["sigma_raw"], _ = sigma_split(X, lab, cell, "raw"); r["sigma_noise"], _ = sigma_split(X, lab, cell, "noise")
    r["ds_fraction"], r["n_tuned"] = ds_fraction(X, lab)
    sd = np.sqrt(np.mean([((Y[lab == c] - Y[lab == c].mean(0)) ** 2).mean(0) for c in range(8)], 0))
    keep = sd > 1e-6 * np.median(sd[sd > 0])
    Z = Y[:, keep] / sd[keep]                          # diagnostic frame: each unit scaled by its within-class SD
    r["tfres_z"] = ladder(Z, lab); r["sign_tfres_z"] = sign_law(Z, lab); r["n_units_z"] = int(keep.sum())
    if not full:
        return r
    r["resid_shuf"] = [ladder(ci.shuffle_within_class(Y, lab, np.random.default_rng(s)), lab) for s in SEEDS]
    r["resid_pool"] = [ladder(ci.pool_residuals(Y, lab, np.random.default_rng(s)), lab) for s in SEEDS]
    r["sigma_heldout"] = sigma_heldout(X, lab, cell)
    r["overlap_profile_tfres"], r["dK_tfres"] = overlap_profile(Y, lab)
    r["overlap_profile_pooled"], r["dK_pooled"] = overlap_profile(X, lab)
    r["order_tfres"] = order_effect(Y, lab)
    r["random_axis_pooled"], r["random_axis_tfres"] = random_axis(X), random_axis(Y)
    r["ds_thirds_tfres"] = ds_thirds(X, lab, cell, "tfres")
    r["ds_thirds_pooled"] = ds_thirds(X, lab, cell, "pooled")
    if n_perm:
        r["perm_tfres"] = permutations(X, lab, tfl, cell, "tfres", n_perm)
        r["perm_pooled"] = permutations(X, lab, tfl, cell, "pooled", n_perm)
    return r
