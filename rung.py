"""rung: the axis-resolved dimensionality-scaling instrument, reusable.

A rung is one step of the subsampling ladder; the tool reads every rung
against its matched floor. (Released as theta-zoom through 1.1.0; that
name stays importable as an alias for one release. 1.3 added the per-rung
deficit, the late fraction and the antipodal-pair check; 1.4 the exact four-term
split of the deficit; 1.5 the sector fit of the class-mean kernel, the
declared accumulation order, and the circular-shift null for labels in
time order; 1.6 the finite-sample bias null and the sign-law readout of the
split, the time-blocked null, pair-preserving order averaging for paired
axes, and, for language models, the last-token report and both feature
frames; 1.6.1 a per-sample validity mask, e.g. a data release's invalid
intervals, applied before every statistic; 1.7 the label-blind floor of the
principal-angle dissimilarity between classes, predicted from the pooled
spectrum and the samples' gain kurtosis; 1.7.1 correctness fixes: every
accumulation order is read against a floor at its own rung sizes, undefined
statistics get no p-value, the split is computed in float64 on centred data,
and degenerate designs raise instead of returning a number; 1.8 the
paper's two-term account, the candidate screen, the feature ladder, blocks
and attention heads; 1.8.1 readings that call a permutation p unresolved
when the permutation budget cannot give p < 0.05; 1.9 `rung report`, one
HTML page per result, and head scores whose class-union ladders come from
class-block sums of the Gram matrix, exact and about 14 times faster.)

Single-file, numpy-only implementation of the paper's estimator:

    theta_obs = theta_floor + delta

with the label-blind floor, the 500-permutation per-layer null, the
order-averaged statistic for unordered class axes, and the
nuisance-preserving stratified null. Conventions are byte-compatible with
the paper's runs (see scripts/run37_inferential_nulls.py).

Quick start on your own data (any modality; rows = samples, cols = features):

    import numpy as np
    from rung import zoom

    X = ...        # (n_samples, n_features), e.g. trials x neurons or
                   # prompts x hidden units at one layer
    labels = ...   # (n_samples,) integer class labels for the declared axis

    res = zoom(X, labels, n_perm=500, k_orders=50)
    print(res["delta"], res["p_two"])          # declared-order shift + null
    print(res["delta_orderavg"], res["p_two_orderavg"])  # order-free shift

    # nuisance-preserving second null (e.g. carriers, topics, sessions):
    res = zoom(X, labels, strata=my_nuisance_ids, n_perm=500)
    print(res["delta"], res["strat_p_two"])    # label reading beyond nuisance

For a language model, collect last-token hidden states at a layer for a
prompt battery (one forward pass per prompt) and call zoom() per layer, or
use the CLI with the paper's battery shipped in axes/ (7 axes, 8 classes x 16
prompts): `rung llm --model M --axis axes/language_type.json`, then
`rung plot battery.json --out profile.png`.
The paper's full batteries and artifacts live in scripts_canonical/ and
data_canonical/ of this repository.
"""
import numpy as np

__version__ = "1.9.0"

BIN_COUNTS_DEFAULT = (1, 2, 3, 4, 6, 8)


def _subset_pr(K, idx):
    """PR of the row-centered subset via the precomputed full Gram matrix."""
    Ks = K[np.ix_(idx, idx)]
    rm = Ks.mean(axis=1)
    tm = rm.mean()
    Kc = Ks - rm[:, None] - rm[None, :] + tm
    tr = float(np.trace(Kc))
    tr2 = float((Kc * Kc).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def _slope(sizes, prs):
    x = np.log(np.asarray(sizes, float))
    y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def _ladder(K, members, order, bin_counts):
    """Accumulate classes in `order`; return (slope, rung sizes, rung PRs)."""
    sizes, prs = [], []
    for c in bin_counts:
        sel = np.concatenate([members[o] for o in order[:c]])
        sizes.append(len(sel))
        prs.append(_subset_pr(K, sel))
    return _slope(sizes, prs), sizes, prs


def _class_blocks(K, lab, nc):
    """Class-block sums of a Gram matrix K (n x n) for labels lab in 0..nc-1: the diagonal sums d_c, S = P^T K P,
    Q = P^T (K o K) P, G[c,d,e] = sum over rows i of class c of (K P)[i,d] (K P)[i,e], and the class sizes. Every rung of a
    class ladder is a union of whole classes, and its row-centred PR follows from these sums alone (`_union_ladder_slopes`).
    K should come from globally centred features: the PR is the same, the sums lose less to cancellation."""
    P = np.zeros((len(lab), nc))
    P[np.arange(len(lab)), lab] = 1.0
    R = K @ P
    return P.T @ np.diag(K), P.T @ K @ P, P.T @ (K * K) @ P, np.einsum("ic,id,ie->cde", P, R, R), P.sum(axis=0)


def _union_ladder_slopes(blocks, orders, bin_counts):
    """The ladder slope of every order in `orders` (rung k = the first bin_counts[k] classes), from class-block sums: for
    the union indicator u with m rows, Tr Kc = u.d - u'Su/m and ||Kc||_F^2 = u'Qu - 2 sum_c u_c u'G_c u/m + (u'Su)^2/m^2,
    exactly the row-centred PR that `_ladder` computes row by row, at O(C^3) per rung instead of O(n^2)."""
    d, S, Q, G, m_c = blocks
    orders = np.asarray(orders)
    U = np.zeros((len(orders), len(bin_counts), len(m_c)))
    for k, c in enumerate(bin_counts):
        np.put_along_axis(U[:, k], orders[:, :c], 1.0, axis=1)
    m = U @ m_c
    sall = np.einsum("...c,cd,...d->...", U, S, U)
    trc = U @ d - sall / m
    fro = np.einsum("...c,cd,...d->...", U, Q, U) - 2.0 * np.einsum("...c,cde,...d,...e->...", U, G, U, U) / m + sall ** 2 / m ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(np.isfinite(fro) & (fro > 0), trc * trc / fro, 1.0)
    x = np.log(m)
    y = np.log(np.maximum(pr, 1e-9))
    xc = x - x.mean(axis=1, keepdims=True)
    return (xc * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / (xc * xc).sum(axis=1)


def _zscore(X):
    """Per-feature z-score in float64; constant features (SD at most 1e-12 of the largest) become zero columns."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = Xc.std(axis=0, keepdims=True)
    keep = s > 1e-12 * max(float(s.max()), 1e-300)
    return np.where(keep, Xc / np.where(keep, s, 1.0), 0.0)


def _null_stats(obs, null):
    """Two-sided permutation reading of `obs` against `null`: (z, p, null mean, null SD). The p-value counts null
    deviations from the null mean at least as large as the observed one, within a floating-point tolerance (a null
    whose every draw equals the observed value gives p = 1), with the (1 + count) / (B + 1) convention. An undefined
    observed value or null (NaN, inf, no draws) returns NaN for every field: never a p-value."""
    null = np.asarray(null, dtype=np.float64)
    if null.size == 0 or not np.isfinite(obs) or not np.all(np.isfinite(null)):
        nan = float("nan")
        return nan, nan, nan, nan
    nm, ns = float(null.mean()), float(null.std())
    tol = 64 * np.finfo(float).eps * (1.0 + abs(obs) + abs(nm))
    dev = abs(obs - nm)
    z = (obs - nm) / ns if ns > tol else 0.0
    p = float((1 + (np.abs(null - nm) >= dev - tol).sum()) / (len(null) + 1))
    return float(z), p, nm, ns


def spectrum(X):
    """Blind-probe diagnostics for a samples-by-features array:
    full-set effective dimension, leading-eigenvalue variance fraction, and
    the variance fraction of the single largest feature. An effective
    dimension near 1 with a leading fraction near 1 means a rogue dimension
    dominates and the linear participation ratio is blind to any declared
    axis; standardize=True in zoom() is the repair."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    v = Xc.var(axis=0)
    tot = lam.sum()
    return {"d_eff_full": float(tot ** 2 / (lam ** 2).sum()) if tot > 0 else float("nan"),
            "top1_eig_frac": float(lam[0] / tot) if tot > 0 else float("nan"),
            "top1_dim_var_frac": float(v.max() / v.sum()) if v.sum() > 0 else float("nan")}


def zoom(X, labels, n_perm=500, k_orders=50, k_null_orders=20,
         n_floor_draws=20, bin_counts=None, strata=None, seed=0,
         floor_seed=None, perm_seed=None, order_seed=None, standardize=False,
         antipode=None, order=None, shift_null=False, time_blocks=None, keep_null=False, floor_by_rung=False):
    """Axis-resolved decomposition of the dimensionality-scaling exponent.

    Parameters
    ----------
    X : (n, d) array, samples by features.
    labels : (n,) int array, class labels of the declared axis. Classes are
        accumulated in label order (0, 1, 2, ...); for unordered axes the
        order-averaged outputs are the partition-level statistics.
    n_perm : label permutations for the per-layer null (0 disables).
    k_orders : random accumulation orders for the order-averaged shift.
    k_null_orders : orders averaged per permuted labeling (conservative if
        smaller than k_orders; Appendix I reads each permuted labelling over
        its own random orders).
    n_floor_draws : random same-size subset draws defining the floor.
    bin_counts : ladder rung class counts; defaults to (1,2,3,4,6,8)
        truncated to the number of classes.
    standardize : z-score every feature before the Gram (default False). Use
        it when the spectrum diagnostics show one dimension dominating the
        variance (effective dimension near 1); see spectrum().
    strata : optional (n,) array of nuisance-stratum ids. When given, a
        second null permutes labels only within strata (nuisance-preserving);
        significance against it supports a label reading beyond composition.
    seed : rng seed (single stream) unless the three explicit seeds are
        given. floor_seed / perm_seed / order_seed reproduce the paper's
        convention (scripts/run37): floor rng = 100*layer+1, permutations
        rng 3700, orders rng 3701, each an independent stream.

    order : the declared accumulation order: None (the classes in sorted
        label order), "sequential" (the same), "antipodal" (one member of each
        antipodal pair first: 0, K/2, 1, K/2+1, ...; the paired order of Appendix E
        contrast), or an explicit list of class labels. The order-averaged
        outputs do not depend on it.
    shift_null : labels are in time order (one per frame) and a second null
        rolls the whole label sequence by a random offset, which keeps the
        labels' autocorrelation and breaks only their alignment with the
        frames (state axes); reported as
        shift_p_two / shift_z.
    time_blocks : optional (n,) array of acquisition-block ids (rows in time
        order cut into contiguous epochs, e.g. block_ids(n, 48)). A further
        null permutes labels only within blocks, which keeps each block's
        class counts: a shift that drift or slow state could produce reads as
        ordinary under it. Reported as tb_p_two / tb_z (declared order) and
        tb_p_two_orderavg. zoom always reports the consecutive same-class rate
        of the rows against its label-blind expectation (meaningful when rows
        are in acquisition order) and warns when labels cluster in time.
    keep_null : also return the permutation draws (null_draws, strat_null_draws).
    floor_by_rung : draw the floor rung by rung (every draw of the first rung, then the next), the order of the
        paper's generators; with their floor seed and draw count it replays their floor exactly. Default: draw by draw.
    antipode : optional dict {class: its antipodal class}. With antipodal pairs,
        the share of the climb left after half the circle (Appendices E and K) reads `late_fraction` (the deficit at half the
        classes over the deficit at one class) as the share of the climb the
        second half of the classes carries; that reading needs the first half
        of the declared order to hold one member of each pair, which this
        checks and reports as `antipode_first_half_has_no_pair`.

    Returns dict with delta, theta_obs, theta_floor, p_two, z,
    delta_orderavg (+sd, p_two_orderavg, z_orderavg), the per-rung ladders
    (pr_obs, pr_floor, deficit) with late_fraction, and, when strata is
    given, strat_p_two / strat_z. When the strata pair classes (two classes
    share stratum ids, as the members of a minimal pair or contrast do), the
    pair-preserving order average is added: the shift averaged over every
    order that keeps partners adjacent (delta_pairavg) against labels
    permuted within strata over the same orders (pairavg_p_two); a full
    order average dilutes a pairwise contrast, this one does not.
    """
    X = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("zoom: X holds NaN or inf; drop or mask those samples first (--valid)")
    if n_floor_draws < 1:
        raise ValueError("zoom: n_floor_draws must be at least 1")
    if standardize:
        # per-feature z-score: the repair for populations dominated by a rogue
        # dimension; off by default so --paper-seeds
        # still reproduces the published cells
        X = _zscore(X)
    # centre every feature on the full population before the Gram: subset centring is exact in algebra but loses
    # the covariance to cancellation when features carry offsets large against their spread
    X = X - X.mean(axis=0, keepdims=True)
    labels = np.asarray(labels)
    if not float((X * X).sum()) > 0:
        # every row identical (e.g. a layer read at a token every prompt shares): PR is 0/0 on every rung, no reading
        nan = float("nan")
        return {"degenerate": "no variance", "delta": nan, "theta_obs": nan, "theta_floor": nan, "theta_floor_se": nan,
                "delta_orderavg": nan, "delta_orderavg_sd": nan, "p_two": nan, "z": nan, "p_two_orderavg": nan, "z_orderavg": nan,
                "n_classes": int(len(np.unique(labels))), "rung_sizes": [], "n_perm": int(n_perm), "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]), "n_floor_draws": int(n_floor_draws)}
    X = X / (X.std() + 1e-9)
    classes = np.unique(labels)
    n_classes = len(classes)
    lab = np.searchsorted(classes, labels)
    n = len(lab)
    if bin_counts is None:
        bc = tuple(c for c in BIN_COUNTS_DEFAULT if c <= n_classes)
        if bc[-1] != n_classes:
            bc = bc + (n_classes,)
    else:
        bc = tuple(bin_counts)
    if len(bc) < 3:
        raise ValueError("need >=3 ladder rungs; use >=3 classes")

    if floor_seed is None and perm_seed is None and order_seed is None:
        rng = np.random.default_rng(seed)
        frng = prng = orng = rng
    else:
        frng = np.random.default_rng(floor_seed if floor_seed is not None else seed)
        prng = np.random.default_rng(perm_seed if perm_seed is not None else seed)
        orng = np.random.default_rng(order_seed if order_seed is not None else seed)
    n_samples = X.shape[0]
    if n_samples > 5000:
        import warnings
        warnings.warn(f"zoom: {n_samples} samples -> full Gram of {8*n_samples**2/1e9:.1f} GB in float64 and "
                      f"time growing as n^2 (about {(n_samples/6400)**2*3.4:.0f} min per 100 permutations on one core); "
                      "subsample stimuli per class (rung sizes stay matched) or run per-layer jobs in parallel.", stacklevel=2)
    K = X @ X.T
    members = [np.where(lab == c)[0] for c in range(n_classes)]
    canonical = declared_order(classes, order)
    run = same_class_rate(lab)

    th_obs, sizes, pr_obs = _ladder(K, members, canonical, bc)
    logs = np.zeros((n_floor_draws, len(sizes)))
    cells = ([(i, k) for k in range(len(sizes)) for i in range(n_floor_draws)] if floor_by_rung
             else [(i, k) for i in range(n_floor_draws) for k in range(len(sizes))])
    for i, k in cells:
        logs[i, k] = np.log(max(_subset_pr(K, frng.choice(n, sizes[k], replace=False)), 1e-9))
    th_floor = _slope(sizes, np.exp(logs.mean(axis=0)))
    # the slope is linear in its response, so the floor's slope is the mean of the single-draw slopes and its standard
    # error is their SD over sqrt(draws): the uncertainty a reported shift inherits from the floor
    draw_slopes = np.array([_slope(sizes, np.exp(lg)) for lg in logs])
    th_floor_sd = float(draw_slopes.std(ddof=1)) if n_floor_draws > 1 else float("nan")
    d_obs = th_obs - th_floor

    # every accumulation order is read against a floor at its own rung sizes: with unequal classes another order
    # has other sizes. Equal sizes reuse the declared floor; new sizes draw from a separate stream so the floor,
    # permutation and order streams of a given seed are unchanged.
    floors = {tuple(sizes): th_floor}
    xrng = np.random.default_rng([floor_seed if floor_seed is not None else seed, 7919])

    def floor_for(sz):
        key = tuple(sz)
        if key not in floors:
            lg = np.array([[np.log(max(_subset_pr(K, xrng.choice(n, s, replace=False)), 1e-9)) for s in key]
                           for _ in range(n_floor_draws)])
            floors[key] = _slope(key, np.exp(lg.mean(axis=0)))
        return floors[key]

    def shift_of(mem, o):
        th, sz, _ = _ladder(K, mem, o, bc)
        return th - floor_for(sz)

    floor_means = np.exp(logs.mean(axis=0))
    deficit = [float(np.log(max(o, 1e-9)) - np.log(max(f, 1e-9))) for o, f in zip(pr_obs, floor_means)]
    out = {"delta": d_obs, "theta_obs": th_obs, "theta_floor": th_floor,
           "theta_floor_sd": th_floor_sd, "theta_floor_se": th_floor_sd / np.sqrt(n_floor_draws),
           "rung_sizes": sizes, "rung_classes": list(bc), "n_classes": n_classes,
           "pr_obs": [float(v) for v in pr_obs],
           "pr_floor": [float(v) for v in floor_means],
           # per-rung log-PR deficit below the floor (zero at the top rung, where the
           # ladder and the floor hold the same samples) and the share of the climb
           # that remains after half the classes: with antipodal pairs, the share of the climb
           # left after half the circle (Appendix K), when the declared order lists one member of each antipodal
           # pair before any of their partners
           "deficit": deficit,
           "n_perm": int(n_perm), "k_orders": int(k_orders),
           "n_floor_draws": int(n_floor_draws), "n_samples": int(n_samples),
           "n_features": int(X.shape[1])}
    out["order"] = [classes[i].item() if hasattr(classes[i], "item") else classes[i] for i in canonical]
    out["same_class_rate"] = run
    if run["excess_se"] > 3 and not run["grouped_by_class"] and time_blocks is None:
        import warnings
        warnings.warn(f"zoom: consecutive rows share a class {run['rate']:.2f} of the time against {run['expected']:.2f} "
                      f"for unordered labels ({run['excess_se']:.0f} SE above): if rows are in acquisition order, "
                      "classes cluster in time and drift can enter the shift; add time_blocks (rung data "
                      "--time-blocks E) for the null that permutes labels only within time blocks.", stacklevel=2)
    half = n_classes // 2
    if half in bc and deficit[0] != 0:
        out["late_rung_classes"] = int(half)
        out["late_fraction"] = float(deficit[bc.index(half)] / deficit[0])
    if antipode is not None:
        pairs = {int(np.searchsorted(classes, k)): int(np.searchsorted(classes, v)) for k, v in antipode.items()}
        first = list(canonical[:half])
        clash = [(a, pairs[a]) for a in first if a in pairs and pairs[a] in first]
        out["antipode_first_half_has_no_pair"] = not clash
        if clash:
            import warnings
            warnings.warn("zoom: the first half of the declared order holds an antipodal pair "
                          f"{clash[:3]}; the share of the climb left after half the circle needs one member of each pair before any partner "
                          "(relabel the classes so the pairs sit half a cycle apart).", stacklevel=2)

    orders = [orng.permutation(n_classes).tolist() for _ in range(k_orders)]
    d_orders = [shift_of(members, o) for o in orders]
    out["delta_orderavg"] = float(np.mean(d_orders)) if d_orders else float("nan")
    out["delta_orderavg_sd"] = float(np.std(d_orders)) if d_orders else float("nan")

    if n_perm:
        null_c, null_a = [], []
        for _ in range(n_perm):
            pl = lab[prng.permutation(n)]
            mem = [np.where(pl == c)[0] for c in range(n_classes)]
            null_c.append(_ladder(K, mem, canonical, bc)[0] - th_floor)
            null_a.append(np.mean([shift_of(mem, o) for o in orders[:k_null_orders]])
                          if k_null_orders > 0 and orders else float("nan"))
        for tag, obs, null in (("", d_obs, null_c),
                               ("_orderavg", out["delta_orderavg"], null_a)):
            z, pv, nm, ns = _null_stats(obs, null)
            out[f"z{tag}"], out[f"p_two{tag}"] = z, pv
            out[f"null_mean{tag}"], out[f"null_sd{tag}"] = nm, ns
        if keep_null:
            out["null_draws"] = [float(v) for v in null_c]

    if strata is not None and n_perm:
        strata = np.asarray(strata)
        if strata.shape != (n,):
            raise ValueError(f"strata must hold one stratum id per row ({n}); got {strata.shape}")
        smem = {s: np.where(strata == s)[0] for s in np.unique(strata)}
        null_s, null_sa = [], []
        for _ in range(n_perm):
            sl = lab.copy()
            for idx in smem.values():
                sl[idx] = sl[idx[prng.permutation(len(idx))]]
            mem = [np.where(sl == c)[0] for c in range(n_classes)]
            null_s.append(_ladder(K, mem, canonical, bc)[0] - th_floor)
            # the same within-stratum relabelling read by the order-averaged statistic (README rule 3 with rule 4)
            null_sa.append(np.mean([shift_of(mem, o) for o in orders[:k_null_orders]])
                           if k_null_orders > 0 and orders else float("nan"))
        out["strat_z"], out["strat_p_two"], out["strat_null_mean"], out["strat_null_sd"] = _null_stats(d_obs, null_s)
        (out["strat_z_orderavg"], out["strat_p_two_orderavg"],
         out["strat_null_mean_orderavg"], out["strat_null_sd_orderavg"]) = _null_stats(out["delta_orderavg"], null_sa)
        if keep_null:
            out["strat_null_draws"] = [float(v) for v in null_s]

    if shift_null and n_perm:
        null_sh = []
        for _ in range(n_perm):
            sl = np.roll(lab, int(prng.integers(1, n)))
            mem = [np.where(sl == c)[0] for c in range(n_classes)]
            null_sh.append(_ladder(K, mem, canonical, bc)[0] - th_floor)
        out["shift_z"], out["shift_p_two"], out["shift_null_mean"], out["shift_null_sd"] = _null_stats(d_obs, null_sh)

    if time_blocks is not None and n_perm:
        tb = np.asarray(time_blocks)
        if tb.shape != (n,):
            raise ValueError(f"time_blocks must hold one block id per row ({n}); got {tb.shape}")
        bmem = [np.where(tb == b)[0] for b in np.unique(tb)]
        null_t, null_ta = [], []
        for _ in range(n_perm):
            tl = lab.copy()
            for idx in bmem:
                tl[idx] = tl[idx[prng.permutation(len(idx))]]
            mem = [np.where(tl == c)[0] for c in range(n_classes)]
            null_t.append(_ladder(K, mem, canonical, bc)[0] - th_floor)
            null_ta.append(np.mean([shift_of(mem, o) for o in orders[:k_null_orders]])
                           if k_null_orders > 0 and orders else float("nan"))
        for tag, obs, null in (("", d_obs, null_t), ("_orderavg", out["delta_orderavg"], null_ta)):
            z, pv, nm, ns = _null_stats(obs, null)
            out[f"tb_z{tag}"], out[f"tb_p_two{tag}"] = z, pv
            out[f"tb_null_mean{tag}"], out[f"tb_null_sd{tag}"] = nm, ns
        out["n_time_blocks"] = len(bmem)

    if strata is not None:
        groups = pair_groups(lab, np.asarray(strata), n_classes)
        if len(groups) >= 2 and max(len(g) for g in groups) >= 2:
            pp, n_pp = pair_preserving_orders(groups, k_max=5000, rng=orng)
            sub = pp if len(pp) <= max(k_orders, 50) else [pp[i] for i in orng.choice(len(pp), max(k_orders, 50), replace=False)]
            out["pair_groups"] = [[classes[i].item() if hasattr(classes[i], "item") else classes[i] for i in g] for g in groups]
            out["n_pair_orders"] = int(n_pp)
            out["delta_pairavg"] = float(np.mean([shift_of(members, o) for o in pp]))
            if n_perm:
                obs_sub = float(np.mean([shift_of(members, o) for o in sub]))
                smem = [np.where(np.asarray(strata) == s_)[0] for s_ in np.unique(strata)]
                null_p = []
                for _ in range(n_perm):
                    sl = lab.copy()
                    for idx in smem:
                        sl[idx] = sl[idx[prng.permutation(len(idx))]]
                    mem = [np.where(sl == c)[0] for c in range(n_classes)]
                    null_p.append(np.mean([shift_of(mem, o) for o in sub]))
                out["pairavg_z"], out["pairavg_p_two"], out["pairavg_null_mean"], out["pairavg_null_sd"] = _null_stats(obs_sub, null_p)
                out["n_pair_orders_in_null"] = len(sub)

    out["n_floor_sequences"] = len(floors)   # rung-size sequences read against their own floor
    return out


def same_class_rate(lab):
    """How often consecutive rows share a class, against the expectation for the same labels in random order
    (sum n_c (n_c - 1) / (n (n - 1))). Meaningful when rows are in acquisition order: a rate far above the expectation
    means classes were presented in runs, and slow drift can then enter a shift."""
    lab = np.asarray(lab); n = len(lab)
    if n < 3:
        return {"rate": float("nan"), "expected": float("nan"), "excess_se": 0.0}
    rate = float(np.mean(lab[1:] == lab[:-1]))
    cnt = np.unique(lab, return_counts=True)[1].astype(float)
    e = float((cnt * (cnt - 1)).sum() / (n * (n - 1)))
    se = float(np.sqrt(max(e * (1 - e), 1e-12) / (n - 1)))
    runs = int(1 + np.sum(lab[1:] != lab[:-1]))
    # rows stored grouped by class (one run per class, as in a prompt battery) say nothing about acquisition time
    return {"rate": rate, "expected": e, "excess_se": float((rate - e) / se), "n_runs": runs,
            "grouped_by_class": bool(runs == len(cnt))}


def valid_from_intervals(starts, stops, intervals):
    """Per-sample validity from invalid time intervals (e.g. a release's invalid_times for a probe): a sample whose window
    [start, stop] overlaps an interval (start < interval stop and stop > interval start) is invalid. Returns a bool mask."""
    starts, stops = np.asarray(starts, float).ravel(), np.asarray(stops, float).ravel()
    valid = np.ones(len(starts), bool)
    for a, z in np.asarray(intervals, float).reshape(-1, 2):
        valid &= ~((starts < z) & (stops > a))
    return valid


def apply_valid(valid, *arrays):
    """Keep only the valid samples (rows) of every array given; None passes through. Statistics then see the remaining
    samples only: class sizes may become unequal, which every rung and floor already handles."""
    valid = np.asarray(valid).astype(bool).ravel()
    return [None if a is None else np.asarray(a)[valid] for a in arrays]


def block_ids(n, size):
    """Contiguous acquisition blocks: rows 0..size-1 are block 0, and so on (rows must be in time order)."""
    return np.arange(n) // int(size)


def pair_groups(lab, strata, n_classes):
    """Classes that share any stratum id belong together (the two members of a minimal pair, the arms of a
    contrast); returns the connected groups of class indices, each sorted."""
    parent = list(range(n_classes))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for s_ in np.unique(strata):
        cs = np.unique(lab[strata == s_])
        for c in cs[1:]:
            parent[find(int(c))] = find(int(cs[0]))
    groups = {}
    for c in range(n_classes):
        groups.setdefault(find(c), []).append(c)
    return sorted(groups.values())


def pair_preserving_orders(groups, k_max=5000, rng=None):
    """Every accumulation order that keeps each group's classes adjacent: the groups in any order, each group's
    classes in any order. Enumerated when there are at most k_max, else k_max drawn at random. Returns
    (orders, total count)."""
    import itertools
    import math
    total = math.factorial(len(groups)) * math.prod(math.factorial(len(g)) for g in groups)
    if total <= k_max:
        orders = [sum((list(p) for p in inner), [])
                  for outer in itertools.permutations(groups)
                  for inner in itertools.product(*[list(itertools.permutations(g)) for g in outer])]
        return orders, total
    rng = rng if rng is not None else np.random.default_rng(0)
    orders = []
    for _ in range(k_max):
        outer = [groups[i] for i in rng.permutation(len(groups))]
        orders.append(sum(([g[i] for i in rng.permutation(len(g))] for g in outer), []))
    return orders, total


def declared_order(classes, order=None):
    """The accumulation order as indices into the sorted class list: None or
    "sequential" for the sorted labels, "antipodal" for one member of each
    antipodal pair first (0, K/2, 1, K/2+1, ...), or an explicit list of class
    labels."""
    classes = np.asarray(classes); K = len(classes)
    if order is None or (isinstance(order, str) and order == "sequential"):
        return list(range(K))
    if isinstance(order, str) and order == "antipodal":
        if K % 2:
            raise ValueError("the antipodal order needs an even number of classes")
        h = K // 2
        return [i for pair in zip(range(h), range(h, K)) for i in pair]
    missing = [o for o in order if not np.any(classes == o)]
    if missing:
        raise ValueError(f"order names classes the labels do not hold: {missing[:5]}")
    idx = [int(np.searchsorted(classes, o)) for o in order]
    if sorted(idx) != list(range(K)):
        raise ValueError("order must list every class exactly once")
    return idx


def sectors(X, labels, order=None):
    """The harmonic content of the class-mean kernel for a cyclic axis (the
    camera-ready's Section 4): the classes, taken in sorted label order (or in
    `order`) as equally spaced points on a circle,
    have mean response vectors whose Pearson correlation across features is
    averaged into a circular profile C(k) over class separations k; its
    discrete Fourier transform is exact on the class grid. Reports the
    coefficients q_l (for eight classes the paper's a, c1, b2, c3, b4), the
    sector balance q2/|q1| (the quadrupole-to-dipole ratio b2/|c1|), the even
    and odd amplitudes, and the coherence of adjacent and antipodal classes.
    Which accumulation order climbs faster is set by how much the classes'
    fluctuations overlap (Section 5, Appendix E), not by these class-mean coherences."""
    X = np.asarray(X, dtype=np.float64); labels = np.asarray(labels)
    classes = np.unique(labels); K = len(classes)
    if K < 3:
        raise ValueError("sectors needs at least three classes on the cycle")
    idx = declared_order(classes, order)
    M = np.stack([X[labels == classes[i]].mean(axis=0) for i in idx])
    Cm = np.corrcoef(M)
    prof = np.array([np.mean([Cm[i, (i + k) % K] for i in range(K)]) for k in range(K)])
    ang = 2 * np.pi * np.arange(K) / K
    coef = {"a": float(prof.mean())}
    for l in range(1, K // 2 + 1):
        w = (1.0 if (K % 2 == 0 and l == K // 2) else 2.0) / K
        coef[f"q{l}"] = float(w * np.sum(prof * np.cos(l * ang)))
    even = [coef[f"q{l}"] for l in range(2, K // 2 + 1, 2)]
    odd = [coef[f"q{l}"] for l in range(1, K // 2 + 1, 2)]
    out = {"n_classes": K, "profile": [float(v) for v in prof], "coefficients": coef,
           "A_even": float(np.sqrt(sum(v * v for v in even))), "A_odd": float(np.sqrt(sum(v * v for v in odd))),
           "even_share": float(sum(v * v for v in even) / max(sum(v * v for v in even + odd), 1e-300)),
           "C_adjacent": float(prof[1])}
    if "q2" in coef:
        out["balance"] = float(coef["q2"] / abs(coef["q1"])) if coef["q1"] != 0 else float("inf")
    if K == 8:
        out["named"] = {"a": coef["a"], "c1": coef["q1"], "b2": coef["q2"], "c3": coef["q3"], "b4": coef["q4"]}
    if K % 2 == 0:
        out["C_antipodal"] = float(prof[K // 2])
    return out


def _gamma_nodes(kappa, n_nodes=48):
    """Quadrature nodes and weights for a trial gain tau with E tau = 1 and E tau^2 = kappa (Gamma shape s = 1/(kappa - 1)):
    generalized Gauss-Laguerre by Golub-Welsch, numpy only; tau = 1 when kappa <= 1."""
    if kappa <= 1 + 1e-9:
        return np.ones(1), np.ones(1)
    s = 1.0 / (kappa - 1.0); a = s - 1.0
    i = np.arange(n_nodes)
    J = np.diag(2 * i + a + 1) + np.diag(np.sqrt((i[1:]) * (i[1:] + a)), 1) + np.diag(np.sqrt((i[1:]) * (i[1:] + a)), -1)
    x, V = np.linalg.eigh(J)
    w = V[0] ** 2
    return x / s, w / w.sum()


def _projector_weights(lam, n, k, tau, w, n_grid=1500, eta_rel=1e-5):
    """Expected weight m_a of the top-k sample subspace of n - 1 centred samples on population eigenvector a, from the
    deterministic equivalent of the sample resolvent for x = sqrt(tau) C^(1/2) z: Qbar(z) = (-z + g(z) C)^-1,
    g = E[tau / (1 + tau delta)], delta = Tr C Qbar / (n - 1); m_a = the spectral weight of a above the k-th sample eigenvalue."""
    nu = n - 1
    hi = (lam.sum() / nu + lam.max()) * (1 + np.sqrt(tau.max())) * 4
    lo = max(lam[lam > 0].min() * 1e-3, hi * 1e-9)
    ls = np.geomspace(hi, lo, n_grid)
    eta = eta_rel * lam.max()
    d = -lam.sum() / (nu * (ls[0] + 1j * eta))
    Q = np.empty((n_grid, len(lam)))
    for i, l in enumerate(ls):
        z = l + 1j * eta
        for _ in range(4000):
            dn = np.sum(lam / (lam * np.sum(w * tau / (1 + tau * d)) - z)) / nu
            if abs(dn - d) < 1e-12 * (1 + abs(d)):
                d = dn
                break
            d = 0.5 * d + 0.5 * dn
        Q[i] = (1.0 / (lam * np.sum(w * tau / (1 + tau * d)) - z)).imag / np.pi
    cum_a = np.vstack([np.zeros(len(lam)), np.cumsum(0.5 * (Q[1:] + Q[:-1]) * (-np.diff(ls))[:, None], axis=0)])
    cum = cum_a.sum(1)
    j = int(np.searchsorted(cum, k))
    if j >= n_grid:
        raise RuntimeError("floor law: the spectral grid does not reach k sample eigenvalues")
    f = (k - cum[j - 1]) / (cum[j] - cum[j - 1])
    return cum_a[j - 1] + f * (cum_a[j] - cum_a[j - 1])


def principal_angle_dissimilarity(X, labels, k=5):
    """1 - mean over class pairs of ||U_c' U_d||_F^2 / k, U_c the top-k directions of class c's residuals (k capped at the
    smallest class size - 1)."""
    X = np.asarray(X, dtype=np.float64); labels = np.asarray(labels)
    classes = np.unique(labels)
    R = [X[labels == c] - X[labels == c].mean(0) for c in classes]
    kk = min(k, min(len(r) for r in R) - 1, X.shape[1])
    if kk < 1:
        raise ValueError("principal angles need k >= 1 and at least two samples in every class")
    U = [np.linalg.svd(r, full_matrices=False)[2][:kk].T for r in R]
    ov = [np.sum((U[a].T @ U[b]) ** 2) / kk for a in range(len(U)) for b in range(a + 1, len(U))]
    return float(1 - np.mean(ov))


def cov_floors(X, labels, k=5, n_perm=0, seed=0):
    """The label-blind floor of the principal-angle dissimilarity between classes, predicted analytically.

    With the labels ignored, two classes are independent samples of one distribution, so the expected overlap of their top-k
    residual subspaces is Tr(M_c M_d), M the expected top-k sample projector at each class's size; M is diagonal in the
    eigenbasis of the pooled covariance and follows from a random-matrix law (the deterministic equivalent of the sample
    resolvent, elliptical form) whose inputs are the pooled spectrum and the trial gain's kurtosis (each sample's squared
    radius kurtosis, net of the Gaussian quadratic form's own 1 + 2/d_eff). Returns the observed value, the floor (with and
    without the gain term), the floor-corrected value and the ceiling-normalized structure part (observed - floor) / (1 - floor);
    with n_perm > 0 also the permutation floor for comparison. Like the PR floor of Eq. 2, the floor depends on the design:
    compare corrected or normalized values across designs, never raw ones."""
    X = np.asarray(X, dtype=np.float64); labels = np.asarray(labels)
    classes = np.unique(labels)
    sizes = [int((labels == c).sum()) for c in classes]
    kk = min(k, min(sizes) - 1, X.shape[1])
    if kk < 1:
        raise ValueError("cov_floors needs k >= 1 and at least two samples in every class")
    Xc = X - X.mean(0)
    G = Xc @ Xc.T if Xc.shape[0] < Xc.shape[1] else Xc.T @ Xc          # nonzero spectrum from the smaller Gram
    lam = np.clip(np.linalg.eigvalsh(G), 0, None) / (len(Xc) - 1)
    if lam.max() <= 0:
        raise ValueError("cov_floors: the samples have no variance")
    lam = lam[lam > lam.max() * 1e-12]
    if kk >= len(lam):
        # every class's top-kk subspace is the whole supported space: dissimilarity and its floor are both zero
        obs = principal_angle_dissimilarity(X, labels, k)
        return {"k": kk, "class_sizes": sizes, "observed": obs, "floor": 0.0, "floor_spectrum_only": 0.0,
                "gain_kurtosis": float("nan"), "d_eff": float(lam.sum() ** 2 / np.sum(lam ** 2)), "corrected": obs,
                "structure": obs, "floor_share": float("nan"), "full_space": True}
    d_eff = float(lam.sum() ** 2 / np.sum(lam ** 2))
    r2 = np.sum(Xc * Xc, axis=1)
    kappa = float(np.mean(r2 ** 2) / np.mean(r2) ** 2 / (1 + 2 / d_eff))

    def law(tau, w):
        m = {n: _projector_weights(lam, n, kk, tau, w) for n in sorted(set(sizes))}
        ov = [m[sizes[a]] @ m[sizes[b]] / kk for a in range(len(sizes)) for b in range(a + 1, len(sizes))]
        return float(1 - np.mean(ov))
    obs = principal_angle_dissimilarity(X, labels, k)
    fl = law(*_gamma_nodes(kappa)); fl0 = law(np.ones(1), np.ones(1))
    out = {"k": kk, "class_sizes": sizes, "observed": obs, "floor": fl, "floor_spectrum_only": fl0, "gain_kurtosis": kappa,
           "d_eff": d_eff, "corrected": obs - fl, "structure": (obs - fl) / (1 - fl) if fl < 1 else float("nan"),
           "floor_share": fl / obs if obs > 0 else float("nan")}
    if n_perm:
        rng = np.random.default_rng(seed)
        out["floor_permutation"] = float(np.mean([principal_angle_dissimilarity(X, labels[rng.permutation(len(labels))], k)
                                                  for _ in range(n_perm)]))
    return out


def _print_cov_floors(r):
    print(f"principal-angle dissimilarity (k = {r['k']}, class sizes {min(r['class_sizes'])}-{max(r['class_sizes'])}): "
          f"observed {r['observed']:.3f}, label-blind floor {r['floor']:.3f} ({100 * r['floor_share']:.0f}% of observed; "
          f"spectrum alone {r['floor_spectrum_only']:.3f}, gain kurtosis {r['gain_kurtosis']:.2f})")
    print(f"  floor-corrected {r['corrected']:+.3f}, structure part (ceiling-normalized) {r['structure']:+.3f}"
          + (f"; permutation floor {r['floor_permutation']:.3f}" if "floor_permutation" in r else ""))


def _print_sectors(r):
    c = r["coefficients"]
    named = r.get("named")
    head = (f"a {named['a']:+.3f}, c1 {named['c1']:+.3f}, b2 {named['b2']:+.3f}, c3 {named['c3']:+.3f}, b4 {named['b4']:+.3f}"
            if named else ", ".join(f"{k} {v:+.3f}" for k, v in c.items()))
    print(f"sectors: class-mean kernel harmonics {head}")
    line = f"  even (orientation-like) amplitude {r['A_even']:.3f}, odd (direction-like) {r['A_odd']:.3f}, even share {r['even_share']:.2f}"
    if "balance" in r:
        line += f"; sector balance q2/|q1| = {r['balance']:.2f}"
    print(line)
    if "C_antipodal" in r:
        print(f"  adjacent classes cohere at C = {r['C_adjacent']:.2f}, antipodal at {r['C_antipodal']:.2f}: "
              "the order the fluctuations favour is read from their overlaps (Section 5, Appendix E), not from these")


def split_by(X, labels, score, n_subsets=3, seed=0, **zoom_kwargs):
    """Section 4's direction-selective thirds as one call: sort the features by a
    per-feature score (a direction-selectivity index, a tuning strength, a
    weight norm), take the top and bottom 1/n_subsets of them and a random
    subset of the same size, and run zoom() on each with its own floor. The
    shift depends on the number of features (a random third of a recording's
    neurons differs from the full population by -0.08 to +0.14), so subsets are
    compared with each other at this matched size and never with the full
    population."""
    X = np.asarray(X); score = np.asarray(score, dtype=float)
    if score.shape != (X.shape[1],):
        raise ValueError(f"score must have one value per feature ({X.shape[1]}); got {score.shape}")
    if int(n_subsets) < 2:
        raise ValueError("split_by needs n_subsets >= 2")
    m = X.shape[1] // n_subsets
    if m < 1:
        raise ValueError(f"split_by: {X.shape[1]} features cannot make {n_subsets} subsets of at least one feature")
    order = np.argsort(score)
    rng = np.random.default_rng(seed)
    subsets = {"top": order[-m:], "bottom": order[:m], "random": rng.choice(X.shape[1], m, replace=False)}
    out = {"n_features_per_subset": int(m), "n_features_total": int(X.shape[1]), "subsets": {}}
    for name, idx in subsets.items():
        r = zoom(X[:, np.sort(idx)], labels, **zoom_kwargs)
        r["median_score"] = float(np.median(score[idx]))
        out["subsets"][name] = r
    return out


def _plot_data(result, out_png, title=None):
    """Ladder figure for one `zoom` result: observed rungs against the matched
    floor on log-log axes, with the shift and its permutation p annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = result["rung_sizes"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(sizes, result["pr_floor"], "o-", color="0.55", label="matched floor (random subsets)")
    ax.plot(sizes, result["pr_obs"], "o-", color="#c23b3b", label="declared axis (classes accumulated)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("samples in rung"); ax.set_ylabel("participation ratio")
    p = result.get("p_two"); ptxt = f"; permutation p = {p:.3f}" if p is not None else ""
    ax.set_title(title or f"shift delta = {result['delta']:+.3f}{ptxt}\n"
                 f"theta_obs {result['theta_obs']:+.3f} = floor {result['theta_floor']:+.3f} + delta", fontsize=8)
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    for axis_ in (ax.xaxis, ax.yaxis):
        axis_.set_major_formatter(ScalarFormatter()); axis_.set_minor_formatter(NullFormatter())
    ax.set_xticks(sizes); ax.set_xticklabels([str(v) for v in sizes], fontsize=7)
    yv = np.array(result["pr_obs"] + result["pr_floor"], float)
    yt = np.unique(np.round(np.geomspace(yv.min(), yv.max(), 5), 1 if yv.max() < 20 else 0))
    ax.set_yticks(yt); ax.set_yticklabels([f"{v:g}" for v in yt], fontsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return out_png


def _p_unresolved(p, n_perm, alpha=0.05):
    """A Monte Carlo p that cannot fall below `alpha` at this budget: with B permutations the smallest p is 1/(B + 1), so
    for B < 1/alpha - 1 a p >= alpha says nothing about separation. Returns the plain-language reason, or None when p is
    readable (p < alpha, or a budget that resolves alpha)."""
    if not n_perm or p is None or not np.isfinite(p) or p < alpha:
        return None
    res = 1.0 / (n_perm + 1)
    if res < alpha:
        return None
    reached = int(round(p * (n_perm + 1))) - 1
    return (f"unresolved: {n_perm} permutations cannot give p below {res:.3f} "
            + ("(no permutation reached the observed value)" if reached <= 0 else f"({reached} of {n_perm} reached it)")
            + "; rerun with --n-perm 200 or more")


def _second_null_verdict(p_label, p_second, alpha=0.05, n_perm=None):
    """One statistic against a nuisance-keeping second null: 'unresolved' when the permutation budget cannot give p below
    alpha, 'unseparated' when the label permutations do not separate it to begin with, otherwise 'survives' or 'absorbed'
    by the second null."""
    if n_perm and 1.0 / (n_perm + 1) >= alpha:
        return "unresolved"
    if p_label >= alpha:
        return "unseparated"
    return "survives" if p_second < alpha else "absorbed"


def summarize_data(result, verbose=True):
    """Plain-language reading of one `rung data` result under the paper's rules: the split of the exponent with the floor's
    own uncertainty, the declared and order-averaged shifts against their nulls, the nuisance-preserving, time-blocked and
    circular-shift nulls, the share of the climb left after half the circle, and, when present, the split into within-class and class-mean parts, the sign
    law, the two-term account, the sectors, the principal-angle floor, the candidates, the feature ladder and the blocks."""
    n_perm = int(result.get("n_perm", 0)); floor_p = 1.0 / (n_perm + 1) if n_perm else None
    se = result.get("theta_floor_se")
    lines = [f"theta_obs = {result['theta_obs']:+.3f} = floor {result['theta_floor']:+.3f} + shift {result['delta']:+.3f} "
             f"({result['n_classes']} classes, rungs {result['rung_sizes']})"
             + (f"; the floor's standard error {se:.3f} ({result['n_floor_draws']} draws) is the uncertainty the shift inherits from it"
                if se is not None and np.isfinite(se) else "")]
    verdict = {"delta": result["delta"]}
    cert = None
    if "p_two" in result:
        cert = bool(result["p_two"] < 0.05)
        unres = _p_unresolved(result["p_two"], n_perm)
        lines.append(f"declared order: z = {result['z']:+.1f}, two-sided Monte Carlo p = {result['p_two']:.3f} "
                     f"(resolution {floor_p:.3f}); "
                     + ("label-linked at p < 0.05" if cert else unres or "not separated from the label-permutation null"))
        verdict["certified"] = cert
        if unres:
            verdict["p_unresolved"] = True
    if "delta_orderavg" in result and np.isfinite(result["delta_orderavg"]):
        oa = result["delta_orderavg"]; same = (oa > 0) == (result["delta"] > 0)
        txt = f"order-averaged shift {oa:+.3f} (SD over orders {result['delta_orderavg_sd']:.3f})"
        if "p_two_orderavg" in result:
            txt += f", p = {result['p_two_orderavg']:.3f}"
        txt += "; same sign as the declared order" if same else "; OPPOSITE sign to the declared order: the declared-order value is a path property, report the order-averaged one for the partition"
        lines.append(txt); verdict["orderavg_same_sign"] = bool(same)
        sd_o = result.get("delta_orderavg_sd")
        if sd_o is not None and np.isfinite(sd_o) and 2 * sd_o >= abs(result["delta"]):
            lines.append(f"the declared shift ({result['delta']:+.3f}) lies within two SDs over random class orders ({sd_o:.3f}) of zero: "
                         "other orders of the same classes can give the opposite sign, so the declared order is one path among many; "
                         "read the order-averaged shift (README rule 3)")
            verdict["read_orderavg"] = True
        elif sd_o is not None and np.isfinite(sd_o) and abs(result["delta"] - oa) > 2 * sd_o:
            lines.append(f"the declared shift ({result['delta']:+.3f}) lies more than two SDs over random class orders ({sd_o:.3f}) from "
                         f"their average ({oa:+.3f}): the declared order is an atypical path through these classes; read the "
                         "order-averaged shift for the partition (README rule 3)")
            verdict["read_orderavg"] = True
    for key, name, what, kept, lost in (
            ("strat", "nuisance-preserving null (labels permuted within strata)", "label_linked_beyond_strata",
             "survives within-stratum relabelling: label-linked beyond the nuisance",
             "is ABSORBED by within-stratum relabelling: it reads nuisance composition, not the labels"),
            ("tb", f"time-blocked null ({result.get('n_time_blocks')} blocks; labels permuted within blocks, which keeps drift and slow state)",
             "beyond_drift", "survives: drift cannot account for it", "does not survive: drift or slow state could produce it")):
        if f"{key}_p_two" not in result:
            continue
        txt = f"{name}: p = {result[f'{key}_p_two']:.3f}"
        if f"{key}_p_two_orderavg" in result:
            txt += f", order-averaged p = {result[f'{key}_p_two_orderavg']:.3f}"
        parts = []
        for tag, sfx in (("declared shift", ""), ("order-averaged shift", "_orderavg")):
            p_lab, p_2nd = result.get(f"p_two{sfx}"), result.get(f"{key}_p_two{sfx}")
            if p_lab is None or p_2nd is None or not (np.isfinite(p_lab) and np.isfinite(p_2nd)):
                continue
            v = _second_null_verdict(p_lab, p_2nd, n_perm=n_perm)
            verdict[what + sfx] = {"survives": True, "absorbed": False}.get(v)
            parts.append(f"the {tag} " + {"survives": kept, "absorbed": lost,
                                            "unseparated": "is not separated from the label-permutation null to begin with",
                                            "unresolved": f"cannot be read at {n_perm} permutations (p cannot fall below "
                                                          f"{floor_p:.3f}); rerun with --n-perm 200 or more"}[v])
        if parts:
            txt += "; " + "; ".join(parts)
        lines.append(txt)
    if "shift_p_two" in result:
        survives = result["shift_p_two"] < 0.05
        unres = _p_unresolved(result["shift_p_two"], n_perm)
        lines.append(f"circular-shift null (labels in time order): p = {result['shift_p_two']:.3f}; "
                     + ("the shift survives rolling the label sequence: the alignment of labels with frames matters beyond their slowness"
                        if survives else unres or "the shift does NOT survive rolling the label sequence: a slowly varying label would read this way by drift alone"))
        verdict["survives_circular_shift"] = bool(survives) if not unres else None
    if "deficit" in result:
        lines.append("per-rung log-PR deficit below the floor, by classes accumulated "
                     + ", ".join(f"{c}: {d:+.3f}" for c, d in zip(result.get("rung_classes", []), result["deficit"])))
        if "late_fraction" in result:
            lf = result["late_fraction"]; h = result["late_rung_classes"]
            txt = f"share of the climb remaining after {h} of {result['n_classes']} classes: {lf:.2f}"
            if "antipode_first_half_has_no_pair" in result:
                if result["antipode_first_half_has_no_pair"]:
                    txt += (" (with antipodal pairs, the share of the climb left after half the circle: the first half holds one member of each pair; near 0 = the code has every class mean by half the classes, "
                            "it reads the quotient; large = the second half still adds directions, the code distinguishes a class from its antipode)")
                else:
                    txt += " (the first half of the declared order holds an antipodal pair, so this is not the share left after half the circle; relabel)"
            lines.append(txt); verdict["late_fraction"] = float(lf)
    if "decompose" in result:
        dc = result["decompose"]; f = dc["delta_split_free"]
        cm = f["Tb"] + f["Cb"] + f["Cx"]
        lines.append(f"split (Section 5): within-class part D {f['D']:+.3f} + class-mean part (Tb + Cb + Cx) {cm:+.3f} = shift {dc['delta']:+.3f}"
                     " (decompose's own floor draws)")
        verdict.update(within_part=float(f["D"]), class_mean_part=float(cm))
        cp = dc.get("class_pr"); sl0 = dc.get("sign_law") or {}
        if cp and dc.get("rungs"):
            n_pc = dc["rungs"][-1]["size"] / max(dc["rung_classes"][-1], 1)
            v = [x for x in cp.values() if np.isfinite(x)]
            if v:
                lines.append(f"sampling: {n_pc:.0f} samples per class against each class's own fluctuation dimension {min(v):.1f} to {max(v):.1f} "
                             f"(median {float(np.median(v)):.1f})" + (f"; PR(W) {sl0['PR_W']:.1f} pools the classes" if "PR_W" in sl0 else "")
                             + ("" if n_pc > max(v) else "; below a class's own dimension the within-class part can read near zero "
                                "where structure exists (Section 2)"))
        sl = dc.get("sign_law")
        if sl and "degenerate" not in sl:
            lines.append(f"sign law (Eq. 3): full-set class-mean remainder c {sl['c_full']:+.3f}, negative exactly when PR(W)/PR(B) = "
                         f"{sl['ratio']:.1f} exceeds 1 + 2(1 - a)/beta = {sl['threshold_full']:.1f} (a {sl['alignment']:.2f}, beta {sl['trB_trW']:.3f}); "
                         + ("its sign predicts the class-mean part's, and here they agree" if np.sign(sl["c_full"]) == np.sign(cm)
                            else "its sign predicts the class-mean part's, and here they DIFFER (a first rung with class-mean variance, or a small part)"))
        if "net_of_bias" in dc:
            lines.append(f"finite-sample bias null (one covariance shape for every class): within-class part {dc['bias_null']['delta_split_free']['D']:+.3f} "
                         f"in the surrogates, {dc['net_of_bias']['delta_split_free']['D']:+.3f} net of them")
    if "account" in result:
        a_ = result["account"]
        lines.append(f"two-term account (Figure 3): within-class part {a_['measured']['within']:+.3f} against the overlap law {a_['within_pred']['mean_field']:+.3f} "
                     f"and the finite-sample identity {a_['within_pred']['finite']:+.3f}; class-mean part {a_['measured']['class_mean']:+.3f} against "
                     f"Eq. 3 through the rungs {a_['class_mean_pred']:+.3f}; shift {a_['measured']['delta']:+.3f} against their sum "
                     f"{a_['delta_pred_mean_field']:+.3f} ({'sign agrees' if a_['signs_agree']['mean_field'] else 'sign differs'}); "
                     f"beyond mean field: separation {a_['within_pred']['population'] - a_['within_pred']['mean_field']:+.3f}, sampling "
                     f"{a_['within_pred']['finite'] - a_['within_pred']['population']:+.3f} (s {a_['sampling_term']:.3f})")
    if "sectors" in result:
        sc = result["sectors"]
        txt = f"class-mean kernel: even amplitude {sc['A_even']:.3f}, odd {sc['A_odd']:.3f}"
        if "balance" in sc:
            txt += f", sector balance {sc['balance']:.2f}"
        lines.append(txt)
    if "cov_floors" in result:
        cf = result["cov_floors"]
        lines.append(f"principal-angle dissimilarity {cf['observed']:.3f} against its label-blind floor {cf['floor']:.3f}: corrected "
                     f"{cf['corrected']:+.3f}, structure part {cf['structure']:+.3f}; compare corrected values across designs, never raw ones")
    if "candidates" in result:
        c_ = result["candidates"]
        if "pick" in c_:
            lines.append("candidates ranked by z: " + ", ".join(f"{k} {c_['screen'][k]['z']:+.1f}" for k in c_["ranking"]))
        else:
            lines.append("candidates: " + "; ".join(f"half {p_['half']} picks {p_['pick']} ({'confirmed' if p_['confirmed'] else 'not confirmed'} on the other)"
                                                     for p_ in c_["picks"]))
    if "feature_ladder" in result:
        fl = result["feature_ladder"]
        lines.append(f"feature ladder: theta_obs {fl['theta_obs']:+.3f} = floor {fl['theta_floor']:+.3f} + shift {fl['delta']:+.3f} "
                     f"(exchangeable control {fl['delta_control']:+.3f})")
    if "blocks" in result:
        lines.append("blocks: " + ", ".join(f"K {r['block_size']}: sorted {r['sorted']['delta']:+.3f} / random {r['random']['delta']:+.3f}"
                                            for r in result["blocks"]["rows"]))
    if result["delta"] < 0:
        lines.append("negative shift: accumulating the declared classes adds fewer dimensions than random draws (class means that "
                     "concentrate the spectrum, or a path that withholds the dominant fluctuation, give this sign; read the split)")
    if verbose:
        for l in lines:
            print(l)
    return {"lines": lines, "verdict": verdict}


def build_axis(rows, text_field, label_field, n_classes=8, n_per_class=16,
               strata_field=None, classes=None, min_chars=20, max_chars=400, seed=0):
    """Turn labelled records into an axis JSON ({class: [prompts]}) and an
    optional strata sidecar. `rows` is any iterable of dicts (a Hugging Face
    dataset split works). Classes are the `n_classes` most frequent labels
    unless `classes` is given; `n_per_class` texts are drawn per class with a
    fixed seed; texts shorter than `min_chars` are skipped and longer than
    `max_chars` are cut at the last space before the limit."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    pool = defaultdict(list)
    cut = defaultdict(int)
    for r in rows:
        t = r.get(text_field); lab = r.get(label_field)
        if t is None or lab is None:
            continue
        t = str(t).strip()
        if len(t) < min_chars:
            continue
        was_cut = len(t) > max_chars
        if was_cut:
            head = t[:max_chars]
            t = head[:head.rfind(" ")] if " " in head else head
        pool[str(lab)].append((t, None if strata_field is None else str(r.get(strata_field)), was_cut))
    if classes is None:
        classes = [c for c, _ in sorted(pool.items(), key=lambda kv: -len(kv[1]))[:n_classes]]
    axis, strata = {}, {}
    for c in classes:
        items = pool.get(str(c), [])
        if len(items) < n_per_class:
            raise ValueError(f"class {c!r} has {len(items)} usable texts, fewer than n_per_class={n_per_class}")
        pick = rng.choice(len(items), n_per_class, replace=False)
        axis[str(c)] = [items[i][0] for i in pick]
        cut[str(c)] = int(sum(items[i][2] for i in pick))
        if strata_field is not None:
            strata[str(c)] = [items[i][1] for i in pick]
    if any(cut.values()):
        import warnings
        warnings.warn("build_axis: texts cut at max_chars (at the last space) per class: "
                      + ", ".join(f"{c} {k}" for c, k in cut.items())
                      + ". A cut text ends mid-sentence, and when the classes differ in how many are cut its last token "
                      "follows the class: raise max_chars, or keep only texts that fit.", stacklevel=2)
    return axis, (strata if strata_field is not None else None)


def axis_from_dataset(name, text_field, label_field, out_json, config=None, split="train",
                      n_classes=8, n_per_class=16, strata_field=None, classes=None,
                      min_chars=20, max_chars=400, seed=0):
    """`rung axis`: build an axis JSON (+ strata sidecar) from a Hugging
    Face dataset. Requires the `datasets` package (`pip install -e ".[models]"`)."""
    import json as _json
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    axis, strata = build_axis(ds, text_field, label_field, n_classes, n_per_class, strata_field, classes, min_chars, max_chars, seed)
    _json.dump(axis, open(out_json, "w"), indent=1)
    if strata is not None:
        side = out_json[:-5] + ".strata.json" if out_json.endswith(".json") else out_json + ".strata.json"
        _json.dump(strata, open(side, "w"), indent=1)
    print(f"wrote {out_json}: {len(axis)} classes x {n_per_class} prompts" + (" + strata sidecar" if strata is not None else ""))
    return axis, strata


# ----------------------------- command line -----------------------------

def _expand_axis_paths(paths):
    """Files, or directories (all *.json except INDEX.json and *.strata.json)."""
    import glob, os
    out = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(glob.glob(os.path.join(p, "*.json"))):
                b = os.path.basename(f)
                if b == "INDEX.json" or b.endswith(".strata.json"):
                    continue
                out.append(f)
        else:
            out.append(p)
    return out


def _load_axis(axis_file, use_strata=True):
    """axis_file: JSON {class_name: [prompt, ...]}. Optional sidecar
    <name>.strata.json with the same shape holding a nuisance stratum per
    prompt (topic, template, carrier id) for the second null."""
    import json as _json, os
    axes = _json.load(open(axis_file))
    prompts, labels, strata = [], [], []
    side = axis_file[:-5] + ".strata.json" if axis_file.endswith(".json") else None
    smap = _json.load(open(side)) if (use_strata and side and os.path.exists(side)) else None
    for ci, (cname, plist) in enumerate(axes.items()):
        for k, p in enumerate(plist):
            prompts.append(p)
            labels.append(ci)
            if smap is not None:
                strata.append(smap[cname][k])
    name = os.path.splitext(os.path.basename(axis_file))[0]
    return name, list(axes.keys()), prompts, np.array(labels), (np.array(strata) if smap is not None else None)


def frames_disagree(r_linear, r_z):
    """True when the order-averaged shift's excess over its null has opposite signs in the linear and the per-unit
    z-scored frame, each beyond one null SD (zoom outputs with n_perm > 0). There the linear reading follows a few
    high-variance units, and both frames are reported."""
    el = r_linear["delta_orderavg"] - r_linear["null_mean_orderavg"]
    ez = r_z["delta_orderavg"] - r_z["null_mean_orderavg"]
    return bool(el * ez < 0 and abs(el) > r_linear["null_sd_orderavg"] and abs(ez) > r_z["null_sd_orderavg"])


def last_token_report(last_ids, labels, decode=None):
    """Distinct last tokens per class (the read-out token of a last-token battery). A class whose prompts end on few
    distinct tokens starts its ladder on a low-dimensional rung at the first layers, so part of a shift there can be
    the last token's identity (Appendix I); when every class ends on the same token, the last token cannot carry the
    shift. Returns counts per class, the shared tokens, and warnings."""
    last_ids = np.asarray(last_ids); labels = np.asarray(labels)
    classes = np.unique(labels)
    distinct = [int(len(np.unique(last_ids[labels == c]))) for c in classes]
    shared = [int(t) for t in np.unique(last_ids) if len(np.unique(labels[last_ids == t])) > 1]
    vals, cnt = np.unique(last_ids, return_counts=True)
    one = len(vals) == 1
    warn = []
    if one:
        warn.append("every class ends on the same last token: the last token cannot carry the shift, and the "
                    "last-token-matched comparison cannot run")
    elif min(distinct) < 6:
        k = min(distinct)
        warn.append(f"a class ends on only {k} distinct last token{'s' if k != 1 else ''}: at the first layers its rung "
                    "spans few dimensions and part of the shift there can be the last token's identity; the matched "
                    "comparison (--match-last-token) needs at least 6 per class, so read the layers past the first few "
                    "or add prompts with varied endings")
    out = {"distinct_per_class": distinct, "n_shared_tokens": len(shared), "one_token_everywhere": bool(one),
           "top_token_share": float(cnt.max() / cnt.sum()), "warnings": warn}
    if decode is not None:
        out["examples"] = [[decode(int(t)) for t in np.unique(last_ids[labels == c])[:8]] for c in classes]
    return out


def last_token_arms(X, labels, last_ids, n_sub=20, n_perm=100, k_orders=20, seed=1500, standardize=False):
    """The last-token control of Appendix I for one layer: n_sub subsamples with m prompts per class ending on m
    distinct last tokens (m = the smallest class's count of distinct last tokens), and n_sub size-matched random
    subsamples with no token matching; the order-averaged shift and its permutation p for each. The lexical part of
    a lift is the drop from the random arm to the matched arm; if the random arm also loses the lift, the subsample
    is too small to tell. standardize reads the arms in the z-scored frame. The arms use ordinary label
    permutations: axis strata are not carried into them. Returns None when m < 6."""
    X = np.asarray(X); labels = np.asarray(labels); last_ids = np.asarray(last_ids)
    classes = np.unique(labels)
    m = min(len(np.unique(last_ids[labels == c])) for c in classes)
    if m < 6:
        return None
    rng = np.random.default_rng(seed)
    arms = {"matched": [], "random": []}
    for _ in range(n_sub):
        keep = []
        for c in classes:
            idx = np.where(labels == c)[0]
            for t in rng.permutation(np.unique(last_ids[idx]))[:m]:
                keep.append(rng.choice(idx[last_ids[idx] == t]))
        arms["matched"].append(np.sort(np.array(keep)))
        arms["random"].append(np.sort(np.concatenate([rng.choice(np.where(labels == c)[0], m, replace=False) for c in classes])))
    out = {"m_per_class": int(m), "frame": "z" if standardize else "linear"}
    for arm, subs in arms.items():
        vals = [zoom(X[k], labels[k], n_perm=n_perm, k_orders=k_orders, k_null_orders=k_orders,
                     seed=int(rng.integers(1 << 30)), standardize=standardize) for k in subs]
        out[arm] = {"delta_orderavg_median": float(np.median([v["delta_orderavg"] for v in vals])),
                    "p_two_orderavg_median": float(np.median([v["p_two_orderavg"] for v in vals]))}
    return out


def llm_battery(model_name, axis_files, device="cpu", n_perm=500, k_orders=50,
                max_len=128, out_path=None, paper_seeds=False, revision=None,
                use_strata=True, standardize=False, dtype=None, frames="both", match_last_token=False):
    """Per-layer zoom() over last-token hidden states for one or more axes.

    axis_files: list of axis JSON paths and/or directories. The model is loaded
    once (optionally at a Hugging Face `revision`, e.g. a Pythia checkpoint
    "step10000") and every axis is encoded and analyzed. If <axis>.strata.json
    exists next to an axis file, the stratified (nuisance-preserving) null is
    computed as well (and, when its strata pair classes, the pair-preserving
    order average). frames: "linear" (the paper's estimator), "z" (every unit
    z-scored first) or "both" (default): with both, the order-averaged shift
    and its null are reported in each frame and layers where the two disagree
    in sign beyond one null SD are flagged (a few high-variance units can set
    the linear sign). Every axis gets a last-token report; match_last_token
    adds the matched and size-matched random arms per layer. Returns
    {"model", "revision", "axes": {name: {...}}}.
    """
    import json as _json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    files = _expand_axis_paths(list(axis_files))
    print(f"{len(files)} axis file(s); loading {model_name}"
          f"{' @ ' + revision if revision else ''} on {device}...", flush=True)
    # dtype: None = float16 on mps/cuda, float32 on cpu; "bf16" repairs float16 overflow in early training
    # checkpoints, "fp32" is the reference
    dtype = {None: (torch.float16 if device in ("mps", "cuda") else torch.float32), "fp16": torch.float16,
             "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
    model = _from_pretrained(AutoModelForCausalLM, model_name, dtype, revision).to(device)
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision) if revision \
        else AutoTokenizer.from_pretrained(model_name)
    model.eval()
    if frames not in ("linear", "z", "both"):
        raise ValueError("frames is 'linear', 'z' or 'both'")
    frame_list = {"linear": [False], "z": [True], "both": [False, True]}[frames]
    if standardize:
        frame_list = [True]

    results = {"model": model_name, "revision": revision, "n_perm": n_perm, "rung_version": __version__,
               "k_orders": k_orders, "frames": "z" if standardize else frames, "axes": {}}
    for af in files:
        name, classes, prompts, labels, strata = _load_axis(af, use_strata)
        print(f"[{name}] {len(classes)} classes, {len(prompts)} prompts"
              f"{' + strata' if strata is not None else ''}", flush=True)
        states, last = [], []
        n_tok = np.array([len(tok(p).input_ids) for p in prompts])
        trunc = [int(((n_tok > max_len) & (labels == c)).sum()) for c in range(len(classes))]
        if any(trunc):
            print(f"  WARNING: prompts longer than --max-len {max_len} tokens are cut from the right, so their read-out token is "
                  f"not their last; per class {trunc}: raise --max-len or shorten the prompts", flush=True)
        with torch.no_grad():
            for i, p in enumerate(prompts):
                ids = tok(p, return_tensors="pt", truncation=True,
                          max_length=max_len).input_ids.to(device)
                last.append(int(ids[0, -1]))
                hs = model(ids, output_hidden_states=True).hidden_states
                states.append([h[0, -1, :].float().cpu().numpy() for h in hs])
                if i and i % 50 == 0:
                    print(f"    encoded {i}/{len(prompts)}", flush=True)
        ltr = last_token_report(last, labels, decode=lambda t: tok.decode([t]))
        print(f"  last tokens: distinct per class {ltr['distinct_per_class']}", flush=True)
        for w in ltr["warnings"]:
            print(f"  WARNING: {w}", flush=True)
        keys = ("delta", "p_two", "z", "null_mean", "null_sd", "delta_orderavg",
                "delta_orderavg_sd", "p_two_orderavg", "z_orderavg",
                "null_mean_orderavg", "null_sd_orderavg", "theta_floor", "theta_floor_se",
                "strat_p_two", "strat_z", "strat_null_mean", "strat_null_sd",
                "strat_p_two_orderavg", "strat_z_orderavg", "strat_null_mean_orderavg", "strat_null_sd_orderavg",
                "delta_pairavg", "pairavg_p_two", "pairavg_z", "pairavg_null_mean", "pairavg_null_sd")
        layers = []
        for l in range(len(states[0])):
            X = np.stack([s[l] for s in states])
            row = {"layer": l}
            for zf in frame_list:
                if paper_seeds:
                    r = zoom(X, labels, n_perm=n_perm, k_orders=k_orders, strata=strata,
                             floor_seed=100 * l + 1, perm_seed=3700, order_seed=3701, standardize=zf)
                else:
                    r = zoom(X, labels, n_perm=n_perm, k_orders=k_orders, strata=strata, seed=l, standardize=zf)
                suffix = "_z" if (zf and len(frame_list) == 2) else ""
                row.update({k + suffix: r[k] for k in keys if k in r})
            if r.get("degenerate"):
                row["degenerate"] = r["degenerate"]
                layers.append(row)
                print(f"    L{l:02d} degenerate ({r['degenerate']}): every prompt's state is the same here; no reading", flush=True)
                continue
            msg = (f"    L{l:02d} delta={row['delta']:+.3f} p={row.get('p_two', 1):.4f}"
                   f" | avg={row['delta_orderavg']:+.3f} p={row.get('p_two_orderavg', 1):.4f}")
            if "delta_orderavg_z" in row:
                msg += f" | z-frame avg={row['delta_orderavg_z']:+.3f} p={row.get('p_two_orderavg_z', 1):.4f}"
                if "null_mean_orderavg" in row and "null_mean_orderavg_z" in row:
                    row["frames_disagree"] = frames_disagree(
                        {k: row[k] for k in ("delta_orderavg", "null_mean_orderavg", "null_sd_orderavg")},
                        {k: row[k + "_z"] for k in ("delta_orderavg", "null_mean_orderavg", "null_sd_orderavg")})
                    if row["frames_disagree"]:
                        msg += "  <- FRAMES DISAGREE in sign"
            if "strat_p_two" in row:
                msg += f" | strat p={row['strat_p_two']:.4f}"
            if "delta_pairavg" in row:
                msg += f" | pair-avg={row['delta_pairavg']:+.3f} p={row.get('pairavg_p_two', 1):.4f}"
            if match_last_token:
                # the arms read the battery's first frame (linear unless only z was asked) at its budget, capped at
                # the control's own 100 permutations and 20 orders
                arms = last_token_arms(X, labels, last, n_perm=min(n_perm, 100), k_orders=min(k_orders, 20),
                                       standardize=frame_list[0])
                row["last_token_arms"] = arms
                if arms is not None:
                    msg += (f" | matched {arms['matched']['delta_orderavg_median']:+.3f} vs random "
                            f"{arms['random']['delta_orderavg_median']:+.3f} (m={arms['m_per_class']})")
            layers.append(row)
            print(msg, flush=True)
        results["axes"][name] = {"classes": classes, "n_prompts": len(prompts), "truncated_per_class": trunc,
                                 "stratified": strata is not None, "last_token": ltr, "layers": layers}
        if any("frames_disagree" in r for r in layers):
            nd = sum(r.get("frames_disagree", False) for r in layers)
            results["axes"][name]["frames_disagree_layers"] = nd
            if nd:
                print(f"  WARNING: the linear and z-scored frames disagree in sign at {nd} layer(s): there the linear "
                      "reading follows a few high-variance units; report both", flush=True)
    if out_path:
        _json.dump(results, open(out_path, "w"), indent=1)
        print(f"wrote {out_path}", flush=True)
    return results


def _plot_battery(path, out_png):
    """One panel per axis from a `rung llm` JSON: declared-path delta,
    order-averaged deltabar, and their permutation-null bands (plus the
    stratified-null band when present)."""
    import json as _json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _json.load(open(path))
    axes_d = d["axes"] if "axes" in d else {"battery": {"layers": d["layers"], "classes": d.get("classes", [])}}
    n = len(axes_d)
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 3.6), squeeze=False)
    for ax, (name, a) in zip(axs[0], axes_d.items()):
        n_all = len(a["layers"])
        L = [l for l in a["layers"] if not l.get("degenerate")]      # a layer with no variance has no reading
        x = np.array([l["layer"] for l in L]) / max(n_all - 1, 1)
        def arr(k):
            return np.array([l[k] for l in L]) if k in L[0] else None
        for mk, sk, col, al, lab in (("null_mean", "null_sd", "#c0392b", 0.12, "declared-path null"),
                                     ("null_mean_orderavg", "null_sd_orderavg", "#2c3e50", 0.18, "order-averaged null"),
                                     ("strat_null_mean", "strat_null_sd", "#8e44ad", 0.18, "stratified null")):
            m, s = arr(mk), arr(sk)
            if m is not None and s is not None:
                ax.fill_between(x, m - 2 * s, m + 2 * s, color=col, alpha=al, lw=0,
                                label=f"{lab} ($\\pm 2$ SD)")
        ax.plot(x, arr("delta"), "-o", ms=3, color="#c0392b", label="declared path $\\delta$")
        ax.plot(x, arr("delta_orderavg"), "--s", ms=3, color="#2c3e50", label="order-averaged $\\bar\\delta$")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("normalized depth"); ax.set_ylabel("$\\delta$")
        ax.set_title(f"{d.get('model', '')}{' @ ' + d['revision'] if d.get('revision') else ''}\n{name}",
                     fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print(f"wrote {out_png}", flush=True)



def _bh_mask(ps, q=0.05):
    """Benjamini-Hochberg: which of the p-values are certified at level q."""
    ps = np.asarray(ps, float); n = len(ps)
    if n == 0:
        return np.zeros(0, bool)
    order = np.argsort(ps); thresh = q * np.arange(1, n + 1) / n
    passed = ps[order] <= thresh
    if not passed.any():
        return np.zeros(n, bool)
    k = int(np.max(np.where(passed)[0])) + 1
    return ps <= thresh[k - 1]


def _frame_class(layer):
    """How the two feature frames read one layer's order-averaged shift: 'agree' or 'disagree' in sign when each excess over
    its own null exceeds one null SD, 'one' when only one frame does, 'neither' when neither does; None without both frames."""
    keys = ("delta_orderavg", "null_mean_orderavg", "null_sd_orderavg")
    if not all(k in layer and k + "_z" in layer for k in keys):
        return None
    el = layer["delta_orderavg"] - layer["null_mean_orderavg"]
    ez = layer["delta_orderavg_z"] - layer["null_mean_orderavg_z"]
    bl, bz = abs(el) > layer["null_sd_orderavg"], abs(ez) > layer["null_sd_orderavg_z"]
    if bl and bz:
        return "agree" if el * ez > 0 else "disagree"
    return "one" if (bl or bz) else "neither"


def _bh_count(ps, q=0.05):
    ps = np.asarray(ps, float); n = len(ps)
    if n == 0:
        return 0
    order = np.argsort(ps); thresh = q * np.arange(1, n + 1) / n
    passed = ps[order] <= thresh
    if not passed.any():
        return 0
    k = int(np.max(np.where(passed)[0])) + 1
    return int((ps <= thresh[k - 1]).sum())


def summarize(path, out_json=None, verbose=True):
    """Apply the paper's reading rules to a `rung llm` JSON and print a
    plain-language verdict per axis. Rules: certification counts at two-sided
    p<0.05 and after Benjamini-Hochberg (q<0.05) for the declared path and the
    order-averaged statistic; declared-path profile shape (embedding sign, peak,
    zero crossing); when a stratified null exists, whether the declared-path
    signal survives it (label-linked) or is absorbed (nuisance composition)."""
    import json as _json
    d = _json.load(open(path))
    axes_d = d["axes"] if "axes" in d else {"battery": {"layers": d["layers"]}}
    report = {"model": d.get("model"), "revision": d.get("revision"), "axes": {}}
    n_perm = d.get("n_perm", 500)
    floor_p = 1.0 / (n_perm + 1)
    for name, a in axes_d.items():
        n_degenerate = sum(1 for l in a["layers"] if l.get("degenerate"))
        L = [l for l in a["layers"] if not l.get("degenerate")]; n = len(L)
        if n == 0:
            report["axes"][name] = {"reading": ["every layer is degenerate (no variance)"]}
            continue
        bh_possible = floor_p <= 0.05 / n
        pd_ = [l.get("p_two", 1.0) for l in L]; pa = [l.get("p_two_orderavg", 1.0) for l in L]
        kd, kd_bh = sum(p < 0.05 for p in pd_), _bh_count(pd_)
        ka, ka_bh = sum(p < 0.05 for p in pa), _bh_count(pa)
        dl = np.array([l["delta"] for l in L]); da = np.array([l["delta_orderavg"] for l in L])
        emb, fin = dl[0], dl[-1]
        exc = dl - emb; peak_l = int(np.argmax(exc)); peak = float(exc[peak_l])
        cross = next((i for i in range(1, n) if np.sign(dl[i]) != np.sign(emb) and dl[i] != 0), None)
        lines = []
        # partition-level reading; certified layers are counted by the sign of their shift (a two-sided test certifies both)
        pa_ = np.asarray(pa, float); raw_m = pa_ < 0.05; bh_m = _bh_mask(pa_)
        kp, kn = int((raw_m & (da > 0)).sum()), int((raw_m & (da < 0)).sum())
        kp_bh, kn_bh = int((bh_m & (da > 0)).sum()), int((bh_m & (da < 0)).sum())
        counts = (f"order-averaged statistic certified positive at {kp}/{n} layers ({kp_bh}/{n} after BH) and negative at "
                  f"{kn}/{n} ({kn_bh}/{n} after BH), mean deltabar {da.mean():+.3f}")
        if kp_bh / n >= 0.5 and kp_bh > kn_bh:
            lines.append(f"label-linked at most depths, the declared classes lifting the exponent: {counts}.")
        elif kn_bh / n >= 0.5 and kn_bh > kp_bh:
            lines.append(f"label-linked at most depths, the declared classes lowering the exponent: {counts}.")
        elif kp_bh + kn_bh > 0:
            lines.append(f"weakly label-linked{', with mixed signs' if kp_bh and kn_bh else ''}: {counts}.")
        else:
            lines.append(f"no certified label linkage under the order-free statistic ({ka}/{n} raw, 0 after BH); "
                         f"treat this partition as unorganized at this design.")
        # declared-path shape
        shape = (f"declared path: embedding {emb:+.3f} -> final {fin:+.3f}, peak excess {peak:+.3f} at layer {peak_l}"
                 f"/{n-1}, certified {kd}/{n} ({kd_bh}/{n} after BH)")
        if emb < 0 and fin > 0:
            shape += (f"; negative-to-positive crossover at layer {cross}: a PATH property (the order-averaged "
                      f"line is the partition-level claim; the crossover shape depends on the class order).")
        elif emb > 0 and fin < emb - 0.02:
            shape += "; positive at the embedding and declining, content-like: organization inherited from tokens, diluted with depth."
        elif abs(emb) < 0.02 and abs(fin) < 0.02 and kd_bh == 0:
            shape += "; flat near zero along the declared path."
        else:
            shape += "."
        lines.append(shape)
        # stratified null
        if "strat_p_two" in L[0]:
            ps_ = [l["strat_p_two"] for l in L]; ks, ks_bh = sum(p < 0.05 for p in ps_), _bh_count(ps_)
            tag = "after BH" if bh_possible else "after BH, at a coarse permutation floor"
            if kd > 0 and ks_bh == 0:
                lines.append(f"stratified null absorbs the declared-path signal ({ks}/{n} layers survive; {tag}): "
                             f"consistent with the ordinary floor reading nuisance composition rather than the label.")
            elif ks_bh > 0:
                lines.append(f"survives the stratified null at {ks}/{n} layers ({ks_bh}/{n} {tag}): "
                             f"label-linked beyond the declared nuisance structure.")
            else:
                lines.append(f"stratified null: nothing to absorb ({ks}/{n} layers).")
        else:
            lines.append("no strata given: the ordinary floor cannot separate label from nuisance composition; "
                         "add <axis>.strata.json (topics/templates/carriers) before reading this as representation.")
        if n_degenerate:
            lines.append(f"{n_degenerate} layer(s) skipped as degenerate: every prompt's state is the same there (a shared read-out token)")
        lt = a.get("last_token") or {}
        for w in lt.get("warnings", []):
            lines.append("final tokens: " + w)
        arms = [l["last_token_arms"] for l in L if l.get("last_token_arms")]
        if arms:
            mt = [x["matched"]["delta_orderavg_median"] for x in arms]; rd = [x["random"]["delta_orderavg_median"] for x in arms]
            lines.append(f"final-token control: matched {mt[0]:+.3f} / size-matched {rd[0]:+.3f} at the first layer, "
                         f"{mt[-1]:+.3f} / {rd[-1]:+.3f} at the last (m = {arms[0]['m_per_class']} per class); the lexical part of a "
                         "lift is the drop from size-matched to matched")
        frame_cls = [_frame_class(l) for l in L]
        if any(c is not None for c in frame_cls):
            cnt = {k: [i for i, c in enumerate(frame_cls) if c == k] for k in ("agree", "disagree", "one", "neither")}
            txt = (f"frames (each order-averaged shift against its own null, beyond one null SD): the linear and per-unit z-scored "
                   f"readings agree in sign at {len(cnt['agree'])}/{n} layers, disagree at {len(cnt['disagree'])}")
            if cnt["one"]:
                txt += (f", and one frame sits inside its null at {len(cnt['one'])} (layers "
                        + ", ".join(str(L[i].get("layer", i)) for i in cnt["one"]) + ")")
            if cnt["neither"]:
                txt += f"; both inside their nulls at {len(cnt['neither'])}"
            if cnt["disagree"] or cnt["one"]:
                txt += "; where the frames disagree or only one separates, report both (the linear reading can follow a few high-variance units)"
            lines.append(txt)
        if bh_possible:
            lines.append(f"caveat: per-layer counts are descriptive (~{0.05*n:.1f} false positives expected at p<0.05); "
                         f"the BH counts and the order-averaged statistic carry the inference.")
        else:
            lines.append(f"RESOLUTION WARNING: n_perm={n_perm} gives a p floor of {floor_p:.4f}, above the BH threshold "
                         f"{0.05/n:.4f} for {n} layers: BH can certify layers only when many share small p-values, so the BH "
                         f"counts above understate what a finer null would resolve. "
                         f"Rerun with --n-perm 500 (the paper's setting) or more before drawing conclusions.")
        report["axes"][name] = {"n_perm": n_perm, "bh_resolvable": bh_possible,
                                "certified_declared": [kd, kd_bh], "certified_orderavg": [ka, ka_bh],
                                "certified_orderavg_by_sign": {"positive": [kp, kp_bh], "negative": [kn, kn_bh]},
                                "frames": {k: sum(c == k for c in frame_cls) for k in ("agree", "disagree", "one", "neither")},
                                "embedding": float(emb), "final": float(fin), "peak_excess": peak,
                                "peak_layer": peak_l, "crossover_layer": cross, "reading": lines}
        if verbose:
            hdr = f"== {report['model'] or ''}{' @ ' + report['revision'] if report.get('revision') else ''} :: {name} ({n} layers) =="
            print(hdr); [print("  - " + s) for s in lines]
    if out_json:
        _json.dump(report, open(out_json, "w"), indent=1)
    return report


# ---- One page per result ----

def _result_kind(d):
    """'heads' (`rung llm --heads`), 'llm' (a battery), 'features' (a feature ladder) or 'data' (a zoom result): the
    dispatch of `rung plot` and `rung summarize`, plus head scores."""
    if "heads" in d and isinstance(d["heads"], list):
        return "heads"
    if "layers" in d or "axes" in d:
        return "llm"
    if "mode" in d and "pr_obs" in d and "rung_sizes" not in d:
        return "features"
    return "data"


def _fmt(v):
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    return "nan" if not np.isfinite(v) else f"{v:.6g}"


def _plot_heads(d, out_png):
    """Every head's permutation z as a layer-by-head map; heads named as known (e.g. found by path patching) outlined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hs = d["heads"]
    L, I = max(h["layer"] for h in hs) + 1, max(h["index"] for h in hs) + 1
    Z = np.full((L, I), np.nan)
    for h in hs:
        Z[h["layer"], h["index"]] = h["z"]
    fig, ax = plt.subplots(figsize=(min(8.6, 1.6 + 0.42 * I), min(6.0, 1.2 + 0.32 * L)))
    v = float(np.nanmax(np.abs(Z))) or 1.0
    im = ax.imshow(Z, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto", origin="lower")
    for name in d.get("known", []):
        l_, i_ = (int(x) for x in name.split("."))
        ax.add_patch(plt.Rectangle((i_ - 0.5, l_ - 0.5), 1, 1, fill=False, lw=1.6, ec="0.1"))
    ax.set_xlabel("head"); ax.set_ylabel("layer")
    ax.set_title("permutation z of each head's write along the axis" + (" (outlined: known heads)" if d.get("known") else ""), fontsize=9)
    fig.colorbar(im, ax=ax, label="z", shrink=0.85)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def _heads_lines(d, top=10):
    hs = sorted(d["heads"], key=lambda h: -h["z"])
    lines = [f"{len(hs)} heads of {d.get('model')} scored along {d.get('axis')} ({d.get('n_perm')} label permutations, "
             f"{d.get('k_orders')} random class orders each); ranked by z: "
             + ", ".join(f"{h['head']} {h['z']:+.1f}" for h in hs[:top])]
    if d.get("known"):
        aucs = ", ".join(f"{k[4:]} {v:.2f}" for k, v in d.items() if k.startswith("AUC_"))
        lines.append(f"against the {len(d['known'])} known heads, AUC (0.5 = chance): {aucs}")
    if all("within" in h for h in hs[:top]):
        lines.append("of the top heads, " + ", ".join(f"{h['head']} {'fluctuations' if h['within'] > h['class_mean'] else 'class means'}"
                                                     for h in hs[:top]) + " carry most of the shift (the split: which part is larger)")
    lines.append("a training-free pre-screen: test the top heads causally (patching or ablation) before calling them a circuit")
    return lines


def _headline(d, kind):
    """(label, value) rows for the numbers a reader quotes, each with the setting that qualifies it."""
    rows = []
    if kind == "heads":
        rows += [("model", f"{d.get('model')}" + (f" @ {d['revision']}" if d.get("revision") else "")),
                 ("axis", f"{d.get('axis')}: {len(d.get('classes', []))} classes, {d.get('n_prompts')} prompts"),
                 ("label permutations per head", f"{d.get('n_perm')} (each with {d.get('k_orders')} random class orders)"),
                 ("seeds", "the paper's (Appendix I)" if d.get("paper_seeds") else "default")]
        rows += [(f"AUC of {k[4:]} against the known heads", f"{v:.3f}") for k, v in d.items() if k.startswith("AUC_")]
        return rows
    if kind == "llm":
        axes_ = d["axes"] if "axes" in d else {"battery": d}
        rows += [("model", f"{d.get('model')}" + (f" @ {d['revision']}" if d.get("revision") else "")),
                 ("axes", ", ".join(axes_)), ("layers", str(max(len(a_["layers"]) for a_ in axes_.values()))),
                 ("label permutations per layer", str(d.get("n_perm"))), ("random class orders", str(d.get("k_orders")))]
        return rows
    se = d.get("theta_floor_se")
    rows.append(("theta_obs = floor + shift", f"{d['theta_obs']:+.3f} = {d['theta_floor']:+.3f} {'-' if d['delta'] < 0 else '+'} {abs(d['delta']):.3f}"))
    if se is not None and np.isfinite(se):
        rows.append(("the floor's standard error", f"{se:.3f} ({d.get('n_floor_draws')} draws); a shift inherits it"))
    if kind == "features":
        rows.append(("exchangeable control (random in both arms)", f"{d['delta_control']:+.3f}"))
        if "p_two" in d:
            rows.append(("against relabellings of the features", f"z = {d['z']:+.1f}, two-sided p = {d['p_two']:.3f} ({d['n_null']} draws)"))
        rows.append(("design", f"rungs {d['sizes']} of {d['n_features']} features, {d['n_samples']} samples"))
        return rows
    if "p_two" in d and d.get("n_perm"):
        rows.append(("declared order against label permutations",
                     f"z = {d['z']:+.1f}, two-sided p = {d['p_two']:.3f} ({d['n_perm']} permutations; p cannot fall below "
                     f"{1.0 / (d['n_perm'] + 1):.3f})"))
    if "delta_orderavg" in d and np.isfinite(d["delta_orderavg"]):
        txt = f"{d['delta_orderavg']:+.3f} (SD over {d.get('k_orders')} random orders {d['delta_orderavg_sd']:.3f})"
        if "p_two_orderavg" in d:
            txt += f", p = {d['p_two_orderavg']:.3f}"
        rows.append(("order-averaged shift", txt))
    if "decompose" in d:
        f = d["decompose"]["delta_split_free"]
        rows.append(("within-class part D", f"{f['D']:+.3f}"))
        rows.append(("class-mean part Tb + Cb + Cx", f"{f['Tb'] + f['Cb'] + f['Cx']:+.3f} (decompose's own floor draws)"))
    rows.append(("design", f"{d.get('n_samples')} samples by {d.get('n_features')} features, {d.get('n_classes')} classes, "
                           f"rungs {d.get('rung_sizes')}"))
    return rows


def _plot_summary(d, out_png):
    """A `rung data` result at a glance, from the numbers it stored (nothing recomputed): the declared and order-averaged
    shifts against their label-permutation nulls (mean +- 2 SD; the order-averaged one with its SD over random orders),
    the split into the within-class and class-mean parts, and the class-mean harmonics when sectors were run. Drawn at
    the report's display width (8.6 in at 100 px/in), so type is true size. Returns False when the result holds none."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = []
    if "null_mean" in d and np.isfinite(d.get("null_sd", np.nan)):
        panels.append("nulls")
    if "decompose" in d:
        panels.append("split")
    if isinstance(d.get("sectors"), dict) and "named" in d["sectors"]:
        panels.append("sectors")
    if not panels:
        return False
    fig, axes = plt.subplots(1, len(panels), figsize=(8.6 * len(panels) / 3, 2.9), squeeze=False)
    red, grey, blue = "#c23b3b", "0.62", "#2f5d8a"
    for ax, kind in zip(axes[0], panels):
        if kind == "nulls":
            rows = [("declared", d["delta"], d["null_mean"], d["null_sd"], None)]
            if np.isfinite(d.get("null_sd_orderavg", np.nan)):
                rows.append(("order-\naveraged", d["delta_orderavg"], d["null_mean_orderavg"], d["null_sd_orderavg"], d.get("delta_orderavg_sd")))
            for i, (lab, v, m, sd, osd) in enumerate(rows):
                ax.barh(i, 4 * sd, left=m - 2 * sd, height=0.4, color=grey, alpha=0.55)
                ax.errorbar(v, i, xerr=osd if osd else None, fmt="o", color=red, capsize=3, ms=5)
            ax.set_yticks(range(len(rows))); ax.set_yticklabels([r_[0] for r_ in rows])
            ax.axvline(0, color="0.3", lw=0.6); ax.set_ylim(-0.7, len(rows) - 0.3)
            ax.set_xlabel("shift (grey: null, mean $\\pm$ 2 SD;\nbar: SD over random orders)", fontsize=8)
            ax.set_title("shift vs label permutations")
        elif kind == "split":
            f = d["decompose"]["delta_split_free"]
            vals = [f["D"], f["Tb"] + f["Cb"] + f["Cx"], d["decompose"]["delta"]]
            ax.bar([0, 1, 2], vals, color=[red, blue, "none"], edgecolor=[red, blue, "0.2"], width=0.6)
            ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["within-\nclass D", "class\nmeans", "shift"])
            ax.axhline(0, color="0.3", lw=0.6)
            lo, hi = min(0, *vals), max(0, *vals); pad = 0.12 * (hi - lo or 1)
            ax.set_ylim(lo - pad, hi + pad)
            for x_, v in enumerate(vals):
                ax.text(x_, v + (0.02 if v >= 0 else -0.02) * (hi - lo or 1), f"{v:+.3f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8)
            ax.set_title("what carries the shift")
        else:
            nm = d["sectors"]["named"]
            keys = [k for k in ("a", "c1", "b2", "c3", "b4") if k in nm]
            ax.bar(range(len(keys)), [nm[k] for k in keys], width=0.6,
                   color=[grey if k == "a" else (red if k.startswith("c") else blue) for k in keys])
            ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys); ax.axhline(0, color="0.3", lw=0.6)
            ax.set_xlabel("red: odd (direction)\nblue: even (orientation)", fontsize=8)
            ax.set_title("class-mean harmonics")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return True


def report(path, out_html=None, title=None):
    """One self-contained HTML page for a rung result JSON (`rung data`, `rung features` or `rung llm`): the figure that
    `rung plot` draws, the reading that `rung summarize` prints, the numbers a reader quotes with their settings, and
    every scalar the run returned. Nothing is recomputed. Returns the path written."""
    import base64
    import html
    import json as _json
    import os
    import tempfile
    d = _json.load(open(path))
    kind = _result_kind(d)
    out_html = out_html or os.path.splitext(path)[0] + ".html"
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "figure.png")
        if kind == "heads":
            _plot_heads(d, png)
            reading = [("", _heads_lines(d))]
        elif kind == "llm":
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):   # _plot_battery announces its (temporary) file
                _plot_battery(path, png)
            reading = [(name, a_["reading"]) for name, a_ in summarize(path, verbose=False)["axes"].items()]
        elif kind == "features":
            _plot_features(d, png)
            reading = [("", _feature_ladder_lines(d))]
        else:
            _plot_data(d, png)
            reading = [("", summarize_data(d, verbose=False)["lines"])]
        img = base64.b64encode(open(png, "rb").read()).decode() if os.path.exists(png) else None
        png2 = os.path.join(tmp, "summary.png")
        img2 = base64.b64encode(open(png2, "rb").read()).decode() if kind == "data" and _plot_summary(d, png2) else None
    e = html.escape
    name = os.path.basename(path)
    title = title or f"rung: {name}"
    scal = {k: v for k, v in d.items() if not isinstance(v, (list, dict))}
    if kind == "data" and "decompose" in d:
        scal.update({f"decompose.{k}": v for k, v in d["decompose"]["delta_split_free"].items() if not isinstance(v, (list, dict))})
    parts = [f"<h1>{e(title)}</h1>",
             f"<p class=meta>{e(name)} &middot; {e({'heads': 'an attention-head', 'llm': 'a language-model', 'features': 'a feature-ladder', 'data': 'a data'}[kind])} "
             f"result &middot; made by rung {e(str(d.get('rung_version', '(version not recorded)')))}; page by rung {e(__version__)}</p>"]
    if img:
        parts.append(f'<figure><img alt="the result\'s figure" src="data:image/png;base64,{img}"></figure>')
    if img2:
        parts.append(f'<figure><img alt="the shift against its nulls, its split and harmonics" src="data:image/png;base64,{img2}"></figure>')
    parts.append("<h2>Reading</h2>")
    for head, lines in reading:
        if head:
            parts.append(f"<h3>{e(head)}</h3>")
        parts.append("<ul>" + "".join(f"<li>{e(line.strip())}</li>" for line in lines) + "</ul>")
    parts.append("<h2>Numbers</h2><table>" + "".join(f"<tr><th>{e(a_)}</th><td>{e(b_)}</td></tr>" for a_, b_ in _headline(d, kind)) + "</table>")
    parts.append("<details><summary>Every number the run returned</summary><table>"
                 + "".join(f"<tr><th>{e(k)}</th><td>{e(_fmt(v))}</td></tr>" for k, v in sorted(scal.items())) + "</table></details>")
    parts.append('<p class=meta>Reading rules: <a href="https://github.com/adilamin89/structure-in-the-zoom#reading-the-numbers">'
                 "README, Reading the numbers</a>. The paper: Structure is in the zoom (NeurReps 2026).</p>")
    css = """:root{--bg:#fff;--fg:#1f2328;--mute:#59636e;--line:#d1d9e0;--head:#f6f8fa}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mute:#9198a1;--line:#3d444d;--head:#151b23}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;margin:0}
main{max-width:860px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
h3{font-size:15px;margin:16px 0 4px}.meta{color:var(--mute);font-size:13px}figure{margin:16px 0}
img{max-width:100%;height:auto;max-height:440px;background:#fff;border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
th{background:var(--head);font-weight:600;width:38%}td{font-variant-numeric:tabular-nums;overflow-wrap:anywhere}
li{margin:4px 0}details{margin:20px 0}summary{cursor:pointer;color:var(--mute)}a{color:#0969da}"""
    page = (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{e(title)}</title><style>{css}</style></head><body><main>{''.join(parts)}</main></body></html>")
    with open(out_html, "w") as fh:
        fh.write(page)
    return out_html


# ---- The four-term split of a rung's deficit (Section 5 and Appendix K) ----
# For trials Y (n x N) with class labels, C = B + W (the between-class covariance of the class means about the
# subset mean, weighted by occupancy; the pooled within-class covariance) and
#     log PR = 2 log(b1 + w1) - log(b2 + w2 + 2x),   b1 = Tr B, w1 = Tr W, b2 = Tr B^2, w2 = Tr W^2, x = Tr(BW).
# The deficit of a ladder rung below its floor (ladder minus floor, the floor's log terms averaged over random
# same-size draws) splits EXACTLY into A = 2 [log(b1 + w1)] (trace), P = -[log w2] (fluctuation pooling),
# Cb = -[log(1 + b2/w2)] (between-class) and Cx (the mean-fluctuation coupling); the shift delta, the slope of the
# deficit against log n, splits into the same four slopes. A and P each carry the within-class scale
# S = 2 Delta log w1 with opposite signs, so they regroup exactly as A + P = D + Tb with D = Delta[2 log w1 - log w2]
# (the pooled within-class effective dimension, scale-free) and Tb = 2 Delta log(1 + b1/w1) (the trace share of the
# class means). The private slope is the slope of P (or D) against log(k/K) over the rungs below the top one: 1 when
# every class brings its own fluctuation subspace, 0 when the variability recurs across classes. Label permutations
# (n_shuffle) give each term's sampling baseline; `excess` is observed minus that mean.
FREE = ("D", "Tb", "Cb", "Cx")


def traces(Y, lab):
    """The five traces of C = B + W for trials Y (n x N) with labels lab."""
    Y = np.asarray(Y, np.float64); n = Y.shape[0]
    xbar = Y.mean(0)
    classes = np.unique(lab)
    Mm = np.stack([np.sqrt((lab == c).sum() / n) * (Y[lab == c].mean(0) - xbar) for c in classes])
    R = np.empty_like(Y)
    for c in classes:
        m = lab == c
        R[m] = Y[m] - Y[m].mean(0)
    R /= np.sqrt(n)
    b1 = float((Mm ** 2).sum()); w1 = float((R ** 2).sum())
    b2 = float(((Mm @ Mm.T) ** 2).sum()); w2 = float(((R @ R.T) ** 2).sum()); x = float(((Mm @ R.T) ** 2).sum())
    return b1, w1, b2, w2, x


def logterms(t):
    b1, w1, b2, w2, x = t
    return {"lt": 2 * np.log(b1 + w1), "lw2": np.log(w2), "lb": np.log1p(b2 / w2), "lbx": np.log1p((b2 + 2 * x) / w2),
            "ld": 2 * np.log(w1) - np.log(w2), "tb": 2 * np.log1p(b1 / w1), "lw1": 2 * np.log(w1),
            "pr": (b1 + w1) ** 2 / (b2 + w2 + 2 * x), "traces": [float(v) for v in t]}


def pr_centered(X):
    """The centered participation ratio (trial-space Gram), the paper's estimator."""
    Xc = np.asarray(X, np.float64); Xc = Xc - Xc.mean(0)
    G = Xc @ Xc.T
    tr, tr2 = float(np.trace(G)), float((G * G).sum())
    return tr * tr / tr2 if tr2 > 0 else 1.0


def _lsq_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def class_pr(X, labels):
    """Each class's own fluctuation dimension: the participation ratio of its residuals around its mean, {class: PR}.
    The sampling threshold of Section 2 compares a class's sample count with this number; PR(W), which pools the
    classes, can sit above or below it."""
    X = np.asarray(X, np.float64); labels = np.asarray(labels)
    out = {}
    for c in np.unique(labels):
        Z = X[labels == c] - X[labels == c].mean(0)
        G = Z @ Z.T if len(Z) <= Z.shape[1] else Z.T @ Z
        t2 = float((G * G).sum())
        out[int(c) if np.issubdtype(type(c), np.integer) else str(c)] = float(np.trace(G) ** 2 / t2) if t2 > 0 else float("nan")
    return out


def sign_law(X, labels):
    """The sign of the class-mean part, read from the full set (Section 5, Eq. 3; Appendix K). With beta = Tr B / Tr W,
    r = PR(W) / PR(B) and the alignment a = xi PR(W), xi = Tr(WB) / (Tr W Tr B) (how much the class means lie along
    the dominant fluctuations), the class-mean term of the full set, c = log[(1 + beta)^2 / (1 + beta^2 r + 2 a beta)],
    is negative exactly when r > 1 + 2 (1 - a) / beta. The x-free threshold 1 + 2 / beta (a = 0) is sufficient for a
    negative term when the alignment is positive, not the rule."""
    X = np.asarray(X, np.float64); labels = np.asarray(labels)
    Xc = X - X.mean(0)
    classes = np.unique(labels)
    n = len(Xc)
    cm = np.stack([Xc[labels == c].mean(0) for c in classes])
    p = np.array([(labels == c).mean() for c in classes])
    Wh = (Xc - cm[np.searchsorted(classes, labels)]) / np.sqrt(n)
    Bh = np.sqrt(p)[:, None] * cm
    w1, b1 = float((Wh * Wh).sum()), float((Bh * Bh).sum())
    w2 = float(((Wh @ Wh.T) ** 2).sum()) if n <= Xc.shape[1] else float(((Wh.T @ Wh) ** 2).sum())
    b2 = float(((Bh @ Bh.T) ** 2).sum())
    x = float(((Wh @ Bh.T) ** 2).sum())
    if w1 <= 0 or b1 <= 0:
        nan = float("nan")
        return {"trB_trW": (b1 / w1 if w1 > 0 else nan), "PR_W": (w1 ** 2 / w2 if w1 > 0 else nan), "PR_B": nan,
                "ratio": nan, "xi": nan, "alignment": nan, "threshold_full": nan, "threshold_x_free": nan,
                "c_full": (0.0 if b1 <= 0 and w1 > 0 else nan), "c_without_x": (0.0 if b1 <= 0 and w1 > 0 else nan),
                "x_share": nan, "class_mean_negative": False, "law_says_negative": False,
                "degenerate": "no class-mean variance" if b1 <= 0 else "no within-class variance"}
    beta, prw, prb = b1 / w1, w1 ** 2 / w2, b1 ** 2 / b2
    xi = x / (w1 * b1)
    a = xi * prw
    c_full = float(np.log((1 + beta) ** 2 / (1 + (b2 + 2 * x) / w2)))
    thr = 1 + 2 * (1 - a) / beta
    return {"trB_trW": beta, "PR_W": prw, "PR_B": prb, "ratio": prw / prb, "xi": xi, "alignment": a,
            "threshold_full": thr, "threshold_x_free": 1 + 2 / beta, "c_full": c_full,
            "c_without_x": float(np.log((1 + beta) ** 2 / (1 + b2 / w2))), "x_share": 2 * x / (b2 + 2 * x),
            "class_mean_negative": bool(c_full < 0), "law_says_negative": bool(prw / prb > thr)}


def bias_null(X, labels, mode="shape", seed=0):
    """A surrogate with the data's class means and class sizes whose fluctuations carry no class dependence of the
    kind named by `mode` (the finite-sample bias null of Appendix K). Each class's residuals are standardized per
    feature, all residual rows are pooled and redealt to the classes at random, and each class is rescaled:
      'shape'    every class gets the pooled per-feature SD profile scaled to its own total variance (one covariance
                 shape for all classes; what the within part reads at these counts when fluctuations do not depend
                 on the class);
      'variance' every class keeps its own per-feature SDs over the pooled correlation structure.
    Returns (Y, labels) with rows grouped by class."""
    X = np.asarray(X, np.float64); labels = np.asarray(labels)
    classes = np.unique(labels)
    mu = {c: X[labels == c].mean(0) for c in classes}
    sd = {c: X[labels == c].std(0) for c in classes}
    Z = np.empty_like(X)
    for c in classes:
        s_ = sd[c].copy(); s_[s_ <= 0] = 1.0
        Z[labels == c] = (X[labels == c] - mu[c]) / s_
    sd_pool = np.sqrt(np.mean([sd[c] ** 2 for c in classes], 0))
    if mode == "shape":
        tgt = {c: np.sqrt((sd[c] ** 2).sum() / max((sd_pool ** 2).sum(), 1e-30)) * sd_pool for c in classes}
    elif mode == "variance":
        tgt = sd
    else:
        raise ValueError("mode is 'shape' or 'variance'")
    rng = np.random.default_rng(seed)
    for _attempt in range(50):
        rows = Z[rng.permutation(len(Z))]
        parts, labs, i, ok = [], [], 0, True
        for c in classes:
            k = int((labels == c).sum())
            Zc = rows[i:i + k].copy(); i += k
            Zc -= Zc.mean(0); s_ = Zc.std(0)
            if np.any((s_ <= 1e-12) & (tgt[c] > 0)):
                ok = False          # this deal left a class constant in a feature it must vary in: deal again
                break
            s_[s_ <= 0] = 1.0; Zc /= s_
            parts.append(mu[c] + Zc * tgt[c]); labs.append(np.full(k, c))
        if ok:
            return np.concatenate(parts), np.concatenate(labs)
    raise ValueError("bias_null: no deal of the residuals gives every class variance in every feature it varies in "
                     "(classes too small or features too discrete for this surrogate)")


def decompose(X, labels, order=None, bin_counts=None, n_null=10, seed=42, min_size=10, n_shuffle=0, n_bias=0,
              floor_by_rung=False):
    """The exact split of every rung's deficit and of the shift for the ladder that accumulates the classes in
    `order` (default: sorted labels) with `bin_counts` classes per rung (default: the paper's ladder). Returns the
    per-rung terms, the four slopes of the shift (and the scale-free regrouping), the private slopes, exactness
    checks, the sign-law readout of the full set, with n_shuffle > 0 the label-permutation baseline and the excess
    over it, and with n_bias > 0 the same split on bias_null(mode='shape') surrogates (seeds 0..n_bias-1) and the
    data net of it: the within part counts as class-dependent fluctuation structure only beyond that null.
    floor_by_rung draws the floor rung by rung, as the paper's generators do (see zoom)."""
    X = np.asarray(X, np.float64); labels = np.asarray(labels)
    if not np.all(np.isfinite(X)):
        raise ValueError("decompose: X holds NaN or inf")
    X = X - X.mean(0)     # centre in float64 before any product (the split is shift-invariant in algebra)
    w_full = traces(X, labels)[1]
    if w_full <= 0:
        raise ValueError("decompose: the classes have no within-class variance; the split is undefined")
    if order is None:
        order = [int(c) for c in np.unique(labels)]
    if bin_counts is None:
        # the same rungs as zoom: the paper's ladder truncated to the classes, with the full set always the top rung
        bin_counts = tuple(c for c in BIN_COUNTS_DEFAULT if c <= len(order))
        if bin_counts[-1] != len(order):
            bin_counts = bin_counts + (len(order),)
    out = _decompose_once(X, labels, order, bin_counts, n_null, seed, min_size, floor_by_rung)
    if out is None:
        return out
    out["sign_law"] = sign_law(X, labels)
    out["class_pr"] = class_pr(X, labels)
    if n_bias > 0:
        bias = []
        for s_ in range(n_bias):
            Y, ly = bias_null(X, labels, "shape", s_)
            r = _decompose_once(Y, ly, order, bin_counts, n_null, seed, min_size, floor_by_rung)
            if r is not None:
                bias.append(r)
        if bias:
            out["bias_null"] = {"n": len(bias), "delta": float(np.mean([r["delta"] for r in bias])),
                                "delta_split_free": {k: float(np.mean([r["delta_split_free"][k] for r in bias])) for k in FREE}}
            out["net_of_bias"] = {"delta": out["delta"] - out["bias_null"]["delta"],
                                  "delta_split_free": {k: out["delta_split_free"][k] - out["bias_null"]["delta_split_free"][k]
                                                       for k in FREE}}
    if n_shuffle <= 0:
        return out
    keys = ("A", "P", "Cb", "Cx", "D", "Tb", "S")
    shuf = []
    for s in range(n_shuffle):
        srng = np.random.default_rng(700 + s)
        r = _decompose_once(X, labels[srng.permutation(len(labels))], order, bin_counts, n_null, seed, min_size, floor_by_rung)
        if r is not None:
            shuf.append(r)
    if not shuf:
        return out

    def _sl(r, k):
        return r["delta_split"][k] if k in r["delta_split"] else (r["delta_scale"] if k == "S" else r["delta_split_free"][k])
    out["n_shuffle"] = len(shuf)
    def _mean_or_none(key):
        v = [r[key] for r in shuf]
        return float(np.mean(v)) if all(x is not None for x in v) else None
    out["shuffle"] = {"delta": float(np.mean([r["delta"] for r in shuf])),
                      "delta_split": {k: float(np.mean([_sl(r, k) for r in shuf])) for k in keys},
                      "private_slope": _mean_or_none("private_slope"),
                      "private_slope_dim": _mean_or_none("private_slope_dim"),
                      "rungs": [{k: float(np.mean([r["rungs"][i][k] for r in shuf])) for k in ("deficit",) + keys} for i in range(len(out["rungs"]))]}
    if "at_four" in out and all("at_four" in r for r in shuf):
        out["shuffle"]["at_four"] = {k: float(np.mean([r["at_four"][k] for r in shuf])) for k in ("deficit",) + keys}
    out["excess"] = {"delta": out["delta"] - out["shuffle"]["delta"],
                     "delta_split": {k: _sl(out, k) - out["shuffle"]["delta_split"][k] for k in keys},
                     "private_slope": (out["private_slope"] - out["shuffle"]["private_slope"]
                                       if out["private_slope"] is not None and out["shuffle"]["private_slope"] is not None else None),
                     "private_slope_dim": (out["private_slope_dim"] - out["shuffle"]["private_slope_dim"]
                                           if out["private_slope_dim"] is not None and out["shuffle"]["private_slope_dim"] is not None else None)}
    if "at_four" in out["shuffle"]:
        out["excess"]["at_four"] = {k: out["at_four"][k] - out["shuffle"]["at_four"][k] for k in ("deficit",) + keys}
    return out


def _decompose_once(X, labels, order, bin_counts, n_null, seed, min_size, floor_by_rung=False):
    rng = np.random.default_rng(seed)
    members = [np.where(labels == c)[0] for c in order]
    K = len(order)
    sizes, ks, obs = [], [], []
    for c in bin_counts:
        sel = np.concatenate(members[:c])
        if len(sel) < min_size:
            continue
        sizes.append(int(len(sel))); ks.append(int(c))
        t = logterms(traces(X[sel], labels[sel])); t["pr_check"] = pr_centered(X[sel]); obs.append(t)
    if len(sizes) < 3:
        return None
    dropped = [int(c) for c in bin_counts if c not in ks]
    flo = [dict(lt=0.0, lw2=0.0, lb=0.0, lbx=0.0, ld=0.0, tb=0.0, lw1=0.0, pr=0.0, traces=np.zeros(5)) for _ in sizes]
    cells = ([k for k in range(len(sizes)) for _ in range(n_null)] if floor_by_rung
             else [k for _ in range(n_null) for k in range(len(sizes))])
    for k in cells:
        idx = rng.choice(len(X), sizes[k], replace=False)
        t = logterms(traces(X[idx], labels[idx]))
        for key in flo[k]:
            if key == "traces":
                flo[k][key] = flo[k][key] + np.array(t[key]) / n_null
            else:
                flo[k][key] += (np.log(t["pr"]) if key == "pr" else t[key]) / n_null
    rows = []
    for k, s in enumerate(sizes):
        o, f = obs[k], flo[k]
        A = o["lt"] - f["lt"]; P = -(o["lw2"] - f["lw2"]); Cb = -(o["lb"] - f["lb"])
        Cx = -((o["lbx"] - o["lb"]) - (f["lbx"] - f["lb"]))
        deficit = float(np.log(o["pr"]) - f["pr"])
        D = o["ld"] - f["ld"]; Tb = o["tb"] - f["tb"]; S = o["lw1"] - f["lw1"]
        rows.append({"classes": ks[k], "size": s, "deficit": deficit, "A": float(A), "P": float(P), "Cb": float(Cb), "Cx": float(Cx),
                     "D": float(D), "Tb": float(Tb), "S": float(S),
                     "log_pr_obs": float(np.log(o["pr"])), "log_pr_floor": float(f["pr"]),
                     "traces_obs": o["traces"], "traces_floor_mean": [float(v) for v in f["traces"]],
                     "sum_check": float(A + P + Cb + Cx - deficit), "regroup_check": float(D + Tb - A - P),
                     "pr_identity_rel": float(abs(o["pr"] - o["pr_check"]) / o["pr_check"])})
    logn = np.log(sizes)
    split = {key: _lsq_slope(logn, [r[key] for r in rows]) for key in ("deficit", "A", "P", "Cb", "Cx", "D", "Tb", "S")}
    below = ks[-1] == K and len(ks) >= 3
    out = {"rungs": rows, "delta": split["deficit"], "delta_split": {key: split[key] for key in ("A", "P", "Cb", "Cx")},
           "delta_split_free": {key: split[key] for key in ("D", "Tb", "Cb", "Cx")}, "delta_scale": split["S"],
           "delta_split_check": float(sum(split[key] for key in ("A", "P", "Cb", "Cx")) - split["deficit"]),
           "private_slope": _lsq_slope(np.log(np.array(ks[:-1]) / K), [r["P"] for r in rows[:-1]]) if below else None,
           "private_slope_dim": _lsq_slope(np.log(np.array(ks[:-1]) / K), [r["D"] for r in rows[:-1]]) if below else None,
           "max_sum_check": float(max(abs(r["sum_check"]) for r in rows)), "max_regroup_check": float(max(abs(r["regroup_check"]) for r in rows)),
           "max_pr_identity_rel": float(max(r["pr_identity_rel"] for r in rows)),
           "rung_classes": ks, "rungs_below_min_size": dropped}
    if 4 in ks and ks[-1] == K:
        r4 = rows[ks.index(4)]
        out["at_four"] = {key: r4[key] for key in ("deficit", "A", "P", "Cb", "Cx", "D", "Tb", "S")}
        late = {key: -r4[key] for key in ("deficit", "A", "P", "Cb", "Cx")}   # the top rung is zero in every term
        out["late_share"] = {key: (late[key] / late["deficit"] if late["deficit"] != 0 else None) for key in ("A", "P", "Cb", "Cx")}
    return out


def largest_term(d, keys=("A", "P", "Cb", "Cx")):
    """The term with the largest magnitude among `keys` in a split dict."""
    return max(keys, key=lambda k: abs(d[k]))


def _print_decomposition(r):
    f = r["delta_split_free"]
    cm = f["Tb"] + f["Cb"] + f["Cx"]
    print(f"decompose (Section 5): within-class part D {f['D']:+.3f} + class-mean part (Tb + Cb + Cx) {cm:+.3f} = shift {r['delta']:+.3f}")
    print(f"  four terms: trace A {r['delta_split']['A']:+.3f} + pooling P {r['delta_split']['P']:+.3f} + between-class Cb {r['delta_split']['Cb']:+.3f} "
          f"+ coupling Cx {r['delta_split']['Cx']:+.3f}; A + P = D + Tb (Tb {f['Tb']:+.3f}), the within-class scale S {r['delta_scale']:+.3f} cancelling between A and P")
    ex = r.get("excess", {}).get("delta_split")
    if ex is not None:
        print(f"  net of {r['n_shuffle']} label permutations: D {ex['D']:+.3f} Tb {ex['Tb']:+.3f} Cb {ex['Cb']:+.3f} Cx {ex['Cx']:+.3f} S {ex['S']:+.3f} -> largest {largest_term(ex, FREE)}; "
              + (f"private slope (D vs log k/K) {r['excess']['private_slope_dim']:+.2f}" if r['excess']['private_slope_dim'] is not None
                 else "private slope undefined (the top rung is not the full class set)"))
    elif r.get("private_slope_dim") is not None:
        print(f"  private slope (D vs log k/K) {r['private_slope_dim']:+.2f}; run with --n-shuffle for the sampling baseline")
    print(f"  exactness: sum {r['max_sum_check']:.1e}, regroup {r['max_regroup_check']:.1e}, PR identity {r['max_pr_identity_rel']:.1e}")
    sl = r.get("sign_law")
    if sl and "degenerate" in sl:
        print(f"  sign law: undefined ({sl['degenerate']})")
    elif sl:
        agree = sl["class_mean_negative"] == sl["law_says_negative"]
        print(f"  sign law (Eq. 3): full-set class-mean remainder c {sl['c_full']:+.3f}, whose sign predicts the class-mean part's; "
              f"PR(W)/PR(B) = {sl['PR_W']:.1f}/{sl['PR_B']:.1f} = {sl['ratio']:.1f} against 1 + 2(1 - a)/beta = {sl['threshold_full']:.1f} "
              f"(a {sl['alignment']:.2f}, beta = Tr B/Tr W {sl['trB_trW']:.3f}; with a = 0 the threshold is 1 + 2/beta = {sl['threshold_x_free']:.1f}, "
              "sufficient when the alignment is positive) " + ("(sign as the law says)" if agree else "(MISMATCH with the law: report it)"))
        if r.get("rungs"):
            n_pc = r["rungs"][-1]["size"] / max(r["rung_classes"][-1], 1)
            cp = r.get("class_pr")
            own = (f"each class's own fluctuation dimension (PR of its residuals) {min(cp.values()):.1f} to {max(cp.values()):.1f}, "
                   f"median {float(np.median(list(cp.values()))):.1f}; " if cp else "")
            print(f"  samples per class {n_pc:.0f} against {own}PR(W) {sl['PR_W']:.1f}, which pools the classes (README rule 8: the "
                  "within-class part needs each class sampled beyond its own fluctuation dimension; --account prints the sampling term)")
    if "net_of_bias" in r:
        b = r["bias_null"]["delta_split_free"]; nb = r["net_of_bias"]["delta_split_free"]
        print(f"  bias null ({r['bias_null']['n']} surrogates, one covariance shape for every class): D {b['D']:+.3f} Tb {b['Tb']:+.3f} "
              f"Cb {b['Cb']:+.3f} Cx {b['Cx']:+.3f}; net of it: D {nb['D']:+.3f}, delta {r['net_of_bias']['delta']:+.3f}")


# ---- Choosing an axis among candidate labels (Section 8, Appendix H) ----

def _one_sided_p(obs, null):
    """One-sided permutation p for a positive excess, (1 + #{null >= obs}) / (B + 1), with the floating-point tolerance of
    _null_stats; NaN when the statistic or the null is undefined."""
    null = np.asarray(null, dtype=np.float64)
    if null.size == 0 or not np.isfinite(obs) or not np.all(np.isfinite(null)):
        return float("nan")
    tol = 64 * np.finfo(float).eps * (1.0 + abs(obs) + abs(float(null.mean())))
    return float((1 + (null >= obs - tol).sum()) / (len(null) + 1))


def _joint_cells(arrays):
    """One integer id per row for the joint cell of several label arrays."""
    return np.unique(np.stack([np.asarray(a) for a in arrays], 1), axis=0, return_inverse=True)[1].ravel()


def candidates(X, label_sets, strata=None, halves=None, n_perm=200, seed=0, standardize=False, alpha=0.05,
               n_floor_draws=20, floor_seed=None):
    """Which of several candidate labels organizes a population: each candidate's shift against label permutations, ranked
    by z = (shift - null mean) / null SD, which does not depend on the floor draw. The screen asks for a positive shift,
    so its p is one-sided.

    label_sets : {name: labels}, one integer label per row each (three or more classes).
    strata : None (plain permutations); "others" (each candidate permuted inside the joint cells of the other
        candidates, so its null keeps the label's association with them: the paper's screen); an array (one nuisance
        stratum per row, the same for every candidate); or a dict {name: array or None}.
    halves : None (one screen on all rows); "oddeven" (row parity, a split that uses no label); or one 0/1 per row. With
        halves the screen runs on each half: a candidate is discovered on a half at p <= alpha / (number of candidates)
        and replicated when the other half gives p <= alpha; each half's pick is its largest z, confirmed when it also
        ranks first on the other half with p <= alpha.
    Permutations for candidate i on half h use the stream seed + 100 (h + 1) + i (seed + i without halves), the paper's
    screen at seed 0; the floor (which sets the shift, never z or p) is drawn rung by rung from floor_seed when given.
    """
    X = np.asarray(X, dtype=np.float64)
    names = list(label_sets)
    L = {k: np.asarray(v) for k, v in label_sets.items()}
    n = len(X)
    if len(names) < 2:
        raise ValueError("candidates: give at least two candidate labels")
    for k, v in L.items():
        if v.shape != (n,):
            raise ValueError(f"candidates: label set {k!r} has shape {v.shape}, expected one label per row ({n})")

    def strata_of(name, rows):
        if strata is None:
            return None
        if isinstance(strata, str):
            if strata != "others":
                raise ValueError("strata is None, 'others', an array or a dict")
            return _joint_cells([L[o][rows] for o in names if o != name])
        if isinstance(strata, dict):
            s_ = strata.get(name)
            return None if s_ is None else np.asarray(s_)[rows]
        return np.asarray(strata)[rows]

    def label_null(Xs, y, st, rng):
        """Ladder slopes of the declared order for n_perm relabellings (inside strata when given), drawn as the paper's
        screen draws them: one permutation per stratum in sorted stratum order. The floor cancels in z and p."""
        Xs = _zscore(Xs) if standardize else Xs
        Xs = Xs - Xs.mean(axis=0, keepdims=True)
        G = Xs @ Xs.T
        classes = np.unique(y)
        lab = np.searchsorted(classes, y)
        bc = _rung_classes(len(classes), None)
        order = list(range(len(classes)))
        groups = [np.where(st == v)[0] for v in np.unique(st)] if st is not None else None
        obs = _ladder(G, [np.where(lab == c)[0] for c in order], order, bc)[0]
        null = []
        for _ in range(n_perm):
            if groups is None:
                yp = lab[rng.permutation(len(lab))]
            else:
                yp = lab.copy()
                for i in groups:
                    yp[i] = lab[i][rng.permutation(len(i))]
            null.append(_ladder(G, [np.where(yp == c)[0] for c in order], order, bc)[0])
        return obs, np.asarray(null)

    def screen(rows, off):
        res = {}
        for i, name in enumerate(names):
            y = L[name][rows]
            st = strata_of(name, rows)
            r = zoom(X[rows], y, n_perm=0, k_orders=0, seed=seed + off + i, standardize=standardize,
                     n_floor_draws=n_floor_draws, floor_seed=floor_seed, floor_by_rung=floor_seed is not None)
            obs, null = label_null(X[rows], y, st, np.random.default_rng(seed + off + i))
            z, _, nm, ns = _null_stats(obs, null)
            row = {"delta": r["delta"], "theta_floor": r["theta_floor"], "theta_floor_se": r["theta_floor_se"],
                   "z": z, "p_one": _one_sided_p(obs, null), "null_mean": nm - r["theta_floor"], "null_sd": ns,
                   "n_classes": r["n_classes"], "stratified": st is not None}
            if st is not None:
                cell = np.unique(st, return_counts=True)[1]
                row["median_stratum_size"] = float(np.median(cell))
                if np.median(cell) < 4:
                    import warnings
                    warnings.warn(f"candidates: the strata of {name!r} hold a median of {np.median(cell):.0f} rows; a "
                                  "stratified null with so little to permute reads almost any shift as large. Use "
                                  "coarser strata (leave random-label controls out of 'others').", stacklevel=3)
            res[name] = row
        return res

    out = {"names": names, "n_perm": int(n_perm), "alpha": alpha,
           "strata": strata if isinstance(strata, str) or strata is None else "given"}
    if halves is None:
        res = screen(np.arange(n), 0)
        out["screen"] = res
        out["ranking"] = sorted(names, key=lambda k: -res[k]["z"] if np.isfinite(res[k]["z"]) else np.inf)
        out["pick"] = out["ranking"][0]
        return out
    h = (np.arange(n) % 2) if (isinstance(halves, str) and halves == "oddeven") else np.asarray(halves).astype(int)
    if h.shape != (n,) or not set(np.unique(h)) <= {0, 1}:
        raise ValueError("halves must be 'oddeven' or one 0/1 per row")
    H = [screen(np.where(h == 0)[0], 100), screen(np.where(h == 1)[0], 200)]
    m = len(names)
    out["halves"] = H
    out["per_candidate"] = {k: {"z": [H[0][k]["z"], H[1][k]["z"]], "p_one": [H[0][k]["p_one"], H[1][k]["p_one"]],
                                "delta": [H[0][k]["delta"], H[1][k]["delta"]],
                                "discovered": [bool(H[0][k]["p_one"] <= alpha / m), bool(H[1][k]["p_one"] <= alpha / m)],
                                "replicated": [bool(H[0][k]["p_one"] <= alpha / m and H[1][k]["p_one"] <= alpha),
                                               bool(H[1][k]["p_one"] <= alpha / m and H[0][k]["p_one"] <= alpha)]}
                            for k in names}
    out["picks"] = []
    for a, b in ((0, 1), (1, 0)):
        pick = max(names, key=lambda k: H[a][k]["z"] if np.isfinite(H[a][k]["z"]) else -np.inf)
        rank = 1 + sum(H[b][k]["z"] > H[b][pick]["z"] for k in names if k != pick)
        out["picks"].append({"half": a, "pick": pick, "rank_on_other": int(rank), "p_on_other": H[b][pick]["p_one"],
                             "confirmed": bool(rank == 1 and H[b][pick]["p_one"] <= alpha)})
    return out


def _print_candidates(r):
    if "screen" in r:
        print(f"candidates ({r['n_perm']} permutations each, strata: {r['strata']}), ranked by z:")
        for k in r["ranking"]:
            s_ = r["screen"][k]
            print(f"  {k:12s} z {s_['z']:+6.1f}  one-sided p {s_['p_one']:.3f}  shift {s_['delta']:+.3f} "
                  f"(floor SE {s_['theta_floor_se']:.3f}, {s_['n_classes']} classes)")
        return
    print(f"candidates on two halves ({r['n_perm']} permutations each, strata: {r['strata']}); discovery at "
          f"p <= {r['alpha']}/{len(r['names'])}, replication at p <= {r['alpha']}:")
    for k in r["names"]:
        c = r["per_candidate"][k]
        print(f"  {k:12s} z {c['z'][0]:+6.1f} / {c['z'][1]:+6.1f}   p {c['p_one'][0]:.3f} / {c['p_one'][1]:.3f}   "
              f"discovered {sum(c['discovered'])}/2, replicated {sum(c['replicated'])}/2")
    for pk in r["picks"]:
        print(f"  pick on half {pk['half']}: {pk['pick']} (rank {pk['rank_on_other']} on the other half, p {pk['p_on_other']:.3f}): "
              + ("confirmed" if pk["confirmed"] else "not confirmed"))


# ---- The two-term account (Section 5, Figure 3, Appendix K) ----
# The within-class part read three ways: the mean-field overlap law ln[K/(1+(K-1) sigma)]/ln K with sigma from split halves;
# the overlap identity in the population limit (W1); and the identity at each rung's sample counts (W2, App K's PR_fin, whose
# one-class form is Eq. 2). The class-mean part read from Eq. 3's full-set remainder carried through the rungs by the class
# means' variance fraction. Their sum against the measured shift is Figure 3 for one dataset.

def _split_half_K(Ra, Rb):
    """Split-half overlap matrix from per-class residuals of two independent halves, both directions averaged."""
    def tr(Z):
        return float((Z * Z).sum() / len(Z))
    C = len(Ra)
    K = np.zeros((C, C))
    ta = [tr(Z) for Z in Ra]
    tb = [tr(Z) for Z in Rb]
    for c in range(C):
        for d in range(C):
            ab = Ra[c] @ Rb[d].T
            ba = Rb[c] @ Ra[d].T
            v1 = (ab * ab).sum() / (len(Ra[c]) * len(Rb[d])) / (ta[c] * tb[d])
            v2 = (ba * ba).sum() / (len(Rb[c]) * len(Ra[d])) / (tb[c] * ta[d])
            K[c, d] = 0.5 * (v1 + v2)
    return K


def _class_halves(lab, seed):
    """A random half of each class (seeded): 0/1 per row."""
    rng = np.random.default_rng(seed)
    h = np.zeros(len(lab), int)
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        h[idx[rng.permutation(len(idx))[len(idx) // 2:]]] = 1
    return h


def split_half_sigma(X, labels, halves=None, seed=0):
    """The classes' mean fluctuation overlap sigma (Section 5, Appendix K): the mean off-diagonal entry of the overlap
    matrix K_cd = Tr(S_c S_d) / (Tr S_c Tr S_d) over its mean diagonal entry, with S_c and S_d taken from two independent
    halves (residuals from each half's own class mean), so the diagonal carries no sampling term. sigma = 1 when every class
    fluctuates alike, 0 when the classes fluctuate in disjoint directions. halves: one 0/1 per row (default: a random half of
    each class). Returns (sigma, K)."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    h = _class_halves(labels, seed) if halves is None else np.asarray(halves).astype(int)
    Ra, Rb = [], []
    for c in np.unique(labels):
        A, B = X[(labels == c) & (h == 0)], X[(labels == c) & (h == 1)]
        if len(A) < 2 or len(B) < 2:
            raise ValueError(f"split_half_sigma: class {c!r} needs at least two rows in each half")
        Ra.append(A - A.mean(0))
        Rb.append(B - B.mean(0))
    K = _split_half_K(Ra, Rb)
    off = ~np.eye(len(K), dtype=bool)
    return float(K[off].mean() / np.diag(K).mean()), K


def class_moments(X, labels, seed=0, n_split=10):
    """Unbiased per-class moments for the overlap identity at finite samples (Appendix K): t_c = Tr Sigma_c,
    T_cd = Tr(Sigma_c Sigma_d) (c != d from full-class residuals; c = d from n_split random split halves, each centred on
    its own mean, so no fourth-moment bias enters), m4_c = E||x - mu_c||^4, and the class sizes n_c, in sorted label order."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    C = len(classes)
    rng = np.random.default_rng(seed)
    R, nu = [], np.zeros(C)
    for i, c in enumerate(classes):
        Fc = X[labels == c]
        if len(Fc) < 4:
            raise ValueError(f"class_moments: class {c!r} has {len(Fc)} rows; the moments need at least four")
        R.append(Fc - Fc.mean(0))
        nu[i] = len(Fc) - 1
    t = np.array([(r * r).sum() for r in R]) / nu
    m4 = np.array([((r * r).sum(1) ** 2).mean() for r in R]) * ((nu + 1) / nu) ** 2
    D = X.shape[1]
    if max(len(r) for r in R) >= D:
        S = [r.T @ r for r in R]
        blk = lambda c, d: float((S[c] * S[d]).sum())
    else:
        blk = lambda c, d: float(((R[c] @ R[d].T) ** 2).sum())
    T = np.zeros((C, C))
    for c in range(C):
        for d in range(c + 1, C):
            T[c, d] = T[d, c] = blk(c, d) / (nu[c] * nu[d])
    for i, c in enumerate(classes):
        Fc = X[labels == c]
        acc = 0.0
        for _ in range(n_split):
            p = rng.permutation(len(Fc))
            A, B = Fc[p[: len(Fc) // 2]], Fc[p[len(Fc) // 2:]]
            A = A - A.mean(0)
            B = B - B.mean(0)
            v = float(((A.T @ A) * (B.T @ B)).sum()) if A.shape[1] <= len(A) else float(((A @ B.T) ** 2).sum())
            acc += v / ((len(A) - 1) * (len(B) - 1))
        T[i, i] = acc / n_split
    return {"t": t, "T": T, "m4": m4, "n": nu + 1}


def pr_population_identity(w, mom):
    """Appendix K's identity, 1/PR(W_A) = sum alpha_c alpha_d K_cd, with class weights w (sample counts)."""
    w = np.asarray(w, float)
    return float((w @ mom["t"]) ** 2 / (w @ mom["T"] @ w))


def pr_finite_identity(m, mom, gauss=False):
    """The identity at finite samples (Appendix K): m samples of each class, each class centred on its own sample mean,
    PR_fin = (sum nu_c t_c)^2 / [sum nu_c nu_d T_cd + sum nu_c (m4_c - T_cc)], nu_c = m_c - 1; for one Gaussian class it
    is Eq. 2. gauss=True uses Gaussian fourth moments."""
    nu = np.maximum(np.asarray(m, float) - 1, 0)
    m4 = mom["t"] ** 2 + 2 * np.diag(mom["T"]) if gauss else mom["m4"]
    return float((nu @ mom["t"]) ** 2 / (nu @ mom["T"] @ nu + nu @ (m4 - np.diag(mom["T"]))))


def _rung_classes(n_classes, bin_counts):
    if bin_counts is None:
        bc = tuple(c for c in BIN_COUNTS_DEFAULT if c <= n_classes)
        return bc if bc[-1] == n_classes else bc + (n_classes,)
    return tuple(int(c) for c in bin_counts)


def _account_within(mom, counts, lab_pos, ks, sigma, rng, n_comp, mom_x=None):
    """The within-class part predicted on one declared ladder: rung k holds the first k classes (positions in the declared
    order; counts, moments and labels already in that order); the floor's rung of the same size holds the class counts of
    random subsets (n_comp compositions)."""
    K = len(counts)
    sizes = [int(counts[:k].sum()) for k in ks]
    comps = [np.array([np.bincount(lab_pos[rng.choice(len(lab_pos), s_, replace=False)], minlength=K) for _ in range(n_comp)])
             for s_ in sizes]
    lad_w = [np.where(np.arange(K) < k, counts, 0) for k in ks]
    x = np.log(sizes)
    out = {"sizes": sizes, "rung_classes": list(ks)}
    for name, M, g in (("finite", mom, False), ("finite_gauss", mom, True), ("finite_heldout", mom_x, False)):
        if M is None:
            continue
        lad = [np.log(pr_finite_identity(w, M, g)) for w in lad_w]
        flo = [float(np.mean([np.log(pr_finite_identity(w, M, g)) for w in C_])) for C_ in comps]
        out[name] = _lsq_slope(x, np.subtract(lad, flo))
    lad1 = [np.log(pr_population_identity(w, mom)) for w in lad_w]
    flo1 = [float(np.mean([np.log(pr_population_identity(w, mom)) for w in C_])) for C_ in comps]
    out["population"] = _lsq_slope(x, np.subtract(lad1, flo1))
    out["mean_field"] = float(np.log(K / (1 + (K - 1) * sigma)) / np.log(K))
    out["mean_field_rungs"] = _lsq_slope(x, [np.log(k / (1 + (k - 1) * sigma)) for k in ks])
    return out


def _classmean_term_rungs(X, lab_pos, ks):
    """Eq. 3's full-set class-mean remainder c = log[(1+beta)^2 / (1 + (b2 + 2x)/w2)] carried through the declared rungs: a
    rung of k classes holds a fraction u_k = b1_k / b1 of the full set's class-mean variance (b1_k from its own class means),
    with the full set's W and the class means' shape fixed, so its term is c_k = log[(1 + u_k beta)^2 / (1 + (u_k^2 b2 +
    2 u_k x)/w2)]; every floor rung holds all classes and the full term c. Returns the slope of c_k - c on log rung size,
    c and the u_k."""
    Xc = X - X.mean(0)
    K = lab_pos.max() + 1
    n = len(Xc)
    cnt = np.bincount(lab_pos, minlength=K)
    cm = np.stack([Xc[lab_pos == c].mean(0) for c in range(K)])
    Wh = (Xc - cm[lab_pos]) / np.sqrt(n)
    Bh = np.sqrt(cnt / n)[:, None] * cm
    w1, b1 = float((Wh * Wh).sum()), float((Bh * Bh).sum())
    w2 = float(((Wh @ Wh.T) ** 2).sum()) if n <= Xc.shape[1] else float(((Wh.T @ Wh) ** 2).sum())
    b2 = float(((Bh @ Bh.T) ** 2).sum())
    xx = float(((Wh @ Bh.T) ** 2).sum())
    if not b1 > 1e-12 * w1:                    # no class-mean variance: the term is zero on every rung
        return 0.0, 0.0, [0.0] * len(ks)
    beta = b1 / w1
    c_full = float(np.log((1 + beta) ** 2 / (1 + (b2 + 2 * xx) / w2)))
    ys, us, sizes = [], [], []
    for k in ks:
        p = cnt[:k] / cnt[:k].sum()
        mu = (p[:, None] * cm[:k]).sum(0)
        u = float((p * ((cm[:k] - mu) ** 2).sum(1)).sum()) / b1
        us.append(u)
        sizes.append(int(cnt[:k].sum()))
        ys.append(np.log((1 + u * beta) ** 2 / (1 + (u ** 2 * b2 + 2 * u * xx) / w2)) - c_full)
    return _lsq_slope(np.log(sizes), ys), c_full, us


def account(X, labels, order=None, bin_counts=None, halves=None, seed=0, n_null=20, n_comp=200, heldout=True, min_size=10,
            floor_by_rung=False):
    """The paper's two-term account for one dataset (Section 5, Figure 3, Appendix K), nothing fitted.

    Measured (decompose): the within-class part D and the class-mean part Tb + Cb + Cx of the declared ladder, and the shift.
    Predicted within-class part: the mean-field overlap law ln[K/(1+(K-1) sigma)]/ln K with sigma from split halves
    (Figure 3A's x-axis); the overlap identity in the population limit; the identity at each rung's sample counts (finite;
    App K's PR_fin), which also reads how far the floor's first rungs are under-sampled. Predicted class-mean part: Eq. 3's
    full-set remainder carried through the rungs (Figure 3B). Their sums against the measured shift (Figure 3C), with sign
    agreement. Mean field leaves out two measured corrections, reported as beyond_mean_field: overlaps that depend on the
    classes' separation (population identity minus mean field) and the floor's sampling (finite-sample identity minus
    population identity), whose first-order size is s = (m - 1) PR_class / (n ln K), m = K/[1 + (K-1) sigma], at n
    samples per class (Figure 3 opens the markers with s > 0.05).
    With heldout, sigma and the moments come from one half of each class and the part is measured on the other, both ways
    (App K's held-out test). halves: one 0/1 per row for sigma's split (default: a random half of each class)."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    if not np.all(np.isfinite(X)):
        raise ValueError("account: X holds NaN or inf")
    X = X - X.mean(0)
    classes = np.unique(labels)
    K = len(classes)
    if K < 3:
        raise ValueError("account: needs at least three classes")
    pos = declared_order(classes, order)                         # declared position -> class index
    rank = np.empty(K, int)
    rank[pos] = np.arange(K)
    lab_pos = rank[np.searchsorted(classes, labels)]               # each row's class position in the declared order
    ks_all = _rung_classes(K, bin_counts)
    order_labels = [classes[i].item() if hasattr(classes[i], "item") else classes[i] for i in pos]

    def ladder_ks(lp):
        cnt = np.bincount(lp, minlength=K)
        return [k for k in ks_all if cnt[:k].sum() >= min_size]

    def measured(Xs, lp):
        r = decompose(Xs, lp, order=list(range(K)), bin_counts=tuple(ks_all), n_null=n_null, seed=seed, min_size=min_size,
                      floor_by_rung=floor_by_rung)
        if r is None:
            raise ValueError("account: the ladder has fewer than three rungs of min_size samples")
        f = r["delta_split_free"]
        return {"delta": r["delta"], "within": f["D"], "class_mean": f["Tb"] + f["Cb"] + f["Cx"]}

    def moments_pos(Xs, lp, s_):
        m = class_moments(Xs, lp, seed=s_)
        return m                                                     # lp holds positions 0..K-1, so sorted order = declared

    rng = np.random.default_rng([seed, 1301])
    out = {"n_classes": K, "order": order_labels, "rung_classes_requested": list(ks_all)}
    meas = measured(X, lab_pos)
    sig, Kmat = split_half_sigma(X, lab_pos, halves=halves, seed=seed)
    mom = moments_pos(X, lab_pos, seed)
    cnt = np.bincount(lab_pos, minlength=K)
    ks = ladder_ks(lab_pos)
    wp = _account_within(mom, cnt, lab_pos, ks, sig, rng, n_comp)
    cm_pred, c_full, us = _classmean_term_rungs(X, lab_pos, ks)
    pr_class = float(np.mean(mom["t"] ** 2 / np.diag(mom["T"])))
    m_eff = K / (1 + (K - 1) * sig)
    n_pc = float(cnt.mean())
    s_term = float((m_eff - 1) * pr_class / (n_pc * np.log(K)))
    out.update({"measured": meas, "sigma": sig, "overlap_matrix": Kmat.tolist(),
                "within_pred": {k: wp[k] for k in ("mean_field", "mean_field_rungs", "population", "finite", "finite_gauss")},
                "class_mean_pred": cm_pred, "c_full": c_full, "u": us, "rung_sizes": wp["sizes"], "rung_classes": wp["rung_classes"],
                "delta_pred_mean_field": wp["mean_field"] + cm_pred, "delta_pred_finite": wp["finite"] + cm_pred,
                "PR_class": pr_class, "PR_W": pr_population_identity(cnt, mom), "samples_per_class": n_pc,
                "m": m_eff, "sampling_term": s_term,
                "beyond_mean_field": {"separation": wp["population"] - wp["mean_field"], "sampling": wp["finite"] - wp["population"]}})
    out["signs_agree"] = {"mean_field": bool(np.sign(out["delta_pred_mean_field"]) == np.sign(meas["delta"])),
                          "finite": bool(np.sign(out["delta_pred_finite"]) == np.sign(meas["delta"])),
                          "class_mean": bool(np.sign(cm_pred) == np.sign(meas["class_mean"]))}
    if heldout:
        h = _class_halves(lab_pos, seed + 1)
        hv = []
        for est, mh in ((0, 1), (1, 0)):
            S_, M_ = h == est, h == mh
            sig_s, _ = split_half_sigma(X[S_], lab_pos[S_], seed=seed + 2 + est)
            mom_s = moments_pos(X[S_], lab_pos[S_], seed + 3 + est)
            mom_m = moments_pos(X[M_], lab_pos[M_], seed + 5 + mh)
            lp_m = lab_pos[M_]
            cnt_m = np.bincount(lp_m, minlength=K)
            wph = _account_within(mom_m, cnt_m, lp_m, ladder_ks(lp_m), sig_s, rng, n_comp, mom_x=mom_s)
            mm = measured(X[M_], lp_m)
            hv.append({"estimated_on": est, "sigma": sig_s, "measured_within": mm["within"],
                       "mean_field": wph["mean_field"], "finite_heldout": wph["finite_heldout"]})
        out["heldout"] = {"halves": hv,
                          "measured_within": float(np.mean([q["measured_within"] for q in hv])),
                          "mean_field": float(np.mean([q["mean_field"] for q in hv])),
                          "finite_heldout": float(np.mean([q["finite_heldout"] for q in hv]))}
    return out


def _print_account(a):
    m, w = a["measured"], a["within_pred"]
    print(f"account (Section 5, Figure 3; nothing fitted), {a['n_classes']} classes, {a['samples_per_class']:.0f} samples per class:")
    print(f"  within-class part: measured {m['within']:+.3f} | mean-field overlap law {w['mean_field']:+.3f} (sigma {a['sigma']:.3f}, "
          f"split halves) | identity at the rungs' sample counts {w['finite']:+.3f} | identity, population limit {w['population']:+.3f}")
    print(f"  class-mean part:   measured {m['class_mean']:+.3f} | Eq. 3's remainder carried through the rungs {a['class_mean_pred']:+.3f} "
          f"(full-set remainder c {a['c_full']:+.3f})")
    sa = a["signs_agree"]
    print(f"  shift:             measured {m['delta']:+.3f} | two terms, mean field {a['delta_pred_mean_field']:+.3f} "
          f"({'sign agrees' if sa['mean_field'] else 'SIGN DIFFERS'}) | two terms, finite-sample identity {a['delta_pred_finite']:+.3f} "
          f"({'sign agrees' if sa['finite'] else 'SIGN DIFFERS'})")
    s_ = a["sampling_term"]
    kern = w["population"] - w["mean_field"]
    fin = w["finite"] - w["population"]
    print(f"  beyond mean field (Section 5): overlaps that depend on separation {kern:+.3f}, the floor's sampling {fin:+.3f}; "
          f"first-order sampling term s = (m - 1) PR_class / (n ln K) = {s_:.3f} (m {a['m']:.2f}, PR_class {a['PR_class']:.1f}, "
          f"n {a['samples_per_class']:.0f})" + ("; s > 0.05, an open marker in Figure 3: read the finite-sample identity" if s_ > 0.05 else ""))
    if "heldout" in a:
        hv = a["heldout"]
        print(f"  held out (sigma and moments on one half, part measured on the other, both ways): measured {hv['measured_within']:+.3f}, "
              f"mean field {hv['mean_field']:+.3f}, finite-sample identity {hv['finite_heldout']:+.3f}")


# ---- The neuron or feature ladder (Figure 1's third reading, Appendix D) ----
FEATURE_SIZES_DEFAULT = (50, 100, 200, 500, 1000, 2000, 5000)


def _feature_pr_fn(X):
    """PR of the samples restricted to a subset of features, each sample centred across the features the subset holds
    (Appendix D's neuron-ladder convention). Uses the full feature Gram when it fits in about 2 GB, else each subset's
    smaller Gram."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)        # a per-sample constant across features: subset centring removes it exactly
    X = X / (X.std() + 1e-12)
    d = X.shape[1]
    if 8.0 * d * d <= 2e9:
        G = X.T @ X
        return lambda idx: _subset_pr(G, np.asarray(idx))

    def pr(idx):
        Y = X[:, np.asarray(idx)]
        Y = Y - Y.mean(axis=1, keepdims=True)
        S = Y.T @ Y if Y.shape[1] <= Y.shape[0] else Y @ Y.T
        tr, tr2 = float(np.trace(S)), float((S * S).sum())
        return tr * tr / tr2 if tr2 > 0 else 1.0
    return pr


def feature_ladder(X, positions=None, order=None, sizes=None, n_anchors=50, n_floor_draws=50, seed=42, floor_seed=1000,
                   control_seeds=(2000, 3000), n_null=0, null_seed=4000):
    """A ladder over features instead of samples (Figure 1's neighbouring neurons, Appendix D). Rung n holds n features:
    with positions (n_features x D coordinates, e.g. cell positions), the n features nearest a random anchor feature, PR
    averaged over n_anchors anchors; with order (every feature index once, e.g. sorted by a score), the first n. The floor
    holds n random features, PR averaged over n_floor_draws draws: the same aggregation in both arms, so no Jensen
    asymmetry. Returns theta_obs, theta_floor and the shift delta = theta_obs - theta_floor, the rung PRs, and the
    exchangeable control (random features in both arms, which must vanish in expectation). With n_null > 0, the shift
    against n_null relabellings of the features (positions or order permuted), for a p-value in one population.
    Default seeds reproduce the paper's Table 4."""
    X = np.asarray(X)
    d = X.shape[1]
    if (positions is None) == (order is None):
        raise ValueError("feature_ladder: give positions or order (exactly one)")
    sizes = [s_ for s_ in (sizes or FEATURE_SIZES_DEFAULT) if s_ <= d]
    if len(sizes) < 3:
        raise ValueError(f"feature_ladder: {d} features leave fewer than three rung sizes; pass smaller sizes")
    if positions is not None:
        positions = np.asarray(positions, dtype=float)
        positions = positions.reshape(len(positions), -1)
        if len(positions) != d:
            raise ValueError(f"feature_ladder: positions have {len(positions)} rows for {d} features")
    else:
        order = np.asarray(order).astype(int)
        if sorted(order.tolist()) != list(range(d)):
            raise ValueError("feature_ladder: order must list every feature index exactly once")
    pr = _feature_pr_fn(X)

    def mean_pr_ladder(subsets_per_size):
        return [float(np.mean([pr(idx) for idx in subs])) for subs in subsets_per_size]

    def observed(pos_, ord_, rng):
        if pos_ is not None:
            subs = []
            for ns in sizes:
                cur = []
                for _ in range(n_anchors):
                    a = rng.integers(d)
                    cur.append(np.argsort(((pos_ - pos_[a]) ** 2).sum(axis=1))[:ns])
                subs.append(cur)
            return mean_pr_ladder(subs)
        return [pr(ord_[:ns]) for ns in sizes]

    def random_ladder(rng):
        return mean_pr_ladder([[rng.choice(d, ns, replace=False) for _ in range(n_floor_draws)] for ns in sizes])

    pr_obs = observed(positions, order, np.random.default_rng(seed))
    pr_floor = random_ladder(np.random.default_rng(floor_seed))
    th_obs, th_floor = _slope(sizes, pr_obs), _slope(sizes, pr_floor)
    ca = random_ladder(np.random.default_rng(control_seeds[0]))
    cb = random_ladder(np.random.default_rng(control_seeds[1]))
    out = {"mode": "positions" if positions is not None else "order", "sizes": sizes, "n_features": int(d),
           "n_samples": int(X.shape[0]), "n_anchors": int(n_anchors), "n_floor_draws": int(n_floor_draws),
           "pr_obs": pr_obs, "pr_floor": pr_floor, "theta_obs": th_obs, "theta_floor": th_floor, "delta": th_obs - th_floor,
           "delta_control": _slope(sizes, ca) - _slope(sizes, cb)}
    if n_null:
        nrng = np.random.default_rng(null_seed)
        null = []
        for _ in range(n_null):
            p_ = nrng.permutation(d)
            null.append(_slope(sizes, observed(positions[p_] if positions is not None else None,
                                               p_[order] if order is not None else None, nrng)) - th_floor)
        z, p, nm, ns = _null_stats(out["delta"], null)
        out.update({"n_null": int(n_null), "z": z, "p_two": p, "null_mean": nm, "null_sd": ns})
    return out


def _feature_ladder_lines(r):
    """The plain-language reading of one `feature_ladder` result, line by line."""
    how = "nearest neighbours of random anchors" if r["mode"] == "positions" else "features in the given order"
    lines = [f"feature ladder ({how}; rungs {r['sizes']} of {r['n_features']} features, {r['n_samples']} samples):",
             f"  theta_obs {r['theta_obs']:+.3f} = floor {r['theta_floor']:+.3f} + shift {r['delta']:+.3f}; "
             f"exchangeable control (random in both arms) {r['delta_control']:+.3f}"]
    if "p_two" in r:
        lines.append(f"  against {r['n_null']} relabellings of the features: z = {r['z']:+.1f}, two-sided p = {r['p_two']:.3f}")
    else:
        lines.append("  (add --n-null N for a p-value in this population; the paper reads the sign across recordings)")
    return lines


def _print_feature_ladder(r):
    for line in _feature_ladder_lines(r):
        print(line)


# ---- Coarse-graining features into blocks (Appendix G) ----

def preferred_phase(X, labels, harmonic=1):
    """Each feature's preferred phase on a cyclic axis: the phase of the l-th angular harmonic of its class means, the
    classes taken in sorted label order at angles 2 pi k / K (harmonic 1: preferred direction; 2: twice the preferred
    orientation). In [0, 2 pi). Computed from the labels: for inference, compute it on other trials than the ladder's."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    M = np.stack([X[labels == c].mean(axis=0) for c in classes])
    ang = np.linspace(0, 2 * np.pi, len(classes), endpoint=False)
    return np.angle((M * np.exp(1j * harmonic * ang)[:, None]).sum(axis=0)) % (2 * np.pi)


def block_features(X, size, order=None, seed=7):
    """Average consecutive groups of `size` features into one, the features taken in `order` (e.g. np.argsort of a
    preference) or in a random order (seed); the remainder is dropped."""
    X = np.asarray(X)
    d = X.shape[1]
    size = int(size)
    if size < 1 or size > d:
        raise ValueError(f"block_features: block size {size} for {d} features")
    order = np.random.default_rng(seed).permutation(d) if order is None else np.asarray(order).astype(int)
    nb = d // size
    return X[:, order[: nb * size]].reshape(X.shape[0], nb, size).mean(axis=2)


def blocks(X, labels, score, sizes=(1, 2, 4, 8, 16, 32), random_seed=7, n_perm=0, seed=0, floor_seed_base=None, **zoom_kwargs):
    """Appendix G's coarse-graining: at each block size K, the declared ladder on blocks of K features sorted by `score`
    (ascending; e.g. preferred_phase) and on random blocks, each against its own floor. Which features a block pools decides
    which part of the shift survives. floor_seed_base: the floor at block size K is drawn from seed floor_seed_base + K
    (Appendix G's table: 1000, ten draws), else from seed."""
    X = np.asarray(X)
    score = np.asarray(score, dtype=float).ravel()
    if score.shape != (X.shape[1],):
        raise ValueError(f"blocks: score must hold one value per feature ({X.shape[1]}); got {score.shape}")
    sorted_order = np.argsort(score)
    rows = []
    for K in sizes:
        row = {"block_size": int(K), "n_blocks": int(X.shape[1] // K)}
        for tag, order in (("sorted", sorted_order), ("random", None)):
            fs = {} if floor_seed_base is None else {"floor_seed": floor_seed_base + int(K)}
            r = zoom(block_features(X, K, order, random_seed), labels, n_perm=n_perm, seed=seed, **fs, **zoom_kwargs)
            row[tag] = {k: r[k] for k in ("delta", "theta_obs", "theta_floor", "theta_floor_se", "delta_orderavg",
                                          "p_two", "z") if k in r}
        rows.append(row)
    return {"rows": rows}


def _print_blocks(b):
    print("blocks (Appendix G): shift on blocks sorted by the score, and on random blocks, each against its own floor")
    for r in b["rows"]:
        s_, q = r["sorted"], r["random"]
        print(f"  K = {r['block_size']:3d} ({r['n_blocks']:6d} blocks): sorted {s_['delta']:+.3f}  random {q['delta']:+.3f}"
              f"   (floor SE {max(s_['theta_floor_se'], q['theta_floor_se']):.3f})")


# ---- Attention heads (Appendix I's known circuit) ----
# Each head's write at the final token: its block of the attention output projection's input times its rows of W_O, a vector in
# the residual stream. The projection module per architecture; the run stops unless the summed writes plus the bias reproduce
# the projection's output, so an unlisted or changed architecture fails loudly.
_HEAD_PROJ = {
    "gpt2": (lambda m: m.transformer.h, lambda b: b.attn.c_proj, "conv1d"),
    "gpt_neox": (lambda m: m.gpt_neox.layers, lambda b: b.attention.dense, "linear"),
    "gpt_neo": (lambda m: m.transformer.h, lambda b: b.attn.attention.out_proj, "linear"),
    **{t: (lambda m: m.model.layers, lambda b: b.self_attn.o_proj, "linear")
       for t in ("llama", "mistral", "qwen2", "qwen3", "olmo", "olmo2", "gemma", "gemma2", "phi3")},
}


def _from_pretrained(cls, name, dtype, revision=None):
    """Load with the dtype keyword the installed transformers expects (dtype from 4.56, torch_dtype before)."""
    import transformers
    major, minor = (int(v) for v in transformers.__version__.split(".")[:2])
    kw = {"dtype" if (major, minor) >= (4, 56) else "torch_dtype": dtype}
    if revision:
        kw["revision"] = revision
    return cls.from_pretrained(name, **kw)


def head_writes(model_name, prompts, device="cpu", revision=None, max_len=128, dtype=None, layers=None):
    """Every attention head's write into the residual stream at each prompt's final token: an array (layers, heads,
    prompts, d_model) in float32. layers: the layer indices to keep (default all)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = {None: torch.float32, "fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    model = _from_pretrained(AutoModelForCausalLM, model_name, dt, revision).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision) if revision else AutoTokenizer.from_pretrained(model_name)
    mt = model.config.model_type
    if mt not in _HEAD_PROJ:
        raise SystemExit(f"--heads knows the attention output projection of {sorted(_HEAD_PROJ)}; {model_name} is {mt!r}")
    blocks_of, proj_of, kind = _HEAD_PROJ[mt]
    blocks = list(blocks_of(model))
    keep = list(range(len(blocks))) if layers is None else list(layers)
    cfg = model.config
    nh = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head")
    dh = getattr(cfg, "head_dim", None) or (getattr(cfg, "hidden_size", None) or cfg.n_embd) // nh
    Z = {}

    def pre_hook(i):
        def f(mod, args):
            Z[i] = args[0][0, -1].detach().float()
        return f
    hooks = [proj_of(blocks[i]).register_forward_pre_hook(pre_hook(i)) for i in keep]
    d_model = None
    out = None
    try:
        with torch.no_grad():
            for p_i, p in enumerate(prompts):
                ids = tok(p, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
                model(ids)
                for j, i in enumerate(keep):
                    P = proj_of(blocks[i])
                    W = P.weight.float()
                    z = Z[i]
                    if kind == "conv1d":                                  # GPT-2 Conv1D: weight (in, out)
                        w = torch.stack([z[h * dh:(h + 1) * dh] @ W[h * dh:(h + 1) * dh] for h in range(nh)])
                    else:                                                 # nn.Linear: weight (out, in)
                        w = torch.stack([z[h * dh:(h + 1) * dh] @ W[:, h * dh:(h + 1) * dh].T for h in range(nh)])
                    if out is None:
                        d_model = w.shape[1]
                        gb = len(keep) * nh * len(prompts) * d_model * 4 / 1e9
                        if gb > 4:
                            raise SystemExit(f"--heads would hold {gb:.1f} GB of head writes; pass --heads-layers to keep fewer layers")
                        out = np.zeros((len(keep), nh, len(prompts), d_model), np.float32)
                    if p_i == 0:
                        full = P(z[None].to(P.weight.dtype)).float()[0]
                        rec = w.sum(0) + (P.bias.float() if getattr(P, "bias", None) is not None else 0)
                        err = float((full - rec).abs().max())
                        if err > 1e-3 * (1 + float(full.abs().max())):
                            raise SystemExit(f"--heads: the head writes of layer {i} do not reproduce the projection's output "
                                             f"(max error {err:.2e}); the projection map for {mt!r} is wrong for this model")
                    out[j, :, p_i] = w.cpu().numpy()
                if p_i and p_i % 50 == 0:
                    print(f"    encoded {p_i}/{len(prompts)}", flush=True)
    finally:
        for hk in hooks:
            hk.remove()
    return out, keep


def _auc(scores, positive):
    pos, neg = scores[positive], scores[~positive]
    return float((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())


def head_scores(H, labels, layers, paper_seeds=False, n_perm=100, k_orders=50, n_floor_draws=20, seed=0, split=True):
    """Score every head along the declared axis: the order-averaged shift of its write and its permutation z (each null
    draw averaged over its own random orders), the class-mean variance fraction Tr B / Tr C, and with split the shift's
    within-class and class-mean parts (decompose), which say whether a head scores through its class means or its
    fluctuations. paper_seeds replays Appendix I's run: floor seed 100 layer + head + 1, orders from seed 3801, null draws
    from seed 4800 (a label permutation, then its own k_orders orders, per draw)."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    lab = np.searchsorted(classes, labels)
    nc = len(classes)
    n = len(lab)
    bc = _rung_classes(nc, None)
    mem = [np.where(lab == c)[0] for c in range(nc)]
    orng = np.random.default_rng(3801 if paper_seeds else seed + 3801)
    orders = [orng.permutation(nc).tolist() for _ in range(k_orders)]
    nrng = np.random.default_rng(4800 if paper_seeds else seed + 4800)
    null_draws = []
    for _ in range(n_perm):
        pl = lab[nrng.permutation(n)]
        null_draws.append((pl, [nrng.permutation(nc).tolist() for _ in range(k_orders)]))
    rows = []
    for j, l in enumerate(layers):
        for h in range(H.shape[1]):
            X = H[j, h].astype(np.float64)
            Xs = X / (X.std() + 1e-9)
            K = Xs @ Xs.T
            sizes = _ladder(K, mem, list(range(nc)), bc)[1]
            frng = np.random.default_rng(l * 100 + h + 1 if paper_seeds else [seed, l, h])
            lg = np.array([[np.log(max(_subset_pr(K, frng.choice(n, s_, replace=False)), 1e-9)) for s_ in sizes]
                           for _ in range(n_floor_draws)])
            th_f = _slope(sizes, np.exp(lg.mean(axis=0)))
            # every ladder below is a union of whole classes: class-block sums of the centred Gram give its PRs exactly
            Xg = Xs - Xs.mean(axis=0)
            Kg = Xg @ Xg.T
            dbar = float(_union_ladder_slopes(_class_blocks(Kg, lab, nc), orders, bc).mean()) - th_f
            vals = np.array([_union_ladder_slopes(_class_blocks(Kg, pl, nc), os_, bc).mean() - th_f for pl, os_ in null_draws])
            z = float((dbar - vals.mean()) / vals.std()) if vals.std() > 0 else 0.0
            Xc = X - X.mean(0)
            M = np.stack([X[lab == c].mean(0) for c in range(nc)]) - X.mean(0)
            frac = float((np.bincount(lab) @ (M ** 2).sum(1)) / max((Xc ** 2).sum(), 1e-300))
            row = {"head": f"{l}.{h}", "layer": int(l), "index": int(h), "delta_orderavg": dbar, "z": z,
                   "p_two": _null_stats(dbar, vals)[1], "classmean_frac": frac}
            if split:
                dc = decompose(X, lab, n_null=10, seed=l * 100 + h + 1)
                if dc is not None:
                    f = dc["delta_split_free"]
                    row.update(within=f["D"], class_mean=f["Tb"] + f["Cb"] + f["Cx"], delta_declared=dc["delta"])
            rows.append(row)
        print(f"  layer {l}: top z " + ", ".join(f"{r['head']} {r['z']:+.1f}" for r in sorted(rows[-H.shape[1]:], key=lambda r: -r["z"])[:3]),
              flush=True)
    return rows


def llm_heads(model_name, axis_file, device="cpu", revision=None, max_len=128, dtype=None, paper_seeds=False, n_perm=100,
              k_orders=50, seed=0, layers=None, known=None, out_path=None):
    """`rung llm --heads`: every attention head scored along one declared axis (Appendix I's known-circuit test). known: head
    names 'layer.index' of a set found by other means (e.g. path patching); the AUC of z, of the class-mean fraction and
    of each part against it is reported. A training-free pre-screen that ranks heads for causal follow-up."""
    import json as _json
    name, cls, prompts, labels, _ = _load_axis(axis_file, use_strata=False)
    print(f"[{name}] {len(cls)} classes, {len(prompts)} prompts; head writes of {model_name}", flush=True)
    H, keep = head_writes(model_name, prompts, device=device, revision=revision, max_len=max_len, dtype=dtype, layers=layers)
    rows = head_scores(H, labels, keep, paper_seeds=paper_seeds, n_perm=n_perm, k_orders=k_orders, seed=seed)
    res = {"model": model_name, "revision": revision, "axis": name, "classes": cls, "n_prompts": len(prompts),
           "paper_seeds": bool(paper_seeds), "n_perm": n_perm, "k_orders": k_orders, "heads": rows}
    if known:
        names = np.array([r["head"] for r in rows])
        pos = np.isin(names, list(known))
        if pos.any() and (~pos).any():
            res["known"] = list(known)
            for key in ("z", "classmean_frac", "within", "class_mean", "delta_orderavg"):
                if all(key in r for r in rows):
                    res[f"AUC_{key}"] = _auc(np.array([r[key] for r in rows]), pos)
            zr = (-np.array([r["z"] for r in rows])).argsort().argsort() + 1
            res["rank_z"] = {h: int(zr[names == h][0]) for h in known if h in names}
            print("  AUC against the known heads: " + ", ".join(f"{k[4:]} {v:.3f}" for k, v in res.items() if k.startswith("AUC_")), flush=True)
    top = sorted(rows, key=lambda r: -r["z"])[:10]
    print("  heads ranked by z: " + ", ".join(f"{r['head']} {r['z']:+.1f}" for r in top), flush=True)
    if out_path:
        _json.dump(res, open(out_path, "w"), indent=1)
        print(f"wrote {out_path}", flush=True)
    return res


# ---- Loading recordings: arrays, MATLAB files and Neurodata Without Borders (NWB) ----

def _load_array(path):
    """npy, npz (path.npz:key), mat (path.mat:variable; v7.3 files need h5py), csv or whitespace text. Without a key, an
    npz or mat file must hold exactly one array."""
    import os
    key = None
    for ext in (".npz", ".mat"):
        if ext + ":" in path:
            path, key = path.split(ext + ":", 1)
            path += ext
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        names = list(z.files)
    elif path.endswith(".mat"):
        try:
            from scipy.io import loadmat
            z = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
        except NotImplementedError:                 # MATLAB v7.3 = HDF5; h5py reads MATLAB's column-major arrays transposed
            try:
                import h5py
            except ImportError:
                raise SystemExit(f"{path} is a MATLAB v7.3 file: pip install h5py (or pip install -e '.[nwb]')")
            f = h5py.File(path, "r")
            z = {k: np.asarray(f[k]).T for k in f.keys() if isinstance(f[k], h5py.Dataset)}
        names = list(z)
    else:
        if not os.path.exists(path):
            raise SystemExit(f"no such file: {path}")
        return np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)
    if key is None:
        if len(names) != 1:
            raise SystemExit(f"{path} holds {len(names)} arrays ({', '.join(names[:8])}); name one as {path}:<name>")
        key = names[0]
    if key not in names:
        raise SystemExit(f"{path} has no array {key!r}; it holds {', '.join(names[:8])}")
    return np.asarray(z[key])


def nwb_counts(path, intervals, label, strata=None, window=None, units_where=None):
    """Trials-by-units spike counts from an NWB file, read with h5py (pip install -e '.[nwb]'). intervals: a table under
    /intervals (e.g. 'trials', 'drifting_gratings_presentations'); label (and strata): its columns, coded as integers in
    sorted order of their values, rows whose label is missing or NaN dropped; window: (start, stop) in seconds after each
    interval's start_time, else [start_time, stop_time); units_where: (column, value) keeps the units whose column equals
    value, looked up in /units or, through peak_channel_id or electrodes, in the electrodes table. Returns a dict with X
    (rows = intervals, columns = units), labels, label_values, strata, strata_values, times (each row's counting window,
    for --sample-times), unit_ids and, when the file has /intervals/invalid_times, invalid (start, stop) and their tags."""
    try:
        import h5py
    except ImportError:
        raise SystemExit("rung nwb needs h5py: pip install h5py (or pip install -e '.[nwb]')")

    def col(g, name):
        a = g[name][()]
        if a.dtype.kind in "SO":
            a = np.array([v.decode() if isinstance(v, bytes) else str(v) for v in a], dtype=object)
        return a

    with h5py.File(path, "r") as f:
        if "intervals" not in f or intervals not in f["intervals"]:
            have = list(f["intervals"].keys()) if "intervals" in f else []
            raise SystemExit(f"{path}: no /intervals/{intervals}; tables: {', '.join(have)}")
        g = f["intervals"][intervals]
        start, stop = g["start_time"][()].astype(float), g["stop_time"][()].astype(float)
        lab_raw = col(g, label)
        str_raw = col(g, strata) if strata else None
        u = f["units"]
        st, sti = u["spike_times"][()], u["spike_times_index"][()]
        uid = u["id"][()] if "id" in u else np.arange(len(sti))
        keep_u = np.ones(len(sti), bool)
        if units_where is not None:
            cname, val = units_where
            if cname in u:
                vals = col(u, cname)
            else:
                el = f["general/extracellular_ephys/electrodes"]
                if cname not in el:
                    raise SystemExit(f"{path}: neither /units nor the electrodes table has a column {cname!r}")
                emap = dict(zip(el["id"][()], col(el, cname)))
                if "peak_channel_id" in u:
                    vals = np.array([emap.get(c, "") for c in u["peak_channel_id"][()]], dtype=object)
                elif "electrodes" in u and "electrodes_index" in u:
                    ei, eix = u["electrodes"][()], u["electrodes_index"][()]
                    first = np.concatenate([[0], eix[:-1]])
                    ids = el["id"][()]
                    vals = np.array([emap.get(ids[ei[a]], "") for a in first], dtype=object)
                else:
                    raise SystemExit(f"{path}: cannot map units to the electrodes table")
            keep_u = np.array([str(v) == str(val) for v in vals])
            if not keep_u.any():
                raise SystemExit(f"{path}: no unit has {cname} = {val}")
        inv = None
        if "invalid_times" in f["intervals"]:
            gi = f["intervals"]["invalid_times"]
            inv = np.stack([gi["start_time"][()], gi["stop_time"][()]], 1).astype(float)
            tags = None
            if "tags" in gi and "tags_index" in gi:
                tg, tix = col(gi, "tags"), gi["tags_index"][()]
                b = np.concatenate([[0], tix])
                tags = [[str(t) for t in tg[b[i]:b[i + 1]]] for i in range(len(tix))]
    def missing(v):
        if isinstance(v, str):
            return v.strip().lower() in ("", "nan", "null", "none")
        return not np.isfinite(float(v))
    ok = np.array([not missing(v) for v in lab_raw])
    if str_raw is not None:
        ok &= np.array([not missing(v) for v in str_raw])
    a0 = start[ok] + (window[0] if window else 0.0)
    a1 = start[ok] + window[1] if window else stop[ok]
    bounds = np.concatenate([[0], sti])
    cols = np.where(keep_u)[0]
    Xc = np.zeros((int(ok.sum()), len(cols)), dtype=np.int32)
    for j, uj in enumerate(cols):
        sp = np.sort(st[bounds[uj]:bounds[uj + 1]])
        Xc[:, j] = np.searchsorted(sp, a1) - np.searchsorted(sp, a0)

    def code(v):
        try:
            vv = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            vv = np.asarray([str(x) for x in v], dtype=object)
        values = np.unique(vv)
        return np.searchsorted(values, vv), [x.item() if hasattr(x, "item") else x for x in values]
    lab, lvals = code(lab_raw[ok])
    out = {"X": Xc, "labels": lab, "label_values": lvals, "times": np.stack([a0, a1], 1), "unit_ids": uid[cols]}
    if str_raw is not None:
        out["strata"], out["strata_values"] = code(str_raw[ok])
    if inv is not None:
        out["invalid"] = inv
        if tags is not None:
            out["invalid_tags"] = tags
    return out


def _plot_features(r, out_png):
    """Feature-ladder figure: the observed ladder (neighbours or the given order) against random features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(r["sizes"], r["pr_floor"], "o-", color="0.55", label="floor (random features)")
    ax.plot(r["sizes"], r["pr_obs"], "o-", color="#2c6fbb",
            label="nearest neighbours of anchors" if r["mode"] == "positions" else "features in the given order")
    ax.set_xscale("log"); ax.set_yscale("log")
    for axis_ in (ax.xaxis, ax.yaxis):
        axis_.set_major_formatter(ScalarFormatter()); axis_.set_minor_formatter(NullFormatter())
    ax.set_xticks(r["sizes"]); ax.set_xticklabels([str(v) for v in r["sizes"]], fontsize=7)
    ax.set_xlabel("features in rung"); ax.set_ylabel("participation ratio")
    ax.set_title(f"shift {r['delta']:+.3f}: theta_obs {r['theta_obs']:+.3f} = floor {r['theta_floor']:+.3f} + shift", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return out_png


def _json_default(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    raise TypeError(type(v))


def _int_list(s):
    return [int(v) for v in s.split(",") if v.strip()]


def main():
    import argparse
    import json as _json
    import os
    fmt = argparse.ArgumentDefaultsHelpFormatter
    ap = argparse.ArgumentParser(
        prog="rung", formatter_class=fmt,
        description="Axis-resolved dimensionality scaling: theta_obs = theta_floor + delta. A reported exponent split into "
                    "the floor that random subsets of the same sizes return and the shift a declared axis adds, with "
                    "permutation and nuisance-preserving nulls.")
    ap.add_argument("--version", action="version", version=f"rung {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", formatter_class=fmt, help="any samples-by-features array with one label per row")
    d.add_argument("X", help="samples-by-features array (npy, npz:key, mat:variable, csv, txt)")
    d.add_argument("labels", nargs="?", default=None, help="integer class labels, one per sample (optional with --candidates)")
    d.add_argument("--transpose", action="store_true", help="X is stored features by samples: transpose it on loading")
    d.add_argument("--strata", default=None, help="nuisance stratum ids, one per sample, for the nuisance-preserving null")
    d.add_argument("--n-perm", type=int, default=500, help="label permutations for the shift's null (0 disables)")
    d.add_argument("--k-orders", type=int, default=50, help="random class orders for the order-averaged shift")
    d.add_argument("--rungs", default=None, help="classes per rung, comma-separated (default 1,2,3,4,6,8 truncated to the classes, plus the full set)")
    d.add_argument("--seed", type=int, default=0, help="random seed for floors, permutations and orders")
    d.add_argument("--floor-draws", type=int, default=20, help="random-subset ladders averaged into the floor")
    d.add_argument("--floor-seed", type=int, default=None, help="a separate seed for the floor (default: --seed's stream)")
    d.add_argument("--floor-by-rung", action="store_true", help="draw the floor rung by rung, the order of the paper's generators")
    d.add_argument("--paper-floor", action="store_true", help="the floor of the paper's imaging, Allen and macaque generators: ten draws, seed 42, rung by rung (replays their printed shifts and splits exactly)")
    d.add_argument("--out", default=None, help="write the result JSON here")
    d.add_argument("--plot", default=None, help="write a ladder figure (PNG) here")
    d.add_argument("--standardize", action="store_true", help="z-score every feature first (the repair when one dimension dominates the spectrum)")
    d.add_argument("--split-by", default=None, help="per-feature score file (one value per feature): also run the top, bottom and a random third of the features at matched size, each with its own floor (Section 4)")
    d.add_argument("--order", default=None, help="the declared accumulation order: sequential (default), antipodal (one member of each antipodal pair first), or a comma-separated list of class labels")
    d.add_argument("--sectors", action="store_true", help="the harmonics of the class-mean kernel for a cyclic axis (labels in cyclic order, class k at angle 2 pi k/K): sector balance, even and odd amplitudes, coherence of adjacent and antipodal classes (Section 4)")
    d.add_argument("--null-shift", action="store_true", help="labels are in time order: add the circular-shift null, which rolls the label sequence and keeps its autocorrelation")
    d.add_argument("--antipode", default=None, help="JSON {class: antipodal class}; checks that the declared order's first half holds one member of each pair and reports late_fraction as the share of the climb left after half the circle (Appendices E and K)")
    d.add_argument("--decompose", action="store_true", help="the shift split into its within-class part D and class-mean part Tb + Cb + Cx (exact four-term split, Section 5, Appendix K), with the sign law of Eq. 3")
    d.add_argument("--n-shuffle", type=int, default=5, help="with --decompose: label permutations for each term's sampling baseline (0 disables)")
    d.add_argument("--n-bias", type=int, default=0, help="with --decompose: surrogates of the finite-sample bias null (class means kept, one covariance shape for every class)")
    d.add_argument("--account", action="store_true", help="the two-term account, Figure 3 for this dataset: the measured within-class part against the overlap law and the finite-sample identity, the class-mean part against Eq. 3 through the rungs, their sum against the shift, and the sampling term")
    d.add_argument("--account-halves", default=None, help="with --account: one 0/1 per sample for sigma's split halves (default: a random half of each class)")
    d.add_argument("--no-heldout", action="store_true", help="with --account: skip the held-out arm (sigma and moments from one half, the part measured on the other)")
    d.add_argument("--time-blocks", default=None, help="rows are in acquisition order: an integer E cuts them into contiguous blocks of E rows, or a file gives one block id per row; adds the null that permutes labels only within blocks")
    d.add_argument("--cov-floors", action="store_true", help="the principal-angle dissimilarity between classes against its label-blind floor, predicted from the pooled spectrum and the samples' gain kurtosis (Appendix H)")
    d.add_argument("--cov-k", type=int, default=5, help="with --cov-floors: top residual directions per class")
    d.add_argument("--cov-perm", type=int, default=0, help="with --cov-floors: also this many label permutations, to check the predicted floor")
    d.add_argument("--valid", default=None, help="per-sample validity (1 keep, 0 drop), e.g. presentations a data release marks invalid; dropped before every statistic")
    d.add_argument("--sample-times", default=None, help="n x 2 array of each sample's [start, stop]; with --invalid-intervals builds the validity mask")
    d.add_argument("--invalid-intervals", default=None, help="k x 2 array of invalid [start, stop] intervals (e.g. Allen's invalid_times); a sample overlapping any is dropped")
    d.add_argument("--candidates", nargs="+", default=None, help="label files to screen against each other: each one's shift against its permutations, ranked by z (Section 8, Appendix H)")
    d.add_argument("--candidate-strata", choices=["none", "others"], default="none", help="with --candidates: permute each candidate plainly, or inside the joint cells of the other candidates (the paper's screen); --strata adds a nuisance to either")
    d.add_argument("--halves", default=None, help="with --candidates: 'oddeven' (row parity) or a file of 0/1 per row: discover on one half, replicate on the other")
    d.add_argument("--block-by", default=None, help="coarse-grain features into blocks sorted by a score: phase1 (preferred direction from the labels' class means), phase2 (preferred orientation), random, or a score file (Appendix G)")
    d.add_argument("--block-size", default="1,2,4,8,16,32", help="with --block-by: features per block, comma-separated")
    d.add_argument("--block-perm", type=int, default=0, help="with --block-by: label permutations per block size (0: shift and floor SE only)")

    fe = sub.add_parser("features", formatter_class=fmt, help="a ladder over features (neighbouring neurons, or features in a given order) against random features")
    fe.add_argument("X", help="samples-by-features array")
    fe.add_argument("--transpose", action="store_true", help="X is stored features by samples")
    g = fe.add_mutually_exclusive_group(required=True)
    g.add_argument("--positions", help="n_features x D coordinates (e.g. cell positions): rungs of nearest neighbours of random anchors")
    g.add_argument("--order", help="every feature index once: rungs of the first n features")
    g.add_argument("--score", help="one value per feature: rungs of the n highest-scoring features")
    fe.add_argument("--sizes", default=",".join(str(v) for v in FEATURE_SIZES_DEFAULT), help="features per rung (sizes above the feature count are dropped)")
    fe.add_argument("--anchors", type=int, default=50, help="anchors per rung (with --positions)")
    fe.add_argument("--floor-draws", type=int, default=50, help="random feature subsets per floor rung")
    fe.add_argument("--n-null", type=int, default=0, help="relabellings of the features for a p-value in this population")
    fe.add_argument("--seed", type=int, default=42, help="anchor seed (the floor uses seed + 958, the paper's 1000 at the default)")
    fe.add_argument("--out", default=None, help="write the result JSON here")
    fe.add_argument("--plot", default=None, help="write the ladder figure (PNG) here")

    nw = sub.add_parser("nwb", formatter_class=fmt, help="trials-by-units spike counts, labels and times from an NWB file (needs h5py)")
    nw.add_argument("nwb_file")
    nw.add_argument("--intervals", required=True, help="table under /intervals, e.g. trials or drifting_gratings_presentations")
    nw.add_argument("--label", required=True, help="the table's column that declares the axis")
    nw.add_argument("--strata", default=None, help="a second column written as strata (e.g. temporal_frequency)")
    nw.add_argument("--window", type=float, nargs=2, default=None, metavar=("START", "STOP"), help="count window in seconds after each interval's start (default: the interval itself)")
    nw.add_argument("--units-where", default=None, help="COLUMN=VALUE keeps matching units (looked up in /units or the electrodes table, e.g. location=VISp)")
    nw.add_argument("--out", required=True, help="output prefix: writes PREFIX_X.npy, _labels.npy, _times.npy, _units.npy, _meta.json (+ _strata.npy, _invalid.npy)")

    m = sub.add_parser("llm", formatter_class=fmt, help="every layer of a Hugging Face model on a prompt battery")
    m.add_argument("--model", required=True)
    m.add_argument("--axis", required=True, nargs="+", help="axis JSON file(s) {class: [prompts]} and/or directories of them")
    m.add_argument("--revision", default=None, help="Hugging Face revision, e.g. a training checkpoint")
    m.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    m.add_argument("--n-perm", type=int, default=500, help="label permutations per layer")
    m.add_argument("--k-orders", type=int, default=50, help="random class orders for the order-averaged shift")
    m.add_argument("--max-len", type=int, default=128, help="tokens per prompt")
    m.add_argument("--no-strata", action="store_true", help="ignore *.strata.json sidecars")
    m.add_argument("--out", default=None, help="write JSON here")
    m.add_argument("--paper-seeds", action="store_true", help="the paper's seeding convention (reproduces the published per-layer cells)")
    m.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None, help="model dtype; default float16 on mps/cuda, float32 on cpu; bf16 repairs float16 overflow in early checkpoints")
    m.add_argument("--standardize", action="store_true", help="z-score every unit per layer before the Gram (z frame only)")
    m.add_argument("--frames", choices=["linear", "z", "both"], default="both", help="feature frame(s): the linear Gram (the paper's), per-unit z-scored, or both with a sign-disagreement flag")
    m.add_argument("--match-last-token", action="store_true", help="per layer, the last-token-matched and size-matched random subsamples (Appendix I's last-token control)")
    m.add_argument("--heads", action="store_true", help="score every attention head's write at the final token along the axis: order-averaged shift, permutation z, class-mean fraction and the split (Appendix I's known circuit); a training-free pre-screen that ranks heads for causal follow-up")
    m.add_argument("--heads-layers", default=None, help="with --heads: layer indices to keep, comma-separated (default all)")
    m.add_argument("--heads-known", default=None, help="with --heads: heads found by other means, as layer.index, comma-separated: prints the AUC of each score against them")
    m.add_argument("--heads-perm", type=int, default=100, help="with --heads: label permutations per head, each averaged over --k-orders orders")

    s = sub.add_parser("summarize", formatter_class=fmt, help="plain-language reading of a rung JSON (the paper's rules)")
    s.add_argument("battery_json")
    s.add_argument("--out", default=None, help="also write the report as JSON")

    x = sub.add_parser("axis", formatter_class=fmt, help="build an axis JSON (+ strata sidecar) from a Hugging Face dataset")
    x.add_argument("--dataset", required=True, help="dataset name, e.g. Rowan/hellaswag")
    x.add_argument("--config", default=None)
    x.add_argument("--split", default="train")
    x.add_argument("--text-field", required=True)
    x.add_argument("--label-field", required=True)
    x.add_argument("--strata-field", default=None, help="nuisance field written to the .strata.json sidecar")
    x.add_argument("--classes", nargs="*", default=None, help="explicit class values (default: the most frequent)")
    x.add_argument("--n-classes", type=int, default=8)
    x.add_argument("--n-per-class", type=int, default=16)
    x.add_argument("--min-chars", type=int, default=20)
    x.add_argument("--max-chars", type=int, default=400)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--out", required=True)

    p = sub.add_parser("plot", formatter_class=fmt, help="the figure for a rung JSON: depth profile (llm), ladder (data) or feature ladder (features)")
    p.add_argument("battery_json")
    p.add_argument("--out", default="battery.png")

    rp = sub.add_parser("report", formatter_class=fmt, help="one self-contained HTML page for a rung JSON: figure, reading, numbers")
    rp.add_argument("result_json")
    rp.add_argument("--out", default=None, help="the page (default: the JSON's name with .html)")
    rp.add_argument("--title", default=None)

    a = ap.parse_args()
    if a.cmd == "summarize":
        dj = _json.load(open(a.battery_json))
        if "layers" in dj or "axes" in dj:
            summarize(a.battery_json, a.out)
        elif "mode" in dj and "pr_obs" in dj and "rung_sizes" not in dj:
            _print_feature_ladder(dj)
        else:
            rep = summarize_data(dj)
            if a.out:
                _json.dump(rep, open(a.out, "w"), indent=1)
    elif a.cmd == "plot":
        dj = _json.load(open(a.battery_json))
        if "layers" in dj or "axes" in dj:
            _plot_battery(a.battery_json, a.out)
        elif "mode" in dj and "pr_obs" in dj and "rung_sizes" not in dj:
            _plot_features(dj, a.out); print("wrote", a.out)
        else:
            _plot_data(dj, a.out); print("wrote", a.out)
    elif a.cmd == "report":
        print("wrote", report(a.result_json, a.out, title=a.title))
    elif a.cmd == "axis":
        axis_from_dataset(a.dataset, a.text_field, a.label_field, a.out, config=a.config, split=a.split,
                          n_classes=a.n_classes, n_per_class=a.n_per_class, strata_field=a.strata_field,
                          classes=a.classes, min_chars=a.min_chars, max_chars=a.max_chars, seed=a.seed)
    elif a.cmd == "nwb":
        uw = None
        if a.units_where:
            if "=" not in a.units_where:
                raise SystemExit("--units-where takes COLUMN=VALUE")
            uw = tuple(a.units_where.split("=", 1))
        r = nwb_counts(a.nwb_file, a.intervals, a.label, strata=a.strata, window=a.window, units_where=uw)
        np.save(a.out + "_X.npy", r["X"]); np.save(a.out + "_labels.npy", r["labels"])
        np.save(a.out + "_times.npy", r["times"]); np.save(a.out + "_units.npy", r["unit_ids"])
        if "strata" in r:
            np.save(a.out + "_strata.npy", r["strata"])
        if "invalid" in r:
            np.save(a.out + "_invalid.npy", r["invalid"])
        meta = {"nwb_file": os.path.basename(a.nwb_file), "intervals": a.intervals, "label": a.label,
                "label_values": r["label_values"], "strata": a.strata, "strata_values": r.get("strata_values"),
                "window": a.window, "units_where": a.units_where, "n_samples": int(r["X"].shape[0]),
                "n_units": int(r["X"].shape[1]), "invalid_tags": r.get("invalid_tags")}
        _json.dump(meta, open(a.out + "_meta.json", "w"), indent=1, default=_json_default)
        print(f"{a.nwb_file}: {r['X'].shape[0]} intervals x {r['X'].shape[1]} units; {len(r['label_values'])} classes of "
              f"{a.label} {r['label_values'][:12]}" + (f"; {len(r['invalid'])} invalid intervals in {a.out}_invalid.npy "
              "(filter them by the tags in _meta.json before --invalid-intervals)" if "invalid" in r else ""))
        print(f"next: rung data {a.out}_X.npy {a.out}_labels.npy --sample-times {a.out}_times.npy ...")
    elif a.cmd == "features":
        X = _load_array(a.X)
        X = X.T if a.transpose else X
        order = None
        if a.order:
            order = _load_array(a.order).astype(int).ravel()
        elif a.score:
            sc = _load_array(a.score).astype(float).ravel()
            order = np.argsort(-sc, kind="stable")
        pos = _load_array(a.positions) if a.positions else None
        r = feature_ladder(X, positions=pos, order=order, sizes=_int_list(a.sizes), n_anchors=a.anchors,
                           n_floor_draws=a.floor_draws, seed=a.seed, floor_seed=a.seed + 958, n_null=a.n_null)
        _print_feature_ladder(r)
        r["rung_version"] = __version__
        if a.out:
            _json.dump(r, open(a.out, "w"), indent=1, default=_json_default); print("wrote", a.out)
        if a.plot:
            _plot_features(r, a.plot); print("wrote", a.plot)
    elif a.cmd == "data":
        X = _load_array(a.X)
        X = X.T if a.transpose else X
        if a.labels is None and not a.candidates:
            raise SystemExit("rung data needs labels (or --candidates)")
        labels = _load_array(a.labels).astype(int).ravel() if a.labels else None
        strata = _load_array(a.strata).astype(int).ravel() if a.strata else None
        cand = {os.path.splitext(os.path.basename(f))[0]: _load_array(f).astype(int).ravel() for f in (a.candidates or [])}
        n_rows = len(X)
        for name_, arr_ in ([("labels", labels)] if labels is not None else []) + list(cand.items()) + (
                [("strata", strata)] if strata is not None else []):
            if len(arr_) != n_rows:
                raise SystemExit(f"{name_} has {len(arr_)} entries for {n_rows} rows of X (rows must be samples; --transpose flips X)")
        antipode = None
        if a.antipode:
            antipode = {int(k): int(v) for k, v in _json.load(open(a.antipode)).items()}
        order = a.order
        if order not in (None, "sequential", "antipodal"):
            order = [int(v) for v in order.split(",")]
        rungs = tuple(_int_list(a.rungs)) if a.rungs else None
        tb = None
        if a.time_blocks:
            tb = block_ids(n_rows, int(a.time_blocks)) if a.time_blocks.isdigit() else _load_array(a.time_blocks).astype(int).ravel()
        halves_c = None
        if a.halves:
            halves_c = "oddeven" if a.halves == "oddeven" else _load_array(a.halves).astype(int).ravel()
        acc_h = _load_array(a.account_halves).astype(int).ravel() if a.account_halves else None
        valid = None
        if a.valid:
            valid = _load_array(a.valid).astype(bool).ravel()
        if a.invalid_intervals:
            if not a.sample_times:
                raise SystemExit("--invalid-intervals needs --sample-times (each sample's start and stop)")
            st = _load_array(a.sample_times).reshape(-1, 2)
            vi = valid_from_intervals(st[:, 0], st[:, 1], _load_array(a.invalid_intervals))
            valid = vi if valid is None else valid & vi
        if valid is not None:
            if len(valid) != n_rows:
                raise SystemExit(f"validity mask has {len(valid)} entries for {n_rows} samples")
            X, labels, strata, tb, acc_h = apply_valid(valid, X, labels, strata, tb, acc_h)   # blocks keep their acquisition-time ids
            cand = {k: v[valid] for k, v in cand.items()}
            if halves_c is not None and not isinstance(halves_c, str):
                halves_c = halves_c[valid]
            print(f"validity mask: {len(X)} of {n_rows} samples kept"
                  + ("; class sizes " + str(dict(zip(*[v.tolist() for v in np.unique(labels, return_counts=True)]))) if labels is not None else ""))
        if a.paper_floor:
            a.floor_draws, a.floor_seed, a.floor_by_rung = 10, 42, True
        fkw = {"n_floor_draws": a.floor_draws, "floor_seed": a.floor_seed, "floor_by_rung": a.floor_by_rung}
        dkw = {"n_null": a.floor_draws, "seed": a.floor_seed if a.floor_seed is not None else a.seed, "floor_by_rung": a.floor_by_rung}
        sp = spectrum(X)
        print(f"spectrum: d_eff_full={sp['d_eff_full']:.1f} top1_eig_frac={sp['top1_eig_frac']:.2f} top1_dim_var_frac={sp['top1_dim_var_frac']:.2f}"
              + ("  <- one dimension dominates; consider --standardize" if sp["top1_eig_frac"] > 0.5 else ""))
        r = {}
        if labels is not None:
            r = zoom(X, labels, n_perm=a.n_perm, k_orders=a.k_orders, strata=strata, seed=a.seed, standardize=a.standardize,
                     antipode=antipode, order=order, shift_null=a.null_shift, time_blocks=tb, bin_counts=rungs, **fkw)
            if not a.standardize:
                rz = zoom(X, labels, n_perm=0, k_orders=a.k_orders, seed=a.seed, standardize=True, order=order, bin_counts=rungs, **fkw)
                r["delta_z_frame"], r["delta_orderavg_z_frame"] = rz["delta"], rz["delta_orderavg"]
                oa = f" (order-averaged {rz['delta_orderavg']:+.3f})" if np.isfinite(rz["delta_orderavg"]) else ""
                print(f"frame check: per-feature z-scored delta {rz['delta']:+.3f}{oa}"
                      + ("  <- OPPOSITE sign to the linear frame: a few high-variance features set the linear reading"
                         if np.sign(rz["delta"]) != np.sign(r["delta"]) else ""))
            for k, v in r.items():
                if not isinstance(v, list):
                    print(f"{k}: {v}")
            if a.split_by:
                score = _load_array(a.split_by).astype(float).ravel()
                sp_r = split_by(X, labels, score, seed=a.seed, n_perm=a.n_perm, k_orders=a.k_orders, strata=strata,
                                standardize=a.standardize, antipode=antipode, order=order, bin_counts=rungs)
                r["split_by"] = sp_r
                print(f"split by score, {sp_r['n_features_per_subset']} of {sp_r['n_features_total']} features per subset (matched size, own floors):")
                for name, rr in sp_r["subsets"].items():
                    print(f"  {name:7s}: delta {rr['delta']:+.3f}" + (f", p = {rr['p_two']:.3f}" if "p_two" in rr else "")
                          + (f", late fraction {rr['late_fraction']:.2f}" if "late_fraction" in rr else "") + f", median score {rr['median_score']:.3g}")
            if a.sectors:
                # the kernel's cycle is the labels' own order (class k sits at angle 2 pi k / K); the accumulation
                # order of --order is a different thing and does not enter here
                sc = sectors(X, labels)
                _print_sectors(sc); r["sectors"] = sc
            if a.cov_floors:
                cf = cov_floors(X, labels, k=a.cov_k, n_perm=a.cov_perm, seed=a.seed)
                _print_cov_floors(cf); r["cov_floors"] = cf
            dec_order = None if order in (None, "sequential") else [int(np.unique(labels)[i]) for i in declared_order(np.unique(labels), order)]
            if a.decompose:
                Xd = _zscore(X) if a.standardize else X
                rd = decompose(Xd, labels, order=dec_order, bin_counts=rungs, n_shuffle=a.n_shuffle, n_bias=a.n_bias, **dkw)
                if rd is None:
                    print("decompose: the ladder has fewer than three rungs of ten samples; nothing to split")
                else:
                    _print_decomposition(rd); r["decompose"] = rd
            if a.account:
                Xa = _zscore(X) if a.standardize else X
                ra = account(Xa, labels, order=dec_order, bin_counts=rungs, halves=acc_h, seed=dkw["seed"], heldout=not a.no_heldout,
                             n_null=a.floor_draws, floor_by_rung=a.floor_by_rung)
                _print_account(ra); r["account"] = ra
            if a.block_by:
                bsz = _int_list(a.block_size)
                if a.block_by in ("phase1", "phase2"):
                    bscore = preferred_phase(X, labels, harmonic=int(a.block_by[-1]))
                elif a.block_by == "random":
                    bscore = np.random.default_rng(a.seed).random(X.shape[1])
                else:
                    bscore = _load_array(a.block_by).astype(float).ravel()
                rb = blocks(X, labels, bscore, sizes=bsz, n_perm=a.block_perm, seed=a.seed, k_orders=0 if not a.block_perm else a.k_orders,
                            order=order, bin_counts=rungs)
                _print_blocks(rb); r["blocks"] = rb
        if cand:
            cs = ({os.path.splitext(os.path.basename(a.labels))[0]: labels, **cand} if labels is not None else cand)
            if len(cs) < len(cand) + (labels is not None):
                raise SystemExit("--candidates: two label files share a name; rename one")
            if a.candidate_strata == "others":
                cst = {k: (_joint_cells([v for kk, v in cs.items() if kk != k] + ([strata] if strata is not None else []))) for k in cs}
            else:
                cst = strata
            rc = candidates(X, cs, strata=cst, halves=halves_c, n_perm=a.n_perm if a.n_perm else 200, seed=a.seed,
                            standardize=a.standardize, n_floor_draws=a.floor_draws, floor_seed=a.floor_seed)
            rc["strata"] = a.candidate_strata + (" + --strata" if strata is not None else "")
            _print_candidates(rc); r["candidates"] = rc
            if labels is None:
                r.update({"delta": float("nan"), "theta_obs": float("nan"), "theta_floor": float("nan"), "n_classes": 0, "rung_sizes": []})
        if valid is not None:
            r["n_samples_total"], r["n_samples_valid"] = int(len(valid)), int(valid.sum())
        r["rung_version"] = __version__
        if a.out:
            _json.dump(r, open(a.out, "w"), indent=1, default=_json_default)
            print("wrote", a.out)
        if a.plot and labels is not None:
            _plot_data(r, a.plot); print("wrote", a.plot)
    elif a.heads:
        for af in _expand_axis_paths(a.axis):
            out = a.out if (a.out and len(_expand_axis_paths(a.axis)) == 1) else (
                (a.out[:-5] if a.out and a.out.endswith(".json") else (a.out or "heads")) + "_" + os.path.splitext(os.path.basename(af))[0] + ".json")
            llm_heads(a.model, af, device=a.device, revision=a.revision, max_len=a.max_len, dtype=a.dtype,
                      paper_seeds=a.paper_seeds, n_perm=a.heads_perm, k_orders=a.k_orders, seed=0,
                      layers=_int_list(a.heads_layers) if a.heads_layers else None,
                      known=[v.strip() for v in a.heads_known.split(",")] if a.heads_known else None, out_path=out)
    else:
        llm_battery(a.model, a.axis, a.device, a.n_perm, a.k_orders, a.max_len,
                    a.out, paper_seeds=a.paper_seeds, revision=a.revision,
                    use_strata=not a.no_strata, standardize=a.standardize, dtype=a.dtype,
                    frames=a.frames, match_last_token=a.match_last_token)


if __name__ == "__main__":
    main()
