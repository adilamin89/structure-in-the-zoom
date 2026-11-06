"""rung — the axis-resolved dimensionality-scaling instrument, reusable.

A rung is one step of the subsampling ladder; the tool reads every rung
against its matched floor. (Released as theta-zoom through 1.1.0; that
name stays importable as an alias for one release. 1.3 added the per-rung
deficit, the late fraction and the stall check; 1.4 the exact four-term
split of the deficit; 1.5 the sector fit of the class-mean kernel, the
declared accumulation order, and the circular-shift null for labels in
time order; 1.6 the finite-sample bias null and the sign-law readout of the
split, the time-blocked null, pair-preserving order averaging for paired
axes, and, for language models, the last-token report and both feature
frames.)

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
         antipode=None, order=None, shift_null=False, time_blocks=None):
    """Axis-resolved decomposition of the dimensionality-scaling exponent.

    Parameters
    ----------
    X : (n, d) array — samples by features.
    labels : (n,) int array — class labels of the declared axis. Classes are
        accumulated in label order (0, 1, 2, ...); for unordered axes the
        order-averaged outputs are the partition-level statistics.
    n_perm : label permutations for the per-layer null (0 disables).
    k_orders : random accumulation orders for the order-averaged shift.
    k_null_orders : orders averaged per permuted labeling (conservative if
        smaller than k_orders; see the paper's Appendix G).
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
    antipode : optional dict {class: its antipodal class}. The stall test of
        Appendix K reads `late_fraction` (the deficit at half the
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
    if standardize:
        # per-feature z-score: the repair for populations dominated by a rogue
        # dimension; off by default so --paper-seeds
        # still reproduces the published cells
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-9)
    X = X / (X.std() + 1e-9)
    labels = np.asarray(labels)
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
    for i in range(n_floor_draws):
        for k, s in enumerate(sizes):
            logs[i, k] = np.log(max(
                _subset_pr(K, frng.choice(n, s, replace=False)), 1e-9))
    th_floor = _slope(sizes, np.exp(logs.mean(axis=0)))
    d_obs = th_obs - th_floor

    floor_means = np.exp(logs.mean(axis=0))
    deficit = [float(np.log(max(o, 1e-9)) - np.log(max(f, 1e-9))) for o, f in zip(pr_obs, floor_means)]
    out = {"delta": d_obs, "theta_obs": th_obs, "theta_floor": th_floor,
           "rung_sizes": sizes, "rung_classes": list(bc), "n_classes": n_classes,
           "pr_obs": [float(v) for v in pr_obs],
           "pr_floor": [float(v) for v in floor_means],
           # per-rung log-PR deficit below the floor (zero at the top rung, where the
           # ladder and the floor hold the same samples) and the share of the climb
           # that remains after half the classes: the stall test of the paper's
           # Appendix K when the declared order lists one member of each antipodal
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
                          f"{clash[:3]}; the stall test needs one member of each pair before any partner "
                          "(relabel the classes so the pairs sit half a cycle apart).", stacklevel=2)

    orders = [orng.permutation(n_classes).tolist() for _ in range(k_orders)]
    d_orders = [_ladder(K, members, o, bc)[0] - th_floor for o in orders]
    out["delta_orderavg"] = float(np.mean(d_orders))
    out["delta_orderavg_sd"] = float(np.std(d_orders))

    if n_perm:
        null_c, null_a = [], []
        for _ in range(n_perm):
            pl = lab[prng.permutation(n)]
            mem = [np.where(pl == c)[0] for c in range(n_classes)]
            null_c.append(_ladder(K, mem, canonical, bc)[0] - th_floor)
            null_a.append(np.mean(
                [_ladder(K, mem, o, bc)[0]
                 for o in orders[:k_null_orders]]) - th_floor)
        for tag, obs, null in (("", d_obs, null_c),
                               ("_orderavg", out["delta_orderavg"], null_a)):
            null = np.asarray(null)
            nm, ns = float(null.mean()), float(null.std())
            out[f"z{tag}"] = (obs - nm) / ns if ns > 0 else 0.0
            out[f"p_two{tag}"] = float(
                (1 + (np.abs(null - nm) >= abs(obs - nm)).sum())
                / (len(null) + 1))
            out[f"null_mean{tag}"], out[f"null_sd{tag}"] = nm, ns

    if strata is not None and n_perm:
        strata = np.asarray(strata)
        smem = {s: np.where(strata == s)[0] for s in np.unique(strata)}
        null_s = []
        for _ in range(n_perm):
            sl = lab.copy()
            for idx in smem.values():
                sl[idx] = sl[idx[prng.permutation(len(idx))]]
            mem = [np.where(sl == c)[0] for c in range(n_classes)]
            null_s.append(_ladder(K, mem, canonical, bc)[0] - th_floor)
        null_s = np.asarray(null_s)
        nm, ns = float(null_s.mean()), float(null_s.std())
        out["strat_z"] = (d_obs - nm) / ns if ns > 0 else 0.0
        out["strat_p_two"] = float(
            (1 + (np.abs(null_s - nm) >= abs(d_obs - nm)).sum())
            / (len(null_s) + 1))
        out["strat_null_mean"], out["strat_null_sd"] = nm, ns

    if shift_null and n_perm:
        null_sh = []
        for _ in range(n_perm):
            sl = np.roll(lab, int(prng.integers(1, n)))
            mem = [np.where(sl == c)[0] for c in range(n_classes)]
            null_sh.append(_ladder(K, mem, canonical, bc)[0] - th_floor)
        null_sh = np.asarray(null_sh)
        nm, ns = float(null_sh.mean()), float(null_sh.std())
        out["shift_z"] = (d_obs - nm) / ns if ns > 0 else 0.0
        out["shift_p_two"] = float(
            (1 + (np.abs(null_sh - nm) >= abs(d_obs - nm)).sum())
            / (len(null_sh) + 1))
        out["shift_null_mean"], out["shift_null_sd"] = nm, ns

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
            null_ta.append(np.mean([_ladder(K, mem, o, bc)[0] for o in orders[:k_null_orders]]) - th_floor)
        for tag, obs, null in (("", d_obs, null_t), ("_orderavg", out["delta_orderavg"], null_ta)):
            null = np.asarray(null)
            nm, ns = float(null.mean()), float(null.std())
            out[f"tb_z{tag}"] = (obs - nm) / ns if ns > 0 else 0.0
            out[f"tb_p_two{tag}"] = float((1 + (np.abs(null - nm) >= abs(obs - nm)).sum()) / (len(null) + 1))
            out[f"tb_null_mean{tag}"], out[f"tb_null_sd{tag}"] = nm, ns
        out["n_time_blocks"] = len(bmem)

    if strata is not None:
        groups = pair_groups(lab, np.asarray(strata), n_classes)
        if len(groups) >= 2 and max(len(g) for g in groups) >= 2:
            pp, n_pp = pair_preserving_orders(groups, k_max=5000, rng=orng)
            sub = pp if len(pp) <= max(k_orders, 50) else [pp[i] for i in orng.choice(len(pp), max(k_orders, 50), replace=False)]
            out["pair_groups"] = [[classes[i].item() if hasattr(classes[i], "item") else classes[i] for i in g] for g in groups]
            out["n_pair_orders"] = int(n_pp)
            out["delta_pairavg"] = float(np.mean([_ladder(K, members, o, bc)[0] for o in pp]) - th_floor)
            if n_perm:
                obs_sub = float(np.mean([_ladder(K, members, o, bc)[0] for o in sub]) - th_floor)
                smem = [np.where(np.asarray(strata) == s_)[0] for s_ in np.unique(strata)]
                null_p = []
                for _ in range(n_perm):
                    sl = lab.copy()
                    for idx in smem:
                        sl[idx] = sl[idx[prng.permutation(len(idx))]]
                    mem = [np.where(sl == c)[0] for c in range(n_classes)]
                    null_p.append(np.mean([_ladder(K, mem, o, bc)[0] for o in sub]) - th_floor)
                null_p = np.asarray(null_p)
                nm, ns = float(null_p.mean()), float(null_p.std())
                out["pairavg_z"] = (obs_sub - nm) / ns if ns > 0 else 0.0
                out["pairavg_p_two"] = float((1 + (np.abs(null_p - nm) >= abs(obs_sub - nm)).sum()) / (len(null_p) + 1))
                out["pairavg_null_mean"], out["pairavg_null_sd"] = nm, ns
                out["n_pair_orders_in_null"] = len(sub)

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
    fluctuations overlap (Section 6), not by these class-mean coherences."""
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
              "the order the fluctuations favour is read from their overlaps (Section 6), not from these")


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
    m = X.shape[1] // n_subsets
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
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return out_png


def summarize_data(result, verbose=True):
    """Plain-language reading of one `zoom` result (the paper's rules for a
    single array): sign, certification at the permutation resolution, the
    order-averaged statistic, and the stratified verdict when present."""
    n_perm = int(result.get("n_perm", 0)); floor_p = 1.0 / (n_perm + 1) if n_perm else None
    lines = [f"theta_obs = {result['theta_obs']:+.3f} = floor {result['theta_floor']:+.3f} + shift {result['delta']:+.3f} "
             f"({result['n_classes']} classes, rungs {result['rung_sizes']})"]
    verdict = {"delta": result["delta"]}
    if "p_two" in result:
        cert = result["p_two"] < 0.05
        lines.append(f"declared order: z = {result['z']:+.1f}, two-sided Monte Carlo p = {result['p_two']:.3f} "
                     f"(resolution {floor_p:.3f}); {'label-linked at p < 0.05' if cert else 'not separated from the label-permutation null'}")
        verdict["certified"] = bool(cert)
    if "delta_orderavg" in result:
        oa = result["delta_orderavg"]; same = (oa > 0) == (result["delta"] > 0)
        txt = f"order-averaged shift {oa:+.3f} (SD over orders {result['delta_orderavg_sd']:.3f})"
        if "p_two_orderavg" in result:
            txt += f", p = {result['p_two_orderavg']:.3f}"
        txt += "; same sign as the declared order" if same else "; OPPOSITE sign to the declared order: the declared-order value is a path property, report the order-averaged one for the partition"
        lines.append(txt); verdict["orderavg_same_sign"] = bool(same)
    if "strat_p_two" in result:
        absorbed = result["strat_p_two"] >= 0.05
        lines.append(f"nuisance-preserving null: p = {result['strat_p_two']:.3f}; "
                     + ("the shift is ABSORBED by within-stratum relabeling: it reads nuisance composition, not the labels"
                        if absorbed else "the shift survives within-stratum relabeling: label-linked beyond the nuisance"))
        verdict["label_linked_beyond_strata"] = bool(not absorbed)
    if "deficit" in result:
        lines.append("per-rung log-PR deficit below the floor, by classes accumulated "
                     + ", ".join(f"{c}: {d:+.3f}" for c, d in zip(result.get("rung_classes", []), result["deficit"])))
        if "late_fraction" in result:
            lf = result["late_fraction"]; h = result["late_rung_classes"]
            txt = f"share of the climb remaining after {h} of {result['n_classes']} classes: {lf:.2f}"
            if "antipode_first_half_has_no_pair" in result:
                if result["antipode_first_half_has_no_pair"]:
                    txt += (" (stall test: the first half holds one member of each antipodal pair; near 0 = the code has every class mean by half the classes, "
                            "it reads the quotient; large = the second half still adds directions, the code distinguishes a class from its antipode)")
                else:
                    txt += " (the first half of the declared order holds an antipodal pair, so this is not the stall test; relabel)"
            lines.append(txt); verdict["late_fraction"] = float(lf)
    if "shift_p_two" in result:
        survives = result["shift_p_two"] < 0.05
        lines.append(f"circular-shift null (labels in time order): p = {result['shift_p_two']:.3f}; "
                     + ("the shift survives rolling the label sequence: the alignment of labels with frames matters beyond their slowness"
                        if survives else "the shift does NOT survive rolling the label sequence: a slowly varying label would read this way by drift alone"))
        verdict["survives_circular_shift"] = bool(survives)
    if "sectors" in result:
        sc = result["sectors"]
        txt = f"class-mean kernel: even amplitude {sc['A_even']:.3f}, odd {sc['A_odd']:.3f}"
        if "balance" in sc:
            txt += f", sector balance {sc['balance']:.2f}"
        lines.append(txt)
    if result["delta"] < 0:
        lines.append("negative shift: accumulating the declared classes adds fewer dimensions than random draws (conditioning branch; low-rank between-class structure with isotropic within-class variability gives this sign)")
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
    for r in rows:
        t = r.get(text_field); lab = r.get(label_field)
        if t is None or lab is None:
            continue
        t = str(t).strip()
        if len(t) < min_chars:
            continue
        if len(t) > max_chars:
            cut = t[:max_chars]
            t = cut[:cut.rfind(" ")] if " " in cut else cut
        pool[str(lab)].append((t, None if strata_field is None else str(r.get(strata_field))))
    if classes is None:
        classes = [c for c, _ in sorted(pool.items(), key=lambda kv: -len(kv[1]))[:n_classes]]
    axis, strata = {}, {}
    for c in classes:
        items = pool.get(str(c), [])
        if len(items) < n_per_class:
            raise ValueError(f"class {c!r} has {len(items)} usable texts, fewer than n_per_class={n_per_class}")
        pick = rng.choice(len(items), n_per_class, replace=False)
        axis[str(c)] = [items[i][0] for i in pick]
        if strata_field is not None:
            strata[str(c)] = [items[i][1] for i in pick]
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

def _load_array(path):
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    return np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)


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


def last_token_arms(X, labels, last_ids, n_sub=20, n_perm=100, k_orders=20, seed=1500):
    """The last-token control of Appendix I for one layer: n_sub subsamples with m prompts per class ending on m
    distinct last tokens (m = the smallest class's count of distinct last tokens), and n_sub size-matched random
    subsamples with no token matching; the order-averaged shift and its permutation p for each. The lexical part of
    a lift is the drop from the random arm to the matched arm; if the random arm also loses the lift, the subsample
    is too small to tell. Returns None when m < 6."""
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
    out = {"m_per_class": int(m)}
    for arm, subs in arms.items():
        vals = [zoom(X[k], labels[k], n_perm=n_perm, k_orders=k_orders, k_null_orders=k_orders,
                     seed=int(rng.integers(1 << 30))) for k in subs]
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
    kw = {"torch_dtype": dtype}
    if revision:
        kw["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).to(device)
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision) if revision \
        else AutoTokenizer.from_pretrained(model_name)
    model.eval()
    if frames not in ("linear", "z", "both"):
        raise ValueError("frames is 'linear', 'z' or 'both'")
    frame_list = {"linear": [False], "z": [True], "both": [False, True]}[frames]
    if standardize:
        frame_list = [True]

    results = {"model": model_name, "revision": revision, "n_perm": n_perm,
               "k_orders": k_orders, "frames": "z" if standardize else frames, "axes": {}}
    for af in files:
        name, classes, prompts, labels, strata = _load_axis(af, use_strata)
        print(f"[{name}] {len(classes)} classes, {len(prompts)} prompts"
              f"{' + strata' if strata is not None else ''}", flush=True)
        states, last = [], []
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
                "null_mean_orderavg", "null_sd_orderavg",
                "strat_p_two", "strat_z", "strat_null_mean", "strat_null_sd",
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
                arms = last_token_arms(X, labels, last)
                row["last_token_arms"] = arms
                if arms is not None:
                    msg += (f" | matched {arms['matched']['delta_orderavg_median']:+.3f} vs random "
                            f"{arms['random']['delta_orderavg_median']:+.3f} (m={arms['m_per_class']})")
            layers.append(row)
            print(msg, flush=True)
        results["axes"][name] = {"classes": classes, "n_prompts": len(prompts),
                                 "stratified": strata is not None, "last_token": ltr, "layers": layers}
        if "frames_disagree" in layers[0]:
            nd = sum(r["frames_disagree"] for r in layers)
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
        L = a["layers"]
        x = np.arange(len(L)) / max(len(L) - 1, 1)
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
        L = a["layers"]; n = len(L)
        bh_possible = floor_p <= 0.05 / n
        pd_ = [l.get("p_two", 1.0) for l in L]; pa = [l.get("p_two_orderavg", 1.0) for l in L]
        kd, kd_bh = sum(p < 0.05 for p in pd_), _bh_count(pd_)
        ka, ka_bh = sum(p < 0.05 for p in pa), _bh_count(pa)
        if not bh_possible:
            # BH cannot certify any layer at this permutation count; fall back to raw counts
            kd_bh, ka_bh = kd, ka
        dl = np.array([l["delta"] for l in L]); da = np.array([l["delta_orderavg"] for l in L])
        emb, fin = dl[0], dl[-1]
        exc = dl - emb; peak_l = int(np.argmax(exc)); peak = float(exc[peak_l])
        cross = next((i for i in range(1, n) if np.sign(dl[i]) != np.sign(emb) and dl[i] != 0), None)
        lines = []
        # partition-level reading
        frac = ka_bh / n
        if frac >= 0.5:
            lines.append(f"label-linked at most depths: order-averaged statistic certified at {ka}/{n} layers "
                         f"({ka_bh}/{n} {'after BH' if bh_possible else 'raw'}), mean deltabar {da.mean():+.3f}.")
        elif ka_bh > 0:
            lines.append(f"weakly label-linked: order-averaged statistic certified at {ka}/{n} layers "
                         f"({ka_bh}/{n} {'after BH' if bh_possible else 'raw'}), mean deltabar {da.mean():+.3f}.")
        else:
            lines.append(f"no certified label linkage under the order-free statistic ({ka}/{n} raw, 0 after BH); "
                         f"treat this partition as unorganized at this design.")
        # declared-path shape
        shape = (f"declared path: embedding {emb:+.3f} -> final {fin:+.3f}, peak excess {peak:+.3f} at layer {peak_l}"
                 f"/{n-1}, certified {kd}/{n} ({kd_bh}/{n} {'after BH' if bh_possible else 'raw'})")
        if emb < 0 and fin > 0:
            shape += (f"; negative-to-positive crossover at layer {cross} — a PATH property (the order-averaged "
                      f"line is the partition-level claim; the crossover shape depends on the class order).")
        elif emb > 0 and fin < emb - 0.02:
            shape += "; positive at the embedding and declining — content-like: organization inherited from tokens, diluted with depth."
        elif abs(emb) < 0.02 and abs(fin) < 0.02 and kd_bh == 0:
            shape += "; flat near zero along the declared path."
        else:
            shape += "."
        lines.append(shape)
        # stratified null
        if "strat_p_two" in L[0]:
            ps_ = [l["strat_p_two"] for l in L]; ks, ks_bh = sum(p < 0.05 for p in ps_), _bh_count(ps_)
            if not bh_possible:
                ks_bh = ks
            tag = "after BH" if bh_possible else "raw, BH not resolvable"
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
        if bh_possible:
            lines.append(f"caveat: per-layer counts are descriptive (~{0.05*n:.1f} false positives expected at p<0.05); "
                         f"the BH counts and the order-averaged statistic carry the inference.")
        else:
            lines.append(f"RESOLUTION WARNING: n_perm={n_perm} gives a p floor of {floor_p:.4f}, above the BH threshold "
                         f"{0.05/n:.4f} for {n} layers, so no layer can be BH-certified at this run; counts above are raw. "
                         f"Rerun with --n-perm 500 (the paper's setting) or more before drawing conclusions.")
        report["axes"][name] = {"n_perm": n_perm, "bh_resolvable": bh_possible,
                                "certified_declared": [kd, kd_bh], "certified_orderavg": [ka, ka_bh],
                                "embedding": float(emb), "final": float(fin), "peak_excess": peak,
                                "peak_layer": peak_l, "crossover_layer": cross, "reading": lines}
        if verbose:
            hdr = f"== {report['model'] or ''}{' @ ' + report['revision'] if report.get('revision') else ''} :: {name} ({n} layers) =="
            print(hdr); [print("  - " + s) for s in lines]
    if out_json:
        _json.dump(report, open(out_json, "w"), indent=1)
    return report


# ---- The four-term split of a rung's deficit (Section 6 and Appendix K) ----
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
    Y = np.asarray(Y, np.float32); n = Y.shape[0]
    xbar = Y.mean(0)
    classes = np.unique(lab)
    Mm = np.stack([np.sqrt((lab == c).sum() / n) * (Y[lab == c].mean(0) - xbar) for c in classes]).astype(np.float32)
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


def sign_law(X, labels):
    """The sign of the class-mean part, read from the full set (Section 6, Appendix K). With beta = Tr B / Tr W,
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
    rows = Z[np.random.default_rng(seed).permutation(len(Z))]
    parts, labs, i = [], [], 0
    for c in classes:
        k = int((labels == c).sum())
        Zc = rows[i:i + k].copy(); i += k
        Zc -= Zc.mean(0); s_ = Zc.std(0); s_[s_ <= 0] = 1.0; Zc /= s_
        parts.append(mu[c] + Zc * tgt[c]); labs.append(np.full(k, c))
    return np.concatenate(parts), np.concatenate(labs)


def decompose(X, labels, order=None, bin_counts=None, n_null=10, seed=42, min_size=10, n_shuffle=0, n_bias=0):
    """The exact split of every rung's deficit and of the shift for the ladder that accumulates the classes in
    `order` (default: sorted labels) with `bin_counts` classes per rung (default: the paper's ladder). Returns the
    per-rung terms, the four slopes of the shift (and the scale-free regrouping), the private slopes, exactness
    checks, the sign-law readout of the full set, with n_shuffle > 0 the label-permutation baseline and the excess
    over it, and with n_bias > 0 the same split on bias_null(mode='shape') surrogates (seeds 0..n_bias-1) and the
    data net of it: the within part counts as class-dependent fluctuation structure only beyond that null."""
    X = np.asarray(X, np.float32); labels = np.asarray(labels)
    if order is None:
        order = [int(c) for c in np.unique(labels)]
    if bin_counts is None:
        bin_counts = tuple(c for c in BIN_COUNTS_DEFAULT if c <= len(order))
    out = _decompose_once(X, labels, order, bin_counts, n_null, seed, min_size)
    if out is None:
        return out
    out["sign_law"] = sign_law(X, labels)
    if n_bias > 0:
        bias = []
        for s_ in range(n_bias):
            Y, ly = bias_null(X, labels, "shape", s_)
            r = _decompose_once(Y.astype(np.float32), ly, order, bin_counts, n_null, seed, min_size)
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
        r = _decompose_once(X, labels[srng.permutation(len(labels))], order, bin_counts, n_null, seed, min_size)
        if r is not None:
            shuf.append(r)
    if not shuf:
        return out

    def _sl(r, k):
        return r["delta_split"][k] if k in r["delta_split"] else (r["delta_scale"] if k == "S" else r["delta_split_free"][k])
    out["n_shuffle"] = len(shuf)
    out["shuffle"] = {"delta": float(np.mean([r["delta"] for r in shuf])),
                      "delta_split": {k: float(np.mean([_sl(r, k) for r in shuf])) for k in keys},
                      "private_slope": float(np.mean([r["private_slope"] for r in shuf])),
                      "private_slope_dim": float(np.mean([r["private_slope_dim"] for r in shuf])),
                      "rungs": [{k: float(np.mean([r["rungs"][i][k] for r in shuf])) for k in ("deficit",) + keys} for i in range(len(out["rungs"]))]}
    if "at_four" in out and all("at_four" in r for r in shuf):
        out["shuffle"]["at_four"] = {k: float(np.mean([r["at_four"][k] for r in shuf])) for k in ("deficit",) + keys}
    out["excess"] = {"delta": out["delta"] - out["shuffle"]["delta"],
                     "delta_split": {k: _sl(out, k) - out["shuffle"]["delta_split"][k] for k in keys},
                     "private_slope": out["private_slope"] - out["shuffle"]["private_slope"],
                     "private_slope_dim": out["private_slope_dim"] - out["shuffle"]["private_slope_dim"]}
    if "at_four" in out["shuffle"]:
        out["excess"]["at_four"] = {k: out["at_four"][k] - out["shuffle"]["at_four"][k] for k in ("deficit",) + keys}
    return out


def _decompose_once(X, labels, order, bin_counts, n_null, seed, min_size):
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
    flo = [dict(lt=0.0, lw2=0.0, lb=0.0, lbx=0.0, ld=0.0, tb=0.0, lw1=0.0, pr=0.0, traces=np.zeros(5)) for _ in sizes]
    for _ in range(n_null):
        for k, s in enumerate(sizes):
            idx = rng.choice(len(X), s, replace=False)
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
           "max_pr_identity_rel": float(max(r["pr_identity_rel"] for r in rows))}
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
    ex = r.get("excess", {}).get("delta_split")
    print(f"decompose: delta {r['delta']:+.3f} = A {r['delta_split']['A']:+.3f} + P {r['delta_split']['P']:+.3f} + Cb {r['delta_split']['Cb']:+.3f} + Cx {r['delta_split']['Cx']:+.3f}"
          f"  (regrouped: D {r['delta_split_free']['D']:+.3f} + Tb {r['delta_split_free']['Tb']:+.3f}; scale S {r['delta_scale']:+.3f} cancels between A and P)")
    if ex is not None:
        print(f"  net of {r['n_shuffle']} label permutations: D {ex['D']:+.3f} Tb {ex['Tb']:+.3f} Cb {ex['Cb']:+.3f} Cx {ex['Cx']:+.3f} S {ex['S']:+.3f} -> largest {largest_term(ex, FREE)}; "
              f"private slope (D vs log k/K) {r['excess']['private_slope_dim']:+.2f}")
    else:
        print(f"  private slope (D vs log k/K) {r['private_slope_dim']:+.2f}; run with --n-shuffle for the sampling baseline")
    print(f"  exactness: sum {r['max_sum_check']:.1e}, regroup {r['max_regroup_check']:.1e}, PR identity {r['max_pr_identity_rel']:.1e}")
    sl = r.get("sign_law")
    if sl:
        agree = sl["class_mean_negative"] == sl["law_says_negative"]
        print(f"  sign law (full set): Tr B/Tr W {sl['trB_trW']:.3f}, PR(W)/PR(B) = {sl['PR_W']:.1f}/{sl['PR_B']:.1f} = {sl['ratio']:.1f}, "
              f"alignment a {sl['alignment']:.2f}; threshold 1 + 2(1 - a)/beta = {sl['threshold_full']:.1f} "
              f"(x-free {sl['threshold_x_free']:.1f}); class-mean term {sl['c_full']:+.3f} "
              + ("(sign as the law says)" if agree else "(MISMATCH with the law: report it)"))
    if "net_of_bias" in r:
        b = r["bias_null"]["delta_split_free"]; nb = r["net_of_bias"]["delta_split_free"]
        print(f"  bias null ({r['bias_null']['n']} surrogates, one covariance shape for every class): D {b['D']:+.3f} Tb {b['Tb']:+.3f} "
              f"Cb {b['Cb']:+.3f} Cx {b['Cx']:+.3f}; net of it: D {nb['D']:+.3f}, delta {r['net_of_bias']['delta']:+.3f}")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        prog="rung",
        description="Axis-resolved dimensionality scaling: "
                    "theta_obs = theta_floor + delta, with exact permutation "
                    "nulls and a nuisance-preserving second null.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="run on arrays (npy/csv/txt)")
    d.add_argument("X", help="samples-by-features array")
    d.add_argument("labels", help="integer class labels, one per sample")
    d.add_argument("--strata", default=None, help="nuisance stratum ids for the second null")
    d.add_argument("--n-perm", type=int, default=500)
    d.add_argument("--k-orders", type=int, default=50)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default=None, help="write the result JSON here")
    d.add_argument("--plot", default=None, help="write a ladder figure (PNG) here")
    d.add_argument("--standardize", action="store_true", help="z-score every feature first (the repair when one dimension dominates the spectrum)")
    d.add_argument("--split-by", default=None, help="per-feature score file (npy/csv/txt, one value per feature): also run the top, bottom and a random third of the features at matched size, each with its own floor (Section 4)")
    d.add_argument("--order", default=None, help="the declared accumulation order: sequential (default), antipodal (one member of each antipodal pair first), or a comma-separated list of class labels")
    d.add_argument("--sectors", action="store_true", help="fit the harmonics of the class-mean kernel for a cyclic axis (labels in cyclic order, class k at angle 2 pi k/K): the sector balance, the even and odd amplitudes, and the coherence of adjacent and antipodal classes (Section 4)")
    d.add_argument("--null-shift", action="store_true", help="labels are in time order: add the circular-shift null, which rolls the label sequence and keeps its autocorrelation")
    d.add_argument("--antipode", default=None, help="JSON {class: antipodal class}; checks the declared order for the stall test and reports late_fraction as it")
    d.add_argument("--decompose", action="store_true", help="the exact four-term split of every rung's deficit and of the shift (trace, pooling, between-class, coupling) and its scale-free regrouping A + P = D + Tb (Section 6, Appendix K); with --n-shuffle > 0, net of label permutations")
    d.add_argument("--n-shuffle", type=int, default=5, help="label permutations for the sampling baseline of each decomposed term (0 disables)")
    d.add_argument("--n-bias", type=int, default=0, help="with --decompose: surrogates of the finite-sample bias null (class means kept, one covariance shape for every class); the within part is read net of it")
    d.add_argument("--time-blocks", default=None, help="rows are in acquisition order: an integer E cuts them into contiguous blocks of E rows, or a file gives one block id per row; adds the null that permutes labels only within blocks")

    m = sub.add_parser("llm", help="per-layer battery on a Hugging Face model")
    m.add_argument("--model", required=True)
    m.add_argument("--axis", required=True, nargs="+",
                   help="axis JSON file(s) {class: [prompts]} and/or directories of them")
    m.add_argument("--revision", default=None, help="HF revision, e.g. a training checkpoint")
    m.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    m.add_argument("--n-perm", type=int, default=500)
    m.add_argument("--k-orders", type=int, default=50)
    m.add_argument("--max-len", type=int, default=128)
    m.add_argument("--no-strata", action="store_true", help="ignore *.strata.json sidecars")
    m.add_argument("--out", default=None, help="write JSON here")
    m.add_argument("--paper-seeds", action="store_true",
                   help="use the paper's seeding convention (reproduces the published per-layer cells)")
    m.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default=None,
                   help="model dtype; default float16 on mps/cuda, float32 on cpu; bf16 repairs float16 overflow in early checkpoints")
    m.add_argument("--standardize", action="store_true", help="z-score every unit per layer before the Gram (z frame only)")
    m.add_argument("--frames", choices=["linear", "z", "both"], default="both",
                   help="feature frame(s): the linear Gram (the paper's), per-unit z-scored, or both with a sign-disagreement flag (default)")
    m.add_argument("--match-last-token", action="store_true",
                   help="per layer, the last-token-matched and size-matched random subsamples (Appendix I's last-token control)")

    s = sub.add_parser("summarize", help="plain-language reading of a `rung llm` or `rung data` JSON (the paper's rules)")
    s.add_argument("battery_json")
    s.add_argument("--out", default=None, help="also write the report as JSON")

    x = sub.add_parser("axis", help="build an axis JSON (+ strata sidecar) from a Hugging Face dataset")
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

    p = sub.add_parser("plot", help="depth-profile figure(s) from a `rung llm` JSON, or the ladder figure from a `rung data` JSON")
    p.add_argument("battery_json")
    p.add_argument("--out", default="battery.png")

    a = ap.parse_args()
    if a.cmd == "summarize":
        import json as _json
        d = _json.load(open(a.battery_json))
        if "layers" in d or "axes" in d:
            summarize(a.battery_json, a.out)
        else:
            rep = summarize_data(d)
            if a.out:
                _json.dump(rep, open(a.out, "w"), indent=1)
    elif a.cmd == "plot":
        import json as _json
        d = _json.load(open(a.battery_json))
        if "layers" in d or "axes" in d:
            _plot_battery(a.battery_json, a.out)
        else:
            _plot_data(d, a.out); print("wrote", a.out)
    elif a.cmd == "axis":
        axis_from_dataset(a.dataset, a.text_field, a.label_field, a.out, config=a.config, split=a.split,
                          n_classes=a.n_classes, n_per_class=a.n_per_class, strata_field=a.strata_field,
                          classes=a.classes, min_chars=a.min_chars, max_chars=a.max_chars, seed=a.seed)
    elif a.cmd == "data":
        X = _load_array(a.X)
        labels = _load_array(a.labels).astype(int)
        strata = _load_array(a.strata).astype(int) if a.strata else None
        sp = spectrum(X)
        print(f"spectrum: d_eff_full={sp['d_eff_full']:.1f} top1_eig_frac={sp['top1_eig_frac']:.2f} top1_dim_var_frac={sp['top1_dim_var_frac']:.2f}"
              + ("  <- one dimension dominates; consider --standardize" if sp["top1_eig_frac"] > 0.5 else ""))
        antipode = None
        if a.antipode:
            import json as _json
            antipode = {int(k): int(v) for k, v in _json.load(open(a.antipode)).items()}
        order = a.order
        if order not in (None, "sequential", "antipodal"):
            order = [int(v) for v in order.split(",")]
        tb = None
        if a.time_blocks:
            tb = block_ids(len(labels), int(a.time_blocks)) if a.time_blocks.isdigit() else _load_array(a.time_blocks).astype(int)
        r = zoom(X, labels, n_perm=a.n_perm, k_orders=a.k_orders, strata=strata, seed=a.seed, standardize=a.standardize,
                 antipode=antipode, order=order, shift_null=a.null_shift, time_blocks=tb)
        if not a.standardize:
            rz = zoom(X, labels, n_perm=0, k_orders=a.k_orders, seed=a.seed, standardize=True, order=order)
            r["delta_z_frame"], r["delta_orderavg_z_frame"] = rz["delta"], rz["delta_orderavg"]
            print(f"frame check: per-feature z-scored delta {rz['delta']:+.3f} (order-averaged {rz['delta_orderavg']:+.3f})"
                  + ("  <- OPPOSITE sign to the linear frame: a few high-variance features set the linear reading"
                     if np.sign(rz["delta"]) != np.sign(r["delta"]) else ""))
        for k, v in r.items():
            if not isinstance(v, list):
                print(f"{k}: {v}")
        if a.split_by:
            score = _load_array(a.split_by).astype(float).ravel()
            sp_r = split_by(X, labels, score, seed=a.seed, n_perm=a.n_perm, k_orders=a.k_orders, strata=strata, standardize=a.standardize, antipode=antipode, order=order)
            r["split_by"] = sp_r
            print(f"split by score, {sp_r['n_features_per_subset']} of {sp_r['n_features_total']} features per subset (matched size, own floors):")
            for name, rr in sp_r["subsets"].items():
                print(f"  {name:7s}: delta {rr['delta']:+.3f}" + (f", p = {rr['p_two']:.3f}" if "p_two" in rr else "")
                      + (f", late fraction {rr['late_fraction']:.2f}" if "late_fraction" in rr else "") + f", median score {rr['median_score']:.3g}")
        if a.sectors:
            # the kernel's cycle is the labels' own order (class k sits at angle 2 pi k / K); the
            # accumulation order of --order is a different thing and does not enter here
            sc = sectors(X, labels)
            _print_sectors(sc); r["sectors"] = sc
        if a.decompose:
            Xd = (X - X.mean(0)) / (X.std(0) + 1e-12) if a.standardize else X
            rd = decompose(Xd, labels, order=None if order in (None, "sequential") else [int(np.unique(labels)[i]) for i in declared_order(np.unique(labels), order)],
                           n_null=20, seed=a.seed, n_shuffle=a.n_shuffle, n_bias=a.n_bias)
            if rd is None:
                print("decompose: the ladder has fewer than three rungs of ten samples; nothing to split")
            else:
                _print_decomposition(rd); r["decompose"] = rd
        if a.out:
            import json as _json
            _json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in r.items()}, open(a.out, "w"), indent=1)
            print("wrote", a.out)
        if a.plot:
            _plot_data(r, a.plot); print("wrote", a.plot)
    else:
        llm_battery(a.model, a.axis, a.device, a.n_perm, a.k_orders, a.max_len,
                    a.out, paper_seeds=a.paper_seeds, revision=a.revision,
                    use_strata=not a.no_strata, standardize=a.standardize, dtype=a.dtype,
                    frames=a.frames, match_last_token=a.match_last_token)


if __name__ == "__main__":
    main()
