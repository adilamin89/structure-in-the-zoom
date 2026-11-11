"""Comparison methods for Section 10 and Appendix L of the extended version: rung's readings beside manifold capacity, CKA / RSA,
a linear probe, intrinsic dimension and neural-collapse order parameters, plus the effective-basis descriptors. Every function
takes X (n samples x d features) and integer labels.

rung (this repository's rung.py; the extended version's numbers were computed with rung 1.13.0, which 1.14 replays):
  rung_reading       zoom (order-averaged shift, label-permutation null), decompose (exact split, declared order and an
                     order average), classmean_share (between arm, wild null), sign_law (alignment a, PR(W), PR(B))
Manifold capacity (Chung, Lee & Sompolinsky, PRX 2018; Cohen et al., Nat Commun 2020):
  capacity_mft       a port of mftma/manifold_analysis_correlation.py (github.com/schung039/neural_manifolds_replicaMFT,
                     commit d56eda1): each manifold centred on its own mean, normalized by its centre norm, reduced to its
                     span, the centre coordinate appended, then each_manifold_analysis_D1 (PRX eqs. 16-17, 28, 29). The
                     constrained minimization of ||v - t||^2 subject to v.s <= -kappa is solved at kappa = 0 by Moreau's
                     decomposition (t - v = the projection of t onto the cone of the points, a non-negative least squares),
                     exact for kappa = 0 and checked against cvxopt's QP (the authors' solver) in g1_selftest.py. The
                     centre-correlation step (fun_FA, a Stiefel-manifold optimization) is not ported: the centres are
                     analysed as they are (PRX 2018 setting) and their correlation is reported beside the capacity. With
                     the README's random manifolds the port gives 0.0400 / 1.473 / 36.05 against the README's 0.04 / 1.47 /
                     36.17 (g1_selftest.json). With 8 classes fun_FA's rank search (up to 11 + the 95% rank) exceeds the
                     P - 1 = 7 centre dimensions, so the step is not defined for these batteries.
  capacity_sim       simulation capacity (Cohen et al. 2020, Methods): the data centred on the global mean, projected onto
                     random N-dimensional subspaces; for every one of the 2^(P-1) - 1 dichotomies of the P manifolds, linear
                     separability of all points through the origin (an LP); N_c where half the dichotomies separate (linear
                     interpolation in N), alpha_sim = P / N_c.
Others: cka_label (linear CKA with the one-hot label; identity CKA = beta sqrt(PR / (K - 1)) for equal classes),
  rsa_label (Spearman between pairwise distances and the same/different-class model RDM), probe (cross-validated logistic
  regression with a permuted-label control, Hewitt & Liang 2019), twonn (Facco et al. 2017, as Ansuini et al. 2019),
  collapse (Papyan et al. 2020: NC1 = Tr(S_W S_B^+) / K; NC2 equinorm and equiangularity), pr_global.
Effective basis: harmonics (class-mean structure factor on a cyclic label with a class-position permutation null),
  ring_h1 (persistent H1 of the class means against Gaussian points with the means' covariance), crossed (class x carrier
  two-way decomposition of a crossed battery).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, nnls
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rung  # noqa: E402

# rung 1.14 averages over all class orders exactly when there are at most ten; the random-order average replays 1.13.0
_RANDOM_ORDERS = {"order_average": "random"} if "order_average" in rung.zoom.__code__.co_varnames else {}


def _lab(labels):
    classes, lab = np.unique(np.asarray(labels), return_inverse=True)
    return lab, len(classes)


def zframe(X):
    """Per-feature z-score (the correlation frame); features whose float64 SD is at most 1e-6 x the median SD are dropped."""
    X = np.asarray(X, np.float64)
    sd = X.std(0)
    keep = sd > 1e-6 * np.median(sd)
    return (X[:, keep] - X[:, keep].mean(0)) / sd[keep], int((~keep).sum())


def pr_global(X):
    return rung.pr_centered(X) if X.shape[0] <= X.shape[1] else _pr_cov(X)


def _pr_cov(X):
    Xc = np.asarray(X, np.float64) - X.mean(0)
    C = Xc.T @ Xc
    return float(np.trace(C) ** 2 / np.sum(C * C))


# ---------------------------------------------------------------------------------------------------------------- rung
def rung_reading(X, labels, n_perm=200, k_orders=50, k_null_orders=20, n_split_orders=20, seed=0, order=None, workers=4,
                 n_share_perm=200, strata=None):
    """The shift (declared and order-averaged) with its label-permutation null, the exact split for the declared order and
    averaged over n_split_orders random orders, the between arm (class-mean share, wild null), the sign-law quantities."""
    X = np.asarray(X, np.float64)
    lab, K = _lab(labels)
    z = rung.zoom(X, lab, n_perm=n_perm, k_orders=k_orders, k_null_orders=k_null_orders, seed=seed, order=order,
                  workers=workers, strata=strata, **_RANDOM_ORDERS)
    out = {k: z.get(k) for k in ("delta", "theta_obs", "theta_floor", "theta_floor_se", "p_two", "z", "delta_orderavg",
                                 "delta_orderavg_sd", "p_two_orderavg", "z_orderavg", "strat_p_two", "strat_z",
                                 "delta_pairavg", "pairavg_p_two")}
    dec = rung.decompose(X, lab, order=order, n_null=10, seed=42, workers=workers, n_bias=0)
    if dec is None or "undefined" in dec:
        out["split_declared"] = {"undefined": str(dec.get("undefined") if dec else "none")}
    else:
        f = dec["delta_split_free"]
        out["split_declared"] = {"delta": dec["delta"], "D": f["D"], "class_mean": f["Tb"] + f["Cb"] + f["Cx"]}
    rng = np.random.default_rng(seed + 9001)
    Ds, CMs, ds = [], [], []
    for _ in range(n_split_orders):
        o = [int(c) for c in rng.permutation(K)]
        d = rung.decompose(X, lab, order=o, n_null=10, seed=42, workers=workers, n_bias=0)
        if d is None or "undefined" in d:
            continue
        f = d["delta_split_free"]
        Ds.append(f["D"]); CMs.append(f["Tb"] + f["Cb"] + f["Cx"]); ds.append(d["delta"])
    out["split_orderavg"] = {"n_orders": len(Ds), "delta": float(np.mean(ds)) if ds else float("nan"),
                             "D": float(np.mean(Ds)) if Ds else float("nan"),
                             "class_mean": float(np.mean(CMs)) if CMs else float("nan"),
                             "D_sd": float(np.std(Ds, ddof=1)) if len(Ds) > 1 else float("nan")}
    cs = rung.classmean_share(X, lab, n_perm=n_share_perm, seed=seed, null="wild", strata=strata)
    out["share"] = {k: cs.get(k) for k in ("share", "null_mean", "null_sd", "z", "p_one", "excess")}
    sl = rung.sign_law(X, lab)
    out["sign_law"] = {k: sl.get(k) for k in ("trB_trW", "PR_W", "PR_B", "ratio", "alignment", "c_full")}
    return out


# ---------------------------------------------------------------------------------------------------------------- capacity
def _maxproj(t_vec, sD1):
    D = sD1.shape[0] - 1
    idx = np.argmax(t_vec[:D].T @ sD1[:D], axis=1)
    s0 = np.vstack([sD1[:D, idx], np.ones((1, t_vec.shape[1]))])
    return s0, np.sum(t_vec * s0, axis=0)


def _project_cone(t, S):
    """kappa = 0: v* = argmin ||v - t||^2 s.t. S'v <= 0 is t - S alpha*, alpha* = argmin_{alpha >= 0} ||t - S alpha||
    (Moreau); returns v*, alpha*."""
    alpha, _ = nnls(S, t, maxiter=50 * S.shape[1])
    return t - S @ alpha, alpha


def each_manifold_D1(sD1, n_t, rng, eps=1e-8, t_vec=None, solver="nnls"):
    """Port of each_manifold_analysis_D1 at kappa = 0 (PRX 2018 eqs. 16-17 capacity, 28 radius, 29 dimension)."""
    D1, m = sD1.shape
    D = D1 - 1
    if t_vec is None:
        t_vec = rng.standard_normal((D1, n_t))
    ss, gg = _maxproj(t_vec, sD1)
    s_all = np.empty((D1, n_t))
    for i in range(n_t):
        t = t_vec[:, i]
        if gg[i] < 0:
            s_all[:, i] = ss[:, i]
            continue
        if solver == "nnls":
            v_f, alpha = _project_cone(t, sD1)
        else:
            v_f, alpha = _qp_cvxopt(t, sD1)
        if np.linalg.norm(v_f - t) < eps:
            s_all[:, i] = ss[:, i]
        else:
            s_all[:, i] = (t - v_f) / np.sum(alpha)
    max_ts = np.maximum(np.sum(t_vec * s_all, axis=0), 0)
    s_sum = np.sum(s_all ** 2, axis=0)
    lamb = np.where(s_sum > 0, max_ts / np.where(s_sum > 0, s_sum, 1), 0)
    a_M = 1 / np.mean(lamb ** 2 * s_sum)
    ds0 = s_all - s_all.mean(axis=1, keepdims=True)
    ds = ds0[:-1] / s_all[-1]
    R_M = np.sqrt(np.mean(np.sum(ds ** 2, axis=0)))
    t_hat = t_vec[:D] / np.sqrt(np.sum(t_vec[:D] ** 2, axis=0, keepdims=True))
    s_hat = s_all[:D] / np.sqrt(np.sum(s_all[:D] ** 2, axis=0, keepdims=True) + 1e-12)
    D_M = D * np.mean(np.sum(t_hat * s_hat, axis=0)) ** 2
    return float(a_M), float(R_M), float(D_M)


def _qp_cvxopt(t, sD1):
    """The authors' solver (cvxopt QP: min 0.5 v'v - t'v s.t. sD1' v <= 0), for the self-test only."""
    from cvxopt import matrix, solvers
    solvers.options.update({"show_progress": False, "abstol": 1e-12, "reltol": 1e-12, "feastol": 1e-12, "maxiters": 1000})
    D1, m = sD1.shape
    out = solvers.qp(matrix(np.eye(D1)), matrix(-t.astype(np.double)), matrix(sD1.T.astype(np.double)),
                     matrix(np.zeros(m)))
    return np.array(out["x"])[:, 0], np.array(out["z"])[:, 0]


def capacity_mft(X, labels, n_t=200, seed=0, solver="nnls"):
    """Per-manifold capacity, radius and dimension (mean-field, kappa = 0) and the centre correlation before any removal.
    Averages as the authors': capacity 1 / mean(1 / a_M), radius and dimension arithmetic means."""
    X = np.asarray(X, np.float64)
    lab, P = _lab(labels)
    rng = np.random.default_rng(seed)
    Xg = X - X.mean(0)
    cents = np.stack([Xg[lab == c].mean(0) for c in range(P)])        # P x d
    cm = cents - cents.mean(0)
    cn = np.linalg.norm(cm, axis=1, keepdims=True)
    C0 = (cm @ cm.T) / (cn @ cn.T)
    rho_c = float((np.abs(C0).sum() - P) / (P * (P - 1)))
    aa, rr, dd = [], [], []
    for c in range(P):
        Xr = Xg[lab == c].T                                                # d x m
        mu = Xr.mean(axis=1, keepdims=True)
        S_r = (Xr - mu) / np.linalg.norm(mu)
        Dd, m = S_r.shape
        if Dd > m:
            Q, Rq = np.linalg.qr(S_r)
            Q = Q * np.where(np.diag(Rq) < 0, -1.0, 1.0)      # positive diagonal of R: the reduced coordinates no longer
            S_r = Q.T @ S_r                                    # depend on Householder's sign choices (rotation-exact)
        sD1 = np.vstack([S_r, np.ones((1, S_r.shape[1]))])
        a, r, d = each_manifold_D1(sD1, n_t, rng, solver=solver)
        aa.append(a); rr.append(r); dd.append(d)
    aa, rr, dd = map(np.array, (aa, rr, dd))
    return {"capacity": float(1 / np.mean(1 / aa)), "radius": float(rr.mean()), "dimension": float(dd.mean()),
            "center_corr": rho_c, "per_manifold": {"a": aa.tolist(), "R": rr.tolist(), "D": dd.tolist()}}


def _separable(Z, y):
    """Is there w with y_i w.z_i >= 1 for every i (homogeneous, after global centring)? An LP feasibility problem."""
    n, N = Z.shape
    A = -(y[:, None] * Z)
    res = linprog(np.zeros(N), A_ub=A, b_ub=-np.ones(n), bounds=[(None, None)] * N, method="highs")
    return res.status == 0


def _sep_batch(args):
    Z, ys = args
    return [_separable(Z, y) for y in ys]


def capacity_sim(X, labels, n_proj=3, seed=0, max_dich=None, n_max=None, n_jobs=None):
    """Simulation capacity: fraction f(N) of dichotomies (all 2^(P-1) - 1, or max_dich random ones) linearly separable after a
    random projection to N dimensions, averaged over n_proj projections. N is searched upward (1, 2, 4, 8, ...)
    until f >= 0.5, the bracket refined by integer bisection; N_c by linear interpolation at f = 0.5, alpha_sim = P / N_c.
    n_max caps N (default min(n, d)); no crossing below it gives alpha_sim < P / n_max, reported as an upper bound."""
    X = np.asarray(X, np.float64)
    lab, P = _lab(labels)
    rng = np.random.default_rng(seed)
    Xg = X - X.mean(0)
    Xg = Xg / np.sqrt(np.mean(np.sum(Xg ** 2, axis=1)))
    dich = [np.array([1] + list(b)) for b in itertools.product([1, -1], repeat=P - 1)]
    dich = [d for d in dich if not np.all(d == 1)]
    if max_dich is not None and len(dich) > max_dich:
        dich = [dich[i] for i in rng.choice(len(dich), max_dich, replace=False)]
    n_max = n_max or min(X.shape[0], X.shape[1])
    cache = {}
    import os
    n_jobs = n_jobs or int(os.environ.get("SIM_JOBS", "1"))      # dichotomies split across processes: identical LPs, identical result
    pool = None
    if n_jobs > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(n_jobs, mp_context=mp.get_context("spawn"))

    def f(N):
        if N not in cache:
            vals = []
            for _ in range(n_proj):
                G = rng.standard_normal((X.shape[1], N)) / np.sqrt(N)
                Z = Xg @ G
                ys = [d[lab].astype(float) for d in dich]
                if pool is None:
                    vals.append(np.mean([_separable(Z, y) for y in ys]))
                else:
                    chunks = [ys[i::n_jobs] for i in range(n_jobs)]
                    vals.append(np.mean([r for part in pool.map(_sep_batch, [(Z, c) for c in chunks]) for r in part]))
            cache[N] = float(np.mean(vals))
        return cache[N]
    grid = [1]                                          # doubling, then integer bisection: the same 1-wide bracket at ~half the evaluations
    while grid[-1] < n_max:
        grid.append(min(n_max, 2 * grid[-1]))
    lo, hi = None, None
    for N in grid:
        if f(N) >= 0.5:
            hi = N
            break
        lo = N
    if hi is None:
        if pool is not None:
            pool.shutdown()
        return {"alpha_sim": float("nan"), "alpha_sim_upper": float(P / n_max), "N_c": float("nan"),
                "curve": {int(k): v for k, v in sorted(cache.items())}, "n_dichotomies": len(dich), "n_proj": n_proj,
                "search": "doubling"}
    if lo is None:
        Nc = float(hi)
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if f(mid) >= 0.5:
                hi = mid
            else:
                lo = mid
        Nc = lo + (0.5 - f(lo)) * (hi - lo) / (f(hi) - f(lo))
    if pool is not None:
        pool.shutdown()
    return {"alpha_sim": float(P / Nc), "N_c": float(Nc), "curve": {int(k): v for k, v in sorted(cache.items())},
            "n_dichotomies": len(dich), "n_proj": n_proj, "search": "doubling"}


# ---------------------------------------------------------------------------------------------------------------- CKA / RSA
def cka_label(X, labels):
    X = np.asarray(X, np.float64)
    lab, K = _lab(labels)
    Xc = X - X.mean(0)
    Y = np.eye(K)[lab]
    Yc = Y - Y.mean(0)
    hsic = np.sum((Yc.T @ Xc) ** 2)
    nx = np.sqrt(np.sum((Xc @ Xc.T) ** 2)) if Xc.shape[0] <= Xc.shape[1] else np.sqrt(np.sum((Xc.T @ Xc) ** 2))
    ny = np.sqrt(np.sum((Yc.T @ Yc) ** 2))
    return float(hsic / (nx * ny))


def classmean_share_plain(X, labels):
    X = np.asarray(X, np.float64)
    lab, K = _lab(labels)
    Xc = X - X.mean(0)
    tb = sum((lab == c).sum() * np.sum(Xc[lab == c].mean(0) ** 2) for c in range(K))
    return float(tb / np.sum(Xc ** 2))


def _sqdist(X):
    Xc = np.asarray(X, np.float64) - X.mean(0)
    g = np.sum(Xc ** 2, axis=1)
    D2 = g[:, None] + g[None, :] - 2 * Xc @ Xc.T
    return np.clip(D2, 0, None)


def rsa_label(X, labels, n_perm=200, seed=0, model="category", D2=None):
    """Spearman correlation between pairwise Euclidean distances and a model RDM: 'category' (0 same class, 1 different)
    or 'circular' (the classes' circular distance on the cycle). Null: label permutations (Mantel)."""
    lab, K = _lab(labels)
    if D2 is None:
        D2 = _sqdist(X)
    iu = np.triu_indices(len(lab), 1)
    d = np.sqrt(D2[iu])

    def mrdm(l):
        a, b = l[iu[0]], l[iu[1]]
        if model == "category":
            return (a != b).astype(float)
        k = np.abs(a - b)
        return np.minimum(k, K - k).astype(float)
    from scipy.stats import rankdata
    rd = rankdata(d)
    def rho(l):
        return float(np.corrcoef(rd, rankdata(mrdm(l)))[0, 1])
    obs = rho(lab)
    rng = np.random.default_rng(seed)
    null = np.array([rho(rng.permutation(lab)) for _ in range(n_perm)]) if n_perm else np.array([])
    out = {"rsa": obs}
    if n_perm:
        out.update(null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                   p_one=float((1 + np.sum(null >= obs)) / (n_perm + 1)), z=float((obs - null.mean()) / null.std(ddof=1)))
    return out


# ---------------------------------------------------------------------------------------------------------------- probe
def probe(X, labels, groups=None, n_splits=5, seed=0, n_control=10, C=1.0, max_iter=20000, span=True, tol=1e-8):
    """Held-out accuracy of a standardized L2 logistic regression; folds stratified, or grouped when groups is given (BLiMP
    pairs, cyclic carriers held out). Control: the same procedure on permuted labels (n_control draws).
    span=True fits each fold in the coordinates of its standardized training rows (their right singular vectors): the L2
    solution lies in that span and the penalty is unchanged by an orthonormal basis, so the decision function equals the
    full-feature fit's up to the optimizer's tolerance, at a cost set by n_train, not d. At sklearn's default tol (1e-4) lbfgs
    stops after 15-30 iterations on these layers, short of the optimum, and the stopping point alone moves held-out accuracy by
    1-2 prompts per fold; tol 1e-8 reaches the penalized optimum (g1_selftest.py T6: its objective is the lowest in every fold)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    X = np.asarray(X, np.float64)
    lab, K = _lab(labels)

    def acc(y, s):
        if groups is None:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=s)
            splits = cv.split(X, y)
        else:
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s)
            splits = cv.split(X, y, groups)
        hits = 0
        for tr, te in splits:
            if span and X.shape[1] > len(tr):
                sc = StandardScaler().fit(X[tr])
                Ztr, Zte = sc.transform(X[tr]), sc.transform(X[te])
                _, sv, Vt = np.linalg.svd(Ztr, full_matrices=False)
                Vt = Vt[sv > sv.max() * 1e-10]
                clf = LogisticRegression(C=C, max_iter=max_iter, tol=tol)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    clf.fit(Ztr @ Vt.T, y[tr])
                hits += int(np.sum(clf.predict(Zte @ Vt.T) == y[te]))
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(X[tr], y[tr])
            hits += int(np.sum(clf.predict(X[te]) == y[te]))
        return hits / len(y)
    a = acc(lab, seed)
    rng = np.random.default_rng(seed + 77)
    ctrl = []
    for i in range(n_control):
        if groups is None:
            yp = rng.permutation(lab)
        else:                                   # permute class labels within each group (keeps group-class balance)
            yp = lab.copy()
            for g in np.unique(groups):
                m = np.where(groups == g)[0]
                yp[m] = rng.permutation(lab[m])
        ctrl.append(acc(yp, seed + 1 + i))
    ctrl = np.array(ctrl)
    return {"acc": float(a), "control_mean": float(ctrl.mean()) if n_control else float("nan"),
            "control_max": float(ctrl.max()) if n_control else float("nan"), "selectivity": float(a - ctrl.mean()) if n_control else float("nan"),
            "chance": 1.0 / K, "n_control": n_control}


# ---------------------------------------------------------------------------------------------------------------- ID
def twonn(X, discard=0.1, D2=None):
    """TwoNN (Facco et al. 2017): mu = r2 / r1 per point, d from the regression of -log(1 - F(mu)) on log(mu) through the
    origin, the largest `discard` fraction of mu dropped (Ansuini et al. 2019's setting). Points with r1 = 0 (duplicates)
    are dropped and counted."""
    if D2 is None:
        D2 = _sqdist(X)
    D2 = D2.copy()
    np.fill_diagonal(D2, np.inf)
    srt = np.sort(D2, axis=1)[:, :2]
    r1, r2 = np.sqrt(srt[:, 0]), np.sqrt(srt[:, 1])
    ok = r1 > 1e-12 * np.median(r2)
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    keep = int(np.floor(n * (1 - discard)))
    F = np.arange(1, n + 1) / n
    x = np.log(mu[:keep]); y = -np.log(1 - F[:keep])
    d = float(np.sum(x * y) / np.sum(x * x))
    return {"id": d, "n_dup": int(np.sum(~ok)), "n": int(n)}


# ---------------------------------------------------------------------------------------------------------------- collapse
def collapse(X, labels):
    """NC1 = Tr(S_W S_B^+) / K in the span of the centred class means; NC2: equinorm (CV of the centred mean norms) and
    equiangularity (SD of pairwise cosines, and their mean against -1/(K-1))."""
    X = np.asarray(X, np.float64)
    lab, K = _lab(labels)
    Xc = X - X.mean(0)
    M = np.stack([Xc[lab == c].mean(0) for c in range(K)])
    R = Xc - M[lab]
    U, s, _ = np.linalg.svd(M.T, full_matrices=False)
    U = U[:, s > s.max() * 1e-10]
    SB = (U.T @ M.T) @ (M @ U) / K
    RU = R @ U
    SW = RU.T @ RU / len(lab)
    nc1 = float(np.trace(SW @ np.linalg.inv(SB)) / K)
    nrm = np.linalg.norm(M, axis=1)
    cos = (M @ M.T) / np.outer(nrm, nrm)
    iu = np.triu_indices(K, 1)
    return {"nc1": nc1, "equinorm_cv": float(nrm.std() / nrm.mean()), "cos_sd": float(cos[iu].std()),
            "cos_mean": float(cos[iu].mean()), "etf_cos": -1.0 / (K - 1)}


# ---------------------------------------------------------------------------------------------------------------- basis
def harmonics(X, labels, n_perm=500, seed=0, order=None):
    """Structure factor of the class means on a cycle: with m_c the centred class means in cyclic order, the power at
    harmonic l is |sum_c m_c e^{2 pi i l c / K}|^2, normalized over l = 1..floor(K/2) (each l counted once with its
    conjugate). Null: the class positions on the circle permuted."""
    X = np.asarray(X, np.float64)
    lab, K = _lab(labels)
    M = np.stack([X[lab == c].mean(0) for c in range(K)])
    M = M - M.mean(0)
    if order is not None:
        M = M[list(order)]
    G = M @ M.T

    def spec(G):
        c = np.arange(K)
        out = []
        for l in range(1, K // 2 + 1):
            e = np.exp(2j * np.pi * l * c / K)
            p = float(np.real(np.conj(e) @ G @ e))
            out.append(p * (1 if (K % 2 == 0 and l == K // 2) else 2))
        out = np.array(out)
        return out / out.sum()
    obs = spec(G)
    rng = np.random.default_rng(seed)
    null = np.array([spec(G[np.ix_(p, p)]) for p in (rng.permutation(K) for _ in range(n_perm))])
    p1 = float((1 + np.sum(null[:, 0] >= obs[0])) / (n_perm + 1))
    return {"share": obs.tolist(), "dominant_l": int(np.argmax(obs) + 1), "l1_share": float(obs[0]),
            "l1_null_mean": float(null[:, 0].mean()), "l1_p": p1, "null_mean": null.mean(0).tolist()}


def ring_h1(X, labels, n_null=300, seed=0):
    """Persistent H1 of the K class means (Euclidean, Vietoris-Rips): the longest bar's lifetime over the means' diameter,
    against K Gaussian points with the means' own covariance (same spectrum, no arrangement)."""
    from ripser import ripser
    X = np.asarray(X, np.float64)
    lab, K = _lab(labels)
    M = np.stack([X[lab == c].mean(0) for c in range(K)])
    M = M - M.mean(0)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    Z = M @ Vt[:K - 1].T                                           # coordinates in the means' own span (K - 1 dims)
    s = s[:K - 1]

    def life(P):
        dg = ripser(P, maxdim=1)["dgms"][1]
        diam = np.max(np.linalg.norm(P[:, None] - P[None], axis=2))
        if len(dg) == 0:
            return 0.0
        return float(np.max(dg[:, 1] - dg[:, 0]) / diam)
    obs = life(Z)
    rng = np.random.default_rng(seed)
    sd = s / np.sqrt(K - 1)
    null = np.array([life(rng.standard_normal((K, len(sd))) * sd) for _ in range(n_null)])
    return {"h1_life": obs, "null_mean": float(null.mean()), "null_q95": float(np.quantile(null, 0.95)),
            "p_one": float((1 + np.sum(null >= obs)) / (n_null + 1))}


def crossed(X, K, T):
    """Two-way decomposition of a crossed battery (rows class-major: index = class * T + carrier): x = mu + a_c + b_t + e_ct.
    Returns the reconstructions (additive mu + a + b, interaction-only mu + a + e) and the energy shares of the within-class
    variation (b + e) carried by the shared carrier term b and by the class-specific interaction e."""
    X = np.asarray(X, np.float64)
    Y = X.reshape(K, T, -1)
    mu = Y.mean(axis=(0, 1))
    a = Y.mean(axis=1) - mu
    b = Y.mean(axis=0) - mu
    e = Y - mu - a[:, None] - b[None]
    eb, ee = K * np.sum(b ** 2), np.sum(e ** 2)
    add = (mu + a[:, None] + b[None]).reshape(K * T, -1)
    inter = (mu + a[:, None] + e).reshape(K * T, -1)
    return {"carrier_share": float(eb / (eb + ee)), "interaction_share": float(ee / (eb + ee)),
            "between_share": float(T * np.sum(a ** 2) / (T * np.sum(a ** 2) + eb + ee))}, add, inter
