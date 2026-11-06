"""H5c (Oct 26; ledger J15): an analytic floor law for the principal-angle dissimilarity, the counterpart of Eq. 2 for PR.

pa_k (h4_r5_power.cov_stats) = 1 - mean over class pairs of ||U_c^T U_d||_F^2 / k, U_c the top-k residual subspace of class c.
With the labels ignored, two classes are independent samples of one distribution, so
    E ||U_c^T U_d||_F^2 = Tr(E P_c E P_d) = Tr(M^2),   M = E[projector onto the top-k sample subspace of nu = n - 1 samples],
exactly. For an elliptical population x = sqrt(tau) C^(1/2) z (z Gaussian, E tau = 1: tau is the trial's overall gain), M is
diagonal in C's eigenbasis by reflection symmetry, M = diag(m_a) with sum_a m_a = k, and the floor is
    floor_pa(n) = 1 - sum_a m_a^2 / k.
m_a comes from the deterministic equivalent of the sample resolvent (Silverstein and Bai 1995; elliptical / separable form,
Paul and Silverstein 2009, Couillet and Liao 2022 Thm 2.6):
    Qbar(z) = (-z I + g(z) C)^(-1),   g(z) = E_tau[ tau / (1 + tau delta(z)) ],   delta(z) = (1/nu) Tr C Qbar(z),
whose spectral measure along eigenvector a, mu_a(dl) = (1/pi) Im Qbar_aa(l + i0) dl, is the expected squared overlap of the
sample eigenvectors at l with population eigenvector a. The top-k sample subspace is the part of the spectrum above the
quantile l_k with p * int_{l_k} rho = k (rho = (1/(p pi)) Im Tr Qbar), so m_a = mu_a([l_k, inf)).
Inputs: the population spectrum (lambda_a) and the gain distribution. 'gamma' takes the gain's kurtosis only
(tau ~ Gamma with E tau = 1, E tau^2 = kappa); 'one' is the Gaussian (tau = 1, the spectrum alone); an array of tau samples
takes the whole distribution. Cross-class overlaps for labelled classes use Tr(M_c M_d) with each class's own M in its own basis.
"""
import numpy as np
from scipy.special import roots_genlaguerre


def gain_nodes(gain, n_nodes=48):
    """(tau values, weights) for E_tau[.]: 'one' -> tau = 1; ('gamma', kappa) -> Gamma(shape s, scale 1/s), s = 1/(kappa - 1);
    an array -> its empirical distribution (normalized to mean 1)."""
    if isinstance(gain, str) and gain == "one":
        return np.ones(1), np.ones(1)
    if isinstance(gain, tuple) and gain[0] == "gamma":
        kap = float(gain[1])
        if kap <= 1 + 1e-9:
            return np.ones(1), np.ones(1)
        s = 1.0 / (kap - 1.0)
        if s > 50:                                    # kappa < 1.02: Laguerre weights overflow; equal-weight quantile nodes
            from scipy.stats import gamma
            return gamma.ppf((np.arange(64) + 0.5) / 64, s, scale=1 / s), np.full(64, 1 / 64)
        x, w = roots_genlaguerre(n_nodes, s - 1)      # int x^(s-1) e^-x f(x) dx
        w = w / w.sum()
        return x / s, w
    t = np.asarray(gain, float)
    return t / t.mean(), np.full(len(t), 1.0 / len(t))


def _solve_delta(z, lam, nu, tau, w, d0):
    """delta(z) by damped fixed point from the warm start d0."""
    d = d0
    for it in range(4000):
        g = np.sum(w * tau / (1 + tau * d))
        dn = np.sum(lam / (lam * g - z)) / nu
        if abs(dn - d) < 1e-12 * (1 + abs(d)):
            return dn, it
        d = 0.5 * d + 0.5 * dn
    return d, -1


def projector_weights(lam, n, k, gain="one", n_grid=6000, eta_rel=1e-5):
    """m_a = expected weight of the top-k sample subspace of nu = n - 1 centred samples on population eigenvector a.
    lam: population eigenvalues (any order; returned m in the same order). Returns (m, info)."""
    lam = np.asarray(lam, float)
    p, nu = len(lam), n - 1
    k = min(k, nu, p)
    tau, w = gain_nodes(gain)
    lmax_pop = lam.max()
    # sample eigenvalues live in (0, ~ (1 + sqrt(p/nu))^2 * scale]; generous log grid over the nonzero part
    hi = (lam.sum() / nu + lmax_pop) * (1 + np.max(tau) ** 0.5) * 4
    lo = max(lam[lam > 0].min() * 1e-3, hi * 1e-9)
    ls = np.geomspace(hi, lo, n_grid)                  # descending: continuation from large l where delta ~ -Tr C / (nu z)
    eta = eta_rel * lmax_pop
    d = -lam.sum() / (nu * (ls[0] + 1j * eta))
    Qaa = np.empty((n_grid, p))
    rho = np.empty(n_grid)
    bad = 0
    for i, l in enumerate(ls):
        z = l + 1j * eta
        d, it = _solve_delta(z, lam, nu, tau, w, d)
        bad += it < 0
        g = np.sum(w * tau / (1 + tau * d))
        q = (1.0 / (lam * g - z)).imag / np.pi       # mu_a density at l
        Qaa[i] = q
        rho[i] = q.sum()
    # integrate from the top down: cumulative count of sample eigenvalues above l (p * int rho), trapezoid on the grid
    dl = -np.diff(ls)
    cum_a = np.vstack([np.zeros(p), np.cumsum(0.5 * (Qaa[1:] + Qaa[:-1]) * dl[:, None], axis=0)])
    cum = cum_a.sum(1)
    j = np.searchsorted(cum, k)                        # first grid index where k eigenvalues lie above
    if j >= n_grid:
        raise RuntimeError("grid does not reach k eigenvalues")
    f = (k - cum[j - 1]) / (cum[j] - cum[j - 1])
    m = cum_a[j - 1] + f * (cum_a[j] - cum_a[j - 1])
    info = {"k": k, "nu": nu, "l_k": float(ls[j - 1] + f * (ls[j] - ls[j - 1])), "nonconverged": int(bad),
            "total_nonzero": float(cum[-1]), "sum_m": float(m.sum())}
    return m, info


def floor_pa(lam, n, k=5, gain="one", **kw):
    """Label-blind floor of pa_k at n trials per class: 1 - sum m_a^2 / k."""
    m, info = projector_weights(lam, n, k, gain, **kw)
    return float(1 - np.sum(m ** 2) / info["k"]), info


def kappa_gain(Xc, lam=None):
    """Trial-gain kurtosis E tau^2 / (E tau)^2 from centred trials Xc (rows): the radius kurtosis E r^4 / (E r^2)^2 divided by
    the Gaussian quadratic form's own factor 1 + 2 / d_eff, d_eff = (Tr C)^2 / Tr C^2 (E (z'Cz)^2 = (Tr C)^2 + 2 Tr C^2)."""
    r2 = np.sum(Xc * Xc, axis=1)
    if lam is None:
        lam = np.linalg.eigvalsh(Xc.T @ Xc / (len(Xc) - 1))
    deff = lam.sum() ** 2 / np.sum(lam ** 2)
    return float(np.mean(r2 ** 2) / np.mean(r2) ** 2 / (1 + 2 / deff)), float(deff)


def overlap_labelled(Ms):
    """Mean over class pairs of Tr(M_c M_d) for expected projectors given as (V_c, m_c) (columns of V_c = eigenvectors)."""
    out = []
    for c in range(len(Ms)):
        for d in range(c + 1, len(Ms)):
            Vc, mc = Ms[c]; Vd, md = Ms[d]
            O = (Vc.T @ Vd) ** 2
            out.append(float(mc @ O @ md))
    return float(np.mean(out))
