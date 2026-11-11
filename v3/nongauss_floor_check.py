"""The floor of Eq. 2 for samples of any distribution (Appendix M of the extended version), checked by simulation.

For n iid samples with unknown mean and the unbiased sample covariance S,
  E Tr S^2 = [n Tr Sigma^2 + (Tr Sigma)^2]/(n - 1) + kappa/n,   kappa = E|y|^4 - (Tr Sigma)^2 - 2 Tr Sigma^2,
hence the ratio of expectations PR_null = d(n - 1)/[n + d + eps d (n - 1)/n] with eps = kappa/(Tr Sigma)^2: n counts as about
n/(1 + eps). Three distributions on a 60-dimensional power-law spectrum at n = 40, 20,000 draws each: Gaussian (eps = 0), a
heavy-tailed shared gain (y = g z, E g^2 = 1) and independent Poisson units. Prints simulation, formula and Eq. 2 side by side.
Out: stdout (v3/nongauss_floor_check.log holds the run). Run: python v3/nongauss_floor_check.py"""
import numpy as np

rng = np.random.default_rng(0)
D, n, R = 60, 40, 20000
lam = 1.0 / np.arange(1, D + 1) ** 0.8
t1, t2 = lam.sum(), (lam ** 2).sum(); d = t1 ** 2 / t2


def draw(kind, m):
    z = rng.standard_normal((m, D)) * np.sqrt(lam)
    if kind == "gauss":
        return z
    if kind == "gamma_gain":           # y = g z, E g^2 = 1, heavy-tailed
        g2 = rng.gamma(0.5, 2.0, size=(m, 1)); return z * np.sqrt(g2)
    if kind == "poisson":              # independent Poisson units, rates lam*20 (variance = rate)
        r = lam * 20; return (rng.poisson(r, size=(m, D)) - r) / np.sqrt(20)


for kind in ("gauss", "gamma_gain", "poisson"):
    big = draw(kind, 400000); mu4 = np.mean(np.sum(big ** 2, 1) ** 2); S = np.cov(big.T); T1, T2 = np.trace(S), np.trace(S @ S)
    kappa = mu4 - T1 ** 2 - 2 * T2
    tr1, tr2 = [], []
    for _ in range(R):
        X = draw(kind, n); Sm = np.cov(X.T); tr1.append(np.trace(Sm)); tr2.append(np.sum(Sm * Sm))
    pred2 = (n * T2 + T1 ** 2) / (n - 1) + kappa / n
    dd = T1 ** 2 / T2; eps = kappa / T1 ** 2
    print(f"{kind:11s} eps {eps:+.3f}  E TrS2 sim {np.mean(tr2):.4f} pred {pred2:.4f} gauss-pred {(n*T2+T1**2)/(n-1):.4f} | "
          f"PR ratio sim {np.mean(tr1)**2/np.mean(tr2):.3f} formula {dd*(n-1)/(n+dd+eps*dd*(n-1)/n):.3f} Eq2 {dd*(n-1)/(dd+n):.3f}")
