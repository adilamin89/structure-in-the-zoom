"""What each comparison measure is blind to by construction, on a planted design (Section 10 and Appendix L of the extended version).

Design: K = 8 classes x 64 samples in d = 128. Class means: Gaussian, centred. Each class's fluctuations have rank 20 with a 1/j spectrum
along an orthonormal basis U0 whose first two columns lie in the span of the class means and the other 18 in its complement (the
shared-subspace design: every class fluctuates along the same U0). Transformations:
  T1  one random rotation of feature space applied to everything;
  T2  each class's residuals rotated by its own random rotation acting only on the complement of the means' span (class means,
      each class's spectrum and the in-span part kept): shared subspace -> class-specific subspaces;
  T3  every residual x 2 (class means kept);
  T4  every centred class mean x 2 (residuals kept).
Statistics: rung's order-averaged shift, within part D and class-mean part (g1_methods.rung_reading, no nulls, fixed seeds so every
arm uses the same floor draws and orders), beta = Tr B / Tr C, label CKA, category RSA, a 5-fold probe, mean-field capacity / radius /
dimension (seed 0: the same Gaussian t-vectors in every arm), simulation capacity, TwoNN, global PR, NC1. Checks: A1 the exact
invariances, A2 the statistics each transformation must move, A3 D rises under T2.
With class-mean scale 1 the classes barely overlap (beta 0.970, probe 1.000); `--mean-scale 0.15` reruns the same transformations
with overlapping classes (base beta near 0.3). Needs scipy and scikit-learn. Out: v3/g1_invariance.json / v3/g1_invariance_ms0.15.json.
Run: python v3/g1_invariance.py [--mean-scale S]"""
import sys
import json
import time
from pathlib import Path

import numpy as np

import g1_methods as gm

K, NC, D_, R_ = 8, 64, 128, 20
rng = np.random.default_rng(20261108)
MS = float(sys.argv[sys.argv.index("--mean-scale") + 1]) if "--mean-scale" in sys.argv else 1.0
M = rng.standard_normal((K, D_))
M -= M.mean(0)
M *= MS
Us, _, _ = np.linalg.svd(M.T, full_matrices=False)
S = Us[:, :K - 1]                                                     # span of the centred means (7 dims)
Qp = np.linalg.svd(np.eye(D_) - S @ S.T)[0][:, :D_ - (K - 1)]        # its complement (121 dims)
U0 = np.hstack([S[:, :2], Qp[:, :R_ - 2]])                           # 2 in-span + 18 complement directions
lam = 1.0 / np.arange(1, R_ + 1)
lab = np.repeat(np.arange(K), NC)
Z = rng.standard_normal((K * NC, R_)) * np.sqrt(lam)
Z -= np.stack([Z[lab == c].mean(0) for c in range(K)])[lab]           # exact zero-mean residuals per class
RES = Z @ U0.T
BASE = (M[lab], RES)


def rot(n, r):
    q, rr = np.linalg.qr(r.standard_normal((n, n)))
    return q * np.sign(np.diag(rr))


def build(means, res):
    return means + res


O = rot(D_, np.random.default_rng(1))
arms = {"base": build(*BASE), "T1": build(*BASE) @ O.T}
res2 = RES.copy()
r2 = np.random.default_rng(2)
for c in range(K):
    Oc = Qp @ rot(D_ - (K - 1), r2) @ Qp.T + S @ S.T                 # identity on the means' span, a rotation on its complement
    res2[lab == c] = RES[lab == c] @ Oc.T
arms["T2"] = build(M[lab], res2)
arms["T3"] = build(M[lab], 2 * RES)
arms["T4"] = build(2 * M[lab], RES)


def stats(X):
    out = {}
    rr = gm.rung_reading(X, lab, n_perm=0, k_orders=50, k_null_orders=1, n_split_orders=20, seed=0, n_share_perm=0, workers=1)
    out["shift_orderavg"] = rr["delta_orderavg"]
    out["D_orderavg"] = rr["split_orderavg"]["D"]
    out["classmean_orderavg"] = rr["split_orderavg"]["class_mean"]
    out["beta"] = gm.classmean_share_plain(X, lab)
    out["cka"] = gm.cka_label(X, lab)
    out["rsa"] = gm.rsa_label(X, lab, n_perm=0)["rsa"]
    out["probe"] = gm.probe(X, lab, n_control=0)["acc"]
    c = gm.capacity_mft(X, lab, n_t=200, seed=0)
    out.update(mft_capacity=c["capacity"], mft_radius=c["radius"], mft_dimension=c["dimension"])
    out["sim_capacity"] = gm.capacity_sim(X, lab, n_proj=2, seed=0)["alpha_sim"]
    out["twonn"] = gm.twonn(X)["id"]
    out["pr"] = gm.pr_global(X)
    out["nc1"] = gm.collapse(X, lab)["nc1"]
    return out


t0 = time.time()
res = {}
for k, X in arms.items():
    res[k] = stats(X)
    print(f"{k}: " + "  ".join(f"{a} {b:+.4f}" for a, b in res[k].items()) + f"  ({time.time() - t0:.0f} s)", flush=True)

SLOPES = {"shift_orderavg", "D_orderavg", "classmean_orderavg"}


def change(a, b, key):
    return abs(b - a) if key in SLOPES else abs(b - a) / max(abs(a), 1e-12)


def moved(a, b, key):
    return abs(b - a) > 0.02 if key in SLOPES else abs(b - a) / max(abs(a), 1e-12) > 0.05


rel = {t: {k: change(res["base"][k], res[t][k], k) for k in res["base"]} for t in ("T1", "T2", "T3", "T4")}
A1 = {"T1": [k for k in res["base"] if k not in ("probe", "sim_capacity")],
      "T2": ["beta", "nc1", "mft_capacity", "mft_radius", "mft_dimension"],
      "T3": ["D_orderavg"], "T4": ["D_orderavg"]}
a1 = {t: {k: rel[t][k] for k in ks} for t, ks in A1.items()}
a1_pass = all(v < 1e-8 for d in a1.values() for v in d.values())
A2 = {"T2": ["D_orderavg"], "T3": ["beta", "mft_radius"], "T4": ["beta", "mft_radius"]}
a2 = {t: {k: moved(res["base"][k], res[t][k], k) for k in ks} for t, ks in A2.items()}
a2_pass = all(v for d in a2.values() for v in d.values())
a3 = res["T2"]["D_orderavg"] > res["base"]["D_orderavg"]
print("A1 exact invariances (max change):", {t: f"{max(d.values()):.1e}" for t, d in a1.items()}, "PASS" if a1_pass else "FAIL")
print("A2 moved:", a2, "PASS" if a2_pass else "FAIL")
print(f"A3 D base {res['base']['D_orderavg']:+.4f} -> T2 {res['T2']['D_orderavg']:+.4f}", "PASS" if a3 else "FAIL")
print("relative / absolute change per arm (who moves):")
for k in res["base"]:
    print(f"  {k:20s} " + "  ".join(f"{t} {rel[t][k]:.2e}" for t in rel))
json.dump({"design": {"K": K, "n_per_class": NC, "d": D_, "rank": R_, "seed": 20261108, "mean_scale": MS}, "values": res, "change": rel,
           "A1": {"values": a1, "pass": a1_pass}, "A2": {"values": a2, "pass": a2_pass}, "A3": {"pass": bool(a3)}},
          open(Path(__file__).with_suffix(".json") if MS == 1.0 else Path(__file__).with_name(f"g1_invariance_ms{MS:g}.json"), "w"),
          indent=1, default=float)
print(f"done ({time.time() - t0:.0f} s)")
