"""Why the floor law sits 4.5% below Allen's permutation floors.
Candidates: (a) the elliptical model holds and the plug-in gain kurtosis / Gamma shape is off; (b) spike counts are not elliptical.
Per population (masking path, as principal_angle_floor_allen.py): 'ell' = Gaussian with the population's covariance, each sample's radius resampled from the real
presentations (R4b's radius model), at the real class sizes, 20 draws; 'law_E' = the law with the empirical squared-radius distribution
(64 quantiles); 'boot' = real presentations drawn without labels into the class sizes (exact label-blind distribution), 20 draws.
If ell matches the permutation floor, the model holds and (a) is the cause; if ell misses as the law does, (b).
Out: data_canonical/principal_angle_floor_allen_cause.json"""
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N_SIM = 20


def pa(X, y, k=5):
    sys.path.insert(0, str(HERE))
    import importlib.util
    spec = importlib.util.spec_from_file_location("h4r5", HERE / "covariance_tools_power.py"); R5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R5)
    return R5.cov_stats(X, y)["pa_5"]


def run_session(s):
    warnings.filterwarnings("ignore")
    sys.path.insert(0, str(HERE))
    import importlib.util
    import allen_engine as E
    spec = importlib.util.spec_from_file_location("r4", HERE / "principal_angle_floor_allen.py"); R4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R4)
    spec = importlib.util.spec_from_file_location("h4r5", HERE / "covariance_tools_power.py"); R5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(R5)
    S = E.load_session(s); out = []
    for area, idx in E.populations(S).items():
        P = E.pop_view(S, area, idx); X = P["X"].astype(np.float64); y = P["lab"]
        sizes = [int((y == c).sum()) for c in range(8)]; ys = np.repeat(np.arange(8), sizes)
        Xc = X - X.mean(0); n, p = Xc.shape
        C = Xc.T @ Xc / (n - 1); lam, V = np.linalg.eigh(C); lam = np.clip(lam, 0, None); A = V * np.sqrt(lam)
        r2 = np.sum(Xc * Xc, 1) / np.mean(np.sum(Xc * Xc, 1)); rad = np.sqrt(r2)
        rng = np.random.default_rng(s % 100000 + len(out))
        ell, boot = [], []
        for _ in range(N_SIM):
            Z = rng.standard_normal((len(ys), p)); Z /= np.linalg.norm(Z, axis=1, keepdims=True) / np.sqrt(p)
            ell.append(R5.cov_stats((Z * rng.choice(rad, len(ys))[:, None]) @ A.T, ys)["pa_5"])
            boot.append(R5.cov_stats(X[rng.permutation(n)][:len(ys)], ys)["pa_5"])
        q = np.quantile(r2, (np.arange(64) + 0.5) / 64)
        out.append({"session": s, "area": str(area), "N": int(p), "ell": float(np.mean(ell)), "boot": float(np.mean(boot)),
                    "law_E": R4.law(lam[lam > lam.max() * 1e-12], sizes, q)})
        print(s, area, {k: round(v, 3) for k, v in out[-1].items() if isinstance(v, float)}, flush=True)
    return out


def main():
    sess = [x["session"] for x in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
    with ProcessPoolExecutor(8, mp_context=get_context("spawn")) as ex:
        rows = [r for part in ex.map(run_session, sess) for r in part]
    r4 = {(r["session"], r["area"]): r for r in json.load(open(HERE.parent / "data_canonical" / "principal_angle_floor_allen.json"))["rows"]}
    for r in rows:
        t = r4[(r["session"], r["area"])]; r.update({"perm": t["floor_perm"], "law_A": t["law_A"], "kappa": t["kappa"]})
    rel = lambda a, b: float(np.median([(r[a] - r[b]) / r[b] for r in rows]))  # noqa: E731
    absrel = lambda a, b: float(np.median([abs(r[a] - r[b]) / r[b] for r in rows]))  # noqa: E731
    v = {"n": len(rows), "boot_vs_perm_signed": rel("boot", "perm"), "boot_vs_perm_abs": absrel("boot", "perm"),
         "ell_vs_perm_signed": rel("ell", "perm"), "ell_vs_perm_abs": absrel("ell", "perm"),
         "lawE_vs_ell_signed": rel("law_E", "ell"), "lawE_vs_perm_signed": rel("law_E", "perm"), "lawA_vs_perm_signed": rel("law_A", "perm"),
         "lawA_vs_ell_signed": rel("law_A", "ell")}
    print("SUMMARY", json.dumps(v, indent=1), flush=True)
    json.dump({"note": "descriptive: why the floor law sits below Allen's permutation floors", "rows": rows, "summary": v}, open(HERE.parent / "data_canonical" / "principal_angle_floor_allen_cause.json", "w"), indent=1)


if __name__ == "__main__":
    main()
