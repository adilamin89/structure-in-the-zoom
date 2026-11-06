"""H1-R18 (C10 after H2-R39, Oct 22): the finite-sample transfer between the within-class and class-mean parts at the real
design, for Allen (every population, TF-residualized arm, ~75 presentations per class, K = 8) and the pvc-11 arrays (200 per
class, K = 12). The bias is a property of the estimator at the design, so it is measured with the data's own counts, units and
means: B7's surrogates built from the population itself (cr_h1_b7_corr_vs_var logic, generalized to K classes; seeds 7, 8, 9):
  (ii) identical-class null: one shared covariance shape (pooled per-neuron SD profile x a class scalar matching each class's
       total variance) over one pooled correlation structure -> within and class-mean parts that arise from estimation alone;
  (i)  shared correlation, each class's own per-neuron SDs (B7's arm on Allen / macaque).
P1-P4, the sign law and B2's predictions are read against (ii), not against zero.
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler h1_bias_null.py [n_workers]   Out: data_canonical/h1_bias_null.json"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE.parent / "data_canonical" / "h1_bias_null.json"
SEEDS = (7, 8, 9)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def surrogates(Y, lab, K, rng_seed):
    mu = {c: Y[lab == c].mean(0) for c in range(K)}; sd = {c: Y[lab == c].std(0) for c in range(K)}
    Z = np.empty_like(Y, dtype=np.float64)
    for c in range(K):
        s = sd[c].copy(); s[s <= 0] = 1.0; Z[lab == c] = (Y[lab == c] - mu[c]) / s
    sd_pool = np.sqrt(np.mean([sd[c] ** 2 for c in range(K)], 0))
    targets = {"i": sd, "ii": {c: np.sqrt((sd[c] ** 2).sum() / max((sd_pool ** 2).sum(), 1e-30)) * sd_pool for c in range(K)}}
    counts = np.bincount(lab, minlength=K); out = {}
    for mode, tgt in targets.items():
        rows = Z[np.random.default_rng(rng_seed).permutation(len(Z))]; parts, labs, i = [], [], 0
        for c in range(K):
            Zc = rows[i:i + counts[c]].copy(); i += counts[c]
            Zc -= Zc.mean(0); s = Zc.std(0); s[s <= 0] = 1.0; Zc /= s
            parts.append(mu[c] + Zc * tgt[c]); labs.append(np.full(counts[c], c))
        out[mode] = (np.concatenate(parts), np.concatenate(labs))
    return out


def allen_one(sid):
    import h1_allen_engine as E
    S = E.load_session(sid); res = []
    for a, idx in E.populations(S).items():
        P = E.pop_view(S, a, idx)   # H5-R1 masking path
        X, lab = P["X"], P["lab"]
        Y = E.tf_residualize(X, lab, P["cell"])
        acc = {"i": [], "ii": []}
        for s in SEEDS:
            for mode, (Yz, l) in surrogates(Y, lab, 8, s).items():
                acc[mode].append(E.ladder(Yz, l))   # surrogate rows are class-ordered; ladder uses the labels
        res.append({"session": sid, "area": a, "N": int(X.shape[1]),
                    **{m: {k: float(np.mean([q[k] for q in v])) for k in v[0]} for m, v in acc.items()}})
    print(f"allen {sid} {len(res)} populations DONE", flush=True)
    return res


def pvc_one(m):
    M = _load("m", "cr_h1_m_pvc11.py"); ow = _load("ow", "cr_order_and_withinbin_controls.py")
    X, lab, info = M.load_array(m); acc = {"i": [], "ii": []}
    for s in SEEDS:
        for mode, (Yz, l) in surrogates(X, lab, 12, s).items():
            acc[mode].append(ow.ladder((Yz @ Yz.T).astype(np.float64), l, M.sets_from(l, list(range(12))), np.random.default_rng(42)))
    r = {"monkey": m, "N": int(X.shape[1]), **{md: {k: float(np.mean([q[k] for q in v])) for k in v[0]} for md, v in acc.items()}}
    print(f"pvc11 monkey{m} (ii) within {r['ii']['within']:+.3f} class-means {r['ii']['class_means']:+.3f} | (i) within {r['i']['within']:+.3f} DONE", flush=True)
    return r


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    sids = [s["session"] for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        allen = [r for rs in ex.map(allen_one, sids) for r in rs]
        pvc = list(ex.map(pvc_one, (1, 2, 3)))
    visp = [r for r in allen if r["area"] == "VISp"]
    summ = {f"{grp}_{m}_{k}_mean": float(np.mean([r[m][k] for r in rs])) for grp, rs in (("allen_all", allen), ("allen_VISp", visp))
            for m in ("i", "ii") for k in ("delta", "within", "class_means")}
    summ.update({f"{grp}_{m}_{k}_range": [float(np.min([r[m][k] for r in rs])), float(np.max([r[m][k] for r in rs]))]
                 for grp, rs in (("allen_all", allen), ("allen_VISp", visp)) for m in ("ii",) for k in ("within", "class_means")})
    print("SUMMARY", json.dumps(summ, indent=1), flush=True)
    json.dump({"allen": allen, "pvc11": pvc, "summary": summ}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
