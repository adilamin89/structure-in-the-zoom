"""Run74_allen_permutations_modal.py's per-population test (200 label permutations at fixed
class sizes, one 50-draw floor, rng 74 per population) recomputed from the local counts, each population through
allen_engine.pop_view (ALLEN_VARIANT). Estimator code verbatim from run74; 'none' must reproduce
data_canonical/run74_allen_permutations.json (1e-6). The verdict block is rebuilt exactly as run74's local entrypoint does,
against allen_expansion_all_sessions.json (allen_expansion_local.py output).
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler allen_permutations_local.py [n_workers]   Out: data_canonical/run74_allen_permutations.json"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
MIN_UNITS, RUNGS, N_NULL, N_PERM = 50, (1, 2, 3, 4, 6, 8), 50, 200


def pr_sub(G, idx):
    S = G[np.ix_(idx, idx)]; r = S.mean(0)
    C = S - r[None, :] - r[:, None] + r.mean()
    return float(np.trace(C) ** 2 / (C * C).sum())


def ladder_slope(G, lab, logn):
    sets = [np.concatenate([np.where(lab == b)[0] for b in range(c)]) for c in RUNGS]
    return float(np.polyfit(logn, [np.log(pr_sub(G, s)) for s in sets], 1)[0])


def process_session(sid):
    sys.path.insert(0, str(HERE))
    import allen_engine as E
    t0 = time.time()
    S = E.load_session(sid)
    counts = S["X"].astype(np.float32)
    out = {"session": sid, "status": "ok", "areas": {}, "h5_variant": E.VARIANT}
    for area, idx in E.populations(S).items():
        P = E.pop_view(S, area, idx)
        X = counts[np.ix_(P["rows"], P["cols"])].astype(np.float64); G = X @ X.T
        labels = P["lab"]
        sizes = [int(sum((labels == b).sum() for b in range(c))) for c in RUNGS]; logn = np.log(sizes)
        rng = np.random.default_rng(74)
        floor = float(np.polyfit(logn, [np.mean([np.log(pr_sub(G, rng.choice(len(X), s, replace=False))) for _ in range(N_NULL)]) for s in sizes], 1)[0])
        obs = ladder_slope(G, labels, logn) - floor
        perm = [ladder_slope(G, rng.permutation(labels), logn) - floor for _ in range(N_PERM)]
        p = (1 + sum(x >= obs for x in perm)) / (N_PERM + 1)
        out["areas"][area] = {"n_units": int(X.shape[1]), "n_pres": int(X.shape[0]), "delta": obs, "p": float(p),
                              "perm_mean": float(np.mean(perm)), "perm_sd": float(np.std(perm)),
                              "z": float((obs - np.mean(perm)) / (np.std(perm) + 1e-12))}
    print(f"[{sid}] {len(out['areas'])} areas ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    exp = json.load(open(DATA / "allen_expansion_all_sessions.json"))["results"]
    stored = {(s["session"], a): v["delta_dir8"] for s in exp for a, v in s.get("areas", {}).items()}
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        results = list(ex.map(process_session, [s["session"] for s in exp]))
    pops = [(r["session"], a, v) for r in results if r.get("status") == "ok" for a, v in r["areas"].items()]
    diff = [abs(v["delta"] - stored[(s, a)]) for s, a, v in pops if (s, a) in stored]
    visp = [v for _, a, v in pops if a == "VISp"]
    verdict = {"n_populations": len(pops), "R1_median_absdiff": float(np.median(diff)) if diff else None, "R1": bool(diff and np.median(diff) < 0.04),
               "P1_VISp_sig": int(sum(v["p"] <= 0.05 for v in visp)), "P1_VISp_n": len(visp), "P1": bool(sum(v["p"] <= 0.05 for v in visp) >= 30),
               "P2_all_sig": int(sum(v["p"] <= 0.05 for _, _, v in pops)), "P2": bool(sum(v["p"] <= 0.05 for _, _, v in pops) >= 140),
               "count_p_min": int(sum(v["p"] <= 1 / 200 for _, _, v in pops)),
               "statuses": {r["session"]: r["status"] for r in results if r.get("status") != "ok"}}
    out = {"design": {"rungs": RUNGS, "n_null": N_NULL, "n_perm": N_PERM, "min_units": MIN_UNITS}, "results": results, "verdict": verdict}
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(DATA / "run74_allen_permutations.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
