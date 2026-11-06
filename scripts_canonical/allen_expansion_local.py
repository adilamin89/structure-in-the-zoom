"""Allen_expansion.py's per-area outputs (delta_dir8, delta_ori4, the three-seed shuffled-label
control, the random-trial delta, physics) recomputed from the local counts (raw/allen_counts) instead of the
Modal NWB extraction, each population through allen_engine.pop_view (ALLEN_VARIANT). The estimator is allen_expansion's
code verbatim (float32 counts as on Modal, one rng 42 per session in sorted-area order, shuffle rngs 500+s / 900+s); 'none' must
reproduce data_canonical/allen_expansion_all_sessions.json to 1e-6 (float32 matmuls on another BLAS).
Out: data_canonical/allen_expansion_all_sessions.json (same schema; canonical copy kept as
allen_expansion_all_sessions.canonical.json for the gate) and data_canonical/allen_floors_all_sessions.json (modal_allen_floors.py's
fields from the same replay: theta_obs/floor of the direction ladder, rung sizes, the random-stimulus ladder; Figure 1C's Allen points).
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler allen_expansion_local.py [sid ...]"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import allen_engine as E  # noqa: E402

DATA = HERE.parent / "data_canonical"
BIN_COUNTS_DIR = [1, 2, 3, 4, 6, 8]
BIN_COUNTS_ORI = [1, 2, 3, 4]
N_NULL = 10
N_SHUFFLE = 3


def pr_trace(X):
    Xc = X - X.mean(axis=0)
    n = Xc.shape[0]
    if n < 3:
        return 1.0
    G = (Xc @ Xc.T).astype(np.float64) / (n - 1)
    tr = float(np.trace(G))
    tr2 = float((G * G).sum())
    if not np.isfinite(tr) or not np.isfinite(tr2) or tr2 <= 0:
        return 1.0
    return tr * tr / tr2


def slope(sizes, prs):
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.maximum(np.asarray(prs, dtype=float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def ladder(X, labels, classes, bin_counts, rng):
    members = [np.where(labels == c)[0] for c in classes]
    sizes, prs = [], []
    for c in bin_counts:
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        sizes.append(len(sel))
        prs.append(pr_trace(X[sel]))
    if len(sizes) < 3:
        return None, None
    th_o = slope(sizes, prs)
    null_logs = np.zeros((N_NULL, len(sizes)))
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            null_logs[d, k] = np.log(max(pr_trace(X[rng.choice(len(X), s, replace=False)]), 1e-9))
    return th_o - slope(sizes, np.exp(null_logs.mean(axis=0))), (sizes, th_o)


def spacing_r(vals):
    vals = np.sort(np.asarray(vals))
    gaps = np.diff(vals)
    gaps = gaps[gaps > 0]
    if len(gaps) < 5:
        return None
    rr = np.minimum(gaps[1:], gaps[:-1]) / np.maximum(gaps[1:], gaps[:-1])
    return float(np.mean(rr))


def physics(X):
    Xc = X - X.mean(axis=0)
    n, d = Xc.shape
    G = (Xc @ Xc.T).astype(np.float64) / (n - 1)
    lam = np.linalg.eigvalsh(G)[::-1]
    lam = lam[lam > 1e-12]
    q = n / d
    edge = np.median(lam) * (1 + np.sqrt(q)) ** 2 / ((1 - np.sqrt(q)) ** 2 if q < 1 else 1.0)
    struct = lam[lam > edge]
    fc = float(struct.sum() / lam.sum()) if len(lam) else None
    k = np.arange(1, min(len(lam), 50) + 1)
    alpha = -slope(k, lam[:len(k)]) if len(lam) >= 10 else None
    gap = float(lam[3] / lam[4]) if len(lam) > 4 else None
    return {"f_c": fc, "n_struct": int(len(struct)), "alpha_top50": alpha, "mp_edge": float(edge),
            "r_struct": spacing_r(struct), "r_2x_edge": spacing_r(lam[lam > 2 * edge]), "r_top30": spacing_r(lam[:30]),
            "gap_l4_l5": gap, "pr_full": pr_trace(X), "eigenvalues_top500": lam[:500].tolist()}


def session(sid):
    S = E.load_session(sid)
    counts = S["X"].astype(np.float32)                     # allen_expansion's float32 counts
    rng = np.random.default_rng(42)
    result = {"session": sid, "status": "ok", "n_presentations": int(len(S["lab"])), "directions": S["dirs"].tolist(),
              "areas": {}, "h5_variant": E.VARIANT}
    floors = {k: v for k, v in result.items() if k != "areas"}; floors["areas"] = {}
    for area, idx in E.populations(S).items():            # sorted VIS areas >= 50 units = allen_expansion's loop
        P = E.pop_view(S, area, idx)
        X = counts[np.ix_(P["rows"], P["cols"])]
        ori = S["dirs"][P["lab"]]
        dirs = S["dirs"]
        ori_mod = ori % 180.0
        oris = np.sort(np.unique(ori_mod))
        d_dir, meta_dir = ladder(X, ori, dirs, BIN_COUNTS_DIR, rng)
        d_ori, _ = ladder(X, ori_mod, oris, BIN_COUNTS_ORI, rng)
        shuf = []
        for s in range(N_SHUFFLE):
            srng = np.random.default_rng(500 + s)
            d_s, _ = ladder(X, ori[srng.permutation(len(ori))], dirs, BIN_COUNTS_DIR, np.random.default_rng(900 + s))
            if d_s is not None:
                shuf.append(d_s)
        # addendum: below 200 presentations the 200 rung cannot be drawn; keep the rungs below n plus n
        rand_sizes = [50, 100, 200, min(400, len(X)), len(X)] if len(X) >= 200 else [r for r in (50, 100) if r < len(X)] + [len(X)]
        rl_obs = [np.mean([pr_trace(X[rng.choice(len(X), s, replace=False)]) for _ in range(3)]) for s in rand_sizes]
        rl_nul = [np.mean([pr_trace(X[rng.choice(len(X), s, replace=False)]) for _ in range(N_NULL)]) for s in rand_sizes]
        result["areas"][area] = {"n_units": int(X.shape[1]), "n_pres": int(X.shape[0]),
                                 "delta_dir8": d_dir, "delta_ori4": d_ori,
                                 "shuffled_mean": float(np.mean(shuf)) if shuf else None,
                                 "shuffled_sd": float(np.std(shuf)) if shuf else None,
                                 "delta_rand": slope(rand_sizes, rl_obs) - slope(rand_sizes, rl_nul),
                                 "physics": physics(X)}
        sizes_dir, th_o_dir = meta_dir if meta_dir else (None, None)
        floors["areas"][area] = {
            "n_units": int(X.shape[1]), "n_pres": int(X.shape[0]), "delta_dir8": d_dir, "delta_ori4": d_ori,
            "theta_obs_dir": th_o_dir, "theta_floor_dir": (th_o_dir - d_dir) if d_dir is not None else None,
            "sizes_dir": [int(z) for z in sizes_dir] if sizes_dir else None,
            "rand_sizes": [int(z) for z in rand_sizes], "rand_obs_pr": [float(z) for z in rl_obs],
            "rand_null_pr": [float(z) for z in rl_nul],
            "theta_obs_rand": slope(rand_sizes, rl_obs), "theta_floor_rand": slope(rand_sizes, rl_nul),
            "pr_full": pr_trace(X)}
        print(f"[{sid}] {area}: n={X.shape[1]} pres={X.shape[0]} d_dir={d_dir:+.4f} d_ori={d_ori:+.4f} "
              f"shuf={result['areas'][area]['shuffled_mean']:+.4f}", flush=True)
    return result, floors


def main():
    canon = DATA / "allen_expansion_all_sessions.canonical.json"
    if not canon.exists():
        shutil.copy(DATA / "allen_expansion_all_sessions.json", canon)
    stored = json.load(open(canon))
    want = [int(x) for x in sys.argv[1:]] or [s["session"] for s in stored["results"]]
    out = json.load(open(DATA / "allen_expansion_all_sessions.json"))
    flo = json.load(open(DATA / "allen_floors_all_sessions.json"))
    pos = {s["session"]: i for i, s in enumerate(out["results"])}
    fpos = {s["session"]: i for i, s in enumerate(flo["results"])}
    for sid in want:
        out["results"][pos[sid]], flo["results"][fpos[sid]] = session(sid)
    json.dump(out, open(DATA / "allen_expansion_all_sessions.json", "w"), indent=1)
    json.dump(flo, open(DATA / "allen_floors_all_sessions.json", "w"), indent=1)
    print("wrote", DATA / "allen_expansion_all_sessions.json", "and allen_floors_all_sessions.json DONE", flush=True)


if __name__ == "__main__":
    main()
