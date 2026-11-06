"""The Allen axis screen closer to
discovery, with label-independent halves and time-aware nulls. Allen VISp, 32 sessions, raw counts, presentations in time order (start times).
Split S1 = odd / even presentation in time (label-independent). The first presentation is dropped (it has no predecessor).
Axes (labels fixed before the split): direction (8; rungs 1,2,3,4,6,8), tf (5; 1..5), running tertiles (3; 1,2,3), random (8 equal
classes, rng 0), prevdir = previous presentation's direction (8), epoch = quartiles of presentation time (4; 1..4).
Nulls per axis per half, 200 permutations each: 'nuisance' = inside strata of the other stimulus/behaviour variables (as
allen_axis_screen; prevdir, epoch and random inside direction x TF), and 'nuisance_epoch' = those strata intersected with
contiguous 80-presentation (~4 min) epochs (the epoch axis keeps only its nuisance strata). Replicability per axis =
max(p_A, p_B), Holm over the 6 axes at 0.05 per session.
Out: data_canonical/allen_axes_halves.json"""
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parent / "raw" / "allen_running"
OUT = HERE.parent / "data_canonical" / "allen_axes_halves.json"
AXES = ("direction", "tf", "running", "random", "prevdir", "epoch")
RUNGS = {"direction": (1, 2, 3, 4, 6, 8), "tf": (1, 2, 3, 4, 5), "running": (1, 2, 3), "random": (1, 2, 3, 4, 6, 8),
         "prevdir": (1, 2, 3, 4, 6, 8), "epoch": (1, 2, 3, 4)}
N_PERM = 200
EPOCH = 80


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shift(ow, G, y, axis):
    """allen_axis_screen.shift with this test's rung table."""
    m = [np.where(y == b)[0] for b in range(y.max() + 1)]
    if len(m) < max(RUNGS[axis]) or any(len(x) == 0 for x in m[:max(RUNGS[axis])]):   # An empty class in this half
        return {"delta": float("nan"), "within": float("nan"), "class_means": float("nan")}
    sets = [np.concatenate(m[:c]) for c in RUNGS[axis]]
    return ow.ladder(G, y, sets, np.random.default_rng(42))


def holm(ps, alpha=0.05):
    order = np.argsort(ps)
    rej = np.zeros(len(ps), bool)
    for i, j in enumerate(order):
        if ps[j] <= alpha / (len(ps) - i):
            rej[j] = True
        else:
            break
    return rej


def run_one(s):
    E = _load("E", "allen_engine.py")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    R2 = _load("r2", "allen_axis_screen.py")
    t0 = time.time()
    sid = s["session"]
    S = E.load_session(sid)
    idx = E.populations(S).get("VISp")
    if idx is None:
        return None
    r = np.load(RUN / f"session_{sid}.npz")
    c = np.load(E.COUNTS / f"session_{sid}.npz")
    assert np.allclose(r["start"], c["start"]), sid
    order = np.argsort(c["start"], kind="stable")
    sp = r["speed"].astype(np.float64)[order]
    lab, tf = S["lab"][order], S["tf"][order]
    X = S["X"][order][:, idx]
    n = len(lab)
    rnd = np.empty(n, int)
    rnd[np.random.default_rng(0).permutation(n)] = np.arange(n) % 8
    L = {"direction": lab, "tf": tf, "running": R2.tertiles(sp), "random": rnd,
         "prevdir": np.r_[-1, lab[:-1]], "epoch": (np.arange(n) * 4) // n}
    keep = np.arange(1, n)                                             # drop the first presentation
    X, L = X[keep], {k: v[keep] for k, v in L.items()}
    blocks = np.arange(len(keep)) // EPOCH                             # 4-min blocks on the full session timeline
    # masking path: history labels (previous stimulus, epoch, blocks) come from the full timeline; then only the
    # population's valid presentations and kept units enter; halves alternate over the presentations that remain
    V = E.pop_view(S, "VISp", idx)
    vt = np.zeros(len(S["lab"]), bool); vt[V["rows"]] = True; vt = vt[order][keep]
    X, L, blocks = X[vt][:, np.isin(idx, V["cols"])], {k: v[vt] for k, v in L.items()}, blocks[vt]
    rank = np.arange(len(X))
    h = rank % 2
    strata = {"direction": L["tf"] * 3 + L["running"], "tf": L["direction"] * 3 + L["running"],
              "running": L["direction"] * 5 + L["tf"]}
    for a in ("random", "prevdir", "epoch"):
        strata[a] = L["direction"] * 5 + L["tf"]
    row = {"session": sid, "N": int(len(idx))}
    for hv in (0, 1):
        sel = np.where(h == hv)[0]
        G = (X[sel] @ X[sel].T).astype(np.float64)
        res = {}
        for ai, axis in enumerate(AXES):
            y = L[axis][sel]
            obs = shift(ow, G, y, axis)
            res[axis] = {**obs, "class_sizes": np.bincount(y).tolist()}
            for nm, st in (("nuisance", strata[axis][sel]),
                           ("nuisance_epoch", strata[axis][sel] if axis == "epoch" else strata[axis][sel] * 1000 + blocks[sel])):
                rng = np.random.default_rng(500 + 10 * hv + ai + (0 if nm == "nuisance" else 100))
                null = []
                for _ in range(N_PERM):
                    yp = y.copy()
                    for u in np.unique(st):
                        i = np.where(st == u)[0]
                        yp[i] = y[i][rng.permutation(len(i))]
                    null.append(shift(ow, G, yp, axis)["delta"])
                null = np.array(null)
                res[axis][nm] = {"mean": float(null.mean()), "sd": float(null.std()), "z": float((obs["delta"] - null.mean()) / null.std()),
                                 "p_one": float((1 + (null >= obs["delta"]).sum()) / (N_PERM + 1)) if np.isfinite(obs["delta"]) else float("nan")}
        row[f"half{hv}"] = res
    for nm in ("nuisance", "nuisance_epoch"):
        prep = np.array([max(row["half0"][a][nm]["p_one"], row["half1"][a][nm]["p_one"]) for a in AXES])
        row[f"replicated_{nm}"] = dict(zip(AXES, holm(prep).tolist()))
        row[f"p_rep_{nm}"] = dict(zip(AXES, prep.tolist()))
    print(sid, " | ".join(f"{a} z {row['half0'][a]['nuisance_epoch']['z']:+.1f}/{row['half1'][a]['nuisance_epoch']['z']:+.1f}" for a in AXES),
          "| replicated:", [a for a in AXES if row["replicated_nuisance_epoch"][a]], f"({time.time() - t0:.0f}s)", flush=True)
    return row


def score(rows):
    v = {"n": len(rows)}
    for nm in ("nuisance", "nuisance_epoch"):
        v[nm] = {a: {"replicated": sum(r[f"replicated_{nm}"][a] for r in rows),
                     "p_le_0.05_half0": sum(r["half0"][a][nm]["p_one"] <= 0.05 for r in rows),
                     "p_le_0.05_half1": sum(r["half1"][a][nm]["p_one"] <= 0.05 for r in rows),
                     "z_median_half0": float(np.nanmedian([r["half0"][a][nm]["z"] for r in rows])),
                     "z_median_half1": float(np.nanmedian([r["half1"][a][nm]["z"] for r in rows])),
                     "n_undefined": int(sum(not np.isfinite(r[h][a]["delta"]) for r in rows for h in ("half0", "half1")))} for a in AXES}
    for a in AXES:
        v.setdefault("delta_median", {})[a] = [float(np.nanmedian([r[h][a]["delta"] for r in rows])) for h in ("half0", "half1")]
        v.setdefault("class_means_median", {})[a] = [float(np.nanmedian([r[h][a]["class_means"] for r in rows])) for h in ("half0", "half1")]
    for sel, conf in (("half0", "half1"), ("half1", "half0")):
        picks = [max(AXES, key=lambda a: np.nan_to_num(r[sel][a]["nuisance_epoch"]["z"], nan=-np.inf)) for r in rows]
        v[f"pick_{sel}"] = {a: picks.count(a) for a in AXES}
    return v


def main():
    sess = json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]
    only = [int(a) for a in sys.argv[1:] if a.isdigit()]   # Rerun only these sessions (the rest are copied from the unmasked run)
    if only:
        sess = [x for x in sess if x["session"] in only]
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        global N_PERM
        N_PERM = 5
        r = run_one(sess[0])
        print({k: v for k, v in r.items() if k.startswith("rep")})
        return
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        rows = [r for r in ex.map(run_one, sess) if r is not None]
    out = {"n_perm": N_PERM, "rows": rows, "score": score(rows)}
    print("SCORE", json.dumps(out["score"], indent=1), flush=True)
    json.dump(out, open(OUT.with_name(OUT.stem + "_partial.json") if only else OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
