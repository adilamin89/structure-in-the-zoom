"""Choose an axis among competitors on one half of each Allen session, test it on the other.

Allen VISp, 32 sessions (E.populations, all units), raw counts (the pooled arm), halves E.halves(cell) (alternate within each
direction x TF cell). Four axes, labels fixed before the split:
  direction  8 classes, ascending angle, rungs 1,2,3,4,6,8
  tf         5 classes, ascending temporal frequency, rungs 1..5
  running    tertiles of the presentation's mean running speed by rank over the session (ties by presentation order), rungs 1,2,3
  random     8 equal classes (rng seed 0 per session), rungs 1,2,3,4,6,8
Shift = cr_order_and_withinbin_controls.ladder (centred-Gram PR, ten-draw floor rng 42) inside one half. Null: 200 permutations of
the axis's labels inside strata of the other declared variables, inside the half (direction in TF x running tertile; tf in
direction x running tertile; running in direction x TF cell; random in direction x TF cell). z = (shift - null mean) / null SD.
Pick on one half = largest z; confirmed on the other = ranks first by z there AND p <= 0.05 there. Both ways.
Out: data_canonical/allen_axis_screen.json"""
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
RUN = HERE.parent / "raw" / "allen_running"
OUT = HERE.parent / "data_canonical" / "allen_axis_screen.json"
AXES = ("direction", "tf", "running", "random")
RUNGS = {"direction": (1, 2, 3, 4, 6, 8), "tf": (1, 2, 3, 4, 5), "running": (1, 2, 3), "random": (1, 2, 3, 4, 6, 8)}
N_PERM = 200


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tertiles(sp):
    """Rank tertiles (0 slowest), ties broken by presentation order (stable sort)."""
    order = np.argsort(sp, kind="stable")
    lab = np.empty(len(sp), int)
    lab[order] = (np.arange(len(sp)) * 3) // len(sp)
    return lab


def labels(S, sp, sid):
    n = len(S["lab"])
    rnd = np.empty(n, int)
    rnd[np.random.default_rng(0).permutation(n)] = np.arange(n) % 8
    return {"direction": S["lab"], "tf": S["tf"], "running": tertiles(sp), "random": rnd}


def strata(L, axis):
    if axis == "direction":
        return L["tf"] * 3 + L["running"]
    if axis == "tf":
        return L["direction"] * 3 + L["running"]
    return L["direction"] * 5 + L["tf"]


def shift(ow, G, y, axis):
    m = [np.where(y == b)[0] for b in range(y.max() + 1)]
    sets = [np.concatenate(m[:c]) for c in RUNGS[axis]]
    return ow.ladder(G, y, sets, np.random.default_rng(42))


def screen_half(ow, X, L, sel, seed):
    G = (X[sel] @ X[sel].T).astype(np.float64)
    out = {}
    for a, axis in enumerate(AXES):
        y = L[axis][sel]
        st = strata(L, axis)[sel]
        obs = shift(ow, G, y, axis)
        rng = np.random.default_rng(seed + a)
        null = []
        for _ in range(N_PERM):
            yp = y.copy()
            for s in np.unique(st):
                i = np.where(st == s)[0]
                yp[i] = y[i][rng.permutation(len(i))]
            null.append(shift(ow, G, yp, axis)["delta"])
        null = np.array(null)
        out[axis] = {**obs, "null_mean": float(null.mean()), "null_sd": float(null.std()), "null_max": float(null.max()),
                     "z": float((obs["delta"] - null.mean()) / null.std()), "p_one": float((1 + (null >= obs["delta"]).sum()) / (N_PERM + 1)),
                     "class_sizes": np.bincount(y).tolist()}
    return out


def run_one(s):
    E = _load("E", "allen_engine.py")
    ow = _load("ow", "cr_order_and_withinbin_controls.py")
    t0 = time.time()
    sid = s["session"]
    S = E.load_session(sid)
    idx = E.populations(S).get("VISp")
    if idx is None:
        return None
    r = np.load(RUN / f"session_{sid}.npz")
    c = np.load(E.COUNTS / f"session_{sid}.npz")
    assert np.allclose(r["start"], c["start"]), sid
    V = E.pop_view(S, "VISp", idx, extra={"speed": r["speed"]})   # masking path: the population's rows only
    sp = V["speed"].astype(np.float64)
    L = labels({"lab": V["lab"], "tf": V["tf"]}, sp, sid)
    X = V["X"]
    h = E.halves(V["cell"])
    row = {"session": sid, "N": int(X.shape[1]), "speed_sd": float(sp.std()),
           "half0": screen_half(ow, X, L, np.where(h == 0)[0], 100), "half1": screen_half(ow, X, L, np.where(h == 1)[0], 200)}
    for sel, conf in (("half0", "half1"), ("half1", "half0")):
        zs = {a: row[sel][a]["z"] for a in AXES}
        pick = max(zs, key=zs.get)
        zc = {a: row[conf][a]["z"] for a in AXES}
        rank = 1 + sum(zc[a] > zc[pick] for a in AXES if a != pick)
        raw = {a: row[sel][a]["delta"] for a in AXES}
        row[f"pick_{sel}"] = {"pick": pick, "pick_raw_shift": max(raw, key=raw.get), "rank_on_other": int(rank),
                              "p_on_other": row[conf][pick]["p_one"],
                              "confirmed": bool(rank == 1 and row[conf][pick]["p_one"] <= 0.05)}
    print(sid, " | ".join(f"{a} z {row['half0'][a]['z']:+.1f}/{row['half1'][a]['z']:+.1f} d {row['half0'][a]['delta']:+.3f}/{row['half1'][a]['delta']:+.3f}" for a in AXES),
          "| pick0", row["pick_half0"]["pick"], row["pick_half0"]["confirmed"], "pick1", row["pick_half1"]["pick"], row["pick_half1"]["confirmed"],
          f"({time.time() - t0:.0f}s)", flush=True)
    return row


def score(rows):
    v = {"n": len(rows)}
    for sel in ("half0", "half1"):
        k = f"pick_{sel}"
        v[k] = {"picked": {a: sum(r[k]["pick"] == a for r in rows) for a in AXES},
                "picked_raw_shift": {a: sum(r[k]["pick_raw_shift"] == a for r in rows) for a in AXES},
                "confirmed": sum(r[k]["confirmed"] for r in rows),
                "confirmed_direction": sum(r[k]["confirmed"] and r[k]["pick"] == "direction" for r in rows)}
    for h in ("half0", "half1"):
        v[h] = {a: {"p_le_0.05": sum(r[h][a]["p_one"] <= 0.05 for r in rows), "z_median": float(np.median([r[h][a]["z"] for r in rows])),
                    "delta_median": float(np.median([r[h][a]["delta"] for r in rows])),
                    "within_median": float(np.median([r[h][a]["within"] for r in rows])),
                    "class_means_median": float(np.median([r[h][a]["class_means"] for r in rows])),
                    "rank_first": sum(max(AXES, key=lambda b: r[h][b]["z"]) == a for r in rows)} for a in AXES}
    for sel, conf in (("half0", "half1"), ("half1", "half0")):
        med = {a: v[sel][a]["z_median"] for a in AXES}
        pick = max(med, key=med.get)
        v[f"pooled_pick_{sel}"] = {"pick": pick, "confirmed_on_other": sum(max(AXES, key=lambda b: r[conf][b]["z"]) == pick and r[conf][pick]["p_one"] <= 0.05 for r in rows)}
    c = min(v["pick_half0"]["confirmed"], v["pick_half1"]["confirmed"])
    v["verdict"] = "STRONG" if c >= 24 else ("ABOVE CHANCE" if c >= 14 else "MISS")
    v["random_sanity_ok"] = all(v[h]["random"]["p_le_0.05"] <= 4 for h in ("half0", "half1"))
    return v


def main():
    sess = json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        global N_PERM
        N_PERM = 5
        print(run_one(sess[0])["pick_half0"])
        return
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        rows = [r for r in ex.map(run_one, sess) if r is not None]
    out = {"n_perm": N_PERM, "rows": rows, "score": score(rows)}
    print("SCORE", json.dumps(out["score"], indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
