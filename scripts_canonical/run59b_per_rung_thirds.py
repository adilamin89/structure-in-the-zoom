"""Run 59b (S78, 2026-09-05) - The rung-four stall, drawn: the per-rung log-PR
deficit below the floor for the three DSI-sorted thirds of run 59, on all
eight grating recordings (the data behind the Section 7 figure).

WHY: Section 7 says an even code repeats its four orientation classes under
the four-class shift, so its ladder has every class mean it will ever have by
four classes and the last rungs add no new directions; the dipole is what
keeps the ladder climbing. Run 59 recorded only the slope difference per
third. This run records the ladder itself: at each rung the log participation
ratio of the accumulated classes minus the mean log participation ratio of
ten random subsets of the same size (the deficit; zero at the top rung by
construction, where the ladder and the floor hold the same trials). The S77
full-population split of this quantity (late rungs carry 0.61-0.74 of the
climb on the full-field drifting recordings, early rungs 0.55-0.66 on the
localized ones) was computed from orientation_zoom.json in the same way.

DESIGN: run 59's subsets exactly (tuned neurons by F-test p < 0.01, thirds by
DSI, the random third with seed 0), the eight-class direction ladder
[1, 2, 3, 4, 6, 8] in angular order, ten-draw floors drawn in run 59's order
(seed 42, draw-major) so each subset's slope difference reproduces run 59's
delta to numerical precision (checked and stored). Per subset:
deficit_k = log PR_k - mean_draws log PR_floor,k; the late fraction is the
deficit at four classes over the deficit at one class, the share of the climb
that remains after four classes. The full population's ladder is read from
orientation_zoom.json.

REGISTERED EXPECTATIONS (written before the run):
S1: the orientation-only third's late fraction is below the direction-
    selective third's in 8 of 8 recordings (the even code has climbed
    further by four classes).
S2: the direction-selective third's late fraction exceeds 0.5 on the three
    full-field drifting recordings (the dipole carries the late rungs there).
S3 (consistency): each subset's slope difference matches run 59's stored
    delta to 1e-6.
A miss is reported at full volume.

Out: ../data_canonical/run59b_per_rung_thirds.json (+ .log)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = DATA / "run59b_per_rung_thirds.json"
spec = importlib.util.spec_from_file_location("run2", HERE / "run2_calibrated_corotating.py")
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
NB = 8


def short(name):
    kind = "D" if "drifting" in name else "L" if "local" in name else "C"
    return kind + name.split("GT")[1][0]


def ladder_rungs(Xt, bl, rng):
    """The observed ladder and its ten-draw floor, rung by rung, with the
    floor draws in ladder_delta's order (draw-major) so the slope difference
    reproduces run 59."""
    members = [np.where(bl == b)[0] for b in range(NB)]
    sizes, obs = [], []
    for c in r2.BIN_COUNTS:
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        sizes.append(int(len(sel)))
        obs.append(float(np.log(max(r2.pr_c(Xt[sel]), 1e-9))))
    nl = np.zeros((r2.N_NULL, len(sizes)))
    for d in range(r2.N_NULL):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(r2.pr_c(Xt[rng.choice(len(Xt), s, replace=False)]), 1e-9))
    flo = nl.mean(axis=0).tolist()
    delta = r2.slope(sizes, np.exp(obs)) - r2.slope(sizes, np.exp(flo))
    return sizes, obs, flo, float(delta)


def main():
    t0 = time.time()
    oz = {r["name"]: r for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    r59 = json.load(open(DATA / "run59_shift_by_direction_selectivity.json"))["rows"]
    names = [n for n in oz if "gratings_" in n and "static" not in n]
    out = {"design": {"n_null": r2.N_NULL, "bin_counts": r2.BIN_COUNTS, "late_rung_index": 3}, "rows": {}}
    for name in names:
        tag = short(name)
        dat = np.load(RAW / f"{name}.npy", allow_pickle=True).item()
        X = np.asarray(dat["sresp"], np.float32); X /= X.std() + 1e-9
        phi = np.asarray(dat["istim"], float).ravel() % (2 * np.pi)
        bl = np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, NB + 1)) - 1, 0, NB - 1)
        Xt = np.ascontiguousarray(X.T); del dat, X
        n, N = Xt.shape
        counts = np.bincount(bl, minlength=NB)
        M = np.stack([Xt[bl == k].mean(0) for k in range(NB)])
        grand = Xt.mean(0)
        ssb = sum(counts[k] * (M[k] - grand) ** 2 for k in range(NB))
        ssw = sum(((Xt[bl == k] - M[k]) ** 2).sum(0) for k in range(NB))
        F = (ssb / (NB - 1)) / (ssw / (n - NB) + 1e-12)
        tuned = (1 - stats.f.cdf(F, NB - 1, n - NB)) < 0.01
        R = M - M.min(0, keepdims=True) + 1e-9
        ang = 2 * np.pi * np.arange(NB) / NB
        dsi = np.abs((R * np.exp(1j * ang[:, None])).sum(0) / R.sum(0))
        tidx = np.where(tuned)[0]; nsub = len(tidx) // 3
        order = tidx[np.argsort(dsi[tidx])]
        rng = np.random.default_rng(0)
        subsets = {"DS": order[-nsub:], "nonDS": order[:nsub], "random": rng.choice(tidx, nsub, replace=False)}
        full_obs = [float(np.log(v[2])) for v in oz[name]["struct_ladder"]]
        full_flo = [float(np.log(v[2])) for v in oz[name]["rand_ladder"]]
        full_def = [o - f for o, f in zip(full_obs, full_flo)]
        row = {"name": name, "n_trials": int(n), "n_neurons": int(N), "n_tuned": int(len(tidx)), "n_sub": int(nsub),
               "full": {"sizes": [int(v[1]) for v in oz[name]["struct_ladder"]], "log_pr": full_obs, "log_pr_floor": full_flo,
                        "deficit": full_def, "delta": float(oz[name]["delta"]),
                        "late_fraction": float(full_def[3] / full_def[0]) if full_def[0] != 0 else None},
               "subsets": {}}
        print(f"[{tag}] trials {n} neurons {N} tuned {len(tidx)} sub {nsub} | full deficit "
              + " ".join(f"{d:+.3f}" for d in full_def) + f" | late {row['full']['late_fraction']:.2f}", flush=True)
        for sname, idx in subsets.items():
            Xs = np.ascontiguousarray(Xt[:, idx])
            sizes, obs, flo, delta = ladder_rungs(Xs, bl, np.random.default_rng(42))
            deficit = [o - f for o, f in zip(obs, flo)]
            d59 = float(r59[tag]["subsets"][sname]["delta"])
            row["subsets"][sname] = {"sizes": sizes, "log_pr": obs, "log_pr_floor": flo, "deficit": deficit, "delta": delta,
                                     "delta_run59": d59, "delta_match_1e6": bool(abs(delta - d59) < 1e-6),
                                     "late_fraction": float(deficit[3] / deficit[0]) if deficit[0] != 0 else None}
            print(f"  {sname:6s}: deficit " + " ".join(f"{d:+.3f}" for d in deficit)
                  + f" | late {row['subsets'][sname]['late_fraction']:.2f} | delta {delta:+.4f} (run59 {d59:+.4f}) | {time.time()-t0:.0f}s", flush=True)
            del Xs
        out["rows"][tag] = row
        json.dump(out, open(OUT, "w"), indent=1)
        del Xt
    rows = out["rows"]; tags = list(rows)
    late = {s: [rows[t]["subsets"][s]["late_fraction"] for t in tags] for s in ("DS", "random", "nonDS")}
    late["full"] = [rows[t]["full"]["late_fraction"] for t in tags]
    drifting = [t for t in tags if t.startswith("D")]
    verdict = {"tags": tags, "late_fraction": late,
               "S1_nonDS_below_DS_count": int(sum(a < b for a, b in zip(late["nonDS"], late["DS"]))),
               "S2_DS_late_gt_0.5_drifting": [bool(rows[t]["subsets"]["DS"]["late_fraction"] > 0.5) for t in drifting],
               "S3_delta_matches_run59": bool(all(rows[t]["subsets"][s]["delta_match_1e6"] for t in tags for s in ("DS", "random", "nonDS")))}
    verdict["S1"] = verdict["S1_nonDS_below_DS_count"] == 8
    verdict["S2"] = all(verdict["S2_DS_late_gt_0.5_drifting"])
    verdict["S3"] = verdict["S3_delta_matches_run59"]
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
