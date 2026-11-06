"""H5b-R6 (Oct 26; registration in data_canonical/ex0_REGISTERED.md; Codex item 15): the held-out-mode test of h4_r3 against a
pooled-fluctuation control that reproduces each half's TF fitting. Allen VISp, 32 sessions, masking path (variant from the sandbox,
registered B), the h4_r3 halves (E.halves on direction x TF cells) and k in {0, 3}.
Pseudo-trials of one half: its raw counts' direction x TF cell means plus raw residuals (from those cell means) permuted across all of
that half's presentations (whole vectors; seeds 7..16); the half's own TF residualization (h4_r3.half_parts) then reruns on them.
Arms: real (h4_r3.measure on the real halves); conditional (measured half pseudo, modes from the real estimator half); full (both halves
pseudo with independent seeds, modes refitted on the pseudo estimator half). Control parts = means over the 10 seeds; net = real - control.
Out: data_canonical/h5b_r6_halfcontrol.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402

spec = importlib.util.spec_from_file_location("r3", HERE / "h4_r3_heldout_modes.py")
R3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R3)
SEEDS = tuple(range(7, 17))
KS = (0, 3)


def pseudo(X, cell, h, s, rng):
    """X with half s replaced by pseudo-trials: cell means + residuals pooled across the half's presentations."""
    m = np.where(h == s)[0]
    Xs = X[m].astype(np.float64)
    mu = {b: Xs[cell[m] == b].mean(0) for b in np.unique(cell[m])}
    M = np.stack([mu[b] for b in cell[m]])
    Xp = X.astype(np.float64).copy()
    Xp[m] = M + (Xs - M)[rng.permutation(len(m))]
    return Xp


def parts(Pm, V, k):
    r = R3.measure(Pm, V, k)
    return {"class_means": r["class_means"], "within": r["within"], "delta": r["delta"]}


def main():
    rows = []
    for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]:
        sid = s["session"]
        S = E.load_session(sid)
        idx = E.populations(S).get("VISp")
        if idx is None:
            continue
        V = E.pop_view(S, "VISp", idx)   # H5-R1 masking path
        X, lab, cell = V["X"].astype(np.float64), V["lab"], V["cell"]
        h = E.halves(cell)
        P = [R3.half_parts(X, lab, cell, h, 0), R3.half_parts(X, lab, cell, h, 1)]
        row = {"session": sid, "N": int(X.shape[1]), "n_half": [int((h == 0).sum()), int((h == 1).sum())]}
        for est, meas in ((0, 1), (1, 0)):
            w = {"real": {f"k{k}": parts(P[meas], P[est]["V"], k) for k in KS}}
            cond = {f"k{k}": [] for k in KS}; full = {f"k{k}": [] for k in KS}
            for sd in SEEDS:
                rng = np.random.default_rng(1000 * sd + 10 * est + meas)
                Pm = R3.half_parts(pseudo(X, cell, h, meas, rng), lab, cell, h, meas)
                Pe = R3.half_parts(pseudo(X, cell, h, est, rng), lab, cell, h, est)
                for k in KS:
                    cond[f"k{k}"].append(parts(Pm, P[est]["V"], k))
                    full[f"k{k}"].append(parts(Pm, Pe["V"], k))
            for name, arm in (("conditional", cond), ("full", full)):
                w[name] = {kk: {q: float(np.mean([x[q] for x in v])) for q in ("class_means", "within", "delta")} for kk, v in arm.items()}
                w[f"net_{name}"] = {kk: {q: w["real"][kk][q] - w[name][kk][q] for q in ("class_means", "within")} for kk in arm}
            row[f"est{est}_meas{meas}"] = w
        rows.append(row)
        x = row["est0_meas1"]
        print(sid, f"real cm k0 {x['real']['k0']['class_means']:+.3f} k3 {x['real']['k3']['class_means']:+.3f} | cond cm k0 "
              f"{x['conditional']['k0']['class_means']:+.3f} k3 {x['conditional']['k3']['class_means']:+.3f} w {x['conditional']['k3']['within']:+.3f} | "
              f"full cm k3 {x['full']['k3']['class_means']:+.3f} w {x['full']['k3']['within']:+.3f}", flush=True)
    summ = {}
    r9 = {r["session"]: r for r in json.load(open(HERE.parent / "data_canonical" / "h4_r9_r3_mechanism.json"))["rows"]}
    for way in ("est0_meas1", "est1_meas0"):
        d = {}
        for arm in ("conditional", "full"):
            d[arm] = {f"k{k}": {"net_cm_negative": int(sum(r[way][f"net_{arm}"][f"k{k}"]["class_means"] < 0 for r in rows)),
                                "control_within_median": float(np.median([r[way][arm][f"k{k}"]["within"] for r in rows])),
                                "control_cm_median": float(np.median([r[way][arm][f"k{k}"]["class_means"] for r in rows])),
                                "net_within_median": float(np.median([r[way][f"net_{arm}"][f"k{k}"]["within"] for r in rows]))} for k in KS}
        d["real_k3_cm_negative"] = int(sum(r[way]["real"]["k3"]["class_means"] < 0 for r in rows))
        d["vs_h4_r9_ctl_k3_median_absdiff"] = float(np.median([abs(r[way]["conditional"]["k3"]["class_means"] - r9[r["session"]][way]["k3"]["ctl_class_means"]) for r in rows]))
        summ[way] = d
    W = ("est0_meas1", "est1_meas0")
    summ["predictions"] = {
        "P1": all(summ[w][a]["k3"]["net_cm_negative"] >= 22 for w in W for a in ("conditional", "full")),
        "P2": all(abs(summ[w][a][f"k{k}"]["control_within_median"]) <= 0.03 for w in W for a in ("conditional", "full") for k in KS),
        "P3": all(summ[w]["vs_h4_r9_ctl_k3_median_absdiff"] <= 0.03 for w in W)}
    print("SUMMARY", json.dumps(summ, indent=1), flush=True)
    json.dump({"registration": "H5b-R6 (ex0_REGISTERED.md)", "variant": E.VARIANT, "rows": rows, "summary": summ},
              open(HERE.parent / "data_canonical" / "h5b_r6_halfcontrol.json", "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
