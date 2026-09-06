"""Run 63 (S78b, 2026-09-06) - The deficit decomposed: where along the ladder
does the shift come from, mean structure, fluctuation pooling, or the
coupling between the two? On the data and on the population model.

WHY (the fluctuation-feedback question): the population model of Section 7
reproduces the shift where the even sector dominates and falls 0.08-0.14
short on drifting GT1 and the three localized recordings, and its
within-class subspaces at opposite directions are more aligned than the
data's. The question is which part of the ladder it misses. The
participation ratio of any trial subset is PR = (Tr C)^2 / Tr C^2 with
C = B + W, B the between-class covariance of the subset (its class means
about the subset mean, weighted by class occupancy) and W the pooled
within-class covariance. So

    log PR = 2 log(b1 + w1) - log(b2 + w2 + 2x),
    b1 = Tr B, w1 = Tr W, b2 = Tr B^2, w2 = Tr W^2, x = Tr(BW),

and the deficit of a ladder rung below its floor splits EXACTLY into
    A  = 2 [log(b1 + w1)]_ladder-floor           (the trace, or mean term),
    P  = -[log w2]_ladder-floor                  (fluctuation pooling: how
         the pooled within-class covariance's second moment differs when
         classes are accumulated rather than drawn at random; private
         class-specific fluctuations make it negative),
    Cb = -[log(1 + b2 / w2)]_ladder-floor        (the between-class term),
    Cx = -[log(1 + (b2 + 2x)/w2) - log(1 + b2/w2)]_ladder-floor
         (the coupling term: the overlap of the between-class structure
         with the fluctuation covariance, Tr(BW), the term a
         fluctuation-feedback theory renormalizes).
This is the analogue of integrating out Gaussian fluctuations around a
reference and reading the corrections order by order: A and Cb are the
mean-structure terms, P the fluctuation term, Cx the feedback term.

DESIGN: for each of the eight grating recordings, on the full tuned
population and on run 59's three thirds, the eight-class direction ladder
[1, 2, 3, 4, 6, 8] with ten random-subset floors (run 59b's draw order), the
five traces per rung on the ladder subset and on every floor draw (trial-space
Gram forms, exact), the four terms per rung, and the check that they sum to
the deficit and that (b1 + w1)^2/(b2 + w2 + 2x) equals the centered PR to 1e-6.
The same on the population model of run 60b (resampled arm at matched
within-class dimensionality, seed 1, the recording's K), full population and
its DSI thirds. Reported per subset: the four terms at every rung, their
values at four classes, and each term's share of the climb after four classes.

REGISTERED EXPECTATIONS (written before the run):
D1 (the private fluctuation term exists in the data): on the full tuned
    population the pooling term P at four classes is negative in 8 of 8
    recordings, and its magnitude is larger for the direction-selective third
    than for the orientation-only third in 8 of 8.
D2 (it carries the direction-selective third's late climb): for the
    direction-selective third, the pooling term's share of the climb after
    four classes exceeds the between term's share in at least 6 of 8.
D3 (it is what the model lacks): the data-minus-model difference of P at
    four classes on the full population is largest on the four recordings the
    model under-predicts (D1, L1, L2, L3 hold at least three of the four
    largest differences).
D4 (the model over-couples): the model's coupling term Cx at four classes is
    larger in magnitude than the data's in 8 of 8 (its gain modes follow the
    class means, so Tr(BW) is larger).
A miss is reported at full volume.

Out: ../data_canonical/run63_deficit_decomposition.json (+ .log)
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
OUT = DATA / "run63_deficit_decomposition.json"


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


r2 = load_module("run2", "run2_calibrated_corotating.py")
r60 = load_module("run60", "run60_mixture_model.py")
r60b = load_module("run60b", "run60b_mixture_matchedK.py")
NB = 8
BIN_COUNTS = r2.BIN_COUNTS
N_NULL = r2.N_NULL


def short(name):
    kind = "D" if "drifting" in name else "L" if "local" in name else "C"
    return kind + name.split("GT")[1][0]


def traces(Y, lab):
    """The five traces of C = B + W for trials Y (n x N) with class labels lab."""
    Y = np.asarray(Y, np.float32); n = Y.shape[0]
    xbar = Y.mean(0)
    classes = np.unique(lab)
    Mm = np.stack([np.sqrt((lab == c).sum() / n) * (Y[lab == c].mean(0) - xbar) for c in classes]).astype(np.float32)
    R = np.empty_like(Y)
    for c in classes:
        m = lab == c
        R[m] = Y[m] - Y[m].mean(0)
    R /= np.sqrt(n)
    b1 = float((Mm ** 2).sum()); w1 = float((R ** 2).sum())
    b2 = float(((Mm @ Mm.T) ** 2).sum()); w2 = float(((R @ R.T) ** 2).sum()); x = float(((Mm @ R.T) ** 2).sum())
    return b1, w1, b2, w2, x


def logterms(t):
    b1, w1, b2, w2, x = t
    return {"lt": 2 * np.log(b1 + w1), "lw2": np.log(w2), "lb": np.log1p(b2 / w2), "lbx": np.log1p((b2 + 2 * x) / w2),
            "pr": (b1 + w1) ** 2 / (b2 + w2 + 2 * x)}


def decompose(Xt, bl, rng):
    """Per rung: the four terms of the deficit (ladder minus floor), with the
    floor's log terms averaged over ten draws, and the exactness checks."""
    members = [np.where(bl == b)[0] for b in range(NB)]
    sizes, obs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate(members[:c])
        if len(sel) < 10:
            continue
        sizes.append(int(len(sel)))
        t = logterms(traces(Xt[sel], bl[sel]))
        t["pr_check"] = float(r2.pr_c(Xt[sel]))
        obs.append(t)
    flo = [dict(lt=0.0, lw2=0.0, lb=0.0, lbx=0.0, pr=0.0) for _ in sizes]
    for d in range(N_NULL):
        for k, s in enumerate(sizes):
            idx = rng.choice(len(Xt), s, replace=False)
            t = logterms(traces(Xt[idx], bl[idx]))
            for key in flo[k]:
                flo[k][key] += (np.log(t["pr"]) if key == "pr" else t[key]) / N_NULL
    rows = []
    for k, s in enumerate(sizes):
        o, f = obs[k], flo[k]
        A = o["lt"] - f["lt"]; P = -(o["lw2"] - f["lw2"]); Cb = -(o["lb"] - f["lb"]); Cx = -((o["lbx"] - o["lb"]) - (f["lbx"] - f["lb"]))
        deficit = np.log(o["pr"]) - f["pr"]
        rows.append({"classes": BIN_COUNTS[k] if len(sizes) == len(BIN_COUNTS) else None, "size": s, "deficit": float(deficit),
                     "A": float(A), "P": float(P), "Cb": float(Cb), "Cx": float(Cx),
                     "sum_check": float(A + P + Cb + Cx - deficit), "pr_identity_rel": float(abs(o["pr"] - o["pr_check"]) / o["pr_check"])})
    if len(rows) == len(BIN_COUNTS):
        i4 = BIN_COUNTS.index(4); r4 = rows[i4]
        late_climb = {key: float(0.0 - r4[key]) for key in ("deficit", "A", "P", "Cb", "Cx")}  # the top rung is zero in every term
        share = {key: float(late_climb[key] / late_climb["deficit"]) if late_climb["deficit"] != 0 else None for key in ("A", "P", "Cb", "Cx")}
        return {"rungs": rows, "at_four": {key: r4[key] for key in ("deficit", "A", "P", "Cb", "Cx")}, "late_share": share}
    return {"rungs": rows, "at_four": None, "late_share": None}


def data_subsets(name):
    dat = np.load(RAW / f"{name}.npy", allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel() % (2 * np.pi)
    bl = np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, NB + 1)) - 1, 0, NB - 1)
    Xt = np.ascontiguousarray(X.T); del dat, X
    n, N = Xt.shape
    counts = np.bincount(bl, minlength=NB)
    M = np.stack([Xt[bl == k].mean(0) for k in range(NB)]); grand = Xt.mean(0)
    ssb = sum(counts[k] * (M[k] - grand) ** 2 for k in range(NB)); ssw = sum(((Xt[bl == k] - M[k]) ** 2).sum(0) for k in range(NB))
    F = (ssb / (NB - 1)) / (ssw / (n - NB) + 1e-12); tuned = (1 - stats.f.cdf(F, NB - 1, n - NB)) < 0.01
    Rr = M - M.min(0, keepdims=True) + 1e-9; ang = 2 * np.pi * np.arange(NB) / NB
    dsi = np.abs((Rr * np.exp(1j * ang[:, None])).sum(0) / Rr.sum(0))
    tidx = np.where(tuned)[0]; nsub = len(tidx) // 3; order = tidx[np.argsort(dsi[tidx])]
    rng = np.random.default_rng(0)
    subsets = {"full": tidx, "DS": order[-nsub:], "nonDS": order[:nsub], "random": rng.choice(tidx, nsub, replace=False)}
    return Xt, bl, subsets


def model_subsets(name, tag):
    bl, meas, arrays = r60.measure(name)
    K = json.load(open(DATA / "run60b_mixture_matchedK_armB.json"))["rows"][tag]["K_matched"]
    rng = np.random.default_rng(107)
    d_vec = r60b.d_vector("B", meas, arrays, rng)
    Xs, R = r60.make_mixture(len(bl), bl, d_vec, arrays["amp_tuned"], meas["within_corr"], K, 1.0, rng)
    mdsi = r60.dsi_osi(R)[0]; order = np.argsort(mdsi); nsub = len(mdsi) // 3
    subsets = {"full": np.arange(len(mdsi)), "DS": order[-nsub:], "nonDS": order[:nsub], "random": rng.choice(len(mdsi), nsub, replace=False)}
    return Xs, bl, subsets, int(K)


def main():
    t0 = time.time()
    oz = {r["name"]: r for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    names = [n for n in oz if "gratings_" in n and "static" not in n]
    out = {"design": {"bin_counts": BIN_COUNTS, "n_null": N_NULL, "model": "run60b arm B, matched K, seed 1"}, "rows": {}}
    for name in names:
        tag = short(name); row = {"name": name, "data": {}, "model": {}}
        Xt, bl, subs = data_subsets(name)
        for sname, idx in subs.items():
            Xs = np.ascontiguousarray(Xt[:, idx])
            row["data"][sname] = decompose(Xs, bl, np.random.default_rng(42)); del Xs
            a4 = row["data"][sname]["at_four"]; sh = row["data"][sname]["late_share"]
            print(f"[{tag}] data {sname:6s}: at four classes deficit {a4['deficit']:+.3f} = A {a4['A']:+.3f} + P {a4['P']:+.3f} + Cb {a4['Cb']:+.3f} + Cx {a4['Cx']:+.3f} | "
                  f"late shares A {sh['A']:.2f} P {sh['P']:.2f} Cb {sh['Cb']:.2f} Cx {sh['Cx']:.2f} | {time.time()-t0:.0f}s", flush=True)
        del Xt
        Xm, blm, msubs, K = model_subsets(name, tag); row["model_K"] = K
        for sname, idx in msubs.items():
            Xs = np.ascontiguousarray(Xm[:, idx])
            row["model"][sname] = decompose(Xs, blm, np.random.default_rng(42)); del Xs
            a4 = row["model"][sname]["at_four"]; sh = row["model"][sname]["late_share"]
            print(f"[{tag}] model {sname:6s}: at four classes deficit {a4['deficit']:+.3f} = A {a4['A']:+.3f} + P {a4['P']:+.3f} + Cb {a4['Cb']:+.3f} + Cx {a4['Cx']:+.3f} | "
                  f"late shares A {sh['A']:.2f} P {sh['P']:.2f} Cb {sh['Cb']:.2f} Cx {sh['Cx']:.2f} | {time.time()-t0:.0f}s", flush=True)
        del Xm
        checks = [r["sum_check"] for s in row["data"].values() for r in s["rungs"]] + [r["sum_check"] for s in row["model"].values() for r in s["rungs"]]
        prs = [r["pr_identity_rel"] for s in row["data"].values() for r in s["rungs"]] + [r["pr_identity_rel"] for s in row["model"].values() for r in s["rungs"]]
        row["max_sum_check"] = float(np.max(np.abs(checks))); row["max_pr_identity_rel"] = float(np.max(prs))
        print(f"[{tag}] exactness: max |A+P+Cb+Cx-deficit| {row['max_sum_check']:.2e}, max PR identity rel {row['max_pr_identity_rel']:.2e}", flush=True)
        out["rows"][tag] = row
        json.dump(out, open(OUT, "w"), indent=1)
    rows = out["rows"]; tags = list(rows)
    P4_full = {t: rows[t]["data"]["full"]["at_four"]["P"] for t in tags}
    d1a = int(sum(P4_full[t] < 0 for t in tags))
    d1b = int(sum(abs(rows[t]["data"]["DS"]["at_four"]["P"]) > abs(rows[t]["data"]["nonDS"]["at_four"]["P"]) for t in tags))
    d2 = int(sum(rows[t]["data"]["DS"]["late_share"]["P"] > rows[t]["data"]["DS"]["late_share"]["Cb"] for t in tags))
    diffP = {t: rows[t]["data"]["full"]["at_four"]["P"] - rows[t]["model"]["full"]["at_four"]["P"] for t in tags}
    top4 = sorted(tags, key=lambda t: diffP[t])[:4]   # most negative data-minus-model P = the data's pooling term more negative than the model's
    d3 = int(sum(t in ("D1", "L1", "L2", "L3") for t in top4))
    d4 = int(sum(abs(rows[t]["model"]["full"]["at_four"]["Cx"]) > abs(rows[t]["data"]["full"]["at_four"]["Cx"]) for t in tags))
    verdict = {"tags": tags, "P_at_four_full_data": P4_full, "P_at_four_full_model": {t: rows[t]["model"]["full"]["at_four"]["P"] for t in tags},
               "diff_P_data_minus_model": diffP, "top4_most_negative_diff": top4,
               "D1_P_negative_count": d1a, "D1_DS_gt_nonDS_count": d1b, "D1": bool(d1a == 8 and d1b == 8),
               "D2_count": d2, "D2": bool(d2 >= 6), "D3_count_in_top4": d3, "D3": bool(d3 >= 3), "D4_count": d4, "D4": bool(d4 == 8),
               "max_sum_check": max(rows[t]["max_sum_check"] for t in tags), "max_pr_identity_rel": max(rows[t]["max_pr_identity_rel"] for t in tags)}
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
