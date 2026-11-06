"""H1-R29 (B1c, Oct 22): mining the B1 miss. Imaging subsampled to Allen's unit counts keeps a NEGATIVE class-mean part, so N
alone does not flip Allen's sign. What does? The full-set class-mean term rewrites exactly as
    c = log[(1 + beta)^2 / (1 + beta^2 PR_W / PR_B + 2 a beta)],   beta = Tr B / Tr W,   a = xi PR_W,   xi = Tr WB / (Tr W Tr B)
(a = how much the class means lie along the fluctuations' own dominant directions, relative to the fluctuation spectrum; checked
to 2e-16 on Allen). At matched N (64, 128, 256 units; 10 random draws, the B1 seeds 1000 N + d for imaging) in both datasets and
both frames (raw; each unit scaled by its pooled within-class SD):
  heterogeneity  CV of per-unit within-class variance; share of Tr W held by the top 5% of units
  sign terms     beta, PR_W, PR_B, xi, a, c (and the ladder's class-mean part: imaging from cr_h1_b1_neuron_count.json at the same
                 draws; Allen computed here on the TF-residualized input)
  attribution    exact counterfactuals at each matched N: c recomputed with one factor (beta, PR_W, PR_B, a) set to the other
                 dataset's median, the rest kept -> which factor flips the sign.
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler cr_h1_b1c_frames.py   Out: data_canonical/cr_h1_b1c.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402

RAW = HERE.parent / "raw"
OUT = HERE.parent / "data_canonical" / "cr_h1_b1c.json"
NS, DRAWS = (64, 128, 256), 10


def within_sd(X, lab):
    return np.sqrt(np.mean([((X[lab == c] - X[lab == c].mean(0)) ** 2).mean(0) for c in range(8)], 0))


def terms(X, lab):
    X = X - X.mean(0)
    cm = np.stack([X[lab == c].mean(0) for c in range(8)]); p = np.bincount(lab, minlength=8) / len(lab)
    n = len(X); Wh = (X - cm[lab]) / np.sqrt(n); Bh = np.sqrt(p)[:, None] * cm
    w1, b1 = (Wh * Wh).sum(), (Bh * Bh).sum()
    w2 = ((Wh @ Wh.T) ** 2).sum(); b2 = ((Bh @ Bh.T) ** 2).sum(); x = ((Wh @ Bh.T) ** 2).sum()
    v = (Wh * Wh).sum(0); vs = np.sort(v)[::-1]; k = max(1, int(round(0.05 * len(v))))
    beta, prw, prb, xi = b1 / w1, w1 ** 2 / w2, b1 ** 2 / b2, x / (w1 * b1)
    return {"beta": float(beta), "PR_W": float(prw), "PR_B": float(prb), "xi": float(xi), "a": float(xi * prw),
            "c": float(np.log((1 + beta) ** 2 / (1 + (b2 + 2 * x) / w2))), "cv_var": float(v.std() / v.mean()),
            "top5_share": float(vs[:k].sum() / vs.sum())}


def c_of(beta, prw, prb, a):
    return float(np.log((1 + beta) ** 2 / (1 + beta ** 2 * prw / prb + 2 * a * beta)))


def frames(X, lab):
    sd = within_sd(X, lab); keep = sd > 1e-6 * np.median(sd[sd > 0])
    return {"raw": X, "z": X[:, keep] / sd[keep]}


def main():
    t0 = time.time(); out = {"imaging": [], "allen": []}
    b1 = {r["name"]: r for r in json.load(open(HERE.parent / "data_canonical" / "cr_h1_b1_neuron_count.json"))["recordings"]}
    for fp in sorted(RAW.glob("gratings*.npy")):
        dat = np.load(fp, allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        lab = E.ow.bins(np.asarray(dat["istim"], float).ravel(), 8); del dat
        for N in NS:
            for d in range(DRAWS):
                idx = np.sort(np.random.default_rng(1000 * N + d).choice(X.shape[1], N, replace=False))
                Xs = np.asarray(X[:, idx], np.float64)
                b1row = [r for r in b1[fp.stem]["rows"] if r["N"] == N and r["draw"] == d][0]
                for fr, Z in frames(Xs, lab).items():
                    out["imaging"].append({"name": fp.stem, "N": N, "draw": d, "frame": fr, **terms(Z, lab),
                                           "class_means_ladder": b1row[fr]["class_means"]})
        print(f"imaging {fp.stem} done ({time.time() - t0:.0f}s)", flush=True)
    sids = [s["session"] for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
    for sid in sids:
        S = E.load_session(sid); idx0 = E.populations(S).get("VISp")
        if idx0 is None:
            continue
        P = E.pop_view(S, "VISp", idx0)   # H5-R1 masking path
        lab = P["lab"]; Y = E.tf_residualize(P["X"], lab, P["cell"]); n0 = P["X"].shape[1]
        for N in [n for n in NS if n <= n0] + [n0]:
            for d in range(DRAWS if N < n0 else 1):
                cols = np.sort(np.random.default_rng(1000 * N + d).choice(n0, N, replace=False)) if N < n0 else np.arange(n0)
                for fr, Z in frames(Y[:, cols], lab).items():
                    out["allen"].append({"session": sid, "N": int(N), "full": bool(N == n0), "draw": d, "frame": fr,
                                         **terms(Z, lab), "class_means_ladder": E.ladder(Z, lab)["class_means"]})
        print(f"allen {sid} done ({time.time() - t0:.0f}s)", flush=True)
    summ = {}
    for N in NS:
        for fr in ("raw", "z"):
            I = [r for r in out["imaging"] if r["N"] == N and r["frame"] == fr]; A = [r for r in out["allen"] if r["N"] == N and r["frame"] == fr]
            if not A:
                continue
            med = lambda rs, k: float(np.median([r[k] for r in rs]))
            s = {f"{ds}_{k}": med(rs, k) for ds, rs in (("img", I), ("allen", A)) for k in ("beta", "PR_W", "PR_B", "xi", "a", "c", "cv_var", "top5_share", "class_means_ladder")}
            s["img_cm_negative_frac"] = float(np.mean([r["class_means_ladder"] < 0 for r in I])); s["allen_cm_negative_frac"] = float(np.mean([r["class_means_ladder"] < 0 for r in A]))
            base = (s["img_beta"], s["img_PR_W"], s["img_PR_B"], s["img_a"])
            s["c_img_medians"] = c_of(*base)
            for j, k in enumerate(("beta", "PR_W", "PR_B", "a")):
                cf = list(base); cf[j] = s[f"allen_{k}"]; s[f"c_img_with_allen_{k}"] = c_of(*cf)
            abase = (s["allen_beta"], s["allen_PR_W"], s["allen_PR_B"], s["allen_a"])
            s["c_allen_medians"] = c_of(*abase)
            for j, k in enumerate(("beta", "PR_W", "PR_B", "a")):
                cf = list(abase); cf[j] = s[f"img_{k}"]; s[f"c_allen_with_img_{k}"] = c_of(*cf)
            summ[f"N{N}_{fr}"] = s
            print(f"N {N} {fr}: " + " ".join(f"{k} {v:+.3f}" for k, v in s.items()), flush=True)
    out["summary"] = summ
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
