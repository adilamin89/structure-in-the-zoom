"""Run 64a (S78b, 2026-09-06) - Is the class-private within-class variability
localized in neuron space? The diagnostic that chooses the next population
model.

WHY: run 63 puts the direction-aligned shift in the fluctuation-pooling term:
accumulating classes pools progressively more class-private low-dimensional
variability, and the population model (shared gain modes weighted by the
rate) has half the data's pooling term on the recordings it under-predicts.
Two constructions give class-private low-dimensional variability: (a)
independent per-neuron gain fluctuations with heavy-tailed amplitudes, whose
covariance is diagonal-heavy, whose effective dimension is set by the fourth
moment of the amplitude distribution, d_eff = N <s^2>^2 / <s^4>, and whose
top modes are localized on the few neurons most active at the class; (b)
class-specific shared modes, extended over the population. The two differ in
neuron space.

DESIGN: for each grating recording, the tuned neurons, and each of the eight
direction classes (all trials of the class): the within-class residual matrix
R_c; per-neuron residual variance s_j^2; the diagonal participation ratio
PR_diag = (sum s^2)^2 / sum s^4 (the effective dimension of the independent
part alone); the full within-class participation ratio PR_full on the same
trials (centered Gram); the top-10 eigenvectors of the within-class
covariance in neuron space (through the trial-space eigendecomposition) and
each one's neuron-space participation ratio 1 / sum_j v_j^4 (N for an
extended mode, tens for a localized one); the fraction of within-class
variance in the top 10 modes; the spread of log s_j^2. The same on the
population model of run 60b (resampled arm, matched K, seed 1).

REGISTERED EXPECTATIONS (written before the run):
Z1 (localized): the neuron-space participation ratio of the top within-class
    eigenvector, averaged over classes, is smaller on the data than on the
    model in 8 of 8 recordings.
Z2 (the fourth moment carries the dimensionality): on the data PR_diag is
    within a factor of 3 of PR_full, averaged over classes, in at least 6 of 8
    recordings (independent heavy-tailed variability alone gives the
    within-class dimensionality); on the model PR_diag exceeds PR_full by more
    than a factor of 3 in 8 of 8 (its dimensionality comes from shared modes).
A miss is reported at full volume.

Out: ../data_canonical/run64a_within_class_localization.json (+ .log)
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUT = DATA / "run64a_within_class_localization.json"


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


r63 = load_module("run63", "run63_deficit_decomposition.py")
NB = 8
TOP = 10


def class_stats(Xt, bl, idx):
    """Per class on neurons idx: PR_diag, PR_full, neuron-space PR of the top
    eigenvectors, top-10 variance fraction, spread of log variances."""
    out = []
    Y = np.ascontiguousarray(Xt[:, idx], dtype=np.float32)
    for c in range(NB):
        R = Y[bl == c]; R = R - R.mean(0)
        n = R.shape[0]
        s2 = (R ** 2).mean(0).astype(np.float64)
        pr_diag = float(s2.sum() ** 2 / (s2 ** 2).sum())
        G = (R @ R.T).astype(np.float64) / max(n - 1, 1)
        w, U = np.linalg.eigh(G); order = np.argsort(w)[::-1]; w, U = w[order], U[:, order]
        pr_full = float(w.sum() ** 2 / (w ** 2).sum())
        top_frac = float(w[:TOP].sum() / w.sum())
        # neuron-space eigenvectors v_k = R^T u_k / sqrt((n-1) w_k)
        V = (R.T @ U[:, :TOP]) / np.sqrt(np.maximum((n - 1) * w[:TOP], 1e-12))
        V = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
        pr_neuron = (1.0 / (V ** 4).sum(0)).tolist()
        spread = float(np.std(np.log(s2 + 1e-12)))
        out.append({"class": c, "n_trials": int(n), "pr_diag": pr_diag, "pr_full": pr_full, "pr_diag_over_full": pr_diag / pr_full,
                    "top10_var_frac": top_frac, "pr_neuron_top": pr_neuron, "log_var_spread": spread})
    return out


def summarize(cls, N):
    return {"pr_diag_mean": float(np.mean([c["pr_diag"] for c in cls])), "pr_full_mean": float(np.mean([c["pr_full"] for c in cls])),
            "ratio_mean": float(np.mean([c["pr_diag_over_full"] for c in cls])),
            "pr_neuron_top1_mean": float(np.mean([c["pr_neuron_top"][0] for c in cls])),
            "pr_neuron_top10_mean": float(np.mean([np.mean(c["pr_neuron_top"]) for c in cls])),
            "top10_var_frac_mean": float(np.mean([c["top10_var_frac"] for c in cls])),
            "log_var_spread_mean": float(np.mean([c["log_var_spread"] for c in cls])), "N": int(N)}


def main():
    t0 = time.time()
    oz = {r["name"]: r for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    names = [n for n in oz if "gratings_" in n and "static" not in n]
    out = {"design": {"top": TOP, "model": "run60b arm B, matched K, seed 1"}, "rows": {}}
    for name in names:
        tag = r63.short(name)
        Xt, bl, subs = r63.data_subsets(name)
        dcls = class_stats(Xt, bl, subs["full"]); dsum = summarize(dcls, len(subs["full"])); del Xt
        Xm, blm, msubs, K = r63.model_subsets(name, tag)
        mcls = class_stats(Xm, blm, msubs["full"]); msum = summarize(mcls, len(msubs["full"])); del Xm
        out["rows"][tag] = {"name": name, "data": {"summary": dsum, "classes": dcls}, "model": {"summary": msum, "classes": mcls, "K": K}}
        print(f"[{tag}] data : PR_diag {dsum['pr_diag_mean']:7.1f} PR_full {dsum['pr_full_mean']:6.1f} ratio {dsum['ratio_mean']:5.2f} | neuron-space PR top1 {dsum['pr_neuron_top1_mean']:7.1f} top10 {dsum['pr_neuron_top10_mean']:7.1f} of N {dsum['N']} | top10 var {dsum['top10_var_frac_mean']:.2f} | log-var spread {dsum['log_var_spread_mean']:.2f} | {time.time()-t0:.0f}s", flush=True)
        print(f"[{tag}] model: PR_diag {msum['pr_diag_mean']:7.1f} PR_full {msum['pr_full_mean']:6.1f} ratio {msum['ratio_mean']:5.2f} | neuron-space PR top1 {msum['pr_neuron_top1_mean']:7.1f} top10 {msum['pr_neuron_top10_mean']:7.1f} of N {msum['N']} | top10 var {msum['top10_var_frac_mean']:.2f} | log-var spread {msum['log_var_spread_mean']:.2f}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)
    rows = out["rows"]; tags = list(rows)
    z1 = int(sum(rows[t]["data"]["summary"]["pr_neuron_top1_mean"] < rows[t]["model"]["summary"]["pr_neuron_top1_mean"] for t in tags))
    z2d = int(sum(rows[t]["data"]["summary"]["ratio_mean"] <= 3.0 for t in tags))
    z2m = int(sum(rows[t]["model"]["summary"]["ratio_mean"] > 3.0 for t in tags))
    verdict = {"Z1_count": z1, "Z1": bool(z1 == 8), "Z2_data_within_3x_count": z2d, "Z2_model_beyond_3x_count": z2m, "Z2": bool(z2d >= 6 and z2m == 8),
               "data_pr_neuron_top1": {t: rows[t]["data"]["summary"]["pr_neuron_top1_mean"] for t in tags},
               "model_pr_neuron_top1": {t: rows[t]["model"]["summary"]["pr_neuron_top1_mean"] for t in tags},
               "data_ratio": {t: rows[t]["data"]["summary"]["ratio_mean"] for t in tags}, "model_ratio": {t: rows[t]["model"]["summary"]["ratio_mean"] for t in tags}}
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
