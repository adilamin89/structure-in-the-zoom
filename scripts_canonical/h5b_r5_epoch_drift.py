"""H5b-R5 (Oct 26; registration in data_canonical/ex0_REGISTERED.md): why 'time in session' (H4-R8's epoch axis) replicates in some
Allen VISp sessions and not others. Masking path (variant from the sandbox, registered B); replicate labels and the epoch shift from
the masked H4-R8 rerun in the same sandbox (data_canonical/h4_r8_allen_axes.json). Presentations in time order; epoch = quartile of
presentation time on the full timeline (as R8). R = counts - session direction x TF cell mean; R_e = R - epoch mean. Epoch covariances
W_e; K_ef = Tr(W_e W_f) / (Tr W_e Tr W_f) with the diagonal from independent split halves of each epoch (h4_r5_power.cov_stats's
estimator); sigma_epoch = mean off-diagonal / mean diagonal; drift = 1 - sigma_epoch. Mean-field law for four classes:
log(4 / (1 + 3 sigma)) / log 4.
Out: data_canonical/h5b_r5_epoch_drift.json"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402


def sigma_epochs(R, ep):
    K = len(np.unique(ep))
    W, T = [], np.zeros((K, K))
    for e in range(K):
        r = R[ep == e]; r = r - r.mean(0); W.append(r.T @ r / (len(r) - 1))
    t = np.array([np.trace(w) for w in W])
    for e in range(K):
        for f in range(K):
            T[e, f] = np.sum(W[e] * W[f])
    for e in range(K):
        a, b = R[ep == e][0::2], R[ep == e][1::2]
        a, b = a - a.mean(0), b - b.mean(0)
        T[e, e] = np.sum((a.T @ a / (len(a) - 1)) * (b.T @ b / (len(b) - 1)))
    Kn = T / np.outer(t, t)
    off = ~np.eye(K, dtype=bool)
    return float(Kn[off].mean() / np.diag(Kn).mean())


def main():
    r8 = {r["session"]: r for r in json.load(open(HERE.parent / "data_canonical" / "h4_r8_allen_axes.json"))["rows"]}
    rows = []
    for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]:
        sid = s["session"]
        S = E.load_session(sid)
        idx = E.populations(S).get("VISp")
        if idx is None:
            continue
        c = np.load(E.COUNTS / f"session_{sid}.npz")
        n = len(c["start"])
        order = np.argsort(c["start"], kind="stable")
        epoch_full = np.empty(n, int); epoch_full[order] = (np.arange(n) * 4) // n      # quartiles of time, full timeline
        V = E.pop_view(S, "VISp", idx, extra={"epoch": epoch_full})   # H5-R1 masking path
        X, cell, ep = V["X"].astype(np.float64), V["cell"], V["epoch"]
        q = r8[sid]
        if np.bincount(ep, minlength=4).min() < 20 or not all(np.isfinite(q[h]["epoch"]["delta"]) for h in ("half0", "half1")):
            print(sid, "epoch axis undefined after masking (an epoch with < 20 presentations): excluded", np.bincount(ep, minlength=4), flush=True)
            continue
        M = {b: X[cell == b].mean(0) for b in np.unique(cell)}
        R = X - np.stack([M[b] for b in cell])
        R = R - np.stack([R[ep == e].mean(0) for e in range(4)])[ep]
        sig = sigma_epochs(R, ep)
        ze = [q[h]["epoch"]["nuisance_epoch"]["z"] for h in ("half0", "half1")]
        we = [q[h]["epoch"]["within"] for h in ("half0", "half1")]
        rows.append({"session": sid, "n": int(len(X)), "per_epoch": np.bincount(ep, minlength=4).tolist(), "sigma_epoch": sig,
                     "drift": 1 - sig, "mean_field_within": float(np.log(4 / (1 + 3 * sig)) / np.log(4)),
                     "epoch_z": ze, "epoch_z_mean": float(np.mean(ze)), "epoch_within": we, "epoch_within_mean": float(np.mean(we)),
                     "epoch_null_sd": [q[h]["epoch"]["nuisance_epoch"]["sd"] for h in ("half0", "half1")],
                     "replicated": bool(q["replicated_nuisance_epoch"]["epoch"])})
        x = rows[-1]
        print(sid, f"n {x['n']} drift {x['drift']:.3f} mf {x['mean_field_within']:+.3f} | epoch z {ze[0]:+.1f}/{ze[1]:+.1f} "
              f"within {x['epoch_within_mean']:+.3f} | replicated {x['replicated']}", flush=True)
    rep = np.array([r["replicated"] for r in rows]); dr = np.array([r["drift"] for r in rows])
    summ = {"n": len(rows), "replicated": int(rep.sum()),
            "drift_median_replicated": float(np.median(dr[rep])), "drift_median_not": float(np.median(dr[~rep])),
            "P1_mannwhitney_p": float(stats.mannwhitneyu(dr[rep], dr[~rep], alternative="greater").pvalue),
            "P2_spearman_drift_z": float(stats.spearmanr(dr, [r["epoch_z_mean"] for r in rows])[0]),
            "P3_pearson_meanfield_within": float(np.corrcoef([r["mean_field_within"] for r in rows], [r["epoch_within_mean"] for r in rows])[0, 1]),
            "null_sd_median_replicated": float(np.median([np.mean(r["epoch_null_sd"]) for r in rows if r["replicated"]])),
            "null_sd_median_not": float(np.median([np.mean(r["epoch_null_sd"]) for r in rows if not r["replicated"]])),
            "not_replicated_z_pos_both": int(sum((not r["replicated"]) and min(r["epoch_z"]) > 0 for r in rows))}
    summ["predictions"] = {"P1": summ["P1_mannwhitney_p"] <= 0.05, "P2": summ["P2_spearman_drift_z"] >= 0.5,
                           "P3": summ["P3_pearson_meanfield_within"] >= 0.5}
    print("SUMMARY", json.dumps(summ, indent=1), flush=True)
    json.dump({"registration": "H5b-R5 (ex0_REGISTERED.md)", "variant": E.VARIANT, "rows": rows, "summary": summ},
              open(HERE.parent / "data_canonical" / "h5b_r5_epoch_drift.json", "w"), indent=1)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
