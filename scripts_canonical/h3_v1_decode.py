"""H3 part A2: the direction shift beside and with the decoders and noise-correlation measures people use on V1
(registrations H3-R4..R6, ex0_REGISTERED.md).

Input as Table 9 (cr_withinclass_controls 'orig'): deconvolved responses / global SD, eight direction classes (run72.bins).
  R4 decoding: multinomial logistic (C = 1, z-scored on the training folds), 5-fold stratified CV, random neuron subsets of
     N = 64, 256, 1024 (5 draws); DS thirds cross-fitted as cr_ds_thirds_crossfit.select (select on one half, decode inside
     the other by 5-fold CV), at N = 256 and 1024 neurons drawn from each third (5 draws).
  R5 noise correlations: 2,000 tuned neurons (run72.tuned_mask, seed 0); r_ij(c) per class; NCchange = mean over pairs of the
     SD over classes of r_ij(c), minus the same statistic with class labels permuted (10 draws); pooled mean |r| and mean r.
  R6 covariance-aware decoding: z-score + PCA to 32 dims fitted on the training folds; LDA vs regularized QDA (0.1, 0.5);
     LDA trained on training projections shuffled within class per PC (correlations removed in PC space), tested on real.
Shift numbers come from Table 9's artifact (cr_withinclass_controls.json) and cr_ds_thirds_crossfit.json.
Out: data_canonical/h3_v1_decode.json
"""
import importlib.util
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the eight Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
DATA = HERE.parent / "data_canonical"
OUT = DATA / "h3_v1_decode.json"
NS = (64, 256, 1024)
N_DRAW = 5
N_NC = 2000
N_NC_NULL = 10


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decode_acc(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    hits = 0
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        hits += int((m.predict(sc.transform(X[te])) == y[te]).sum())
    return hits / len(y)


def nc_sd(X, y, nc=8):
    """Mean over neuron pairs of the SD over classes of the within-class correlation r_ij(c)."""
    d = X.shape[1]
    s1 = np.zeros((d, d))
    s2 = np.zeros((d, d))
    for c in range(nc):
        Z = X[y == c] - X[y == c].mean(0)
        Z /= Z.std(0) + 1e-12
        R = Z.T @ Z / len(Z)
        s1 += R
        s2 += R * R
    var = np.clip(s2 / nc - (s1 / nc) ** 2, 0, None)
    iu = np.triu_indices(d, 1)
    return float(np.sqrt(var[iu]).mean()), s1 / nc


def cov_decoders(X, y, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(seed)
    hits = {"lda": 0, "qda_0.1": 0, "qda_0.5": 0, "lda_shuffled_train": 0}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        pca = PCA(32, svd_solver="randomized", random_state=seed).fit(sc.transform(X[tr]))
        Ztr, Zte = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[te]))
        hits["lda"] += int((LinearDiscriminantAnalysis().fit(Ztr, y[tr]).predict(Zte) == y[te]).sum())
        for r in (0.1, 0.5):
            q = QuadraticDiscriminantAnalysis(reg_param=r).fit(Ztr, y[tr])
            hits[f"qda_{r}"] += int((q.predict(Zte) == y[te]).sum())
        Zs = Ztr.copy()
        for c in np.unique(y[tr]):
            idx = np.where(y[tr] == c)[0]
            perm = np.argsort(rng.random((len(idx), Zs.shape[1])), axis=0)
            Zs[idx] = np.take_along_axis(Ztr[idx], perm, axis=0)
        hits["lda_shuffled_train"] += int((LinearDiscriminantAnalysis().fit(Zs, y[tr]).predict(Zte) == y[te]).sum())
    return {k: v / len(y) for k, v in hits.items()}


def run_one(fp):
    warnings.filterwarnings("ignore")
    r72 = _load("run72", "run72_paper_designs_split.py")
    dsx = _load("dsx", "cr_ds_thirds_crossfit.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    raw = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    y = r72.bins(np.asarray(dat["istim"], float).ravel(), 8)
    del dat
    X = raw / (raw.std() + 1e-9)
    del raw
    rng = np.random.default_rng(0)
    row = {"name": fp.stem, "n_trials": int(len(y)), "n_neurons": int(X.shape[1])}
    # R4 random subsets
    dec = {}
    for N in NS:
        accs = [decode_acc(X[:, rng.choice(X.shape[1], N, replace=False)], y, seed=d) for d in range(N_DRAW)]
        dec[str(N)] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs))}
    row["decode"] = dec
    print(f"{fp.stem[:30]} decode {dec} ({time.time() - t0:.0f}s)", flush=True)
    # R4 DS thirds, cross-fitted (same halves as cr_ds_thirds_crossfit)
    half = np.zeros(len(y), int)
    for k in range(8):
        idx = np.where(y == k)[0]
        half[idx[1::2]] = 1
    thirds = {}
    for sel_h, meas_h in ((0, 1), (1, 0)):
        ds, nds = dsx.select(X[half == sel_h], y[half == sel_h])
        Xm, ym = X[half == meas_h], y[half == meas_h]
        res = {}
        for tag, cols in (("DS", ds), ("nonDS", nds)):
            res[tag] = {str(N): float(np.mean([decode_acc(Xm[:, rng.choice(cols, N, replace=False)], ym, seed=d)
                                               for d in range(N_DRAW)])) for N in (256, 1024)}
        thirds[f"sel{sel_h}_meas{meas_h}"] = res
    row["thirds_decode"] = thirds
    print(f"{fp.stem[:30]} thirds {thirds} ({time.time() - t0:.0f}s)", flush=True)
    # R5 noise correlations
    tuned = np.where(r72.tuned_mask(X, y))[0]
    cols = np.random.default_rng(0).choice(tuned, min(N_NC, len(tuned)), replace=False)
    Xn = X[:, cols].astype(np.float64)
    obs, Rbar = nc_sd(Xn, y)
    nrng = np.random.default_rng(1)
    nulls = [nc_sd(Xn, nrng.permutation(y))[0] for _ in range(N_NC_NULL)]
    iu = np.triu_indices(len(cols), 1)
    row["noise_corr"] = {"n_tuned": int(len(tuned)), "n_used": int(len(cols)), "sd_over_classes": obs,
                         "null_mean": float(np.mean(nulls)), "null_sd": float(np.std(nulls)),
                         "NCchange": float(obs - np.mean(nulls)), "mean_abs_r": float(np.abs(Rbar[iu]).mean()),
                         "mean_r": float(Rbar[iu].mean())}
    print(f"{fp.stem[:30]} NC {row['noise_corr']} ({time.time() - t0:.0f}s)", flush=True)
    # R6 covariance-aware decoders
    row["cov_decoders"] = cov_decoders(X, y)
    print(f"{fp.stem[:30]} cov {row['cov_decoders']} ({time.time() - t0:.0f}s)", flush=True)
    return row


def score(rows):
    from scipy import stats
    t9 = {r["name"]: r["orig"] for r in json.load(open(DATA / "cr_withinclass_controls.json"))["rows"]}
    th = {r["name"]: r for r in json.load(open(DATA / "cr_ds_thirds_crossfit.json"))["rows"]}
    names = [r["name"] for r in rows]
    delta = np.array([t9[n]["delta"] for n in names])
    within = np.array([t9[n]["within"] for n in names])
    cm = np.array([t9[n]["class_means"] for n in names])
    acc = np.array([r["decode"]["256"]["mean"] for r in rows])
    ncc = np.array([r["noise_corr"]["NCchange"] for r in rows])
    gain = {k: np.array([r["cov_decoders"][k] - r["cov_decoders"]["lda"] for r in rows]) for k in ("qda_0.1", "qda_0.5")}
    shuf = np.array([r["cov_decoders"]["lda"] - r["cov_decoders"]["lda_shuffled_train"] for r in rows])

    def sp(a, b):
        return float(stats.spearmanr(a, b)[0])

    def partial(a, b, c):
        """Pearson partial correlation of a and b given c."""
        ra = a - np.polyval(np.polyfit(c, a, 1), c)
        rb = b - np.polyval(np.polyfit(c, b, 1), c)
        return float(np.corrcoef(ra, rb)[0, 1])
    out = {"n": len(rows),
           "Q6_acc256_vs_delta": sp(acc, delta), "Q6_acc256_vs_within": sp(acc, within),
           "Q6_acc256_vs_classmean": sp(acc, cm),
           "Q7_NCchange_vs_within": sp(ncc, within), "Q7_NCchange_vs_delta": sp(ncc, delta),
           "Q7_partial_within_NCchange_given_acc": partial(within, ncc, acc),
           "Q7_partial_within_acc_given_NCchange": partial(within, acc, ncc),
           "Q8_qda0.1_gain_vs_within": sp(gain["qda_0.1"], within), "Q8_qda0.5_gain_vs_within": sp(gain["qda_0.5"], within),
           "Q8_qda0.1_gain_vs_delta": sp(gain["qda_0.1"], delta),
           "Q9_shuffle_loss_vs_NCchange": sp(shuf, ncc), "Q9_shuffle_loss_vs_within": sp(shuf, within)}
    cnt = {}
    for k in ("sel0_meas1", "sel1_meas0"):
        for N in ("256", "1024"):
            cnt[f"{k}_N{N}_decode_DS_gt_nonDS"] = int(sum(r["thirds_decode"][k]["DS"][N] > r["thirds_decode"][k]["nonDS"][N]
                                                        for r in rows))
        cnt[f"{k}_delta_DS_gt_nonDS"] = int(sum(th[r["name"]][k]["DS"] > th[r["name"]][k]["nonDS"] for r in rows))
    out["thirds_counts"] = cnt
    return out


def r6b_one(fp):
    """H3-R6b: off-ceiling covariance-aware decoding (256 neurons, 16 standardized PCs, QDA reg by inner CV)."""
    warnings.filterwarnings("ignore")
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    r72 = _load("run72", "run72_paper_designs_split.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    raw = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    y = r72.bins(np.asarray(dat["istim"], float).ravel(), 8)
    del dat
    X = raw / (raw.std() + 1e-9)
    del raw
    rng = np.random.default_rng(0)
    lda_acc, qda_acc, regs = [], [], []
    for d in range(N_DRAW):
        Xd = X[:, rng.choice(X.shape[1], 256, replace=False)]
        hl = hq = 0
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=d).split(Xd, y):
            sc = StandardScaler().fit(Xd[tr])
            pca = PCA(16, random_state=d).fit(sc.transform(Xd[tr]))
            Ptr, Pte = pca.transform(sc.transform(Xd[tr])), pca.transform(sc.transform(Xd[te]))
            s2 = StandardScaler().fit(Ptr)
            Ptr, Pte = s2.transform(Ptr), s2.transform(Pte)
            hl += int((LinearDiscriminantAnalysis().fit(Ptr, y[tr]).predict(Pte) == y[te]).sum())
            g = GridSearchCV(QuadraticDiscriminantAnalysis(), {"reg_param": [0.01, 0.1, 0.3, 0.6, 0.9]}, cv=3).fit(Ptr, y[tr])
            regs.append(g.best_params_["reg_param"])
            hq += int((g.predict(Pte) == y[te]).sum())
        lda_acc.append(hl / len(y))
        qda_acc.append(hq / len(y))
    row = {"name": fp.stem, "lda": float(np.mean(lda_acc)), "qda": float(np.mean(qda_acc)),
           "gain": float(np.mean(np.array(qda_acc) - np.array(lda_acc))),
           "gain_sd_over_draws": float(np.std(np.array(qda_acc) - np.array(lda_acc))), "chosen_reg": regs}
    print(f"{fp.stem[:30]} R6b lda {row['lda']:.3f} qda {row['qda']:.3f} gain {row['gain']:+.4f} "
          f"(sd {row['gain_sd_over_draws']:.4f}) ({time.time() - t0:.0f}s)", flush=True)
    return row


def r6b():
    from scipy import stats
    files = sorted(RAW.glob("gratings*.npy"))
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(r6b_one, files))
    t9 = {r["name"]: r["orig"] for r in json.load(open(DATA / "cr_withinclass_controls.json"))["rows"]}
    a2 = {r["name"]: r for r in json.load(open(OUT))["rows"]}
    gain = np.array([r["gain"] for r in rows])
    within = np.array([t9[r["name"]]["within"] for r in rows])
    lda = np.array([r["lda"] for r in rows])
    rw = within - np.polyval(np.polyfit(lda, within, 1), lda)
    rg = gain - np.polyval(np.polyfit(lda, gain, 1), lda)
    sc = {"Q8b_gain_vs_within": float(stats.spearmanr(gain, within)[0]),
          "gain_vs_delta": float(stats.spearmanr(gain, [t9[r["name"]]["delta"] for r in rows])[0]),
          "gain_vs_classmean": float(stats.spearmanr(gain, [t9[r["name"]]["class_means"] for r in rows])[0]),
          "gain_vs_NCchange": float(stats.spearmanr(gain, [a2[r["name"]]["noise_corr"]["NCchange"] for r in rows])[0]),
          "partial_gain_within_given_lda": float(np.corrcoef(rg, rw)[0, 1])}
    sc["Q8b_verdict"] = "PASS (COMBINED)" if sc["Q8b_gain_vs_within"] >= 0.5 else "MISS"
    print("SCORE", json.dumps(sc, indent=1), flush=True)
    json.dump({"registration": "H3-R6b (ex0_REGISTERED.md)", "rows": rows, "score": sc},
              open(DATA / "h3_v1_decode_r6b.json", "w"), indent=1)
    print("DONE", flush=True)


def lda_qda(Xd, y, seed):
    """R6b's pipeline on one neuron set: training-fold z-score, 16 PCs, standardized scores, LDA vs QDA (reg by inner CV)."""
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    hl = hq = 0
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xd, y):
        sc = StandardScaler().fit(Xd[tr])
        pca = PCA(16, random_state=seed).fit(sc.transform(Xd[tr]))
        Ptr, Pte = pca.transform(sc.transform(Xd[tr])), pca.transform(sc.transform(Xd[te]))
        s2 = StandardScaler().fit(Ptr)
        Ptr, Pte = s2.transform(Ptr), s2.transform(Pte)
        hl += int((LinearDiscriminantAnalysis().fit(Ptr, y[tr]).predict(Pte) == y[te]).sum())
        g = GridSearchCV(QuadraticDiscriminantAnalysis(), {"reg_param": [0.01, 0.1, 0.3, 0.6, 0.9]}, cv=3).fit(Ptr, y[tr])
        hq += int((g.predict(Pte) == y[te]).sum())
    return hl / len(y), hq / len(y)


def r6c_one(fp):
    """H3-R6c: QDA - LDA gain in the DS vs nonDS third of the same recording (cross-fitted thirds, as R4)."""
    warnings.filterwarnings("ignore")
    r72 = _load("run72", "run72_paper_designs_split.py")
    dsx = _load("dsx", "cr_ds_thirds_crossfit.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    raw = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    y = r72.bins(np.asarray(dat["istim"], float).ravel(), 8)
    del dat
    X = raw / (raw.std() + 1e-9)
    del raw
    half = np.zeros(len(y), int)
    for k in range(8):
        idx = np.where(y == k)[0]
        half[idx[1::2]] = 1
    rng = np.random.default_rng(0)
    row = {"name": fp.stem}
    for sel_h, meas_h in ((0, 1), (1, 0)):
        ds, nds = dsx.select(X[half == sel_h], y[half == sel_h])
        Xm, ym = X[half == meas_h], y[half == meas_h]
        res = {}
        for tag, cols in (("DS", ds), ("nonDS", nds)):
            accs = [lda_qda(Xm[:, rng.choice(cols, 256, replace=False)], ym, d) for d in range(N_DRAW)]
            res[tag] = {"lda": float(np.mean([a[0] for a in accs])), "qda": float(np.mean([a[1] for a in accs])),
                        "gains": [a[1] - a[0] for a in accs]}
        diff = np.array(res["DS"]["gains"]) - np.array(res["nonDS"]["gains"])
        res["gain_diff_mean"] = float(diff.mean())
        res["gain_diff_sd"] = float(diff.std())
        row[f"sel{sel_h}_meas{meas_h}"] = res
        print(f"{fp.stem[:30]} sel{sel_h}: DS lda {res['DS']['lda']:.3f} gain {np.mean(res['DS']['gains']):+.4f} | "
              f"nonDS lda {res['nonDS']['lda']:.3f} gain {np.mean(res['nonDS']['gains']):+.4f} | diff {diff.mean():+.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    return row


def r6c():
    files = sorted(RAW.glob("gratings*.npy"))
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(r6c_one, files))
    th = {r["name"]: r for r in json.load(open(DATA / "cr_ds_thirds_crossfit.json"))["rows"]}
    sc = {}
    for k in ("sel0_meas1", "sel1_meas0"):
        sc[f"Q8c_{k}_gainDS_gt_gainNonDS"] = int(sum(r[k]["gain_diff_mean"] > 0 for r in rows))
        sc[f"{k}_deltaDS_gt_deltaNonDS"] = int(sum(th[r["name"]][k]["DS"] > th[r["name"]][k]["nonDS"] for r in rows))
        sc[f"{k}_ldaDS_gt_ldaNonDS"] = int(sum(r[k]["DS"]["lda"] > r[k]["nonDS"]["lda"] for r in rows))
    sc["Q8c_verdict"] = "PASS" if all(sc[f"Q8c_{k}_gainDS_gt_gainNonDS"] >= 7 for k in ("sel0_meas1", "sel1_meas0")) else "MISS"
    print("SCORE", json.dumps(sc, indent=1), flush=True)
    json.dump({"registration": "H3-R6c (ex0_REGISTERED.md)", "rows": rows, "score": sc},
              open(DATA / "h3_v1_decode_r6c.json", "w"), indent=1)
    print("DONE", flush=True)


def main():
    files = sorted(RAW.glob("gratings*.npy"))
    if len(sys.argv) > 1 and sys.argv[1] == "smoke":
        files = files[:1]
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(run_one, files))
    out = {"registration": "H3-R4..R6 (ex0_REGISTERED.md, e238f3f5)", "rows": rows}
    if len(rows) == 8:
        out["score"] = score(rows)
        print("SCORE", json.dumps(out["score"], indent=1), flush=True)
    json.dump(out, open(OUT if len(rows) == 8 else OUT.with_name("h3_v1_decode_smoke.json"), "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "r6b":
        r6b()
    elif len(sys.argv) > 1 and sys.argv[1] == "r6c":
        r6c()
    else:
        main()
