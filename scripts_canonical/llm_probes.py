"""The shift beside and with linear probes on the same language-model states.

States: the battery's cached last-token states (raw/llm_states/<model>__<battery>.npz, byte-identical prompts). The shift
numbers come from the battery's artifact (llm_battery_<model>.json) or, for the projected arm, from llm_battery's own
estimator functions (gram 'linear', ladder, floor, order averages, seeds) applied to a modified Gram.

  R1 (content / construction): per layer, 5-fold stratified CV x 5 repeats of StandardScaler + multinomial logistic
     (C by inner 3-fold over 10^-3..10^2), a Hewitt-Liang control task (labels permuted once per repeat), a learning
     curve at 2/4/8/12 training prompts per class (C = 1, 20 draws), and Spearman across layers of the 2-per-class
     accuracy with the shift's excess over its null (dbar50 - perm null mean, the battery's artifact).
  R2 (BLiMP design B): 8-class, phenomenon and grammaticality probes, random 5-fold vs pair-held-out GroupKFold, and the
     minimal-pair accuracy (held-out pairs whose good sentence scores higher).
  R3 (COMBINED): the order-averaged shift after projecting out the 7-dim class-mean subspace, against 200 label
     permutations that each project out their own permuted-label subspace; cross-fitted probe accuracy after projection.

Usage: python -X faulthandler llm_probes.py <model_key> <battery>  -> data_canonical/llm_probes_<model_key>__<battery>.json
"""
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("h2b", HERE / "llm_battery.py")
h2b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h2b)
OUTDIR = h2b.OUTDIR
CS = np.logspace(-3, 2, 6)
N_REP = 5
LC_SIZES = (2, 4, 8, 12)
LC_DRAWS = 20


def span_fit(Xtr, Xte):
    """Standardize on the training rows, then express both sets in the training rows' span. An L2-penalized linear
    model's weights lie in that span (w = Xtr^T a; the intercept is unpenalized), and the map is an isometry on it, so
    fits and predictions equal those on the full features, at n_train instead of d dimensions."""
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[s > s[0] * 1e-10].T
    return A @ V, B @ V


def probe_cv(X, y, splitter, groups=None, cv_c=True, seed=0):
    """Held-out predictions (labels and decision scores) for one split."""
    pred = np.empty_like(y)
    score = None
    for tr, te in splitter.split(X, y, groups):
        if cv_c:
            clf = LogisticRegressionCV(Cs=CS, cv=3, max_iter=3000)
        else:
            clf = LogisticRegression(C=1.0, max_iter=3000)
        Ztr, Zte = span_fit(X[tr], X[te])
        m = clf.fit(Ztr, y[tr])
        pred[te] = m.predict(Zte)
        d = m.decision_function(Zte)
        if score is None:
            score = np.zeros((len(y),) + d.shape[1:])
        score[te] = d
    return pred, score


def learning_curve(X, y, rng):
    nc = y.max() + 1
    out = {}
    for m in LC_SIZES:
        accs = []
        for _ in range(LC_DRAWS):
            tr = np.concatenate([rng.choice(np.where(y == c)[0], m, replace=False) for c in range(nc)])
            te = np.setdiff1d(np.arange(len(y)), tr)
            Ztr, Zte = span_fit(X[tr], X[te])
            clf = LogisticRegression(C=1.0, max_iter=3000).fit(Ztr, y[tr])
            accs.append(float((clf.predict(Zte) == y[te]).mean()))
        out[str(m)] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs))}
    return out


def r1_layer(X, y, l):
    accs, ctrl = [], []
    for rep in range(N_REP):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        p, _ = probe_cv(X, y, skf)
        accs.append(float((p == y).mean()))
        yc = np.random.default_rng(1000 + rep).permutation(y)
        pc, _ = probe_cv(X, yc, skf)
        ctrl.append(float((pc == yc).mean()))
    return {"layer": l, "acc": float(np.mean(accs)), "acc_sd": float(np.std(accs)), "control": float(np.mean(ctrl)),
            "selectivity": float(np.mean(accs) - np.mean(ctrl)),
            "learning_curve": learning_curve(X, y, np.random.default_rng(2000 + l))}


def r2_layer(X, y, l):
    pairs = np.arange(len(y)) // 2
    assert np.all(y[0::2] % 2 == 0) and np.all(y[1::2] == y[0::2] + 1), "BLiMP rows must be (good, bad) pairs"
    targets = {"class8": y, "phenomenon": y // 2, "grammatical": (y % 2 == 0).astype(int)}
    row = {"layer": l}
    for split in ("random", "pair_held_out"):
        res = {}
        for name, t in targets.items():
            accs, mp = [], []
            for rep in range(N_REP):
                if split == "random":
                    sp, g = StratifiedKFold(5, shuffle=True, random_state=rep), None
                else:
                    perm = np.random.default_rng(rep).permutation(pairs.max() + 1)
                    sp, g = GroupKFold(5), perm[pairs]
                p, s = probe_cv(X, t, sp, g)
                accs.append(float((p == t).mean()))
                if name == "grammatical" and split == "pair_held_out":
                    # decision_function > 0 favours class 1 = grammatical; a pair is correct when good scores higher
                    mp.append(float((s[0::2] > s[1::2]).mean() + 0.5 * (s[0::2] == s[1::2]).mean()))
            res[name] = {"acc": float(np.mean(accs)), "acc_sd": float(np.std(accs))}
            if mp:
                res[name]["minimal_pair_acc"] = float(np.mean(mp))
        row[split] = res
    return row


def mean_subspace(X, y):
    nc = y.max() + 1
    M = np.stack([X[y == c].mean(0) for c in range(nc)])
    M = M - M.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[: nc - 1].T                                    # features x (nc - 1), orthonormal


def projected_gram(X, y):
    Xs = X.astype(np.float64)
    Xs = Xs / (Xs.std() + 1e-9)                              # h2b.gram('linear') scaling
    Q = mean_subspace(Xs, y)
    XQ = Xs @ Q
    return Xs @ Xs.T - XQ @ XQ.T


def r3_layer(X, y, l, counts, orders200, null_draws):
    nc = y.max() + 1
    n = len(y)
    K = projected_gram(X, y)
    mem = h2b.members_of(y, nc)
    th_obs, sizes = h2b.ladder(K, mem, list(range(nc)), counts)
    th_f = h2b.floor(K, n, sizes, l * 100 + 1)
    dbar = h2b.order_avg(K, mem, orders200, counts, th_f)
    vals = []
    for lab, ords in null_draws:
        Kn = projected_gram(X, lab)                           # each null draw removes its own label subspace
        th_fn = h2b.floor(Kn, n, sizes, l * 100 + 1)
        vals.append(h2b.order_avg(Kn, h2b.members_of(lab, nc), ords, counts, th_fn))
    vals = np.array(vals)
    # cross-fitted sanity: remove the training folds' class-mean subspace, then probe
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        Xs = X.astype(np.float64)
        Q = mean_subspace(Xs[tr], y[tr])
        Xp = Xs - (Xs @ Q) @ Q.T
        Ztr, Zte = span_fit(Xp[tr], Xp[te])
        m = LogisticRegression(C=1.0, max_iter=3000).fit(Ztr, y[tr])
        accs.append(float((m.predict(Zte) == y[te]).mean()))
    return {"layer": l, "delta_declared_proj": float(th_obs - th_f), "floor_proj": float(th_f),
            "dbar200_proj": float(dbar), "null_mean": float(vals.mean()), "null_sd": float(vals.std()),
            "p_one": float((1 + (vals >= dbar).sum()) / (len(vals) + 1)),
            "z": float((dbar - vals.mean()) / vals.std()) if vals.std() > 0 else 0.0,
            "probe_acc_after_projection": float(np.mean(accs))}


def main(mkey, battery):
    z = np.load(h2b.STATES / f"{mkey}__{battery}.npz")
    H, y = z["H"], z["labels"].astype(int)
    prompts, labels, names, counts = h2b.battery_prompts(battery)
    assert np.array_equal(labels, y) and list(z["prompts"]) == prompts
    art = json.load(open(OUTDIR / f"llm_battery_{mkey}.json"))["batteries"][battery]
    out_path = OUTDIR / f"llm_probes_{mkey}__{battery}.json"
    out = {"model_key": mkey, "battery": battery, "n_prompts": int(len(y)), "dtype": str(z["dtype"]),
           "layers": []}
    nc = y.max() + 1
    orng = np.random.default_rng(3801)
    orders200 = [orng.permutation(nc).tolist() for _ in range(200)]
    rng = np.random.default_rng(4800)                         # the battery's perm-null seed, same draws
    null_draws = [(h2b.null_labels("perm", y, rng), [rng.permutation(nc).tolist() for _ in range(50)])
                  for _ in range(200)]
    for l in range(H.shape[0]):
        t0 = time.time()
        X = H[l].astype(np.float64)
        row = r2_layer(X, y, l) if battery == "blimp_B" else r1_layer(X, y, l)
        a = art["layers"][l]
        row["shift"] = {"dbar50": a["dbar50_run37"], "dbar200": a["dbar200"], "perm_null_mean": a["perm"]["null_mean"],
                        "perm_null_sd": a["perm"]["null_sd"], "p_one": a["perm"]["p_one"],
                        "excess": a["dbar50_run37"] - a["perm"]["null_mean"], "z": a["perm"]["z"]}
        if battery != "blimp_B":
            row["projected"] = r3_layer(X, y, l, counts, orders200, null_draws)
        out["layers"].append(row)
        json.dump(out, open(out_path, "w"), indent=1)
        msg = (f"L{l:02d} acc {row['acc']:.3f} ctrl {row['control']:.3f} lc2 {row['learning_curve']['2']['mean']:.3f} "
               f"| shift excess {row['shift']['excess']:+.3f} | proj dbar {row['projected']['dbar200_proj']:+.3f} "
               f"null {row['projected']['null_mean']:+.3f} p {row['projected']['p_one']:.3f} "
               f"probe-after-proj {row['projected']['probe_acc_after_projection']:.3f}"
               if battery != "blimp_B" else
               f"L{l:02d} " + " ".join(f"{s}: " + " ".join(f"{k} {v['acc']:.3f}" for k, v in row[s].items())
                                       for s in ("random", "pair_held_out"))
               + f" minpair {row['pair_held_out']['grammatical']['minimal_pair_acc']:.3f} "
               f"| shift excess {row['shift']['excess']:+.3f}")
        print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
    if battery != "blimp_B":
        L = out["layers"]
        for key in ("2", "12"):
            r, p = stats.spearmanr([x["learning_curve"][key]["mean"] for x in L], [x["shift"]["excess"] for x in L])
            out[f"spearman_lc{key}_vs_excess"] = {"rho": float(r), "n_layers": len(L)}
        r, _ = stats.spearmanr([x["acc"] for x in L], [x["shift"]["excess"] for x in L])
        out["spearman_acc_vs_excess"] = {"rho": float(r), "n_layers": len(L)}
        json.dump(out, open(out_path, "w"), indent=1)
        print("SPEARMAN", {k: v for k, v in out.items() if k.startswith("spearman")}, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
