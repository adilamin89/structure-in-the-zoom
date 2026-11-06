"""H1-R22 (B2), H1-R24 (B4), H1-R25 (B5) on the eight grating recordings (Oct 22). Registrations in ex0_REGISTERED.md.

B2  two-term prediction of delta, nothing fitted. Within part: the mean field ln[8/(1+7 sigma)]/ln 8 (sigma = split-half
    raw sigma of cr_overlap_mining, the paper's Fig 3A). Class-mean part: the full-set term c = log[(1+beta)^2/(1+(b2+2x)/w2)]
    (cr_c8_sign_condition quantities) carried through the ladder. A declared rung of k classes holds a fraction
    u_k = b1_k / b1 of the full set's class-mean variance (b1_k = p-weighted variance of its own class means about their
    mean, from class means only); with the full set's W and the class means' shape fixed, its term is
    c_k = log[(1+u_k beta)^2 / (1+(u_k^2 b2 + 2 u_k x)/w2)]; every floor rung holds the full set's term c (a random subset holds
    every class). The class-mean part is the least-squares slope of c_k - c on the log rung sizes (weights from the rung
    sizes; rung 1: u = 0, top rung: u = 1).
B4  the sigma law held out: sigma from one half of each 7.5-degree sub-bin's presentations (its own split halves), the
    within-class part of the Table 9 ladder on the other half; both ways.
B5  Fig 2C's overlap profile (sixteen 22.5-degree classes, Kn = K_cd / sqrt(K_cc K_dd), cr_failure_interrogation) in three
    frames: as published (reproduction), the correlation frame (each neuron divided by its SD), and with neuron variances
    permuted (neuron i rescaled to the SD of neuron pi(i), seeds 1-3): is the adjacent-minus-antipodal margin dK > 0 kept?
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler cr_h1_b245.py [n_workers]   Out: data_canonical/cr_h1_b245.json"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent / "raw"   # the eight Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
RUNGS = (1, 2, 3, 4, 6, 8)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def full_terms(X, lab, K=8):
    X = X - X.mean(0)
    cm = np.stack([X[lab == c].mean(0) for c in range(K)]); p = np.bincount(lab, minlength=K) / len(lab)
    n = len(X); Wh = (X - cm[lab]) / np.sqrt(n); Bh = np.sqrt(p)[:, None] * cm
    w1, b1 = (Wh * Wh).sum(), (Bh * Bh).sum()
    w2 = ((Wh @ Wh.T) ** 2).sum(); b2 = ((Bh @ Bh.T) ** 2).sum(); x = ((Wh @ Bh.T) ** 2).sum()
    return dict(w1=float(w1), b1=float(b1), w2=float(w2), b2=float(b2), x=float(x), cm=cm, counts=np.bincount(lab, minlength=K))


def classmean_prediction(T, sizes):
    """B2's class-mean part from the full-set terms and the class means alone."""
    cm, cnt = T["cm"], T["counts"]; beta = T["b1"] / T["w1"]
    c_full = np.log((1 + beta) ** 2 / (1 + (T["b2"] + 2 * T["x"]) / T["w2"]))
    ys, us = [], []
    for k in RUNGS:
        p = cnt[:k] / cnt[:k].sum(); mu = (p[:, None] * cm[:k]).sum(0)
        b1k = float((p * ((cm[:k] - mu) ** 2).sum(1)).sum())          # the rung's own B, per presentation
        u = b1k / T["b1"]; us.append(u)
        ck = np.log((1 + u * beta) ** 2 / (1 + (u ** 2 * T["b2"] + 2 * u * T["x"]) / T["w2"]))
        ys.append(ck - c_full)
    return float(np.polyfit(np.log(sizes), ys, 1)[0]), float(c_full), us


def kn_profile(G, lab16, om):
    idx = [np.where(lab16 == c)[0] for c in range(16)]
    tr, _, T = om.overlaps(G, idx); K = om.K_of(tr, tr, T); kd = np.diag(K)
    a = np.arange(16); sep = np.minimum(np.abs(np.subtract.outer(a, a)), 16 - np.abs(np.subtract.outer(a, a)))
    Kn = K / np.sqrt(np.outer(kd, kd))
    prof = [float(Kn[sep == s].mean()) for s in range(9)]
    return prof, prof[1] - prof[8]


def run_one(fp):
    ow = _load("ow", "cr_order_and_withinbin_controls.py"); om = _load("om", "cr_overlap_mining.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab = ow.bins(phi, 8); sub48 = om.bins(phi, 48); lab16 = ow.bins(phi, 16)
    row = {"name": fp.stem}
    # B2
    m8 = [np.where(lab == b)[0] for b in range(8)]
    sizes = [sum(len(m8[j]) for j in range(k)) for k in RUNGS]
    T = full_terms(X.astype(np.float64), lab)
    cmp_, c_full, us = classmean_prediction(T, sizes)
    row["B2"] = {"classmean_pred": cmp_, "c_full": c_full, "u": us, "beta": T["b1"] / T["w1"]}
    # B4
    h = om.halves_by_sub(sub48); b4 = []
    for sh, mh in ((0, 1), (1, 0)):
        S, Mm = h == sh, h == mh
        hh = om.halves_by_sub(sub48[S])
        Ra = om.residuals(X[S][hh == 0], lab[S][hh == 0], sub48[S][hh == 0], "raw")
        Rb = om.residuals(X[S][hh == 1], lab[S][hh == 1], sub48[S][hh == 1], "raw")
        sig = om.sigma(om.split_half_K(Ra, Rb))
        Xm, lm = X[Mm], lab[Mm]; mm = [np.where(lm == b)[0] for b in range(8)]
        lad = ow.ladder((Xm @ Xm.T).astype(np.float64), lm, [np.concatenate(mm[:c]) for c in RUNGS], np.random.default_rng(42))
        b4.append({"sigma": sig, "pred": float(np.log(8 / (1 + 7 * sig)) / np.log(8)), **lad})
    row["B4"] = b4
    # B5
    G = (X @ X.T).astype(np.float64); prof, dK = kn_profile(G, lab16, om); del G
    row["B5"] = {"published": {"profile": prof, "dK": dK}}
    sd = X.std(0).astype(np.float64); ok = sd > 1e-6 * np.median(sd[sd > 0])
    Z = X[:, ok] / sd[ok].astype(np.float32)
    prof, dK = kn_profile((Z @ Z.T).astype(np.float64), lab16, om); row["B5"]["correlation"] = {"profile": prof, "dK": dK}
    perm = []
    for s in (1, 2, 3):
        pi = np.random.default_rng(s).permutation(int(ok.sum()))
        Zp = Z * sd[ok][pi].astype(np.float32)
        prof, dK = kn_profile((Zp @ Zp.T).astype(np.float64), lab16, om); perm.append({"profile": prof, "dK": dK})
    row["B5"]["variance_permuted"] = perm
    print(f"{fp.stem[9:30]:22s} B2 cm-pred {cmp_:+.3f} (c_full {c_full:+.3f}) | B4 " +
          " ".join(f"sig {b['sigma']:.3f} pred {b['pred']:+.3f} within {b['within']:+.3f}" for b in b4) +
          f" | B5 dK pub {row['B5']['published']['dK']:+.3f} corr {row['B5']['correlation']['dK']:+.3f} perm "
          + " ".join(f"{p['dK']:+.3f}" for p in perm) + f" ({time.time() - t0:.0f}s) DONE", flush=True)
    return row


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    files = sorted(RAW.glob("gratings*.npy"))
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(run_one, files))
    t9 = {r["name"]: r["orig"] for r in json.load(open(DATA / "cr_order_and_withinbin_controls.json"))["rows"]}
    omj = {r["name"]: r for r in json.load(open(DATA / "cr_overlap_mining.json"))["rows"]}
    fi = {r["name"]: r for r in json.load(open(DATA / "cr_failure_interrogation.json"))["rows"]}
    for r in rows:
        sig = omj[r["name"]]["a"]["raw"]["sigma"]; wp = float(np.log(8 / (1 + 7 * sig)) / np.log(8))
        r["B2"].update(sigma=sig, within_pred=wp, delta_pred=wp + r["B2"]["classmean_pred"], **{f"t9_{k}": v for k, v in t9[r["name"]].items()})
        r["B2"]["error"] = r["B2"]["delta_pred"] - t9[r["name"]]["delta"]
        r["B2"]["classmean_error"] = r["B2"]["classmean_pred"] - t9[r["name"]]["class_means"]
        r["B5"]["published_reproduces"] = float(abs(r["B5"]["published"]["dK"] - fi[r["name"]]["dK"]))
    from scipy import stats
    hp = [np.mean([b["pred"] for b in r["B4"]]) for r in rows]; hw = [np.mean([b["within"] for b in r["B4"]]) for r in rows]
    v = {"B2_abs_err_le_005": int(sum(abs(r["B2"]["error"]) <= 0.05 for r in rows)),
         "B2_errors": [round(r["B2"]["error"], 3) for r in rows],
         "B2_classmean_errors": [round(r["B2"]["classmean_error"], 3) for r in rows],
         "B2_classmean_r": float(stats.pearsonr([r["B2"]["classmean_pred"] for r in rows], [r["B2"]["t9_class_means"] for r in rows])[0]),
         "B4_heldout_r": float(stats.pearsonr(hp, hw)[0]), "B4_heldout_mae": float(np.mean(np.abs(np.array(hp) - hw))),
         "B5_published_max_absdiff": float(max(r["B5"]["published_reproduces"] for r in rows)),
         "B5_corr_dK_pos": int(sum(r["B5"]["correlation"]["dK"] > 0 for r in rows)),
         "B5_perm_dK_pos_all_seeds": int(sum(all(p["dK"] > 0 for p in r["B5"]["variance_permuted"]) for r in rows)),
         "B5_corr_max_at_adjacent": int(sum(int(np.argmax(r["B5"]["correlation"]["profile"][1:])) == 0 for r in rows))}
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(DATA / "cr_h1_b245.json", "w"), indent=1)
    print("wrote cr_h1_b245.json ALL DONE", flush=True)


if __name__ == "__main__":
    main()
