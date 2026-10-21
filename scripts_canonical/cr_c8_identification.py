"""C8 controls (close-read feedback, Oct 21): what the within-class shuffle identifies, and whether the random-axis
calibration is exchangeable as aggregated.

PART A, eight grating recordings, engine of cr_order_and_withinbin_controls (Gram trace, exact W + B split, ten-draw floor,
rng 42; direction ladder 1,2,3,4,6,8 classes). Inputs:
  orig        the paper's input; reproduces cr_order_and_withinbin_controls 'orig'.
  resid       tuning inside each 45-degree class removed first: each presentation replaced by its residual from the
              leave-one-out mean of its 7.5-degree sub-bin, plus its class mean (reproduces 'withinbin_8x6').
  resid_shuf  resid, then each neuron's residuals permuted within its class, independently per neuron (seeds 7, 8, 9):
              class means and per-neuron per-class variances kept, every correlation between neurons removed.
  resid_pool  resid, then the residual VECTORS (all neurons intact) permuted across all presentations and re-centred in
              each class (seeds 7, 8, 9): class means kept, correlations between neurons kept, but every class now draws
              from the same pooled residual distribution, so class dependence of the fluctuation geometry is removed.
PREDICTIONS (written before the run), if the shift is carried by class-dependent correlated fluctuations:
  A1 resid_shuf: delta below resid in 8 of 8 and negative in >= 6 of 8 (the raw-shuffle reversal survives on tuning-free
     residuals).
  A2 resid_pool: within-class part within 0.05 of zero in 8 of 8, delta below resid in 8 of 8 and negative in >= 6 of 8.
  If A1 fails (resid_shuf keeps delta positive in >= 3), the raw shuffle's reversal rode on within-bin tuning covariance and
  Sec 6 / the abstract must say so.

PART B, all 18 recordings, random-stimulus ladder of random_zoom_all_recordings.py (Table 2), same draws (observed arm
rng 42+s for five draws per rung; floor rng 1000+i, ten ladders), PR by the trace formula on the centred Gram block:
  ORIENTATION: two spontaneous files (MP033, MP034) have more neurons than frames; the published loader then kept them
  neurons x frames, so their "random stimulus" ladder subsampled neurons. Both orientations are run; the corrected one
  (frames x neurons) replaces them in the final 18.
  paper    observed rung = log of the mean PR of five draws (as published); floor = mean over ten single-draw ladders.
  matched  observed rung = mean of log PR of the same five draws (aggregation matched to the floor).
PREDICTION B1: 'paper' reproduces Table 2's per-recording delta within 0.002; |matched - paper| <= 0.005 in every recording
and mean matched delta within 0.005 of zero.
Out: data_canonical/cr_c8_identification.json"""
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np
from scipy.io import loadmat

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "raw"   # Stringer et al. 2021 (Figshare 8279387) .npy; spont/ = Stringer et al. 2019 (Science)
OUT = HERE.parent / "data_canonical" / "cr_c8_identification.json"
TABLE2 = HERE.parent / "data_canonical" / "stringer_theta_decomposition_pre_c8.json"   # the published Table 2
spec = importlib.util.spec_from_file_location("ow", HERE / "cr_order_and_withinbin_controls.py")
ow = importlib.util.module_from_spec(spec); spec.loader.exec_module(ow)
ZOOM_SIZES = [2000, 1000, 500, 200, 100, 50]
SEEDS = (7, 8, 9)


def shuffle_within_class(X, lab, rng):
    Y = X.copy()
    for c in np.unique(lab):
        idx = np.where(lab == c)[0]
        perm = np.argsort(rng.random((len(idx), X.shape[1])), axis=0)       # independent permutation per neuron
        Y[idx] = np.take_along_axis(X[idx], perm, axis=0)
    return Y


def pool_residuals(Y, lab, rng):
    cm = {c: Y[lab == c].mean(0) for c in np.unique(lab)}
    R = Y - np.stack([cm[c] for c in lab])
    R = R[rng.permutation(len(R))]                                          # whole residual vectors, neurons intact
    for c in cm:
        m = lab == c; R[m] -= R[m].mean(0); R[m] += cm[c]                   # re-centre in each class, add its mean
    return R


def part_a(fp):
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab = ow.bins(phi, 8)
    m8 = [np.where(lab == b)[0] for b in range(8)]
    sets = [np.concatenate(m8[:c]) for c in (1, 2, 3, 4, 6, 8)]
    lad = lambda Z: ow.ladder((Z @ Z.T).astype(np.float64), lab, sets, np.random.default_rng(42))
    Y = ow.remove_withinbin(X, lab, phi, 6, 8)
    row = {"name": fp.stem, "orig": lad(X), "resid": lad(Y),
           "resid_shuf": [lad(shuffle_within_class(Y, lab, np.random.default_rng(s))) for s in SEEDS],
           "resid_pool": [lad(pool_residuals(Y, lab, np.random.default_rng(s))) for s in SEEDS]}
    f = lambda k: (np.mean([r[k] for r in row["resid_shuf"]]), np.mean([r[k] for r in row["resid_pool"]]))
    print(f"A {fp.stem[:28]:28s} orig d {row['orig']['delta']:+.3f} | resid d {row['resid']['delta']:+.3f} "
          f"w {row['resid']['within']:+.3f} | shuf d {f('delta')[0]:+.3f} w {f('within')[0]:+.3f} | "
          f"pool d {f('delta')[1]:+.3f} w {f('within')[1]:+.3f} ({time.time() - t0:.0f}s)", flush=True)
    return row


def pr_sub(G, idx):
    Gs = G[np.ix_(idx, idx)]
    r = Gs.mean(1, keepdims=True); Gc = Gs - r - r.T + Gs.mean()
    return float(np.trace(Gc) ** 2 / (Gc * Gc).sum())


def slope(ns, ys):
    return float(np.polyfit(np.log(ns), ys, 1)[0])


def part_b(item):
    name, fp, corrected = item
    t0 = time.time()
    if fp.suffix == ".npy":
        X = np.asarray(np.load(fp, allow_pickle=True).item()["sresp"], np.float64).T          # presentations x neurons
    else:
        F = loadmat(fp)["Fsp"]                                                              # neurons x frames
        X = F.T if (corrected or F.shape[0] < F.shape[1]) else F       # published loader kept F when neurons > frames
        if X.shape[0] > 5000:
            X = X[np.sort(np.random.default_rng(42).choice(X.shape[0], size=5000, replace=False))]
        X = np.asarray(X, np.float64)
        if not corrected and X.shape[0] > X.shape[1]:
            X = X.T                                                                         # published decompose() swap
    n = X.shape[0]
    G = X @ X.T
    ns = [n] + [s for s in ZOOM_SIZES if s < n]
    full = np.log(pr_sub(G, np.arange(n))); obs_paper, obs_matched = [full], [full]
    for s in ns[1:]:
        prs = [pr_sub(G, np.random.default_rng(42 + k).choice(n, size=s, replace=False)) for k in range(5)]
        obs_paper.append(np.log(np.mean(prs))); obs_matched.append(np.mean(np.log(prs)))
    floors, flog = [], []
    for i in range(10):
        rng = np.random.default_rng(1000 + i)
        flog.append([np.log(pr_sub(G, rng.choice(n, size=s, replace=False))) for s in ns])
        floors.append(slope(ns, flog[-1]))
    fl = float(np.mean(floors))
    row = {"name": name, "corrected_orientation": corrected, "n_samples": n, "n_features": X.shape[1], "rungs": ns,
           "log_pr_obs_paper": [float(v) for v in obs_paper], "log_pr_obs_matched": [float(v) for v in obs_matched],
           "log_pr_floor_mean": [float(v) for v in np.mean(flog, 0)], "pr_full": float(np.exp(full)), "theta_obs_paper": slope(ns, obs_paper), "theta_obs_matched": slope(ns, obs_matched),
           "floor": fl, "floor_sd": float(np.std(floors)),
           "floor_analytic": slope(ns, [np.log((s - 1) * np.exp(full) / ((s - 1) + np.exp(full))) for s in ns])}
    row["delta_paper"] = row["theta_obs_paper"] - fl; row["delta_matched"] = row["theta_obs_matched"] - fl
    print(f"B {name[:30]:30s} delta paper {row['delta_paper']:+.4f} matched {row['delta_matched']:+.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return row


def main():
    grat = sorted(DATA.glob("gratings*.npy"))
    items = [(fp.stem, fp, False) for fp in sorted(list(DATA.glob("gratings*.npy")) + list(DATA.glob("static*.npy")))]
    for fp in sorted((DATA / "spont").glob("spont_*.mat")):
        items.append((f"spont_{fp.stem}", fp, False))
        if loadmat(fp, variable_names=["Fsp"])["Fsp"].shape[0] > loadmat(fp, variable_names=["Fsp"])["Fsp"].shape[1]:
            items.append((f"spont_{fp.stem}", fp, True))                                     # neurons > frames: fix
    with ProcessPoolExecutor(3, mp_context=mp.get_context("spawn")) as ex:
        rows_b = list(ex.map(part_b, items))
        rows_a = list(ex.map(part_a, grat))
    t2 = {r["name"]: r["shift_obs"] for r in json.load(open(TABLE2))["rows"] if r.get("status") == "ok"}
    for r in rows_b:
        r["delta_table2"] = t2.get(r["name"]) if not r["corrected_orientation"] else None
    pub = [r for r in rows_b if not r["corrected_orientation"]]
    fixed = {r["name"]: r for r in rows_b if r["corrected_orientation"]}
    final = [fixed.get(r["name"], r) for r in pub]                                          # the 18 as they should be
    mean = lambda rs, k: float(np.mean([r[k] for r in rs]))
    sh = [(r["resid"]["delta"], np.mean([q["delta"] for q in r["resid_shuf"]])) for r in rows_a]
    po = [(r["resid"]["delta"], np.mean([q["delta"] for q in r["resid_pool"]]), np.mean([q["within"] for q in r["resid_pool"]]))
          for r in rows_a]
    v = {"A1_shuf_below_resid": int(sum(s < d for d, s in sh)), "A1_shuf_negative": int(sum(s < 0 for _, s in sh)),
         "A2_pool_within_abs_le_0.05": int(sum(abs(w) <= 0.05 for _, _, w in po)),
         "A2_pool_below_resid": int(sum(p < d for d, p, _ in po)), "A2_pool_negative": int(sum(p < 0 for _, p, _ in po)),
         "B1_max_abs_paper_minus_table2": float(max(abs(r["delta_paper"] - r["delta_table2"]) for r in pub)),
         "B1_max_abs_matched_minus_paper": float(max(abs(r["delta_matched"] - r["delta_paper"]) for r in final)),
         "B_final18_mean_delta_paper": mean(final, "delta_paper"), "B_final18_mean_delta_matched": mean(final, "delta_matched"),
         "B_final18_max_abs_delta_matched": float(max(abs(r["delta_matched"]) for r in final)),
         "B_final18_max_abs_delta_paper": float(max(abs(r["delta_paper"]) for r in final))}
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump({"part_a": rows_a, "part_b": rows_b, "part_b_final18": [r["name"] + (" (corrected)" if r["corrected_orientation"] else "") for r in final], "verdict": v}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
