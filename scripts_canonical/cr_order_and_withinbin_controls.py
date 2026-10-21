"""Referee controls (ChatGPT referee, Oct 20, comments 3 and 4) on the eight grating recordings, one fast engine.

Engine: the Gram matrix G = X X^T of a recording (presentations x neurons, global SD scaling as run72) is computed once;
every subset's centred PR and its exact W + B split (run72 / App K: within-class part log(w1^2/w2), class-mean part the
remainder) are read from G restricted to the subset, with class-mean projections done on G. Floors: ten random subsets
per rung drawn exactly as run72.ladder (same rng sequence), so the baseline reproduces Table 9's direction columns.

(a) CLASS ORDER. The angular order of the eight 45-degree classes against 40 random orders of the whole classes (same
    rung schedule 1,2,3,4,6,8 classes; each order its own floor): does the circular adjacency matter, beyond class
    heterogeneity? Statistic: the angular delta's rank among the random orders.
(b) WITHIN-BIN TUNING. Directions are continuous, so a 45-degree class mixes directions and W holds residual tuning. Each
    presentation is replaced by its residual from the held-out (leave-one-out) mean of its fine sub-bin, plus its class
    mean: class means kept, within-bin tuning removed. Sub-bins of 7.5 and 3.75 degrees. Re-measured: the direction
    ladder (delta, within, class-mean parts) and the sixteen-class order effect (paired - sequential, run72 design B,
    sub-bins 3.75 degrees inside each 22.5-degree class).

PREDICTIONS (written before the run):
  a: the angular order's delta is above the median of the random orders in >= 7 of 8 (adjacent classes share fluctuation
     directions, Sec 4); class heterogeneity alone keeps the random orders' deltas positive.
  b: with within-bin tuning removed, delta stays positive in 8 of 8 and the within-class part positive in 8 of 8; the
     order effect's within-class part stays negative in >= 7 of 8.
Out: data_canonical/cr_order_and_withinbin_controls.json"""
import json
import multiprocessing as mp
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_order_and_withinbin_controls.json"
N_NULL, N_ORDERS = 10, 40


def bins(phi, nb):
    return np.clip(np.digitize(phi % (2 * np.pi), np.linspace(0, 2 * np.pi, nb + 1)) - 1, 0, nb - 1)


def parts_from_gram(Gs, lab):
    """(log PR total, log PR within) of the centred subset whose Gram block is Gs (k x k), labels lab."""
    k = len(Gs)
    r = Gs.mean(1, keepdims=True); Gc = Gs - r - r.T + Gs.mean()
    tot = np.trace(Gc) ** 2 / (Gc * Gc).sum()
    _, inv = np.unique(lab, return_inverse=True)
    A = np.zeros((k, inv.max() + 1)); A[np.arange(k), inv] = 1.0
    A /= A.sum(0)                                                     # column c averages class c
    PG = A.T @ Gs                                                     # class means of rows: C x k
    M = PG[inv]                                                       # row i -> its class-mean row
    PGP = (PG @ A)[inv][:, inv]                                       # class-mean x class-mean block
    GR = Gs - M - M.T + PGP                                           # Gram of class residuals
    wit = np.trace(GR) ** 2 / (GR * GR).sum()
    return np.log(tot), np.log(wit)


def ladder(G, lab, rung_sets, rng):
    sizes, lad, flo = [], [], []
    for sel in rung_sets:
        sizes.append(len(sel)); lad.append(parts_from_gram(G[np.ix_(sel, sel)], lab[sel]))
        acc = np.zeros(2)
        for _ in range(N_NULL):
            idx = rng.choice(len(G), len(sel), replace=False)
            acc += np.array(parts_from_gram(G[np.ix_(idx, idx)], lab[idx])) / N_NULL
        flo.append(acc)
    x = np.log(sizes); d = np.array(lad) - np.array(flo)
    s_tot = float(np.polyfit(x, d[:, 0], 1)[0]); s_w = float(np.polyfit(x, d[:, 1], 1)[0])
    return {"delta": s_tot, "within": s_w, "class_means": s_tot - s_w}


def remove_withinbin(X, lab, phi, n_sub_per_class, n_class):
    sub = bins(phi, n_class * n_sub_per_class)
    Y = X.copy()
    for b in np.unique(sub):
        m = sub == b; nb = m.sum()
        if nb < 2:
            continue
        S = X[m].sum(0)
        Y[m] = X[m] - (S - X[m]) / (nb - 1)                           # residual from the leave-one-out sub-bin mean
    for c in np.unique(lab):
        m = lab == c; Y[m] += X[m].mean(0)                             # add the class mean back
    return Y


def designs(lab8, lab16):
    m8 = [np.where(lab8 == b)[0] for b in range(8)]
    m16 = [np.where(lab16 == b)[0] for b in range(16)]
    setsA = [np.concatenate(m8[:c]) for c in (1, 2, 3, 4, 6, 8)]
    binsP = [[j for k in range(r + 1) for j in (k, k + 8)] for r in range(8)]
    binsS = [[j for k in range(r + 1) for j in (2 * k, 2 * k + 1)] for r in range(8)]
    P = [np.concatenate([m16[j] for j in s]) for s in binsP]; S = [np.concatenate([m16[j] for j in s]) for s in binsS]
    return m8, setsA, P, S


def run_one(fp):
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab8, lab16 = bins(phi, 8), bins(phi, 16)
    m8, setsA, P, S = designs(lab8, lab16)
    G = (X @ X.T).astype(np.float64)
    row = {"name": fp.stem, "orig": ladder(G, lab8, setsA, np.random.default_rng(42))}
    po = ladder(G, lab16, P, np.random.default_rng(42)); so = ladder(G, lab16, S, np.random.default_rng(43))
    row["orig_order_effect"] = {k: po[k] - so[k] for k in po}
    for nsub in (6, 12):
        Y = remove_withinbin(X, lab8, phi, nsub, 8); GY = (Y @ Y.T).astype(np.float64)
        row[f"withinbin_8x{nsub}"] = ladder(GY, lab8, setsA, np.random.default_rng(42))
    Y = remove_withinbin(X, lab16, phi, 6, 16); GY = (Y @ Y.T).astype(np.float64)
    po = ladder(GY, lab16, P, np.random.default_rng(42)); so = ladder(GY, lab16, S, np.random.default_rng(43))
    row["withinbin_order_effect_16x6"] = {k: po[k] - so[k] for k in po}
    print(f"{fp.stem[:30]:30s} WITHIN-BIN orig d {row['orig']['delta']:+.3f} w {row['orig']['within']:+.3f} -> 7.5deg d {row['withinbin_8x6']['delta']:+.3f} w {row['withinbin_8x6']['within']:+.3f}; 3.75deg d {row['withinbin_8x12']['delta']:+.3f} | order-effect within {row['orig_order_effect']['within']:+.3f} -> {row['withinbin_order_effect_16x6']['within']:+.3f} ({time.time() - t0:.0f}s)", flush=True)
    rng = np.random.default_rng(1)
    rand = []
    for _ in range(N_ORDERS):
        perm = rng.permutation(8)
        sets = [np.concatenate([m8[j] for j in perm[:c]]) for c in (1, 2, 3, 4, 6, 8)]
        rand.append(ladder(G, lab8, sets, np.random.default_rng(42)))
    rd = np.array([r["delta"] for r in rand])
    row["class_order"] = {"random_delta_mean": float(rd.mean()), "random_delta_sd": float(rd.std()),
                          "random_delta_min": float(rd.min()), "random_delta_max": float(rd.max()),
                          "angular_rank_frac": float((rd < row["orig"]["delta"]).mean()),
                          "random_within_mean": float(np.mean([r["within"] for r in rand]))}
    o, c = row["orig"], row["class_order"]; w6 = row["withinbin_8x6"]; oe = row["withinbin_order_effect_16x6"]
    print(f"{fp.stem[:30]:30s} orig d {o['delta']:+.3f} w {o['within']:+.3f} | random orders d {c['random_delta_mean']:+.3f}"
          f"+-{c['random_delta_sd']:.3f} (angular above {100 * c['angular_rank_frac']:.0f}%) | within-bin removed d {w6['delta']:+.3f}"
          f" w {w6['within']:+.3f} | order effect w {row['orig_order_effect']['within']:+.3f} -> {oe['within']:+.3f}"
          f" ({time.time() - t0:.0f}s)", flush=True)
    return row


def main():
    files = sorted(RAW.glob("gratings*.npy"))
    with mp.get_context("fork").Pool(4) as pool:
        rows = pool.map(run_one, files)
    v = {"a_angular_above_median": int(sum(r["class_order"]["angular_rank_frac"] > 0.5 for r in rows)),
         "b_delta_positive": int(sum(r["withinbin_8x6"]["delta"] > 0 for r in rows)),
         "b_delta_positive_12": int(sum(r["withinbin_8x12"]["delta"] > 0 for r in rows)),
         "b_within_positive": int(sum(r["withinbin_8x6"]["within"] > 0 for r in rows)),
         "b_order_within_negative": int(sum(r["withinbin_order_effect_16x6"]["within"] < 0 for r in rows))}
    print("VERDICT", json.dumps(v), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
