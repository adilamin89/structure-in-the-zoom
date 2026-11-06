"""Does the second half of the direction ladder gain from odd class MEANS or from odd
FLUCTUATION structure? Eight grating recordings, the paper's input; tuning-free residuals Y (remove_withinbin 8 x 6: leave-one-
out 7.5-degree sub-bin residual plus the 45-degree class mean); antipodal pairs (c, c + 4) of the eight classes.
  orig   Y.
  arm 1  even means: each class keeps its own residual vectors (re-centred), its mean replaced by the pair's common mean
         (odd mean harmonics removed exactly; possible odd covariance structure kept).
  arm 2  even fluctuations: each class keeps its own mean; the residual vectors of the two classes of a pair pooled, permuted
         and re-split at the class counts, re-centred (odd residual-distribution differences removed; seeds 7, 8, 9).
Measured (identical schedules): the Table 9 ladder per rung (declared minus ten-draw floor, rng 42, total and within parts),
its full slope and its LATE-HALF gain = least-squares slope over the rungs holding 4, 6 and 8 classes; and the 8-class
within-class order effect (identity, 8 origins; correlation_vs_variance.order_within).
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler odd_means_vs_fluctuations.py [n_workers]   Out: data_canonical/odd_means_vs_fluctuations.json"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
OUT = HERE.parent / "data_canonical" / "odd_means_vs_fluctuations.json"
RUNGS = (1, 2, 3, 4, 6, 8)
SEEDS = (7, 8, 9)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def ladder_rungs(ow, Y, lab):
    """ow.ladder with its per-rung curves kept: d[k] = (declared - floor) for (total, within) at rung k."""
    G = (Y @ Y.T).astype(np.float64); m = [np.where(lab == b)[0] for b in range(8)]
    sets = [np.concatenate(m[:c]) for c in RUNGS]; rng = np.random.default_rng(42)
    lad, flo = [], []
    for sel in sets:
        lad.append(ow.parts_from_gram(G[np.ix_(sel, sel)], lab[sel]))
        acc = np.zeros(2)
        for _ in range(ow.N_NULL):
            idx = rng.choice(len(G), len(sel), replace=False)
            acc += np.array(ow.parts_from_gram(G[np.ix_(idx, idx)], lab[idx])) / ow.N_NULL
        flo.append(acc)
    x = np.log([len(s) for s in sets]); d = np.array(lad) - np.array(flo); late = slice(3, 6)
    out = {"x": x.tolist(), "d_total": d[:, 0].tolist(), "d_within": d[:, 1].tolist(),
           "delta": float(np.polyfit(x, d[:, 0], 1)[0]), "within": float(np.polyfit(x, d[:, 1], 1)[0]),
           "late_total": float(np.polyfit(x[late], d[late, 0], 1)[0]), "late_within": float(np.polyfit(x[late], d[late, 1], 1)[0])}
    out["class_means"] = out["delta"] - out["within"]
    return out


def arm_even_means(Y, lab):
    Z = Y.copy()
    mu = {c: Y[lab == c].mean(0) for c in range(8)}
    for c in range(8):
        m = lab == c; Z[m] = Y[m] - mu[c] + 0.5 * (mu[c] + mu[(c + 4) % 8])
    return Z


def arm_even_fluct(Y, lab, rng):
    Z = Y.copy()
    for c in range(4):
        a, b = np.where(lab == c)[0], np.where(lab == c + 4)[0]
        Ra, Rb = Y[a] - Y[a].mean(0), Y[b] - Y[b].mean(0)
        P = np.concatenate([Ra, Rb])[rng.permutation(len(a) + len(b))]
        Pa, Pb = P[:len(a)], P[len(a):]
        Z[a] = Pa - Pa.mean(0) + Y[a].mean(0); Z[b] = Pb - Pb.mean(0) + Y[b].mean(0)
    return Z


def run_one(fp):
    ow = _load("ow", "cr_order_and_withinbin_controls.py"); om = _load("om", "overlap_identity_checks.py")
    b7 = _load("b7", "correlation_vs_variance.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab = ow.bins(phi, 8); Y = ow.remove_withinbin(X, lab, phi, 6, 8)
    row = {"name": fp.stem}
    for name, Z in (("orig", Y), ("even_means", arm_even_means(Y, lab))):
        row[name] = {**ladder_rungs(ow, Z, lab), "order_within": b7.order_within(ow, om, Z, lab)}
    ef = []
    for s in SEEDS:
        Z = arm_even_fluct(Y, lab, np.random.default_rng(s))
        ef.append({**ladder_rungs(ow, Z, lab), "order_within": b7.order_within(ow, om, Z, lab)})
    row["even_fluct"] = {"seeds": ef, **{k: float(np.mean([e[k] for e in ef])) for k in ("delta", "within", "class_means", "late_total", "late_within", "order_within")}}
    o, a1, a2 = row["orig"], row["even_means"], row["even_fluct"]
    print(f"{fp.stem[9:30]:22s} late gain orig {o['late_total']:+.3f} | even means {a1['late_total']:+.3f} | even fluct {a2['late_total']:+.3f} "
          f"| order within {o['order_within']:+.3f} / {a1['order_within']:+.3f} / {a2['order_within']:+.3f} | delta {o['delta']:+.3f} / "
          f"{a1['delta']:+.3f} / {a2['delta']:+.3f} ({time.time() - t0:.0f}s) DONE", flush=True)
    return row


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(run_one, sorted(RAW.glob("gratings*.npy"))))
    keep = lambda r, a: r[a]["late_total"] / r["orig"]["late_total"]
    v = {"even_means_keeps_ge_half_late": int(sum(keep(r, "even_means") >= 0.5 for r in rows)),
         "even_fluct_loses_ge_half_late": int(sum(keep(r, "even_fluct") <= 0.5 for r in rows)),
         "late_kept_even_means": [round(keep(r, "even_means"), 2) for r in rows],
         "late_kept_even_fluct": [round(keep(r, "even_fluct"), 2) for r in rows],
         "order_within_orig_meanseven_flucteven": [[round(r[a]["order_within"], 3) for a in ("orig", "even_means", "even_fluct")] for r in rows]}
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
