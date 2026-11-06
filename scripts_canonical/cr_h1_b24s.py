"""H1-R27 (B2/B4 sharpened, referee items 2 + 9, Oct 22) and H1-R28 (sign-law counterexample search, referee item 9).

R27 prospective prediction. Halves are temporal (epochs of 48 consecutive presentations; A = even, B = odd). From half A only,
frozen: sigma (split halves inside A, alternate presentations of each 7.5-degree sub-bin, raw residuals; cr_overlap_mining),
the full-set terms (beta, b2, x, w2) and the class means. Predicted for half B at B's class sizes and rung schedule:
  within part   (secant) ln[8/(1+7 sigma)]/ln 8, and (multi-rung) the least-squares slope over B's log rung sizes of
                log[k/(1 + (k-1) sigma)] for the rungs' k = 1,2,3,4,6,8 classes (the mean-field declared curve; the floor holds
                all eight classes at every rung, a constant);
  class-mean    cr_h1_b245.classmean_prediction (A's terms, B's rung sizes);
  delta         the sum.
Measured: Table 9 ladder on half B. Both ways. Baseline: for each recording and half, the mean measured value of the OTHER
mice's recordings of the same protocol on the same half (low-contrast GT1 <-> GT2, one each). Reported: absolute errors,
calibration slope and intercept (measured on predicted, 16 points), per protocol and mouse.

R28 counterexample search, full data, each recording. Class orders: the 8 rotations x 2 reflections of the circle plus 32 random
orders (rng 5); rung schedules {1,2,3,4,6,8}, {1,2,4,8}, {1,...,8}, {2,4,6,8}, {1,3,5,8}. Exact class-mean remainder per rung
(log PR total - log PR within) on the declared rung and on the ten-draw floor (rng 42; floor on a 40-point grid of sizes,
interpolated in log size), its
least-squares slope = the ladder's class-mean part; counterexample = a ladder whose class-mean part has the opposite sign to the
full-set term c. Curves of declared and floor remainders saved for the 16 cyclic orders at {1,2,3,4,6,8}.
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler cr_h1_b24s.py [n_workers]   Out: data_canonical/cr_h1_b24s.json"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the eight Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_h1_b24s.json"
E = 48
RUNGS = (1, 2, 3, 4, 6, 8)
SCHEDULES = {"S1": (1, 2, 3, 4, 6, 8), "S2": (1, 2, 4, 8), "S3": (1, 2, 3, 4, 5, 6, 7, 8), "S4": (2, 4, 6, 8), "S5": (1, 3, 5, 8)}


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def prospective(X, lab, sub48, ep, ow, om, b245):
    out = []
    for a, b in ((0, 1), (1, 0)):
        A, B = ep == a, ep == b
        hh = om.halves_by_sub(sub48[A])
        Ra = om.residuals(X[A][hh == 0], lab[A][hh == 0], sub48[A][hh == 0], "raw")
        Rb = om.residuals(X[A][hh == 1], lab[A][hh == 1], sub48[A][hh == 1], "raw")
        sig = om.sigma(om.split_half_K(Ra, Rb))
        mB = [np.where(lab[B] == c)[0] for c in range(8)]
        sizes = [sum(len(mB[j]) for j in range(k)) for k in RUNGS]
        secant = float(np.log(8 / (1 + 7 * sig)) / np.log(8))
        multi = float(np.polyfit(np.log(sizes), [np.log(k / (1 + (k - 1) * sig)) for k in RUNGS], 1)[0])
        cm_pred, c_full_A, _ = b245.classmean_prediction(b245.full_terms(X[A].astype(np.float64), lab[A]), sizes)
        XB, lB = X[B], lab[B]
        meas = ow.ladder((XB @ XB.T).astype(np.float64), lB, [np.concatenate(mB[:c]) for c in RUNGS], np.random.default_rng(42))
        out.append({"from": int(a), "to": int(b), "sigma_A": sig, "within_secant": secant, "within_multirung": multi,
                    "classmean_pred": cm_pred, "c_full_A": c_full_A, "delta_pred_secant": secant + cm_pred,
                    "delta_pred_multirung": multi + cm_pred, "measured": meas})
    return out


def counterexamples(X, lab, ow):
    G = (X @ X.T).astype(np.float64); n = len(G); m = [np.where(lab == b)[0] for b in range(8)]
    full = ow.parts_from_gram(G, lab); c_full = float(full[0] - full[1])
    # the floor's class-mean remainder is a smooth function of subset size: ten draws on a grid of sizes, interpolated in log size
    grid = np.unique(np.concatenate([np.round(np.geomspace(min(len(q) for q in m) // 2, n, 40)).astype(int), [n]]))
    rng0 = np.random.default_rng(42); fgrid = []
    for size in grid:
        vals = [ow.parts_from_gram(G[np.ix_(i, i)], lab[i]) for i in (rng0.choice(n, size, replace=False) for _ in range(ow.N_NULL))]
        fgrid.append(float(np.mean([v[0] - v[1] for v in vals])))
    floor = lambda size: float(np.interp(np.log(size), np.log(grid), fgrid))
    cyc = [[(o + s * j) % 8 for j in range(8)] for o in range(8) for s in (1, -1)]
    rng = np.random.default_rng(5); rnd = [list(rng.permutation(8)) for _ in range(32)]
    res, curves = [], []
    for kind, orders in (("cyclic", cyc), ("random", rnd)):
        for oi, order in enumerate(orders):
            for sname, sched in SCHEDULES.items():
                sets = [np.concatenate([m[j] for j in order[:k]]) for k in sched]
                dec = []
                for sel in sets:
                    v = ow.parts_from_gram(G[np.ix_(sel, sel)], lab[sel]); dec.append(float(v[0] - v[1]))
                flo = [floor(len(s)) for s in sets]
                x = np.log([len(s) for s in sets]); y = np.array(dec) - np.array(flo)
                slope = float(np.polyfit(x, y, 1)[0])
                res.append({"kind": kind, "order": [int(j) for j in order], "schedule": sname, "classmean_slope": slope,
                            "opposite_to_c_full": bool(np.sign(slope) != np.sign(c_full))})
                if kind == "cyclic" and sname == "S1":
                    curves.append({"order": [int(j) for j in order], "x": x.tolist(), "declared": dec, "floor": flo})
    return {"c_full": c_full, "ladders": res, "n_ladders": len(res), "n_opposite": int(sum(r["opposite_to_c_full"] for r in res)),
            "curves_cyclic_S1": curves}


def run_one(fp):
    ow = _load("ow", "cr_order_and_withinbin_controls.py"); om = _load("om", "cr_overlap_mining.py"); b245 = _load("b245", "cr_h1_b245.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab, sub48 = ow.bins(phi, 8), om.bins(phi, 48); ep = (np.arange(len(phi)) // E) % 2
    row = {"name": fp.stem, "prospective": prospective(X, lab, sub48, ep, ow, om, b245)}
    print(f"{fp.stem[9:30]:22s} R27 " + " | ".join(f"pred {p['delta_pred_multirung']:+.3f} meas {p['measured']['delta']:+.3f} "
          f"(within {p['within_multirung']:+.3f}/{p['measured']['within']:+.3f}, cm {p['classmean_pred']:+.3f}/{p['measured']['class_means']:+.3f})"
          for p in row["prospective"]) + f" ({time.time() - t0:.0f}s)", flush=True)
    row["counterexamples"] = counterexamples(X, lab, ow)
    ce = row["counterexamples"]
    print(f"{fp.stem[9:30]:22s} R28 c_full {ce['c_full']:+.3f}: {ce['n_opposite']}/{ce['n_ladders']} ladders opposite ({time.time() - t0:.0f}s) DONE", flush=True)
    return row


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(run_one, sorted(RAW.glob("gratings*.npy"))))
    mouse = [r["name"].split("_GT")[1][0] for r in rows]
    prot = [next(k for k in ("low_contrast", "drifting", "local") if k in r["name"]) for r in rows]
    pts = []
    for i, r in enumerate(rows):
        for p in r["prospective"]:
            others = [rows[j] for j in range(8) if prot[j] == prot[i] and mouse[j] != mouse[i]]
            base = {k: float(np.mean([[q for q in o["prospective"] if q["to"] == p["to"]][0]["measured"][k] for o in others])) for k in ("delta", "within", "class_means")}
            pts.append({"name": r["name"], "to": p["to"], "pred_delta": p["delta_pred_multirung"], "pred_delta_secant": p["delta_pred_secant"],
                        "pred_within": p["within_multirung"], "pred_cm": p["classmean_pred"], "base": base, "meas": p["measured"]})
    err = lambda k_pred, k_meas: [abs(q[k_pred] - q["meas"][k_meas]) for q in pts]
    berr = lambda k: [abs(q["base"][k] - q["meas"][k]) for q in pts]
    pd_, md_ = np.array([q["pred_delta"] for q in pts]), np.array([q["meas"]["delta"] for q in pts])
    cal = np.polyfit(pd_, md_, 1)
    v = {"R27_delta_mae_model": float(np.mean(err("pred_delta", "delta"))), "R27_delta_mae_secant": float(np.mean(err("pred_delta_secant", "delta"))),
         "R27_delta_mae_protocol_baseline": float(np.mean(berr("delta"))),
         "R27_within_mae_model": float(np.mean(err("pred_within", "within"))), "R27_within_mae_baseline": float(np.mean(berr("within"))),
         "R27_cm_mae_model": float(np.mean(err("pred_cm", "class_means"))), "R27_cm_mae_baseline": float(np.mean(berr("class_means"))),
         "R27_delta_abs_err_le_005": f"{sum(e <= 0.05 for e in err('pred_delta', 'delta'))}/{len(pts)}",
         "R27_calibration_slope_intercept": [float(cal[0]), float(cal[1])],
         "R27_r_pred_meas_delta": float(np.corrcoef(pd_, md_)[0, 1]),
         "R28_opposite_total": f"{sum(r['counterexamples']['n_opposite'] for r in rows)}/{sum(r['counterexamples']['n_ladders'] for r in rows)}",
         "R28_opposite_per_recording": {r["name"]: r["counterexamples"]["n_opposite"] for r in rows}}
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump({"rows": rows, "points": pts, "verdict": v}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
