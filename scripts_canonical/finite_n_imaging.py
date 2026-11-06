"""The overlap identity at finite samples on the eight grating recordings: the same arms as finite_n_identity.arms against the
within part of the direction ladder (cr_order_and_withinbin_controls.json 'orig', the Fig 3A imaging points) on all trials, and on the
held-out halves of twoterm_imaging.json (sigma from one half, the part measured on the other): W2x takes its moments from the estimator
half and its counts from the measured half. Loading, bins and halves as twoterm_imaging.py; each half's sigma is reproduced and asserted.
Run: python -X faulthandler finite_n_imaging.py [n_workers]   Out: data_canonical/finite_n_imaging.json"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent / "raw"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def run_one(fp):
    import finite_n_identity as P
    ow = _load("ow", "cr_order_and_withinbin_controls.py"); om = _load("om", "overlap_identity_checks.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab = ow.bins(phi, 8); sub48 = om.bins(phi, 48)
    b245 = next(r for r in json.load(open(DATA / "twoterm_imaging.json"))["rows"] if r["name"] == fp.stem)
    t9 = next(r for r in json.load(open(DATA / "cr_order_and_withinbin_controls.json"))["rows"] if r["name"] == fp.stem)["orig"]
    cnt = np.bincount(lab, minlength=8)
    mom = P.moments(X, lab, seed=8_000)
    a = P.arms(mom, cnt, lab, b245["B2"]["sigma"], 14_000)
    assert abs(a["W0"] - b245["B2"]["within_pred"]) < 1e-9
    row = {"name": fp.stem, "counts": cnt.tolist(), "measured_within": t9["within"], "full": {k: v for k, v in a.items()},
           "t": mom["t"].tolist(), "T": mom["T"].tolist(), "m4": mom["m4"].tolist(), "Kn": mom["Kn"].tolist(), "halves": []}
    h = om.halves_by_sub(sub48)
    for (sh, mh), b4 in zip(((0, 1), (1, 0)), b245["B4"]):
        S, M = h == sh, h == mh
        hh = om.halves_by_sub(sub48[S])
        Ra = om.residuals(X[S][hh == 0], lab[S][hh == 0], sub48[S][hh == 0], "raw")
        Rb = om.residuals(X[S][hh == 1], lab[S][hh == 1], sub48[S][hh == 1], "raw")
        sig = om.sigma(om.split_half_K(Ra, Rb))
        assert abs(sig - b4["sigma"]) < 1e-6, (fp.stem, sig, b4["sigma"])
        momS = P.moments(X[S], lab[S], seed=8_100 + sh); momM = P.moments(X[M], lab[M], seed=8_200 + mh)
        cM = np.bincount(lab[M], minlength=8)
        ah = P.arms(momM, cM, lab[M], sig, 14_100 + sh, mom_x=momS)
        assert abs(ah["W0"] - b4["pred"]) < 1e-9
        row["halves"].append({"est_half": sh, "measured_within": b4["within"], "sigma": sig, **ah})
    print(f"{fp.stem[9:32]:24s} full: meas {t9['within']:+.3f} W0 {a['W0']:+.3f} W1 {a['W1']:+.3f} W2 {a['W2']:+.3f} | halves meas "
          + " ".join(f"{q['measured_within']:+.3f}" for q in row["halves"]) + " W0 " + " ".join(f"{q['W0']:+.3f}" for q in row["halves"])
          + " W2x " + " ".join(f"{q['W2x']:+.3f}" for q in row["halves"]) + f" | n/class {cnt.mean():.0f} PR_W {a['PR_W_pop']:.0f} "
          f"PR_class {np.mean(a['PR_class_pop']):.0f} ({time.time() - t0:.0f}s) DONE", flush=True)
    return row


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    files = sorted(RAW.glob("gratings*.npy"))
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(run_one, files))
    full = np.array([[r["full"][k] - r["measured_within"] for k in ("W0", "W1", "W2", "W2g")] for r in rows])
    half = np.array([[np.mean([q[k] - q["measured_within"] for q in r["halves"]]) for k in ("W0", "W1", "W2", "W2x")] for r in rows])
    se_full = float(np.std(full[:, 0], ddof=1) / np.sqrt(len(rows)))
    se_half = float(np.std(half[:, 0], ddof=1) / np.sqrt(len(rows)))
    v = {"bar_full_2se": 0.040, "bar_halves_2se": 0.038, "se_full_W0_recomputed": se_full, "se_halves_W0_recomputed": se_half,
         "mean_err_full": dict(zip(("W0", "W1", "W2", "W2g"), full.mean(0).tolist())),
         "mean_err_halves": dict(zip(("W0", "W1", "W2", "W2x"), half.mean(0).tolist())),
         "kernel_term_full": [r["full"]["W1"] - r["full"]["W0"] for r in rows],
         "finite_n_term_full": [r["full"]["W2"] - r["full"]["W1"] for r in rows],
         "r_W0_full": float(np.corrcoef([r["full"]["W0"] for r in rows], [r["measured_within"] for r in rows])[0, 1]),
         "r_W2_full": float(np.corrcoef([r["full"]["W2"] for r in rows], [r["measured_within"] for r in rows])[0, 1]),
         "r_W2x_halves": float(np.corrcoef([np.mean([q["W2x"] for q in r["halves"]]) for r in rows],
                                           [np.mean([q["measured_within"] for q in r["halves"]]) for r in rows])[0, 1]),
         "r_W0_halves": float(np.corrcoef([np.mean([q["W0"] for q in r["halves"]]) for r in rows],
                                          [np.mean([q["measured_within"] for q in r["halves"]]) for r in rows])[0, 1])}
    v["P2_full_W2_within_bar"] = abs(v["mean_err_full"]["W2"]) <= v["bar_full_2se"]
    v["P2_halves_W2x_within_bar"] = abs(v["mean_err_halves"]["W2x"]) <= v["bar_halves_2se"]
    v["P2_met"] = v["P2_full_W2_within_bar"] and v["P2_halves_W2x_within_bar"]
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(DATA / "finite_n_imaging.json", "w"), indent=1, default=float)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
