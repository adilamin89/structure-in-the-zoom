"""Is law_E's 2.0% gap to law_A (bar 2%) the overdispersion
of raw r^2 (which still carries the Gaussian quadratic form's own spread) or the gain distribution's shape beyond kurtosis?
E' = the empirical r^2 quantiles rescaled about 1 to the deconvolved kurtosis kappa_tau (shape kept, second moment matched to law_A's).
Draw 0 of every (recording, n) cell, R4's neurons. Out: data_canonical/principal_angle_floor_gain_shape.json"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def one(fp):
    sys.path.insert(0, str(HERE))
    import principal_angle_floor_imaging as R
    import principal_angle_floor_law as FL
    ow = R._load("ow", "cr_order_and_withinbin_controls.py")
    dat = np.load(fp, allow_pickle=True).item()
    Xf = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); Xf /= Xf.std() + 1e-9
    yf = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    out = []
    for n in R.NS:
        rng = np.random.default_rng(30000 * n)
        [rng.choice(np.where(yf == c)[0], n, replace=False) for c in range(8)]
        cols = rng.choice(Xf.shape[1], R.N_NEUR, replace=False)
        Xa = Xf[:, cols].astype(np.float64); Ta = Xa - Xa.mean(0)
        lam, _, kap, deff, q = R.gain_and_spectrum(Ta)
        qs = 1 + (q - 1) * np.sqrt((kap - 1) / np.var(q))          # E' : shape of r^2, variance = kappa_tau - 1
        qs = np.clip(qs, 1e-6, None)
        A = FL.floor_pa(lam, n, R.K, ("gamma", kap), n_grid=R.GRID)[0]
        E = FL.floor_pa(lam, n, R.K, q, n_grid=R.GRID)[0]
        Ep = FL.floor_pa(lam, n, R.K, qs, n_grid=R.GRID)[0]
        out.append({"name": fp.stem, "n": n, "A": A, "E": E, "Eprime": Ep, "kappa_tau": kap, "kappa_raw": float(np.mean(q ** 2) / np.mean(q) ** 2)})
        print(fp.stem[:28], n, f"A {A:.4f} E {E:.4f} E' {Ep:.4f} kappa_tau {kap:.2f} raw {out[-1]['kappa_raw']:.2f}", flush=True)
    return out


def main():
    sys.path.insert(0, str(HERE))
    import principal_angle_floor_imaging as R
    with ProcessPoolExecutor(8, mp_context=get_context("spawn")) as ex:
        rows = [x for p in ex.map(one, sorted(R.RAW.glob("gratings*.npy"))) for x in p]
    perm = {(x["name"], x["n"]): x["perm"] for x in json.load(open(HERE.parent / "data_canonical" / "principal_angle_floor_imaging.json"))["rows"]
            if x["draw"] == 0}                                          # draw 0's own permutation floor (same neurons)
    for r in rows:
        r["perm_cell"] = perm[(r["name"], r["n"])]
    s = {"A_minus_E_over_perm": float(np.median([abs(r["A"] - r["E"]) / r["perm_cell"] for r in rows])),
         "A_minus_Eprime_over_perm": float(np.median([abs(r["A"] - r["Eprime"]) / r["perm_cell"] for r in rows])),
         "Eprime_relerr_vs_perm": float(np.median([abs(r["Eprime"] - r["perm_cell"]) / r["perm_cell"] for r in rows])),
         "A_relerr_vs_perm_draw0": float(np.median([abs(r["A"] - r["perm_cell"]) / r["perm_cell"] for r in rows])),
         "E_minus_A_signed_median": float(np.median([(r["E"] - r["A"]) / r["perm_cell"] for r in rows]))}
    print("SUMMARY", json.dumps(s, indent=1))
    json.dump({"note": "descriptive: gain shape beyond kurtosis", "rows": rows, "summary": s},
              open(HERE.parent / "data_canonical" / "principal_angle_floor_gain_shape.json", "w"), indent=1)


if __name__ == "__main__":
    main()
