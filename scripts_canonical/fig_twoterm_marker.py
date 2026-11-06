"""Fig 3A's open marker for the 72 lattice and spin-wave ensembles from the sampling law: the within part sits below the overlap
law by s = (m - 1) PR_class / (n ln K) at first order (Eq. 2 in both ladders), m = K / [1 + (K - 1) sigma], K = 8, PR_class the
classes' population PR (t_c^2 / T_cc, mean over classes), n the configurations per class. Open when s > 0.05. Printed beside it: the
earlier rule (n / PR_class < 2 with the sample PR of one class), the full finite-n term W2 - W1 and W2's error (finite_n_lattice.json).
Out: data_canonical/fig_twoterm_marker.json"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data_canonical"
K, BAR = 8, 0.05


def main():
    sc = json.load(open(DATA / "finite_n_lattice.json"))
    pol = {(r["T"], r["ens"]): r for r in json.load(open(DATA / "xy_polar_twoterm.json"))["rows"]}
    lat = {(r["L"], r["T"], r["per_class"], r["ens"]): r for r in json.load(open(DATA / "xy_twoterm.json"))["lattice"]}
    out = []
    for group in ("polar", "nem", "sw"):
        for r in sc[group]["rows"]:
            n = float(np.mean(r["counts"])); prc = float(np.mean(r["PR_class_pop"])); sig = r["sigma_H1"]
            m = K / (1 + (K - 1) * sig); s = (m - 1) * prc / (n * np.log(K))
            if group == "polar":
                old = pol[(r["T"], r["ens"])]["n_over_d"]
            elif group == "nem":
                q = lat[(r["L"], r["T"], r["per_class"], r["ens"])]; old = q["per_class"] / q["PR_class"]
            else:
                old = None                                     # spin waves: no printed n/d (drawn filled)
            row = {"group": group, "key": r["key"], "n_per_class": n, "PR_class_pop": prc, "n_over_PRclass_pop": n / prc, "sigma": sig,
                   "s_first_order": float(s), "finite_n_W2_minus_W1": r["W2"] - r["W1"], "W2_err": r["W2"] - r["measured_within"],
                   "W0_err": r["W0"] - r["measured_within"], "n_over_d_printed": old,
                   "open_old": (old is not None and old < 2), "open_new": bool(s > BAR)}
            row["flips"] = row["open_old"] != row["open_new"]
            out.append(row)
    for g in ("polar", "nem", "sw"):
        rs = [r for r in out if r["group"] == g]
        print(f"{g}: open old {sum(r['open_old'] for r in rs)}, new {sum(r['open_new'] for r in rs)} of {len(rs)}; flips "
              f"{sum(r['flips'] for r in rs)}", flush=True)
        for r in rs:
            if r["flips"]:
                print(f"   {r['key']:22s} n/PR_pop {r['n_over_PRclass_pop']:.2f} (printed n/d {r['n_over_d_printed'] or float('nan'):.2f}) sigma "
                      f"{r['sigma']:.2f} s {r['s_first_order']:.3f} W2-W1 {r['finite_n_W2_minus_W1']:+.3f} W0 err {r['W0_err']:+.3f}", flush=True)
    json.dump(out, open(DATA / "fig_twoterm_marker.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
