"""H5 (Oct 25): generator for data_canonical/allen_withinclass_partial.json, which was written by an inline analysis with no script
(App H: 'the within-class PR of the first rung, uncorrelated with the amplitude (r = +0.02), predicts the shift independently with
the opposite sign (partial r = -0.47)'). Joins allen_expansion_all_sessions.json (delta_dir8, n_units), allen_multipoles_all_sessions.json
(b_quadrupole) and allen_withinclass_pr.json (pr_within_rung1, corr_within_mean) over the 167 populations; partial r = Pearson r of
OLS residuals on the controls (with intercept). 'none' must reproduce the canonical JSON.
Out: data_canonical/allen_withinclass_partial.json"""
import json
from pathlib import Path

import numpy as np

DC = Path(__file__).resolve().parent.parent / "data_canonical"


def byk(fn):
    return {(s["session"], a): v for s in json.load(open(DC / fn))["results"] if s.get("status") == "ok" for a, v in s["areas"].items()}


def partial(y, x, controls):
    Z = np.column_stack([np.ones(len(y))] + list(controls))
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


ex, mp, wc = byk("allen_expansion_all_sessions.json"), byk("allen_multipoles_all_sessions.json"), byk("allen_withinclass_pr.json")
keys = [k for k in ex if k in mp and k in wc]
d = np.array([ex[k]["delta_dir8"] for k in keys]); b2 = np.array([mp[k]["b_quadrupole"] for k in keys])
w1 = np.array([wc[k]["pr_within_rung1"] for k in keys]); wcc = np.array([wc[k]["corr_within_mean"] for k in keys])
nu = np.array([ex[k]["n_units"] for k in keys], float)
out = {"n": len(keys), "pooled": float(np.corrcoef(d, b2)[0, 1]), "r_delta_wcpr1": float(np.corrcoef(d, w1)[0, 1]),
       "r_b2_wcpr1": float(np.corrcoef(b2, w1)[0, 1]), "partial_b2_given_wcpr1": partial(d, b2, [w1]),
       "partial_b2_given_wcpr1_nunits": partial(d, b2, [w1, nu]), "partial_b2_given_wccorr": partial(d, b2, [wcc]),
       "partial_b2_given_both": partial(d, b2, [w1, wcc]), "partial_wcpr1_given_b2": partial(d, w1, [b2])}
print(json.dumps(out, indent=1))
json.dump(out, open(DC / "allen_withinclass_partial.json", "w"))
print("DONE")
