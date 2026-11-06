"""C11 request (Oct 23, Fig 3): the frozen two-term prediction of delta on the three pvc-11 arrays, nothing fitted, exactly as for
imaging (cr_h1_b245) and Allen (h1_allen_b2_predict): within_pred = the mean-field sigma law with split-half sigma at K = 12,
ln[12/(1 + 11 sigma)]/ln 12 (cr_h1_m_pvc11.json raw.mean_field, recomputed here and asserted equal); classmean_pred =
cr_h1_b245.classmean_prediction logic with the macaque rung schedule [1, 2, 3, 4, 6, 8, 12] in the ladder's class order (0..11 by
30 degrees) and the full-set terms at K = 12; delta_pred = the sum. Measured: cr_h1_m_pvc11.json raw.ladder. Raw frame, the same
units and window as cr_h1_m_pvc11 (load_array). Out: data_canonical/cr_h1_m_pvc11_twoterm.json"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


M = _load("m", "cr_h1_m_pvc11.py"); B = _load("b245", "cr_h1_b245.py"); om = _load("om", "cr_overlap_mining.py")
RUNGS, K = (1, 2, 3, 4, 6, 8, 12), 12


def classmean_prediction(T, sizes):
    """cr_h1_b245.classmean_prediction with the K = 12 rung schedule (identical algebra)."""
    cm, cnt = T["cm"], T["counts"]; beta = T["b1"] / T["w1"]
    c_full = np.log((1 + beta) ** 2 / (1 + (T["b2"] + 2 * T["x"]) / T["w2"]))
    ys = []
    for k in RUNGS:
        p = cnt[:k] / cnt[:k].sum(); mu = (p[:, None] * cm[:k]).sum(0)
        u = float((p * ((cm[:k] - mu) ** 2).sum(1)).sum()) / T["b1"]
        ys.append(np.log((1 + u * beta) ** 2 / (1 + (u ** 2 * T["b2"] + 2 * u * T["x"]) / T["w2"])) - c_full)
    return float(np.polyfit(np.log(sizes), ys, 1)[0]), float(c_full)


stored = {r["monkey"]: r for r in json.load(open(HERE.parent / "data_canonical" / "cr_h1_m_pvc11.json"))["arrays"]}
rows = []
for m in (1, 2, 3):
    X, lab, info = M.load_array(m)
    h = M.halves(lab)
    sig = om.sigma(om.split_half_K(om.residuals(X[h == 0], lab[h == 0], lab[h == 0], "raw"), om.residuals(X[h == 1], lab[h == 1], lab[h == 1], "raw")))
    wp = float(np.log(K / (1 + (K - 1) * sig)) / np.log(K))
    assert abs(wp - stored[m]["raw"]["mean_field"]) < 1e-9, (wp, stored[m]["raw"]["mean_field"])
    sizes = [int((lab < k).sum()) for k in RUNGS]
    cmp_, c_full = classmean_prediction(B.full_terms(X.astype(np.float64), lab, K=K), sizes)
    meas = stored[m]["raw"]["ladder"]
    rows.append({"monkey": m, "N": int(X.shape[1]), "sigma": float(sig), "within_pred": wp, "classmean_pred": cmp_, "delta_pred": wp + cmp_,
                 "c_full": c_full, "measured_delta": meas["delta"], "measured_within": meas["within"], "measured_class_means": meas["class_means"],
                 "error_delta": wp + cmp_ - meas["delta"]})
    r = rows[-1]
    print(f"monkey{m}: pred delta {r['delta_pred']:+.3f} (within {wp:+.3f}, cm {cmp_:+.3f}) | measured {r['measured_delta']:+.3f} "
          f"(within {r['measured_within']:+.3f}, cm {r['measured_class_means']:+.3f}) | c_full {c_full:+.3f}", flush=True)
json.dump({"rungs": RUNGS, "rows": rows}, open(HERE.parent / "data_canonical" / "cr_h1_m_pvc11_twoterm.json", "w"), indent=1)
print("ALL DONE")
