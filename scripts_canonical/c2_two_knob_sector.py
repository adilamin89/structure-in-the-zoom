"""Why delta falls with the even share across the 8 imaging recordings
but rises with the even amplitude across the 167 Allen populations.

Two-knob reading (Section 7's model): at fixed within-class geometry, a larger even
(quadrupole) share makes antipodal classes alike, so accumulating classes adds fewer
new directions and delta falls; the amount of angular modulation at all (how much the
class means differ, 1 - a in C = a + sum of harmonics, C(0) = 1) and the class-dependent
gain that rides with it raise delta. Across the imaging recordings the share varies
(the localized ensemble is dipole-dominant); across Allen populations the correlation
profile is nearly flat (a ~ 0.9: untuned mean rate dominates) and the "quadrupole
amplitude" mostly measures how much tuning a population has.

Tests on existing artifacts only (no new run):
  imaging: delta (run72 design A, all neurons) and its within-class / class-mean parts
           vs modulation M = 1 - a and even share E = (b2 + b4) / (b2 + b4 + |c1| + |c3|)
           from multipole_harmonics_8dir.json;
  Allen:   delta_dir8 (allen_expansion_all_sessions.json, ten-draw floor) vs M and
           E' = b2 / (b2 + |c1|) (the four-term Allen fit has a single odd harmonic),
           partial correlations, within-class PR (allen_withinclass_pr.json).
Output: data_canonical/c2_two_knob_sector.json
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

DATA = Path(__file__).resolve().parent.parent / "data_canonical"


def partial(x, y, z):
    """Partial correlation of x and y given the columns of z (least-squares residuals)."""
    Z = np.column_stack([np.ones(len(x))] + [np.asarray(c, float) for c in z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def imaging():
    mh = {r["name"]: r for r in json.load(open(DATA / "multipole_harmonics_8dir.json"))["rows"]
          if r.get("status") == "ok"}
    split = json.load(open(DATA / "run72_paper_designs_split.json"))["rows"]
    rows = []
    for key, rec in split.items():
        h = mh[rec["name"]]["coef"]
        A = rec["all"]["A"]
        even = h["b2"] + h["b4"]
        odd = abs(h["c1"]) + abs(h["c3"])
        rows.append(dict(
            key=key, name=rec["name"], delta=A["delta"], within=A["slope_noise"], means=A["slope_probe"],
            a=h["a"], M=1 - h["a"], E=even / (even + odd), b2_over_c1=h["b2"] / h["c1"],
            b2=h["b2"], c1=h["c1"], even=even, odd=odd,
            order_effect=rec["all"]["order_effect"], order_within=rec["all"]["order_noise"],
            order_means=rec["all"]["order_probe"]))
    d = np.array([r["delta"] for r in rows])
    out = {"rows": rows, "n": len(rows)}
    for var in ["E", "b2_over_c1", "M", "even", "odd"]:
        x = np.array([r[var] for r in rows])
        out[f"spearman_delta_{var}"] = float(stats.spearmanr(d, x)[0])
        out[f"pearson_delta_{var}"] = float(np.corrcoef(d, x)[0, 1])
        for part in ["within", "means"]:
            p = np.array([r[part] for r in rows])
            out[f"spearman_{part}_{var}"] = float(stats.spearmanr(p, x)[0])
    E = np.array([r["E"] for r in rows]); M = np.array([r["M"] for r in rows])
    out["partial_delta_E_given_M"] = partial(d, E, [M])
    out["partial_delta_M_given_E"] = partial(d, M, [E])
    out["pearson_E_M"] = float(np.corrcoef(E, M)[0, 1])
    # within stimulus type: does the sign hold inside the drifting (D) and full-field (D+C) groups?
    for grp, keys in {"fullfield": ["D1", "D2", "D3", "C1", "C2"], "drifting": ["D1", "D2", "D3"]}.items():
        sub = [r for r in rows if r["key"] in keys]
        out[f"spearman_delta_E_{grp}"] = float(stats.spearmanr([r["delta"] for r in sub],
                                                               [r["E"] for r in sub])[0])
    return out


def allen():
    mp = json.load(open(DATA / "allen_multipoles_all_sessions.json"))["results"]
    ex = json.load(open(DATA / "allen_expansion_all_sessions.json"))["results"]
    wc = json.load(open(DATA / "allen_withinclass_pr.json"))["results"]
    exd = {(s["session"], a): v for s in ex if s.get("status") == "ok" for a, v in s["areas"].items()}
    wcd = {(s["session"], a): v for s in wc if s.get("status") == "ok" for a, v in s["areas"].items()}
    rows = []
    for s in mp:
        if s.get("status") != "ok":
            continue
        for a, v in s["areas"].items():
            k = (s["session"], a)
            if k not in exd:
                continue
            b2, c1, b4 = v["b_quadrupole"], v["c_dipole"], v["b4"]
            rows.append(dict(session=s["session"], area=a, delta=exd[k]["delta_dir8"], a=v["a"], M=1 - v["a"],
                             b2=b2, c1=c1, b4=b4, A_even=float(np.hypot(b2, b4)),
                             E=b2 / (b2 + abs(c1)) if (b2 + abs(c1)) > 0 else np.nan,
                             wcpr1=wcd.get(k, {}).get("pr_within_rung1", np.nan), n_units=v["n_units"]))
    d = np.array([r["delta"] for r in rows])
    g = lambda key: np.array([r[key] for r in rows], float)
    M, E, Aev, b2, wc = g("M"), g("E"), g("A_even"), g("b2"), g("wcpr1")
    ok = np.isfinite(E) & np.isfinite(wc)
    out = {"n": len(rows), "n_ok": int(ok.sum()),
           "a_median": float(np.median(g("a"))), "a_range": [float(g("a").min()), float(g("a").max())],
           "E_median": float(np.nanmedian(E)), "E_range": [float(np.nanmin(E)), float(np.nanmax(E))],
           "r_delta_b2": float(np.corrcoef(d, b2)[0, 1]),
           "r_delta_Aeven": float(np.corrcoef(d, Aev)[0, 1]),
           "r_delta_M": float(np.corrcoef(d, M)[0, 1]),
           "r_delta_E": float(np.corrcoef(d[ok], E[ok])[0, 1]),
           "spearman_delta_E": float(stats.spearmanr(d[ok], E[ok])[0]),
           "r_b2_M": float(np.corrcoef(b2, M)[0, 1]),
           "r_E_M": float(np.corrcoef(E[ok], M[ok])[0, 1]),
           "partial_delta_E_given_M": partial(d[ok], E[ok], [M[ok]]),
           "partial_delta_M_given_E": partial(d[ok], M[ok], [E[ok]]),
           "partial_delta_b2_given_M": partial(d, b2, [M]),
           "partial_delta_E_given_M_wcpr1": partial(d[ok], E[ok], [M[ok], wc[ok]]),
           "partial_delta_M_given_E_wcpr1": partial(d[ok], M[ok], [E[ok], wc[ok]])}
    # VISp alone (32 animals, one stimulus, one area)
    v = np.array([r["area"] == "VISp" for r in rows]) & ok
    out["visp_n"] = int(v.sum())
    out["visp_r_delta_M"] = float(np.corrcoef(d[v], M[v])[0, 1])
    out["visp_r_delta_E"] = float(np.corrcoef(d[v], E[v])[0, 1])
    out["visp_partial_delta_E_given_M"] = partial(d[v], E[v], [M[v]])
    return out


def direction_selective():
    """run59 (registered Sep 5): equal-size DSI thirds per recording; and the ordering split counts (run72)."""
    v = json.load(open(DATA / "run59_shift_by_direction_selectivity.json"))["verdict"]
    ds, nds, frac = map(np.array, (v["delta_DS"], v["delta_nonDS"], v["ds_fraction"]))
    split = json.load(open(DATA / "run72_paper_designs_split.json"))["rows"]
    om = np.array([r["all"]["order_probe"] for r in split.values()])
    ow = np.array([r["all"]["order_noise"] for r in split.values()])
    mp = json.load(open(DATA / "allen_multipoles_all_sessions.json"))["results"]
    ratio = [abs(a["c_dipole"]) / a["b_quadrupole"] for s in mp if s.get("status") == "ok"
             for a in s["areas"].values() if a["b_quadrupole"] > 0]
    im = json.load(open(DATA / "multipole_harmonics_8dir.json"))["rows"]
    im_ratio = [r["coef"]["c1"] / r["coef"]["b2"] for r in im if r.get("status") == "ok"]
    return {"ds_gt_nonds": int((ds > nds).sum()), "delta_DS_range": [ds.min(), ds.max()],
            "delta_nonDS_range": [nds.min(), nds.max()], "gap_range": [(ds - nds).min(), (ds - nds).max()],
            "ds_fraction_range": [frac.min(), frac.max()],
            "rho_full_vs_dsfrac": v["H4_rho_full_vs_DSfrac"],
            "order_means_positive": int((om > 0.01).sum()), "order_means_near_zero": int((abs(om) <= 0.02).sum()),
            "order_means_values": om.tolist(), "order_within_negative": int((ow < 0).sum()),
            "allen_c1_over_b2_median": float(np.median(ratio)),
            "allen_c1_over_b2_iqr": np.percentile(ratio, [25, 75]).tolist(),
            "imaging_c1_over_b2_range": [min(im_ratio), max(im_ratio)]}


if __name__ == "__main__":
    res = {"imaging": imaging(), "allen": allen(), "direction_selective": direction_selective()}
    print("DS", res["direction_selective"])
    json.dump(res, open(DATA / "c2_two_knob_sector.json", "w"), indent=1, default=float)
    im, al = res["imaging"], res["allen"]
    print("IMAGING n=8")
    for r in im["rows"]:
        print(f"  {r['key']}: delta {r['delta']:+.3f} within {r['within']:+.3f} means {r['means']:+.3f} | "
              f"a {r['a']:.2f} M {r['M']:.2f} E {r['E']:.2f} b2/c1 {r['b2_over_c1']:.2f}")
    for k, v in im.items():
        if k not in ("rows", "n"):
            print(f"  {k}: {v:+.3f}")
    print("ALLEN")
    for k, v in al.items():
        print(f"  {k}: {v}")
