"""Run 1 - Allen partial correlations + hierarchical (mixed / cluster-robust) stats.

Tests the two-knobs resolution of the quadrupole paradox on the real 167
populations, and answers the standard pseudoreplication objection.

REGISTERED EXPECTATIONS (before run):
P1: pooled Pearson r(delta_dir8, b_quadrupole) reproduces ~+0.41 (sanity).
P2: partial r(delta, b2 | harmonic shape b/(b+c)) SURVIVES (stays clearly
    positive) - amplitude, not sector shape, carries the association.
P3: partial r(delta, b2 | dipole, b4, cardinal, n_units, pr_full) survives at
    reduced magnitude - b2 is not merely a proxy for population size/rank.
    (The decisive within-class-PR control needs raw NWBs; camera-ready.)
P4: mixed model (session random intercept, area fixed effects): b2 coefficient
    positive with p < 0.05 under cluster-robust and REML inference.

Data: theta_project/data_canonical/allen_{expansion,multipoles}_all_sessions.json
Out:  feedback_runs/run1_allen_partial_mixed.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
DC = HERE.parent / "data_canonical"

exp = json.load(open(DC / "allen_expansion_all_sessions.json"))["results"]
mul = json.load(open(DC / "allen_multipoles_all_sessions.json"))["results"]
mul_by_sess = {m["session"]: m for m in mul}

rows = []
for s in exp:
    ms = mul_by_sess.get(s["session"])
    if ms is None:
        continue
    for area, a in s["areas"].items():
        am = ms["areas"].get(area)
        if am is None:
            continue
        rows.append(dict(
            session=s["session"], area=area, n_units=a["n_units"],
            delta=a["delta_dir8"], b2=am["b_quadrupole"], c1=am["c_dipole"],
            b4=am["b4"], cardinal=am["cardinal_fraction"],
            pr_full=a["physics"]["pr_full"],
            shape=am["b_quadrupole"] / (am["b_quadrupole"] + abs(am["c_dipole"]) + 1e-9),
        ))
df = pd.DataFrame(rows)
print(f"assembled {len(df)} populations, {df.session.nunique()} sessions, "
      f"{df.area.nunique()} areas", flush=True)


def partial_r(df, y, x, controls):
    """Pearson r between residuals of y and x after OLS on controls."""
    Z = np.column_stack([np.ones(len(df))] + [df[c].values for c in controls])
    ry = df[y].values - Z @ np.linalg.lstsq(Z, df[y].values, rcond=None)[0]
    rx = df[x].values - Z @ np.linalg.lstsq(Z, df[x].values, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


out = {"n_pop": len(df), "n_sessions": int(df.session.nunique())}

r0 = float(np.corrcoef(df.delta, df.b2)[0, 1])
out["pooled_r"] = r0
print(f"P1 pooled r(delta, b2) = {r0:+.3f}  (paper: +0.41)", flush=True)

tests = {
    "given_shape": ["shape"],
    "given_dipole": ["c1"],
    "given_cardinal": ["cardinal"],
    "given_b4": ["b4"],
    "given_size_rank": ["n_units", "pr_full"],
    "given_all": ["shape", "c1", "b4", "cardinal", "n_units", "pr_full"],
}
out["partial"] = {}
for name, ctl in tests.items():
    pr = partial_r(df, "delta", "b2", ctl)
    out["partial"][name] = pr
    print(f"  partial r(delta, b2 | {','.join(ctl)}) = {pr:+.3f}", flush=True)

# --- hierarchical inference ---
ols = smf.ols("delta ~ b2 + c1 + cardinal + C(area)", data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["session"]})
out["cluster_robust"] = {"b2_coef": float(ols.params["b2"]),
                         "b2_se": float(ols.bse["b2"]),
                         "b2_p": float(ols.pvalues["b2"])}
print(f"P4 cluster-robust OLS (session clusters): b2 = "
      f"{ols.params['b2']:+.3f} ± {ols.bse['b2']:.3f}, p = {ols.pvalues['b2']:.2e}",
      flush=True)

mm = smf.mixedlm("delta ~ b2 + c1 + cardinal + C(area)", data=df,
                 groups=df["session"]).fit(reml=True)
out["mixed_model"] = {"b2_coef": float(mm.params["b2"]),
                      "b2_se": float(mm.bse["b2"]),
                      "b2_p": float(mm.pvalues["b2"])}
print(f"P4 mixed model (session random intercept): b2 = "
      f"{mm.params['b2']:+.3f} ± {mm.bse['b2']:.3f}, p = {mm.pvalues['b2']:.2e}",
      flush=True)

# per-mouse collapse sanity (paper: r=0.39)
pm = df.groupby("session")[["delta", "b2"]].mean()
out["per_mouse_r"] = float(np.corrcoef(pm.delta, pm.b2)[0, 1])
print(f"per-mouse collapse r = {out['per_mouse_r']:+.3f}  (paper: +0.39)", flush=True)

json.dump(out, open(HERE / "run1_allen_partial_mixed.json", "w"), indent=1)
print("DONE run1", flush=True)
