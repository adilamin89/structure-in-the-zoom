"""Run 39 — Allen mixed model with A_even / A_odd entering DIRECTLY (round-4 item 6).

WHY: the paper defines the composite even-sector amplitude A_even =
sqrt(b2^2 + b4^2) and the odd amplitude A_odd = |c1| but reports them only
as raw correlations (r = +0.41 / -0.04); the hierarchical inference (run1)
was fit on b2 alone. This run refits the run1 models with A_even and A_odd
as the predictors, closing the composite-vs-component gap.

REGISTERED EXPECTATIONS (written before the run):
A1: A_even coefficient positive with p < 0.05 under both cluster-robust OLS
    and REML mixed model (session random intercept, area fixed effects) —
    mirroring b2's +0.86 +/- 0.17, p = 4e-7 (b2 dominates the composite).
A2: A_odd coefficient not significant (raw r = -0.04).
A3: partial r(delta, A_even | A_odd, cardinal, n_units, pr_full) stays
    clearly positive (amplitude association not a size/rank proxy).

Data: same assembly as run1 (allen_{expansion,multipoles}_all_sessions.json,
167 populations, 32 sessions).
Out: ../data_canonical/run39_allen_aeven_mixed.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
DC = HERE.parent / "data_canonical"
OUT = DC / "run39_allen_aeven_mixed.json"

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
            delta=a["delta_dir8"],
            A_even=float(np.sqrt(am["b_quadrupole"] ** 2 + am["b4"] ** 2)),
            A_odd=abs(am["c_dipole"]),
            cardinal=am["cardinal_fraction"],
            pr_full=a["physics"]["pr_full"],
        ))
df = pd.DataFrame(rows)
print(f"assembled {len(df)} populations, {df.session.nunique()} sessions",
      flush=True)

out = {"n_pop": len(df), "n_sessions": int(df.session.nunique())}

out["raw_r_A_even"] = float(np.corrcoef(df.delta, df.A_even)[0, 1])
out["raw_r_A_odd"] = float(np.corrcoef(df.delta, df.A_odd)[0, 1])
print(f"raw r(delta, A_even) = {out['raw_r_A_even']:+.3f}  (paper: +0.41)")
print(f"raw r(delta, A_odd)  = {out['raw_r_A_odd']:+.3f}  (paper: -0.04)")


def partial_r(df, y, x, controls):
    Z = np.column_stack([np.ones(len(df))] + [df[c].values for c in controls])
    ry = df[y].values - Z @ np.linalg.lstsq(Z, df[y].values, rcond=None)[0]
    rx = df[x].values - Z @ np.linalg.lstsq(Z, df[x].values, rcond=None)[0]
    return float(np.corrcoef(ry, rx)[0, 1])


out["partial_A_even_given_all"] = partial_r(
    df, "delta", "A_even", ["A_odd", "cardinal", "n_units", "pr_full"])
print(f"A3 partial r(delta, A_even | A_odd, cardinal, n_units, pr_full) = "
      f"{out['partial_A_even_given_all']:+.3f}", flush=True)

formula = "delta ~ A_even + A_odd + cardinal + C(area)"

ols = smf.ols(formula, data=df).fit(
    cov_type="cluster", cov_kwds={"groups": df["session"]})
out["cluster_robust"] = {
    k: {"coef": float(ols.params[k]), "se": float(ols.bse[k]),
        "p": float(ols.pvalues[k])} for k in ("A_even", "A_odd")}
print(f"A1 cluster-robust: A_even = {ols.params['A_even']:+.3f} ± "
      f"{ols.bse['A_even']:.3f}, p = {ols.pvalues['A_even']:.2e} | "
      f"A_odd = {ols.params['A_odd']:+.3f} ± {ols.bse['A_odd']:.3f}, "
      f"p = {ols.pvalues['A_odd']:.2e}", flush=True)

mm = smf.mixedlm(formula, data=df, groups=df["session"]).fit(reml=True)
out["mixed_model"] = {
    k: {"coef": float(mm.params[k]), "se": float(mm.bse[k]),
        "p": float(mm.pvalues[k])} for k in ("A_even", "A_odd")}
print(f"A1 mixed model: A_even = {mm.params['A_even']:+.3f} ± "
      f"{mm.bse['A_even']:.3f}, p = {mm.pvalues['A_even']:.2e} | "
      f"A_odd = {mm.params['A_odd']:+.3f} ± {mm.bse['A_odd']:.3f}, "
      f"p = {mm.pvalues['A_odd']:.2e}", flush=True)

json.dump(out, open(OUT, "w"), indent=1)
print("DONE run39", flush=True)
