"""C6 (Oct 20): why the calibrated model falls short of cortex's fluctuation overlap at 135 degrees. Reads only artifacts.

The model (run48 make_partial) builds each class's fluctuations as K shared random modes with each neuron's loading
rescaled by (1 - s) + s * the class's normalized mean. For delocalized random modes Tr(S_c S_d) is dominated by the
mean of the product of the two gain vectors, so at s = 1 the model's shape overlap Kn(dphi) should equal the square of
its class-mean correlation, built from (a, b2, c1, b4) without the l = 3 term. Compared with cortex: measured Kn against
the square of the measured class-mean correlation (all five terms).
Inputs: cr_model_overlap_calibration.json (model sweep, measured profile), multipole_harmonics_8dir.json.
Out: printed table + data_canonical/cr_model_overlap_square.json"""
import json
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data_canonical"
mh = {r["name"]: r["coef"] for r in json.load(open(DATA / "multipole_harmonics_8dir.json"))["rows"] if "coef" in r}
cal = json.load(open(DATA / "cr_model_overlap_calibration.json"))["rows"]
NAMES = {"GT1": "gratings_drifting_GT1_2019_04_12_1", "GT2": "gratings_drifting_GT2_2019_04_05_1", "GT3": "gratings_drifting_GT3_2019_04_05_1"}
d = np.radians([45, 90, 135, 180])
out = {}
for t, n in NAMES.items():
    c = mh[n]
    c_model = (c["a"] + c["b2"] * np.cos(2 * d) + c["c1"] * np.cos(d) + c["b4"] * np.cos(4 * d)) / (c["a"] + c["b2"] + c["c1"] + c["b4"])
    c_data = c["a"] + c["b2"] * np.cos(2 * d) + c["c1"] * np.cos(d) + c["c3"] * np.cos(3 * d) + c["b4"] * np.cos(4 * d)
    kn_model = np.array(list(cal[t]["sweep"]["1.0"]["profile"].values()))
    kn_data = np.array(list(cal[t]["measured_profile"].values()))
    out[t] = {"kn_model_s1": kn_model.tolist(), "c_model_sq": (c_model ** 2).tolist(), "kn_data": kn_data.tolist(),
              "c_data_sq": (c_data ** 2).tolist(), "ratio_data": (kn_data / c_data ** 2).tolist(),
              "max_abs_model_minus_csq": float(np.abs(kn_model - c_model ** 2).max())}
    print(f"{t}: model Kn {kn_model.round(3)} vs C_model^2 {(c_model ** 2).round(3)} (max diff {out[t]['max_abs_model_minus_csq']:.3f}) | "
          f"data Kn {kn_data.round(3)} vs C^2 {(c_data ** 2).round(3)}, ratio {(kn_data / c_data ** 2).round(2)}")
r = np.array([out[t]["ratio_data"] for t in out])
print(f"CHECK model = C^2 within {max(o['max_abs_model_minus_csq'] for o in out.values()):.3f}; data/C^2 range {r.min():.1f}-{r.max():.1f};"
      f" at 90-135 deg {r[:, 1:3].min():.1f}-{r[:, 1:3].max():.1f}, at 180 deg {r[:, 3].min():.1f}-{r[:, 3].max():.1f}")
json.dump(out, open(DATA / "cr_model_overlap_square.json", "w"), indent=1)
