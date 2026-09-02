"""Harmonic decomposition of the class-mean signal correlation C(dphi) over the
eight direction classes, for every Stringer grating recording (S74, 2026-09-02).

WHY: the paper quotes a three-term fit (a, b cos2, c cos) with R^2 = 0.90, a
cos4 term b4 = 0.07 (R^2 0.97), a cos3 term c3 = 0.064, and the even/odd
variance ratio (b^2 + b4^2)/(c^2 + c3^2) = 4.8 on GT3. Only a, b, c had a
stored artifact. This script closes the provenance gap and tests the
per-recording claim.

DEFINITION (reproduces stringer_cos2theta_fit.json to 4 decimals): responses
sresp (stimuli x neurons); direction classes = 8 equal bins of istim on
[0, 2pi); class-mean vectors over neurons; C_ij = Pearson correlation between
class-mean vectors i and j across neurons; circular profile C(k * 45 deg) =
mean over i of C_{i, i+k}. With eight classes the cosines cos(0..4 dphi) are
the complete symmetric Fourier basis on this grid, so the five coefficients
(a, c1, b2, c3, b4) are the exact discrete Fourier transform of the profile
(b4 is the Nyquist term, weight 1/8); least squares over all 64 matrix
entries returns the same coefficients because the harmonics are orthogonal
there. R^2 values are over the 64 entries.

REGISTERED EXPECTATIONS (written before the run):
  E1: GT3 drifting reproduces the stored fit: a 0.3947, b2 0.3298, c1 0.1402
      (|diff| < 1e-3) and the reviewed b4 = 0.0716, c3 = 0.0638; even/odd 4.80.
  E2: paper claim "quadrupole dominance holds in each of the ten recordings":
      |b2| > |c1| in all 8 grating recordings (the 2 static sessions have a
      4-degree stimulus span and degenerate direction bins; excluded).
      Exploratory pass on 2026-09-02 found the three localized-grating
      recordings dipole-dominant (b2/c1 0.50-0.78); E2 is therefore expected
      to FAIL for those three, and the paper text will be corrected.
  E3: entry-coherence sign: the exact interpolant gives C(22.5) > C(180) in
      all 8 gratings (consistent with the 8/8 sequential > paired reversal).

Out: ../data_canonical/multipole_harmonics_8dir.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = HERE.parent / "data_canonical" / "multipole_harmonics_8dir.json"
NB = 8


def load(f):
    dat = np.load(f, allow_pickle=True).item()
    return dat["sresp"].T.astype(np.float64), np.asarray(dat["istim"]).ravel() % (2 * np.pi)


def profile_and_dft(X, phi):
    b = np.floor(phi / (2 * np.pi) * NB).astype(int) % NB
    counts = np.bincount(b, minlength=NB)
    if counts.min() < 10:
        return None, counts, None
    M = np.stack([X[b == k].mean(0) for k in range(NB)])
    Cm = np.corrcoef(M)
    prof = np.array([np.mean([Cm[i, (i + k) % NB] for i in range(NB)]) for k in range(NB)])
    ang = 2 * np.pi * np.arange(NB) / NB
    coef = {"a": float(prof.mean()),
            "c1": float(2 / NB * np.sum(prof * np.cos(ang))),
            "b2": float(2 / NB * np.sum(prof * np.cos(2 * ang))),
            "c3": float(2 / NB * np.sum(prof * np.cos(3 * ang))),
            "b4": float(1 / NB * np.sum(prof * np.cos(4 * ang)))}
    return prof, counts, (Cm, coef)


def r2_over_matrix(Cm, coef, terms):
    idx = np.arange(NB)
    D = 2 * np.pi * ((idx[None, :] - idx[:, None]) % NB) / NB
    basis = {"a": np.ones_like(D), "c1": np.cos(D), "b2": np.cos(2 * D), "c3": np.cos(3 * D), "b4": np.cos(4 * D)}
    pred = sum(coef[t] * basis[t] for t in terms)
    y = Cm.ravel()
    return float(1 - ((y - pred.ravel()) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def C_at(coef, deg):
    d = np.radians(deg)
    return float(coef["a"] + coef["c1"] * np.cos(d) + coef["b2"] * np.cos(2 * d)
                 + coef["c3"] * np.cos(3 * d) + coef["b4"] * np.cos(4 * d))


def main():
    rows = []
    for f in sorted(RAW.glob("*.npy")):
        X, phi = load(f)
        prof, counts, rest = profile_and_dft(X, phi)
        row = {"name": f.stem, "n_neurons": int(X.shape[1]), "n_stimuli": int(X.shape[0]),
               "bin_counts": counts.tolist()}
        if prof is None:
            row["status"] = "degenerate_bins"
            rows.append(row)
            print(f"  {f.stem[:34]:34s} degenerate bins {counts.tolist()}")
            continue
        Cm, coef = rest
        eo = (coef["b2"] ** 2 + coef["b4"] ** 2) / (coef["c1"] ** 2 + coef["c3"] ** 2)
        row.update({"status": "ok", "profile_deg": [0, 45, 90, 135, 180], "profile": prof[:5].tolist(),
                    "coef": coef, "b2_over_c1": coef["b2"] / coef["c1"], "even_over_odd_variance": eo,
                    "r2_3term": r2_over_matrix(Cm, coef, ["a", "b2", "c1"]),
                    "r2_4term": r2_over_matrix(Cm, coef, ["a", "b2", "c1", "b4"]),
                    "r2_5term": r2_over_matrix(Cm, coef, ["a", "b2", "c1", "b4", "c3"]),
                    "C_22p5_interp": C_at(coef, 22.5), "C_180": C_at(coef, 180.0),
                    "quadrupole_dominant": bool(abs(coef["b2"]) > abs(coef["c1"]))})
        rows.append(row)
        print(f"  {f.stem[:34]:34s} a {coef['a']:.4f} c1 {coef['c1']:+.4f} b2 {coef['b2']:+.4f} c3 {coef['c3']:+.4f} "
              f"b4 {coef['b4']:+.4f} | b2/c1 {row['b2_over_c1']:.2f} even/odd {eo:.2f} | R2 {row['r2_3term']:.3f}/"
              f"{row['r2_4term']:.3f}/{row['r2_5term']:.3f} | C22.5 {row['C_22p5_interp']:.3f} C180 {row['C_180']:.3f} "
              f"| quad>dip {row['quadrupole_dominant']}")
    ok = [r for r in rows if r["status"] == "ok"]
    gt3 = next(r for r in ok if "drifting_GT3" in r["name"])
    e1 = (abs(gt3["coef"]["a"] - 0.3947) < 1e-3 and abs(gt3["coef"]["b2"] - 0.3298) < 1e-3
          and abs(gt3["coef"]["c1"] - 0.1402) < 1e-3 and abs(gt3["coef"]["b4"] - 0.0716) < 1e-3
          and abs(gt3["coef"]["c3"] - 0.0638) < 1e-3)
    e2_n = sum(r["quadrupole_dominant"] for r in ok)
    e3_n = sum(r["C_22p5_interp"] > r["C_180"] for r in ok)
    verdict = {"E1_gt3_reproduces_stored_fit": bool(e1),
               "E2_quadrupole_dominant_count": f"{e2_n}/{len(ok)}",
               "E2_pass_all": bool(e2_n == len(ok)),
               "E2_failing_recordings": [r["name"] for r in ok if not r["quadrupole_dominant"]],
               "E3_adjacent_more_coherent_count": f"{e3_n}/{len(ok)}"}
    print("VERDICT", json.dumps(verdict, indent=1))
    json.dump({"definition": "8 equal direction bins of istim; Pearson correlation of class-mean vectors across "
               "neurons; circular profile mean over base class; exact DFT (b4 = Nyquist, weight 1/8); "
               "R2 over all 64 matrix entries", "rows": rows, "verdict": verdict},
              open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
