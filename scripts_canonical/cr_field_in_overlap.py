"""Is the fluctuation overlap that cortex has beyond C(dphi)^2 a single shared population field?

Fact (cr_model_overlap_square.py): the calibrated model's shape overlap Kn(dphi) equals the square of its class-mean
correlation by construction; cortex's Kn is 1.1-5.2 x C^2, most at 90-135 deg. Earlier analyses found a
population field: one slow, shared fluctuation mode common to all neurons (the top mode; in mouse V1 spontaneous activity
one field suffices). A field present in every stimulus class adds the same fluctuation direction to all classes regardless
of how alike their means are, raising Kn most where C^2 is small.
Test (three drifting + all eight grating recordings): the field direction u = top eigenvector of the pooled within-class
covariance on half A of the presentations (alternate trials within each class); project u out of every presentation in
half B and recompute the split-half shape-overlap profile (8 classes, 45..180 deg) on half B's own two halves; compare with
C^2 (C from multipole_harmonics_8dir). Also: share of within-class variance on u, and |cos| of each class's own top
within-class eigenvector with u (is the field common to the classes?).
EXPECTATIONS (written before the run):
F1: removing the field lowers Kn at 90 and 135 deg by more (relative) than at 45 deg in >= 6 of 8 recordings.
F2: after removal the ratio Kn / C^2 at 90-135 deg falls by at least a third (median over recordings).
F3: each class's top within-class mode aligns with the pooled field, |cos| >= 0.7, in >= 6 of 8 recordings (median class).
A miss of F1/F2 means cortex's extra overlap is spread over many shared modes (mouse V1 has ~59 above the edge), not one
field; then the follow-up model needs the measured overlap kernel itself.
Out: data_canonical/cr_field_in_overlap.json"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent / "raw"
OUT = DATA / "cr_field_in_overlap.json"


def bins8(phi):
    return np.clip(np.digitize(phi % (2 * np.pi), np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)


def kn_profile(X, bl):
    A, B = [], []
    for c in range(8):
        idx = np.where(bl == c)[0]; a, b = idx[0::2], idx[1::2]
        A.append(X[a] - X[a].mean(0)); B.append(X[b] - X[b].mean(0))
    tr = lambda Z: float((Z * Z).sum() / len(Z))
    K = np.zeros((8, 8))
    for c in range(8):
        for d in range(8):
            ab = A[c] @ B[d].T; ba = B[c] @ A[d].T
            K[c, d] = 0.5 * ((ab * ab).sum() / (len(A[c]) * len(B[d]) * tr(A[c]) * tr(B[d]))
                             + (ba * ba).sum() / (len(B[c]) * len(A[d]) * tr(B[c]) * tr(A[d])))
    Kn = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))
    sep = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
    return np.array([Kn[sep == k].mean() for k in range(1, 5)])


def top_mode(Z, k=1):
    """Top right singular vector(s) of a centred presentations x neurons block."""
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return vt[:k]


def main():
    mh = {r["name"]: r["coef"] for r in json.load(open(DATA / "multipole_harmonics_8dir.json"))["rows"] if "coef" in r}
    d = np.radians([45, 90, 135, 180]); rows = []
    for fp in sorted(RAW.glob("gratings*.npy")):
        dat = np.load(fp, allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        bl = bins8(np.asarray(dat["istim"], float).ravel()); del dat
        half = np.zeros(len(bl), int)
        for c in range(8):
            idx = np.where(bl == c)[0]; half[idx[1::2]] = 1
        XA, bA, XB, bB = X[half == 0], bl[half == 0], X[half == 1], bl[half == 1]
        WA = np.concatenate([XA[bA == c] - XA[bA == c].mean(0) for c in range(8)])
        u = top_mode(WA)[0]
        share = float(((WA @ u) ** 2).sum() / (WA * WA).sum())
        cos = [float(abs(top_mode(XA[bA == c] - XA[bA == c].mean(0))[0] @ u)) for c in range(8)]
        kn_raw = kn_profile(XB, bB)
        XBr = XB - np.outer(XB @ u, u)
        kn_rem = kn_profile(XBr, bB)
        c = mh[fp.stem]; C = c["a"] + c["b2"] * np.cos(2 * d) + c["c1"] * np.cos(d) + c["c3"] * np.cos(3 * d) + c["b4"] * np.cos(4 * d)
        row = {"name": fp.stem, "field_share_of_within": share, "class_top_cos_median": float(np.median(cos)),
               "kn_raw": kn_raw.tolist(), "kn_field_removed": kn_rem.tolist(), "c_sq": (C ** 2).tolist(),
               "ratio_raw": (kn_raw / C ** 2).tolist(), "ratio_removed": (kn_rem / C ** 2).tolist()}
        rows.append(row)
        print(f"{fp.stem[9:30]:22s} field share {share:.3f} class-top |cos| median {np.median(cos):.2f} | Kn raw {kn_raw.round(3)}"
              f" -> removed {kn_rem.round(3)} | ratio 90-135 {np.mean(row['ratio_raw'][1:3]):.2f} -> {np.mean(row['ratio_removed'][1:3]):.2f}", flush=True)
    f1 = sum(np.mean(1 - np.array(r["kn_field_removed"][1:3]) / np.array(r["kn_raw"][1:3])) > (1 - r["kn_field_removed"][0] / r["kn_raw"][0]) for r in rows)
    f2 = float(np.median([1 - np.mean(r["ratio_removed"][1:3]) / np.mean(r["ratio_raw"][1:3]) for r in rows]))
    f3 = sum(r["class_top_cos_median"] >= 0.7 for r in rows)
    v = {"F1_count": int(f1), "F2_median_relative_drop_ratio_90_135": f2, "F3_count": int(f3)}
    print("VERDICT", json.dumps(v), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
