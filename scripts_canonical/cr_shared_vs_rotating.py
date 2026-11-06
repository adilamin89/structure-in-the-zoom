"""Two ways for classes to share fluctuation structure. Which one cortex uses, per grating recording.

(a) SHARED MODES, REWEIGHTED (common principal components, Flury 1984): every class uses one eigenbasis U and sets its own
    mode variances, Sigma_c = U Lambda_c U^T.
(b) ROTATING MODES: each class's modes turn away from the common basis (off-diagonal Sigma_c in U).
(The camera-ready model is a third construction, neuron-wise gain on shared random modes, whose overlap is C^2 by
construction: cr_model_overlap_square.py.)
Matrix construction: basis V = top R = 300 right singular vectors of half A's pooled within-class residuals (alternate
trials within each class); every class's residuals in both halves projected on V (R x R covariances, the tensor
S[c, half]); U = eigenvectors of the class-and-half mean covariance. Overlap from full matrices,
K_full(c, d) = Tr(S_c^A S_d^B) / (Tr S_c^A Tr S_d^B), and from diagonals in U only (way a),
K_cpc(c, d) = sum_k (U^T S_c^A U)_kk (U^T S_d^B U)_kk / (same traces); symmetrized over halves; shape-normalized Kn profiles at
45..180 deg. Truncation check: K_full in V against the full-space split-half profile of cr_model_overlap_calibration /
cr_field_in_overlap (same estimator).
REGISTERED (before the run), the decision rule: way (a) carries the overlap if the CPC profile is within 0.03 of the full
profile at every separation in >= 6 of 8 recordings; way (b) is needed where it misses by > 0.05 at any of 90-180 deg.
Also reported: the share of the profile's adjacent-minus-antipodal excess (dK at 8 classes) that way (a) reproduces.
Out: data_canonical/cr_shared_vs_rotating.json"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent / "raw"
OUT = DATA / "cr_shared_vs_rotating.json"
R = 300
SEP = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))


def profile(K):
    Kn = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))
    return np.array([Kn[SEP == k].mean() for k in range(1, 5)])


def main():
    rows = []
    for fp in sorted(RAW.glob("gratings*.npy")):
        dat = np.load(fp, allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        bl = np.clip(np.digitize(np.asarray(dat["istim"], float).ravel() % (2 * np.pi), np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7); del dat
        idx = [np.where(bl == c)[0] for c in range(8)]
        res = {h: [X[i[h::2]] - X[i[h::2]].mean(0) for i in idx] for h in (0, 1)}
        _, sv, vt = np.linalg.svd(np.concatenate(res[0]), full_matrices=False)
        V = vt[:R].T                                                    # N x R
        cap = float((sv[:R] ** 2).sum() / (sv ** 2).sum())
        S = np.stack([np.stack([(Z @ V).T @ (Z @ V) / len(Z) for Z in res[h]]) for h in (0, 1)])   # (half, class, R, R)
        U = np.linalg.eigh(S.mean((0, 1)))[1]
        D = np.einsum("rk,hcrs,sk->hck", U, S, U)                       # diagonals in the common basis
        tr = np.einsum("hcrr->hc", S)
        Kf = 0.5 * (np.einsum("crs,dsr->cd", S[0], S[1]) + np.einsum("crs,dsr->cd", S[1], S[0])) / np.outer(tr[0], tr[1])
        Kc = 0.5 * (np.einsum("ck,dk->cd", D[0], D[1]) + np.einsum("ck,dk->cd", D[1], D[0])) / np.outer(tr[0], tr[1])
        pf, pc = profile(Kf), profile(Kc)
        dk_f, dk_c = pf[0] - pf[3], pc[0] - pc[3]
        rot_share = float(1 - (np.einsum("hck,hck->hc", D, D).sum() / np.einsum("hcrs,hcrs->hc", S, S).sum()))
        row = {"name": fp.stem, "variance_captured_R": cap, "kn_full": pf.tolist(), "kn_cpc": pc.tolist(),
               "max_abs_gap": float(np.abs(pf - pc).max()), "gap_90_180": float(np.abs(pf - pc)[1:].max()),
               "dK_full": float(dk_f), "dK_cpc": float(dk_c), "offdiag_share_of_TrS2": rot_share}
        rows.append(row)
        print(f"{fp.stem[9:30]:22s} captured {cap:.2f} | Kn full {pf.round(3)} cpc {pc.round(3)} | max gap {row['max_abs_gap']:.3f}"
              f" | dK full {dk_f:+.3f} cpc {dk_c:+.3f} | off-diagonal share of Tr S^2 {rot_share:.2f}", flush=True)
    v = {"way_a_within_0.03_count": int(sum(r["max_abs_gap"] <= 0.03 for r in rows)),
         "way_b_needed_count": int(sum(r["gap_90_180"] > 0.05 for r in rows))}
    print("VERDICT", json.dumps(v), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
