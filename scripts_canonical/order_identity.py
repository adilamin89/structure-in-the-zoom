"""Is the order law (origin-averaged within-class order effect vs the
adjacent-minus-antipodal fluctuation overlap dK, r = -0.997 over the eight grating recordings, App E) an identity?

Steps and their status (stated in the registration): log PR(W_A) from the class overlaps (tr_c = Tr S_c, T_cd =
Tr S_c S_d) is an exact algebraic identity (overlap_identity_checks.within_curve); the origin average of the identity's order
effect is a function of the whole 16 x 16 overlap matrix; dK = Kn(22.5 deg) - Kn(180 deg) is one scalar of it. Tests:
 (a) held-out: temporally blocked halves (first / second half of the session's presentations); from half A, the scalar
     dK_A (with a leave-one-recording-out linear map fitted across recordings) and the full-kernel identity (no fit)
     each predict half B's origin-averaged within-class order effect (identity on B); both directions.
 (b) constrained-kernel challenge: transforms of each recording's empirical class covariances that are computable from
     the overlaps: convex circulant mixtures S'_c = sum_e A_ce S_e (T' = A T A^T, tr' = A tr), and random relabelings of the
     16 classes; for each transform dK' and the identity's origin-averaged order effect; at matched dK' (bins of 0.02) the
     spread of the effect and whether opposite signs occur.
Out: data_canonical/order_identity.json"""
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DC = HERE.parent / "data_canonical"


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load("m", "overlap_identity_checks.py")
FI = load("fi", "fluctuation_overlap_profile.py")
SEP = np.minimum(np.abs(np.subtract.outer(np.arange(16), np.arange(16))),
                 16 - np.abs(np.subtract.outer(np.arange(16), np.arange(16))))


def oe_mean(tr, T, counts):
    return float(FI.oe_from(lambda rungs: M.within_curve(tr, T, rungs, counts), counts).mean())


def dK_of(tr, T):
    K = M.K_of(tr, tr, T)
    kd = np.diag(K)
    Kn = K / np.sqrt(np.outer(kd, kd))
    return float(Kn[SEP == 1].mean() - Kn[SEP == 8].mean())


def per_recording(fp):
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel() % (2 * np.pi)
    del dat
    G = (X @ X.T).astype(np.float64)
    lab16 = M.bins(phi, 16)
    idx = [np.where(lab16 == c)[0] for c in range(16)]
    counts = np.array([len(i) for i in idx], float)
    tr, _, T = M.overlaps(G, idx)
    rec = {"name": fp.stem, "dK": dK_of(tr, T), "oe_identity_full": oe_mean(tr, T, counts)}
    n = len(lab16)
    halves = {}
    for h, sel in (("A", np.arange(n) < n // 2), ("B", np.arange(n) >= n // 2)):
        ih = [np.where((lab16 == c) & sel)[0] for c in range(16)]
        ch = np.array([len(i) for i in ih], float)
        t_h, _, T_h = M.overlaps(G, ih)
        halves[h] = {"dK": dK_of(t_h, T_h), "oe": oe_mean(t_h, T_h, ch), "counts": ch.tolist()}
    rec["halves"] = halves
    # (b) transforms
    rng = np.random.default_rng(16)
    cloud = []
    for s in range(1, 9):
        Ps = np.roll(np.eye(16), s, axis=1)
        for a in (0.1, 0.2, 0.3, 0.4, 0.5):
            A = (1 - a) * np.eye(16) + a * 0.5 * (Ps + Ps.T)
            cloud.append({"kind": f"mix_s{s}_a{a}", "dK": dK_of(A @ tr, A @ T @ A.T), "oe": oe_mean(A @ tr, A @ T @ A.T, counts)})
    for k in range(40):
        p = rng.permutation(16)
        cloud.append({"kind": f"relabel{k}", "dK": dK_of(tr[p], T[np.ix_(p, p)]), "oe": oe_mean(tr[p], T[np.ix_(p, p)], counts)})
    rec["cloud"] = cloud
    print(f"{fp.stem[:34]:34s} dK {rec['dK']:+.3f} oe_full {rec['oe_identity_full']:+.4f} | A dK {halves['A']['dK']:+.3f} "
          f"oe {halves['A']['oe']:+.4f} | B dK {halves['B']['dK']:+.3f} oe {halves['B']['oe']:+.4f}", flush=True)
    return rec


def main():
    ref = {r["name"]: r for r in json.load(open(DC / "fluctuation_overlap_profile.json"))["rows"]}
    files = sorted(M.RAW.glob("gratings*.npy"))
    rows = [per_recording(fp) for fp in files]
    for r in rows:   # reproduction of the paper's dK
        assert abs(r["dK"] - ref[r["name"]]["dK"]) < 1e-6, (r["name"], r["dK"], ref[r["name"]]["dK"])
    out = {"rows": rows}
    # (a) held-out predictions, both directions
    errs = {"full": [], "scalar": []}
    for src, tgt in (("A", "B"), ("B", "A")):
        x = np.array([r["halves"][src]["dK"] for r in rows])
        y = np.array([r["halves"][tgt]["oe"] for r in rows])
        full = np.array([r["halves"][src]["oe"] for r in rows])
        for i in range(len(rows)):
            m = np.arange(len(rows)) != i
            b = np.polyfit(x[m], y[m], 1)
            errs["scalar"].append(abs(np.polyval(b, x[i]) - y[i]))
            errs["full"].append(abs(full[i] - y[i]))
        out[f"r_full_{src}{tgt}"] = float(np.corrcoef(full, y)[0, 1])
        out[f"r_scalar_{src}{tgt}"] = float(np.corrcoef(x, y)[0, 1])
    out["median_abs_err"] = {k: float(np.median(v)) for k, v in errs.items()}
    out["errors"] = {k: [float(e) for e in v] for k, v in errs.items()}
    # (b) matched-dK spread over all recordings' transforms
    cl = [(c["dK"], c["oe"]) for r in rows for c in r["cloud"]] + [(r["dK"], r["oe_identity_full"]) for r in rows]
    cl = np.array(cl)
    bins = np.floor(cl[:, 0] / 0.02)
    spreads = []
    for b in np.unique(bins):
        v = cl[bins == b, 1]
        if len(v) >= 3:
            spreads.append({"dK_bin": float(b * 0.02), "n": int(len(v)), "oe_min": float(v.min()), "oe_max": float(v.max()),
                            "opposite_signs": bool(v.min() < 0 < v.max())})
    out["matched_dK"] = spreads
    out["max_spread_at_matched_dK"] = float(max(s["oe_max"] - s["oe_min"] for s in spreads))
    out["n_bins_opposite_sign"] = int(sum(s["opposite_signs"] for s in spreads))
    out["cloud_r"] = float(np.corrcoef(cl[:, 0], cl[:, 1])[0, 1])
    print("held-out median |err| full vs scalar:", out["median_abs_err"], {k: v for k, v in out.items() if k.startswith("r_")})
    print("matched-dK: max spread", out["max_spread_at_matched_dK"], "bins with opposite signs", out["n_bins_opposite_sign"],
          "of", len(spreads), "cloud r", out["cloud_r"])
    json.dump(out, open(DC / "order_identity.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
