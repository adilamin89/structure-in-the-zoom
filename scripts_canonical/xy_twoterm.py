"""The two-term prediction of delta (twoterm_imaging.classmean_prediction + the mean-field sigma law), frozen and
imported read-only, applied to the XY lattice ensembles (regenerated exactly) and the spin waves.
Out: data_canonical/xy_twoterm.json"""
import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lattice_ground_truth as h  # noqa: E402
import xy_asymptote as xa  # noqa: E402
import xy_groundtruth as gt  # noqa: E402
import nematic_polar_delta as nem  # noqa: E402


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


B245 = _load("b245", "twoterm_imaging.py")
OM = _load("om", "overlap_identity_checks.py")


def predict(cfg, seed):
    _, _, psi = h.order_params(cfg)
    lab = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
    F = nem.features(cfg)
    F = F - F.mean(0)
    F /= F.std()
    b1, w1, b2, w2, x = xa.traces_fs(F, lab)
    cm = np.stack([F[lab == c].mean(0) for c in range(8)])
    T = dict(w1=w1, b1=b1, w2=w2, b2=b2, x=x, cm=cm, counts=np.bincount(lab, minlength=8))
    cnt = T["counts"]
    sizes = [int(cnt[:k].sum()) for k in B245.RUNGS]
    cm_pred, c_full, _ = B245.classmean_prediction(T, sizes)
    rng = np.random.default_rng(seed)
    Ra, Rb = [], []
    for c in range(8):
        R = F[lab == c] - F[lab == c].mean(0)
        p = rng.permutation(len(R))
        Ra.append(R[p[: len(R) // 2]])
        Rb.append(R[p[len(R) // 2:]])
    sig = OM.sigma(OM.split_half_K(Ra, Rb))
    mf = float(np.log(8 / (1 + 7 * sig)) / np.log(8))
    return {"pred_within": mf, "pred_class_mean": cm_pred, "pred_delta": mf + cm_pred, "sigma_H1": sig, "c_full": c_full}


def main():
    out = {"lattice": [], "spinwave": []}
    for f in sorted(glob.glob(str(h.OUTDIR / "xy_asymptote_L*_T*.json"))):
        art = json.load(open(f))
        L, T = art["L"], art["T"]
        for per, rows in art["cells"].items():
            for r in rows:
                cfg = xa.ensemble(L, T, int(per) * 8 // 5, r["ens"])
                p = predict(cfg, 600 + r["ens"])
                rec = {"L": L, "T": T, "per_class": int(per), "ens": r["ens"], **p,
                       "delta": r["delta"], "within": r["within"], "class_mean": r["class_mean"], "PR_class": r["PR_class"]}
                out["lattice"].append(rec)
                print(f"lattice L={L} T={T} {per}/class e{r['ens']}: pred d {p['pred_delta']:+.3f} (w {p['pred_within']:+.3f} "
                      f"m {p['pred_class_mean']:+.3f}) | meas d {r['delta']:+.3f} (w {r['within']:+.3f} m {r['class_mean']:+.3f})",
                      flush=True)
                json.dump(out, open(h.OUTDIR / "xy_twoterm.json", "w"), indent=1)
    gtd = json.load(open(h.OUTDIR / "xy_groundtruth.json"))
    for per in (25, 200):
        for s, r in enumerate(gtd[f"spinwave_{per}"]["rows"]):
            cfg = gt.spinwave(8 * per, 40_000 + 100 * per + s)
            p = predict(cfg, 700 + s)
            rec = {"per_class": per, "seed": s, **p, "delta": r["delta"], "within": r["within_part"],
                   "class_mean": r["class_mean_part"]}
            out["spinwave"].append(rec)
            print(f"spinwave {per}/class s{s}: pred d {p['pred_delta']:+.3f} (w {p['pred_within']:+.3f} m {p['pred_class_mean']:+.3f})"
                  f" | meas d {r['delta']:+.3f}", flush=True)
    allr = out["lattice"] + out["spinwave"]
    out["sign_agree"] = int(sum(np.sign(r["pred_delta"]) == np.sign(r["delta"]) for r in allr))
    out["n"] = len(allr)
    lat = out["lattice"]
    out["spearman_lattice"] = float(stats.spearmanr([r["pred_delta"] for r in lat], [r["delta"] for r in lat])[0])
    out["mean_err"] = {k: float(np.mean([r[f"pred_{k}"] - r[m] for r in allr]))
                       for k, m in (("within", "within"), ("class_mean", "class_mean"), ("delta", "delta"))}
    print("sign agreement", out["sign_agree"], "/", out["n"], "Spearman (lattice)", out["spearman_lattice"],
          "mean pred - meas", out["mean_err"], flush=True)
    json.dump(out, open(h.OUTDIR / "xy_twoterm.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
