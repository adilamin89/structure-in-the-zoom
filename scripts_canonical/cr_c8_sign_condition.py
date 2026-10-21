"""C8 mining: what sets the sign of the class-mean part. At the full stimulus set the class-mean term of log PR is
c = log[(1 + b1/w1)^2 / (1 + (b2 + 2x)/w2)] (App K split; b1 = Tr B, w1 = Tr W, b2 = Tr B^2, w2 = Tr W^2, x = Tr WB). Without
x it is negative exactly when PR_W / PR_B > 1 + 2 w1/b1: the class means concentrate variance when the fluctuations are
far higher-dimensional than the means, relative to the means' strength. The ladder's class-mean part is the slope of
(declared - floor) class-mean terms, which starts at -c_floor and ends at 0, so its sign follows c at the full set.
PREDICTION (before the run): c < 0 in 8 of 8; PR_W/PR_B exceeds 1 + 2 w1/b1 in >= 6 of 8 (x deepens the rest);
x's share of the denominator 2x/(b2 + 2x) is reported. Out: data_canonical/cr_c8_sign_condition.json"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "raw"   # Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
rows = []
for fp in sorted(DATA.glob("gratings*.npy")):
    d = np.load(fp, allow_pickle=True).item()
    X = np.asarray(d["sresp"], np.float64).T; X /= X.std()
    phi = np.asarray(d["istim"], float).ravel()
    lab = np.clip(np.digitize(phi % (2 * np.pi), np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
    X -= X.mean(0)
    cm = np.stack([X[lab == c].mean(0) for c in range(8)]); p = np.bincount(lab, minlength=8) / len(lab)
    n = len(X); Wh = (X - cm[lab]) / np.sqrt(n); Bh = np.sqrt(p)[:, None] * cm
    w1, b1 = (Wh * Wh).sum(), (Bh * Bh).sum()
    w2 = ((Wh @ Wh.T) ** 2).sum(); b2 = ((Bh @ Bh.T) ** 2).sum(); x = ((Wh @ Bh.T) ** 2).sum()
    prW, prB = w1 ** 2 / w2, b1 ** 2 / b2
    c = np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2)); c0 = np.log((1 + b1 / w1) ** 2 / (1 + b2 / w2))
    r = {"name": fp.stem, "trB_trW": b1 / w1, "PR_W": prW, "PR_B": prB, "ratio": prW / prB, "threshold": 1 + 2 * w1 / b1,
         "c_full": c, "c_without_x": c0, "x_share": 2 * x / (b2 + 2 * x)}
    rows.append(r)
    print(f"{fp.stem[:28]:28s} TrB/TrW {r['trB_trW']:.3f} PR_W {prW:6.1f} PR_B {prB:4.2f} ratio {r['ratio']:6.1f} "
          f"thr {r['threshold']:5.1f} c {c:+.3f} (no x {c0:+.3f}) x-share {r['x_share']:.2f}", flush=True)
v = {"c_negative": int(sum(r["c_full"] < 0 for r in rows)), "inequality_holds": int(sum(r["ratio"] > r["threshold"] for r in rows)),
     "c_without_x_negative": int(sum(r["c_without_x"] < 0 for r in rows))}
print("VERDICT", v)
json.dump({"rows": rows, "verdict": v}, open(HERE.parent / "data_canonical" / "cr_c8_sign_condition.json", "w"), indent=1)


def model_side():
    """The calibrated model at s = 0 (shared modes, Sec 6 +0.080) and s = 1, same statistics."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run6", HERE / "run6_corotation_calibration.py")
    r6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r6)
    (n_stim, n_neur), bl, r_within, pr_within = r6.r2.measure_gt3()
    out = []
    for s in (0.0, 1.0):
        X = np.asarray(r6.make_partial(n_stim, 4000, bl, r_within, pr_within, s), np.float64); X -= X.mean(0)
        lab = np.asarray(bl); K = lab.max() + 1
        cm = np.stack([X[lab == c].mean(0) for c in range(K)]); p = np.bincount(lab, minlength=K) / len(lab)
        Wh = (X - cm[lab]) / np.sqrt(len(X)); Bh = np.sqrt(p)[:, None] * cm
        w1, b1 = (Wh * Wh).sum(), (Bh * Bh).sum()
        w2 = ((Wh @ Wh.T) ** 2).sum(); b2 = ((Bh @ Bh.T) ** 2).sum(); x = ((Wh @ Bh.T) ** 2).sum()
        r = {"s": s, "trB_trW": b1 / w1, "PR_W": w1 ** 2 / w2, "PR_B": b1 ** 2 / b2, "threshold": 1 + 2 * w1 / b1,
             "c_full": float(np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2)))}
        r["ratio"] = r["PR_W"] / r["PR_B"]; out.append(r)
        print(f"model s={s}: TrB/TrW {r['trB_trW']:.3f} PR_W {r['PR_W']:.1f} PR_B {r['PR_B']:.2f} ratio {r['ratio']:.1f} "
              f"thr {r['threshold']:.1f} c {r['c_full']:+.3f}", flush=True)
    return out


if __name__ == "__main__":
    res = json.load(open(HERE.parent / "data_canonical" / "cr_c8_sign_condition.json"))
    res["model"] = model_side()
    json.dump(res, open(HERE.parent / "data_canonical" / "cr_c8_sign_condition.json", "w"), indent=1)
