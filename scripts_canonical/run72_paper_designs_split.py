"""Run 72 (2026-10-01): the paper's designs, exactly, for all neurons and tuned neurons, with the noise-side / probe split.
Registration: ../data_canonical/run72_REGISTERED.md. Design A = orientation_zoom's direction ladder (8 bins; rungs
1,2,3,4,6,8). Design B = accumulation_order.py (16 bins; 8 rungs of 2 bins; paired (k,k+8) vs sequential (2k,2k+1); a
floor per ladder). Split per subset: log PR = log(w1^2/w2) [noise side] + log((1+b1/w1)^2/(1+(b2+2x)/w2)) [probe]."""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
RAW = HERE.parent.parent.parent / "basin_memory" / "data" / "stringer_v1" / "natimg"
OUT = DATA / "run72_paper_designs_split.json"
SUPP = HERE.parent.parent / "neurreps_supplement" / "data" / "antipodal_order.json"
spec = importlib.util.spec_from_file_location("run63", HERE / "run63_deficit_decomposition.py")
r63 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r63)
N_NULL = 10


def logparts(Y, lab):
    b1, w1, b2, w2, x = r63.traces(Y, lab)
    return {"logpr": np.log((b1 + w1) ** 2 / (b2 + w2 + 2 * x)), "noise": np.log(w1 ** 2 / w2),
            "probe": np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2)), "b1": b1}


def slope(sizes, ys):
    return float(np.polyfit(np.log(sizes), ys, 1)[0])


def ladder(Xt, lab, rung_sets, rng):
    """rung_sets: list of index arrays (cumulative). Returns per-rung ladder parts, floor-mean parts, slopes, delta."""
    lad, flo, sizes = [], [], []
    for sel in rung_sets:
        sizes.append(len(sel)); lad.append(logparts(Xt[sel], lab[sel]))
        acc = {k: 0.0 for k in lad[-1]}
        for _ in range(N_NULL):
            idx = rng.choice(len(Xt), len(sel), replace=False); p = logparts(Xt[idx], lab[idx])
            for k in acc: acc[k] += p[k] / N_NULL
        flo.append(acc)
    out = {"sizes": sizes, "ladder": lad, "floor": flo}
    for k in ("logpr", "noise", "probe"):
        out[f"slope_{k}"] = slope(sizes, [l[k] - f[k] for l, f in zip(lad, flo)])
    out["delta"] = out["slope_logpr"]
    return out


def bins(phi, nb):
    return np.clip(np.digitize(phi % (2 * np.pi), np.linspace(0, 2 * np.pi, nb + 1)) - 1, 0, nb - 1)


def tuned_mask(Xt, lab8):
    n = len(lab8); M = np.stack([Xt[lab8 == k].mean(0) for k in range(8)]); g = Xt.mean(0); c = np.bincount(lab8, minlength=8)
    ssb = sum(c[k] * (M[k] - g) ** 2 for k in range(8)); ssw = sum(((Xt[lab8 == k] - M[k]) ** 2).sum(0) for k in range(8))
    F = (ssb / 7) / (ssw / (n - 8) + 1e-12)
    return (1 - stats.f.cdf(F, 7, n - 8)) < 0.01


def harmonic_prediction(Xt, lab16, sets):
    """Predicted Tr B per subset of 16 bins from per-harmonic powers of the same population's class means."""
    th = 2 * np.pi * np.arange(16) / 16
    M = np.stack([Xt[lab16 == j].mean(0) for j in range(16)])          # 16 x N
    C = np.fft.fft(M, axis=0) / 16                                       # c_l per neuron
    P = (np.abs(C) ** 2).sum(1)                                          # power at each l = 0..15
    pred = []
    for S in sets:
        pred.append(float(sum(P[l] * (1 - abs(np.exp(1j * l * th[S]).mean()) ** 2) for l in range(1, 16))))
    return pred


def main():
    t0 = time.time()
    oz = {r["name"]: r["delta"] for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    supp = {r["name"]: r for r in json.load(open(SUPP))["rows"]}
    out = {"registration": "run72_REGISTERED.md", "rows": {}}
    for fp in sorted(RAW.glob("gratings*.npy")):
        name = fp.stem; tag = r63.short(name)
        dat = np.load(fp, allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        phi = np.asarray(dat["istim"], float).ravel(); del dat
        lab8, lab16 = bins(phi, 8), bins(phi, 16)
        pops = {"all": np.arange(X.shape[1]), "tuned": np.where(tuned_mask(X, lab8))[0]}
        row = {"name": name, "n_neurons": {k: int(len(v)) for k, v in pops.items()}}
        m8 = [np.where(lab8 == b)[0] for b in range(8)]; m16 = [np.where(lab16 == b)[0] for b in range(16)]
        setsA = [np.concatenate(m8[:c]) for c in (1, 2, 3, 4, 6, 8)]
        binsP = [[j for k in range(r + 1) for j in (k, k + 8)] for r in range(8)]
        binsS = [[j for k in range(r + 1) for j in (2 * k, 2 * k + 1)] for r in range(8)]
        for pname, cols in pops.items():
            Xp = np.ascontiguousarray(X[:, cols])
            A = ladder(Xp, lab8, setsA, np.random.default_rng(42))
            Bp = ladder(Xp, lab16, [np.concatenate([m16[j] for j in s]) for s in binsP], np.random.default_rng(42))
            Bs = ladder(Xp, lab16, [np.concatenate([m16[j] for j in s]) for s in binsS], np.random.default_rng(43))
            predP, predS = harmonic_prediction(Xp, lab16, binsP), harmonic_prediction(Xp, lab16, binsS)
            row[pname] = {"A": A, "B_paired": Bp, "B_sequential": Bs, "order_effect": Bp["delta"] - Bs["delta"],
                          "order_noise": Bp["slope_noise"] - Bs["slope_noise"], "order_probe": Bp["slope_probe"] - Bs["slope_probe"],
                          "TrB_seq_minus_paired": [Bs["ladder"][i]["b1"] - Bp["ladder"][i]["b1"] for i in range(4)],
                          "pred_seq_minus_paired": [predS[i] - predP[i] for i in range(4)]}
            del Xp
            r = row[pname]
            print(f"[{tag}] {pname:5s} N={len(cols)} | A delta {A['delta']:+.3f} (paper {oz.get(name, float('nan')):+.3f}) noise {A['slope_noise']:+.3f} probe {A['slope_probe']:+.3f} | "
                  f"B order {r['order_effect']:+.3f} (paper {supp[name]['order_effect']:+.3f}) = noise {r['order_noise']:+.3f} + probe {r['order_probe']:+.3f} | {time.time()-t0:.0f}s", flush=True)
        del X
        out["rows"][tag] = row
        json.dump(out, open(OUT, "w"), indent=1, default=float)
    R = out["rows"]; tags = list(R)
    v = {"R1_A_within_0.03": int(sum(abs(R[t]["all"]["A"]["delta"] - oz[R[t]["name"]]) < 0.03 for t in tags)),
         "R1_B_negative": int(sum(R[t]["all"]["order_effect"] < 0 for t in tags)),
         "R1_B_within_0.03": int(sum(abs(R[t]["all"]["order_effect"] - supp[R[t]["name"]]["order_effect"]) < 0.03 for t in tags))}
    for p in ("all", "tuned"):
        v[f"P1_{p}"] = int(sum(R[t][p]["A"]["slope_noise"] > R[t][p]["A"]["delta"] and R[t][p]["A"]["slope_probe"] < 0 for t in tags))
        v[f"P3_{p}"] = int(sum(np.sign(a) == np.sign(b) for t in tags for a, b in zip(R[t][p]["TrB_seq_minus_paired"], R[t][p]["pred_seq_minus_paired"])))
    v["R1"] = bool(v["R1_A_within_0.03"] == 8 and v["R1_B_negative"] == 8 and v["R1_B_within_0.03"] >= 7)
    v["P1"] = bool(v["P1_all"] >= 7 and v["P1_tuned"] >= 7); v["P3"] = bool(v["P3_all"] >= 26 and v["P3_tuned"] >= 26)
    out["verdict"] = v; print("VERDICT", json.dumps(v), flush=True)
    json.dump(out, open(OUT, "w"), indent=1, default=float); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
