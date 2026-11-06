"""Score the lattice ground truth from lattice_ground_truth_*.json; every criterion printed before its verdict.
Out: data_canonical/lattice_ground_truth_score.json"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

D = Path(__file__).resolve().parent.parent / "data_canonical"


def ising():
    rows = {}
    for f in sorted(glob.glob(str(D / "lattice_ground_truth_ising_L*.json"))):
        r = json.load(open(f))
        rows[(r["L"], round(r["t_over_tc"], 2))] = r
    out = {"points": {}}
    for (L, t), r in sorted(rows.items()):
        s = r["summary"]["centered_absm"]
        v = np.array([c["second"]["centered_absm"] for c in r["chains"]])
        ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=v.std(ddof=1) / np.sqrt(len(v)))
        out["points"][f"L{L}_t{t}"] = {"mean": s["mean"], "se": s["se"], "ci95": list(ci), "n_neg": s["n_neg"],
                                        "n": len(v), "first_half_mean": s["first_half_mean"], "binder_U": r["binder_U"],
                                        "uncentered_signed": r["summary"]["uncentered_signed"]["mean"]}
        print(f"Ising L={L} t={t}: centered|m| {s['mean']:+.4f} ± {s['se']:.4f} CI [{ci[0]:+.3f},{ci[1]:+.3f}] "
              f"neg {s['n_neg']}/{len(v)} (first half {s['first_half_mean']:+.4f}) U={r['binder_U']:.4f} "
              f"| uncentered signed {r['summary']['uncentered_signed']['mean']:+.4f}")
    tc = sorted((L, p) for (L, t), p in rows.items() if t == 1.0)
    g1 = [(L, out["points"][f"L{L}_t1.0"]) for L, _ in tc]
    out["G1"] = {"per_L": {L: {"ci_excludes_0": p["ci95"][1] < 0, "n_neg": p["n_neg"]} for L, p in g1},
                 "pass": all(p["ci95"][1] < 0 and p["n_neg"] >= 14 for _, p in g1)}
    print("G1", out["G1"])
    Ls = np.array([L for L, _ in g1 if L <= 128], float)
    if len(Ls) >= 3:
        y = np.array([abs(out["points"][f"L{int(L)}_t1.0"]["mean"]) for L in Ls])
        se = np.array([out["points"][f"L{int(L)}_t1.0"]["se"] for L in Ls])
        A = np.vstack([np.ones_like(Ls), np.log(Ls)]).T
        coef, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        x = -coef[1]
        # chain-level SE of the exponent by resampling chain means within each L
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(2000):
            yb = np.abs(y + rng.normal(0, se))
            boots.append(-np.linalg.lstsq(A, np.log(yb), rcond=None)[0][1])
        ch64_128 = (y[2] - y[1]) / y[1]
        out["scaling"] = {"L": Ls.tolist(), "abs_delta": y.tolist(), "x_loglog": float(x), "x_se": float(np.std(boots)),
                          "change_64_to_128": float(ch64_128),
                          "H_const": bool(abs(ch64_128) < 0.10), "H_decay": bool(x - 2 * np.std(boots) > 0.1),
                          "predict_L256_power": float(np.exp(coef[0]) * 256 ** coef[1])}
        print("scaling", out["scaling"])
    return out


def xy():
    pts = {}
    for f in sorted(glob.glob(str(D / "lattice_ground_truth_xy_L*.json"))):
        r = json.load(open(f))
        k = (r["L"], r["J1"], r["T"])
        if k not in pts or r.get("start", "hot") == "cold":   # amendment 2: the long-equilibration rerun supersedes
            pts[k] = r
    out = {"points": {}}
    for (L, J1, T), r in sorted(pts.items()):
        s = r["summary"]
        e = s["equil_check"]
        v = np.array([x["delta_director"] for x in r["ensembles"]])
        ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=v.std(ddof=1) / np.sqrt(len(v)))
        eq = abs(e["hot_E"]["mean"] - e["cold_E"]["mean"]) <= 2 * np.hypot(e["hot_E"]["se"], e["cold_E"]["se"])
        rec = {"delta_director": s["delta_director"]["mean"], "se": s["delta_director"]["se"], "ci95": list(ci),
               "n_pos": s["delta_director"]["n_pos"], "perm": s["delta_permuted"]["mean"], "perm_se": s["delta_permuted"]["se"],
               "scalar_S2": s["delta_scalar_S2"]["mean"], "scalar_se": s["delta_scalar_S2"]["se"],
               "S2": s["S2"]["mean"], "M1": s["M1"]["mean"], "chi2": s["chi2"], "equilibrated": bool(eq),
               "E_hot": e["hot_E"]["mean"], "E_cold": e["cold_E"]["mean"], "dir_ac": s.get("director_ac_lag1", {}).get("mean"),
               "odd_share": s["odd_share"]["mean"], "odd_share_perm": s["odd_share_perm"]["mean"],
               "harmonics": s["kernel_harmonics_mean"], "harmonics_perm": s["kernel_harmonics_perm_mean"]}
        out["points"][f"L{L}_J{J1}_T{T}"] = rec
        print(f"XY L={L} J1={J1} T={T}: dir {rec['delta_director']:+.4f}±{rec['se']:.4f} CI [{ci[0]:+.3f},{ci[1]:+.3f}]"
              f" pos {rec['n_pos']}/8 | perm {rec['perm']:+.4f}±{rec['perm_se']:.4f} | S2-sorted {rec['scalar_S2']:+.4f}"
              f" | S2 {rec['S2']:.3f} M1 {rec['M1']:.3f} chi2 {rec['chi2']:.1f} | E hot {rec['E_hot']:.4f} cold {rec['E_cold']:.4f}"
              f" eq={eq} ac={rec['dir_ac']} | odd {rec['odd_share']:.3f} (perm {rec['odd_share_perm']:.3f})")
    for J1 in sorted({k[1] for k in pts}):
        for L in sorted({k[0] for k in pts if k[1] == J1}):
            Ts = sorted(T for (l, j, T) in pts if l == L and j == J1)
            chi = [pts[(L, J1, T)]["summary"]["chi2"] for T in Ts]
            Tstar = Ts[int(np.argmax(chi))]
            i = Ts.index(Tstar)
            near = [Ts[k] for k in (i - 1, i, i + 1) if 0 <= k < len(Ts)]
            sig = {T: out["points"][f"L{L}_J{J1}_T{T}"]["ci95"][0] > 0 for T in near}
            neg = {T: out["points"][f"L{L}_J{J1}_T{T}"]["ci95"][1] < 0 for T in near}
            out[f"Tstar_L{L}_J{J1}"] = {"Tstar": Tstar, "chi2": dict(zip(map(str, Ts), chi)),
                                        "pos_ci_near": {str(k): v for k, v in sig.items()},
                                        "neg_ci_near": {str(k): v for k, v in neg.items()},
                                        "G_pass": bool(sig[Tstar] and sum(sig.values()) >= 2)}
            print(f"T* L={L} J1={J1}: {Tstar}; positive-CI near T*: {sig}; negative-CI: {neg}")
    return out


def xytime():
    out = {}
    for f in sorted(glob.glob(str(D / "lattice_ground_truth_xytime_*.json"))):
        r = json.load(open(f))
        c = r["chains"]
        d = [x["delta_director"] for x in c if np.isfinite(x["delta_director"])]
        t = [x["delta_time"] for x in c if np.isfinite(x["delta_time"])]
        out[f"L{r['L']}_T{r['T']}_J{r['J1']}"] = {"director_mean": float(np.mean(d)), "time_mean": float(np.mean(t)),
                                                  "n": len(d), "bins_nonempty": [x["bins_nonempty"] for x in c],
                                                  "corr": float(np.corrcoef(d, t)[0, 1]) if len(d) > 2 else None}
        print("old design", f, out[f"L{r['L']}_T{r['T']}_J{r['J1']}"])
    return out


if __name__ == "__main__":
    res = {"ising": ising(), "xy": xy(), "xytime": xytime()}
    json.dump(res, open(D / "lattice_ground_truth_score.json", "w"), indent=1, default=float)
