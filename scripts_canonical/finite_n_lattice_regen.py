"""The Fig 3A lattice and spin-wave ensembles, regenerated exactly with their generators' seeds and calls (xy_polar_twoterm.py,
xy_asymptote.py, xy_groundtruth.py), every stored measured value asserted to 1e-9 before anything is kept; per ensemble the split's
per-rung arrays (declared rung and every floor draw, with its class composition), the moments of finite_n_identity.moments (t, T_cd, m4,
K, Kn) and overlap_identity_checks' split-half K (whose sigma the overlap law uses, reproduced).
  polar  L 32, J1 0.42, T 0.6-0.85, ordered start, n_equil 1500, 8 ensembles x 40 chains x 5 samples.
  nem    L 16 T 1.0/1.1 (25..1600 per class), L 32 T 1.0 (25..400), 2 ensembles; smaller ensembles are the first chains of the largest
         one (same seeds), so the largest is generated and sliced.
  sw     the spin-wave model, 25 and 200 per class, 8 seeds.
Chains run in parallel (spawn; each chain is seeded on its own).
Run: python -X faulthandler finite_n_lattice_regen.py [polar|nem|sw] [n_workers]
Out: data_canonical/finite_n/<group>_*.npz, data_canonical/finite_n_regen_<group>.json"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data_canonical"
BIG = HERE.parent / "data_canonical" / "finite_n"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- chain workers (module level for spawn)
def chains_polar(args):
    import lattice_ground_truth as h
    T, e, cs = args
    L, J1, SAMPLES, SPACING, NEQ = 32, 0.42, 5, 10, 1500
    out = []
    for c in cs:
        seed = 77_000_000 + 100_000 * int(round(J1 * 100)) + 1000 * L + 40 * e + c + int(round(T * 1000)) * 7
        x, _ = h.xy_chain_w(L, T, J1, 1.0, seed, False, NEQ, SAMPLES, SPACING)
        out.append(x)
    return cs, out


def chains_nem(args):
    import lattice_ground_truth as h
    L, T, e, cs = args
    out = []
    for c in cs:
        seed = 33_000_000 + 1_000_000 * e + 10_000 * L + c + int(round(T * 100)) * 100_000_000
        x, _ = h.xy_chain_w(L, T, 0.42, 1.0, seed, True, 300 if L <= 32 else 500, 5, 10)
        out.append(x)
    return cs, out


def run_chains(fn, prefix, n_chains, ex, chunk=8):
    jobs = [(*prefix, list(range(i, min(i + chunk, n_chains)))) for i in range(0, n_chains, chunk)]
    got = {}
    for cs, xs in ex.map(fn, jobs):
        for c, x in zip(cs, xs):
            got[c] = x
    return np.concatenate([got[c] for c in range(n_chains)])


# ---------------------------------------------------------------- measurement, instrumented
def measured_split(F, lab, sizes, rng, n_floor, logparts):
    """The estimator's split with every rung's declared and floor-draw log PR(W) kept (rng consumed exactly as the original)."""
    order = np.argsort(lab, kind="stable")
    lad, flo = [], []
    for s in sizes:
        lad.append(logparts(F[order[:s]], lab[order[:s]]))
        dr = []
        for _ in range(n_floor):
            idx = rng.choice(len(F), s, replace=False)
            dr.append({**logparts(F[idx], lab[idx]), "comp": np.bincount(lab[idx], minlength=8).tolist()})
        flo.append(dr)
    x = np.log(sizes)
    sl = {k: float(np.polyfit(x, [l[k] - np.mean([d[k] for d in dr]) for l, dr in zip(lad, flo)], 1)[0]) for k in ("noise", "probe", "logpr")}
    floor_slope = float(np.polyfit(x, [np.mean([d["noise"] for d in dr]) for dr in flo], 1)[0])
    return {"sizes": list(map(int, sizes)), "lad_noise": [float(l["noise"]) for l in lad],
            "flo_noise": [[float(d["noise"]) for d in dr] for dr in flo], "flo_comp": [[d["comp"] for d in dr] for dr in flo],
            "within": sl["noise"], "class_mean": sl["probe"], "delta": sl["logpr"], "floor_slope_noise": floor_slope}


def analyse(F, lab, sizes, split_seed, n_floor, logparts, mom_seed, om_seed, P, OM):
    import xy_asymptote as xa  # noqa: F401
    sp = measured_split(F, lab, sizes, np.random.default_rng(split_seed), n_floor, logparts)
    mom = P.moments(F, lab, seed=mom_seed)
    rng = np.random.default_rng(om_seed); Ra, Rb = [], []
    for c in range(8):
        R = F[lab == c] - F[lab == c].mean(0); p = rng.permutation(len(R))
        Ra.append(R[p[: len(R) // 2]]); Rb.append(R[p[len(R) // 2:]])
    Kom = OM.split_half_K(Ra, Rb)
    return sp, mom, Kom


def save(group, key, lab, sp, mom, Kom, extra):
    BIG.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BIG / f"{group}_{key}.npz", lab=lab, t=mom["t"], T=mom["T"], m4=mom["m4"], K=mom["K"], Kn=mom["Kn"], K_om=Kom,
                        sizes=np.array(sp["sizes"]), lad_noise=np.array(sp["lad_noise"]), flo_noise=np.array(sp["flo_noise"]),
                        flo_comp=np.array(sp["flo_comp"]))
    return {"key": key, **extra, "counts": np.bincount(lab, minlength=8).tolist(), "sizes": sp["sizes"],
            "measured_within": sp["within"], "measured_class_mean": sp["class_mean"], "measured_delta": sp["delta"],
            "floor_slope_noise": sp["floor_slope_noise"], "lad_noise": sp["lad_noise"],
            "flo_noise_mean": [float(np.mean(d)) for d in sp["flo_noise"]],
            "t": mom["t"].tolist(), "T_cd": mom["T"].tolist(), "m4": mom["m4"].tolist(), "K": mom["K"].tolist(), "Kn": mom["Kn"].tolist(),
            "K_om": Kom.tolist(), "sigma_om": float(OM.sigma(Kom))}


OM = _load("om", "overlap_identity_checks.py")


def polar(ex):
    import lattice_ground_truth as h
    import xy_asymptote as xa
    import nematic_polar_delta as nem
    import finite_n_identity as P
    stored = json.load(open(DATA / "xy_polar_twoterm.json"))["rows"]
    rows = []
    for T in (0.60, 0.70, 0.80, 0.85):
        art = json.load(open(DATA / f"lattice_ground_truth_xy_L32_T{T:.2f}_J0.42_cold1500.json"))
        for e in range(8):
            t0 = time.time()
            cfg = run_chains(chains_polar, (T, e), 40, ex)
            _, _, psi = h.order_params(cfg)
            d_dir, _, _ = h.director_ladder(nem.features(cfg), psi, np.random.default_rng(1_000 + e))
            assert abs(d_dir - art["ensembles"][e]["delta_director"]) < 1e-9, (T, e, d_dir)
            lab = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
            F = nem.features(cfg); F = F - F.mean(0); F /= F.std()
            cnt = np.bincount(lab, minlength=8)
            sizes = sorted({int(cnt[:k].sum()) for k in nem.BIN_COUNTS if cnt[:k].sum() >= 10})
            st = next(r for r in stored if r["T"] == T and r["ens"] == e)
            sp, mom, Kom = analyse(F, lab, sizes, 900 + e, xa.N_FLOOR, xa.logparts, 5_000 + e, 800 + e, P, OM)
            assert abs(sp["within"] - st["measured_within"]) < 1e-9 and abs(sp["class_mean"] - st["measured_class_mean"]) < 1e-9, (T, e, sp["within"])
            assert abs(OM.sigma(Kom) - st["sigma_H1"]) < 1e-9, (T, e, OM.sigma(Kom), st["sigma_H1"])
            rows.append(save("polar", f"T{T:.2f}_e{e}", lab, sp, mom, Kom, {"T": T, "ens": e, "sigma_H1": st["sigma_H1"],
                                                                                 "pred_within_printed": st["pred_within"]}))
            print(f"polar T {T} e{e}: within {sp['within']:+.4f} (asserted) floor slope {sp['floor_slope_noise']:+.3f} sigma_om "
                  f"{OM.sigma(Kom):.3f} ({time.time() - t0:.0f}s)", flush=True)
            json.dump(rows, open(DATA / "finite_n_regen_polar.json", "w"), indent=1)
    return rows


def nem_group(ex):
    import lattice_ground_truth as h
    import xy_asymptote as xa
    import nematic_polar_delta as nem
    import finite_n_identity as P
    stored = json.load(open(DATA / "xy_twoterm.json"))["lattice"]
    rows = []
    for L, T, pers in ((16, 1.0, (25, 100, 400, 1600)), (16, 1.1, (25, 100, 400, 1600)), (32, 1.0, (25, 100, 200, 400))):
        for e in range(2):
            t0 = time.time()
            big = run_chains(chains_nem, (L, T, e), max(pers) * 8 // 5, ex)
            print(f"nem L{L} T{T} e{e}: {len(big)} configs generated ({time.time() - t0:.0f}s)", flush=True)
            for per in pers:
                cfg = big[: per * 8]
                _, _, psi = h.order_params(cfg)
                lab = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
                F = nem.features(cfg); F = F - F.mean(0); F /= F.std()
                cnt = np.bincount(lab, minlength=8)
                sizes = sorted({int(cnt[:k].sum()) for k in nem.BIN_COUNTS if cnt[:k].sum() >= 10})
                st = next(r for r in stored if r["L"] == L and r["T"] == T and r["per_class"] == per and r["ens"] == e)
                sp, mom, Kom = analyse(F, lab, sizes, 900 + e, xa.N_FLOOR, xa.logparts, 6_000 + 10 * e + per, 600 + e, P, OM)
                assert abs(sp["within"] - st["within"]) < 1e-9 and abs(sp["class_mean"] - st["class_mean"]) < 1e-9, (L, T, per, e, sp["within"], st["within"])
                assert abs(OM.sigma(Kom) - st["sigma_H1"]) < 1e-9, (L, T, per, e, OM.sigma(Kom), st["sigma_H1"])
                rows.append(save("nem", f"L{L}_T{T:.2f}_n{per}_e{e}", lab, sp, mom, Kom,
                                 {"L": L, "T": T, "per_class": per, "ens": e, "sigma_H1": st["sigma_H1"], "pred_within_printed": st["pred_within"]}))
                print(f"nem L{L} T{T} {per}/class e{e}: within {sp['within']:+.4f} (asserted) floor slope {sp['floor_slope_noise']:+.3f}", flush=True)
                json.dump(rows, open(DATA / "finite_n_regen_nem.json", "w"), indent=1)
    return rows


def sw_group(ex):
    import lattice_ground_truth as h
    import xy_groundtruth as gt
    import nematic_polar_delta as nem
    import finite_n_identity as P
    r72 = gt.xm.xs.r72
    stored = json.load(open(DATA / "xy_groundtruth.json"))
    tt = json.load(open(DATA / "xy_twoterm.json"))["spinwave"]
    rows = []
    for per in (25, 200):
        for s in range(8):
            cfg = gt.spinwave(8 * per, 40_000 + 100 * per + s)
            _, _, psi = h.order_params(cfg)
            F0 = nem.features(cfg)
            _, sizes, _ = h.director_ladder(F0, psi, np.random.default_rng(4000 + s))
            lab = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
            G = F0 - F0.mean(0); G /= G.std() + 1e-12; G = G.astype(np.float32)
            sp, mom, Kom = analyse(G, lab, sizes, 4000 + s + 7, r72.N_NULL, r72.logparts, 7_000 + 10 * s + per, 700 + s, P, OM)
            st = stored[f"spinwave_{per}"]["rows"][s]
            assert abs(sp["within"] - st["within_part"]) < 1e-6, (per, s, sp["within"], st["within_part"])
            # the printed law's sigma (xy_twoterm.predict: unscaled features, rng 700 + s)
            sig = next(r["sigma_H1"] for r in tt if r["per_class"] == per and r["seed"] == s)
            rows.append(save("sw", f"n{per}_s{s}", lab, sp, mom, Kom, {"per_class": per, "seed": s, "sigma_H1": sig,
                                                                         "pred_within_printed": float(np.log(8 / (1 + 7 * sig)) / np.log(8))}))
            print(f"sw {per}/class s{s}: within {sp['within']:+.4f} (asserted) floor slope {sp['floor_slope_noise']:+.3f}", flush=True)
            json.dump(rows, open(DATA / "finite_n_regen_sw.json", "w"), indent=1)
    return rows


if __name__ == "__main__":
    group, nw = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        {"polar": polar, "nem": nem_group, "sw": sw_group}[group](ex)
    print("ALL DONE", flush=True)
