"""rung against the paper's numbers, and the cause of every difference. Each dataset is read twice with rung: as a user would
(default seed, 20 floor draws drawn draw by draw) and with the paper generator's own floor (its seed, its draw count, drawn
rung by rung: zoom/decompose floor_by_rung=True). When the replay returns the artifact's value to floating-point tolerance,
the default run differs from the paper by its floor draw alone; the spread of the default over 20 seeds gives that draw's
size.
  GT3 direction ladder  Table 9 (cr_order_and_withinbin_controls.ladder: ten draws, rng 42)        raw/gratings_drifting_GT3_*.npy
  Allen VISp 715093703  per-session pooled and TF-residualized arms (allen_engine.ladder, idem)      raw/allen_counts, allen_masks
  macaque, 3 arrays     macaque_pvc11.json (ow.ladder, idem)                                        raw/pvc11/spikes_gratings
  lattices              finite_n_regen_*.json (xy_asymptote.split: six draws, rng 900 + ensemble;   regenerated with the release's
                        spin waves: run72 split, ten draws, rng 4007)                                samplers (numba)
Run: python -X faulthandler rung_acceptance_replay.py      Out: data_canonical/rung_acceptance_replay.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
import rung  # noqa: E402

RAW, DATA = HERE.parent / "raw", HERE.parent / "data_canonical"
OUT = DATA / "rung_acceptance_replay.json"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def read(X, lab, paper_seed, paper_draws, bin_counts=None, spread=True):
    """Shift and within-class part, as a user would and with the paper's floor."""
    out = {}
    d0 = rung.zoom(X, lab, n_perm=0, k_orders=0, seed=0, bin_counts=bin_counts)
    w0 = rung.decompose(X, lab, n_null=20, seed=0, bin_counts=bin_counts)
    out["default"] = {"delta": d0["delta"], "theta_floor_se": d0["theta_floor_se"], "within": w0["delta_split_free"]["D"]}
    rz = rung.zoom(X, lab, n_perm=0, k_orders=0, n_floor_draws=paper_draws, floor_seed=paper_seed, floor_by_rung=True,
                   bin_counts=bin_counts)
    rd = rung.decompose(X, lab, n_null=paper_draws, seed=paper_seed, floor_by_rung=True, bin_counts=bin_counts)
    out["replay"] = {"delta": rz["delta"], "within": rd["delta_split_free"]["D"]}
    if spread:
        ds = [rung.zoom(X, lab, n_perm=0, k_orders=0, seed=s, bin_counts=bin_counts)["delta"] for s in range(20)]
        out["default_20_seeds"] = {"mean": float(np.mean(ds)), "sd": float(np.std(ds, ddof=1)), "min": float(min(ds)), "max": float(max(ds))}
    return out


def row(name, r, paper_delta, paper_within):
    r.update(name=name, paper={"delta": paper_delta, "within": paper_within},
             residual_replay={"delta": r["replay"]["delta"] - paper_delta, "within": r["replay"]["within"] - paper_within})
    res = max(abs(r["residual_replay"]["delta"]), abs(r["residual_replay"]["within"]))
    r["cause"] = ("the floor draw alone (the replay is exact)" if res < 1e-9 else
                  "the floor draw; the replay agrees to float32 rounding (the generator's Gram is float32)" if res < 1e-5 else
                  "NOT the floor draw: the replay differs")
    print(f"{name:34s} default {r['default']['delta']:+.4f} replay {r['replay']['delta']:+.4f} paper {paper_delta:+.4f} "
          f"residual {r['residual_replay']['delta']:+.1e} | within replay {r['replay']['within']:+.4f} paper {paper_within:+.4f} "
          f"| 20-seed default {r.get('default_20_seeds', {}).get('mean', float('nan')):+.4f} +- {r.get('default_20_seeds', {}).get('sd', float('nan')):.4f} "
          f"| {r['cause']}", flush=True)
    return r


def main():
    rows = []
    # GT3, Table 9
    gt3 = sorted(RAW.glob("gratings_drifting_GT3*.npy"))[0]
    d = np.load(gt3, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(d["sresp"], np.float32).T)
    lab = np.clip(np.digitize(np.asarray(d["istim"], float) % (2 * np.pi), np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
    t9 = next(r for r in json.load(open(DATA / "cr_order_and_withinbin_controls.json"))["rows"] if r["name"] == gt3.stem)["orig"]
    rows.append(row("GT3 direction ladder (Table 9)", read(X, lab, 42, 10), t9["delta"], t9["within"]))
    # Allen VISp, one session, both arms
    E = _load("E", "allen_engine.py")
    sid = 715093703
    S = E.load_session(sid)
    V = E.pop_view(S, "VISp", E.populations(S)["VISp"])
    art = json.load(open(DATA / "allen_sessions" / f"session_{sid}.json"))["areas"]["VISp"]
    for arm, Xa in (("pooled", V["X"]), ("tfres", E.tf_residualize(V["X"], V["lab"], V["cell"]))):
        rows.append(row(f"Allen VISp {sid} {arm}", read(Xa, V["lab"], 42, 10), art[arm]["delta"], art[arm]["within"]))
    # macaque
    M = _load("M", "macaque_pvc11.py")
    mac = json.load(open(DATA / "macaque_pvc11.json"))["arrays"]
    for m in (1, 2, 3):
        Xm, lm, _ = M.load_array(m)
        lad = mac[m - 1]["raw"]["ladder"]
        rows.append(row(f"macaque array {m}", read(Xm, lm, 42, 10, bin_counts=(1, 2, 3, 4, 6, 8, 12)), lad["delta"], lad["within"]))
    # lattices, regenerated with the release's seeds (finite_n_lattice_regen.py's calls)
    import lattice_ground_truth as h
    import nematic_polar_delta as nem
    import xy_groundtruth as gt
    R = _load("R", "finite_n_lattice_regen.py")

    def feats(cfg, scale_eps=0.0, f32=False):
        _, _, psi = h.order_params(cfg)
        lb = np.minimum((psi / (np.pi / nem.N_BINS)).astype(int), nem.N_BINS - 1)
        F = nem.features(cfg); F = F - F.mean(0); F /= F.std() + scale_eps
        return (F.astype(np.float32) if f32 else F), lb
    reg = {g: {r["key"]: r for r in json.load(open(DATA / f"finite_n_regen_{g}.json"))} for g in ("nem", "polar", "sw")}
    cases = (("nematic L16 T1.0, 100 per class", "nem", "L16_T1.00_n100_e0",
              lambda: np.concatenate([R.chains_nem((16, 1.0, 0, [c]))[1][0] for c in range(160)]), 900, 6, {}),
             ("polar XY L32 T0.7, ensemble 0", "polar", "T0.70_e0",
              lambda: np.concatenate([R.chains_polar((0.70, 0, [c]))[1][0] for c in range(40)]), 900, 6, {}),
             ("spin waves, 200 per class, seed 0", "sw", "n200_s0", lambda: gt.spinwave(8 * 200, 40_000 + 100 * 200), 4007, 10,
              {"scale_eps": 1e-12, "f32": True}))
    for name, g, key, make, sd, nd, kw in cases:
        F, lb = feats(make(), **kw)
        st = reg[g][key]
        rows.append(row(name, read(F, lb, sd, nd), st["measured_delta"], st["measured_within"]))
    json.dump({"rows": rows}, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
