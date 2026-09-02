"""Render the paper's key tables and headline numbers from the committed
JSON artifacts in data_canonical/. Requires only numpy; no model downloads.

Usage: theta-zoom-render (after pip install .) or python render_all.py
"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data_canonical"


def j(name):
    return json.load(open(DATA / name))


def llm_summary():
    print("\n== LLM battery: depth profiles + descriptive sig/shuffle (runs 17-26) ==")
    files = {"Pythia-160m": "run17_multiclass_battery.json",
             "Pythia-410m": "run18_pythia410m_battery.json",
             "Pythia-1B": "run19_pythia1b_battery.json",
             "Pythia-2.8B": "run26_pythia28b_battery.json",
             "OLMo-1B": "run25_olmo1b_16pc_battery.json"}
    for tag, f in files.items():
        d = j(f)
        for ax in ("world_knowledge", "language_type", "ethical"):
            ls = d["axes"][ax]["layers"]
            de = np.array([l["delta"] for l in ls])
            sh = np.array([l["shuffle_mean"] for l in ls])
            r = np.mean(np.abs(de)) / max(np.mean(np.abs(sh)), 1e-9)
            print(f"  {tag:12s} {ax:16s} emb {de[0]:+.2f} "
                  f"final {de[-1]:+.2f} sig/shuffle {r:.1f}x")


def multipole_summary():
    """Exact 8-direction harmonic decomposition of C(dphi) (paper Sec. 4, App. I)."""
    print("\n== V1 class-mean correlation harmonics (8 direction classes; "
          "multipole_harmonics_8dir) ==")
    d = j("multipole_harmonics_8dir.json")
    for r in d["rows"]:
        if r["status"] != "ok":
            print(f"  {r['name'][:32]:32s} degenerate direction bins")
            continue
        c = r["coef"]
        print(f"  {r['name'][:32]:32s} a {c['a']:.3f} b2 {c['b2']:+.3f} c1 {c['c1']:+.3f} "
              f"b4 {c['b4']:+.3f} c3 {c['c3']:+.3f} | b2/c1 {r['b2_over_c1']:.2f} "
              f"even/odd {r['even_over_odd_variance']:.2f} | quadrupole-dominant "
              f"{r['quadrupole_dominant']}")
    print("  verdict:", d["verdict"])


def tuning_summary():
    """Per-neuron tuning behind the localized-grating dipole (App. I)."""
    print("\n== Localized vs full-field gratings: per-neuron tuning (local_vs_fullfield_tuning) ==")
    d = j("local_vs_fullfield_tuning.json")
    for r in d["rows"]:
        if r["status"] != "ok":
            continue
        print(f"  {r['name'][:32]:32s} OSI {r['median_osi_tuned']:.2f} DSI {r['median_dsi_tuned']:.2f} "
              f"DSI>0.5 {r['frac_dsi_gt_0p5_tuned']:.2f} cardinal {r['cardinal_direction_fraction']:.2f} "
              f"spatial r OSI {r['spatial_r_osi']:+.2f} DSI {r['spatial_r_dsi']:+.2f} "
              f"b2/c1 tuned-only {r['b2_over_c1']['tuned_only']:.2f}")
    print("  verdict:", d["verdict"])


def scale_summary():
    """Sector balance across scales (App. I): additivity, graining flow, Allen per-area."""
    print("\n== Sector balance across scales (sector_balance_scale) ==")
    d = j("sector_balance_scale.json")
    for rr in d["rows"]:
        if rr["status"] != "ok":
            continue
        fl = rr["graining_flow"]
        print(f"  {rr['name'][:32]:32s} b2/c1 {rr['b2_over_c1_measured']:.2f} (from neuron harmonics "
              f"{rr['b2_over_c1_from_neuron_harmonics']:.2f}) | K=64: ori-sorted {fl['ori_sorted'][-1]:.1f} "
              f"dir-sorted {fl['dir_sorted'][-1]:.2f} random {fl['random'][-1]:.2f}")
    for area, a in sorted(d["allen_per_area"].items(), key=lambda kv: -kv[1]["n"]):
        print(f"  Allen {area:6s} n={a['n']:3d} median b2/|c1| {a['median_b2_over_c1']:.1f} quadrupole-dominant {a['frac_quadrupole_dominant']:.2f}")
    print("  verdict:", d["verdict"])


def certified_summary():
    """Permutation-certified layer counts (paper Table 5, run37 artifact)."""
    print("\n== LLM battery: certified layers, declared order . order-averaged "
          "(500 permutations, two-sided p<0.05; paper Table 5) ==")
    d = j("run37_inferential_nulls.json")
    for m, mv in d["models"].items():
        for a, av in mv["axes"].items():
            L = av["layers"]
            nc = sum(1 for l in L if l["p_two"] < 0.05)
            na = sum(1 for l in L if l["p_two_orderavg"] < 0.05)
            print(f"  {m:24s} {a:16s} {nc:2d}/{len(L)} . {na:2d}/{len(L)}")


def cnn_summary():
    print("\n== CNN double dissociation (paper Table 4) ==")
    d = j("run5b_cnn_seeds.json")
    for k, v in d.items():
        print(f"  {k:24s} delta_rot {v['delta_rot_mean']:+.3f} "
              f"(sd {v['delta_rot_sd']:.3f})  "
              f"delta_digit {v['delta_digit_mean']:+.3f}")


def axis_search_summary():
    print("\n== Static-session axis search (paper Table 1 rows) ==")
    d = j("run27_static_axis_search.json")
    for sess, s in d["sessions"].items():
        for ax, a in s["axes"].items():
            print(f"  {sess:34s} {ax:16s} delta {a['delta']:+.4f} "
                  f"shuffle {a['shuffle_mean']:+.4f}")


def cyclic_summary():
    print("\n== Planted-C8 recovery (run28) ==")
    d = j("run28_cyclic_axis_llm.json")
    for ax, a in d["axes"].items():
        ls = a["layers"][1:]
        cr = [l["circulant_r2"] for l in ls]
        er = [l["eff_rank"] for l in ls]
        print(f"  {ax:16s} circulant R2 [{min(cr):.2f}, {max(cr):.2f}]  "
              f"eff_rank [{min(er):.2f}, {max(er):.2f}]")


def ranking_summary():
    print("\n== Cross-model axis-ranking conservation (run22 + extension) ==")
    d = j("run22b_rank_extension.json")
    for pair, v in d["pairs"].items():
        print(f"  {pair:16s} rho {v['rho']:+.3f}")


def main():
    llm_summary()
    certified_summary()
    multipole_summary()
    tuning_summary()
    scale_summary()
    cnn_summary()
    axis_search_summary()
    cyclic_summary()
    ranking_summary()
    print("\nFigure scripts: scripts_canonical/make_fig5_mechanism.py, "
          "scripts_canonical/make_fig6_llm.py")


if __name__ == "__main__":
    main()
