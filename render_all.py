"""Render the paper's key tables and headline numbers from the committed
JSON artifacts in data/. Requires only numpy; no model downloads.

Usage: theta-zoom-render (after pip install .) or python render_all.py
"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"


def j(name):
    return json.load(open(DATA / name))


def llm_summary():
    print("\n== LLM battery: content/structure contrast (paper Table 5) ==")
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
    cnn_summary()
    axis_search_summary()
    cyclic_summary()
    ranking_summary()
    print("\nFigure scripts: scripts/make_fig5_mechanism.py, "
          "scripts/make_fig6_llm.py")


if __name__ == "__main__":
    main()
