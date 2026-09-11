"""Run 68 (S79, 2026-09-06) - Two consistency checks on the V1 recordings
that runs 65-67 made necessary before their sentences enter the paper.

WHY: on the language model, the network and the Allen populations the trace
and pooling terms of run 63 each carry the within-class scale with opposite
signs (A = S + Tb, P = D - S; deficit_split.py), so "the deficit is the
pooling term" reads as "the deficit is the within-class dimension term" only
where the scale term S is small. And the raw-count localization of the
leading within-class mode on the Allen populations (6.7% of units) was rate
heterogeneity: per-unit z-scoring spread it to 31% of units. The paper's
statement that the leading within-class mode of every recording is carried
by 48-121 neurons (run 64a) was made on globally scaled deconvolved
responses, so the same control is owed here.

DESIGN: the eight grating recordings, all tuned neurons (run 63's subset),
the eight-class ladder [1, 2, 3, 4, 6, 8] with ten floors, the refined split
(D, Tb, S, Cb, Cx) at four classes and in the shift; and run 64a's class
statistics (within-class PR, diagonal PR, the neuron-space PR of the top
within-class eigenvector, the top-10 variance fraction) on the recording's
own units and on per-neuron z-scored responses.

REGISTERED EXPECTATIONS (written before the run):
V1 (the scale term is small in cortex): |S| at four classes is below 0.15
    and |S| is smaller than |D| on 8 of 8 recordings, so that the pooling
    term of the paper is the within-class dimension term.
V2 (the localization is not an amplitude effect): on per-neuron z-scored
    responses the neuron-space participation ratio of the leading
    within-class mode stays below 5% of the tuned population on 8 of 8
    recordings (it is 0.3-0.9% on the recording's own units).
A miss is reported at full volume.

POST-RUN NOTE (first pass): V1 held on 8 of 8 (S at four classes -0.06 to
+0.11, D -0.13 to -0.43 against P -0.19 to -0.51); V2 missed on 8 of 8: at
equal weight per neuron the leading within-class mode spans 12-28% of the
population (1,700-4,500 neurons), so the 48-121-neuron figure is where the
mode's amplitude sits, not its extent. The rerun adds three label
permutations so that the shift's split is net of sampling, as in runs
65-67 (--shuffle 3), and keeps the first pass's artifact alongside.

Out: ../data_canonical/run68_v1_scale_and_zscore.json (+ .log)
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUT = DATA / "run68_v1_scale_and_zscore.json"
sys.path.insert(0, str(HERE))
from deficit_split import FREE, decompose, largest_term  # noqa: E402


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


r63 = load_module("run63", "run63_deficit_decomposition.py")
r64 = load_module("run64a", "run64a_within_class_localization.py")
BIN_COUNTS = (1, 2, 3, 4, 6, 8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shuffle", type=int, default=0, help="label permutations per recording (the sampling baseline of each term)")
    args = ap.parse_args()
    t0 = time.time()
    out_path = OUT if args.shuffle == 0 else DATA / "run68_v1_scale_and_zscore_shuffled.json"
    oz = {r["name"]: r for r in json.load(open(DATA / "orientation_zoom.json"))["results"]}
    names = [n for n in oz if "gratings_" in n and "static" not in n]
    out = {"design": {"bin_counts": BIN_COUNTS, "n_null": 10, "n_shuffle": args.shuffle, "subset": "run63 full tuned population"}, "rows": {}}
    for name in names:
        tag = r63.short(name)
        Xt, bl, subs = r63.data_subsets(name)
        idx = subs["full"]; N = int(len(idx))
        Xs = np.ascontiguousarray(Xt[:, idx])
        r = decompose(Xs, bl, list(range(8)), BIN_COUNTS, n_null=10, seed=42, n_shuffle=args.shuffle)
        own = r64.summarize(r64.class_stats(Xt, bl, idx), N)
        Xz = np.ascontiguousarray((Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9))
        zs = r64.summarize(r64.class_stats(Xz, bl, np.arange(N)), N)
        del Xt, Xs, Xz
        a4 = r["at_four"]
        row = {"name": name, "N": N, "delta": r["delta"], "delta_split": r["delta_split"], "delta_split_free": r["delta_split_free"], "delta_scale": r["delta_scale"],
               "at_four": a4, "private_slope": r["private_slope"], "private_slope_dim": r["private_slope_dim"], "rungs": r["rungs"],
               "max_sum_check": r["max_sum_check"], "max_regroup_check": r["max_regroup_check"],
               "localization_own_units": own, "localization_zscored": zs}
        if "excess" in r:
            row["delta_split_excess"] = r["excess"]["delta_split"]; row["at_four_excess"] = r["excess"]["at_four"]; row["private_slope_dim_excess"] = r["excess"]["private_slope_dim"]
            row["delta_split_shuffle"] = r["shuffle"]["delta_split"]
            ex = r["excess"]["delta_split"]
            print(f"      excess split: D {ex['D']:+.3f} Tb {ex['Tb']:+.3f} S {ex['S']:+.3f} Cb {ex['Cb']:+.3f} Cx {ex['Cx']:+.3f} | shuffle D {r['shuffle']['delta_split']['D']:+.3f} Tb {r['shuffle']['delta_split']['Tb']:+.3f} | dim slope excess {r['excess']['private_slope_dim']:+.2f}", flush=True)
        out["rows"][tag] = row
        print(f"[{tag}] N {N}: delta {r['delta']:+.3f} | at four: deficit {a4['deficit']:+.3f} A {a4['A']:+.3f} P {a4['P']:+.3f} | D {a4['D']:+.3f} Tb {a4['Tb']:+.3f} S {a4['S']:+.3f} Cb {a4['Cb']:+.3f} Cx {a4['Cx']:+.3f} | "
              f"slope split D {r['delta_split_free']['D']:+.3f} Tb {r['delta_split_free']['Tb']:+.3f} S {r['delta_scale']:+.3f} | private slope P {r['private_slope']:.2f} dim {r['private_slope_dim']:.2f} | "
              f"top-mode neuron PR own {own['pr_neuron_top1_mean']:.0f} ({own['pr_neuron_top1_mean']/N:.4f} N) z-scored {zs['pr_neuron_top1_mean']:.0f} ({zs['pr_neuron_top1_mean']/N:.4f} N) | "
              f"wc PR own {own['pr_full_mean']:.0f} z {zs['pr_full_mean']:.0f} | diag/full own {own['ratio_mean']:.1f} z {zs['ratio_mean']:.1f} | {time.time()-t0:.0f}s", flush=True)
        json.dump(out, open(out_path, "w"), indent=1)
    rows = out["rows"]; tags = list(rows)
    v1a = int(sum(abs(rows[t]["at_four"]["S"]) < 0.15 for t in tags)); v1b = int(sum(abs(rows[t]["at_four"]["S"]) < abs(rows[t]["at_four"]["D"]) for t in tags))
    v2 = int(sum(rows[t]["localization_zscored"]["pr_neuron_top1_mean"] / rows[t]["N"] < 0.05 for t in tags))
    verdict = {"V1_S_below_0.15_count": v1a, "V1_S_below_D_count": v1b, "V1": bool(v1a == 8 and v1b == 8), "V2_zscored_below_5pct_count": v2, "V2": bool(v2 == 8),
               "S_at_four": {t: rows[t]["at_four"]["S"] for t in tags}, "D_at_four": {t: rows[t]["at_four"]["D"] for t in tags}, "P_at_four": {t: rows[t]["at_four"]["P"] for t in tags},
               "Tb_at_four": {t: rows[t]["at_four"]["Tb"] for t in tags},
               "top_mode_frac_own": {t: rows[t]["localization_own_units"]["pr_neuron_top1_mean"] / rows[t]["N"] for t in tags},
               "top_mode_frac_zscored": {t: rows[t]["localization_zscored"]["pr_neuron_top1_mean"] / rows[t]["N"] for t in tags},
               "top_mode_neurons_zscored": {t: rows[t]["localization_zscored"]["pr_neuron_top1_mean"] for t in tags},
               "private_slope_dim_full": {t: rows[t]["private_slope_dim"] for t in tags}}
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(out_path, "w"), indent=1); print(f"wrote {out_path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
