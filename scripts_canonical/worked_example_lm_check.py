"""App M's language-model worked example: check every number of `rung llm --model EleutherAI/pythia-160m
--axis axes/world_knowledge32.json --paper-seeds --match-last-token` (rung, fresh clone) against the battery's artifacts.
  - declared and order-averaged shift per layer (linear frame) == llm_battery_pythia-160m.json world_knowledge32
    (delta_declared, dbar50_run37), asserted to 1e-8 (float64 summation order; worst printed);
  - order-averaged shift in the z-scored frame == llm_battery's own estimator with gram 'zscore' (the battery's seeds), asserted;
  - permutation p: rung's null is run37's convention (500 permutations, each averaged over the first 20 observed orders,
    perms rng 3700); the battery's is 200 permutations each averaged over 50 fresh orders (rng 4800): both printed, not asserted;
  - distinct last tokens per class == llm_lasttoken_pythia-160m.json world_knowledge32 (asserted); matched / random arms:
    rung's own subsample draws and null (20 subsamples, 100 permutations x 20 orders), printed beside the battery's.
Usage: python worked_example_lm_check.py <lm.json>  -> data_canonical/appm_lm_check.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_battery as hb  # noqa: E402

lm = json.load(open(sys.argv[1]))
ax = lm["axes"]["world_knowledge32"]
h2 = json.load(open(hb.OUTDIR / "llm_battery_pythia-160m.json"))["batteries"]["world_knowledge32"]
lt = json.load(open(hb.OUTDIR / "llm_lasttoken_pythia-160m.json"))["batteries"]["world_knowledge32"]
z = np.load(hb.STATES / "pythia-160m__world_knowledge32.npz")
H, labels = z["H"], z["labels"]
nc = len(np.unique(labels))
orng = np.random.default_rng(3701)
orders50 = [orng.permutation(nc).tolist() for _ in range(50)]
counts = [1, 2, 3, 4, 6, 8]
rows, worst = [], {"declared": 0.0, "orderavg": 0.0, "orderavg_z": 0.0}
for r, h in zip(ax["layers"], h2["layers"]):
    l = r["layer"]
    worst["declared"] = max(worst["declared"], abs(r["delta"] - h["delta_declared"]))
    worst["orderavg"] = max(worst["orderavg"], abs(r["delta_orderavg"] - h["dbar50_run37"]))
    Kz = hb.gram(H[l], "zscore")
    mem = hb.members_of(labels, nc)
    sizes = hb.ladder(Kz, mem, list(range(nc)), counts)[1]
    dz = hb.order_avg(Kz, mem, orders50, counts, hb.floor(Kz, len(labels), sizes, l * 100 + 1))
    worst["orderavg_z"] = max(worst["orderavg_z"], abs(r["delta_orderavg_z"] - dz))
    arms = r.get("last_token_arms") or {}
    rows.append({"layer": l, "delta_orderavg": r["delta_orderavg"], "p_rung": r["p_two_orderavg"],
                 "p_h2_one_sided": h["perm"]["p_one"], "null_mean_rung": r["null_mean_orderavg"],
                 "null_mean_h2": h["perm"]["null_mean"], "delta_orderavg_z": r["delta_orderavg_z"],
                 "p_rung_z": r["p_two_orderavg_z"], "frames_disagree": r.get("frames_disagree"),
                 "matched": arms.get("matched", {}).get("delta_orderavg_median"),
                 "random": arms.get("random", {}).get("delta_orderavg_median"),
                 "h2_matched": lt["matched"][l]["dbar_median"] if isinstance(lt.get("matched"), list) else None})
# float64 summation order differs (rung centres each subset Gram; the battery uses a one-pass numba trace): 1e-8, worst printed
assert worst["declared"] < 1e-8 and worst["orderavg"] < 1e-8 and worst["orderavg_z"] < 1e-8, worst
assert ax["last_token"]["distinct_per_class"] == lt["distinct_last_tokens"], (ax["last_token"]["distinct_per_class"],
                                                                            lt["distinct_last_tokens"])
out = {"max_abs_diff": worst, "distinct_last_tokens": lt["distinct_last_tokens"], "rows": rows,
       "frames_disagree_layers": ax.get("frames_disagree_layers")}
json.dump(out, open(hb.OUTDIR / "appm_lm_check.json", "w"), indent=1)
print(json.dumps({"max_abs_diff": worst, "distinct": lt["distinct_last_tokens"]}, indent=1))
for x in rows:
    print(f"L{x['layer']:02d} dbar {x['delta_orderavg']:+.3f} p_rung {x['p_rung']:.3f} p_h2(one-sided) {x['p_h2_one_sided']:.3f} | "
          f"z-frame {x['delta_orderavg_z']:+.3f} p {x['p_rung_z']:.3f} disagree {x['frames_disagree']} | matched "
          f"{x['matched']} random {x['random']} (H2 matched {x['h2_matched']})")
print("ALL ASSERTS PASS")
