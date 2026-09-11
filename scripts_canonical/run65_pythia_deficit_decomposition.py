"""Run 65 (S79, 2026-09-06) - The four-term split of the deficit in a language
model: which term carries the content axis's shift at the embedding and the
construction axis's rise with depth?

WHY: in V1 the direction-aligned shift is the pooling of class-private
within-class fluctuation modes (run 63: the deficit IS the pooling term on
8 of 8 recordings). The language-model battery (Section 8) reads the same
observable on prompt classes: a content axis (eight knowledge domains) that
is positive at the embedding and declines, and a construction axis (eight
sentence types) that starts negative and is built with depth. Does the
anatomy of the deficit differ between the two, and does the built
organization pop out as prompt-class-private variability the way the
direction code does in cortex? The same exact bookkeeping answers it.

DESIGN: Pythia-160m (13 layers with the embedding) and Pythia-2.8B-deduped
(33 layers), fp16 on MPS as in run 61, last-token hidden states; the two
axes of the released battery, sixteen prompts per class, eight classes, the
declared class order; the ladder [1, 2, 3, 4, 6, 8] classes (16 to 128
prompts) with ten random-subset floors; at every layer the deficit per rung
and its exact split into the trace, pooling, between-class and coupling
terms (deficit_split.py); the shift delta as the slope of the deficit
against log n and its exact split into the four slopes; the private slope
(P against log(k/8)); and five label permutations per layer as the sampling
baseline of every term (sixteen prompts per class give each class's sample
covariance its own random subspace), so every registered quantity is the
excess of the observed term over the label-shuffle mean. The stored
battery shifts (run 17, run 26) are printed next to the recomputed delta.
"Late layers" = the last third of the network, l >= ceil(2 n_blocks / 3):
layers 8-12 on 160m, 22-32 on 2.8B.

REGISTERED EXPECTATIONS (written before the run):
E1 (content at the embedding = lexical means): at layer 0 on the content
    axis, the between-class term is the largest-magnitude term of the four
    in the excess delta-split, on both models.
E2 (the built organization is prompt-class-private variability): on the
    construction axis, over the late layers, the pooling term is the
    largest-magnitude term of the excess delta-split and positive in more
    than half of those layers, on both models.
E3 (the private share grows with depth): the construction axis's excess
    private slope has Spearman >= 0.5 with the layer index on both models;
    the content axis's is reported for contrast.
A miss is reported at full volume.

POST-RUN NOTE (first pass, before the rerun): all three missed, and the
per-layer terms showed why: at every late layer the trace term is +0.4 and
the pooling term -0.3, a change of the within-class SCALE that the two terms
carry with opposite signs (A holds +2 Delta log w1, P holds -2 Delta log w1)
and that cancels in their sum. The exact scale-free regrouping
A + P = D + Tb (D = Delta log d_eff of the pooled within-class covariance,
Tb = the between-class trace share; deficit_split.py) is reported alongside,
descriptively, with the scale-free private slope (D against log(k/8)). The
registered expectations E1-E3 are judged on the terms they named.

Out: ../data_canonical/run65_pythia_deficit_decomposition.json (+ .log)
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
SUPP = HERE.parent.parent / "arxiv_supplement"
OUT = DATA / "run65_pythia_deficit_decomposition.json"
sys.path.insert(0, str(HERE))
from deficit_split import FREE, decompose, largest_term  # noqa: E402

spec = importlib.util.spec_from_file_location("rung", SUPP / "rung.py")
rung = importlib.util.module_from_spec(spec); spec.loader.exec_module(rung)

MODELS = [("pythia-160m", "EleutherAI/pythia-160m", "run17_multiclass_battery.json"),
          ("pythia-2.8b", "EleutherAI/pythia-2.8b-deduped", "run26_pythia28b_battery.json")]
EXTRA_MODELS = {"pythia-410m": ("pythia-410m", "EleutherAI/pythia-410m-deduped", "run18_pythia410m_battery.json"),
                "pythia-1b": ("pythia-1b", "EleutherAI/pythia-1b-deduped", "run19_pythia1b_battery.json")}
AXES = {"content": "world_knowledge", "construction": "language_type"}
NC = 8
BIN_COUNTS = (1, 2, 3, 4, 6, 8)
N_NULL = 10
N_SHUFFLE = 5
KEYS = ("A", "P", "Cb", "Cx")


def encode(model_name, prompts, device="mps", max_len=128):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    states = []
    with torch.no_grad():
        for p in prompts:
            ids = tok(p, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
            hs = model(ids, output_hidden_states=True).hidden_states
            states.append([h[0, -1, :].float().cpu().numpy() for h in hs])
    layers = [np.stack([s[l] for s in states]) for l in range(len(states[0]))]
    del model
    return layers


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated extra models (pythia-410m, pythia-1b) instead of the registered pair; output goes to a suffixed file")
    args = ap.parse_args()
    models = [EXTRA_MODELS[m] for m in args.models.split(",") if m] if args.models else MODELS
    out_path = OUT if not args.models else DATA / f"run65_pythia_deficit_decomposition_{args.models.replace(',', '_').replace('-', '')}.json"
    t0 = time.time()
    out = {"design": {"models": [m[1] for m in models], "axes": AXES, "bin_counts": BIN_COUNTS, "n_null": N_NULL, "n_shuffle": N_SHUFFLE}, "models": {}}
    for tag, model_name, battery in models:
        stored = json.load(open(DATA / battery))["axes"]
        mrow = {"model": model_name, "axes": {}}
        for role, ax in AXES.items():
            _, classes, prompts, labels, _ = rung._load_axis(str(SUPP / "axes" / f"{ax}.json"), use_strata=False)
            layers = encode(model_name, prompts)
            n_blocks = len(layers) - 1; late_from = int(np.ceil(2 * n_blocks / 3))
            rows = []
            for l, X in enumerate(layers):
                X = X / (X.std() + 1e-9)
                r = decompose(X, labels, list(range(NC)), BIN_COUNTS, n_null=N_NULL, seed=l, n_shuffle=N_SHUFFLE)
                ex = r["excess"]["delta_split"]
                row = {"layer": l, "delta": r["delta"], "delta_stored": stored[ax]["layers"][l]["delta"], "delta_shuffle": r["shuffle"]["delta"],
                       "delta_split": r["delta_split"], "delta_split_shuffle": r["shuffle"]["delta_split"], "delta_split_excess": ex,
                       "largest_excess_term": largest_term(ex), "largest_free_term": largest_term(ex, FREE),
                       "delta_split_free": r["delta_split_free"], "delta_scale": r["delta_scale"],
                       "private_slope": r["private_slope"], "private_slope_shuffle": r["shuffle"]["private_slope"],
                       "private_slope_excess": r["excess"]["private_slope"], "private_slope_dim": r["private_slope_dim"],
                       "private_slope_dim_excess": r["excess"]["private_slope_dim"], "at_four": r["at_four"], "at_four_excess": r["excess"]["at_four"],
                       "late_share": r["late_share"], "rungs": r["rungs"], "rungs_shuffle": r["shuffle"]["rungs"],
                       "max_sum_check": r["max_sum_check"], "max_regroup_check": r["max_regroup_check"], "max_pr_identity_rel": r["max_pr_identity_rel"], "late": bool(l >= late_from)}
                rows.append(row)
                print(f"[{tag} {role:12s}] L{l:2d}: delta {r['delta']:+.3f} (stored {row['delta_stored']:+.3f}, shuffle {r['shuffle']['delta']:+.3f}) | excess "
                      f"A {ex['A']:+.3f} P {ex['P']:+.3f} Cb {ex['Cb']:+.3f} Cx {ex['Cx']:+.3f} -> {row['largest_excess_term']:2s} | free D {ex['D']:+.3f} Tb {ex['Tb']:+.3f} S {ex['S']:+.3f} -> {row['largest_free_term']:2s} | "
                      f"private slope P {r['excess']['private_slope']:+.2f} dim {r['excess']['private_slope_dim']:+.2f} | {time.time()-t0:.0f}s", flush=True)
            mrow["axes"][role] = {"axis": ax, "classes": classes, "n_layers": len(layers), "late_from": late_from, "layers": rows}
            del layers
        out["models"][tag] = mrow
        json.dump(out, open(out_path, "w"), indent=1)
    verdict = {}
    for tag, _, _ in models:
        ax = out["models"][tag]["axes"]
        c0 = ax["content"]["layers"][0]
        late = [r for r in ax["construction"]["layers"] if r["late"]]
        e2n = sum(r["largest_excess_term"] == "P" and r["delta_split_excess"]["P"] > 0 for r in late)
        ls = [r["layer"] for r in ax["construction"]["layers"]]
        rho_c = float(spearmanr(ls, [r["private_slope_excess"] for r in ax["construction"]["layers"]])[0])
        rho_k = float(spearmanr(ls, [r["private_slope_excess"] for r in ax["content"]["layers"]])[0])
        rho_cd = float(spearmanr(ls, [r["private_slope_dim_excess"] for r in ax["construction"]["layers"]])[0])
        rho_kd = float(spearmanr(ls, [r["private_slope_dim_excess"] for r in ax["content"]["layers"]])[0])
        late_free = [r["largest_free_term"] for r in late]
        verdict[tag] = {"E1_content_L0_largest": c0["largest_excess_term"], "E1_content_L0_excess_split": c0["delta_split_excess"], "E1": bool(c0["largest_excess_term"] == "Cb"),
                        "refined_descriptive": {"content_L0_largest_free": c0["largest_free_term"],
                                                "construction_late_largest_free_counts": {k: int(late_free.count(k)) for k in FREE},
                                                "construction_late_mean_free": {k: float(np.mean([r["delta_split_excess"][k] for r in late])) for k in FREE + ("S",)},
                                                "content_late_mean_free": {k: float(np.mean([r["delta_split_excess"][k] for r in ax["content"]["layers"] if r["late"]])) for k in FREE + ("S",)},
                                                "rho_layer_dimslope_construction": rho_cd, "rho_layer_dimslope_content": rho_kd},
                        "E2_late_layers": [r["layer"] for r in late], "E2_count_P_largest_positive": int(e2n), "E2": bool(e2n > len(late) / 2),
                        "E2_late_largest_terms": [r["largest_excess_term"] for r in late],
                        "E3_rho_construction": rho_c, "E3_rho_content": rho_k, "E3": bool(rho_c >= 0.5),
                        "max_sum_check": max(r["max_sum_check"] for a in ax.values() for r in a["layers"]),
                        "max_pr_identity_rel": max(r["max_pr_identity_rel"] for a in ax.values() for r in a["layers"])}
    verdict["E1"] = bool(all(verdict[t]["E1"] for t, _, _ in models)); verdict["E2"] = bool(all(verdict[t]["E2"] for t, _, _ in models)); verdict["E3"] = bool(all(verdict[t]["E3"] for t, _, _ in models))
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(out_path, "w"), indent=1); print(f"wrote {out_path} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
