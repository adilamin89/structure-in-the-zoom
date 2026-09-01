"""Run 28 - Planted cyclic (C8) axes on Pythia-160m: does the instrument
recover an embedded symmetry in an LLM? (Referee Option B: "controlled
rotations/permutations in synthetic tasks, cyclic group transformations".)

DESIGN: two axes with C8 ground truth, each 8 classes x 16 prompts built from
16 SHARED carrier templates with only the class token rotated - between-class
difference is solely the planted token and its semantic associations:
  compass:  north, northeast, ..., northwest (adjacent classes share a
            component word: "northeast" overlaps "north" and "east" - the
            cycle is planted lexically AND semantically)
  clock:    midnight, 3am, 6am, 9am, noon, 3pm, 6pm, 9pm (adjacent classes
            share period-of-day semantics)
Comparison axis: world_knowledge (run17 prompts) - no planted cyclic order.

MEASUREMENTS per layer:
  1. delta + shuffle (standard run17 ladder, classes in cycle order)
  2. class-mean kernel C (8x8 corrcoef of class means), eff_rank (run24
     convention: PR of eigenvalues), top1_frac
  3. circulant fit: f(D) = mean of C[i, (i+D) mod 8], symmetrized;
     R2_circ = variance of off-diagonal C explained by the circulant model
  4. Fourier spectrum of f(D) -> dominant harmonic
  5. entry-coherence order test (the V1 antipodal-order law transferred):
     delta under sequential (adjacent-first) vs antipodal-paired
     (0,4,1,5,2,6,3,7) vs 20 random class orders - no re-encoding needed.

REGISTERED EXPECTATIONS (written before the run):
C1: circulant R2 higher for compass/clock than for world_knowledge at most
    layers (the planted order is real; the comparison axis order is arbitrary).
C2: RECOVERY QUESTION (either outcome reportable): if the model embeds the
    cycle geometrically, kernel eff_rank < the ~6 of generic axes (run24),
    with a dominant low harmonic - V1-like. If categories stay equidistant,
    the planted symmetry is NOT recovered and run24's equidistance finding
    extends even to explicitly cyclic prompt structure.
C3: sign(delta_sequential - delta_antipodal) predicted by sign(f(1) - f(4))
    (V1 entry-coherence law: adjacent-first accumulation with higher adjacent
    coherence lowers early-rung dimensionality gain).
C4: shuffle ~ 0 everywhere.

Model: Pythia-160m (cached), MPS. ~10 min.
Out: ../data_canonical/run28_cyclic_axis_llm.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run28_cyclic_axis_llm.json"

spec = importlib.util.spec_from_file_location(
    "r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r17)

COMPASS = ["north", "northeast", "east", "southeast",
           "south", "southwest", "west", "northwest"]
COMPASS_TEMPLATES = [
    "The hikers traveled {d} across the open plateau",
    "Strong winds from the {d} brought colder weather",
    "The ancient trade route ran {d} through the mountain passes",
    "Migrating birds fly {d} when the seasons change",
    "The river bends {d} before reaching the coastal delta",
    "Explorers charted the {d} territories over several expeditions",
    "The storm system is moving {d} at fifteen miles per hour",
    "A narrow trail leads {d} from the village to the ridge",
    "The army marched {d} for three days without rest",
    "Sailors steered {d} guided by the evening stars",
    "The highway extends {d} connecting the two provinces",
    "Herds of caribou drift {d} during the long winter",
    "The {d} face of the mountain receives the least sunlight",
    "Settlers pushed {d} in search of fertile farmland",
    "The railway line was extended {d} to reach the mining towns",
    "Fishing boats headed {d} out of the harbor at dawn",
]

CLOCK = ["midnight", "three in the morning", "six in the morning",
         "nine in the morning", "noon", "three in the afternoon",
         "six in the evening", "nine at night"]
CLOCK_TEMPLATES = [
    "At {t} the city streets were nearly deserted",
    "The train departs at {t} from the central station",
    "By {t} the market vendors had arranged their stalls",
    "The nurses changed shifts at {t} as usual",
    "Around {t} the temperature dropped noticeably",
    "The bakery starts its ovens at {t} every day",
    "At {t} the fishermen returned with their catch",
    "The meeting was scheduled for {t} in the main hall",
    "By {t} most of the passengers had fallen asleep",
    "The factory whistle sounds at {t} across the valley",
    "At {t} the guards completed another patrol round",
    "The ferry crosses the strait at {t} each day",
    "Around {t} the birds began singing in the garden",
    "The observatory opens its dome at {t} for viewing",
    "By {t} the highway traffic had reached its peak",
    "The radio station broadcasts news at {t} daily",
]


def kernel_analysis(class_means):
    """Class-mean kernel + circulant fit + Fourier spectrum (run24 conv.).

    Class means are CENTERED (grand mean of class means subtracted) before
    the kernel: the uncentered kernel is dominated by the shared mean
    direction (corrcoef ~ 1 everywhere, eff_rank ~ 1) and carries no
    between-class geometry. run24's committed eigenvalues (8 summing to 1,
    top1 0.21 on world_knowledge) certify the centered convention.
    """
    cm = class_means - class_means.mean(axis=0)
    C = np.corrcoef(cm)
    lam = np.linalg.eigvalsh(C)
    lam = lam[lam > 1e-10]
    lam_n = lam / lam.sum()
    eff_rank = float(1.0 / (lam_n ** 2).sum())
    top1 = float(np.sort(lam_n)[-1])
    n = C.shape[0]
    # circulant profile f(D), symmetrized over D and n-D
    f = np.zeros(n)
    for D in range(n):
        f[D] = np.mean([C[i, (i + D) % n] for i in range(n)])
    f_sym = (f + f[::-1].take(range(-1, n - 1), mode="wrap")) / 2  # f[(n-D)%n]
    # variance of off-diagonal C explained by circulant model
    Cc = np.array([[f_sym[(j - i) % n] for j in range(n)] for i in range(n)])
    mask = ~np.eye(n, dtype=bool)
    resid = C[mask] - Cc[mask]
    tot = C[mask] - C[mask].mean()
    r2 = float(1.0 - (resid ** 2).sum() / (tot ** 2).sum()) \
        if (tot ** 2).sum() > 0 else 0.0
    # Fourier spectrum of f_sym (real, even -> cosine coefficients)
    fk = np.fft.rfft(f_sym).real
    return {"eff_rank": eff_rank, "top1_frac": top1,
            "f_profile": f_sym.tolist(), "circulant_r2": r2,
            "fourier": fk.tolist(),
            "kernel": C.tolist()}


def ordered_delta(X, labels, order, rng):
    """Ladder delta accumulating classes in the given order."""
    relab = np.zeros_like(labels)
    for new, old in enumerate(order):
        relab[labels == old] = new
    return r17.ladder_delta(X, relab, 8, rng)


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = {
        "compass": {d: [t.format(d=d) for t in COMPASS_TEMPLATES]
                    for d in COMPASS},
        "clock": {t_: [tpl.format(t=t_) for tpl in CLOCK_TEMPLATES]
                  for t_ in CLOCK},
        "world_knowledge": r17.build_axes()["world_knowledge"],
    }

    print("\nloading Pythia-160m (MPS)...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        "EleutherAI/pythia-160m").to("mps")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    out = {"model": "EleutherAI/pythia-160m", "axes": {}}
    antipodal = [0, 4, 1, 5, 2, 6, 3, 7]

    for axis_name, classes in axes.items():
        class_names = list(classes.keys())
        all_prompts, labels = [], []
        for ci, cn in enumerate(class_names):
            for p in classes[cn]:
                all_prompts.append(p)
                labels.append(ci)
        labels = np.array(labels)
        print(f"\n[{axis_name}] 8 classes, {len(all_prompts)} prompts",
              flush=True)

        per_layer = r17.get_hidden_states(model, tokenizer, all_prompts,
                                          device="mps")

        axis_results = {"class_names": class_names, "layers": []}
        for l in range(len(per_layer)):
            X = per_layer[l].astype(np.float32)
            X = X / (X.std() + 1e-9)
            d, sh = r17.ladder_delta(X, labels, 8,
                                     np.random.default_rng(l * 100 + 1))
            if d is None:
                continue
            means = np.stack([X[labels == c].mean(axis=0) for c in range(8)])
            ka = kernel_analysis(means)
            # order test (same states, different accumulation orders)
            d_anti, _ = ordered_delta(X, labels, antipodal,
                                      np.random.default_rng(l * 100 + 1))
            rand_orders = []
            org = np.random.default_rng(l * 7 + 3)
            for _ in range(20):
                perm = org.permutation(8).tolist()
                dr, _ = ordered_delta(X, labels, perm,
                                      np.random.default_rng(l * 100 + 1))
                if dr is not None:
                    rand_orders.append(dr)
            axis_results["layers"].append({
                "layer": l, "delta": d, "shuffle_mean": sh,
                "delta_antipodal": d_anti,
                "rand_order_mean": float(np.mean(rand_orders)),
                "rand_order_sd": float(np.std(rand_orders)),
                "f1_minus_f4": float(ka["f_profile"][1] - ka["f_profile"][4]),
                **{k: ka[k] for k in
                   ["eff_rank", "top1_frac", "circulant_r2", "f_profile",
                    "fourier"]}})
        out["axes"][axis_name] = axis_results
        mid = axis_results["layers"][len(axis_results["layers"]) // 2]
        print(f"  mid-layer: delta {mid['delta']:+.3f} | "
              f"eff_rank {mid['eff_rank']:.2f} | "
              f"circ_r2 {mid['circulant_r2']:.2f} | "
              f"seq-anti {mid['delta'] - mid['delta_antipodal']:+.3f} | "
              f"f1-f4 {mid['f1_minus_f4']:+.3f}", flush=True)
        json.dump(out, open(OUT, "w"), indent=1)

    # C3 scoreboard across layers/axes
    print("\nC3 entry-coherence check (sign(seq-anti) vs sign(f1-f4)):",
          flush=True)
    for axis_name in ["compass", "clock"]:
        hits = tot = 0
        for lr in out["axes"][axis_name]["layers"][1:]:  # skip embedding
            pred = np.sign(lr["f1_minus_f4"])
            obs = np.sign(lr["delta"] - lr["delta_antipodal"])
            if pred != 0 and obs != 0:
                tot += 1
                hits += int(pred == obs)
        print(f"  {axis_name}: {hits}/{tot}", flush=True)

    json.dump(out, open(OUT, "w"), indent=1)
    print("\nDONE run28", flush=True)


if __name__ == "__main__":
    main()
