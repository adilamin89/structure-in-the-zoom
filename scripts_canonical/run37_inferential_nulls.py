"""Run 37 — Inferential permutation nulls + order-averaged delta for the LLM battery.

WHY: runs 17-32 report sig/shuffle ratios from N_SHUF=5 label shuffles —
descriptive, not inferential (round-4 item 3). This run re-encodes
world_knowledge + language_type (prompts byte-identical via run17.build_axes())
on Pythia-160m and Pythia-2.8b-deduped and computes:

  (A) per-layer permutation null: N_PERM=500 label permutations -> z,
      permutation quantile, and two-sided p for the canonical-order delta
      at every layer (replaces the sig/shuffle ratio);
  (B) profile-level inference on PRE-DECLARED stats: integrated excess
      IE = mean_{l>=1}(delta_l - delta_0) and peak excess
      PE = max_{l>=1}(delta_l - delta_0), one-sided p in the pre-declared
      direction (language_type: IE>0, PE>0 — structure built with depth;
      world_knowledge: IE<0 — content diluted with depth);
  (C) order-averaged headline for UNORDERED axes (round-4 item 2):
      deltabar_l +/- SD over K_ORDERS=50 random class-accumulation orders;
      ordered ladders remain only for direction/SF/time/clock/compass;
  (D) order-matched null: IE/PE of the order-averaged profile against
      permutations averaged over K_NULL_ORDERS=20 orders each (the K
      mismatch adds Monte Carlo spread to each null draw -> conservative).

REGISTERED EXPECTATIONS (written before the run):
P1: per-layer z reproduces run32's CI sign structure on 2.8B language_type
    (negative early layers, positive late) and world_knowledge positive at
    most layers, both models.
P2: language_type IE and PE significantly positive (p<0.01) on both models;
    world_knowledge IE significantly negative on 2.8B (declining profile);
    same direction expected at 160m (weaker).
P3: order-averaging preserves profile SHAPE (declining wk / rising lt);
    SD over orders small relative to the profile range.
P4: order-averaged IE/PE remain significant wherever P2 passes.

FLOOR: label-independent, N_NULL_FLOOR=20 random-subset draws (run17
construction), computed once per layer and shared by observed and permuted
deltas — permuting labels cannot move the floor.

GRAM TRICK: the full n x n linear kernel K = X X^T is computed once per
layer; every subset PR is then O(n^2) via double-centering the submatrix
(PR = Tr(Kc)^2 / sum(Kc*Kc), the 1/(n-1) factors cancel). Validated
in-script against run17.pr_c (rel diff < 1e-3) before use.

Out: ../data_canonical/run37_inferential_nulls.json
"""
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run37_inferential_nulls.json"

N_PERM = 500
K_ORDERS = 50
K_NULL_ORDERS = 20
N_NULL_FLOOR = 20
BIN_COUNTS = [1, 2, 3, 4, 6, 8]

spec = importlib.util.spec_from_file_location(
    "r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r17)

MODELS = [
    ("EleutherAI/pythia-160m", "cpu"),
    ("EleutherAI/pythia-2.8b-deduped", "mps"),
]
AXES = ["world_knowledge", "language_type"]
DECLARED_DIRECTION = {  # one-sided direction for IE; PE always tested >0 for lt
    "world_knowledge": {"IE": -1, "PE": +1},
    "language_type": {"IE": +1, "PE": +1},
}


def subset_pr(K, idx):
    """PR of the row-centered subset via the precomputed full Gram."""
    Ks = K[np.ix_(idx, idx)]
    rm = Ks.mean(axis=1)
    tm = rm.mean()
    # double-centering: Kc = H Ks H with H = I - J/n
    Kc = Ks - rm[:, None] - rm[None, :] + tm
    tr = float(np.trace(Kc))
    tr2 = float((Kc * Kc).sum())
    return tr * tr / tr2 if tr2 > 0 else 1.0


def ladder_slope(K, members, order, sizes_cache=None):
    """theta_obs slope for accumulating classes in `order` (list of class ids)."""
    sizes, prs = [], []
    for c in BIN_COUNTS:
        sel = np.concatenate([members[o] for o in order[:c]])
        sizes.append(len(sel))
        prs.append(subset_pr(K, sel))
    return r17.slope(sizes, np.asarray(prs)), sizes


def floor_slope(K, n_total, sizes, rng):
    nl = np.zeros((N_NULL_FLOOR, len(sizes)))
    for d in range(N_NULL_FLOOR):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(subset_pr(K, rng.choice(n_total, s,
                                                          replace=False)),
                                  1e-9))
    return r17.slope(sizes, np.exp(nl.mean(axis=0)))


def profile_stats(deltas):
    e = np.asarray(deltas[1:]) - deltas[0]
    return float(e.mean()), float(e.max())


def one_sided_p(obs, null, direction):
    null = np.asarray(null)
    if direction > 0:
        return float((1 + (null >= obs).sum()) / (len(null) + 1))
    return float((1 + (null <= obs).sum()) / (len(null) + 1))


def analyze_axis(per_layer, labels, n_classes, axis_name):
    """All four components for one (model, axis). per_layer: list of (n,d)."""
    members0 = [np.where(labels == c)[0] for c in range(n_classes)]
    n_total = len(labels)
    canonical = list(range(n_classes))

    # pre-draw permutations and orders once (shared across layers)
    prng = np.random.default_rng(3700)
    perms = [prng.permutation(n_total) for _ in range(N_PERM)]
    orng = np.random.default_rng(3701)
    orders = [orng.permutation(n_classes).tolist() for _ in range(K_ORDERS)]
    null_orders = orders[:K_NULL_ORDERS]

    layers_out = []
    obs_profile, obs_profile_avg = [], []
    perm_profiles = np.zeros((N_PERM, len(per_layer)))
    perm_profiles_avg = np.zeros((N_PERM, len(per_layer)))

    for l in range(len(per_layer)):
        t0 = time.time()
        X = per_layer[l].astype(np.float64)
        X = X / (X.std() + 1e-9)
        K = X @ X.T

        if l == 0:
            # validate Gram trick vs direct pr_c on 3 subsets
            vrng = np.random.default_rng(7)
            for s in (16, 48, 128):
                idx = vrng.choice(n_total, s, replace=False)
                a = subset_pr(K, idx)
                b = r17.pr_c(X[idx])
                assert abs(a - b) / max(abs(b), 1e-9) < 1e-3, (a, b)
            print(f"    [{axis_name}] gram-trick validated", flush=True)

        th_obs, sizes = ladder_slope(K, members0, canonical)
        frng = np.random.default_rng(l * 100 + 1)
        th_f = floor_slope(K, n_total, sizes, frng)
        d_obs = th_obs - th_f
        obs_profile.append(d_obs)

        # (C) order-averaged observed delta
        d_orders = [ladder_slope(K, members0, o)[0] - th_f for o in orders]
        obs_profile_avg.append(float(np.mean(d_orders)))
        d_order_sd = float(np.std(d_orders))

        # (A) canonical permutation null + (D) order-averaged null
        for p, perm in enumerate(perms):
            pl = labels[perm]
            mem = [np.where(pl == c)[0] for c in range(n_classes)]
            perm_profiles[p, l] = ladder_slope(K, mem, canonical)[0] - th_f
            perm_profiles_avg[p, l] = np.mean(
                [ladder_slope(K, mem, o)[0] for o in null_orders]) - th_f

        null_l = perm_profiles[:, l]
        nm, ns = float(null_l.mean()), float(null_l.std())
        z = (d_obs - nm) / ns if ns > 0 else 0.0
        quant = float((null_l <= d_obs).mean())
        p_two = float((1 + (np.abs(null_l - nm) >= abs(d_obs - nm)).sum())
                      / (N_PERM + 1))
        # per-layer inference for the order-averaged headline (same perms,
        # each averaged over K_NULL_ORDERS orders)
        null_a = perm_profiles_avg[:, l]
        nma, nsa = float(null_a.mean()), float(null_a.std())
        d_avg = obs_profile_avg[-1]
        z_avg = (d_avg - nma) / nsa if nsa > 0 else 0.0
        p_two_avg = float((1 + (np.abs(null_a - nma) >= abs(d_avg - nma)).sum())
                          / (N_PERM + 1))
        layers_out.append({
            "layer": l, "delta": float(d_obs),
            "delta_orderavg": d_avg,
            "delta_orderavg_sd": d_order_sd,
            "null_mean": nm, "null_sd": ns,
            "z": float(z), "quantile": quant, "p_two": p_two,
            "null_mean_orderavg": nma, "null_sd_orderavg": nsa,
            "z_orderavg": float(z_avg), "p_two_orderavg": p_two_avg})
        print(f"    L{l:02d} delta={d_obs:+.3f} z={z:+.1f} p2={p_two:.4f} "
              f"| avg={obs_profile_avg[-1]:+.3f}±{d_order_sd:.3f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    # (B)+(D) profile-level stats
    stats = {}
    for tag, prof, nulls in (
            ("canonical", obs_profile, perm_profiles),
            ("orderavg", obs_profile_avg, perm_profiles_avg)):
        ie_obs, pe_obs = profile_stats(prof)
        null_ie = [profile_stats(nulls[p])[0] for p in range(N_PERM)]
        null_pe = [profile_stats(nulls[p])[1] for p in range(N_PERM)]
        for name, obs, null in (("IE", ie_obs, null_ie), ("PE", pe_obs, null_pe)):
            dirn = DECLARED_DIRECTION[axis_name][name]
            ns_ = float(np.std(null))
            stats[f"{name}_{tag}"] = {
                "obs": obs, "direction": dirn,
                "null_mean": float(np.mean(null)), "null_sd": ns_,
                "z": (obs - float(np.mean(null))) / ns_ if ns_ > 0 else 0.0,
                "p_one": one_sided_p(obs, null, dirn)}
    return {"n_classes": n_classes, "n_prompts": n_total,
            "layers": layers_out, "profile_stats": stats}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes_all = r17.build_axes()
    axes = {a: axes_all[a] for a in AXES}

    out = {"n_perm": N_PERM, "k_orders": K_ORDERS,
           "k_null_orders": K_NULL_ORDERS, "n_null_floor": N_NULL_FLOOR,
           "design": "run17 prompts byte-identical", "models": {}}
    if OUT.exists():
        out = json.load(open(OUT))
        print(f"resumed: {[(m, list(v['axes'])) for m, v in out['models'].items()]}",
              flush=True)

    for model_name, device in MODELS:
        mkey = model_name.split("/")[-1]
        out["models"].setdefault(mkey, {"axes": {}})
        if all(a in out["models"][mkey]["axes"] for a in AXES):
            continue
        print(f"\nloading {model_name} ({device})...", flush=True)
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        print(f"  {model.config.num_hidden_layers} layers", flush=True)

        for axis_name in AXES:
            if axis_name in out["models"][mkey]["axes"]:
                continue
            classes = axes[axis_name]
            class_names = list(classes.keys())
            prompts, labels = [], []
            for ci, cn in enumerate(class_names):
                for p in classes[cn]:
                    prompts.append(p)
                    labels.append(ci)
            labels = np.array(labels)
            print(f"\n[{mkey} / {axis_name}] {len(class_names)} classes, "
                  f"{len(prompts)} prompts — encoding...", flush=True)
            per_layer = r17.get_hidden_states(model, tokenizer, prompts,
                                              device=device)
            res = analyze_axis(per_layer, labels, len(class_names), axis_name)
            res["class_names"] = class_names
            out["models"][mkey]["axes"][axis_name] = res
            json.dump(out, open(OUT, "w"), indent=1)
            ps = res["profile_stats"]
            print(f"  => IE canon {ps['IE_canonical']['obs']:+.3f} "
                  f"(p={ps['IE_canonical']['p_one']:.4f}) | "
                  f"PE canon {ps['PE_canonical']['obs']:+.3f} "
                  f"(p={ps['PE_canonical']['p_one']:.4f}) | "
                  f"IE avg p={ps['IE_orderavg']['p_one']:.4f} | "
                  f"PE avg p={ps['PE_orderavg']['p_one']:.4f}", flush=True)

        del model
        if device == "mps":
            torch.mps.empty_cache()

    print("\nDONE run37", flush=True)


if __name__ == "__main__":
    main()
