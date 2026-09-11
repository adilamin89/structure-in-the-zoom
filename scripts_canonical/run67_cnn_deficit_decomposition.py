"""Run 67 (S79, 2026-09-06) - The four-term split of the deficit in the
equivariant network: is the rotation shift carried by the four rotation
means, or by rotation-private variability as in cortex?

WHY: in V1 the direction-aligned shift is the pooling of class-private
within-class modes (run 63) and those modes are localized (run 64a). The
network of Section 6.2 is the ground-truth case where the within-class
variability is known: the same test images at four rotations, so the
within-class variability is the image content, and what changes between
classes is the rotation. If the shift there is carried by the between-class
term, the private-mode statement about cortex is specific to cortex; if the
pooling term carries it there too, the anatomy is generic to any rotated
code and Section 7's contrast must say so.

DESIGN: run 5b's design, retrained (the features were never saved): five
seeds, 20,000 training images at random C4 rotations, two epochs, the plain
net and the orbit-sharing equivariant net; 4,000 test images at balanced
rotations; features at conv1-3 (plain), conv1-3 equivariant (the four
per-orientation stacks) and conv1-3 invariant (their mean over the orbit).
Per seed and layer: the rotation ladder (four classes in order 0, 90, 180,
270 degrees; rungs 1-4) with ten random-subset floors, the four terms per
rung, the four slopes of the shift, the private slope (P against log(k/4) on
rungs 1-3) and five label permutations as the sampling baseline; run 5's own
ladder_delta as the reproduction check against run 5b's stored means; the
digit ladder (eight digits, rungs [1, 2, 3, 4, 6, 8], three permutations)
reported for scale. Means and SDs over the five seeds. Features saved per
seed as float16 (data_canonical/run67_features/, not committed).

REGISTERED EXPECTATIONS (written before the run; seed means of the excess
over the label shuffle):
C1 (the rotation means carry the shift): on the four layer sets where run 5b
    measured a rotation shift above 0.05 (plain conv2, plain conv3,
    equivariant conv2, equivariant conv3), the between-class term is the
    largest-magnitude term of the four and exceeds the pooling term, on all
    four.
C2 (invariance kills every term): on the three invariant layers every
    term's seed-mean magnitude is below 0.02.
C3 (no rotation-private variability): the rotation ladder's excess private
    slope is below 0.3 on all six plain and equivariant layers.
Reproduction check (reported): the recomputed delta_rot's seed mean lies
    within two stored SDs + 0.02 of run 5b's on all nine layer sets.
A miss is reported at full volume.

POST-RUN NOTE (first pass): the retrain reproduced run 5b's nine shifts to
three decimals; C2 and C3 passed; C1 missed because the rotation means
enter the participation ratio through the TRACE (2 log Tr C), not through
Tr B^2: weak means add variance linearly and their second moment is
negligible against the fluctuations'. The scale-free regrouping
A + P = D + Tb (deficit_split.py) names that term Tb, the between-class
trace share, and is reported alongside, descriptively, from the saved
features (--from-saved). C1 is judged on the term it named.

Out: ../data_canonical/run67_cnn_deficit_decomposition.json (+ .log)
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data_canonical"
OUT = DATA / "run67_cnn_deficit_decomposition.json"
FEAT = DATA / "run67_features"
sys.path.insert(0, str(HERE))
from deficit_split import FREE, decompose, largest_term  # noqa: E402

spec = importlib.util.spec_from_file_location("run5", HERE / "run5_equivariant_cnn.py")
r5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r5)

N_SEEDS = 5
NTE, NTR = 4000, 20000
ROT_BINS = (1, 2, 3, 4)
DIGIT_BINS = (1, 2, 3, 4, 6, 8)
N_NULL = 10
N_SHUFFLE_ROT = 5
N_SHUFFLE_DIG = 3
KEYS = ("A", "P", "Cb", "Cx")
POSITIVE_SETS = ["plain/conv2", "plain/conv3", "equivariant/conv2_equi", "equivariant/conv3_equi"]
INV_SETS = ["equivariant/conv1_inv", "equivariant/conv2_inv", "equivariant/conv3_inv"]
EQUI_SETS = ["plain/conv1", "plain/conv2", "plain/conv3", "equivariant/conv1_equi", "equivariant/conv2_equi", "equivariant/conv3_equi"]


def features_for_seed(seed, m):
    """run 5b's data draw and training, verbatim; returns the per-layer test
    features, the rotation labels and the digit labels."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr_idx = rng.choice(len(m["train_x"]), NTR, replace=False)
    Xtr = m["train_x"][tr_idx].astype(np.float32) / 255.0
    ytr = m["train_y"][tr_idx].astype(np.int64)
    rot_tr = rng.integers(0, r5.N_ROT, NTR)
    Xtr = np.stack([np.rot90(x, g) for x, g in zip(Xtr, rot_tr)])
    te_idx = rng.choice(len(m["test_x"]), NTE, replace=False)
    Xte = m["test_x"][te_idx].astype(np.float32) / 255.0
    yte = m["test_y"][te_idx].astype(np.int64)
    rot_te = np.repeat(np.arange(r5.N_ROT), NTE // r5.N_ROT)
    rng.shuffle(rot_te)
    Xte = np.stack([np.rot90(x, g) for x, g in zip(Xte, rot_te)])
    Xtr_t = torch.tensor(np.ascontiguousarray(Xtr)).unsqueeze(1)
    Xte_t = torch.tensor(np.ascontiguousarray(Xte)).unsqueeze(1)
    ytr_t = torch.tensor(ytr)
    feats = {}
    for name, Net in [("plain", r5.PlainNet), ("equivariant", r5.EquivNet)]:
        model = Net().to(r5.DEV)
        r5.train(model, Xtr_t, ytr_t)
        model.eval()
        with torch.no_grad():
            for i in range(0, NTE, 512):
                o = model(Xte_t[i:i + 512].to(r5.DEV))
                if name == "plain":
                    for l, f in enumerate(o[1]):
                        feats.setdefault(f"plain/conv{l+1}", []).append(f.cpu().numpy())
                else:
                    for l, f in enumerate(o[1]):
                        feats.setdefault(f"equivariant/conv{l+1}_equi", []).append(f.cpu().numpy())
                    for l, f in enumerate(o[2]):
                        feats.setdefault(f"equivariant/conv{l+1}_inv", []).append(f.cpu().numpy())
        del model
    return {k: np.concatenate(v) for k, v in feats.items()}, rot_te, yte


def load_saved(seed):
    z = np.load(FEAT / f"seed{seed}.npz")
    feats = {k.replace("__", "/"): z[k].astype(np.float32) for k in z.files if k not in ("rot", "digit")}
    return feats, z["rot"], z["digit"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--from-saved", action="store_true", help="decompose the saved per-seed features instead of retraining")
    args = ap.parse_args()
    t0 = time.time()
    FEAT.mkdir(exist_ok=True)
    stored = json.load(open(DATA / "run5b_cnn_seeds.json"))
    m = None if args.from_saved else r5.load_mnist()
    per_seed = {}
    for seed in range(N_SEEDS):
        if args.from_saved:
            feats, rot_te, yte = load_saved(seed)
        else:
            feats, rot_te, yte = features_for_seed(seed, m)
            np.savez_compressed(FEAT / f"seed{seed}.npz", rot=rot_te, digit=yte, **{k.replace("/", "__"): v.astype(np.float16) for k, v in feats.items()})
        keep = yte < 8
        for layer, X in feats.items():
            X = X / (X.std() + 1e-9)
            rot = decompose(X, rot_te, list(range(r5.N_ROT)), ROT_BINS, n_null=N_NULL, seed=1, n_shuffle=N_SHUFFLE_ROT)
            d_rot_run5, _ = r5.ladder_delta(X, rot_te, r5.ROT_RUNGS, r5.N_ROT, np.random.default_rng(1), n_shuf=0)
            dig = decompose(X[keep], yte[keep], list(range(8)), DIGIT_BINS, n_null=N_NULL, seed=2, n_shuffle=N_SHUFFLE_DIG)
            per_seed.setdefault(layer, []).append({"seed": seed, "rot": rot, "delta_rot_run5": float(d_rot_run5), "digit": dig})
            ex = rot["excess"]["delta_split"]; dx = dig["excess"]["delta_split"]
            print(f"seed{seed} {layer:24s}: rot delta {rot['delta']:+.3f} (run5 {d_rot_run5:+.3f}, shuffle {rot['shuffle']['delta']:+.3f}) | excess A {ex['A']:+.3f} P {ex['P']:+.3f} "
                  f"Cb {ex['Cb']:+.3f} Cx {ex['Cx']:+.3f} -> {largest_term(ex)} | free D {ex['D']:+.3f} Tb {ex['Tb']:+.3f} S {ex['S']:+.3f} -> {largest_term(ex, FREE)} | dim slope {rot['excess']['private_slope_dim']:+.2f} | "
                  f"digit delta {dig['delta']:+.3f} free D {dx['D']:+.3f} Tb {dx['Tb']:+.3f} S {dx['S']:+.3f} dim slope {dig['excess']['private_slope_dim']:+.2f} | {time.time()-t0:.0f}s", flush=True)
        del feats
    out = {"design": {"n_seeds": N_SEEDS, "rot_bins": ROT_BINS, "digit_bins": DIGIT_BINS, "n_null": N_NULL, "n_shuffle_rot": N_SHUFFLE_ROT, "n_shuffle_digit": N_SHUFFLE_DIG},
           "layers": {}}
    for layer, rows in per_seed.items():
        def ms(get):
            v = np.array([get(r) for r in rows], float); return {"mean": float(v.mean()), "sd": float(v.std())}
        summ = {"delta_rot": ms(lambda r: r["rot"]["delta"]), "delta_rot_run5": ms(lambda r: r["delta_rot_run5"]), "delta_rot_shuffle": ms(lambda r: r["rot"]["shuffle"]["delta"]),
                "delta_rot_stored_run5b": {"mean": stored[layer]["delta_rot_mean"], "sd": stored[layer]["delta_rot_sd"]},
                "rot_split": {k: ms(lambda r, k=k: r["rot"]["delta_split"][k]) for k in KEYS},
                "rot_split_excess": {k: ms(lambda r, k=k: r["rot"]["excess"]["delta_split"][k]) for k in KEYS},
                "rot_split_free_excess": {k: ms(lambda r, k=k: r["rot"]["excess"]["delta_split"][k]) for k in FREE + ("S",)},
                "rot_private_slope_dim_excess": ms(lambda r: r["rot"]["excess"]["private_slope_dim"]),
                "digit_split_free_excess": {k: ms(lambda r, k=k: r["digit"]["excess"]["delta_split"][k]) for k in FREE + ("S",)},
                "digit_private_slope_dim_excess": ms(lambda r: r["digit"]["excess"]["private_slope_dim"]),
                "rot_private_slope": ms(lambda r: r["rot"]["private_slope"]), "rot_private_slope_excess": ms(lambda r: r["rot"]["excess"]["private_slope"]),
                "rot_rungs_P_excess": [ms(lambda r, i=i: r["rot"]["rungs"][i]["P"] - r["rot"]["shuffle"]["rungs"][i]["P"]) for i in range(len(ROT_BINS))],
                "rot_rungs_Cb_excess": [ms(lambda r, i=i: r["rot"]["rungs"][i]["Cb"] - r["rot"]["shuffle"]["rungs"][i]["Cb"]) for i in range(len(ROT_BINS))],
                "digit_delta": ms(lambda r: r["digit"]["delta"]), "digit_delta_stored_run5b": {"mean": stored[layer]["delta_digit_mean"], "sd": stored[layer]["delta_digit_sd"]},
                "digit_split_excess": {k: ms(lambda r, k=k: r["digit"]["excess"]["delta_split"][k]) for k in KEYS},
                "digit_private_slope_excess": ms(lambda r: r["digit"]["excess"]["private_slope"]),
                "max_sum_check": max(r["rot"]["max_sum_check"] for r in rows), "max_pr_identity_rel": max(r["rot"]["max_pr_identity_rel"] for r in rows)}
        summ["rot_largest_excess_term"] = largest_term({k: summ["rot_split_excess"][k]["mean"] for k in KEYS})
        summ["rot_largest_free_term"] = largest_term({k: summ["rot_split_free_excess"][k]["mean"] for k in FREE}, FREE)
        summ["digit_largest_free_term"] = largest_term({k: summ["digit_split_free_excess"][k]["mean"] for k in FREE}, FREE)
        out["layers"][layer] = {"summary": summ, "seeds": rows}
        e = summ["rot_split_excess"]; f = summ["rot_split_free_excess"]; g = summ["digit_split_free_excess"]
        print(f"{layer:24s}: rot delta {summ['delta_rot']['mean']:+.3f}±{summ['delta_rot']['sd']:.3f} (run5b {stored[layer]['delta_rot_mean']:+.3f}±{stored[layer]['delta_rot_sd']:.3f}) | excess "
              f"A {e['A']['mean']:+.3f} P {e['P']['mean']:+.3f} Cb {e['Cb']['mean']:+.3f} Cx {e['Cx']['mean']:+.3f} -> {summ['rot_largest_excess_term']} | free D {f['D']['mean']:+.3f}±{f['D']['sd']:.3f} Tb {f['Tb']['mean']:+.3f}±{f['Tb']['sd']:.3f} "
              f"S {f['S']['mean']:+.3f} -> {summ['rot_largest_free_term']} dim slope {summ['rot_private_slope_dim_excess']['mean']:+.2f} | digit {summ['digit_delta']['mean']:+.3f} free D {g['D']['mean']:+.3f} Tb {g['Tb']['mean']:+.3f} S {g['S']['mean']:+.3f} "
              f"-> {summ['digit_largest_free_term']} dim slope {summ['digit_private_slope_dim_excess']['mean']:+.2f}", flush=True)
    L = out["layers"]
    c1 = {s: bool(L[s]["summary"]["rot_largest_excess_term"] == "Cb" and L[s]["summary"]["rot_split_excess"]["Cb"]["mean"] > L[s]["summary"]["rot_split_excess"]["P"]["mean"]) for s in POSITIVE_SETS}
    c2 = {s: bool(max(abs(L[s]["summary"]["rot_split_excess"][k]["mean"]) for k in KEYS) < 0.02) for s in INV_SETS}
    c3 = {s: bool(L[s]["summary"]["rot_private_slope_excess"]["mean"] < 0.3) for s in EQUI_SETS}
    rep = {s: bool(abs(L[s]["summary"]["delta_rot"]["mean"] - stored[s]["delta_rot_mean"]) <= 2 * stored[s]["delta_rot_sd"] + 0.02) for s in L}
    verdict = {"C1_per_set": c1, "C1": bool(all(c1.values())), "C2_per_set": c2, "C2": bool(all(c2.values())), "C3_per_set": c3, "C3": bool(all(c3.values())),
               "reproduction_per_set": rep, "reproduction": bool(all(rep.values())),
               "largest_excess_term": {s: L[s]["summary"]["rot_largest_excess_term"] for s in L},
               "refined_descriptive": {"rot_largest_free_term": {s: L[s]["summary"]["rot_largest_free_term"] for s in L},
                                       "digit_largest_free_term": {s: L[s]["summary"]["digit_largest_free_term"] for s in L},
                                       "rot_Tb_excess_mean": {s: L[s]["summary"]["rot_split_free_excess"]["Tb"]["mean"] for s in L},
                                       "rot_D_excess_mean": {s: L[s]["summary"]["rot_split_free_excess"]["D"]["mean"] for s in L},
                                       "digit_D_excess_mean": {s: L[s]["summary"]["digit_split_free_excess"]["D"]["mean"] for s in L},
                                       "digit_Tb_excess_mean": {s: L[s]["summary"]["digit_split_free_excess"]["Tb"]["mean"] for s in L}},
               "max_sum_check": max(L[s]["summary"]["max_sum_check"] for s in L), "max_pr_identity_rel": max(L[s]["summary"]["max_pr_identity_rel"] for s in L)}
    out["verdict"] = verdict
    print("VERDICT", json.dumps(verdict, indent=1), flush=True)
    json.dump(out, open(OUT, "w"), indent=1); print(f"wrote {OUT} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
