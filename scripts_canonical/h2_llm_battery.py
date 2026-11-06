"""H2 language-model battery: order-averaged shifts with their own nulls (registrations H2-R1..R8, ex0_REGISTERED.md).

The estimator is run37's (global-std scaling, linear Gram, double-centred subset PR, class rungs, 20-draw floor with
seed l*100+1). The C9 finding (App I) is that for an axis with no natural order the declared-order profile belongs to
the path; this script therefore reports, per layer, the shift averaged over random class orders against the null of
that same statistic (label permutations, each averaged over the same number of orders), the declared path's percentile
among orders, the order-averaged depth profile against its null, the spectrum check (z-scored and top-4-PC-removed
estimators, run55), and second nulls (topic-stratified for construction, run45; within-pair swaps for BLiMP, run42
design B, and Baroni 64 pairs, run43b).

Usage:
  python h2_llm_battery.py encode <hf_model> <device> [dtype]   -> raw/llm_states/<model>__<battery>.npz
  python h2_llm_battery.py analyse <model_key> [battery ...]     -> data_canonical/h2_llm_battery_<model_key>.json
Batteries: world_knowledge, language_type (run17.build_axes, byte-identical), blimp_B (run42), baroni64 (run43b's stored
prompts), cyclic batteries added by h2_cyclic_axes.py.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATES = ROOT / "raw" / "llm_states"   # written by encode (several GB, not shipped)
OUTDIR = ROOT / "data_canonical"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r37 = _load("r37", "run37_inferential_nulls.py")
r17 = r37.r17

N_ORDERS = 200          # headline order average
N_NULL = 200            # null draws of the order-averaged statistic
K_NULL_ORDERS = 50      # orders per null draw (equal to the 50-order observed statistic)
N_FLOOR = 20


# ------------------------------------------------------------------ batteries
def battery_prompts(name):
    """Return prompts, labels, pairs flag, class names, rung counts."""
    if name in ("world_knowledge", "language_type"):
        cache = STATES / "run17_axes.json"
        if cache.exists():
            axes = json.load(open(cache))
        else:
            all_axes = r17.build_axes()
            axes = {a: all_axes[a] for a in ("world_knowledge", "language_type")}
            STATES.mkdir(parents=True, exist_ok=True)
            json.dump(axes, open(cache, "w"), indent=1)
        classes = axes[name]
        names = list(classes.keys())
        prompts, labels = [], []
        for ci, cn in enumerate(names):
            for p in classes[cn]:
                prompts.append(p)
                labels.append(ci)
        return prompts, np.array(labels), names, [1, 2, 3, 4, 6, 8]
    if name in ("world_knowledge32", "language_type32"):   # H2-R22: run17's 16 + run23's 16 per class, byte-identical
        base, _, names, counts = battery_prompts(name[:-2])[0], None, battery_prompts(name[:-2])[2], [1, 2, 3, 4, 6, 8]
        r23 = _load("r23", "run23_expanded_prompts.py")
        extra = r23.EXTRA_WORLD if name.startswith("world") else r23.EXTRA_LANG
        assert list(extra.keys()) == names, (list(extra.keys()), names)
        prompts, labels = [], []
        for ci, cn in enumerate(names):
            for p in base[16 * ci:16 * ci + 16] + extra[cn]:
                prompts.append(p)
                labels.append(ci)
        assert all(np.bincount(labels) == 32)
        return prompts, np.array(labels), names, counts
    if name == "blimp_B":
        r42 = _load("r42", "run42_blimp_battery.py")
        _, b_pairs = r42.load_blimp()
        prompts, labels, names = [], [], []
        for ph, p in enumerate(r42.B_PARADIGMS):
            names += [f"{p}_good", f"{p}_bad"]
            for good, bad in b_pairs[p]:
                prompts += [good, bad]
                labels += [2 * ph, 2 * ph + 1]
        return prompts, np.array(labels), names, [1, 2, 3, 4, 6, 8]
    if name == "baroni64":
        art = json.load(open(OUTDIR / "run43b_baroni_64pairs.json"))
        return list(art["prompts"]), np.array(art["labels"]), list(art["class_names"]), [1, 2, 3, 4, 6]
    if name.startswith("cyclic_"):
        cyc = _load("h2cyc", "h2_cyclic_axes.py")
        return cyc.battery(name)
    raise ValueError(name)


PAIRED = {"blimp_B", "baroni64"}


# ------------------------------------------------------------------ encoding
def encode(model_name, device, dtype_name=None, batteries=("world_knowledge", "language_type", "blimp_B", "baroni64"),
           key=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = {"fp16": torch.float16, "fp32": torch.float32, None: None}[dtype_name]
    if dtype is None:
        dtype = torch.float16 if device == "mps" else torch.float32
    tag = "fp16" if dtype == torch.float16 else "fp32"
    mkey = key or model_name.split("/")[-1]
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    STATES.mkdir(parents=True, exist_ok=True)
    for b in batteries:
        out = STATES / f"{mkey}__{b}.npz"
        if out.exists():
            print(f"skip {out.name}", flush=True)
            continue
        prompts, labels, names, _ = battery_prompts(b)
        t0 = time.time()
        per_layer = r17.get_hidden_states(model, tok, prompts, device=device)
        H = np.stack(per_layer)
        np.savez(out, H=H, labels=labels, names=np.array(names), prompts=np.array(prompts),
                 model=model_name, device=device, dtype=tag)
        print(f"{mkey} {b}: H {H.shape} {H.dtype} ({time.time() - t0:.0f}s)", flush=True)
    return mkey


# ------------------------------------------------------------------ estimator
from numba import njit  # noqa: E402


@njit(cache=True)
def subset_pr_fast(K, idx):
    """run37.subset_pr in one pass: for Kc = H Ks H, Tr Kc = Tr Ks - S/n and ||Kc||^2 = ||Ks||^2 - 2||Ks 1||^2/n + S^2/n^2."""
    n = idx.shape[0]
    tr = 0.0
    S = 0.0
    Q = 0.0
    R2 = 0.0
    for a in range(n):
        ia = idx[a]
        tr += K[ia, ia]
        r = 0.0
        for b in range(n):
            v = K[ia, idx[b]]
            r += v
            Q += v * v
        S += r
        R2 += r * r
    trc = tr - S / n
    q = Q - 2.0 * R2 / n + S * S / (n * n)
    return trc * trc / q if q > 0 else 1.0


def ladder(K, members, order, counts):
    sizes, prs = [], []
    for c in counts:
        sel = np.concatenate([members[o] for o in order[:c]])
        sizes.append(len(sel))
        prs.append(subset_pr_fast(K, sel))
    return r17.slope(sizes, np.asarray(prs)), sizes


def floor(K, n, sizes, seed):
    rng = np.random.default_rng(seed)
    nl = np.zeros((N_FLOOR, len(sizes)))
    for d in range(N_FLOOR):
        for k, s in enumerate(sizes):
            nl[d, k] = np.log(max(subset_pr_fast(K, rng.choice(n, s, replace=False)), 1e-9))
    return r17.slope(sizes, np.exp(nl.mean(axis=0)))


def members_of(lab, n_classes):
    return [np.where(lab == c)[0] for c in range(n_classes)]


def order_avg(K, mem, orders, counts, th_f):
    return float(np.mean([ladder(K, mem, o, counts)[0] for o in orders])) - th_f


def gram(X, variant):
    X = X.astype(np.float64)
    if variant == "linear":   # run37: global std scaling (PR is scale invariant)
        X = X / (X.std() + 1e-9)
        return X @ X.T
    Xc = X - X.mean(axis=0, keepdims=True)
    if variant == "zscore":
        Z = Xc / (Xc.std(axis=0, keepdims=True) + 1e-9)
        return Z @ Z.T
    if variant == "pc4":
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        Xr = Xc - (U[:, :4] * S[:4]) @ Vt[:4]
        return Xr @ Xr.T
    raise ValueError(variant)


def spectrum(X):
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    return {"d_eff": float(lam.sum() ** 2 / (lam ** 2).sum()), "top1_share": float(lam[0] / lam.sum())}


def null_labels(kind, labels, rng, strata=None):
    if kind == "perm":
        return labels[rng.permutation(len(labels))]
    if kind == "swap":
        sl = labels.copy()
        for i in np.where(rng.integers(0, 2, len(labels) // 2))[0]:
            sl[2 * i], sl[2 * i + 1] = sl[2 * i + 1], sl[2 * i]
        return sl
    if kind == "strat":
        sl = labels.copy()
        for idx in strata.values():
            sl[idx] = sl[idx[rng.permutation(len(idx))]]
        return sl
    raise ValueError(kind)


def analyse_battery(H, labels, names, counts, battery, variants=("zscore", "pc4")):
    n_layers, n = H.shape[0], H.shape[1]
    nc = len(names)
    mem0 = members_of(labels, nc)
    canonical = list(range(nc))
    # shared draws across layers (profiles need them)
    orng = np.random.default_rng(3701)
    orders50 = [orng.permutation(nc).tolist() for _ in range(50)]        # run37 / run55 (first 20)
    orng2 = np.random.default_rng(3801)
    orders200 = [orng2.permutation(nc).tolist() for _ in range(N_ORDERS)]
    kinds = ["perm"]
    strata = None
    if battery in PAIRED:
        kinds.append("swap")
    if battery == "language_type":
        r45 = _load("r45", "run45_lt_stratified_floor.py")
        assert list(r45.TOPICS.keys()) == list(names), names
        st = np.array([t for cn in names for t in r45.TOPICS[cn]])
        strata = {t: np.where(st == t)[0] for t in np.unique(st)}
        kinds.append("strat")
    draws = {}
    for ki, kind in enumerate(kinds):
        rng = np.random.default_rng(4800 + ki)
        draws[kind] = [(null_labels(kind, labels, rng, strata),
                        [rng.permutation(nc).tolist() for _ in range(K_NULL_ORDERS)]) for _ in range(N_NULL)]
    for kind in kinds:   # class sizes preserved
        assert all(np.array_equal(np.bincount(d[0], minlength=nc), np.bincount(labels, minlength=nc))
                   for d in draws[kind][:5])
    null_prof = {k: np.zeros((N_NULL, n_layers)) for k in kinds}
    null_prof_decl = np.zeros((N_NULL, n_layers))
    rows = []
    for l in range(n_layers):
        t0 = time.time()
        X = H[l]
        K = gram(X, "linear")
        th_obs, sizes = ladder(K, mem0, canonical, counts)
        th_f = floor(K, n, sizes, l * 100 + 1)
        d_decl = th_obs - th_f
        per_order = np.array([ladder(K, mem0, o, counts)[0] for o in orders200]) - th_f
        d50 = order_avg(K, mem0, orders50, counts, th_f)
        d20 = order_avg(K, mem0, orders50[:20], counts, th_f)
        row = {"layer": l, "delta_declared": float(d_decl), "floor": float(th_f), "rung_sizes": sizes,
               "dbar50_run37": d50, "dbar20_run55": d20, "dbar200": float(per_order.mean()),
               "dbar200_sd_over_orders": float(per_order.std()),
               "declared_pct": float((per_order < d_decl).mean())}
        for kind in kinds:
            vals = []
            for i, (lab, ords) in enumerate(draws[kind]):
                mem = members_of(lab, nc)
                vals.append(order_avg(K, mem, ords, counts, th_f))
                if kind == "perm":
                    null_prof_decl[i, l] = ladder(K, mem, canonical, counts)[0] - th_f
            vals = np.array(vals)
            null_prof[kind][:, l] = vals
            nm, ns = float(vals.mean()), float(vals.std())
            row[kind] = {"null_mean": nm, "null_sd": ns, "null_max": float(vals.max()),
                         "p_one": float((1 + (vals >= d50).sum()) / (N_NULL + 1)),
                         "p_two": float((1 + (np.abs(vals - nm) >= abs(d50 - nm)).sum()) / (N_NULL + 1)),
                         "z": (d50 - nm) / ns if ns > 0 else 0.0}
        if battery in ("world_knowledge", "language_type"):
            row["spectrum"] = spectrum(X)
            for v in variants:
                Kv = gram(X, v)
                th_ov, sz = ladder(Kv, mem0, canonical, counts)
                th_fv = floor(Kv, n, sz, l * 100 + 1)
                row[v] = {"delta_declared": float(th_ov - th_fv),
                          "dbar50": order_avg(Kv, mem0, orders50, counts, th_fv),
                          "dbar20_run55": order_avg(Kv, mem0, orders50[:20], counts, th_fv)}
        rows.append(row)
        print(f"  L{l:02d} decl={d_decl:+.3f} dbar50={d50:+.3f} dbar200={row['dbar200']:+.3f} "
              f"pct={row['declared_pct']:.2f} " + " ".join(
                  f"{k}:p1={row[k]['p_one']:.3f}/p2={row[k]['p_two']:.3f}(null {row[k]['null_mean']:+.3f}"
                  f"±{row[k]['null_sd']:.3f})" for k in kinds)
              + (f" z={row['zscore']['dbar50']:+.3f} pc4={row['pc4']['dbar50']:+.3f} d_eff={row['spectrum']['d_eff']:.1f}"
                 if "zscore" in row else "") + f" ({time.time() - t0:.0f}s)", flush=True)

    def ie(p):
        p = np.asarray(p)
        return float((p[1:] - p[0]).mean())
    prof = [r["dbar50_run37"] for r in rows]
    prof_decl = [r["delta_declared"] for r in rows]
    profile = {}
    for tag, obs, nulls in (("orderavg", prof, null_prof["perm"]), ("declared", prof_decl, null_prof_decl)):
        nie = np.array([ie(nulls[i]) for i in range(N_NULL)])
        o = ie(obs)
        nm, ns = float(nie.mean()), float(nie.std())
        profile[tag] = {"IE": o, "null_mean": nm, "null_sd": ns, "z": (o - nm) / ns if ns > 0 else 0.0,
                        "p_two": float((1 + (np.abs(nie - nm) >= abs(o - nm)).sum()) / (N_NULL + 1))}
    return {"battery": battery, "n_prompts": int(n), "class_names": list(names), "rung_counts": counts,
            "null_kinds": kinds, "layers": rows, "profile": profile,
            "null_profiles_perm": null_prof["perm"].tolist()}


def analyse(mkey, batteries):
    out_path = OUTDIR / f"h2_llm_battery_{mkey}.json"
    out = json.load(open(out_path)) if out_path.exists() else {"model_key": mkey, "batteries": {},
                                                               "n_orders": N_ORDERS, "n_null": N_NULL,
                                                               "k_null_orders": K_NULL_ORDERS}
    for b in batteries:
        f = STATES / f"{mkey}__{b}.npz"
        z = np.load(f, allow_pickle=False)
        H = z["H"]
        print(f"[{mkey} / {b}] H {H.shape} {z['dtype']}", flush=True)
        prompts, labels, names, counts = battery_prompts(b)
        assert np.array_equal(labels, z["labels"]), "labels changed since encoding"
        res = analyse_battery(H, z["labels"], [str(x) for x in z["names"]], counts, b)
        res["dtype"] = str(z["dtype"])
        res["device"] = str(z["device"])
        out["batteries"][b] = res
        json.dump(out, open(out_path, "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "encode":
        encode(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None,
               batteries=tuple(sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] != "-" else
               ("world_knowledge", "language_type", "blimp_B", "baroni64"),
               key=sys.argv[6] if len(sys.argv) > 6 else None)
        print("DONE", flush=True)
    else:
        analyse(sys.argv[2], sys.argv[3:] or ["world_knowledge", "language_type", "blimp_B", "baroni64"])
