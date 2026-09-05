"""Run 55 — the physics of a blind probe: why OLMo-1B reads zero on every axis.

WHY: OLMo-1B (allenai/OLMo-1B-hf) reads nearly nothing on any declared axis (Table 5),
while OLMo-2-1B (run54), Mamba (run53) and every NeoX-lineage model carry the content
and construction profiles. The paper's rule says delta = 0 means the probe is blind.
A linear participation ratio PR = (sum lambda)^2 / sum lambda^2 is blind when a few
residual-stream dimensions carry most of the variance ("rogue dimensions", Timkey &
van Schijndel 2021; "massive activations", Sun et al. 2024): every subset then has
d_eff ~ 1-2, both ladders are flat, delta = 0 for any axis. That is measurable and
has three repairs that are also tests.

DESIGN: for each model, encode the world_knowledge and language_type prompts
(run17.build_axes(), byte-identical) at every layer; then per layer
  DIAGNOSTICS on the language_type matrix (128 x d, column-centered): d_eff of the full
  set; top-1 and top-3 eigenvalue variance fractions; fraction of total variance in the
  single largest dimension and in the top 1% of dimensions; kurtosis of the per-dimension
  variances.
  DELTA VARIANTS on both axes (ladder [1,2,3,4,6,8], twenty-draw floors, 100 label
  permutations two-sided, order-averaged over 20 random orders, paper seeds):
    linear   = the paper's estimator (centered Gram);
    zscore   = per-dimension standardization first (diagonal Mahalanobis);
    pc1, pc4 = top-1 / top-4 principal components of the layer removed first;
    kernel   = participation ratio of the double-centered RBF kernel at the median
               pairwise distance (nonlinear, local geometry).
MODELS: OLMo-1B-hf (the blind one), pythia-160m, pythia-2.8b-deduped, OLMo-2-0425-1B,
mamba-2.8b-hf (the readable comparisons).

REGISTERED EXPECTATIONS (written before the run, 2026-09-05 01:05 CDT):
B1: OLMo-1B's top-1 eigenvalue fraction and single-dimension variance fraction are far
    above the other models' at most layers, and its full-set d_eff is far below theirs.
B2: on OLMo-1B, zscore and/or pc-removal restore the profiles the other models show:
    content positive at the embedding and diluting; construction negative early, rising.
B3: the kernel PR reads OLMo-1B where the linear PR is blind (nonzero, sign-structured
    profiles with permutation p < 0.05 at more than chance layers).
B4: on the readable models the zscore and kernel variants preserve the linear sign
    structure (the paper's linear reading is not an artifact of the estimator).
If B1 fails (OLMo-1B's spectrum is unremarkable), the blindness is not rogue-dimension
physics and the outlier stays open; say so.
Out: ../data_canonical/run55_blind_probe_physics.json (log alongside).
"""
import importlib.util, json, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data_canonical" / "run55_blind_probe_physics.json"
spec = importlib.util.spec_from_file_location("r17", HERE / "run17_multiclass_battery.py")
r17 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r17)
spec = importlib.util.spec_from_file_location("r37", HERE / "run37_inferential_nulls.py")
r37 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r37)

MODELS = [("allenai/OLMo-1B-hf", "mps"), ("EleutherAI/pythia-160m", "cpu"),
          ("allenai/OLMo-2-0425-1B", "mps"), ("EleutherAI/pythia-2.8b-deduped", "mps"),
          ("state-spaces/mamba-2.8b-hf", "mps")]
AXES = ["world_knowledge", "language_type"]
BIN_COUNTS = [1, 2, 3, 4, 6, 8]
N_PERM, K_ORDERS, N_FLOOR = 100, 20, 20


def diagnostics(X):
    Xc = X - X.mean(axis=0, keepdims=True)
    v = Xc.var(axis=0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    lam = sv ** 2
    tot = lam.sum()
    k1 = max(1, int(round(0.01 * X.shape[1])))
    vs = np.sort(v)[::-1]
    m = v.mean(); sd = v.std() + 1e-12
    return {"d_eff_full": float(tot ** 2 / (lam ** 2).sum()),
            "top1_eig_frac": float(lam[0] / tot), "top3_eig_frac": float(lam[:3].sum() / tot),
            "top1_dim_var_frac": float(vs[0] / v.sum()), "top1pct_dims_var_frac": float(vs[:k1].sum() / v.sum()),
            "dim_var_kurtosis": float(((v - m) ** 4).mean() / sd ** 4)}


def gram_variants(X, rng):
    Xc = X - X.mean(axis=0, keepdims=True)
    out = {"linear": Xc @ Xc.T}
    Z = Xc / (Xc.std(axis=0, keepdims=True) + 1e-9)
    out["zscore"] = Z @ Z.T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    for k in (1, 4):
        Xr = Xc - (U[:, :k] * S[:k]) @ Vt[:k]
        out[f"pc{k}"] = Xr @ Xr.T
    sq = (Xc ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Xc @ Xc.T, 0)
    med = np.median(np.sqrt(D2[np.triu_indices_from(D2, 1)]))
    out["kernel"] = np.exp(-D2 / (2 * med ** 2 + 1e-12))
    return out


def delta_variant(K, labels, n_classes, seed):
    members = [np.where(labels == c)[0] for c in range(n_classes)]
    order = list(range(n_classes))
    th_obs, sizes = r37.ladder_slope(K, members, order)
    rng = np.random.default_rng(100 * seed + 1)
    th_f = r37.floor_slope(K, len(labels), sizes, rng)
    delta = th_obs - th_f
    prng = np.random.default_rng(3700)
    nulls = []
    for _ in range(N_PERM):
        pl = prng.permutation(labels)
        pm = [np.where(pl == c)[0] for c in range(n_classes)]
        nulls.append(r37.ladder_slope(K, pm, order)[0] - th_f)
    nulls = np.array(nulls)
    p_two = float((np.sum(np.abs(nulls - nulls.mean()) >= abs(delta - nulls.mean())) + 1) / (N_PERM + 1))
    orng = np.random.default_rng(3701)
    davg = float(np.mean([r37.ladder_slope(K, members, list(orng.permutation(n_classes)))[0] - th_f for _ in range(K_ORDERS)]))
    return {"delta": float(delta), "theta_floor": float(th_f), "p_two": p_two, "null_sd": float(nulls.std()), "delta_orderavg": davg}


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    axes_all = r17.build_axes()
    out = json.load(open(OUT)) if OUT.exists() else {"n_perm": N_PERM, "k_orders": K_ORDERS, "n_floor": N_FLOOR, "models": {}}
    for model_name, device in MODELS:
        mkey = model_name.split("/")[-1]
        if mkey in out["models"] and all(a in out["models"][mkey]["axes"] for a in AXES):
            print(f"skip {mkey} (done)", flush=True); continue
        print(f"\nloading {model_name} ({device})...", flush=True)
        dtype = torch.float16 if device == "mps" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        out["models"].setdefault(mkey, {"axes": {}, "diagnostics": []})
        for axis_name in AXES:
            classes = axes_all[axis_name]; names = list(classes.keys())
            prompts, labels = [], []
            for ci, cn in enumerate(names):
                for p in classes[cn]:
                    prompts.append(p); labels.append(ci)
            labels = np.array(labels)
            print(f"[{mkey} / {axis_name}] encoding {len(prompts)} prompts...", flush=True)
            per_layer = r17.get_hidden_states(model, tok, prompts, device=device)
            layers = []
            t0 = time.time()
            for l, X in enumerate(per_layer):
                X = X.astype(np.float64)
                row = {"layer": l}
                if axis_name == "language_type":
                    row["diag"] = diagnostics(X)
                for name, K in gram_variants(X, None).items():
                    row[name] = delta_variant(K, labels, len(names), seed=l)
                layers.append(row)
                d = row.get("diag", {})
                print(f"  L{l:02d} " + " ".join(f"{n}={row[n]['delta']:+.3f}(p{row[n]['p_two']:.2f})" for n in ("linear", "zscore", "pc1", "pc4", "kernel"))
                      + (f" | d_eff={d['d_eff_full']:.1f} top1eig={d['top1_eig_frac']:.2f} top1dim={d['top1_dim_var_frac']:.2f}" if d else "")
                      + f" ({time.time() - t0:.0f}s)", flush=True)
            out["models"][mkey]["axes"][axis_name] = {"class_names": names, "layers": layers}
            json.dump(out, open(OUT, "w"), indent=1)
        del model
        if device == "mps":
            torch.mps.empty_cache()
    print("\nDONE run55", flush=True)


if __name__ == "__main__":
    main()
