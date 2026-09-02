"""theta_zoom — the axis-resolved dimensionality-scaling instrument, reusable.

Single-file, numpy-only implementation of the paper's estimator:

    theta_obs = theta_floor + delta

with the label-blind floor, the 500-permutation per-layer null, the
order-averaged statistic for unordered class axes, and the
nuisance-preserving stratified null. Conventions are byte-compatible with
the paper's runs (see scripts/run37_inferential_nulls.py).

Quick start on your own data (any modality; rows = samples, cols = features):

    import numpy as np
    from theta_zoom import zoom

    X = ...        # (n_samples, n_features), e.g. trials x neurons or
                   # prompts x hidden units at one layer
    labels = ...   # (n_samples,) integer class labels for the declared axis

    res = zoom(X, labels, n_perm=500, k_orders=50)
    print(res["delta"], res["p_two"])          # declared-order shift + null
    print(res["delta_orderavg"], res["p_two_orderavg"])  # order-free shift

    # nuisance-preserving second null (e.g. carriers, topics, sessions):
    res = zoom(X, labels, strata=my_nuisance_ids, n_perm=500)
    print(res["delta"], res["strat_p_two"])    # label reading beyond nuisance

For a language model, collect last-token hidden states at a layer for a
prompt battery (one forward pass per prompt) and call zoom() per layer, or
use the CLI with the paper's battery shipped in axes/ (7 axes, 8 classes x 16
prompts): `theta-zoom llm --model M --axis axes/language_type.json`, then
`theta-zoom plot battery.json --out profile.png`.
The paper's full batteries, artifacts, and registered expectations live in
scripts/ and data/ of this repository.
"""
import numpy as np

BIN_COUNTS_DEFAULT = (1, 2, 3, 4, 6, 8)


def _subset_pr(K, idx):
    """PR of the row-centered subset via the precomputed full Gram matrix."""
    Ks = K[np.ix_(idx, idx)]
    rm = Ks.mean(axis=1)
    tm = rm.mean()
    Kc = Ks - rm[:, None] - rm[None, :] + tm
    tr = float(np.trace(Kc))
    tr2 = float((Kc * Kc).sum())
    return tr * tr / tr2 if np.isfinite(tr2) and tr2 > 0 else 1.0


def _slope(sizes, prs):
    x = np.log(np.asarray(sizes, float))
    y = np.log(np.maximum(np.asarray(prs, float), 1e-9))
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, y, rcond=None)[0][1])


def _ladder(K, members, order, bin_counts):
    sizes, prs = [], []
    for c in bin_counts:
        sel = np.concatenate([members[o] for o in order[:c]])
        sizes.append(len(sel))
        prs.append(_subset_pr(K, sel))
    return _slope(sizes, prs), sizes


def zoom(X, labels, n_perm=500, k_orders=50, k_null_orders=20,
         n_floor_draws=20, bin_counts=None, strata=None, seed=0,
         floor_seed=None, perm_seed=None, order_seed=None):
    """Axis-resolved decomposition of the dimensionality-scaling exponent.

    Parameters
    ----------
    X : (n, d) array — samples by features.
    labels : (n,) int array — class labels of the declared axis. Classes are
        accumulated in label order (0, 1, 2, ...); for unordered axes the
        order-averaged outputs are the partition-level statistics.
    n_perm : label permutations for the per-layer null (0 disables).
    k_orders : random accumulation orders for the order-averaged shift.
    k_null_orders : orders averaged per permuted labeling (conservative if
        smaller than k_orders; see the paper's Appendix G).
    n_floor_draws : random same-size subset draws defining the floor.
    bin_counts : ladder rung class counts; defaults to (1,2,3,4,6,8)
        truncated to the number of classes.
    strata : optional (n,) array of nuisance-stratum ids. When given, a
        second null permutes labels only within strata (nuisance-preserving);
        significance against it supports a label reading beyond composition.
    seed : rng seed (single stream) unless the three explicit seeds are
        given. floor_seed / perm_seed / order_seed reproduce the paper's
        convention (scripts/run37): floor rng = 100*layer+1, permutations
        rng 3700, orders rng 3701, each an independent stream.

    Returns dict with delta, theta_obs, theta_floor, p_two, z,
    delta_orderavg (+sd, p_two_orderavg, z_orderavg), and, when strata is
    given, strat_p_two / strat_z.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X / (X.std() + 1e-9)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    n_classes = len(classes)
    lab = np.searchsorted(classes, labels)
    n = len(lab)
    if bin_counts is None:
        bc = tuple(c for c in BIN_COUNTS_DEFAULT if c <= n_classes)
        if bc[-1] != n_classes:
            bc = bc + (n_classes,)
    else:
        bc = tuple(bin_counts)
    if len(bc) < 3:
        raise ValueError("need >=3 ladder rungs; use >=3 classes")

    if floor_seed is None and perm_seed is None and order_seed is None:
        rng = np.random.default_rng(seed)
        frng = prng = orng = rng
    else:
        frng = np.random.default_rng(floor_seed if floor_seed is not None else seed)
        prng = np.random.default_rng(perm_seed if perm_seed is not None else seed)
        orng = np.random.default_rng(order_seed if order_seed is not None else seed)
    K = X @ X.T
    members = [np.where(lab == c)[0] for c in range(n_classes)]
    canonical = list(range(n_classes))

    th_obs, sizes = _ladder(K, members, canonical, bc)
    logs = np.zeros((n_floor_draws, len(sizes)))
    for i in range(n_floor_draws):
        for k, s in enumerate(sizes):
            logs[i, k] = np.log(max(
                _subset_pr(K, frng.choice(n, s, replace=False)), 1e-9))
    th_floor = _slope(sizes, np.exp(logs.mean(axis=0)))
    d_obs = th_obs - th_floor

    out = {"delta": d_obs, "theta_obs": th_obs, "theta_floor": th_floor,
           "rung_sizes": sizes, "n_classes": n_classes}

    orders = [orng.permutation(n_classes).tolist() for _ in range(k_orders)]
    d_orders = [_ladder(K, members, o, bc)[0] - th_floor for o in orders]
    out["delta_orderavg"] = float(np.mean(d_orders))
    out["delta_orderavg_sd"] = float(np.std(d_orders))

    if n_perm:
        null_c, null_a = [], []
        for _ in range(n_perm):
            pl = lab[prng.permutation(n)]
            mem = [np.where(pl == c)[0] for c in range(n_classes)]
            null_c.append(_ladder(K, mem, canonical, bc)[0] - th_floor)
            null_a.append(np.mean(
                [_ladder(K, mem, o, bc)[0]
                 for o in orders[:k_null_orders]]) - th_floor)
        for tag, obs, null in (("", d_obs, null_c),
                               ("_orderavg", out["delta_orderavg"], null_a)):
            null = np.asarray(null)
            nm, ns = float(null.mean()), float(null.std())
            out[f"z{tag}"] = (obs - nm) / ns if ns > 0 else 0.0
            out[f"p_two{tag}"] = float(
                (1 + (np.abs(null - nm) >= abs(obs - nm)).sum())
                / (len(null) + 1))
            out[f"null_mean{tag}"], out[f"null_sd{tag}"] = nm, ns

    if strata is not None and n_perm:
        strata = np.asarray(strata)
        smem = {s: np.where(strata == s)[0] for s in np.unique(strata)}
        null_s = []
        for _ in range(n_perm):
            sl = lab.copy()
            for idx in smem.values():
                sl[idx] = sl[idx[prng.permutation(len(idx))]]
            mem = [np.where(sl == c)[0] for c in range(n_classes)]
            null_s.append(_ladder(K, mem, canonical, bc)[0] - th_floor)
        null_s = np.asarray(null_s)
        nm, ns = float(null_s.mean()), float(null_s.std())
        out["strat_z"] = (d_obs - nm) / ns if ns > 0 else 0.0
        out["strat_p_two"] = float(
            (1 + (np.abs(null_s - nm) >= abs(d_obs - nm)).sum())
            / (len(null_s) + 1))
        out["strat_null_mean"], out["strat_null_sd"] = nm, ns

    return out


# ----------------------------- command line -----------------------------

def _load_array(path):
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    return np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)


def _llm_battery(model_name, axis_file, device, n_perm, k_orders, max_len,
                 out_path, paper_seeds=False):
    """Per-layer zoom() over last-token hidden states for a prompt battery.

    axis_file: JSON mapping class name -> list of prompts, e.g.
      {"question": ["What causes...", ...], "definition": ["A molecule is...", ...]}
    """
    import json as _json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    axes = _json.load(open(axis_file))
    prompts, labels = [], []
    for ci, (cname, plist) in enumerate(axes.items()):
        for p in plist:
            prompts.append(p)
            labels.append(ci)
    labels = np.array(labels)
    print(f"{len(axes)} classes, {len(prompts)} prompts; loading "
          f"{model_name} on {device}...", flush=True)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    states = []
    with torch.no_grad():
        for i, p in enumerate(prompts):
            ids = tok(p, return_tensors="pt", truncation=True,
                      max_length=max_len).input_ids.to(device)
            hs = model(ids, output_hidden_states=True).hidden_states
            states.append([h[0, -1, :].float().cpu().numpy() for h in hs])
            if i and i % 50 == 0:
                print(f"  encoded {i}/{len(prompts)}", flush=True)

    results = {"model": model_name, "classes": list(axes.keys()),
               "n_perm": n_perm, "k_orders": k_orders, "layers": []}
    for l in range(len(states[0])):
        X = np.stack([s[l] for s in states])
        if paper_seeds:
            r = zoom(X, labels, n_perm=n_perm, k_orders=k_orders,
                     floor_seed=100 * l + 1, perm_seed=3700, order_seed=3701)
        else:
            r = zoom(X, labels, n_perm=n_perm, k_orders=k_orders, seed=l)
        row = {"layer": l,
               **{k: r[k] for k in ("delta", "p_two", "z", "null_mean", "null_sd",
                                    "delta_orderavg", "delta_orderavg_sd",
                                    "p_two_orderavg", "z_orderavg",
                                    "null_mean_orderavg", "null_sd_orderavg")
                  if k in r}}
        results["layers"].append(row)
        print(f"  L{l:02d} delta={r['delta']:+.3f} p={r.get('p_two', 1):.4f}"
              f" | avg={r['delta_orderavg']:+.3f}"
              f" p={r.get('p_two_orderavg', 1):.4f}", flush=True)
    if out_path:
        _json.dump(results, open(out_path, "w"), indent=1)
        print(f"wrote {out_path}", flush=True)
    return results



def _plot_battery(path, out_png):
    """Depth profile from a `theta-zoom llm` JSON: declared-path delta and
    order-averaged deltabar with their permutation-null bands."""
    import json as _json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _json.load(open(path))
    L = d["layers"]
    x = np.arange(len(L)) / max(len(L) - 1, 1)
    dl = np.array([l["delta"] for l in L])
    da = np.array([l["delta_orderavg"] for l in L])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if "null_sd" in L[0]:
        nm = np.array([l["null_mean"] for l in L]); ns = np.array([l["null_sd"] for l in L])
        ax.fill_between(x, nm - 2 * ns, nm + 2 * ns, color="#c0392b", alpha=0.12, lw=0,
                        label="declared-path null ($\\pm 2$ SD)")
    if "null_sd_orderavg" in L[0]:
        nma = np.array([l["null_mean_orderavg"] for l in L]); nsa = np.array([l["null_sd_orderavg"] for l in L])
        ax.fill_between(x, nma - 2 * nsa, nma + 2 * nsa, color="#2c3e50", alpha=0.18, lw=0,
                        label="order-averaged null ($\\pm 2$ SD)")
    ax.plot(x, dl, "-o", ms=3, color="#c0392b", label="declared path $\\delta$")
    ax.plot(x, da, "--s", ms=3, color="#2c3e50", label="order-averaged $\\bar\\delta$")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("normalized depth"); ax.set_ylabel("$\\delta$")
    ax.set_title(f"{d.get('model', '')}: {', '.join(d.get('classes', [])[:3])}...", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print(f"wrote {out_png}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(
        prog="theta-zoom",
        description="Axis-resolved dimensionality scaling: "
                    "theta_obs = theta_floor + delta, with exact "
                    "permutation nulls (see the paper for conventions).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="run on arrays (npy/csv/txt)")
    d.add_argument("X", help="samples-by-features array")
    d.add_argument("labels", help="integer class labels, one per sample")
    d.add_argument("--strata", default=None,
                   help="nuisance stratum ids for the second null")
    d.add_argument("--n-perm", type=int, default=500)
    d.add_argument("--k-orders", type=int, default=50)
    d.add_argument("--seed", type=int, default=0)

    m = sub.add_parser("llm", help="per-layer battery on a Hugging Face model")
    m.add_argument("--model", required=True)
    m.add_argument("--axis", required=True,
                   help="JSON: {class_name: [prompt, ...], ...}")
    m.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    m.add_argument("--n-perm", type=int, default=500)
    m.add_argument("--k-orders", type=int, default=50)
    m.add_argument("--max-len", type=int, default=128)
    m.add_argument("--out", default=None, help="write per-layer JSON here")
    m.add_argument("--paper-seeds", action="store_true",
                   help="use the paper's seeding convention (reproduces run37 cells)")

    p = sub.add_parser("plot", help="depth-profile figure from a `theta-zoom llm` JSON")
    p.add_argument("battery_json")
    p.add_argument("--out", default="battery.png")

    a = ap.parse_args()
    if a.cmd == "plot":
        _plot_battery(a.battery_json, a.out)
    elif a.cmd == "data":
        X = _load_array(a.X)
        labels = _load_array(a.labels).astype(int)
        strata = _load_array(a.strata).astype(int) if a.strata else None
        r = zoom(X, labels, n_perm=a.n_perm, k_orders=a.k_orders,
                 strata=strata, seed=a.seed)
        for k, v in r.items():
            print(f"{k}: {v}")
    else:
        _llm_battery(a.model, a.axis, a.device, a.n_perm, a.k_orders,
                     a.max_len, a.out, paper_seeds=a.paper_seeds)


if __name__ == "__main__":
    main()
