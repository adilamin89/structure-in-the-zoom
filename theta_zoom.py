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
    """Accumulate classes in `order`; return (slope, rung sizes, rung PRs)."""
    sizes, prs = [], []
    for c in bin_counts:
        sel = np.concatenate([members[o] for o in order[:c]])
        sizes.append(len(sel))
        prs.append(_subset_pr(K, sel))
    return _slope(sizes, prs), sizes, prs


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
    n_samples = X.shape[0]
    if n_samples > 5000:
        import warnings
        warnings.warn(f"zoom: {n_samples} samples -> full Gram of {8*n_samples**2/1e9:.1f} GB in float64 and "
                      f"time growing as n^2 (about {(n_samples/6400)**2*3.4:.0f} min per 100 permutations on one core); "
                      "subsample stimuli per class (rung sizes stay matched) or run per-layer jobs in parallel.", stacklevel=2)
    K = X @ X.T
    members = [np.where(lab == c)[0] for c in range(n_classes)]
    canonical = list(range(n_classes))

    th_obs, sizes, pr_obs = _ladder(K, members, canonical, bc)
    logs = np.zeros((n_floor_draws, len(sizes)))
    for i in range(n_floor_draws):
        for k, s in enumerate(sizes):
            logs[i, k] = np.log(max(
                _subset_pr(K, frng.choice(n, s, replace=False)), 1e-9))
    th_floor = _slope(sizes, np.exp(logs.mean(axis=0)))
    d_obs = th_obs - th_floor

    out = {"delta": d_obs, "theta_obs": th_obs, "theta_floor": th_floor,
           "rung_sizes": sizes, "n_classes": n_classes,
           "pr_obs": [float(v) for v in pr_obs],
           "pr_floor": [float(v) for v in np.exp(logs.mean(axis=0))],
           "n_perm": int(n_perm), "k_orders": int(k_orders),
           "n_floor_draws": int(n_floor_draws), "n_samples": int(n_samples),
           "n_features": int(X.shape[1])}

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


def _plot_data(result, out_png, title=None):
    """Ladder figure for one `zoom` result: observed rungs against the matched
    floor on log-log axes, with the shift and its permutation p annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = result["rung_sizes"]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(sizes, result["pr_floor"], "o-", color="0.55", label="matched floor (random subsets)")
    ax.plot(sizes, result["pr_obs"], "o-", color="#c23b3b", label="declared axis (classes accumulated)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("samples in rung"); ax.set_ylabel("participation ratio")
    p = result.get("p_two"); ptxt = f"; permutation p = {p:.3f}" if p is not None else ""
    ax.set_title(title or f"shift delta = {result['delta']:+.3f}{ptxt}\n"
                 f"theta_obs {result['theta_obs']:+.3f} = floor {result['theta_floor']:+.3f} + delta", fontsize=8)
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    for axis_ in (ax.xaxis, ax.yaxis):
        axis_.set_major_formatter(ScalarFormatter()); axis_.set_minor_formatter(NullFormatter())
    ax.set_xticks(sizes); ax.set_xticklabels([str(v) for v in sizes], fontsize=7)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return out_png


def summarize_data(result, verbose=True):
    """Plain-language reading of one `zoom` result (the paper's rules for a
    single array): sign, certification at the permutation resolution, the
    order-averaged statistic, and the stratified verdict when present."""
    n_perm = int(result.get("n_perm", 0)); floor_p = 1.0 / (n_perm + 1) if n_perm else None
    lines = [f"theta_obs = {result['theta_obs']:+.3f} = floor {result['theta_floor']:+.3f} + shift {result['delta']:+.3f} "
             f"({result['n_classes']} classes, rungs {result['rung_sizes']})"]
    verdict = {"delta": result["delta"]}
    if "p_two" in result:
        cert = result["p_two"] < 0.05
        lines.append(f"declared order: z = {result['z']:+.1f}, two-sided Monte Carlo p = {result['p_two']:.3f} "
                     f"(resolution {floor_p:.3f}); {'label-linked at p < 0.05' if cert else 'not separated from the label-permutation null'}")
        verdict["certified"] = bool(cert)
    if "delta_orderavg" in result:
        oa = result["delta_orderavg"]; same = (oa > 0) == (result["delta"] > 0)
        txt = f"order-averaged shift {oa:+.3f} (SD over orders {result['delta_orderavg_sd']:.3f})"
        if "p_two_orderavg" in result:
            txt += f", p = {result['p_two_orderavg']:.3f}"
        txt += "; same sign as the declared order" if same else "; OPPOSITE sign to the declared order: the declared-order value is a path property, report the order-averaged one for the partition"
        lines.append(txt); verdict["orderavg_same_sign"] = bool(same)
    if "strat_p_two" in result:
        absorbed = result["strat_p_two"] >= 0.05
        lines.append(f"nuisance-preserving null: p = {result['strat_p_two']:.3f}; "
                     + ("the shift is ABSORBED by within-stratum relabeling: it reads nuisance composition, not the labels"
                        if absorbed else "the shift survives within-stratum relabeling: label-linked beyond the nuisance"))
        verdict["label_linked_beyond_strata"] = bool(not absorbed)
    if result["delta"] < 0:
        lines.append("negative shift: accumulating the declared classes adds fewer dimensions than random draws (conditioning branch; low-rank between-class structure with isotropic within-class variability gives this sign)")
    if verbose:
        for l in lines:
            print(l)
    return {"lines": lines, "verdict": verdict}


def build_axis(rows, text_field, label_field, n_classes=8, n_per_class=16,
               strata_field=None, classes=None, min_chars=20, max_chars=400, seed=0):
    """Turn labelled records into an axis JSON ({class: [prompts]}) and an
    optional strata sidecar. `rows` is any iterable of dicts (a Hugging Face
    dataset split works). Classes are the `n_classes` most frequent labels
    unless `classes` is given; `n_per_class` texts are drawn per class with a
    fixed seed; texts shorter than `min_chars` are skipped and longer than
    `max_chars` are cut at the last space before the limit."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    pool = defaultdict(list)
    for r in rows:
        t = r.get(text_field); lab = r.get(label_field)
        if t is None or lab is None:
            continue
        t = str(t).strip()
        if len(t) < min_chars:
            continue
        if len(t) > max_chars:
            cut = t[:max_chars]
            t = cut[:cut.rfind(" ")] if " " in cut else cut
        pool[str(lab)].append((t, None if strata_field is None else str(r.get(strata_field))))
    if classes is None:
        classes = [c for c, _ in sorted(pool.items(), key=lambda kv: -len(kv[1]))[:n_classes]]
    axis, strata = {}, {}
    for c in classes:
        items = pool.get(str(c), [])
        if len(items) < n_per_class:
            raise ValueError(f"class {c!r} has {len(items)} usable texts, fewer than n_per_class={n_per_class}")
        pick = rng.choice(len(items), n_per_class, replace=False)
        axis[str(c)] = [items[i][0] for i in pick]
        if strata_field is not None:
            strata[str(c)] = [items[i][1] for i in pick]
    return axis, (strata if strata_field is not None else None)


def axis_from_dataset(name, text_field, label_field, out_json, config=None, split="train",
                      n_classes=8, n_per_class=16, strata_field=None, classes=None,
                      min_chars=20, max_chars=400, seed=0):
    """`theta-zoom axis`: build an axis JSON (+ strata sidecar) from a Hugging
    Face dataset. Requires the `datasets` package (`pip install -e ".[models]"`)."""
    import json as _json
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split) if config else load_dataset(name, split=split)
    axis, strata = build_axis(ds, text_field, label_field, n_classes, n_per_class, strata_field, classes, min_chars, max_chars, seed)
    _json.dump(axis, open(out_json, "w"), indent=1)
    if strata is not None:
        side = out_json[:-5] + ".strata.json" if out_json.endswith(".json") else out_json + ".strata.json"
        _json.dump(strata, open(side, "w"), indent=1)
    print(f"wrote {out_json}: {len(axis)} classes x {n_per_class} prompts" + (" + strata sidecar" if strata is not None else ""))
    return axis, strata


# ----------------------------- command line -----------------------------

def _load_array(path):
    if path.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    return np.loadtxt(path, delimiter="," if path.endswith(".csv") else None)


def _expand_axis_paths(paths):
    """Files, or directories (all *.json except INDEX.json and *.strata.json)."""
    import glob, os
    out = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(glob.glob(os.path.join(p, "*.json"))):
                b = os.path.basename(f)
                if b == "INDEX.json" or b.endswith(".strata.json"):
                    continue
                out.append(f)
        else:
            out.append(p)
    return out


def _load_axis(axis_file, use_strata=True):
    """axis_file: JSON {class_name: [prompt, ...]}. Optional sidecar
    <name>.strata.json with the same shape holding a nuisance stratum per
    prompt (topic, template, carrier id) for the second null."""
    import json as _json, os
    axes = _json.load(open(axis_file))
    prompts, labels, strata = [], [], []
    side = axis_file[:-5] + ".strata.json" if axis_file.endswith(".json") else None
    smap = _json.load(open(side)) if (use_strata and side and os.path.exists(side)) else None
    for ci, (cname, plist) in enumerate(axes.items()):
        for k, p in enumerate(plist):
            prompts.append(p)
            labels.append(ci)
            if smap is not None:
                strata.append(smap[cname][k])
    name = os.path.splitext(os.path.basename(axis_file))[0]
    return name, list(axes.keys()), prompts, np.array(labels), (np.array(strata) if smap is not None else None)


def llm_battery(model_name, axis_files, device="cpu", n_perm=500, k_orders=50,
                max_len=128, out_path=None, paper_seeds=False, revision=None,
                use_strata=True):
    """Per-layer zoom() over last-token hidden states for one or more axes.

    axis_files: list of axis JSON paths and/or directories. The model is loaded
    once (optionally at a Hugging Face `revision`, e.g. a Pythia checkpoint
    "step10000") and every axis is encoded and analyzed. If <axis>.strata.json
    exists next to an axis file, the stratified (nuisance-preserving) null is
    computed as well. Returns {"model", "revision", "axes": {name: {...}}}.
    """
    import json as _json
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    files = _expand_axis_paths(list(axis_files))
    print(f"{len(files)} axis file(s); loading {model_name}"
          f"{' @ ' + revision if revision else ''} on {device}...", flush=True)
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    kw = {"torch_dtype": dtype}
    if revision:
        kw["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw).to(device)
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision) if revision \
        else AutoTokenizer.from_pretrained(model_name)
    model.eval()

    results = {"model": model_name, "revision": revision, "n_perm": n_perm,
               "k_orders": k_orders, "axes": {}}
    for af in files:
        name, classes, prompts, labels, strata = _load_axis(af, use_strata)
        print(f"[{name}] {len(classes)} classes, {len(prompts)} prompts"
              f"{' + strata' if strata is not None else ''}", flush=True)
        states = []
        with torch.no_grad():
            for i, p in enumerate(prompts):
                ids = tok(p, return_tensors="pt", truncation=True,
                          max_length=max_len).input_ids.to(device)
                hs = model(ids, output_hidden_states=True).hidden_states
                states.append([h[0, -1, :].float().cpu().numpy() for h in hs])
                if i and i % 50 == 0:
                    print(f"    encoded {i}/{len(prompts)}", flush=True)
        layers = []
        for l in range(len(states[0])):
            X = np.stack([s[l] for s in states])
            if paper_seeds:
                r = zoom(X, labels, n_perm=n_perm, k_orders=k_orders, strata=strata,
                         floor_seed=100 * l + 1, perm_seed=3700, order_seed=3701)
            else:
                r = zoom(X, labels, n_perm=n_perm, k_orders=k_orders, strata=strata, seed=l)
            keys = ("delta", "p_two", "z", "null_mean", "null_sd", "delta_orderavg",
                    "delta_orderavg_sd", "p_two_orderavg", "z_orderavg",
                    "null_mean_orderavg", "null_sd_orderavg",
                    "strat_p_two", "strat_z", "strat_null_mean", "strat_null_sd")
            row = {"layer": l, **{k: r[k] for k in keys if k in r}}
            layers.append(row)
            msg = (f"    L{l:02d} delta={r['delta']:+.3f} p={r.get('p_two', 1):.4f}"
                   f" | avg={r['delta_orderavg']:+.3f} p={r.get('p_two_orderavg', 1):.4f}")
            if "strat_p_two" in r:
                msg += f" | strat p={r['strat_p_two']:.4f}"
            print(msg, flush=True)
        results["axes"][name] = {"classes": classes, "n_prompts": len(prompts),
                                 "stratified": strata is not None, "layers": layers}
    if out_path:
        _json.dump(results, open(out_path, "w"), indent=1)
        print(f"wrote {out_path}", flush=True)
    return results


def _plot_battery(path, out_png):
    """One panel per axis from a `theta-zoom llm` JSON: declared-path delta,
    order-averaged deltabar, and their permutation-null bands (plus the
    stratified-null band when present)."""
    import json as _json
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = _json.load(open(path))
    axes_d = d["axes"] if "axes" in d else {"battery": {"layers": d["layers"], "classes": d.get("classes", [])}}
    n = len(axes_d)
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 3.6), squeeze=False)
    for ax, (name, a) in zip(axs[0], axes_d.items()):
        L = a["layers"]
        x = np.arange(len(L)) / max(len(L) - 1, 1)
        def arr(k):
            return np.array([l[k] for l in L]) if k in L[0] else None
        for mk, sk, col, al, lab in (("null_mean", "null_sd", "#c0392b", 0.12, "declared-path null"),
                                     ("null_mean_orderavg", "null_sd_orderavg", "#2c3e50", 0.18, "order-averaged null"),
                                     ("strat_null_mean", "strat_null_sd", "#8e44ad", 0.18, "stratified null")):
            m, s = arr(mk), arr(sk)
            if m is not None and s is not None:
                ax.fill_between(x, m - 2 * s, m + 2 * s, color=col, alpha=al, lw=0,
                                label=f"{lab} ($\\pm 2$ SD)")
        ax.plot(x, arr("delta"), "-o", ms=3, color="#c0392b", label="declared path $\\delta$")
        ax.plot(x, arr("delta_orderavg"), "--s", ms=3, color="#2c3e50", label="order-averaged $\\bar\\delta$")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("normalized depth"); ax.set_ylabel("$\\delta$")
        ax.set_title(f"{d.get('model', '')}{' @ ' + d['revision'] if d.get('revision') else ''}\n{name}",
                     fontsize=9)
        ax.legend(fontsize=7, frameon=False)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print(f"wrote {out_png}", flush=True)



def _bh_count(ps, q=0.05):
    ps = np.asarray(ps, float); n = len(ps)
    if n == 0:
        return 0
    order = np.argsort(ps); thresh = q * np.arange(1, n + 1) / n
    passed = ps[order] <= thresh
    if not passed.any():
        return 0
    k = int(np.max(np.where(passed)[0])) + 1
    return int((ps <= thresh[k - 1]).sum())


def summarize(path, out_json=None, verbose=True):
    """Apply the paper's reading rules to a `theta-zoom llm` JSON and print a
    plain-language verdict per axis. Rules: certification counts at two-sided
    p<0.05 and after Benjamini-Hochberg (q<0.05) for the declared path and the
    order-averaged statistic; declared-path profile shape (embedding sign, peak,
    zero crossing); when a stratified null exists, whether the declared-path
    signal survives it (label-linked) or is absorbed (nuisance composition)."""
    import json as _json
    d = _json.load(open(path))
    axes_d = d["axes"] if "axes" in d else {"battery": {"layers": d["layers"]}}
    report = {"model": d.get("model"), "revision": d.get("revision"), "axes": {}}
    n_perm = d.get("n_perm", 500)
    floor_p = 1.0 / (n_perm + 1)
    for name, a in axes_d.items():
        L = a["layers"]; n = len(L)
        bh_possible = floor_p <= 0.05 / n
        pd_ = [l.get("p_two", 1.0) for l in L]; pa = [l.get("p_two_orderavg", 1.0) for l in L]
        kd, kd_bh = sum(p < 0.05 for p in pd_), _bh_count(pd_)
        ka, ka_bh = sum(p < 0.05 for p in pa), _bh_count(pa)
        if not bh_possible:
            # BH cannot certify any layer at this permutation count; fall back to raw counts
            kd_bh, ka_bh = kd, ka
        dl = np.array([l["delta"] for l in L]); da = np.array([l["delta_orderavg"] for l in L])
        emb, fin = dl[0], dl[-1]
        exc = dl - emb; peak_l = int(np.argmax(exc)); peak = float(exc[peak_l])
        cross = next((i for i in range(1, n) if np.sign(dl[i]) != np.sign(emb) and dl[i] != 0), None)
        lines = []
        # partition-level reading
        frac = ka_bh / n
        if frac >= 0.5:
            lines.append(f"label-linked at most depths: order-averaged statistic certified at {ka}/{n} layers "
                         f"({ka_bh}/{n} {'after BH' if bh_possible else 'raw'}), mean deltabar {da.mean():+.3f}.")
        elif ka_bh > 0:
            lines.append(f"weakly label-linked: order-averaged statistic certified at {ka}/{n} layers "
                         f"({ka_bh}/{n} {'after BH' if bh_possible else 'raw'}), mean deltabar {da.mean():+.3f}.")
        else:
            lines.append(f"no certified label linkage under the order-free statistic ({ka}/{n} raw, 0 after BH); "
                         f"treat this partition as unorganized at this design.")
        # declared-path shape
        shape = (f"declared path: embedding {emb:+.3f} -> final {fin:+.3f}, peak excess {peak:+.3f} at layer {peak_l}"
                 f"/{n-1}, certified {kd}/{n} ({kd_bh}/{n} {'after BH' if bh_possible else 'raw'})")
        if emb < 0 and fin > 0:
            shape += (f"; negative-to-positive crossover at layer {cross} — a PATH property (the order-averaged "
                      f"line is the partition-level claim; the crossover shape depends on the class order).")
        elif emb > 0 and fin < emb - 0.02:
            shape += "; positive at the embedding and declining — content-like: organization inherited from tokens, diluted with depth."
        elif abs(emb) < 0.02 and abs(fin) < 0.02 and kd_bh == 0:
            shape += "; flat near zero along the declared path."
        else:
            shape += "."
        lines.append(shape)
        # stratified null
        if "strat_p_two" in L[0]:
            ps_ = [l["strat_p_two"] for l in L]; ks, ks_bh = sum(p < 0.05 for p in ps_), _bh_count(ps_)
            if not bh_possible:
                ks_bh = ks
            tag = "after BH" if bh_possible else "raw, BH not resolvable"
            if kd > 0 and ks_bh == 0:
                lines.append(f"stratified null absorbs the declared-path signal ({ks}/{n} layers survive; {tag}): "
                             f"consistent with the ordinary floor reading nuisance composition rather than the label.")
            elif ks_bh > 0:
                lines.append(f"survives the stratified null at {ks}/{n} layers ({ks_bh}/{n} {tag}): "
                             f"label-linked beyond the declared nuisance structure.")
            else:
                lines.append(f"stratified null: nothing to absorb ({ks}/{n} layers).")
        else:
            lines.append("no strata given: the ordinary floor cannot separate label from nuisance composition; "
                         "add <axis>.strata.json (topics/templates/carriers) before reading this as representation.")
        if bh_possible:
            lines.append(f"caveat: per-layer counts are descriptive (~{0.05*n:.1f} false positives expected at p<0.05); "
                         f"the BH counts and the order-averaged statistic carry the inference.")
        else:
            lines.append(f"RESOLUTION WARNING: n_perm={n_perm} gives a p floor of {floor_p:.4f}, above the BH threshold "
                         f"{0.05/n:.4f} for {n} layers, so no layer can be BH-certified at this run; counts above are raw. "
                         f"Rerun with --n-perm 500 (the paper's setting) or more before drawing conclusions.")
        report["axes"][name] = {"n_perm": n_perm, "bh_resolvable": bh_possible,
                                "certified_declared": [kd, kd_bh], "certified_orderavg": [ka, ka_bh],
                                "embedding": float(emb), "final": float(fin), "peak_excess": peak,
                                "peak_layer": peak_l, "crossover_layer": cross, "reading": lines}
        if verbose:
            hdr = f"== {report['model'] or ''}{' @ ' + report['revision'] if report.get('revision') else ''} :: {name} ({n} layers) =="
            print(hdr); [print("  - " + s) for s in lines]
    if out_json:
        _json.dump(report, open(out_json, "w"), indent=1)
    return report


def main():
    import argparse
    ap = argparse.ArgumentParser(
        prog="theta-zoom",
        description="Axis-resolved dimensionality scaling: "
                    "theta_obs = theta_floor + delta, with exact permutation "
                    "nulls and a nuisance-preserving second null.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="run on arrays (npy/csv/txt)")
    d.add_argument("X", help="samples-by-features array")
    d.add_argument("labels", help="integer class labels, one per sample")
    d.add_argument("--strata", default=None, help="nuisance stratum ids for the second null")
    d.add_argument("--n-perm", type=int, default=500)
    d.add_argument("--k-orders", type=int, default=50)
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--out", default=None, help="write the result JSON here")
    d.add_argument("--plot", default=None, help="write a ladder figure (PNG) here")

    m = sub.add_parser("llm", help="per-layer battery on a Hugging Face model")
    m.add_argument("--model", required=True)
    m.add_argument("--axis", required=True, nargs="+",
                   help="axis JSON file(s) {class: [prompts]} and/or directories of them")
    m.add_argument("--revision", default=None, help="HF revision, e.g. a training checkpoint")
    m.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    m.add_argument("--n-perm", type=int, default=500)
    m.add_argument("--k-orders", type=int, default=50)
    m.add_argument("--max-len", type=int, default=128)
    m.add_argument("--no-strata", action="store_true", help="ignore *.strata.json sidecars")
    m.add_argument("--out", default=None, help="write JSON here")
    m.add_argument("--paper-seeds", action="store_true",
                   help="use the paper's seeding convention (reproduces Table 5 cells)")

    s = sub.add_parser("summarize", help="plain-language reading of a `theta-zoom llm` or `theta-zoom data` JSON (the paper's rules)")
    s.add_argument("battery_json")
    s.add_argument("--out", default=None, help="also write the report as JSON")

    x = sub.add_parser("axis", help="build an axis JSON (+ strata sidecar) from a Hugging Face dataset")
    x.add_argument("--dataset", required=True, help="dataset name, e.g. Rowan/hellaswag")
    x.add_argument("--config", default=None)
    x.add_argument("--split", default="train")
    x.add_argument("--text-field", required=True)
    x.add_argument("--label-field", required=True)
    x.add_argument("--strata-field", default=None, help="nuisance field written to the .strata.json sidecar")
    x.add_argument("--classes", nargs="*", default=None, help="explicit class values (default: the most frequent)")
    x.add_argument("--n-classes", type=int, default=8)
    x.add_argument("--n-per-class", type=int, default=16)
    x.add_argument("--min-chars", type=int, default=20)
    x.add_argument("--max-chars", type=int, default=400)
    x.add_argument("--seed", type=int, default=0)
    x.add_argument("--out", required=True)

    p = sub.add_parser("plot", help="depth-profile figure(s) from a `theta-zoom llm` JSON, or the ladder figure from a `theta-zoom data` JSON")
    p.add_argument("battery_json")
    p.add_argument("--out", default="battery.png")

    a = ap.parse_args()
    if a.cmd == "summarize":
        import json as _json
        d = _json.load(open(a.battery_json))
        if "layers" in d or "axes" in d:
            summarize(a.battery_json, a.out)
        else:
            rep = summarize_data(d)
            if a.out:
                _json.dump(rep, open(a.out, "w"), indent=1)
    elif a.cmd == "plot":
        import json as _json
        d = _json.load(open(a.battery_json))
        if "layers" in d or "axes" in d:
            _plot_battery(a.battery_json, a.out)
        else:
            _plot_data(d, a.out); print("wrote", a.out)
    elif a.cmd == "axis":
        axis_from_dataset(a.dataset, a.text_field, a.label_field, a.out, config=a.config, split=a.split,
                          n_classes=a.n_classes, n_per_class=a.n_per_class, strata_field=a.strata_field,
                          classes=a.classes, min_chars=a.min_chars, max_chars=a.max_chars, seed=a.seed)
    elif a.cmd == "data":
        X = _load_array(a.X)
        labels = _load_array(a.labels).astype(int)
        strata = _load_array(a.strata).astype(int) if a.strata else None
        r = zoom(X, labels, n_perm=a.n_perm, k_orders=a.k_orders, strata=strata, seed=a.seed)
        for k, v in r.items():
            if not isinstance(v, list):
                print(f"{k}: {v}")
        if a.out:
            import json as _json
            _json.dump({k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in r.items()}, open(a.out, "w"), indent=1)
            print("wrote", a.out)
        if a.plot:
            _plot_data(r, a.plot); print("wrote", a.plot)
    else:
        llm_battery(a.model, a.axis, a.device, a.n_perm, a.k_orders, a.max_len,
                    a.out, paper_seeds=a.paper_seeds, revision=a.revision,
                    use_strata=not a.no_strata)


if __name__ == "__main__":
    main()
