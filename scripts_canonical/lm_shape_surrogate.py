"""What carries a language model's topic lift: the class means or how each topic's prompts vary. Pythia-2.8B-deduped on the paper's
topic axis (axes/world_knowledge32.json, 8 topics x 32 prompts), every layer's last-token state (the paper's extraction,
run17_multiclass_battery.get_hidden_states: one prompt at a time, no padding).

At 32 prompts per topic each topic is sampled below its own fluctuation dimension, and the raw within / class-mean split then
moves weight from the within-class part D to the class-mean part even with no structure (Appendix K). So the split is read against
rung's shape surrogate (rung.bias_null(mode='shape'), 10 draws): every topic keeps its mean, size and variance, and all topics share
one fluctuation shape. Per layer and frame (covariance; z-scored per unit), from rung.decompose(n_null=20, seed=42, n_bias=10):
  delta, D, class_means        the data's shift and its raw split
  surrogate_delta              the surrogates' mean shift (the shift that same centres with one shared shape give)
  net_D, net_class_means       the data's split minus the surrogates' (what the topics' own fluctuation shapes add)
  class_dim_split              each topic's own fluctuation dimension from split halves (rung.class_pr(estimate='split'))
For layer 24 (three-quarters depth, the README's prompt-clouds animation) also the ladder (rung.zoom: rung sizes, PR of the topic
ladder and of the floor, declared and order-averaged shift) and every prompt's coordinates in the plane of the topic centres' top
two directions, so the animation reads this file only.
Needs the model extras (torch, transformers) and the model (5.3 GB); fp16 on mps or cuda, fp32 on cpu. --states FILE.npz reuses
cached states (H: layers x prompts x units, labels) and skips the model.
Out: data_canonical/lm_shape_surrogate_pythia-2.8b.json. Run: python scripts_canonical/lm_shape_surrogate.py [--device mps]
[--states FILE.npz] [--save-states FILE.npz]"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import rung  # noqa: E402

MODEL = "EleutherAI/pythia-2.8b-deduped"
AXIS = ROOT / "axes" / "world_knowledge32.json"
OUT = ROOT / "data_canonical" / "lm_shape_surrogate_pythia-2.8b.json"
LAYER = 24


def battery():
    d = json.load(open(AXIS))
    names = list(d)
    prompts = [p for c in names for p in d[c]]
    labels = np.repeat(np.arange(len(names)), [len(d[c]) for c in names])
    return prompts, labels, names


def encode(prompts, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = importlib.util.spec_from_file_location("run17", HERE / "run17_multiclass_battery.py")
    run17 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run17)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    states = run17.get_hidden_states(model, tok, prompts, device=device)
    return np.stack([np.asarray(s, np.float32) for s in states]), {"device": device, "dtype": "fp32" if device == "cpu" else "fp16"}


def split_row(X, y):
    d = rung.decompose(X, y, n_null=20, seed=42, n_bias=10)
    f, b, nb = d["delta_split_free"], d["bias_null"], d["net_of_bias"]["delta_split_free"]
    cm = lambda q: q["Tb"] + q["Cb"] + q["Cx"]
    cd = rung.class_pr(X, y, estimate="split")
    return {"delta": d["delta"], "D": f["D"], "class_means": cm(f), "surrogate_delta": b["delta"],
            "surrogate_D": b["delta_split_free"]["D"], "surrogate_class_means": cm(b["delta_split_free"]),
            "net_D": nb["D"], "net_class_means": cm(nb), "class_dim_split": [cd[k] for k in sorted(cd)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--states", default=None)
    ap.add_argument("--save-states", default=None)
    a = ap.parse_args()
    prompts, labels, names = battery()
    t0 = time.time()
    if a.states:
        z = np.load(a.states)
        H = np.asarray(z["H"], np.float32)
        assert np.array_equal(z["labels"], labels) and list(map(str, z["prompts"])) == prompts, "states are not this axis"
        meta = {"states": Path(a.states).name, "device": str(z["device"]) if "device" in z else None,
                "dtype": str(z["dtype"]) if "dtype" in z else None}
    else:
        H, meta = encode(prompts, a.device)
        if a.save_states:
            np.savez(a.save_states, H=H, labels=labels, prompts=np.array(prompts), names=np.array(names), **meta)
    print(f"states {H.shape} ({time.time() - t0:.0f} s)", flush=True)
    assert H.shape[:2] == (33, len(prompts)) and np.isfinite(H).all()
    out = {"model": MODEL, "axis": AXIS.name, "names": names, "n_per_class": int(np.bincount(labels).min()), "rung": rung.__version__,
           "estimators": "decompose(n_null=20, seed=42, n_bias=10); class_pr(estimate='split'); zoom(n_perm=0, k_orders=50, seed=0)",
           **meta, "frames": {}}
    for frame in ("cov", "z"):
        rows = []
        for L in range(H.shape[0]):
            X = H[L].astype(np.float64)
            X = rung._zscore(X) if frame == "z" else X
            rows.append({"layer": L, **split_row(X, labels)})
            r = rows[-1]
            print(f"{frame} layer {L:2d}: shift {r['delta']:+.3f} surrogate {r['surrogate_delta']:+.3f} | net D {r['net_D']:+.3f} "
                  f"net class means {r['net_class_means']:+.3f}", flush=True)
        out["frames"][frame] = rows
    X = H[LAYER].astype(np.float64)
    zr = rung.zoom(X, labels, n_perm=0, k_orders=50, seed=0)
    M = np.stack([X[labels == c].mean(0) for c in range(len(names))]); Mc = M - M.mean(0)
    _, s, Vt = np.linalg.svd(Mc, full_matrices=False)
    xy = (X - M.mean(0)) @ Vt[:2].T
    tot = float(((X - X.mean(0)) ** 2).sum())
    out["layer24"] = {"layer": LAYER, "rung_sizes": zr["rung_sizes"], "pr_obs": zr["pr_obs"], "pr_floor": zr["pr_floor"],
                      "theta_obs": zr["theta_obs"], "theta_floor": zr["theta_floor"], "delta": zr["delta"],
                      "delta_orderavg": zr["delta_orderavg"], "theta_floor_se": zr["theta_floor_se"],
                      "plane_share_of_class_means": float((s[:2] ** 2).sum() / (s ** 2).sum()),
                      "plane_share_of_all": float((((X - M.mean(0)) @ Vt[:2].T) ** 2).sum() / tot),
                      "xy": np.round(xy, 4).tolist(), "centres_xy": np.round(Mc @ Vt[:2].T, 4).tolist()}
    keep = [r["surrogate_delta"] / r["delta"] for r in out["frames"]["cov"]]
    out["summary"] = {f: {"net_D_min": min(r["net_D"] for r in rows), "net_D_max": max(r["net_D"] for r in rows),
                          "net_D_positive": sum(r["net_D"] > 0 for r in rows), "n_layers": len(rows),
                          "net_class_means_min": min(r["net_class_means"] for r in rows),
                          "net_class_means_max": max(r["net_class_means"] for r in rows),
                          "surrogate_share_of_shift_max": max(r["surrogate_delta"] / r["delta"] for r in rows),
                          "surrogate_share_of_shift_min": min(r["surrogate_delta"] / r["delta"] for r in rows)}
                      for f, rows in out["frames"].items()}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"wrote {OUT.name}; surrogate keeps {min(keep):+.2f} to {max(keep):+.2f} of the shift (cov); {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
