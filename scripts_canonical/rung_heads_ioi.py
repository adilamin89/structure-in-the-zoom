"""`rung llm --heads` on GPT-2 small's indirect-object circuit (Appendix I): the paper-seed replay of llm_ioi.json, the spread of
the default-seed ranking over ten seeds, and the per-head split into within-class and class-mean parts.
Heads: the eleven name-mover and backup heads path patching found (Wang et al. 2023) against the other 133; the two negative
name movers reported by rank. The head writes are deterministic and encoded once; each seed rescores them.
Run: VECLIB_MAXIMUM_THREADS=2 python -X faulthandler rung_heads_ioi.py [n_workers]   Out: data_canonical/rung_heads_ioi.json"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rung  # noqa: E402

DATA = HERE.parent / "data_canonical"
AXIS = HERE.parent / "axes" / "ioi_names.json"
KNOWN = ["9.6", "9.9", "10.0", "10.10", "10.6", "10.2", "10.1", "11.2", "9.7", "9.0", "11.9"]
NAME_MOVERS, NEGATIVE = ["9.6", "9.9", "10.0"], ["10.7", "11.10"]
CACHE = DATA / "rung_heads_ioi_writes.npy"


def aucs(rows):
    names = np.array([r["head"] for r in rows])
    pos, nm = np.isin(names, KNOWN), np.isin(names, NAME_MOVERS)
    z = np.array([r["z"] for r in rows])
    rank = (-z).argsort().argsort() + 1
    out = {"AUC_z": rung._auc(z, pos), "AUC_z_name_movers": rung._auc(z, nm),
           "AUC_classmean_frac": rung._auc(np.array([r["classmean_frac"] for r in rows]), pos),
           "rank_negative": {h: int(rank[names == h][0]) for h in NEGATIVE}}
    for k in ("within", "class_mean"):
        if all(k in r for r in rows):
            out[f"AUC_{k}"] = rung._auc(np.array([r[k] for r in rows]), pos)
    return out


def score_seed(s):
    H = np.load(CACHE)
    _, _, _, labels, _ = rung._load_axis(str(AXIS), use_strata=False)
    t0 = time.time()
    rows = rung.head_scores(H, labels, list(range(H.shape[0])), paper_seeds=False, n_perm=100, k_orders=50, seed=s, split=False)
    a = aucs(rows)
    print(f"seed {s}: AUC_z {a['AUC_z']:.3f}, name movers {a['AUC_z_name_movers']:.3f}, class-mean share {a['AUC_classmean_frac']:.3f}, "
          f"negative ranks {a['rank_negative']} ({time.time() - t0:.0f}s)", flush=True)
    return a


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    _, _, prompts, labels, _ = rung._load_axis(str(AXIS), use_strata=False)
    H, keep = rung.head_writes("openai-community/gpt2", prompts)
    np.save(CACHE, H)
    paper_rows = rung.head_scores(H, labels, keep, paper_seeds=True, n_perm=100, k_orders=50, split=True)
    paper = aucs(paper_rows)
    art = json.load(open(DATA / "llm_ioi.json"))
    paper["replays_llm_ioi"] = bool(paper["AUC_z"] == art["AUC_z"] and paper["AUC_classmean_frac"] == art["AUC_classmean_frac"])
    print("paper seeds:", json.dumps(paper), flush=True)
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        seeds = list(ex.map(score_seed, range(10)))
    sp = {k: {"mean": float(np.mean([a[k] for a in seeds])), "sd": float(np.std([a[k] for a in seeds], ddof=1)),
              "min": float(min(a[k] for a in seeds)), "max": float(max(a[k] for a in seeds))}
          for k in ("AUC_z", "AUC_z_name_movers", "AUC_classmean_frac")}
    pos = [r for r in paper_rows if r["head"] in KNOWN]; oth = [r for r in paper_rows if r["head"] not in KNOWN]
    split = {grp: {"within_median": float(np.median([r["within"] for r in g])), "class_mean_median": float(np.median([r["class_mean"] for r in g])),
                   "class_mean_negative": int(sum(r["class_mean"] < 0 for r in g)), "n": len(g),
                   "classmean_frac_median": float(np.median([r["classmean_frac"] for r in g]))} for grp, g in (("known", pos), ("others", oth))}
    out = {"paper_seeds": paper, "default_seeds": seeds, "default_spread": sp, "split": split, "heads_paper_seeds": paper_rows}
    print("spread:", json.dumps(sp), "\nsplit:", json.dumps(split), flush=True)
    json.dump(out, open(DATA / "rung_heads_ioi.json", "w"), indent=1)
    CACHE.unlink()
    print("wrote rung_heads_ioi.json ALL DONE", flush=True)


if __name__ == "__main__":
    main()
