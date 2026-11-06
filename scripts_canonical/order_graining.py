"""App G's direction-sorted coarse-graining cross-fitted on temporal halves, against random and
rate-and-reliability-matched random blocks. Ladder = the paper's (cr_graining_table.shift estimator) computed through the
trial Gram matrix (llm_battery.ladder / floor with double-centred subsets = centred PR). Out: data_canonical/order_graining.json"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_battery as hb  # noqa: E402

spec = importlib.util.spec_from_file_location("cg", HERE / "cr_graining_table.py")
CG = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CG)
RAW = HERE.parent / "raw"
KS = [1, 4, 16, 32]
hb.N_FLOOR = 10          # App G: ten-draw floor
COUNTS = [1, 2, 3, 4, 6, 8]
TH = np.linspace(0, 2 * np.pi, 8, endpoint=False)


def bins_of(phi):
    return np.clip(np.digitize(np.asarray(phi, float) % (2 * np.pi), np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)


def delta_gram(V, bins, seed):
    """V: units x trials. The paper's ladder via the trial Gram (centred PR of trial subsets)."""
    G = (V.T @ V).astype(np.float64)
    mem = [np.where(bins == b)[0] for b in range(8)]
    th, sizes = hb.ladder(G, mem, list(range(8)), COUNTS)
    return th - hb.floor(G, len(bins), sizes, seed)


def reproduce():
    ref = {r["K"]: r for r in json.load(open(HERE.parent / "data_canonical" / "cr_graining_table.json"))["rows"]}
    d = np.load(CG.RAW, allow_pickle=True).item()
    X = np.asarray(d["sresp"], np.float32)
    X /= X.std() + 1e-9
    bins = np.clip(np.digitize(np.asarray(d["istim"], float), np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
    means = np.stack([X[:, bins == b].mean(axis=1) for b in range(8)])
    pref = np.angle((means * np.exp(1j * TH)[:, None]).sum(0)) % (2 * np.pi)
    K = 32
    n = X.shape[0] // K
    out = {}
    for tag, order in (("pref_sorted", np.argsort(pref)), ("random_blocks", np.random.default_rng(7).permutation(X.shape[0]))):
        V = X[order[: n * K]].reshape(n, K, X.shape[1]).mean(axis=1)
        dl, _, _ = CG.shift(np.ascontiguousarray(V.T), bins, seed=1000 + K)
        out[tag] = dl
        assert abs(dl - ref[K][tag]["delta"]) < 0.005, (tag, dl, ref[K][tag]["delta"])
    print("reproduced GT3 K=32:", out, flush=True)
    return out


def orders_from(XA, bA, rng):
    means = np.stack([XA[:, bA == b].mean(axis=1) for b in range(8)])
    pref = np.angle((means * np.exp(1j * TH)[:, None]).sum(0)) % (2 * np.pi)
    rate = XA.mean(axis=1)
    odd = np.arange(XA.shape[1]) % 2 == 1
    m1 = np.stack([XA[:, (bA == b) & odd].mean(axis=1) for b in range(8)])
    m2 = np.stack([XA[:, (bA == b) & ~odd].mean(axis=1) for b in range(8)])
    a, b = m1 - m1.mean(0), m2 - m2.mean(0)
    rel = (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-12)
    dec = np.digitize(rate, np.quantile(rate, np.linspace(0, 1, 11)[1:-1]))
    matched = np.lexsort((rel, dec))
    return {"dir_sorted": np.argsort(pref), "random": rng.permutation(XA.shape[0]), "matched": matched}


def main():
    res = {"reproduction": reproduce(), "rows": []}
    for fp in sorted(RAW.glob("gratings*.npy")):
        d = np.load(fp, allow_pickle=True).item()
        X = np.asarray(d["sresp"], np.float32)
        X /= X.std() + 1e-9
        bins = bins_of(d["istim"])
        T = X.shape[1]
        halves = {"A": np.arange(T) < T // 2, "B": np.arange(T) >= T // 2}
        rec = {"name": fp.stem, "cross": {}}
        for src, tgt in (("A", "B"), ("B", "A")):
            rng = np.random.default_rng(21)
            orders = orders_from(X[:, halves[src]], bins[halves[src]], rng)
            Xt, bt = X[:, halves[tgt]], bins[halves[tgt]]
            cell = {}
            for arm, order in orders.items():
                vals = []
                for K in KS:
                    n = X.shape[0] // K
                    V = Xt[order[: n * K]].reshape(n, K, Xt.shape[1]).mean(axis=1) if K > 1 else Xt
                    vals.append(delta_gram(V, bt, 1000 + K))
                cell[arm] = vals
            rec["cross"][f"{src}{tgt}"] = cell
            print(fp.stem[:30], src, "->", tgt, {a: [round(v, 3) for v in vs] for a, vs in cell.items()}, flush=True)
        ret = {arm: float(np.mean([rec["cross"][k][arm][-1] / rec["cross"][k][arm][0] for k in rec["cross"]])) for arm in
               ("dir_sorted", "random", "matched")}
        rec["retention_K32"] = ret
        res["rows"].append(rec)
        json.dump(res, open(hb.OUTDIR / "order_graining.json", "w"), indent=1)
    hits = sum(r["retention_K32"]["dir_sorted"] - r["retention_K32"]["matched"] >= 0.3 for r in res["rows"])
    res["P_hits"] = int(hits)
    print("retention K32 per recording:", [(r["name"][9:25], {k: round(v, 2) for k, v in r["retention_K32"].items()})
                                          for r in res["rows"]], "hits", hits, flush=True)
    json.dump(res, open(hb.OUTDIR / "order_graining.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
