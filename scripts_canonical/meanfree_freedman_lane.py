"""The paper's within arm (Appendix K, meanfree_shift_exact.py: the mean-free shift on 32 label-blind halves of the eight grating
recordings) re-tested under the Freedman-Lane null (rung 1.13's default for within_arm), which held its level in
null_calibration.py's heavy-tailed regime where the paper's own-span permutation null ran liberal.

Per half (S1 odd/even, S2 first/second in time; the generator's loader, bins and Gram procedure, imported from scripts_canonical):
  observed  the mean-free shift exactly as the generator computes it (checked against meanfree_shift_exact.json)
  FL        null data = the half's fitted class means + its residuals permuted across rows (unrestricted) or inside the half's
            48-presentation epochs (epoch), then the generator's whole procedure (own class-mean span removed, ladder, floor
            seed 42): one-sided p and z.
Needs the eight grating recordings in raw/ (Figshare release 8279387, as for meanfree_shift_exact.py).
Out: data_canonical/meanfree_freedman_lane.json. Run: python scripts_canonical/meanfree_freedman_lane.py [N_DRAWS] [WORKERS]
(defaults 50 and 4; about three CPU hours)."""
import importlib.util
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EPOCH = 48


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fl_half(XB, yB, eB, ow, R1, n_draws, seed):
    """The generator's measured_half observation and two Freedman-Lane nulls (unrestricted, inside epochs)."""
    GB = (XB @ XB.T).astype(np.float64)

    def shift(G, X, lab):
        Q = R1.mean_subspace(X, lab)
        P = (X @ Q.astype(np.float32)).astype(np.float64)
        return ow.ladder(G - P @ P.T, lab, R1.rung_sets(lab), np.random.default_rng(42))["delta"]

    obs = shift(GB, XB, yB)
    M = np.stack([XB[yB == c].mean(0) for c in range(8)])                  # fitted class means (float32)
    R = XB - M[yB]                                                          # residuals
    RR = (R @ R.T).astype(np.float64); MR = (M @ R.T).astype(np.float64); MM = (M @ M.T).astype(np.float64)
    out = {"obs": float(obs)}
    for name in ("unrestricted", "epoch"):
        rng = np.random.default_rng(seed + (0 if name == "unrestricted" else 1))
        nd = []
        for _ in range(n_draws):
            if name == "unrestricted":
                pi = rng.permutation(len(yB))
            else:
                pi = np.arange(len(yB))
                for b in np.unique(eB):
                    i = np.where(eB == b)[0]
                    pi[i] = i[rng.permutation(len(i))]
            Xs = M[yB] + R[pi]
            # the Gram of the null data from blocks, without the n^2 d product: (M[y] + R[pi]) (M[y] + R[pi])^T
            A = MR[yB][:, pi]
            Gs = MM[yB][:, yB] + A + A.T + RR[pi][:, pi]
            nd.append(shift(Gs, Xs, yB))
        nd = np.array(nd)
        out[name] = {"mean": float(nd.mean()), "sd": float(nd.std()), "max": float(nd.max()),
                     "z": float((obs - nd.mean()) / nd.std()) if nd.std() > 0 else 0.0,
                     "p_one": float((1 + (nd >= obs).sum()) / (n_draws + 1)), "draws": nd.tolist()}
    return out


def run_one(args):
    fp, repo, n_draws = args
    warnings.filterwarnings("ignore")
    sc = Path(repo) / "scripts_canonical"
    sys.path.insert(0, str(sc))
    ow = _load("ow", sc / "cr_order_and_withinbin_controls.py")
    R1 = _load("h4r1", sc / "meanfree_shift.py")
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T)
    X /= X.std() + 1e-9
    y = ow.bins(np.asarray(dat["istim"], float).ravel(), 8)
    t = np.asarray(dat["stimtimes"], float).ravel()
    del dat
    assert len(t) == len(y) and np.all(np.diff(t) > 0), fp.stem
    n = len(y); ep = np.arange(n) // EPOCH
    row = {"name": fp.stem}
    for s, h in {"S1": np.arange(n) % 2, "S2": (np.arange(n) >= n // 2).astype(int)}.items():
        for hv in (0, 1):
            t0 = time.time()
            sel = np.where(h == hv)[0]
            row[f"{s}_half{hv}"] = fl_half(X[sel], y[sel], ep[sel], ow, R1, n_draws, 9000 + 10 * hv + (0 if s == "S1" else 100))
            r = row[f"{s}_half{hv}"]
            print(f"{fp.stem[:30]} {s} half{hv}: shift {r['obs']:+.4f} | FL unr z {r['unrestricted']['z']:+.1f} p {r['unrestricted']['p_one']:.4f}"
                  f" | FL epoch z {r['epoch']['z']:+.1f} p {r['epoch']['p_one']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    return row


def main():
    repo, raw = str(ROOT), ROOT / "raw"
    n_draws = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    files = sorted(raw.glob("gratings*.npy"))
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        rows = list(ex.map(run_one, [(f, repo, n_draws) for f in files]))
    paper = {r["name"]: r for r in json.load(open(Path(repo) / "data_canonical" / "meanfree_shift_exact.json"))["rows"]}
    cells = []
    for r in rows:
        for k, v in r.items():
            if not k.startswith("S"):
                continue
            pp = paper[r["name"]][k]
            cells.append({"name": r["name"], "half": k, "obs": v["obs"], "obs_paper": pp["meanfree"]["delta"],
                          "fl_p_unr": v["unrestricted"]["p_one"], "fl_p_epoch": v["epoch"]["p_one"],
                          "fl_z_unr": v["unrestricted"]["z"], "fl_z_epoch": v["epoch"]["z"],
                          "paper_p_unr": pp["null_unrestricted"]["p_one"], "paper_p_epoch": pp["null_epoch"]["p_one"],
                          "paper_z_unr": (pp["meanfree"]["delta"] - pp["null_unrestricted"]["mean"]) / pp["null_unrestricted"]["sd"],
                          "paper_z_epoch": (pp["meanfree"]["delta"] - pp["null_epoch"]["mean"]) / pp["null_epoch"]["sd"]})
    summ = {"n_cells": len(cells), "n_draws": n_draws,
            "max_abs_obs_vs_paper": max(abs(c["obs"] - c["obs_paper"]) for c in cells),
            "fl_both_p_le_0.05": sum(c["fl_p_unr"] <= 0.05 and c["fl_p_epoch"] <= 0.05 for c in cells),
            "fl_conjunction_p": [max(c["fl_p_unr"] for c in cells), max(c["fl_p_epoch"] for c in cells)],
            "fl_z_min": [min(c["fl_z_unr"] for c in cells), min(c["fl_z_epoch"] for c in cells)],
            "paper_z_min": [min(c["paper_z_unr"] for c in cells), min(c["paper_z_epoch"] for c in cells)]}
    out = {"summary": summ, "cells": cells, "rows": rows}
    print(json.dumps(summ, indent=1), flush=True)
    (ROOT / "data_canonical" / "meanfree_freedman_lane.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
