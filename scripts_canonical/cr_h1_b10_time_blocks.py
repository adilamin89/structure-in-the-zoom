"""H1-R15 (B10, referee item 5, Oct 22): is the direction shift and its overlap modulation ordinary under a null that keeps
acquisition time? Eight grating recordings, the paper's input (global SD scaling). Presentations in stimulus-time order
(stimtimes, asserted increasing) are cut into contiguous epochs of E = 48 presentations (about four minutes; ~6 per class);
an epoch is kept when it holds all eight 45-degree classes (coverage reported). Arms on the kept presentations:
  restricted  stimulus angles permuted only within each epoch (200 permutations; class sizes per epoch kept);
  balanced    each kept epoch downsampled to its smallest class count (first presentations in time), then restricted perms;
  free        angles permuted over all kept presentations (200), the usual null.
Statistics on the same subset: delta (Table 9 ladder; ten-draw floor rng 42, label-blind, shared by all permutations) and
the overlap modulation dK = Kn(22.5) - Kn(180) of Fig 2C (sixteen classes; 40 permutations per arm). One-sided p values.
No state covariates (running, pupil) are in these files. Epoch construction was fixed before any shift was looked at.
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler cr_h1_b10_time_blocks.py [n_workers]   Out: data_canonical/cr_h1_b10.json"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the eight Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_h1_b10.json"
E, N_PERM, N_PERM_DK = 48, 200, 40
RUNGS = (1, 2, 3, 4, 6, 8)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def lpr(G, idx):
    Gs = G[np.ix_(idx, idx)]; r = Gs.mean(1, keepdims=True); Gc = Gs - r - r.T + Gs.mean()
    return np.log(np.trace(Gc) ** 2 / (Gc * Gc).sum())


def make_delta(G, n):
    rng = np.random.default_rng(42)
    def sets(lab):
        m = [np.where(lab == b)[0] for b in range(8)]
        return [np.concatenate(m[:c]) for c in RUNGS]
    def f(lab):
        S = sets(lab); x = np.log([len(s) for s in S])
        if not hasattr(f, "flo"):
            f.flo = np.array([np.mean([lpr(G, rng.choice(n, len(s), replace=False)) for _ in range(10)]) for s in S])
        return float(np.polyfit(x, np.array([lpr(G, s) for s in S]) - f.flo, 1)[0])
    return f


def dK(G, lab16, om):
    idx = [np.where(lab16 == c)[0] for c in range(16)]
    tr, _, T = om.overlaps(G, idx); K = om.K_of(tr, tr, T); kd = np.diag(K); Kn = K / np.sqrt(np.outer(kd, kd))
    a = np.arange(16); sep = np.minimum(np.abs(np.subtract.outer(a, a)), 16 - np.abs(np.subtract.outer(a, a)))
    return float(Kn[sep == 1].mean() - Kn[sep == 8].mean())


def run_arm(X, phi, epoch, ow, om, seed):
    G = (X @ X.T).astype(np.float64); n = len(X)
    lab8, lab16 = ow.bins(phi, 8), ow.bins(phi, 16)
    fd = make_delta(G, n)
    obs = {"delta": fd(lab8), "dK": dK(G, lab16, om)}
    rng = np.random.default_rng(seed); out = {"n": int(n), "obs": obs}
    blocks = [np.where(epoch == e)[0] for e in np.unique(epoch)]
    def perm_restricted():
        p = phi.copy()
        for b in blocks:
            p[b] = phi[b][rng.permutation(len(b))]
        return p
    perm_free = lambda: phi[rng.permutation(n)]
    for name, pf in (("restricted", perm_restricted), ("free", perm_free)):
        nd = [fd(ow.bins(pf(), 8)) for _ in range(N_PERM)]
        nk = [dK(G, ow.bins(pf(), 16), om) for _ in range(N_PERM_DK)]
        out[name] = {"delta_p": float((1 + sum(v >= obs["delta"] for v in nd)) / (N_PERM + 1)), "delta_null_mean": float(np.mean(nd)),
                     "delta_null_sd": float(np.std(nd)), "delta_null_max": float(np.max(nd)),
                     "dK_p": float((1 + sum(v >= obs["dK"] for v in nk)) / (N_PERM_DK + 1)), "dK_null_mean": float(np.mean(nk)),
                     "dK_null_sd": float(np.std(nk)), "dK_null_max": float(np.max(nk))}
    return out


def run_one(fp):
    ow = _load("ow", "cr_order_and_withinbin_controls.py"); om = _load("om", "cr_overlap_mining.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); t = np.asarray(dat["stimtimes"], float).ravel(); del dat
    assert len(t) == len(phi) == len(X) and np.all(np.diff(t) > 0)
    lab8 = ow.bins(phi, 8)
    ep = np.arange(len(phi)) // E
    keep_ep = [e for e in np.unique(ep) if len(np.unique(lab8[ep == e])) == 8]
    keep = np.isin(ep, keep_ep)
    row = {"name": fp.stem, "n_total": int(len(phi)), "n_epochs": int(ep.max() + 1), "n_epochs_kept": len(keep_ep),
           "coverage": float(keep.mean())}
    row["full_coverage"] = run_arm(X[keep], phi[keep], ep[keep], ow, om, 1)
    bal = []
    for e in keep_ep:
        idx = np.where(ep == e)[0]; nmin = min((lab8[idx] == c).sum() for c in range(8))
        bal += [i for c in range(8) for i in idx[lab8[idx] == c][:nmin]]
    bal = np.sort(np.array(bal)); row["balanced_coverage"] = float(len(bal) / len(phi))
    row["balanced"] = run_arm(X[bal], phi[bal], ep[bal], ow, om, 2)
    a, b = row["full_coverage"], row["balanced"]
    print(f"{fp.stem[9:30]:22s} cov {row['coverage']:.2f}/{row['balanced_coverage']:.2f} | delta {a['obs']['delta']:+.3f} p_restr "
          f"{a['restricted']['delta_p']:.4f} (null {a['restricted']['delta_null_mean']:+.3f}+-{a['restricted']['delta_null_sd']:.3f}) p_free "
          f"{a['free']['delta_p']:.4f} | dK {a['obs']['dK']:+.3f} p_restr {a['restricted']['dK_p']:.3f} | balanced delta {b['obs']['delta']:+.3f} "
          f"p {b['restricted']['delta_p']:.4f} dK {b['obs']['dK']:+.3f} p {b['restricted']['dK_p']:.3f} ({time.time() - t0:.0f}s) DONE", flush=True)
    return row


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(run_one, sorted(RAW.glob("gratings*.npy"))))
    pmin, pmin_k = 1 / (N_PERM + 1) + 1e-12, 1 / (N_PERM_DK + 1) + 1e-12
    v = {f"{arm}_{s}_{st}": int(sum(r[arm][s][f"{st}_p"] <= (pmin if st == "delta" else pmin_k) for r in rows))
         for arm in ("full_coverage", "balanced") for s in ("restricted", "free") for st in ("delta", "dK")}
    v["coverage"] = [round(r["coverage"], 2) for r in rows]; v["balanced_coverage"] = [round(r["balanced_coverage"], 2) for r in rows]
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump({"E": E, "rows": rows, "verdict": v}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
