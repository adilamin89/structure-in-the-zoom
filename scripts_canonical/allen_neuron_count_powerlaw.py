"""Does one within-class power-law spectrum,
measured on half A, predict PR(W) under random neuron subsampling and the neuron count where the class-mean term changes sign,
on half B?

Halves are temporal: epochs of 48 consecutive presentations, A = even epochs, B = odd. Per recording (paper input, global SD):
  half A  pooled within-class residuals (45-degree classes) -> eigenvalues (Gram) -> alpha = -slope of log lambda_k on log k,
          ranks 11-500. Predicted PR(W)(N): (sim) a D = 4000 population with spectrum k^-alpha and Haar eigenvectors, random
          N-neuron principal submatrices (10 draws), PR = tr^2 / tr(C^2); (asym) N^{2(1-alpha)} scaled to the sim at N = 4000.
          Class-mean quantities at each N (10 random neuron draws on half A): beta = Tr B / Tr W, PR(B), xi = Tr WB/(Tr W Tr B).
          Predicted full-set class-mean term c(N) = log[(1+beta)^2 / (1 + beta^2 PR_W/PR_B + 2 xi beta PR_W)] with PR_W from the
          power law (exact rewrite of log[(1+b1/w1)^2/(1+(b2+2x)/w2)]: b2/w2 = beta^2 PR_W/PR_B, x/w2 = xi beta PR_W).
  half B  measured PR(W)(N), c(N) (10 draws).
Crossover N*: the smallest grid N from which c(N) stays negative (predicted from A, measured on B).
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler allen_neuron_count_powerlaw.py [n_workers]   Out: data_canonical/allen_neuron_count_powerlaw.json"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
OUT = HERE.parent / "data_canonical" / "allen_neuron_count_powerlaw.json"
GRID = (64, 128, 256, 512, 1024, 4096)
E, DRAWS, D_SIM = 48, 10, 4000


def bins(phi, nb):
    return np.clip(np.digitize(phi % (2 * np.pi), np.linspace(0, 2 * np.pi, nb + 1)) - 1, 0, nb - 1)


def terms(X, lab):
    X = X - X.mean(0)
    cm = np.stack([X[lab == c].mean(0) for c in range(8)]); p = np.bincount(lab, minlength=8) / len(lab)
    n = len(X); Wh = (X - cm[lab]) / np.sqrt(n); Bh = np.sqrt(p)[:, None] * cm
    w1, b1 = (Wh * Wh).sum(), (Bh * Bh).sum()
    w2 = ((Wh @ Wh.T) ** 2).sum(); b2 = ((Bh @ Bh.T) ** 2).sum(); x = ((Wh @ Bh.T) ** 2).sum()
    return {"beta": b1 / w1, "PR_W": w1 ** 2 / w2, "PR_B": b1 ** 2 / b2, "xi": x / (w1 * b1),
            "c": float(np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2)))}


def c_of(beta, prw, prb, xi):
    return float(np.log((1 + beta) ** 2 / (1 + beta ** 2 * prw / prb + 2 * xi * beta * prw)))


def sim_pr(alpha, Ns, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D_SIM, D_SIM)))
    lam = np.arange(1, D_SIM + 1, dtype=float) ** (-alpha)
    C = (Q * lam) @ Q.T
    out = {}
    for N in Ns:
        prs = []
        for d in range(DRAWS):
            idx = rng.choice(D_SIM, N, replace=False); S = C[np.ix_(idx, idx)]
            prs.append(np.trace(S) ** 2 / (S * S).sum())
        out[N] = float(np.mean(prs))
    return out


def crossover(cs, grid):
    neg = [c < 0 for c in cs]
    for i in range(len(grid)):
        if all(neg[i:]):
            return grid[i]
    return None


def run_one(fp):
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab = bins(phi, 8); ep = (np.arange(len(phi)) // E) % 2
    A, B = ep == 0, ep == 1
    nN = X.shape[1]; grid = [g for g in GRID if g < nN] + [nN]
    XA = X[A].astype(np.float64); la = lab[A]
    R = XA - np.stack([XA[la == c].mean(0) for c in range(8)])[la]
    ev = np.sort(np.linalg.eigvalsh((R @ R.T) / len(R)))[::-1]
    k = np.arange(11, 501)
    alpha = float(-np.polyfit(np.log(k), np.log(ev[k - 1]), 1)[0])
    pr_full_A = float(ev.sum() ** 2 / (ev ** 2).sum())
    sim = sim_pr(alpha, [g for g in grid if g <= D_SIM])
    rows = []
    for N in grid:
        tA, tB = [], []
        for d in range(DRAWS if N < nN else 1):
            idx = np.sort(np.random.default_rng(1000 * N + d).choice(nN, N, replace=False)) if N < nN else np.arange(nN)
            tA.append(terms(XA[:, idx], la)); tB.append(terms(X[B][:, idx].astype(np.float64), lab[B]))
        mA = {q: float(np.mean([t[q] for t in tA])) for q in tA[0]}; mB = {q: float(np.mean([t[q] for t in tB])) for q in tB[0]}
        pw_sim = sim.get(N, sim[max(sim)] * 1.0)
        pw_asym = float(sim[max(sim)] * (N / max(sim)) ** (2 * (1 - alpha))) if alpha < 1 else None
        rows.append({"N": int(N), "A": mA, "B": mB, "PR_W_pred_sim": pw_sim, "PR_W_pred_asym": pw_asym,
                     "c_pred_sim": c_of(mA["beta"], pw_sim, mA["PR_B"], mA["xi"]),
                     "c_pred_A_measured_PRW": c_of(mA["beta"], mA["PR_W"], mA["PR_B"], mA["xi"])})
        print(f"{fp.stem[9:30]:22s} N {N:6d} PR_W B {mB['PR_W']:7.1f} pred sim {pw_sim:7.1f} | c B {mB['c']:+.3f} pred {rows[-1]['c_pred_sim']:+.3f} ({time.time() - t0:.0f}s)", flush=True)
    out = {"name": fp.stem, "alpha_A": alpha, "pr_W_full_A": pr_full_A, "rows": rows,
           "crossover_B": crossover([r["B"]["c"] for r in rows], grid), "crossover_pred": crossover([r["c_pred_sim"] for r in rows], grid)}
    print(f"{fp.stem} alpha {alpha:.3f} crossover pred {out['crossover_pred']} measured {out['crossover_B']} DONE", flush=True)
    return out


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        recs = list(ex.map(run_one, sorted(RAW.glob("gratings*.npy"))))
    ok = lambda r: [x for x in r["rows"] if x["N"] >= 256 and x["N"] <= D_SIM]
    v = {"prw_within_1p5_at_N_ge_256": int(sum(all(1 / 1.5 <= x["PR_W_pred_sim"] / x["B"]["PR_W"] <= 1.5 for x in ok(r)) for r in recs)),
         "crossover_within_one_step": int(sum(r["crossover_pred"] is not None and r["crossover_B"] is not None and
                                              abs(GRID.index(r["crossover_pred"]) - GRID.index(r["crossover_B"])) <= 1
                                              if (r["crossover_pred"] in GRID and r["crossover_B"] in GRID) else r["crossover_pred"] == r["crossover_B"]
                                              for r in recs)),
         "alpha": {r["name"]: round(r["alpha_A"], 3) for r in recs},
         "crossovers": {r["name"]: [r["crossover_pred"], r["crossover_B"]] for r in recs}}
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump({"recordings": recs, "verdict": v}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
