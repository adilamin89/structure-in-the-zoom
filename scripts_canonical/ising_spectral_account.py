"""The centred |m|-sorted Ising ladder's delta(T_c) from the spectrum, nothing fitted.
Rung covariance = bulk (uniform mode removed; moments B1, B2 per chain, Wishart-corrected) + magnetization mode with the
rung's own variance v_r = N Var(m); sample PR(n, v) = (B1 + v)^2 / (B2 + v^2 + (B1 + v)^2 / n); floor at each size uses the
full set's v. Ablations: A bulk moments fixed at L = 32's per-spin values; B P(m) replaced by a Gaussian of the same <m^2>
(rung variance = v_full Var(Z | |Z| < z_q)); C log PR linearised in the mode share around 0. Out: data_canonical/ising_spectral_account.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lattice_ground_truth as h  # noqa: E402

iwm = h.iwm
FR = [0.1, 0.25, 0.5, 0.75, 1.0]


def slope(n, y):
    return float(np.polyfit(np.log(n), y, 1)[0])


def logpr(n, v, B1, B2):
    return np.log((B1 + v) ** 2 / (B2 + v ** 2 + (B1 + v) ** 2 / n))


def chain_terms(cfg):
    n, N = cfg.shape
    m = cfg.mean(1)
    sizes = [max(10, int(f * n)) for f in FR]
    order = np.argsort(np.abs(m))
    v_r = [N * float(np.var(m[order[:s]])) for s in sizes]
    v_full = N * float(np.var(m))
    Y = cfg - m[:, None]
    Y = Y - Y.mean(0)
    g = (Y @ Y.T) / n / N
    B1 = float(np.trace(g))
    B2 = float((g * g).sum() - B1 ** 2 / n)
    return sizes, np.array(v_r) / N, v_full / N, B1, B2, float(np.mean(m ** 2))


def gauss_trunc_var(q):
    """Var(Z | |Z| < z_q) for standard normal Z, with P(|Z| < z_q) = q (the rung's quantile): P(m) Gaussian, same <m^2>."""
    from scipy import stats
    out = []
    for qq in q:
        if qq >= 1:
            out.append(1.0)
            continue
        a = stats.norm.ppf(0.5 + qq / 2)
        out.append(1 - 2 * a * stats.norm.pdf(a) / qq)
    return np.array(out)


def predict(sizes, v_r, v_full, B1, B2, mode="full"):
    n = np.array(sizes, float)
    if mode == "C":   # linearised in the share s = v / (B1 + v): log PR ~ log PR(n, 0) - d/ds at 0 * s
        s_r = v_r / (B1 + v_r)
        s_f = v_full / (B1 + v_full)
        eps = 1e-6
        d0 = (logpr(n, eps * B1, B1, B2) - logpr(n, 0.0, B1, B2)) / (eps)   # per unit of v/B1
        return slope(n, d0 * (s_r - s_f))
    lr = logpr(n, v_r, B1, B2)
    lf = logpr(n, v_full, B1, B2)
    return slope(n, lr - lf)


def main():
    out = {"rows": [], "per_L": {}}
    ref = None
    for L, nch in ((32, 16), (64, 16), (128, 16), (256, 8)):
        art = json.load(open(h.OUTDIR / f"lattice_ground_truth_ising_L{L}_t1.00.json"))
        recs = []
        for c in range(nch):
            seed = 910_000 + 1000 * L + 10 * c + 100
            cfg = h.ising_chain(L, h.T_C, seed, 200, 400)
            rng = np.random.default_rng(seed)        # replay the rng exactly as ising_point
            iwm.N_CONFIGS = 200
            for name, cent, sk in iwm.ARMS:
                iwm.delta_arm(cfg[:200], rng, cent, sk)
            meas = None
            for name, cent, sk in iwm.ARMS:
                v = iwm.delta_arm(cfg[200:], rng, cent, sk)
                if name == "centered_absm":
                    meas = v
            assert abs(meas - art["chains"][c]["second"]["centered_absm"]) < 1e-9, (L, c, meas)
            sizes, v_r, v_full, B1, B2, m2 = chain_terms(cfg[200:])
            if ref is None:
                ref = (B1, B2 * L ** 2)     # per-spin-normalised bulk at L = 32 (B1 ~ O(1), B2 * N ~ O(1) if bulk PR ~ N)
            p = predict(sizes, v_r, v_full, B1, B2)
            pA = predict(sizes, v_r, v_full, ref[0], ref[1] / L ** 2)
            q = np.array(sizes, float) / sizes[-1]
            pB = predict(sizes, v_full * gauss_trunc_var(q), v_full, B1, B2)
            pC = predict(sizes, v_r, v_full, B1, B2, "C")
            rec = {"L": L, "chain": c, "measured": meas, "pred": p, "pred_A": pA, "pred_B": pB, "pred_C": pC, "m2": m2,
                   "B1": B1, "B2": B2, "bulk_PR": B1 ** 2 / B2, "share_full": v_full / (B1 + v_full), "v_r": list(map(float, v_r))}
            recs.append(rec)
            print(f"L={L} c{c}: meas {meas:+.3f} pred {p:+.3f} A {pA:+.3f} B {pB:+.3f} C {pC:+.3f} share {rec['share_full']:.3f} "
                  f"bulkPR {rec['bulk_PR']:.0f}", flush=True)
        out["rows"] += recs
        out["per_L"][L] = {k: float(np.mean([r[k] for r in recs])) for k in ("measured", "pred", "pred_A", "pred_B", "pred_C",
                                                                             "m2", "share_full", "bulk_PR")}
        json.dump(out, open(h.OUTDIR / "ising_spectral_account.json", "w"), indent=1)
    Ls = sorted(out["per_L"])
    m2 = np.array([out["per_L"][L]["m2"] for L in Ls])
    for k in ("measured", "pred", "pred_A", "pred_B", "pred_C"):
        y = np.abs([out["per_L"][L][k] for L in Ls])
        out[f"exp_on_m2_{k}"] = float(np.polyfit(np.log(m2), np.log(y), 1)[0])
        out[f"exp_on_L_{k}"] = float(-np.polyfit(np.log(Ls), np.log(y), 1)[0])
    print({k: v for k, v in out.items() if k.startswith("exp")}, flush=True)
    print({L: {k: round(v, 3) for k, v in d.items()} for L, d in out["per_L"].items()}, flush=True)
    json.dump(out, open(h.OUTDIR / "ising_spectral_account.json", "w"), indent=1)
    print("DONE", flush=True)


def rungbulk():
    """R43b: each rung's own bulk moments; floor uses the full set's."""
    out = {}
    for L, nch in ((32, 16), (64, 16), (128, 16), (256, 8)):
        art = json.load(open(h.OUTDIR / f"lattice_ground_truth_ising_L{L}_t1.00.json"))
        preds, meas = [], []
        for c in range(nch):
            seed = 910_000 + 1000 * L + 10 * c + 100
            cfg = h.ising_chain(L, h.T_C, seed, 200, 400)[200:]
            n, N = cfg.shape
            m = cfg.mean(1)
            o = np.argsort(np.abs(m))
            sizes = [max(10, int(f * n)) for f in FR]
            _, _, v_full, B1f, B2f, _ = chain_terms(cfg)
            lr, lf = [], []
            for s_ in sizes:
                sel = o[:s_]
                Y = cfg[sel] - m[sel, None]
                Y = Y - Y.mean(0)
                g = Y @ Y.T / s_ / N
                B1 = float(np.trace(g))
                B2 = float((g * g).sum() - B1 ** 2 / s_)
                v = float(np.var(m[sel]))
                lr.append(logpr(s_, v, B1, B2))
                lf.append(logpr(s_, v_full, B1f, B2f))
            preds.append(slope(np.array(sizes, float), np.array(lr) - np.array(lf)))
            meas.append(art["chains"][c]["second"]["centered_absm"])
        out[L] = {"pred": float(np.mean(preds)), "measured": float(np.mean(meas))}
        print(L, out[L], flush=True)
    res = json.load(open(h.OUTDIR / "ising_spectral_account.json"))
    res["R43b_rung_bulk"] = out
    Ls = sorted(out)
    m2 = [res["per_L"][str(L)]["m2"] if str(L) in res["per_L"] else res["per_L"][L]["m2"] for L in Ls]
    res["R43b_exp_on_m2"] = float(np.polyfit(np.log(m2), np.log([abs(out[L]["pred"]) for L in Ls]), 1)[0])
    print("exp on m2", res["R43b_exp_on_m2"], flush=True)
    json.dump(res, open(h.OUTDIR / "ising_spectral_account.json", "w"), indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rungbulk":
        rungbulk()
    else:
        main()
