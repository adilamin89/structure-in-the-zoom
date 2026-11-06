"""Does a shared correlation matrix with class-specific variances reproduce the shift?

Eight grating recordings, the paper's input (global SD scaling). Tuning-free residuals as withinclass_shuffle_identification 'resid': each
presentation's residual from the leave-one-out mean of its 7.5-degree sub-bin plus its 45-degree class mean
(remove_withinbin 8 x 6); R = Y - class mean. Surrogates (seeds 7, 8, 9):
  (i)  shared correlation, class-specific variance profiles, D_c R_pool D_c: residuals standardized neuron by neuron within
       each class, all standardized vectors pooled, every class redrawn from the pool (one permutation of the pooled rows,
       split at the class counts), re-centred and re-standardized within class, then each class's measured per-neuron SDs and
       mean restored.
  (iii) as (i), but each class's per-neuron SDs predicted from its class means by ONE mean-variance law per
       recording, sigma_ic = a mu_ic^b, fitted on half A (log SD on log mean): do the variance profiles just follow tuning?
  (iv) (as run57) as (i), SDs from the gain on the TUNED response: sigma_ic^2 = v_i + s^2 (m_ic - mu0_i)^2, one s^2 fitted on
       half A, v_i the neuron's class-independent variance.
  (v) as (i), variance linear in the class mean: sigma_ic^2 = v_i + s^2 m_ic (Poisson-like), one s^2 fitted on half A.
  (ii) shared covariance shape, one scalar variance per class: as (i) but each class gets the pooled per-neuron SD profile
       times a scalar that matches the class's total variance (covariances proportional to one matrix): the bias floor of the
       within-class part from class-mean estimation, unequal counts and weights.
Designs: actual class counts; balanced (every class downsampled to the smallest count, first presentations in time); held out
(all surrogate inputs from half A = alternate presentations of each 7.5-degree sub-bin, surrogate sized to half B's counts,
compared with half B's own residual ladder; both ways). Checks: recovered class means (max |diff|), per-neuron per-class SDs
(max relative error), and the shared correlation matrix (r between off-diagonal entries of the surrogate's class-pooled
correlation and R_pool on 1000 random neurons). Ladder: Table 9 engine (ow.ladder, rng 42).
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler correlation_vs_variance.py [n_workers]   Out: data_canonical/correlation_vs_variance.json"""
import importlib.util
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"
OUT = HERE.parent / "data_canonical" / "correlation_vs_variance.json"
SEEDS = (7, 8, 9)
RUNGS = (1, 2, 3, 4, 6, 8)


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def class_moments(Y, lab):
    mu = {c: Y[lab == c].mean(0) for c in range(8)}
    sd = {c: Y[lab == c].std(0) for c in range(8)}
    return mu, sd


def standardized_pool(Y, lab, mu, sd):
    Z = np.empty_like(Y)
    for c in range(8):
        m = lab == c; s = sd[c].copy(); s[s <= 0] = 1.0
        Z[m] = (Y[m] - mu[c]) / s
    return Z


def surrogate(Zpool, counts, mu, sd_target, rng):
    """Every class redrawn from one standardized pool, re-standardized, given its mean and its target per-neuron SDs."""
    rows = Zpool[rng.permutation(len(Zpool))]; out, lab, i = [], [], 0
    for c in range(8):
        Zc = rows[i:i + counts[c]].copy(); i += counts[c]
        if i > len(rows):
            raise ValueError("pool smaller than requested counts")
        Zc -= Zc.mean(0); s = Zc.std(0); s[s <= 0] = 1.0; Zc /= s
        out.append(mu[c] + Zc * sd_target[c])
        lab.append(np.full(counts[c], c))
    return np.concatenate(out).astype(np.float32), np.concatenate(lab)


def mean_variance_law(Y, lab, h):
    """(iii) sigma_ic = a mu_ic^b fitted on half h == 0 (log SD on log mean over neuron-class pairs with positive mean and SD);
    returns a, b and the per-class SDs it predicts from the full data's class means."""
    A = h == 0
    mu = np.concatenate([Y[A & (lab == c)].mean(0) for c in range(8)]); sd = np.concatenate([Y[A & (lab == c)].std(0) for c in range(8)])
    ok = (mu > 0) & (sd > 0)
    b, la = np.polyfit(np.log(mu[ok]), np.log(sd[ok]), 1)
    full_mu = {c: Y[lab == c].mean(0) for c in range(8)}
    pred = {c: np.exp(la) * np.maximum(full_mu[c], np.exp(np.log(mu[ok]).min())) ** b for c in range(8)}
    r = float(np.corrcoef(np.log(mu[ok]), np.log(sd[ok]))[0, 1])
    return float(np.exp(la)), float(b), r, pred


def tuned_gain_law(Y, lab, h):
    """(iv) run57's gain on the TUNED response: sigma_ic^2 = v_i + s^2 t_ic^2, t_ic = m_ic - mu0_i (mu0 = class-averaged mean),
    v_i the neuron's class-independent variance; one s^2 per recording fitted on half A by least squares on the class
    deviations sigma_ic^2 - mean_c sigma_ic^2 = s^2 (t_ic^2 - mean_c t_ic^2). Returns s2, r of that fit, per-class SDs
    (v_i and t from the full data; v_i floored at 0)."""
    A = h == 0
    m = np.stack([Y[A & (lab == c)].mean(0) for c in range(8)]); v = np.stack([Y[A & (lab == c)].var(0) for c in range(8)])
    t2 = (m - m.mean(0)) ** 2
    dv, dt = (v - v.mean(0)).ravel(), (t2 - t2.mean(0)).ravel()
    s2 = float((dv * dt).sum() / (dt * dt).sum()); r = float(np.corrcoef(dv, dt)[0, 1])
    mf = np.stack([Y[lab == c].mean(0) for c in range(8)]); vf = np.stack([Y[lab == c].var(0) for c in range(8)])
    tf2 = (mf - mf.mean(0)) ** 2
    vi = np.maximum(vf.mean(0) - s2 * tf2.mean(0), 0)
    return s2, r, {c: np.sqrt(vi + s2 * tf2[c]) for c in range(8)}


def fano_law(Y, lab, h):
    """(v) variance linear in the class mean (Poisson-like): sigma_ic^2 = v_i + s^2 m_ic, one s^2 per recording fitted on half
    A from the class deviations (sigma_ic^2 - mean_c sigma_ic^2 = s^2 (m_ic - mean_c m_ic)); v_i and m from the full data
    (variance floored at 0). Keeps the direction preference m carries, which t^2 discards."""
    A = h == 0
    m = np.stack([Y[A & (lab == c)].mean(0) for c in range(8)]); v = np.stack([Y[A & (lab == c)].var(0) for c in range(8)])
    dv, dm = (v - v.mean(0)).ravel(), (m - m.mean(0)).ravel()
    s2 = float((dv * dm).sum() / (dm * dm).sum()); r = float(np.corrcoef(dv, dm)[0, 1])
    mf = np.stack([Y[lab == c].mean(0) for c in range(8)]); vf = np.stack([Y[lab == c].var(0) for c in range(8)])
    vi = vf.mean(0) - s2 * mf.mean(0)
    return s2, r, {c: np.sqrt(np.maximum(vi + s2 * mf[c], 0)) for c in range(8)}


def order_within(ow, om, Y, lab):
    """Within-class order effect (paired - sequential) on the EIGHT 45-degree classes (4 rungs), averaged over 8 origins,
    exactly from the overlap identity (fluctuation_overlap_profile.oe_from with within_curve; floors of equal-size arms cancel).
    Eight classes, not sixteen: a surrogate redraws whole 45-degree classes, so it has no 22.5-degree sub-structure."""
    G = (Y @ Y.T).astype(np.float64); idx = [np.where(lab == c)[0] for c in range(8)]
    tr, _, T = om.overlaps(G, idx); counts = np.array([len(i) for i in idx], float); out = []
    for o in range(8):
        P = [[(j + o) % 8 for k in range(r + 1) for j in (k, k + 4)] for r in range(4)]
        S = [[(j + o) % 8 for k in range(r + 1) for j in (2 * k, 2 * k + 1)] for r in range(4)]
        sl = [om.slope(np.log([counts[r].sum() for r in rr]), om.within_curve(tr, T, rr, counts)) for rr in (P, S)]
        out.append(sl[0] - sl[1])
    return float(np.mean(out))


def ladder(ow, Y, lab):
    m = [np.where(lab == b)[0] for b in range(8)]
    return ow.ladder((Y @ Y.T).astype(np.float64), lab, [np.concatenate(m[:c]) for c in RUNGS], np.random.default_rng(42))


def checks(S, lab, mu, sd, Rpool_sub, sub):
    m2, s2 = class_moments(S, lab)
    dmu = max(float(np.abs(m2[c] - mu[c]).max()) for c in range(8))
    ok = [sd[c] > 0 for c in range(8)]
    dsd = max(float(np.abs(s2[c][ok[c]] / sd[c][ok[c]] - 1).max()) for c in range(8))
    Zs = np.concatenate([(S[lab == c][:, sub] - m2[c][sub]) / np.where(s2[c][sub] > 0, s2[c][sub], 1) for c in range(8)])
    R2 = np.corrcoef(Zs.T); iu = np.triu_indices(len(sub), 1)
    return {"max_abs_mean_diff": dmu, "max_rel_sd_err": dsd, "r_offdiag_corr": float(np.corrcoef(R2[iu], Rpool_sub[iu])[0, 1])}


def arms(ow, Y, lab, counts, rng_sub, sd_iii=None, order=None, sd_iv=None, sd_v=None):
    """Surrogates (i), (ii) and, when sd_iii is given, (iii) built from (Y, lab), sized to `counts`; ladders.
    order = om adds the 8-class within-class order effect of seed 7's surrogate."""
    mu, sd = class_moments(Y, lab); Zp = standardized_pool(Y, lab, mu, sd)
    sd_pool = np.sqrt(np.mean([sd[c] ** 2 for c in range(8)], 0))
    targets = {"i": sd, "ii": {c: np.sqrt((sd[c] ** 2).sum() / (sd_pool ** 2).sum()) * sd_pool for c in range(8)}}
    if sd_iii is not None:
        targets["iii"] = sd_iii
    if sd_iv is not None:
        targets["iv"] = sd_iv
    if sd_v is not None:
        targets["v"] = sd_v
    sub = np.sort(rng_sub.choice(Y.shape[1], min(1000, Y.shape[1]), replace=False))
    Rp = np.corrcoef(Zp[:, sub].T)
    res = {}
    for mode, tgt in targets.items():
        lads, chk = [], []
        for s in SEEDS:
            S, l = surrogate(Zp, counts, mu, tgt, np.random.default_rng(s))
            lads.append(ladder(ow, S, l))
            if s == SEEDS[0]:
                chk = checks(S, l, mu, tgt, Rp, sub)
                if order is not None:
                    chk["order_within"] = order_within(ow, order, S, l)
        res[mode] = {"ladders": lads, "checks": chk, **{k: float(np.mean([q[k] for q in lads])) for k in lads[0]}}
    return res


def run_one(fp):
    ow = _load("ow", "cr_order_and_withinbin_controls.py"); om = _load("om", "overlap_identity_checks.py")
    t0 = time.time()
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel(); del dat
    lab = ow.bins(phi, 8); sub48 = om.bins(phi, 48)
    Y = ow.remove_withinbin(X, lab, phi, 6, 8)
    counts = np.bincount(lab, minlength=8)
    row = {"name": fp.stem, "counts": counts.tolist(), "resid": ladder(ow, Y, lab)}
    row["resid"]["order_within"] = order_within(ow, om, Y, lab)
    a, b, rlaw, sd_iii = mean_variance_law(Y, lab, om.halves_by_sub(sub48))
    row["mean_variance_law"] = {"a": a, "b": b, "r_log": rlaw, "fit_on": "half A (alternate presentations of each 7.5-degree sub-bin)"}
    s2, r4, sd_iv = tuned_gain_law(Y, lab, om.halves_by_sub(sub48))
    row["tuned_gain_law"] = {"s2": s2, "r_fit": r4, "fit_on": "half A"}
    s2v, r5, sd_v = fano_law(Y, lab, om.halves_by_sub(sub48))
    row["fano_law"] = {"s2": s2v, "r_fit": r5, "fit_on": "half A"}
    row["full"] = arms(ow, Y, lab, counts, np.random.default_rng(0), sd_iii, om, sd_iv, sd_v)
    # balanced: first n_min presentations of each class in time
    nmin = int(counts.min()); keep = np.sort(np.concatenate([np.where(lab == c)[0][:nmin] for c in range(8)]))
    Yb, lb = Y[keep], lab[keep]
    row["balanced"] = {"n_per_class": nmin, "resid": ladder(ow, Yb, lb), **arms(ow, Yb, lb, np.full(8, nmin), np.random.default_rng(0))}
    # held out
    h = om.halves_by_sub(sub48); ho = []
    for a, b in ((0, 1), (1, 0)):
        A, B = h == a, h == b
        cb = np.bincount(lab[B], minlength=8)
        ho.append({"target_resid": ladder(ow, Y[B], lab[B]), **arms(ow, Y[A], lab[A], np.minimum(cb, np.bincount(lab[A], minlength=8)), np.random.default_rng(0))})
    row["heldout"] = ho
    f = lambda d, m, k: d[m][k]
    print(f"{fp.stem[9:30]:22s} resid d {row['resid']['delta']:+.3f} w {row['resid']['within']:+.3f} | (i) d {f(row['full'], 'i', 'delta'):+.3f} "
          f"w {f(row['full'], 'i', 'within'):+.3f} | (ii) d {f(row['full'], 'ii', 'delta'):+.3f} w {f(row['full'], 'ii', 'within'):+.3f} | "
          f"(iii) d {f(row['full'], 'iii', 'delta'):+.3f} w {f(row['full'], 'iii', 'within'):+.3f} [b {b:.2f}] | bal resid w {row['balanced']['resid']['within']:+.3f} (i) w {f(row['balanced'], 'i', 'within'):+.3f} (ii) w {f(row['balanced'], 'ii', 'within'):+.3f} | "
          f"checks {row['full']['i']['checks']} ({time.time() - t0:.0f}s) DONE", flush=True)
    return row


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(run_one, sorted(RAW.glob("gratings*.npy"))))
    pool = {r["name"]: np.mean([q["within"] for q in r["resid_pool"]]) for r in json.load(open(HERE.parent / "data_canonical" / "withinclass_shuffle_identification.json"))["part_a"]}
    for r in rows:
        r["c8_resid_pool_within"] = float(pool[r["name"]])
    v = {"i_within_between_pool_and_resid": int(sum(min(r["c8_resid_pool_within"], r["resid"]["within"]) < r["full"]["i"]["within"] < max(r["c8_resid_pool_within"], r["resid"]["within"]) for r in rows)),
         "i_reproduces_delta_and_within_005": int(sum(abs(r["full"]["i"]["delta"] - r["resid"]["delta"]) <= 0.05 and abs(r["full"]["i"]["within"] - r["resid"]["within"]) <= 0.05 for r in rows)),
         "ii_within_abs_le_005": int(sum(abs(r["full"]["ii"]["within"]) <= 0.05 for r in rows)),
         "i_within_share_of_resid": [round(r["full"]["i"]["within"] / r["resid"]["within"], 2) for r in rows],
         "ii_within": [round(r["full"]["ii"]["within"], 3) for r in rows],
         "v_reproduces_i_within_005": int(sum(abs(r["full"]["v"]["within"] - r["full"]["i"]["within"]) <= 0.05 for r in rows)),
         "v_reproduces_i_order_005": int(sum(abs(r["full"]["v"]["checks"]["order_within"] - r["full"]["i"]["checks"]["order_within"]) <= 0.05 for r in rows)),
         "iv_reproduces_i_order_005": int(sum(abs(r["full"]["iv"]["checks"]["order_within"] - r["full"]["i"]["checks"]["order_within"]) <= 0.05 for r in rows)),
         "iv_reproduces_i_within_005": int(sum(abs(r["full"]["iv"]["within"] - r["full"]["i"]["within"]) <= 0.05 for r in rows)),
         "iv_within_share_of_i": [round(r["full"]["iv"]["within"] / r["full"]["i"]["within"], 2) for r in rows],
         "iii_reproduces_i_within_005": int(sum(abs(r["full"]["iii"]["within"] - r["full"]["i"]["within"]) <= 0.05 for r in rows)),
         "iii_within_share_of_i": [round(r["full"]["iii"]["within"] / r["full"]["i"]["within"], 2) for r in rows],
         "mean_variance_b": [round(r["mean_variance_law"]["b"], 3) for r in rows],
         "order_within_resid_i_ii_iii": [[round(r["resid"]["order_within"], 3)] + [round(r["full"][m]["checks"]["order_within"], 3) for m in ("i", "ii", "iii", "iv", "v")] for r in rows],
         "balanced_ii_within": [round(r["balanced"]["ii"]["within"], 3) for r in rows],
         "heldout_i_minus_target_within": [round(np.mean([h["i"]["within"] - h["target_resid"]["within"] for h in r["heldout"]]), 3) for r in rows]}
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
