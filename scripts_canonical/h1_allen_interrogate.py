"""H1-R19 (P6 interrogation) and H1-R20 (P5 interrogation), Allen VISp, after the interim misses (Oct 22).

R19 the DS thirds (P6: 13/21 and 12/21 against 24).
  (a) power: the imaging effect at Allen's design. On each grating recording, draws of Allen's design: n_tuned tuned neurons
      (tuned = F p < 0.01 on all trials; n_tuned = the Allen VISp median tuned count) and 75 presentations per 45-degree class;
      thirds by first-harmonic vector strength, cross-fitted on alternate trials (cr_ds_thirds_crossfit design, ten-draw
      floors); 20 draws per recording -> the probability that most > least, and the expected pass count out of 21.
  (b) estimator: Allen DS index computed per temporal frequency (class means within each TF, vector strength, averaged over
      the 5 TFs) instead of on TF-pooled means; thirds cross-fitted as P6; measured on the TF-residualized other half.
  (c) the across-population version is P9 (h1_allen_score.py).
R20 the held-out sigma law (P5: r 0.46 VISp).
  (a) sampling: sigma (noise mode) on 20 random half-splits per population -> its sampling SD, and the reliability of sigma
      across VISp, 1 - mean sampling variance / observed variance; the same for the within part (split-half ladders);
      disattenuated r = r / sqrt(rel_sigma rel_within).
  (b) frame: sigma and the within part in the within-SD frame (each unit scaled by its pooled within-class SD); same-data and
      held-out r.
  (c) n / d_w: per population d_w = p m / (m - 1 - p) from each class's PR at m presentations (Eq. 2 inverted), averaged;
      the ratio m / d_w against App B's threshold (the within part is sampling-limited below ~1).
Run: VECLIB_MAXIMUM_THREADS=1 python -X faulthandler h1_allen_interrogate.py   Out: data_canonical/h1_allen_interrogate.json"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h1_allen_engine as E  # noqa: E402

RAW = HERE.parent / "raw"
OUT = HERE.parent / "data_canonical" / "h1_allen_interrogate.json"
dsx = E.dsx


def ds_index_per_tf(X, lab, tfl):
    v = []
    for t in np.unique(tfl):
        m = tfl == t
        M = np.stack([X[m & (lab == k)].mean(0) for k in range(8)]); R = M - M.min(0, keepdims=True) + 1e-9
        v.append(np.abs((R * np.exp(1j * 2 * np.pi * np.arange(8)[:, None] / 8)).sum(0) / R.sum(0)))
    return np.mean(v, 0)


def tuned_mask(X, lab):
    n = len(lab); counts = np.bincount(lab, minlength=8)
    M = np.stack([X[lab == k].mean(0) for k in range(8)]); g = X.mean(0)
    ssb = sum(counts[k] * (M[k] - g) ** 2 for k in range(8)); ssw = sum(((X[lab == k] - M[k]) ** 2).sum(0) for k in range(8))
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ssb / 7) / (ssw / (n - 8))
    return np.nan_to_num(1 - stats.f.cdf(F, 7, n - 8), nan=1.0) < 0.01


def thirds_per_tf(X, lab, tfl, cell):
    h = E.halves(cell); out = {}
    for sh, mh in ((0, 1), (1, 0)):
        S = h == sh; tun = np.where(tuned_mask(X[S], lab[S]))[0]
        dsi = ds_index_per_tf(X[S], lab[S], tfl[S]); nsub = len(tun) // 3
        if nsub < 20:
            return None
        order = tun[np.argsort(dsi[tun])]; ds, nds = order[-nsub:], order[:nsub]
        Mm = h == mh; Ym = E.tf_residualize(X[Mm], lab[Mm], cell[Mm])
        out[f"sel{sh}_meas{mh}"] = {"DS": E.ladder_delta_only(np.ascontiguousarray(Ym[:, ds]), lab[Mm]),
                                    "nonDS": E.ladder_delta_only(np.ascontiguousarray(Ym[:, nds]), lab[Mm]), "n_sub": int(nsub)}
    return out


def sigma_splits(X, lab, cell, frame_scale=None, n_split=20):
    """sigma (noise mode) and the tfres within part on random half-splits (alternation replaced by random halves per cell)."""
    sig, wit = [], []
    Xs = X if frame_scale is None else X / frame_scale
    for s in range(n_split):
        rng = np.random.default_rng(100 + s); h = np.zeros(len(cell), int)
        for b in np.unique(cell):
            idx = np.where(cell == b)[0]; h[rng.permutation(idx)[: len(idx) // 2]] = 1
        Ra = E.om.residuals(Xs[h == 0], lab[h == 0], cell[h == 0], "noise"); Rb = E.om.residuals(Xs[h == 1], lab[h == 1], cell[h == 1], "noise")
        sig.append(E.om.sigma(E.om.split_half_K(Ra, Rb)))
        wit.append(E.ladder(E.tf_residualize(Xs[h == 0], lab[h == 0], cell[h == 0]), lab[h == 0])["within"])
    return np.array(sig), np.array(wit)


def d_w(Y, lab):
    out = []
    for c in range(8):
        Z = Y[lab == c] - Y[lab == c].mean(0); m = len(Z); G = Z @ Z.T
        p = np.trace(G) ** 2 / (G * G).sum(); out.append(p * m / max(m - 1 - p, 1e-9))
    return float(np.mean(out)), float(np.mean(np.bincount(lab)))


def power_sim(n_tuned, n_per_class=75, draws=20):
    ow = E.ow; res = []
    for fp in sorted(RAW.glob("gratings*.npy")):
        dat = np.load(fp, allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        lab = ow.bins(np.asarray(dat["istim"], float).ravel(), 8); del dat
        tun_all = np.where(tuned_mask(X, lab))[0]; wins = 0; gaps = []
        for d in range(draws):
            rng = np.random.default_rng(d)
            rows = np.sort(np.concatenate([rng.choice(np.where(lab == c)[0], n_per_class, replace=False) for c in range(8)]))
            cols = np.sort(rng.choice(tun_all, min(n_tuned, len(tun_all)), replace=False))
            Xd, ld = np.ascontiguousarray(X[np.ix_(rows, cols)], dtype=np.float64), lab[rows]
            h = np.zeros(len(ld), int)
            for c in range(8):
                idx = np.where(ld == c)[0]; h[idx[1::2]] = 1
            g = []
            for sh, mh in ((0, 1), (1, 0)):
                ds, nds = dsx.select(Xd[h == sh], ld[h == sh])
                if len(ds) < 3:
                    continue
                Xm, lm = Xd[h == mh], ld[h == mh]
                g.append(E.ladder_delta_only(np.ascontiguousarray(Xm[:, ds]), lm) - E.ladder_delta_only(np.ascontiguousarray(Xm[:, nds]), lm))
            if g:
                gaps.append(float(np.mean(g))); wins += int(g[0] > 0)
        res.append({"name": fp.stem, "p_DS_gt_nonDS": wins / draws, "gap_mean": float(np.mean(gaps)), "gap_sd": float(np.std(gaps))})
        print(f"power {fp.stem[9:30]:22s} P(DS>nonDS) {wins / draws:.2f} gap {np.mean(gaps):+.3f}+-{np.std(gaps):.3f}", flush=True)
    return res


def main():
    t0 = time.time()
    sids = [s["session"] for s in json.load(open(HERE.parent / "data_canonical" / "allen_expansion_all_sessions.json"))["results"]]
    rows, ntun = [], []
    for sid in sids:
        S = E.load_session(sid); idx = E.populations(S).get("VISp")
        if idx is None:
            continue
        P = E.pop_view(S, "VISp", idx)   # H5-R1 masking path
        X, lab, tfl, cell = P["X"], P["lab"], P["tf"], P["cell"]
        Y = E.tf_residualize(X, lab, cell)
        sd = np.sqrt(np.mean([((Y[lab == c] - Y[lab == c].mean(0)) ** 2).mean(0) for c in range(8)], 0))
        keep = sd > 1e-6 * np.median(sd[sd > 0])
        r = {"session": sid, "N": int(X.shape[1]), "thirds_per_tf": thirds_per_tf(X, lab, tfl, cell)}
        sg, wt = sigma_splits(X, lab, cell)
        sgz, wtz = sigma_splits(X[:, keep], lab, cell, frame_scale=sd[keep])
        r.update(sigma_split_mean=float(sg.mean()), sigma_split_sd=float(sg.std()), within_split_mean=float(wt.mean()), within_split_sd=float(wt.std()),
                 sigma_z_mean=float(sgz.mean()), sigma_z_sd=float(sgz.std()), within_z_split_mean=float(wtz.mean()), within_z_split_sd=float(wtz.std()))
        r["within_z_full"] = E.ladder(Y[:, keep] / sd[keep], lab)["within"]
        r["d_w"], r["m_per_class"] = d_w(Y, lab); r["m_over_dw"] = r["m_per_class"] / r["d_w"]
        ntun.append(int(tuned_mask(X, lab).sum()))
        rows.append(r)
        print(f"{sid} N {r['N']} sigma {sg.mean():.3f}+-{sg.std():.3f} within {wt.mean():+.3f}+-{wt.std():.3f} | z sigma {sgz.mean():.3f} within "
              f"{wtz.mean():+.3f} | m/d_w {r['m_over_dw']:.2f} ({time.time() - t0:.0f}s)", flush=True)
    mf = lambda s: np.log(8 / (1 + 7 * np.asarray(s))) / np.log(8)
    sig = np.array([r["sigma_split_mean"] for r in rows]); wit = np.array([r["within_split_mean"] for r in rows])
    rel_s = 1 - np.mean([r["sigma_split_sd"] ** 2 for r in rows]) / sig.var()
    rel_w = 1 - np.mean([r["within_split_sd"] ** 2 for r in rows]) / wit.var()
    r_obs = float(stats.pearsonr(mf(sig), wit)[0])
    sz = np.array([r["sigma_z_mean"] for r in rows]); wz = np.array([r["within_z_split_mean"] for r in rows])
    el = [r for r in rows if r["thirds_per_tf"]]
    v = {"R19b_per_tf_sel0_meas1": f"{sum(r['thirds_per_tf']['sel0_meas1']['DS'] > r['thirds_per_tf']['sel0_meas1']['nonDS'] for r in el)}/{len(el)}",
         "R19b_per_tf_sel1_meas0": f"{sum(r['thirds_per_tf']['sel1_meas0']['DS'] > r['thirds_per_tf']['sel1_meas0']['nonDS'] for r in el)}/{len(el)}",
         "R19b_mean_gap": float(np.mean([np.mean([r['thirds_per_tf'][k]['DS'] - r['thirds_per_tf'][k]['nonDS'] for k in ('sel0_meas1', 'sel1_meas0')]) for r in el])) if el else None,
         "allen_visp_median_tuned": int(np.median(ntun)),
         "R20a_sigma_reliability": float(rel_s), "R20a_within_reliability": float(rel_w), "R20a_r_half_split": r_obs,
         "R20a_r_disattenuated": float(r_obs / np.sqrt(max(rel_s, 1e-9) * max(rel_w, 1e-9))) if rel_s > 0 and rel_w > 0 else None,
         "R20b_z_r_half_split": float(stats.pearsonr(mf(sz), wz)[0]), "R20b_z_r_full": float(stats.pearsonr(mf(sz), [r["within_z_full"] for r in rows])[0]),
         "R20b_z_mae": float(np.mean(np.abs(mf(sz) - wz))),
         "R20c_m_over_dw_median": float(np.median([r["m_over_dw"] for r in rows])), "R20c_m_over_dw_range": [float(min(r["m_over_dw"] for r in rows)), float(max(r["m_over_dw"] for r in rows))]}
    print("ALLEN SIDE", json.dumps(v, indent=1), flush=True)
    pw = power_sim(v["allen_visp_median_tuned"])
    v["R19a_power"] = pw; v["R19a_expected_pass_of_21"] = float(21 * np.mean([p["p_DS_gt_nonDS"] for p in pw]))
    print("VERDICT", json.dumps({k: x for k, x in v.items() if k != "R19a_power"}, indent=1), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(OUT, "w"), indent=1)
    print("wrote", OUT, "ALL DONE", flush=True)


if __name__ == "__main__":
    main()
