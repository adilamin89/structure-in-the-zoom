"""Camera-ready mining follow-up (Oct 20), after cr_overlap_mining.py. Expectations written before this run.

Verdicts of the first run (data_canonical/cr_overlap_mining.json): a1 FAIL (noise-only sigma 0.212 / 0.211 / 0.287 for
drifting GT1 / GT2 / GT3: GT2's fluctuations are not the most shared); a2 raised in all three, most in GT2, least in GT3;
a3 pass (Tr B / Tr W 0.264 / 0.280 / 0.158); b1 FAIL (share(22.5)/share(45) = 0.36-0.50: the share grows as Delta^1.0-1.5,
not Delta^2); b2 pass (r = 0.998, linear prediction 19-34% high); b3 pass; c INVALID estimator (the orientation phase of
eight 45-degree classes takes four values, which sit on the 22.5-degree histogram edges: chi^2 and densities were
quantization); d1 FAIL (sign 6/8, Spearman 0.62); d2 5/8 within 0.02 (sampling-limited where |order effect| <= 0.05).

(a4) ORIGIN. Is GT2's drop under within-bin removal (-0.259) a property of its covariance or of the declared origin (the
     ladder starts at the 0-45 degree class)? Direction ladder (rungs 1,2,3,4,6,8 classes) from each of the eight cyclic
     origins, raw and with 7.5-degree within-bin tuning removed (as cr_order_and_withinbin_controls), ten-draw floors.
     Expectation: the drop is a covariance property: origin-mean drop in GT2 >= 2x those of GT1 and GT3.
     Reported per arm: declared and floor slopes of the within and class-mean terms.
(c')  ANISOTROPY, lattice-safe estimator: preferences from 24 classes of 15 degrees on one half (tuned: F-test p < 0.01
     on eight classes, same half), orientation histogram in 8 bins CENTRED on 0, 22.5, ... degrees (lattice points of
     sharp neurons, 7.5 + 15k, never fall on an edge); class-mean correlations of eight classes on the other half; both ways.
     c'1: orientation histogram non-uniform, chi^2 p < 0.001 in >= 6 of 8.
     c'2: sign[R(7,0) + R(3,4) - R(1,2) - R(5,6)] = sign[density(bin 0) - density(bin 4)] in >= 7 of 8.
     c'3: reweighting tuned neurons to a uniform orientation density raises the circulant share in >= 6 of 8 (mean of
          both directions).
     c4 (alternative): neurons preferring orientations near 0 (bin 0) are more broadly tuned (lower orientation vector
          strength) than those near 90 (bin 4) in >= 6 of the 7 recordings whose contrast is positive.
(d4) WHAT IN K CARRIES THE ORDER EFFECT (sixteen classes, full data, declared arms, exact identity):
     variants of K: full; diagonal replaced by its mean; off-diagonal replaced by its separation average (diagonal kept);
     alpha made equal (K kept).
(d5) ORIGIN AVERAGE: the stationary kernel (d1) predicts the order effect averaged over the sixteen cyclic origins:
     within 0.02 in >= 6 of 8, Pearson >= 0.9 (the measured Table 9 value is one origin).
Out: data_canonical/cr_overlap_mining_followup.json"""
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("m", HERE / "cr_overlap_mining.py"); M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
spec = importlib.util.spec_from_file_location("c", HERE / "cr_order_and_withinbin_controls.py"); CO = importlib.util.module_from_spec(spec); spec.loader.exec_module(CO)
RAW, OUT = M.RAW, HERE.parent / "data_canonical" / "cr_overlap_mining_followup.json"
TWO_PI = 2 * np.pi


def ladder_arms(G, lab, rung_sets, rng):
    """As CO.ladder, also returning the declared and floor slopes of each term."""
    sizes, lad, flo = [], [], []
    for sel in rung_sets:
        sizes.append(len(sel)); lad.append(CO.parts_from_gram(G[np.ix_(sel, sel)], lab[sel]))
        acc = np.zeros(2)
        for _ in range(CO.N_NULL):
            idx = rng.choice(len(G), len(sel), replace=False)
            acc += np.array(CO.parts_from_gram(G[np.ix_(idx, idx)], lab[idx])) / CO.N_NULL
        flo.append(acc)
    x = np.log(sizes); L = np.array(lad); F = np.array(flo)
    s = lambda y: float(np.polyfit(x, y, 1)[0])
    out = {"decl_total": s(L[:, 0]), "decl_within": s(L[:, 1]), "floor_total": s(F[:, 0]), "floor_within": s(F[:, 1])}
    out["delta"] = out["decl_total"] - out["floor_total"]; out["within"] = out["decl_within"] - out["floor_within"]
    out["class_means"] = out["delta"] - out["within"]
    return out


def part_a4(X, phi, G):
    lab8 = M.bins(phi, 8); m8 = [np.where(lab8 == b)[0] for b in range(8)]
    Y = CO.remove_withinbin(X, lab8, phi, 6, 8); GY = (Y @ Y.T).astype(np.float64)
    res = []
    for o in range(8):
        sets = [np.concatenate([m8[(o + j) % 8] for j in range(c)]) for c in (1, 2, 3, 4, 6, 8)]
        res.append({"origin": o, "raw": ladder_arms(G, lab8, sets, np.random.default_rng(42)),
                    "noise": ladder_arms(GY, lab8, sets, np.random.default_rng(42))})
    return res


def part_c2(X, phi):
    lab8 = M.bins(phi, 8); lab24 = M.bins(phi, 24); sub48 = M.bins(phi, 48); h = M.halves_by_sub(sub48)
    th24 = np.radians(7.5 + 15 * np.arange(24))
    sep = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
    off = ~np.eye(8, dtype=bool)

    def circ(C):
        cc = np.zeros_like(C)
        for s in range(5):
            cc[sep == s] = C[sep == s].mean()
        r = C[off] - cc[off]; t = C[off] - C[off].mean()
        return float(1 - (r ** 2).sum() / (t ** 2).sum()), C - cc

    def wcorr(Mm, w):
        Mc = Mm - (Mm * w).sum(1, keepdims=True) / w.sum()
        cov = (Mc * w) @ Mc.T; d = np.sqrt(np.diag(cov)); return cov / np.outer(d, d)

    out = []
    for sh, mh in ((0, 1), (1, 0)):
        Xs, l8 = X[h == sh], lab8[h == sh]; l24 = lab24[h == sh]; n = len(l8)
        M8 = np.stack([Xs[l8 == k].mean(0) for k in range(8)]); grand = Xs.mean(0); cnt = np.bincount(l8, minlength=8)
        ssb = sum(cnt[k] * (M8[k] - grand) ** 2 for k in range(8)); ssw = sum(((Xs[l8 == k] - M8[k]) ** 2).sum(0) for k in range(8))
        tuned = (1 - stats.f.cdf((ssb / 7) / (ssw / (n - 8) + 1e-12), 7, n - 8)) < 0.01
        M24 = np.stack([Xs[l24 == k].mean(0) for k in range(24)]); R = M24 - M24.min(0, keepdims=True)
        z2 = (R * np.exp(2j * th24[:, None])).sum(0)
        ori = (np.angle(z2) / 2) % np.pi; osi = np.abs(z2) / (R.sum(0) + 1e-12)
        ob = np.round(ori[tuned] / (np.pi / 8)).astype(int) % 8
        hist = np.bincount(ob, minlength=8); dens = hist / hist.sum() * 8
        osi_t = osi[tuned]
        Xm, lm = X[h == mh], lab8[h == mh]
        Mm = np.stack([Xm[lm == k].mean(0) for k in range(8)])
        s0, Rr = circ(np.corrcoef(Mm))
        w = np.ones(X.shape[1]); w[tuned] = 1.0 / dens[ob]
        sw, _ = circ(wcorr(Mm, w))
        out.append({"ori_hist": hist.tolist(), "chi2_p": float(stats.chisquare(hist).pvalue),
                    "dens0_minus_dens4": float((hist[0] - hist[4]) / hist.sum()),
                    "contrast": float(Rr[7, 0] + Rr[3, 4] - Rr[1, 2] - Rr[5, 6]),
                    "circ": s0, "circ_reweighted": sw,
                    "osi_bin0": float(osi_t[ob == 0].mean()), "osi_bin4": float(osi_t[ob == 4].mean())})
    return out


def part_d45(X, phi, G):
    lab16 = M.bins(phi, 16); idx = [np.where(lab16 == c)[0] for c in range(16)]
    counts = np.array([len(i) for i in idx], float)
    tr, _, T = M.overlaps(G, idx); K = M.K_of(tr, tr, T)
    sep = np.minimum(np.abs(np.subtract.outer(np.arange(16), np.arange(16))), 16 - np.abs(np.subtract.outer(np.arange(16), np.arange(16))))

    def oe(trv, Tm, o=0):
        sl = {}
        for nm, rungs in (("P", [[(j + o) % 16 for k in range(r + 1) for j in (k, k + 8)] for r in range(8)]),
                          ("S", [[(j + o) % 16 for k in range(r + 1) for j in (2 * k, 2 * k + 1)] for r in range(8)])):
            x = np.log([counts[r].sum() for r in rungs]); sl[nm] = M.slope(x, M.within_curve(trv, Tm, rungs, counts))
        return sl["P"] - sl["S"]

    Kdiag = K.copy(); np.fill_diagonal(Kdiag, np.diag(K).mean())
    Koff = K.copy()
    for s in range(1, 9):
        Koff[sep == s] = K[sep == s].mean()
    Kc = Koff.copy(); np.fill_diagonal(Kc, np.diag(K).mean())
    ones = np.ones(16)
    variants = {"full": oe(tr, T), "full_K_equal_alpha": oe(ones, K), "diag_mean": oe(ones, Kdiag),
                "offdiag_sep_avg": oe(ones, Koff), "stationary": oe(ones, Kc)}
    variants["origin_mean_full"] = float(np.mean([oe(tr, T, o) for o in range(16)]))
    variants["origin_sd_full"] = float(np.std([oe(tr, T, o) for o in range(16)]))
    return variants


def main():
    rows = []
    for fp in sorted(RAW.glob("gratings*.npy")):
        t0 = time.time()
        dat = np.load(fp, allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        phi = np.asarray(dat["istim"], float).ravel() % TWO_PI; del dat
        G = (X @ X.T).astype(np.float64)
        row = {"name": fp.stem, "c": part_c2(X, phi), "d": part_d45(X, phi, G)}
        if "drifting" in fp.stem:
            row["a4"] = part_a4(X, phi, G)
            drops = [r["noise"]["delta"] - r["raw"]["delta"] for r in row["a4"]]
            print(f"{fp.stem[9:24]} a4 drop per origin " + " ".join(f"{d:+.3f}" for d in drops) + f" mean {np.mean(drops):+.3f}", flush=True)
        c = row["c"]; d = row["d"]
        print(f"{fp.stem[9:30]:22s} c': chi2p {c[0]['chi2_p']:.1e}/{c[1]['chi2_p']:.1e} d0-d4 {c[0]['dens0_minus_dens4']:+.3f}"
              f" contrast {c[0]['contrast']:+.3f} circ {np.mean([x['circ'] for x in c]):.3f}->{np.mean([x['circ_reweighted'] for x in c]):.3f}"
              f" osi0/4 {c[0]['osi_bin0']:.3f}/{c[0]['osi_bin4']:.3f} | d: " + " ".join(f"{k} {v:+.3f}" for k, v in d.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
        rows.append(row)
        json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
