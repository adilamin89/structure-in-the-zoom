"""Camera-ready mining (Oct 20): explain four existing results with the covariance-overlap identity of App K, per grating recording.
No new finding is sought; each part explains a number already in the paper. Expectations written before the run.

Identity (App K, exact on samples): for the classes A a rung holds, W_A = sum_c p_c S_c (S_c the class-centred covariance,
p_c = n_c / N) and 1 / PR(W_A) = sum_{c,d in A} alpha_c alpha_d K_cd, alpha_c = n_c Tr S_c / sum_d n_d Tr S_d,
K_cd = Tr(S_c S_d) / (Tr S_c Tr S_d). All traces come from the Gram matrix of the presentations. Sharing ratio
sigma = mean off-diagonal K / mean diagonal K (1 = every class fluctuates alike, 0 = disjoint directions), computed with
split halves (alternate presentations within each 7.5-degree sub-bin; each half's residuals built inside the half) so the
diagonal carries no 1/n sampling term.

(a) WHY GT2's SHIFT IS MOSTLY WITHIN-BIN TUNING. Removing tuning inside each 45-degree bin (cr_order_and_withinbin_controls)
    moved delta by -0.050 (GT1), -0.259 (GT2), -0.070 (GT3) with similar tuning shares of W (14%, 17%, 11%). In GT2 the
    within part fell 0.642 -> 0.497 AND the class-mean part 0.350 -> 0.464 (more negative); in GT1 the within part barely
    moved (0.578 -> 0.570). Sigma for raw W, noise-only W (tuning removed) and the tuning part alone.
    a1 (first hypothesis): GT2's noise-only fluctuations are the most shared across classes of the three
        drifting recordings, sigma_noise(GT2) > sigma_noise(GT1) and > sigma_noise(GT3).
    a1' (second expectation, from the within parts left after removal, GT3 0.29 < GT2 0.50 < GT1 0.57, and the
        approximate growth law PR(W_8)/PR(S_1) ~ 8 / (1 + 7 sigma)): sigma_noise orders GT3 > GT2 > GT1, so a1 fails on
        GT3; GT2's noise is shared more than GT1's.
    a2: removing the tuning raises sigma in all three, by the most in GT2 and the least in GT1 (tuning adds class-specific
        directions in GT2, almost none in GT1).
    a3: GT2's shift is the thinnest margin between two large parts: its class means are the strongest relative to its
        fluctuations, Tr B / Tr W at the full set the largest of the three drifting recordings, so shrinking W by removing
        tuning makes its class-mean part the most negative.
(b) THE WITHIN-BIN SHARE vs TUNING SLOPE AND BIN WIDTH. Reliable share of within-class variance carried by tuning inside a
    class of width 90, 45 and 22.5 degrees (cross-half products of 3.75-degree sub-bin means about each half's class mean),
    and the linear prediction share = mean |dmu/dphi|^2 Delta^2 / 12 / Tr S_c with the slope from cross-half central
    differences of 7.5-degree sub-bin means.
    b1: share(22.5) / share(45) in [0.2, 0.4] (about 1/4) in >= 7 of 8; share(90) / share(45) in [2, 4] (curvature).
    b2: predicted vs measured share at 45 degrees: Pearson r >= 0.9 across the 8 recordings, ratio in [0.6, 1.4] in >= 6.
    b3: the three full-field drifting recordings have the three largest slope^2 / noise.
(c) THE CLASS-PAIR ANISOTROPY (separation average explains 79-96%; departures reproducible, split-half r >= 0.99). The
    largest residual sits at an adjacent pair straddling a cardinal drift direction in 6 of 8 recordings (seen before
    registering, descriptive). Tuned neurons (F-test p < 0.01 on eight classes); preferred orientation = half the phase of
    the second harmonic of their eight class means, preferred direction = phase of the first. Cross-fit: preferences
    from one half, class means from the other, both ways.
    c1: the preferred-orientation histogram (8 bins of 22.5 degrees) departs from uniform (chi^2 p < 0.001) in >= 6 of 8.
    c2: sign[R(7,0) + R(3,4) - R(1,2) - R(5,6)] (pairs straddling the 0/180 drift directions minus those straddling 90/270)
        equals sign[density(orientation within 22.5 deg of 0) - density(within 22.5 deg of 90)] in >= 7 of 8.
    c3: reweighting tuned neurons to a uniform preferred-orientation density raises the circulant share in >= 6 of 8.
(d) ORDER EFFECTS FROM THE MEASURED OVERLAPS (sixteen 22.5-degree classes; paired = each rung adds a class and its
    antipode, sequential = adjacent pairs; run72 design B). The declared arms' within-class curves follow from K exactly.
    d1 (stationary kernel): K replaced by its separation average (diagonal by its mean, alpha equal) predicts the sign of
        the within-class order effect (Table 9) in 8 of 8 and its rank across recordings, Spearman >= 0.8.
    d2 (held out): K and alpha from half A predict half B's declared-arm within-class order effect within 0.02 in >= 7 of
        8, both ways (equal trials per class in both halves, so the diagonal's sampling term matches).
    d3: the K-based prediction (d1) ranks the measured within-class order effect better than the principal-angle margin
        (|rho| > 0.67, App K).
Out: data_canonical/cr_overlap_mining.json"""
import json
import time
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_overlap_mining.json"
PRIOR = HERE.parent / "data_canonical" / "cr_order_and_withinbin_controls.json"
TWO_PI = 2 * np.pi


def bins(phi, nb):
    return np.clip(np.digitize(phi % TWO_PI, np.linspace(0, TWO_PI, nb + 1)) - 1, 0, nb - 1)


def centred(B):
    """Double-centred Gram block between class c (rows) and class d (cols)."""
    B = B - B.mean(1, keepdims=True)
    return B - B.mean(0, keepdims=True)


def overlaps(G, idx_rows, idx_cols=None):
    """Tr S_c, and Tr(S_c S_d) for every class pair. idx_cols (if given) = the other half's index sets (split-half)."""
    idx_cols = idx_rows if idx_cols is None else idx_cols
    C = len(idx_rows)
    tr_r = np.array([np.trace(centred(G[np.ix_(I, I)])) / len(I) for I in idx_rows])
    tr_c = np.array([np.trace(centred(G[np.ix_(I, I)])) / len(I) for I in idx_cols])
    T = np.zeros((C, C))
    for c in range(C):
        for d in range(C):
            B = centred(G[np.ix_(idx_rows[c], idx_cols[d])])
            T[c, d] = (B * B).sum() / (len(idx_rows[c]) * len(idx_cols[d]))
    return tr_r, tr_c, T


def K_of(tr_r, tr_c, T):
    return T / np.outer(tr_r, tr_c)


def sigma(K):
    off = ~np.eye(len(K), dtype=bool)
    return float(K[off].mean() / np.diag(K).mean())


def split_half_K(Xa_list, Xb_list):
    """Split-half overlap matrix from per-class residual arrays of two independent halves (both directions averaged)."""
    def tr(Z): return float((Z * Z).sum() / len(Z))
    C = len(Xa_list); K = np.zeros((C, C))
    ta = [tr(Z) for Z in Xa_list]; tb = [tr(Z) for Z in Xb_list]
    for c in range(C):
        for d in range(C):
            ab = Xa_list[c] @ Xb_list[d].T; ba = Xb_list[c] @ Xa_list[d].T
            v1 = (ab * ab).sum() / (len(Xa_list[c]) * len(Xb_list[d])) / (ta[c] * tb[d])
            v2 = (ba * ba).sum() / (len(Xb_list[c]) * len(Xa_list[d])) / (tb[c] * ta[d])
            K[c, d] = 0.5 * (v1 + v2)
    return K


def residuals(X, lab, sub, mode):
    """Per-class centred residuals. mode 'raw': x - class mean; 'noise': x - leave-one-out sub-bin mean, centred per
    class; 'tuning': leave-one-out sub-bin mean - class mean."""
    out = []
    for c in np.unique(lab):
        m = lab == c; Xc = X[m]; sc = sub[m]
        if mode == "raw":
            out.append(Xc - Xc.mean(0)); continue
        L = np.empty_like(Xc)
        for b in np.unique(sc):
            mb = sc == b; nb = mb.sum(); S = Xc[mb].sum(0)
            L[mb] = (S - Xc[mb]) / max(nb - 1, 1) if nb > 1 else Xc[mb]
        R = Xc - L if mode == "noise" else L - Xc.mean(0)
        out.append(R - R.mean(0))
    return out


def halves_by_sub(sub):
    h = np.zeros(len(sub), int)
    for b in np.unique(sub):
        idx = np.where(sub == b)[0]; h[idx[1::2]] = 1
    return h


def within_curve(Klist_tr, T, rung_classes, counts):
    """log PR(W_A) for each rung from the overlap identity. Klist_tr = Tr S_c, T = Tr(S_c S_d)."""
    out = []
    for A in rung_classes:
        A = np.array(A); w = counts[A] * Klist_tr[A]
        num = w.sum() ** 2; den = (np.outer(counts[A], counts[A]) * T[np.ix_(A, A)]).sum()
        out.append(np.log(num / den))
    return np.array(out)


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


def part_a(X, phi, lab8, sub48, G):
    idx8 = [np.where(lab8 == c)[0] for c in range(8)]
    tr, _, T = overlaps(G, idx8)
    counts = np.array([len(i) for i in idx8], float)
    # exactness check of the identity on the declared rungs
    rungs = [list(range(k)) for k in (1, 2, 3, 4, 6, 8)]
    curve = within_curve(tr, T, rungs, counts)
    direct = []
    for A in rungs:
        sel = np.concatenate([idx8[c] for c in A]); Z = np.concatenate([X[idx8[c]] - X[idx8[c]].mean(0) for c in A])
        Gz = Z @ Z.T; direct.append(np.log(np.trace(Gz) ** 2 / (Gz * Gz).sum()))
    exact_err = float(np.abs(curve - np.array(direct)).max())
    h = halves_by_sub(sub48)
    res = {}
    for mode in ("raw", "noise", "tuning"):
        Ra = residuals(X[h == 0], lab8[h == 0], sub48[h == 0], mode)
        Rb = residuals(X[h == 1], lab8[h == 1], sub48[h == 1], mode)
        K = split_half_K(Ra, Rb)
        res[mode] = {"sigma": sigma(K), "K": K.tolist(),
                     "trace_per_class": float(np.mean([(Z * Z).sum() / len(Z) for Z in Ra + Rb]))}
    # class-mean strength: Tr B / Tr W over the full set (p_c weighted)
    mu = np.stack([X[i].mean(0) for i in idx8]); gm = (counts[:, None] * mu).sum(0) / counts.sum()
    trB = float((counts * ((mu - gm) ** 2).sum(1)).sum() / counts.sum()); trW = float((counts * tr).sum() / counts.sum())
    res["trB_over_trW"] = trB / trW
    res["trB_over_trW_noise"] = trB / (trW * res["noise"]["trace_per_class"] / res["raw"]["trace_per_class"])
    res["identity_max_abs_err"] = exact_err
    return res


def part_b(X, phi, lab8):
    h = None; out = {}
    sub96 = bins(phi, 96); h = halves_by_sub(sub96)
    for ncls in (4, 8, 16):
        lab = bins(phi, ncls); shares = []
        for c in range(ncls):
            m = lab == c; Xa = X[m & (h == 0)]; Xb = X[m & (h == 1)]
            ma, mb = Xa.mean(0), Xb.mean(0)
            sa, sb = sub96[m & (h == 0)], sub96[m & (h == 1)]
            acc, ntot = 0.0, 0
            for b in np.intersect1d(np.unique(sa), np.unique(sb)):
                nb = (sub96[m] == b).sum()
                acc += float(((Xa[sa == b].mean(0) - ma) * (Xb[sb == b].mean(0) - mb)).sum()) * nb; ntot += nb
            Xc = X[m]; trW = float(((Xc - Xc.mean(0)) ** 2).sum() / m.sum())
            shares.append(acc / ntot / trW)
        out[f"share_{360 // ncls}"] = float(np.mean(shares))
    # linear prediction at 45 degrees: slope^2 from cross-half central differences of 7.5-degree sub-bin means
    sub48 = bins(phi, 48); h48 = halves_by_sub(sub48)
    Ma = np.stack([X[(sub48 == b) & (h48 == 0)].mean(0) for b in range(48)])
    Mb = np.stack([X[(sub48 == b) & (h48 == 1)].mean(0) for b in range(48)])
    step = np.radians(7.5)
    Da = (np.roll(Ma, -1, 0) - np.roll(Ma, 1, 0)) / (2 * step); Db = (np.roll(Mb, -1, 0) - np.roll(Mb, 1, 0)) / (2 * step)
    slope2 = (Da * Db).sum(1)                                            # per sub-bin, unbiased for |central difference|^2
    lab8 = bins(phi, 8); preds, noise = [], []
    for c in range(8):
        sb = [b for b in range(48) if b // 6 == c]
        Xc = X[lab8 == c]; trW = float(((Xc - Xc.mean(0)) ** 2).sum() / len(Xc))
        tun = float(slope2[sb].mean()) * np.radians(45.0) ** 2 / 12
        preds.append(tun / trW); noise.append(trW)
    out["pred_share_45"] = float(np.mean(preds))
    out["slope2_over_noise"] = float(np.mean(slope2) / np.mean(noise))
    return out


def part_c(X, phi, lab8, sub48):
    h = halves_by_sub(sub48)
    th = np.radians(22.5 + 45 * np.arange(8))
    sep = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
    off = ~np.eye(8, dtype=bool)

    def circ_share(C):
        circ = np.zeros_like(C)
        for s in range(5):
            circ[sep == s] = C[sep == s].mean()
        r = C[off] - circ[off]; t = C[off] - C[off].mean()
        return float(1 - (r ** 2).sum() / (t ** 2).sum()), C - circ

    def wcorr(M, w):
        Mc = M - (M * w).sum(1, keepdims=True) / w.sum()
        cov = (Mc * w) @ Mc.T; d = np.sqrt(np.diag(cov)); return cov / np.outer(d, d)

    out = {"halves": []}
    for sel_h, meas_h in ((0, 1), (1, 0)):
        Xs, ls = X[h == sel_h], lab8[h == sel_h]; n = len(ls)
        M = np.stack([Xs[ls == k].mean(0) for k in range(8)]); grand = Xs.mean(0); cnt = np.bincount(ls, minlength=8)
        ssb = sum(cnt[k] * (M[k] - grand) ** 2 for k in range(8)); ssw = sum(((Xs[ls == k] - M[k]) ** 2).sum(0) for k in range(8))
        tuned = (1 - stats.f.cdf((ssb / 7) / (ssw / (n - 8) + 1e-12), 7, n - 8)) < 0.01
        R = M - M.min(0, keepdims=True)
        ori = (np.angle((R * np.exp(2j * th[:, None])).sum(0)) / 2) % np.pi            # preferred orientation, [0, pi)
        dirn = np.angle((R * np.exp(1j * th[:, None])).sum(0)) % TWO_PI
        o_bin = np.clip((ori[tuned] / (np.pi / 8)).astype(int), 0, 7)                     # 22.5-deg bins from 0
        o_hist = np.bincount(o_bin, minlength=8)
        d_hist = np.bincount(np.clip((dirn[tuned] / (TWO_PI / 8)).astype(int), 0, 7), minlength=8)
        # density near 0 vs near 90 deg orientation (within 22.5 deg): orientation bins 0 and 7 vs 3 and 4
        ori_t = ori[tuned]; near0 = np.mean((ori_t < np.radians(22.5)) | (ori_t >= np.radians(157.5)))
        near90 = np.mean(np.abs(ori_t - np.pi / 2) < np.radians(22.5))
        Xm, lm = X[h == meas_h], lab8[h == meas_h]
        Mm = np.stack([Xm[lm == k].mean(0) for k in range(8)])
        share0, Rres = circ_share(np.corrcoef(Mm))
        w = np.ones(X.shape[1]); dens = o_hist / o_hist.sum() * 8
        w[tuned] = 1.0 / dens[o_bin]
        share_w, Rw = circ_share(wcorr(Mm, w))
        out["halves"].append({
            "n_tuned": int(tuned.sum()), "ori_hist": o_hist.tolist(), "dir_hist": d_hist.tolist(),
            "ori_chi2_p": float(stats.chisquare(o_hist).pvalue), "dir_chi2_p": float(stats.chisquare(d_hist).pvalue),
            "near0_minus_near90": float(near0 - near90),
            "cardinal_pair_contrast": float(Rres[7, 0] + Rres[3, 4] - Rres[1, 2] - Rres[5, 6]),
            "circ_share": share0, "circ_share_reweighted": share_w,
            "max_abs_res": float(np.abs(Rres[off]).max()), "max_abs_res_reweighted": float(np.abs(Rw[off]).max())})
    return out


def part_d(X, phi, G):
    lab16 = bins(phi, 16); idx = [np.where(lab16 == c)[0] for c in range(16)]
    binsP = [[j for k in range(r + 1) for j in (k, k + 8)] for r in range(8)]
    binsS = [[j for k in range(r + 1) for j in (2 * k, 2 * k + 1)] for r in range(8)]

    def arm_slopes(tr, T, counts):
        xs = {}; out = {}
        for nm, rungs in (("P", binsP), ("S", binsS)):
            x = np.log([counts[r].sum() for r in rungs]); xs[nm] = x
            out[nm] = slope(x, within_curve(tr, T, rungs, counts))
        return out["P"] - out["S"]

    counts = np.array([len(i) for i in idx], float)
    tr, _, T = overlaps(G, idx)
    full_declared = arm_slopes(tr, T, counts)
    # d1: stationary kernel (separation average of K, mean diagonal, equal alpha)
    K = K_of(tr, tr, T); sep = np.minimum(np.abs(np.subtract.outer(np.arange(16), np.arange(16))), 16 - np.abs(np.subtract.outer(np.arange(16), np.arange(16))))
    Kc = np.zeros_like(K)
    for s in range(9):
        Kc[sep == s] = K[sep == s].mean()
    ones = np.ones(16); d1 = arm_slopes(ones, Kc, counts)                 # Tr S_c = 1, T = K: alpha_c proportional to n_c
    # d2: held-out halves (alternate presentations within each class)
    h = np.zeros(len(lab16), int)
    for I in idx:
        h[I[1::2]] = 1
    d2 = {}
    for a, b in ((0, 1), (1, 0)):
        ia = [I[h[I] == a] for I in idx]; ib = [I[h[I] == b] for I in idx]
        ca = np.array([len(i) for i in ia], float); cb = np.array([len(i) for i in ib], float)
        tra, _, Ta = overlaps(G, ia); trb, _, Tb = overlaps(G, ib)
        d2[f"{a}->{b}"] = {"pred": arm_slopes(tra, Ta, ca), "meas": arm_slopes(trb, Tb, cb)}
    return {"declared_within_OE_full": full_declared, "d1_stationary_pred": d1, "d2": d2}


def main():
    prior = {r["name"]: r for r in json.load(open(PRIOR))["rows"]}
    rows = []
    for fp in sorted(RAW.glob("gratings*.npy")):
        t0 = time.time()
        dat = np.load(fp, allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        phi = np.asarray(dat["istim"], float).ravel() % TWO_PI; del dat
        lab8, sub48 = bins(phi, 8), bins(phi, 48)
        G = (X @ X.T).astype(np.float64)
        row = {"name": fp.stem}
        row["a"] = part_a(X, phi, lab8, sub48, G)
        row["b"] = part_b(X, phi, lab8)
        row["c"] = part_c(X, phi, lab8, sub48)
        row["d"] = part_d(X, phi, G)
        p = prior[fp.stem]; row["d"]["table9_within_OE"] = p["orig_order_effect"]["within"]
        a, b, c, d = row["a"], row["b"], row["c"], row["d"]
        print(f"{fp.stem[9:30]:22s} a: sigma raw {a['raw']['sigma']:.3f} noise {a['noise']['sigma']:.3f} tuning {a['tuning']['sigma']:.3f}"
              f" B/W {a['trB_over_trW']:.3f} (id err {a['identity_max_abs_err']:.1e}) | b: share 90/45/22.5 {b['share_90']:.3f}/"
              f"{b['share_45']:.3f}/{b['share_22']:.3f} pred45 {b['pred_share_45']:.3f} | c: chi2p {c['halves'][0]['ori_chi2_p']:.1e}"
              f" contrast {c['halves'][0]['cardinal_pair_contrast']:+.3f} n0-n90 {c['halves'][0]['near0_minus_near90']:+.3f} circ"
              f" {c['halves'][0]['circ_share']:.3f}->{c['halves'][0]['circ_share_reweighted']:.3f} | d: T9 {d['table9_within_OE']:+.3f}"
              f" decl {d['declared_within_OE_full']:+.3f} stat {d['d1_stationary_pred']:+.3f} held "
              + " ".join(f"{v['pred']:+.3f}/{v['meas']:+.3f}" for v in d['d2'].values()) + f" ({time.time() - t0:.0f}s)", flush=True)
        rows.append(row)
        json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
