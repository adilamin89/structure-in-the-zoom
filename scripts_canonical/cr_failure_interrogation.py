"""Camera-ready failure interrogation (Oct 20: every failed expectation gets its cause, its correct version and its residual).
Expectations written before the run. Inputs: cr_overlap_mining*.json, cr_order_effect_origins.json, cr_confession_tests.json.

I1 WHY THE ORDER EFFECT DEPENDS ON THE ORIGIN (E2 fail; d1 fail on Table 9). The stationary kernel gives the origin
   average exactly (d5); hypothesis: the origin term is set by the starting classes' own fluctuation concentration K_cc,
   which varies with direction because the code is anisotropic. Per recording, the within-class order effect OE_w(o) at
   the 16 cyclic origins from the exact identity; its departure from the origin mean against the first-rung diagonal
   contrast D(o) = [K(o+8,o+8) - K(o+1,o+1)] / 2 (paired partner minus sequential partner; a more concentrated partner
   lowers that arm's first-rung PR and steepens its climb). Expect: Spearman(OE_w(o) - mean, D(o)) >= +0.6 pooled over
   the 8 recordings; the measured TOTAL order effect at each origin (cr_order_effect_origins) tracks OE_w(o) + the
   class-mean part's own origin term with r >= 0.8 (exact by the split; reported); K_cc across the 16 classes carries a
   cardinal-locked even harmonic (l = 2 or 4 the largest non-constant harmonic) in >= 6 of 8.
I2 SPLIT-HALF ORDER EFFECT (d2: 5/8 within 0.02). Twenty random half-splits within each class give the sampling SD of
   the half-data within-class order effect. Expect |A - B| <= 2 sqrt(2) SD in >= 7 of 8.
I3 WHY THE WITHIN-BIN SHARE IS SUB-QUADRATIC IN BIN WIDTH (b1). Tuning saturates inside a bin when the bin approaches
   the tuning width, so sharper tuning gives a smaller exponent. Expect Spearman(exponent 22.5 -> 45, median
   orientation selectivity of tuned neurons, run59) <= -0.6.
I4 WHY REMOVING TUNING BARELY MOVED GT1's WITHIN PART (a2). From a4 (eight origins, ladder arms): the within part is the
   declared arm minus the floor arm. Expect: in GT1 the floor arm's within slope falls by >= 70% of the declared arm's
   fall; in GT2 by < 50%.
I5 VON MISES WITH SEEDS (T1 within part non-monotone in kappa, one draw each). Ten seeds per kappa; tuning share from
   cross-half sub-bin means (unbiased; the leave-one-out residual inflated noise by 1/(n_b - 1) ~ 11% at 10 stimuli
   per sub-bin and made the old share negative). Expect: tuning-removed within part |mean| <= 0.02 at every kappa;
   raw within-part mean rises with kappa (2 -> 16) and the class-mean part's magnitude falls.
I6 WHICH NEURONS CARRY THE ANISOTROPY (c'2/c'3 fail, c4 pass). Cross-fit: preferences on one half, class means on the
   other. Contrast R(7,0) + R(3,4) - R(1,2) - R(5,6) after dropping tuned neurons preferring orientations near 0
   (bin 0) against dropping the same number of random tuned neurons (5 draws). Expect: dropping the near-0 group lowers
   the contrast by >= 50% in >= 5 of the 7 positive-contrast recordings, random drops by <= 20%.
I7 THE HARMONIC PREDICTION, MADE ON THE CLASS MEANS ALONE (GT3: class-mean part favours adjacency at 16/16 origins, although
   the most even code). The 'class-mean part' is the increment log PR(W + B) - log PR(W); the harmonic prediction is about
   B alone. OE of log PR(B_A) (declared arms, sixteen classes), origin-averaged. Expect: positive (pairing) in all five
   even-led recordings (b/c > 1), GT3 included; then GT3's negative increment is the W-B overlap term.
I8 SIZE OF THE COMPETITION (why the recordings differ, and by how much). Per recording, origin averages over the 16
   origins: fluctuation side = within-class order effect; class-mean side = increment part; their kernel contrasts
   dK = Kn(22.5) - Kn(180) (Kn = K_cd / sqrt(K_cc K_dd), the shape overlap of two classes' fluctuations) and
   dC = C(22.5) - C(180) of the sixteen class means. Expect: Spearman(within side, dK) <= -0.8 (more adjacent-sharing,
   more negative); Spearman(class-mean side, B-alone order effect of I7) >= +0.6; the localized recordings have the
   three largest dK.
Out: data_canonical/cr_failure_interrogation.json"""
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
DC = HERE.parent / "data_canonical"


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod


M = load("m", "cr_overlap_mining.py"); CO = load("co", "cr_order_and_withinbin_controls.py"); FU = load("fu", "cr_overlap_mining_followup.py")
KS = load("ks", "kappa_sweep.py")
OUT = DC / "cr_failure_interrogation.json"


def paths(o):
    P = [[(j + o) % 16 for k in range(r + 1) for j in (k, k + 8)] for r in range(8)]
    S = [[(j + o) % 16 for k in range(r + 1) for j in (2 * k, 2 * k + 1)] for r in range(8)]
    return P, S


def oe_from(curve_fn, counts):
    out = []
    for o in range(16):
        P, S = paths(o); sl = []
        for rungs in (P, S):
            x = np.log([counts[r].sum() for r in rungs]); sl.append(M.slope(x, curve_fn(rungs)))
        out.append(sl[0] - sl[1])
    return np.array(out)


def per_recording(fp, origins_total):
    dat = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(dat["istim"], float).ravel() % (2 * np.pi); del dat
    G = (X @ X.T).astype(np.float64)
    lab16 = M.bins(phi, 16); idx = [np.where(lab16 == c)[0] for c in range(16)]
    counts = np.array([len(i) for i in idx], float)
    tr, _, T = M.overlaps(G, idx); K = M.K_of(tr, tr, T); kd = np.diag(K)
    oe_w = oe_from(lambda rungs: M.within_curve(tr, T, rungs, counts), counts)
    D = np.array([(kd[(o + 8) % 16] - kd[(o + 1) % 16]) / 2 for o in range(16)])
    harm = np.abs(np.fft.rfft(kd - kd.mean()))
    sep16 = np.minimum(np.abs(np.subtract.outer(np.arange(16), np.arange(16))), 16 - np.abs(np.subtract.outer(np.arange(16), np.arange(16))))
    Kn = K / np.sqrt(np.outer(kd, kd)); Cm = np.corrcoef(np.stack([X[i].mean(0) for i in idx]))
    kprof = [float(Kn[sep16 == s].mean()) for s in range(9)]; cprof = [float(Cm[sep16 == s].mean()) for s in range(9)]
    # I7: class means alone
    mu = np.stack([X[i].mean(0) for i in idx])

    def b_curve(rungs):
        out = []
        for A in rungs:
            p = counts[A] / counts[A].sum(); mA = (p[:, None] * mu[A]).sum(0); Z = np.sqrt(p)[:, None] * (mu[A] - mA)
            g = Z @ Z.T; out.append(np.log(np.trace(g) ** 2 / (g * g).sum()))
        return np.array(out)
    oe_b = oe_from(b_curve, counts)
    # I2: twenty random half-splits
    rng = np.random.default_rng(7); halves = []
    for _ in range(20):
        ia, ib = [], []
        for I in idx:
            pm = rng.permutation(I); ia.append(np.sort(pm[: len(I) // 2])); ib.append(np.sort(pm[len(I) // 2:]))
        for part in (ia, ib):
            c = np.array([len(i) for i in part], float); t, _, TT = M.overlaps(G, part)
            P, S = paths(0); sl = []
            for rungs in (P, S):
                sl.append(M.slope(np.log([c[r].sum() for r in rungs]), M.within_curve(t, TT, rungs, c)))
            halves.append(sl[0] - sl[1])
    # I6: anisotropy carriers
    lab8 = M.bins(phi, 8); lab24 = M.bins(phi, 24); sub48 = M.bins(phi, 48); h = M.halves_by_sub(sub48)
    th24 = np.radians(7.5 + 15 * np.arange(24)); i6 = []
    rr = np.random.default_rng(3)
    for sh, mh in ((0, 1), (1, 0)):
        Xs, l8 = X[h == sh], lab8[h == sh]; n = len(l8)
        M8 = np.stack([Xs[l8 == k].mean(0) for k in range(8)]); grand = Xs.mean(0); cnt = np.bincount(l8, minlength=8)
        ssb = sum(cnt[k] * (M8[k] - grand) ** 2 for k in range(8)); ssw = sum(((Xs[l8 == k] - M8[k]) ** 2).sum(0) for k in range(8))
        tuned = (1 - stats.f.cdf((ssb / 7) / (ssw / (n - 8) + 1e-12), 7, n - 8)) < 0.01
        M24 = np.stack([Xs[lab24[h == sh] == k].mean(0) for k in range(24)]); R = M24 - M24.min(0, keepdims=True)
        ori = (np.angle((R * np.exp(2j * th24[:, None])).sum(0)) / 2) % np.pi
        ob = np.full(X.shape[1], -1); ob[tuned] = np.round(ori[tuned] / (np.pi / 8)).astype(int) % 8
        Xm, lm = X[h == mh], lab8[h == mh]; Mm = np.stack([Xm[lm == k].mean(0) for k in range(8)])

        def contrast(keep):
            C = np.corrcoef(Mm[:, keep]); sep = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
            cc = np.zeros_like(C)
            for s in range(5):
                cc[sep == s] = C[sep == s].mean()
            Rr = C - cc; return float(Rr[7, 0] + Rr[3, 4] - Rr[1, 2] - Rr[5, 6])
        allk = np.ones(X.shape[1], bool); base = contrast(allk)
        g0 = ob == 0; drop0 = contrast(~g0)
        tid = np.where(tuned)[0]; rnd = []
        for _ in range(5):
            k = allk.copy(); k[rr.choice(tid, g0.sum(), replace=False)] = False; rnd.append(contrast(k))
        i6.append({"base": base, "drop_near0": drop0, "drop_random_mean": float(np.mean(rnd)), "n_near0": int(g0.sum())})
    tot = np.array([x["delta"] for x in origins_total]); mean_part = np.array([x["class_means"] for x in origins_total])
    w_meas = np.array([x["within"] for x in origins_total])
    return {"Kn_profile": kprof, "C16_profile": cprof, "dK": kprof[1] - kprof[8], "dC": cprof[1] - cprof[8],
            "origin_mean_within": float(w_meas.mean()), "origin_mean_means": float(mean_part.mean()), "origin_mean_total": float(tot.mean()),
            "oe_within_identity": oe_w.tolist(), "D_first_rung_diag": D.tolist(), "Kcc": kd.tolist(),
            "Kcc_harmonics": harm.tolist(), "oe_B_alone": oe_b.tolist(), "oe_B_alone_mean": float(oe_b.mean()),
            "split_half_values": halves, "split_half_sd": float(np.std(halves, ddof=1)),
            "total_vs_within_r": float(np.corrcoef(tot, w_meas)[0, 1]), "total_vs_means_r": float(np.corrcoef(tot, mean_part)[0, 1]),
            "i6": i6}


def i5():
    out = []
    for kappa in KS.KAPPAS:
        vals = []
        for seed in range(10):
            resp, prefs, thetas = KS.make_population(kappa, seed=1000 * seed + int(kappa * 10))
            X = np.ascontiguousarray(resp.T.astype(np.float64)); X /= X.std()
            phi2 = 2 * thetas; lab = CO.bins(phi2, 8); sub = CO.bins(phi2, 48)
            sets = [np.concatenate([np.where(lab == b)[0] for b in range(c)]) for c in (1, 2, 3, 4, 6, 8)]
            Y = CO.remove_withinbin(X, lab, phi2, 6, 8)
            raw = FU.ladder_arms(X @ X.T, lab, sets, np.random.default_rng(seed)); rem = FU.ladder_arms(Y @ Y.T, lab, sets, np.random.default_rng(seed))
            hh = M.halves_by_sub(sub); sh = []
            for c in range(8):
                m = lab == c; Xa, Xb = X[m & (hh == 0)], X[m & (hh == 1)]; sa, sb = sub[m & (hh == 0)], sub[m & (hh == 1)]
                acc = sum(((Xa[sa == b].mean(0) - Xa.mean(0)) * (Xb[sb == b].mean(0) - Xb.mean(0))).sum() * (sub[m] == b).sum()
                          for b in np.intersect1d(np.unique(sa), np.unique(sb)))
                Xc = X[m]; sh.append(acc / m.sum() / (((Xc - Xc.mean(0)) ** 2).sum() / m.sum()))
            vals.append([raw["delta"], raw["within"], raw["class_means"], rem["delta"], rem["within"], float(np.mean(sh))])
        v = np.array(vals); mu, sd = v.mean(0), v.std(0, ddof=1)
        keys = ["delta", "within", "class_means", "removed_delta", "removed_within", "share"]
        out.append({"kappa": kappa, **{k: [float(mu[i]), float(sd[i])] for i, k in enumerate(keys)}})
        print(f"I5 kappa {kappa:4.0f}: " + " ".join(f"{k} {mu[i]:+.3f}+-{sd[i]:.3f}" for i, k in enumerate(keys)), flush=True)
    return out


def main():
    orig = {r["name"]: r["origins"] for r in json.load(open(DC / "cr_order_effect_origins.json"))["rows"]}
    first = {r["name"]: r for r in json.load(open(DC / "cr_overlap_mining.json"))["rows"]}
    fol = {r["name"]: r for r in json.load(open(DC / "cr_overlap_mining_followup.json"))["rows"]}
    run59 = {v["name"]: v for v in json.load(open(DC / "run59_shift_by_direction_selectivity.json"))["rows"].values()}
    mp = {r["name"]: r for r in json.load(open(DC / "multipole_harmonics_8dir.json"))["rows"]}
    rows = []
    for fp in sorted(M.RAW.glob("gratings*.npy")):
        r = per_recording(fp, orig[fp.stem]); r["name"] = fp.stem
        b = first[fp.stem]["b"]; r["share_exponent_22_45"] = float(np.log2(b["share_45"] / b["share_22"]))
        r["median_osi"] = run59[fp.stem]["median_osi_tuned"]; r["b_over_c"] = mp[fp.stem]["b2_over_c1"]
        d2 = first[fp.stem]["d"]["d2"]["0->1"]; r["d2_gap"] = abs(d2["pred"] - d2["meas"])
        if "a4" in fol[fp.stem]:
            a4 = fol[fp.stem]["a4"]
            r["i4"] = {k: float(np.mean([o["noise"][k] - o["raw"][k] for o in a4])) for k in ("decl_within", "floor_within", "within")}
        dev = np.array(r["oe_within_identity"]) - np.mean(r["oe_within_identity"])
        r["i1_spearman"] = float(stats.spearmanr(dev, r["D_first_rung_diag"])[0])
        print(f"{fp.stem[9:30]:22s} I1 rho {r['i1_spearman']:+.2f} Kcc harm(l=1..4) " + " ".join(f"{x:.4f}" for x in r["Kcc_harmonics"][1:5])
              + f" tot~w r {r['total_vs_within_r']:+.2f} tot~m r {r['total_vs_means_r']:+.2f} | I2 gap {r['d2_gap']:.3f} sd {r['split_half_sd']:.3f}"
              f" | I7 B-alone OE {r['oe_B_alone_mean']:+.3f} (b/c {r['b_over_c']:.2f}) | I6 " + " ".join(
                  f"{x['base']:+.3f}->{x['drop_near0']:+.3f}/{x['drop_random_mean']:+.3f}" for x in r["i6"])
              + f" | I8 dK {r['dK']:+.3f} dC {r['dC']:+.3f} w {r['origin_mean_within']:+.3f} m {r['origin_mean_means']:+.3f}"
              + (f" | I4 decl {r['i4']['decl_within']:+.3f} floor {r['i4']['floor_within']:+.3f}" if "i4" in r else ""), flush=True)
        rows.append(r)
        json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    dev_all = np.concatenate([np.array(r["oe_within_identity"]) - np.mean(r["oe_within_identity"]) for r in rows])
    D_all = np.concatenate([r["D_first_rung_diag"] for r in rows])
    summ = {"I1_pooled_spearman": float(stats.spearmanr(dev_all, D_all)[0]),
            "I3_spearman": float(stats.spearmanr([r["share_exponent_22_45"] for r in rows], [r["median_osi"] for r in rows])[0]),
            "I8_within_vs_dK": float(stats.spearmanr([r["origin_mean_within"] for r in rows], [r["dK"] for r in rows])[0]),
            "I8_means_vs_Balone": float(stats.spearmanr([r["origin_mean_means"] for r in rows], [r["oe_B_alone_mean"] for r in rows])[0])}
    print("SUMMARY", json.dumps(summ), flush=True)
    res = {"rows": rows, "summary": summ, "I5": i5()}
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
