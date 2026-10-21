"""Camera-ready tests behind three appendix sentences that report an exception without its physics (Oct 20). Expectations written
before the run.

T1 VON MISES SIGN (App G: "the sign of its ungrained orientation-class shift depends on the tuning width, negative at
   kappa = 2 and 8, positive by 16"). The population of kappa_sweep.py has the same isotropic noise in every class, so its
   only class-specific within-class covariance is tuning inside each 22.5-degree orientation bin. Exact split (App K) at
   kappa = 2, 4, 8, 16, raw and with tuning inside each bin removed (held-out 3.75-degree sub-bin means, as App K).
   Expect: within-bin share and within part rise with kappa; the sign flips where the within part overtakes the class
   means; with the tuning removed the within part is within 0.02 of zero and delta negative at every kappa.
T2 ANALYTIC FLOOR (App K / App L: +0.254 against +0.237, "we do not separate the two"). chun_comparison.py maps every rung
   of both arms through the isotropic inversion PR -> PR (n-1) / (n-1-PR) and still subtracts the empirical floor. So the
   difference of the two shifts is exactly the slope of -log(1 - PR/(n-1)) on the declared rungs minus that on the floor
   rungs. Expect: that term reproduces the difference (algebra) and comes from the first rungs, where the declared ladder
   sits at lower PR than its floor.
T3 GT2 IN GRAINING (App G: "whether b/c is above or below one sorts seven of the eight, and GT2 is the exception";
   GT2 keeps 0.31 of its shift under orientation-sorted blocks at K = 32 although even-led, b/c = 1.6). Mining (a) found
   GT2's shift rests most on tuning inside each 45-degree bin (+0.292 -> +0.032 when removed). Hypothesis: orientation-
   sorted blocks pool opposite directions and average away that direction-specific tuning. Drifting GT1, GT2, GT3,
   orientation-sorted blocks (run51 construction) at K = 1 and 32, raw and with within-bin tuning removed.
   Expect: in GT2 the orientation-sorted K = 32 shift on raw data is within 0.05 of the tuning-removed shift at K = 1,
   and on tuning-removed data orientation sorting keeps >= 0.8 of the shift; GT1 and GT3 keep >= 0.8 either way.
Out: data_canonical/cr_confession_tests.json"""
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname); mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod); return mod



CO = load("co", "cr_order_and_withinbin_controls.py")
FU = load("fu", "cr_overlap_mining_followup.py")
KS = load("ks", "kappa_sweep.py")
OUT = HERE.parent / "data_canonical" / "cr_confession_tests.json"


def t1():
    out = []
    for kappa in KS.KAPPAS:
        resp, prefs, thetas = KS.make_population(kappa, seed=int(kappa * 10))
        X = np.ascontiguousarray(resp.T.astype(np.float64)); X /= X.std()
        phi2 = 2 * thetas; lab = CO.bins(phi2, 8); sets = [np.concatenate([np.where(lab == b)[0] for b in range(c)]) for c in (1, 2, 3, 4, 6, 8)]
        Y = CO.remove_withinbin(X, lab, phi2, 6, 8)
        raw = FU.ladder_arms(X @ X.T, lab, sets, np.random.default_rng(42)); rem = FU.ladder_arms(Y @ Y.T, lab, sets, np.random.default_rng(42))
        share = np.mean([1 - ((Y[lab == c] - Y[lab == c].mean(0)) ** 2).sum() / ((X[lab == c] - X[lab == c].mean(0)) ** 2).sum()
                         for c in range(8)])                          # tuning's share of each class's within-class trace
        out.append({"kappa": kappa, "raw": raw, "tuning_removed": rem, "withinbin_share": float(share)})
        print(f"T1 kappa {kappa:4.0f}: share {share:.3f} | raw delta {raw['delta']:+.3f} within {raw['within']:+.3f} means {raw['class_means']:+.3f}"
              f" | removed delta {rem['delta']:+.3f} within {rem['within']:+.3f} means {rem['class_means']:+.3f}", flush=True)
    return out


def t2():
    dat = np.load(CO.RAW / "gratings_drifting_GT3_2019_04_05_1.npy", allow_pickle=True).item()
    X = np.asarray(dat["sresp"], np.float32); X /= X.std() + 1e-9; Xt = np.ascontiguousarray(X.T)
    bl = np.clip(np.digitize(np.asarray(dat["istim"], float), np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
    members = [np.where(bl == b)[0] for b in range(8)]
    pr = lambda Z: (lambda G: np.trace(G) ** 2 / (G * G).sum())(((Z - Z.mean(0)) @ (Z - Z.mean(0)).T).astype(np.float64))
    rng = np.random.default_rng(42); sizes, prd, prf = [], [], []
    for c in (1, 2, 3, 4, 6, 8):
        sel = np.concatenate(members[:c]); sizes.append(len(sel)); prd.append(pr(Xt[sel]))
        prf.append(np.exp(np.mean([np.log(pr(Xt[rng.choice(len(Xt), len(sel), replace=False)])) for _ in range(10)])))
    n = np.array(sizes, float); prd, prf = np.array(prd), np.array(prf); x = np.log(n)
    s = lambda y: float(np.polyfit(x, y, 1)[0])
    d_emp = s(np.log(prd)) - s(np.log(prf))
    inv = lambda p: p * (n - 1) / (n - 1 - p)
    d_an = s(np.log(inv(prd))) - s(np.log(inv(prf)))
    term = s(-np.log(1 - prd / (n - 1))) - s(-np.log(1 - prf / (n - 1)))
    out = {"sizes": sizes, "pr_declared": prd.tolist(), "pr_floor": prf.tolist(), "delta_empirical": d_emp, "delta_analytic": d_an,
           "difference": d_an - d_emp, "term_slope": term,
           "term_per_rung": (-np.log(1 - prd / (n - 1)) + np.log(1 - prf / (n - 1))).tolist()}
    print(f"T2 GT3: delta emp {d_emp:+.3f} analytic {d_an:+.3f} diff {d_an - d_emp:+.4f} term {term:+.4f}; per rung "
          + " ".join(f"{v:+.4f}" for v in out["term_per_rung"]) + "; PR decl/floor " + " ".join(f"{a:.0f}/{b:.0f}" for a, b in zip(prd, prf)), flush=True)
    return out


def t3():
    out = []
    for stem in ("gratings_drifting_GT1_2019_04_12_1", "gratings_drifting_GT2_2019_04_05_1", "gratings_drifting_GT3_2019_04_05_1"):
        dat = np.load(CO.RAW / f"{stem}.npy", allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        phi = np.asarray(dat["istim"], float).ravel(); del dat
        lab = CO.bins(phi, 8); sets = [np.concatenate([np.where(lab == b)[0] for b in range(c)]) for c in (1, 2, 3, 4, 6, 8)]
        ang = 2 * np.pi * np.arange(8) / 8
        M = np.stack([X[lab == k].mean(0) for k in range(8)]); Mc = M - M.mean(0)
        order = np.argsort((np.angle((Mc * np.exp(2j * ang[:, None])).sum(0)) / 2) % np.pi)
        Y = CO.remove_withinbin(X, lab, phi, 6, 8)
        row = {"name": stem}
        for tag, Z in (("raw", X), ("tuning_removed", Y)):
            for K in (1, 32):
                Zb = Z if K == 1 else np.stack([Z[:, order[i * K:(i + 1) * K]].mean(1) for i in range(len(order) // K)], 1)
                row[f"{tag}_K{K}"] = FU.ladder_arms((Zb @ Zb.T).astype(np.float64), lab, sets, np.random.default_rng(42))["delta"]
        row["retention_raw"] = row["raw_K32"] / row["raw_K1"]; row["retention_removed"] = row["tuning_removed_K32"] / row["tuning_removed_K1"]
        out.append(row)
        print(f"T3 {stem[9:24]}: raw K1 {row['raw_K1']:+.3f} K32 {row['raw_K32']:+.3f} (kept {row['retention_raw']:.2f}) | tuning removed"
              f" K1 {row['tuning_removed_K1']:+.3f} K32 {row['tuning_removed_K32']:+.3f} (kept {row['retention_removed']:.2f})", flush=True)
    return out


def main():
    res = {"T1": t1(), "T2": t2(), "T3": t3()}
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
