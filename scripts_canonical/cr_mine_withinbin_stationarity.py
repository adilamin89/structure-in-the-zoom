"""Mining two referee-control outcomes (C5, Oct 20), per grating recording.

1. WITHIN-BIN TUNING SHARE. The 45-degree direction classes pool continuous directions. Share of each class's within-class
   variance (trace) carried by its 7.5-degree sub-bin means, averaged over classes; and the same share on two disjoint halves
   of the presentations (alternate trials within each sub-bin), its reliable part = the cross-half covariance of sub-bin
   means (noise-free estimate of tuning variance inside a bin). Compared with how much of delta the within-bin removal took
   (cr_order_and_withinbin_controls.json).
2. KERNEL STATIONARITY vs NOISE. The separation average explains 79-96% of the off-diagonal variance of the 8 x 8 class-mean
   correlation (cr_kernel_stationarity.py). Noise ceiling: correlate the 8 x 8 off-diagonal entries between two disjoint
   halves of the presentations (split-half reliability r_hh). If the circulant share approaches r_hh-limited reproducibility,
   the departure from stationarity is estimation noise; if not, it is structure (e.g. cardinal over-representation).
   Also: cardinal minus oblique class-mean norm (mouse V1 oblique effect) per recording.
Out: data_canonical/cr_mine_withinbin_stationarity.json"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the Stringer et al. 2019 grating recordings (figshare), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_mine_withinbin_stationarity.json"
SEP = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
OFF = ~np.eye(8, dtype=bool)


def circ_share(C):
    circ = np.zeros_like(C)
    for s in range(5):
        circ[SEP == s] = C[SEP == s].mean()
    r = C[OFF] - circ[OFF]; t = C[OFF] - C[OFF].mean()
    return float(1 - (r ** 2).sum() / (t ** 2).sum())


def main():
    rows = []
    for fp in sorted(RAW.glob("gratings*.npy")):
        d = np.load(fp, allow_pickle=True).item()
        X = np.asarray(d["sresp"], np.float32).T; X /= X.std() + 1e-9          # presentations x neurons
        phi = np.asarray(d["istim"], float) % (2 * np.pi)
        lab = np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
        sub = np.clip(np.digitize(phi, np.linspace(0, 2 * np.pi, 49)) - 1, 0, 47)  # 7.5-degree sub-bins
        share, rel_share = [], []
        half = np.zeros(len(lab), int)
        for b in range(48):
            idx = np.where(sub == b)[0]; half[idx[1::2]] = 1
        for c in range(8):
            m = lab == c; Xc = X[m]; Wc = ((Xc - Xc.mean(0)) ** 2).sum() / m.sum()
            sb = sub[m]; mu = Xc.mean(0)
            tun = sum(((Xc[sb == b].mean(0) - mu) ** 2).sum() * (sb == b).sum() for b in np.unique(sb)) / m.sum()
            share.append(tun / Wc)
            # reliable tuning variance: cross-half covariance of sub-bin means about the class mean
            hc = half[m]; acc = 0.0; nb = 0
            for b in np.unique(sb):
                a = Xc[(sb == b) & (hc == 0)]; e = Xc[(sb == b) & (hc == 1)]
                if len(a) and len(e):
                    acc += ((a.mean(0) - mu) * (e.mean(0) - mu)).sum() * (sb == b).sum(); nb += 1
            rel_share.append(acc / m.sum() / Wc)
        M = np.stack([X[lab == k].mean(0) for k in range(8)]); C = np.corrcoef(M)
        Ch = [np.corrcoef(np.stack([X[(lab == k) & (half == h)].mean(0) for k in range(8)])) for h in (0, 1)]
        r_hh = float(np.corrcoef(Ch[0][OFF], Ch[1][OFF])[0, 1])
        norms = np.linalg.norm(M - M.mean(0), axis=1)
        card = float(norms[[0, 2, 4, 6]].mean() / norms[[1, 3, 5, 7]].mean())      # bins starting at 0,90,180,270 deg vs 45,...
        row = {"name": fp.stem, "withinbin_share": float(np.mean(share)), "withinbin_share_reliable": float(np.mean(rel_share)),
               "circulant_share": circ_share(C), "splithalf_r_offdiag": r_hh,
               "circulant_share_halves": [circ_share(c) for c in Ch], "cardinal_over_oblique_norm": card}
        rows.append(row)
        print(f"{fp.stem[:30]:30s} within-bin share {row['withinbin_share']:.3f} (reliable {row['withinbin_share_reliable']:.3f}) | "
              f"circulant {row['circulant_share']:.3f}, halves {row['circulant_share_halves'][0]:.3f}/{row['circulant_share_halves'][1]:.3f}, "
              f"split-half r {r_hh:.3f} | cardinal/oblique {card:.3f}", flush=True)
    json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
