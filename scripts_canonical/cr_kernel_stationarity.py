"""Referee check (ChatGPT referee, Oct 20, comment 8): Sec 4 averages the class-mean correlation over class pairs at each
separation, C(dphi), assuming the full 8 x 8 class-pair correlation depends only on the separation. Per grating recording:
the 8 x 8 correlation across neurons of the class-mean vectors (eight 45-degree classes, as multipole_harmonics_8dir), its
circulant projection (each entry replaced by the mean at its circular separation), and the share of the off-diagonal variance
the projection explains (1 - residual SS / total SS about the off-diagonal mean). No prediction registered: a descriptive check.
Out: data_canonical/cr_kernel_stationarity.json"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE.parent / "raw"   # the Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
OUT = HERE.parent / "data_canonical" / "cr_kernel_stationarity.json"


def main():
    rows = []
    sep = np.minimum(np.abs(np.subtract.outer(np.arange(8), np.arange(8))), 8 - np.abs(np.subtract.outer(np.arange(8), np.arange(8))))
    off = ~np.eye(8, dtype=bool)
    for fp in sorted(RAW.glob("gratings*.npy")):
        d = np.load(fp, allow_pickle=True).item()
        X = np.asarray(d["sresp"], np.float32)                                   # neurons x presentations
        lab = np.clip(np.digitize(np.asarray(d["istim"], float) % (2 * np.pi), np.linspace(0, 2 * np.pi, 9)) - 1, 0, 7)
        M = np.stack([X[:, lab == k].mean(1) for k in range(8)])                 # 8 x neurons
        C = np.corrcoef(M)
        circ = np.zeros_like(C)
        for s in range(5):
            circ[sep == s] = C[sep == s].mean()
        r = C[off] - circ[off]; t = C[off] - C[off].mean()
        share = float(1 - (r ** 2).sum() / (t ** 2).sum())
        rows.append({"name": fp.stem, "circulant_share": share, "max_abs_residual": float(np.abs(r).max())})
        print(f"{fp.stem[:32]:32s} separation-average explains {100 * share:.1f}% of off-diagonal variance; max |residual| {np.abs(r).max():.3f}", flush=True)
    json.dump({"rows": rows}, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
