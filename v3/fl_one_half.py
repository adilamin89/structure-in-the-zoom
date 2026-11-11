"""The one label-blind half that missed under the Freedman-Lane null at 50 draws (Section 5 of the extended version: the early half
of a low-contrast recording, gratings_low_contrast_GT1, S2 half 0; p 3/51 unrestricted, 9/51 inside epochs in
data_canonical/meanfree_freedman_lane.json) re-read at 500 draws per null, four chunks of 125 with independent seeds, through
scripts_canonical/meanfree_freedman_lane.py's fl_half (the generator of the other 31 halves).
Needs raw/gratings_low_contrast_GT1_2019_04_09_1.npy. Out: v3/fl_one_half.json. Run: python v3/fl_one_half.py"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FP = ROOT / "raw" / "gratings_low_contrast_GT1_2019_04_09_1.npy"


def chunk(seed):
    sc = ROOT / "scripts_canonical"
    sys.path.insert(0, str(sc))
    import meanfree_freedman_lane as F
    ow = F._load("ow", sc / "cr_order_and_withinbin_controls.py"); R1 = F._load("h4r1", sc / "meanfree_shift.py")
    dat = np.load(FP, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
    y = ow.bins(np.asarray(dat["istim"], float).ravel(), 8); n = len(y); ep = np.arange(n) // 48
    sel = np.where((np.arange(n) >= n // 2).astype(int) == 0)[0]
    return F.fl_half(X[sel], y[sel], ep[sel], ow, R1, 125, seed)


if __name__ == "__main__":
    with ProcessPoolExecutor(4, mp_context=get_context("spawn")) as ex:
        parts = list(ex.map(chunk, [50_000 + 1000 * k for k in range(4)]))
    obs = parts[0]["obs"]
    assert all(abs(p["obs"] - obs) < 1e-12 for p in parts)
    out = {"name": FP.stem, "half": "S2_half0", "obs": obs}
    for name in ("unrestricted", "epoch"):
        nd = np.concatenate([p[name]["draws"] for p in parts])
        out[name] = {"n": len(nd), "mean": float(nd.mean()), "sd": float(nd.std()), "z": float((obs - nd.mean()) / nd.std()),
                     "p_one": float((1 + (nd >= obs).sum()) / (1 + len(nd))), "n_ge": int((nd >= obs).sum())}
    print(json.dumps(out, indent=1))
    (HERE / "fl_one_half.json").write_text(json.dumps(out, indent=1))
