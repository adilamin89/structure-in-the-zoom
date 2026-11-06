"""H2-R25: reproducibility of the declared vs order-averaged depth profile across two prompt samples (run17's 16 vs run23's
16 per class) of the 32/class batteries. Out: data_canonical/h2_profile_halves.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h2_llm_battery as hb  # noqa: E402


def profiles(H, lab):
    nc = 8
    orng = np.random.default_rng(3701)
    orders = [orng.permutation(nc).tolist() for _ in range(50)]
    decl, oav = [], []
    for l in range(H.shape[0]):
        K = hb.gram(H[l], "linear")
        mem = hb.members_of(lab, nc)
        th, sizes = hb.ladder(K, mem, list(range(nc)), [1, 2, 3, 4, 6, 8])
        f = hb.floor(K, len(lab), sizes, l * 100 + 1)
        decl.append(th - f)
        oav.append(hb.order_avg(K, mem, orders, [1, 2, 3, 4, 6, 8], f))
    return np.array(decl), np.array(oav)


def main():
    out = {}
    for m in sys.argv[1:]:
        out[m] = {}
        for ax in ("world_knowledge32", "language_type32"):
            z = np.load(hb.STATES / f"{m}__{ax}.npz")
            H, lab = z["H"], z["labels"]
            within = np.arange(len(lab)) % 32
            res = {}
            for half, sel in (("orig16", within < 16), ("new16", within >= 16)):
                d, o = profiles(H[:, sel], lab[sel])
                res[half] = {"declared": d.tolist(), "orderavg": o.tolist()}
            r_d = float(np.corrcoef(res["orig16"]["declared"][1:], res["new16"]["declared"][1:])[0, 1])
            r_o = float(np.corrcoef(res["orig16"]["orderavg"][1:], res["new16"]["orderavg"][1:])[0, 1])
            res["r_declared"], res["r_orderavg"] = r_d, r_o
            out[m][ax] = res
            print(f"{m} {ax}: profile r across prompt samples declared {r_d:+.2f}, order-averaged {r_o:+.2f}", flush=True)
            json.dump(out, open(hb.OUTDIR / "h2_profile_halves.json", "w"), indent=1)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
