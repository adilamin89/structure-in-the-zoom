"""Score the cyclic day/month axes from cyclic_axes_order_<model>.json; every criterion printed.
R14: months-final kernel harmonics without May and March, from the cached states. Out: data_canonical/cyclic_axes_score.json"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cyclic_axes_order as cy  # noqa: E402

MODELS = ["pythia-160m", "pythia-410m-deduped", "pythia-1b-deduped", "pythia-2.8b-deduped"]


def late(rows, place):
    n = len(rows)
    return [r for r in rows if (r["layer"] >= n / 3 if place == "inner" else r["layer"] >= 1)]


def main():
    out = {}
    for m in MODELS:
        f = cy.hb.OUTDIR / f"cyclic_axes_order_{m}.json"
        if not f.exists():
            continue
        art = json.load(open(f))["batteries"]
        out[m] = {}
        for b, res in art.items():
            place = "inner" if b.split("_")[-1].startswith("inner") else "final"
            rows = res["layers"]
            lr = late(rows, place)
            h1 = sum(r["h1_p_one"] <= 0.05 for r in lr)
            r12 = sum(r["orderavg_carrier"]["p_one"] <= 0.05 for r in lr)
            sig = [r for r in lr if r["h1_p_one"] <= 0.05]
            r10_sign = sum(np.sign(r["D_cycle_minus_far"]["D"]) == np.sign(r["kernel"]["f1_minus_fanti"]) for r in sig)
            r10_p = sum(r["D_cycle_minus_far"]["p_two"] <= 0.05 for r in sig)
            r11 = (abs(rows[0]["cycle_carrier"]["delta"]) < 1e-6 and abs(rows[0]["orderavg_carrier"]["delta"]) < 1e-6) \
                if place == "inner" else None
            rec = {"n_late": len(lr), "R9_h1_above_null": h1, "R9_pass": h1 >= 2 / 3 * len(lr),
                   "R12_orderavg_above_null": r12, "R12_pass": r12 >= 2 / 3 * len(lr),
                   "R10_sign_agree": r10_sign, "R10_n": len(sig), "R10_p_le_05": r10_p,
                   "R10_pass": len(sig) > 0 and r10_sign >= 0.75 * len(sig) and r10_p >= 0.5 * len(sig),
                   "R11_carrier_floor_zero_at_embedding": r11,
                   "R11_ordinary_floor_delta_at_embedding": rows[0]["orderavg_ordinary"]["delta"] if place == "inner" else None,
                   "R13_cycle_p_le_05": sum(r["cycle_carrier"]["p_one"] <= 0.05 for r in lr),
                   "cycle_pct_range": [min(r["cycle_pct_among_orders"] for r in lr), max(r["cycle_pct_among_orders"] for r in lr)],
                   "h1_share_range": [min(r["kernel"]["h1_share"] for r in lr), max(r["kernel"]["h1_share"] for r in lr)],
                   "circularity_max": max(r["kernel"]["cycle_neighbours_in_pc_plane"] for r in lr),
                   "cycle_carrier_range": [min(r["cycle_carrier"]["delta"] for r in lr), max(r["cycle_carrier"]["delta"] for r in lr)],
                   "cycle_null_sd_median": float(np.median([r["cycle_carrier"]["null_sd"] for r in lr])),
                   "orderavg_carrier_range": [min(r["orderavg_carrier"]["delta"] for r in lr),
                                              max(r["orderavg_carrier"]["delta"] for r in lr)]}
            out[m][b] = rec
            print(m, b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, flush=True)
        # R14: months final without May (4) and March (2)
        z = np.load(cy.hb.STATES / f"{m}__cyclic_months_final.npz")
        H, lab = z["H"], z["labels"]
        def h1_frac(keep):
            shares = []
            for l in range(1, H.shape[0]):
                X = H[l].astype(np.float64)
                means = np.stack([X[lab == c].mean(0) for c in keep])
                cm = means - means.mean(0)
                th = np.array([2 * np.pi * c / 12 for c in keep])
                A = np.vstack([np.cos(th), np.sin(th)]).T
                coef, *_ = np.linalg.lstsq(A, cm, rcond=None)
                shares.append(float(((A @ coef) ** 2).sum() / (cm ** 2).sum()))
            return shares
        out[m]["R14_months_final_all12_h1_frac"] = h1_frac(list(range(12)))
        keep = [c for c in range(12) if c not in (2, 4)]
        shares = []
        for l in range(1, H.shape[0]):
            X = H[l].astype(np.float64)
            means = np.stack([X[lab == c].mean(0) for c in keep])
            cm = means - means.mean(0)
            th = np.array([2 * np.pi * c / 12 for c in keep])
            # first-harmonic share of the class-mean variance on the (irregular) 10-point circle: project on cos/sin of the angle
            A = np.vstack([np.cos(th), np.sin(th)]).T
            coef, *_ = np.linalg.lstsq(A, cm, rcond=None)
            fit = A @ coef
            shares.append(float((fit ** 2).sum() / (cm ** 2).sum()))
        out[m]["R14_months_final_no_May_March_h1_share"] = shares
        print(m, "R14 h1 fraction 10 months:", np.round(shares, 2).tolist(), "| all 12:",
              np.round(out[m]["R14_months_final_all12_h1_frac"], 2).tolist(), flush=True)
    json.dump(out, open(cy.hb.OUTDIR / "cyclic_axes_score.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
