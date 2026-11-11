"""rung 1.14's continuous labels and drift check on the eight grating recordings of Stringer et al. 2021 (Figshare 8279387, saved as
raw/gratings_*.npy: sresp neurons x trials, istim radians; rows in presentation order).
  replay   cells_from_continuous(istim, 8, cyclic) + residualize_within_cells(6 sub-cells) + the paper floor (10 draws, seed 42,
           rung by rung) against data_canonical/cr_order_and_withinbin_controls.json 'orig' and 'withinbin_8x6' (delta, within).
           The paper's generator works in float32 (its Gram from float32 rows); rung in float64: agreement to float32 precision.
  origins  continuous(..., n_origins=8) with rung's defaults (20 floor draws, seed 0, exact order average), raw and residualized:
           the declared shift, order-averaged shift and within-class part over the 8 wall positions (5.625-degree steps).
  drift    drift_check(X, 45-degree classes) in row order, 20 random interleaved splits.
Out: data_canonical/continuous_gratings.json (one row per recording, written after each). Run: python scripts_canonical/continuous_gratings.py"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import rung  # noqa: E402

RAW = HERE.parent / "raw"   # the Stringer et al. 2021 grating recordings (Figshare 8279387), saved as .npy
ART = HERE.parent / "data_canonical" / "cr_order_and_withinbin_controls.json"
OUT = HERE.parent / "data_canonical" / "continuous_gratings.json"


def cpu():
    u = resource.getrusage(resource.RUSAGE_SELF)
    return u.ru_utime + u.ru_stime


def replay(X, phi, residualize):
    lab = rung.cells_from_continuous(phi, 8, "width", cyclic=True)
    if residualize:
        sub = lab * 48 + rung.cells_from_continuous(phi, 48, "width", cyclic=True)
        X, share = rung.residualize_within_cells(X, lab, sub)
    # the paper floor's draws (10, seed 42, rung by rung) in the fast Gram arithmetic: the same draws, equal to rows to ~1e-12
    zr = rung.zoom(X, lab, n_perm=0, k_orders=0, n_floor_draws=10, floor_seed=42, floor_by_rung=True, workers=0, fast=True)
    d = rung.decompose(X, lab, n_null=10, seed=42, floor_by_rung=True, n_bias=0, workers=0, fast=True)
    out = {"delta": zr["delta"], "within": d["delta_split_free"]["D"], "split_delta": d["delta"]}
    if residualize:
        out["within_cell_tuning_share"] = share
    return out


def main():
    art = {r_["name"]: r_ for r_ in json.load(open(ART))["rows"]}
    rows = []
    for fp in sorted(RAW.glob("gratings*.npy")):
        t0, c0 = time.time(), cpu()
        dat = np.load(fp, allow_pickle=True).item()
        X = np.ascontiguousarray(np.asarray(dat["sresp"], np.float32).T); X /= X.std() + 1e-9
        phi = np.asarray(dat["istim"], float).ravel(); del dat
        X = X.astype(np.float64)
        a = art[fp.stem]
        row = {"name": fp.stem, "n": int(len(phi)), "n_neurons": int(X.shape[1])}
        for tag, res, key in (("orig", False, "orig"), ("withinbin_8x6", True, "withinbin_8x6")):
            rp = replay(X, phi, res)
            rp["paper"] = {k: a[key][k] for k in ("delta", "within")}
            rp["abs_diff"] = max(abs(rp["delta"] - a[key]["delta"]), abs(rp["within"] - a[key]["within"]))
            row[f"replay_{tag}"] = rp
        print(f"{fp.stem[:34]:34s} replay orig d {row['replay_orig']['delta']:+.4f} (paper {a['orig']['delta']:+.4f}) | 8x6 d "
              f"{row['replay_withinbin_8x6']['delta']:+.4f} (paper {a['withinbin_8x6']['delta']:+.4f}) w "
              f"{row['replay_withinbin_8x6']['within']:+.4f} (paper {a['withinbin_8x6']['within']:+.4f}) | max diff "
              f"{max(row['replay_orig']['abs_diff'], row['replay_withinbin_8x6']['abs_diff']):.1e}", flush=True)
        for tag, res in (("origins_raw", False), ("origins_residualized", True)):
            c = rung.continuous(X, phi, 8, cyclic=True, residualize=res, n_perm=0, k_orders=50, seed=0, workers=0)
            c.pop("first")
            row[tag] = c
            s = c["summary"]
            print(f"  {tag:22s} declared {s['delta']['min']:+.3f}..{s['delta']['max']:+.3f} (SD {s['delta']['sd']:.3f}) | order-avg "
                  f"{s['delta_orderavg']['min']:+.3f}..{s['delta_orderavg']['max']:+.3f} (SD {s['delta_orderavg']['sd']:.3f}) | D "
                  f"{s['within']['min']:+.3f}..{s['within']['max']:+.3f} | class means {s['class_means']['min']:+.3f}..{s['class_means']['max']:+.3f}",
                  flush=True)
        lab8 = rung.cells_from_continuous(phi, 8, cyclic=True)
        dr = rung.drift_check(X, lab8, n_splits=20, seed=0, workers=0)
        row["drift"] = {k: v for k, v in dr.items()}
        print("  " + rung._drift_lines(dr)[0], flush=True)
        row["cpu_s"], row["wall_s"] = cpu() - c0, time.time() - t0
        rows.append(row)
        json.dump({"rung": rung.__version__, "rows": rows}, open(OUT, "w"), indent=1)
    v = {"replay_max_abs_diff": max(max(r_["replay_orig"]["abs_diff"], r_["replay_withinbin_8x6"]["abs_diff"]) for r_ in rows),
         "orderavg_positive_all_origins": sum(r_["origins_raw"]["summary"]["delta_orderavg"]["min"] > 0 for r_ in rows),
         "within_positive_all_origins": sum(r_["origins_raw"]["summary"]["within"]["min"] > 0 for r_ in rows),
         "class_means_negative_all_origins": sum(r_["origins_raw"]["summary"]["class_means"]["max"] < 0 for r_ in rows),
         "drift_flagged": {r_["name"]: r_["drift"].get("flagged") for r_ in rows}}
    print("VERDICT", json.dumps(v), flush=True)
    json.dump({"rung": rung.__version__, "rows": rows, "verdict": v}, open(OUT, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
