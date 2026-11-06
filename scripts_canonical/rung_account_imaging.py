"""`rung data --account` on the eight grating recordings against Figure 3's imaging points and Appendix K's finite-sample test.
Per recording: rung.account with the paper's split halves for sigma (alternate presentations within each 7.5-degree sub-bin) and
the paper's floor (ten draws, seed 42, rung by rung), so the mean-field law, Eq. 3's term and the measured parts replay
twoterm_imaging.json and Table 9; the finite-sample identity and its held-out arm against finite_n_imaging.json (whose moment
splits and floor compositions draw from other streams). Across the eight: sign agreement and Pearson r of the two-term sum
with the shift (Figure 3C), and the held-out mean error of the identity (Appendix K).
Run: VECLIB_MAXIMUM_THREADS=3 python -X faulthandler rung_account_imaging.py [n_workers]   Out: data_canonical/rung_account_imaging.json"""
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW, DATA = HERE.parent / "raw", HERE.parent / "data_canonical"


def run_one(fp):
    sys.path.insert(0, str(HERE.parent))
    import rung
    t0 = time.time()
    d = np.load(fp, allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(d["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(d["istim"], float).ravel(); del d
    bins = lambda nb: np.clip(np.digitize(phi % (2 * np.pi), np.linspace(0, 2 * np.pi, nb + 1)) - 1, 0, nb - 1)
    lab, sub48 = bins(8), bins(48)
    h = np.zeros(len(phi), int)
    for b in np.unique(sub48):
        idx = np.where(sub48 == b)[0]; h[idx[1::2]] = 1
    a = rung.account(X.astype(np.float64), lab, halves=h, n_null=10, seed=42, floor_by_rung=True)
    tt = next(r for r in json.load(open(DATA / "twoterm_imaging.json"))["rows"] if r["name"] == fp.stem)["B2"]
    fi = next(r for r in json.load(open(DATA / "finite_n_imaging.json"))["rows"] if r["name"] == fp.stem)
    row = {"name": fp.stem, "account": a, "paper": {
        "sigma": tt["sigma"], "mean_field": tt["within_pred"], "class_mean_pred": tt["classmean_pred"], "delta_pred": tt["delta_pred"],
        "within": tt["t9_within"], "class_mean": tt["t9_class_means"], "delta": tt["t9_delta"], "finite": fi["full"]["W2"],
        "heldout_within": float(np.mean([q["measured_within"] for q in fi["halves"]])),
        "heldout_finite": float(np.mean([q["W2x"] for q in fi["halves"]]))}}
    p = row["paper"]
    print(f"{fp.stem[9:34]:26s} sigma {a['sigma']:.4f} ({p['sigma']:.4f}) MF {a['within_pred']['mean_field']:+.4f} ({p['mean_field']:+.4f}) "
          f"cm-pred {a['class_mean_pred']:+.4f} ({p['class_mean_pred']:+.4f}) within {a['measured']['within']:+.4f} ({p['within']:+.4f}) "
          f"finite {a['within_pred']['finite']:+.4f} ({p['finite']:+.4f}) held-out {a['heldout']['finite_heldout'] - a['heldout']['measured_within']:+.4f} "
          f"({p['heldout_finite'] - p['heldout_within']:+.4f}) ({time.time() - t0:.0f}s) DONE", flush=True)
    return row


def main():
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    files = sorted(RAW.glob("gratings*.npy"))
    assert len(files) == 8, files
    with ProcessPoolExecutor(nw, mp_context=mp.get_context("spawn")) as ex:
        rows = list(ex.map(run_one, files))
    fv = json.load(open(DATA / "finite_n_imaging.json"))["verdict"]
    A = lambda k: np.array([k(r["account"]) for r in rows])
    Pp = lambda k: np.array([r["paper"][k] for r in rows])
    dpred, dmeas = A(lambda a: a["delta_pred_mean_field"]), A(lambda a: a["measured"]["delta"])
    v = {"n": len(rows),
         "signs_agree_two_terms": int(np.sum(np.sign(dpred) == np.sign(dmeas))),
         "r_two_terms_delta": float(np.corrcoef(dpred, dmeas)[0, 1]),
         "r_paper_two_terms_delta": float(np.corrcoef(Pp("delta_pred"), Pp("delta"))[0, 1]),
         "max_abs_diff_sigma": float(np.max(np.abs(A(lambda a: a["sigma"]) - Pp("sigma")))),
         "max_abs_diff_mean_field": float(np.max(np.abs(A(lambda a: a["within_pred"]["mean_field"]) - Pp("mean_field")))),
         "max_abs_diff_class_mean_pred": float(np.max(np.abs(A(lambda a: a["class_mean_pred"]) - Pp("class_mean_pred")))),
         "max_abs_diff_within": float(np.max(np.abs(A(lambda a: a["measured"]["within"]) - Pp("within")))),
         "max_abs_diff_delta": float(np.max(np.abs(dmeas - Pp("delta")))),
         "mean_err_finite_full": float(np.mean(A(lambda a: a["within_pred"]["finite"]) - A(lambda a: a["measured"]["within"]))),
         "mean_err_finite_heldout": float(np.mean(A(lambda a: a["heldout"]["finite_heldout"] - a["heldout"]["measured_within"]))),
         "paper_mean_err_finite_full": fv["mean_err_full"]["W2"], "paper_mean_err_finite_heldout": fv["mean_err_halves"]["W2x"],
         "signs_agree_class_mean": int(np.sum([r["account"]["signs_agree"]["class_mean"] for r in rows]))}
    print("VERDICT", json.dumps(v, indent=1), flush=True)
    json.dump({"rows": rows, "verdict": v}, open(DATA / "rung_account_imaging.json", "w"), indent=1, default=float)
    print("wrote rung_account_imaging.json ALL DONE", flush=True)


if __name__ == "__main__":
    main()
