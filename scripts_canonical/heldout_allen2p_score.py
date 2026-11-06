"""Runs on data the paper never saw: the Allen Brain Observatory two-photon readings, summarized.

Reads the `rung data` outputs of the 61 Visual Coding two-photon experiments (data_canonical/heldout_allen2p/<id>.json, the release's
own per-presentation response; data_canonical/heldout_allen2p_nobaseline/<id>.json, the same window of the release's DfOverF with no
per-presentation baseline) and the controls (data_canonical/heldout_allen2p_controls.json) and writes
data_canonical/heldout_allen2p_summary.json, which the README's 'Runs on data the paper never saw' section quotes.

Definitions (rung keys): shift = delta (declared angular order) and delta_orderavg (50 random orders); within-class part =
decompose.delta_split_free.D; class-mean part = Tb + Cb + Cx (so the two parts sum to delta); alignment a = decompose.sign_law.alignment;
full-set remainder = decompose.sign_law.c_full; sector balance = sectors.balance; principal-angle floor share = cov_floors.floor_share.

Usage: python scripts_canonical/heldout_allen2p_score.py
"""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_canonical")


def rows_of(sub):
    rows = []
    for fn in sorted(glob.glob(os.path.join(DATA, sub, "*.json"))):
        r = json.load(open(fn))
        d = r["decompose"]; f = d["delta_split_free"]; sl = d["sign_law"]
        cm = f["Tb"] + f["Cb"] + f["Cx"]
        assert abs(f["D"] + cm - r["delta"]) < 1e-9
        rows.append(dict(id=os.path.basename(fn)[:-5], n_cells=r["n_features"], n_presentations=r["n_samples"],
                         delta=r["delta"], p_two=r["p_two"], delta_orderavg=r["delta_orderavg"],
                         delta_orderavg_sd=r["delta_orderavg_sd"], p_two_orderavg=r["p_two_orderavg"],
                         strat_p_two=r.get("strat_p_two"), tb_p_two_orderavg=r.get("tb_p_two_orderavg"),
                         theta_floor=r["theta_floor"], within_part=f["D"], class_mean_part=cm, c_full=sl["c_full"],
                         alignment=sl["alignment"], balance=r["sectors"]["balance"],
                         pa_floor_share=r["cov_floors"]["floor_share"]))
    return rows


def rank_corr(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def describe(rows):
    col = lambda k: np.array([x[k] for x in rows], float)
    oa, poa = col("delta_orderavg"), col("p_two_orderavg")
    sig = col("p_two") <= 0.05; sig_oa = poa <= 0.05
    cm, a = col("class_mean_part"), col("alignment")
    return dict(
        n=len(rows), n_cells_min=int(col("n_cells").min()), n_cells_max=int(col("n_cells").max()),
        orderavg_positive=int((oa > 0).sum()), orderavg_p05=int(sig_oa.sum()), orderavg_median=float(np.median(oa)),
        orderavg_tb_survives=int(((col("tb_p_two_orderavg") <= 0.05) & sig_oa).sum()),
        declared_p05=int(sig.sum()), declared_strat_survives=int(((col("strat_p_two") <= 0.05) & sig).sum()),
        declared_min=float(col("delta").min()), declared_max=float(col("delta").max()),
        order_sd_median=float(np.median(col("delta_orderavg_sd"))),
        within_positive=int((col("within_part") > 0).sum()), within_median=float(np.median(col("within_part"))),
        class_mean_negative=int((cm < 0).sum()), class_mean_positive=int((cm > 0).sum()), class_mean_median=float(np.median(cm)),
        part_takes_remainder_sign=int((np.sign(cm) == np.sign(col("c_full"))).sum()),
        alignment_median=float(np.median(a)), spearman_alignment_class_mean=rank_corr(a, cm),
        even_led=int((col("balance") > 1).sum()), pa_floor_share_median=float(np.median(col("pa_floor_share"))))


def main():
    base, nob = rows_of("heldout_allen2p"), rows_of("heldout_allen2p_nobaseline")
    out = {"release_response": describe(base), "no_baseline": describe(nob)}
    a0 = {r["id"]: r["alignment"] for r in base}
    pair = [(a0[r["id"]], r["alignment"]) for r in nob if r["id"] in a0]
    out["no_baseline"]["alignment_lower_than_release"] = int(sum(b < a for a, b in pair))
    ctl = os.path.join(DATA, "heldout_allen2p_controls.json")
    if os.path.exists(ctl):
        c = json.load(open(ctl))
        R = c["rows"]
        out["controls"] = dict(one_shape_negative=int(sum(r["shape"]["delta_orderavg"] < 0 for r in R)),
                               equal_means_positive=int(sum(r["meanfree"]["delta_orderavg"] > 0 for r in R)),
                               order_sd_median=c["summary"]["median_sd_data"],
                               order_sd_median_one_shape=c["summary"]["shape"]["median_sd"],
                               spearman_order_sd_class_logpr_sd=c["summary"]["spearman_sd_class_logpr_sd"], n=len(R))
    out["rows"] = {"release_response": base, "no_baseline": nob}
    json.dump(out, open(os.path.join(DATA, "heldout_allen2p_summary.json"), "w"), indent=1)
    for k in ("release_response", "no_baseline", "controls"):
        if k in out:
            print(k, json.dumps(out[k]))


if __name__ == "__main__":
    main()
