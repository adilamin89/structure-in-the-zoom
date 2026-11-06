"""Every registered criterion printed with its number before its verdict.

Usage: python llm_battery_score.py  -> data_canonical/llm_battery_score.json (scores whatever artifacts exist; missing ones are listed)
"""
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data_canonical"


def scale(rho):
    """The registered outcome scale for a rank correlation (the 0.3-0.6 band was not given a name: printed as such)."""
    if rho >= 0.6:
        return "REDUNDANT-candidate (rho >= 0.6; partial decides)"
    if rho <= -0.6:
        return "AGAINST"
    if abs(rho) < 0.3:
        return "COMPLEMENTARY"
    return "UNNAMED BAND (0.3 <= |rho| < 0.6)"


def probes(out, mkey):
    for b in ("world_knowledge", "language_type"):
        f = DATA / f"llm_probes_{mkey}__{b}.json"
        if not f.exists():
            out["missing"].append(f.name)
            continue
        a = json.load(open(f))
        L = a["layers"][1:]
        n = len(L)
        if "spearman_lc2_vs_excess" not in a:          # run still in progress: report as partial, score nothing
            out["missing"].append(f"{f.name} (partial: {len(a['layers'])} layers)")
            continue
        acc = np.array([x["acc"] for x in L])
        row = {"n_layers_past_embedding": n, "acc_range": [float(acc.min()), float(acc.max())],
               "embedding_acc": a["layers"][0]["acc"],
               "control_range": [min(x["control"] for x in L), max(x["control"] for x in L)],
               "lc2_range": [min(x["learning_curve"]["2"]["mean"] for x in L), max(x["learning_curve"]["2"]["mean"] for x in L)],
               "Q1_layers_acc_ge_0.90": int((acc >= 0.90).sum())}
        row["Q1_verdict"] = "PASS" if row["Q1_layers_acc_ge_0.90"] >= 0.9 * n else "MISS"
        if "spearman_lc2_vs_excess" in a:
            for k in ("spearman_lc2_vs_excess", "spearman_lc12_vs_excess", "spearman_acc_vs_excess"):
                row[k] = a[k]["rho"]
            row["Q2_scale"] = scale(a["spearman_lc2_vs_excess"]["rho"])
        P = [x["projected"] for x in a["layers"]]
        above = [p["p_one"] <= 0.05 for p in P]
        row["Q5_projected_above_null"] = f"{sum(above)}/{len(P)}"
        row["Q5_dbar_proj_range"] = [min(p["dbar200_proj"] for p in P), max(p["dbar200_proj"] for p in P)]
        row["Q5_unprojected_dbar200_range"] = [min(x["shift"]["dbar200"] for x in a["layers"]),
                                               max(x["shift"]["dbar200"] for x in a["layers"])]
        row["Q5_probe_after_projection_range"] = [min(p["probe_acc_after_projection"] for p in P),
                                                  max(p["probe_acc_after_projection"] for p in P)]
        row["Q5_verdict"] = "PASS (RUNG-UNIQUE)" if sum(above) >= 2 / 3 * len(P) else "MISS"
        out["A1"][f"{mkey}/{b}"] = row
    f = DATA / f"llm_probes_{mkey}__blimp_B.json"
    if not f.exists():
        out["missing"].append(f.name)
        return
    a = json.load(open(f))
    L = a["layers"][1:]
    n = len(L)
    if n < 2:
        out["missing"].append(f"{f.name} (partial: {len(a['layers'])} layers)")
        return
    ph = [min(x["random"]["phenomenon"]["acc"], x["pair_held_out"]["phenomenon"]["acc"]) for x in L]
    mp = np.array([x["pair_held_out"]["grammatical"]["minimal_pair_acc"] for x in L])
    row = {"n_layers_past_embedding": n,
           "Q3_layers_phenomenon_ge_0.95_both_splits": int(sum(v >= 0.95 for v in ph)),
           "phenomenon_min_range": [float(min(ph)), float(max(ph))],
           "class8_random_range": [min(x["random"]["class8"]["acc"] for x in L), max(x["random"]["class8"]["acc"] for x in L)],
           "class8_pairheld_range": [min(x["pair_held_out"]["class8"]["acc"] for x in L),
                                     max(x["pair_held_out"]["class8"]["acc"] for x in L)],
           "grammatical_random_range": [min(x["random"]["grammatical"]["acc"] for x in L),
                                        max(x["random"]["grammatical"]["acc"] for x in L)],
           "grammatical_pairheld_range": [min(x["pair_held_out"]["grammatical"]["acc"] for x in L),
                                          max(x["pair_held_out"]["grammatical"]["acc"] for x in L)],
           "minimal_pair_range": [float(mp.min()), float(mp.max())],
           "Q4_layers_minpair_gt_0.75": int((mp > 0.75).sum())}
    row["Q3_verdict"] = "PASS" if row["Q3_layers_phenomenon_ge_0.95_both_splits"] >= 0.9 * n else "MISS"
    row["Q4_verdict"] = ("PASS (TOOL-UNIQUE)" if row["Q4_layers_minpair_gt_0.75"] >= n / 2 else "MISS") \
        if mkey == "pythia-2.8b-deduped" else "reported (Q4 is scored at 2.8b-d)"
    out["A1"][f"{mkey}/blimp_B"] = row


def generality(out, key, art_file, bkey="batteries"):
    f = DATA / art_file
    if not f.exists():
        out["missing"].append(f.name)
        return
    A = json.load(open(f))[bkey]
    for b in ("world_knowledge", "language_type"):
        if b not in A:
            continue
        R_all = A[b]["layers"]
        # a layer whose states are identical across prompts (BERT [CLS] at the embedding, one token for every prompt)
        # has no spectrum: excluded from counts and listed, as llm_battery does for BLiMP's embedding layer
        degenerate = [r["layer"] for r in R_all if r["perm"]["null_sd"] == 0.0 and r["dbar50_run37"] == 0.0]
        R = [r for r in R_all if r["layer"] not in degenerate]
        n = len(R)
        g1 = [r["dbar50_run37"] > 0 and r["perm"]["p_one"] <= 0.05 for r in R]
        dom = [r for r in R if r.get("spectrum", {}).get("top1_share", 0) > 0.5]
        row = {"n_layers": n, "G1_count": f"{sum(g1)}/{n}", "G1_verdict": "PASS" if sum(g1) >= 0.9 * n else "MISS",
               "G1_failing_layers": [r["layer"] for r, g in zip(R, g1) if not g],
               "dbar50_range": [min(r["dbar50_run37"] for r in R), max(r["dbar50_run37"] for r in R)],
               "dbar200_range": [min(r["dbar200"] for r in R), max(r["dbar200"] for r in R)],
               "floor_range": [min(r["floor"] for r in R), max(r["floor"] for r in R)],
               "declared_pct_range": [min(r["declared_pct"] for r in R), max(r["declared_pct"] for r in R)],
               "G2_orderavg_IE_p_two": A[b]["profile"]["orderavg"]["p_two"],
               "G2_verdict": ("PASS" if A[b]["profile"]["orderavg"]["p_two"] > 0.05 else "MISS") if n >= 3
               else "n/a (fewer than 3 layers: no profile)",
               "degenerate_layers_excluded": degenerate,
               "excess_range": [min(r["dbar50_run37"] - r["perm"]["null_mean"] for r in R),
                                max(r["dbar50_run37"] - r["perm"]["null_mean"] for r in R)],
               "declared_IE_p_two": A[b]["profile"]["declared"]["p_two"],
               "G3_dominated_layers": [r["layer"] for r in dom],
               "G3_repaired_sign": {r["layer"]: {"linear": r["dbar50_run37"], "zscore": r["zscore"]["dbar50"],
                                                 "pc4": r["pc4"]["dbar50"]} for r in dom}}
        out["B"][f"{key}/{b}"] = row


def r10b(out, key, lin_file, z_file):
    """Z-scored estimator with its own null; G1z and E1 (significant opposite signs, linear vs z-scored)."""
    fl, fz = DATA / lin_file, DATA / z_file
    if not (fl.exists() and fz.exists()):
        out["missing"].append(fz.name)
        return
    L = json.load(open(fl))["batteries"]
    Z = json.load(open(fz))["batteries"]
    for b in ("world_knowledge", "language_type"):
        if b not in Z or b not in L:
            continue
        zl = Z[b]["layers"]
        ll = L[b]["layers"]
        assert max(abs(a["dbar50_run37"] - c["zscore"]["dbar50"]) for a, c in zip(zl, ll)) < 1e-9, "z-run != zscore column"
        degenerate = [r["layer"] for r in zl if r["perm"]["null_sd"] == 0.0 and r["dbar50_run37"] == 0.0]
        keep = [i for i, r in enumerate(zl) if r["layer"] not in degenerate]
        g1 = [zl[i]["dbar50_run37"] > 0 and zl[i]["perm"]["p_one"] <= 0.05 for i in keep]

        def ex(r):
            return r["dbar50_run37"] - r["perm"]["null_mean"], r["perm"]["null_sd"]
        opp = []
        for i in keep:
            (ea, sa), (ec, sc) = ex(zl[i]), ex(ll[i])
            if ea * ec < 0 and abs(ea) > sa and abs(ec) > sc:
                opp.append(zl[i]["layer"])
        pos_lin = sum(ex(ll[i])[0] > 0 for i in keep)
        pos_z = sum(ex(zl[i])[0] > 0 for i in keep)
        out["R10b"][f"{key}/{b}"] = {"G1z_count": f"{sum(g1)}/{len(keep)}",
                                      "descriptive_excess_positive_linear": f"{pos_lin}/{len(keep)}",
                                      "descriptive_excess_positive_z": f"{pos_z}/{len(keep)}",
                                      "G2z_orderavg_IE_p_two": Z[b]["profile"]["orderavg"]["p_two"] if len(keep) >= 3 else None,
                                      "G1z_verdict": "PASS" if sum(g1) >= 0.9 * len(keep) else "MISS",
                                      "G1z_failing_layers": [zl[i]["layer"] for i, g in zip(keep, g1) if not g],
                                      "z_dbar50_range": [min(zl[i]["dbar50_run37"] for i in keep),
                                                         max(zl[i]["dbar50_run37"] for i in keep)],
                                      "z_excess_range": [min(ex(zl[i])[0] for i in keep), max(ex(zl[i])[0] for i in keep)],
                                      "E1_opposite_sign_layers": opp, "degenerate_excluded": degenerate,
                                      "linear_G1_count": f"{sum(ll[i]['dbar50_run37'] > 0 and ll[i]['perm']['p_one'] <= 0.05 for i in keep)}/{len(keep)}"}


def main():
    out = {"A1": {}, "B": {}, "R10b": {}, "missing": []}
    for m in ("pythia-160m", "pythia-2.8b-deduped", "pythia-410m-deduped", "pythia-1b-deduped"):
        probes(out, m)
    for key in ("gpt-neo-1.3B", "mamba-130m-hf", "bert-base-uncased_cls", "bert-base-uncased_mean",
                "all-MiniLM-L6-v2_mean", "all-MiniLM-L6-v2_sent", "pythia-160m_attn", "pythia-160m_mlp"):
        generality(out, key, f"llm_architectures_{key}.json")
    generality(out, "OLMo-2-0425-1B (H2 artifact)", "llm_battery_OLMo-2-0425-1B.json")
    generality(out, "pythia-160m residual (H2 artifact)", "llm_battery_pythia-160m.json")
    for key in ("gpt-neo-1.3B", "mamba-130m-hf", "bert-base-uncased_cls", "bert-base-uncased_mean", "all-MiniLM-L6-v2_mean",
                "all-MiniLM-L6-v2_sent", "pythia-160m_attn", "pythia-160m_mlp"):
        r10b(out, key, f"llm_architectures_{key}.json", f"llm_architectures_{key}__zscore.json")
    r10b(out, "OLMo-2-0425-1B", "llm_battery_OLMo-2-0425-1B.json", "llm_architectures_OLMo-2-0425-1B__zscore.json")
    r10b(out, "pythia-160m", "llm_battery_pythia-160m.json", "llm_architectures_pythia-160m__zscore.json")
    for f in ("v1_decoders.json", "llm_ioi.json"):
        p = DATA / f
        if not p.exists():
            out["missing"].append(f)
            continue
        a = json.load(open(p))
        if f == "v1_decoders.json":
            from scipy import stats
            out["A2"] = a.get("score")
            th = {r["name"]: r for r in json.load(open(DATA / "cr_ds_thirds_crossfit.json"))["rows"]}
            for k in ("sel0_meas1", "sel1_meas0"):   # unregistered descriptive: size of the DS-minus-nonDS gap
                dg = [r["thirds_decode"][k]["DS"]["256"] - r["thirds_decode"][k]["nonDS"]["256"] for r in a["rows"]]
                sg = [th[r["name"]][k]["DS"] - th[r["name"]][k]["nonDS"] for r in a["rows"]]
                out["A2"][f"descriptive_{k}_gap_spearman_decode_vs_delta"] = float(stats.spearmanr(dg, sg)[0])
            # protocol-demeaned Pearson (descriptive; three protocols, n = 8): does each mapping hold inside protocols?
            t9 = {r["name"]: r["orig"] for r in json.load(open(DATA / "cr_withinclass_controls.json"))["rows"]}
            prot = np.array([r["name"].split("_GT")[0] for r in a["rows"]])

            def demean(v):
                v = np.asarray(v, float)
                return v - np.array([v[prot == p].mean() for p in prot])
            acc = demean([r["decode"]["256"]["mean"] for r in a["rows"]])
            ncc = demean([r["noise_corr"]["NCchange"] for r in a["rows"]])
            for part in ("class_means", "within", "delta"):
                v = demean([t9[r["name"]][part] for r in a["rows"]])
                out["A2"][f"descriptive_protocol_demeaned_r_acc256_vs_{part}"] = float(np.corrcoef(acc, v)[0, 1])
                out["A2"][f"descriptive_protocol_demeaned_r_NCchange_vs_{part}"] = float(np.corrcoef(ncc, v)[0, 1])
            r6b = DATA / "v1_decoders_r6b.json"
            if r6b.exists():
                out["A2_R6b"] = json.load(open(r6b))["score"]
        else:
            out["B4"] = {k: v for k, v in a.items() if k.startswith("AUC") or k.startswith("rank") or k.startswith("frac")}
            # descriptive (unregistered): discrimination inside the late layers 9-11, where probes saturate
            H = [h for h in a["heads"] if int(h["head"].split(".")[0]) >= 9]
            pos = np.array([h["head"] in a["positive_set"] for h in H])
            for key in ("z", "probe_acc", "classmean_frac", "norm"):
                sc = np.array([h[key] for h in H])
                p, n = sc[pos], sc[~pos]
                out["B4"][f"descriptive_AUC_layers9to11_{key}"] = float((p[:, None] > n[None]).mean()
                                                                        + 0.5 * (p[:, None] == n[None]).mean())
            out["B4"]["descriptive_layers9to11_n_pos_neg"] = [int(pos.sum()), int((~pos).sum())]
    tot = {"linear": [0, 0], "z": [0, 0]}
    for v in out["R10b"].values():
        for k, f in (("linear", "descriptive_excess_positive_linear"), ("z", "descriptive_excess_positive_z")):
            a, b = map(int, v[f].split("/"))
            tot[k][0] += a
            tot[k][1] += b
    out["R10b_totals_excess_positive"] = {k: f"{a}/{b}" for k, (a, b) in tot.items()}
    json.dump(out, open(DATA / "llm_battery_score.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
