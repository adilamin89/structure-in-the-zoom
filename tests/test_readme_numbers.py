"""Every number README.md and PAPER.md print, recomputed from its artifact and checked at their rounding; every file they name
exists. Run: pytest -q tests"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_canonical")
README = "\n".join(open(os.path.join(ROOT, f)).read() for f in ("README.md", "PAPER.md"))


def load(name):
    return json.load(open(os.path.join(DATA, name)))


def printed(value, text, digits):
    """value rounds to `text` at `digits` decimals, and `text` appears in the README."""
    assert f"{value:.{digits}f}".lstrip("+") == text.lstrip("+"), (value, text)
    assert text in README, text


def test_gt3_three_probes():
    gt3 = next(r for r in load("orientation_zoom.json")["results"] if r["name"].startswith("gratings_drifting_GT3"))
    rnd = next(r for r in load("stringer_theta_decomposition.json")["rows"] if r.get("status") == "ok" and r["name"].startswith("gratings_drifting_GT3"))
    sp = next(r for r in load("neuron_zoom_spatial.json")["rows"] if r["name"].startswith("gratings_drifting_GT3"))
    printed(rnd["theta_obs"], "0.253", 3); printed(rnd["theta_floor_empirical"], "0.232", 3); printed(rnd["shift_obs"], "+0.020", 3)
    printed(gt3["theta_structured"], "0.305", 3); printed(gt3["theta_random"], "0.065", 3); printed(gt3["delta"], "0.240", 3)
    printed(sp["theta_spatial"], "0.346", 3); printed(sp["theta_floor"], "0.342", 3)
    for v, t in ((rnd["theta_obs"], "0.25"), (gt3["theta_structured"], "0.30"), (sp["theta_spatial"], "0.35")):
        assert f"{v:.2f}" == t and t in README, (v, t)


def test_random_recordings_stay_at_floor():
    ok = [r for r in load("stringer_theta_decomposition.json")["rows"] if r.get("status") == "ok"]
    assert len(ok) == 18 and "all 18 recordings" in README
    printed(np.mean([r["shift_obs"] for r in ok]), "-0.002", 3)


def test_direction_range_and_permutations():
    rows = [r for r in load("run38_v1_label_permutations.json")["rows"] if r["name"].startswith("gratings")]
    d = [r["dir8"]["delta"] for r in rows]
    assert len(d) == 8 and all(r["dir8"]["n_perm"] == 200 and r["dir8"]["n_perm_geq_obs"] == 0 for r in rows)
    printed(min(d), "0.20", 2); printed(max(d), "0.45", 2)


def test_allen_counts():
    pops = [v["delta_dir8"] for s in load("allen_expansion_all_sessions.json")["results"] if s.get("status") == "ok" for v in s["areas"].values()]
    visp = [s["areas"]["VISp"]["delta_dir8"] for s in load("allen_expansion_all_sessions.json")["results"] if s.get("status") == "ok" and "VISp" in s["areas"]]
    assert (len(pops), sum(v > 0 for v in pops)) == (167, 158) and "158 of 167" in README
    assert (len(visp), sum(v > 0 for v in visp)) == (32, 32) and "32 of 32" in README
    assert 167 - load("allen_twoterm_misses.json")["n_miss"] == 152 and "152 of 167" in README
    # the temporal-frequency-residualized arm: its negative shifts, and the two terms' signs by class at the rungs' sample counts
    # (finite, W2) and in the mean-field form (W0)
    rows = load("allen_finite_n_identity.json")["rows"]
    dm = np.array([q["delta_measured"] for q in rows]); cm = np.array([q["classmean_pred"] for q in rows])
    agree = {k: np.sign(np.array([q[k] for q in rows]) + cm) == np.sign(dm) for k in ("W0", "W2")}
    assert (len(rows), int((dm < 0).sum())) == (167, 16) and "negative in 16 of the 167" in README
    assert (int(agree["W2"].sum()), int((agree["W2"] & (dm < 0)).sum())) == (163, 14) and "14 of those 16 and of 163\n  of 167" in README
    assert (int(agree["W0"].sum()), int((agree["W0"] & (dm < 0)).sum())) == (152, 2) and "gives 2 of the 16 (152 of 167)" in README


def test_decoders_with_the_means_removed():
    q = [r[h]["qda"] for r in load("meanfree_shift_exact.json")["rows"] for h in ("S1_half0", "S1_half1", "S2_half0", "S2_half1")]
    printed(min(x["qda"] for x in q), "0.21", 2); printed(max(x["qda"] for x in q), "0.69", 2)
    assert "chance 0.125" in README


def test_covariance_floors():
    share = load("allen_covariance_floors_decoders.json")["score"]["pa_5"]["floor_share_median"]
    assert f"{100 * share:.0f}%" == "69%" and "69%" in README
    assert f"{100 * load('principal_angle_floor_imaging.json')['score']['law_A_relerr']:.0f}%" == "2%"
    assert f"{100 * load('principal_angle_floor_allen.json')['summary']['law_A_relerr_median']:.0f}%" == "6%"
    assert "2% of the\n  permutation floors in imaging and 6%" in README


def test_every_named_file_exists():
    # table cells escape file globs (a backslash before each asterisk) so that GitHub does not render them as italics: read them as plain globs
    names = set(re.findall(r"([A-Za-z0-9_\-*/]+\.(?:json|py|sh|png))", README.replace("\\*", "*")))
    missing = []
    for n in names:
        if n.startswith(("http", "raw/")) or n in ("X.npy", "r.json", "lt.json", "lm.json", "v1.json", "pairs.json", "my.json",
                                                        "r.png", "lt.png", "v1.png", "strata.json", "gt3.json", "gt3.png",
                                                        "heads.json", "lm.png", "toy.json", "toy.png"):   # command-line examples
            continue
        hits = [p for base in ("", "scripts_canonical", "data_canonical", "figures_canonical", "tests")
                for p in glob.glob(os.path.join(ROOT, base, n))]
        if not hits:
            missing.append(n)
    assert not missing, sorted(missing)


def test_finite_n_rule():
    """README rule 8: samples per class, in units of PR_class, at which the identity evaluated at each rung's sample counts puts the
    within part of eight equal Gaussian classes of PR_class = 30 within 0.05 of the population identity's value on the same rungs (finite_n_rule.json 'rule',
    D = W2 - W1)."""
    for sigma, text in ((0.2, "20"), (0.44, "8"), (0.7, "3")):
        assert f"{load('finite_n_rule.json')['rule'][f'{sigma}_0.05']:.0f}" == text, (sigma, text)
    assert "within 0.05 of the population identity's value on the same rungs" in README and "about 20, 8 and 3 times" in README and "σ = 0.2, 0.44 and 0.7" in README and "PR_class = 30" in README
    grid = load("finite_n_rule.json")["grid"]                  # 'later for classes of lower PR': at PR 5 the deficit is larger
    for sigma, r in ((0.44, 8), (0.7, 4)):
        d5, d30 = (next(q["D"] for q in grid if q["sigma"] == sigma and q["PR_class"] == pr and q["n_over_PR"] == r) for pr in (5, 30))
        assert d5 < d30 < 0, (sigma, d5, d30)
    assert "at higher ratios n/PR_class for classes of" in README


def test_feature_reference_numbers():
    t9 = [r["orig"] for r in load("cr_order_and_withinbin_controls.json")["rows"] if r["name"].startswith("gratings")]
    assert len(t9) == 8
    printed(min(r["within"] for r in t9), "+0.29", 2); printed(max(r["within"] for r in t9), "+0.64", 2)
    printed(max(r["class_means"] for r in t9), "-0.08", 2); printed(min(r["class_means"] for r in t9), "-0.35", 2)
    assert "+0.29 to\n+0.64 and the class-mean part -0.08 to -0.35" in README
    sp = load("neuron_zoom_spatial.json")
    printed(sp["mean_delta_spatial"], "-0.033", 3)
    assert (sp["n_negative"], len(sp["rows"])) == (8, 10) and "negative in 8 of 10 recordings" in README
    printed(load("cr_graining_table.json")["rows"][-1]["random_blocks"]["delta"], "+0.027", 3)
    assert load("cr_graining_table.json")["rows"][-1]["K"] == 32 and "(+0.027 at 32" in README
    io = load("llm_ioi.json")
    printed(io["AUC_z"], "0.92", 2); printed(io["AUC_classmean_frac"], "0.96", 2); printed(io["AUC_probe_acc"], "0.95", 2)
    assert "circuit the class-mean share ranked the patched heads at 0.96, z at 0.92" in README          # rule 14
    tt = load("twoterm_imaging.json")["rows"]
    # imaging tests the account's size, not its sign: every imaging shift is positive (the always-positive baseline gets 8 of 8)
    pr, ms = np.array([r["B2"]["delta_pred"] for r in tt]), np.array([r["B2"]["t9_delta"] for r in tt])
    assert len(ms) == 8 and (ms > 0).all() and "all eight shifts are positive there" in README
    assert f"{np.corrcoef(pr, ms)[0, 1]:.2f}" == "0.88" and "(r = 0.88,\nFigure 3)" in README
    am = load("allen_axis_screen_rescore.json")["b_across_mice"]
    assert all(am[k]["pick"] == "direction" and am[k]["confirmed"] for k in ("pick_set0", "pick_set1"))
    assert "picks direction on 16 Allen mice and again on the other 16" in README
    v = load("cr_ds_thirds_crossfit.json")["verdict"]
    assert v["sel0_meas1"] == v["sel1_meas0"] == 8 and "most direction-selective third shifts more in all eight recordings" in README


def test_acceptance_section():
    """'rung against the paper': each table row and each sentence number re-derived from the three acceptance artifacts."""
    rep = load("rung_acceptance_replay.json")["rows"]
    assert len(rep) == 9
    for r in rep:
        s = r["default_20_seeds"]
        for v in (r["default"]["delta"], r["replay"]["delta"], r["paper"]["delta"]):
            assert f"{v:+.3f}" in README, (r["name"], v)
        assert f"{s['mean']:+.3f} ± {s['sd']:.3f}" in README, r["name"]
        x = max(abs(r["residual_replay"]["delta"]), abs(r["residual_replay"]["within"]))
        assert x < 1e-5, (r["name"], x)                                   # the replay is exact or at float32 rounding
    acc = load("rung_account_imaging.json")["verdict"]
    assert acc["n"] == 8 and f"tracks the shift's size with r = {acc['r_two_terms_delta']:.2f}" in README
    assert f"mean error of {acc['mean_err_finite_heldout']:+.3f}" in README
    hd = load("rung_heads_ioi.json")
    assert hd["paper_seeds"]["replays_llm_ioi"] and f"(AUC {hd['paper_seeds']['AUC_z']:.2f})" in README
    sp = hd["default_spread"]["AUC_z"]
    assert f"the AUC is {sp['mean']:.2f} ± {sp['sd']:.2f}" in README


def test_worked_example_split():
    w = load("worked_example_gt3.json")["decompose"]["delta_split_free"]
    printed(w["D"], "+0.358", 3); printed(w["Tb"] + w["Cb"] + w["Cx"], "-0.127", 3)


def test_runs_on_data_the_paper_never_saw():
    s = load("heldout_allen2p_summary.json")
    b, nb, c = s["release_response"], s["no_baseline"], s["controls"]
    assert b["n"] == 61 and c["n"] == 61 and "61 primary visual cortex experiments" in README
    assert (b["n_cells_min"], b["n_cells_max"]) == (86, 474) and "86 to 474 cells" in README
    for value, text in ((b["orderavg_positive"], "positive in 48 of 61"), (b["orderavg_p05"], "permutations in 49"),
                        (b["orderavg_tb_survives"], "in 47 of those 49"), (b["within_positive"], "positive in 47 of 61"),
                        (b["class_mean_negative"], "negative in 53"), (c["one_shape_negative"], "negative in 56 of 61"),
                        (c["equal_means_positive"], "positive in 57"), (b["even_led"], "orientation-led in 60 of 61"),
                        (nb["alignment_lower_than_release"], "falls in 55 of 61"), (nb["class_mean_positive"], "turns positive in 25")):
        assert re.search(r"\d+", text).group() == str(value) and text in README, (value, text)
    printed(b["orderavg_median"], "+0.07", 2); printed(b["within_median"], "+0.21", 2); printed(b["class_mean_median"], "-0.07", 2)
    printed(b["alignment_median"], "1.38", 2); printed(b["spearman_alignment_class_mean"], "-0.82", 2)
    assert f"{100 * b['pa_floor_share_median']:.0f}%" == "68%" and "median 68%" in README
    printed(nb["alignment_median"], "1.01", 2)
    printed(b["declared_min"], "-0.62", 2); printed(b["declared_max"], "+0.72", 2)
    printed(c["order_sd_median"], "0.19", 2); printed(c["spearman_order_sd_class_logpr_sd"], "0.98", 2)
    printed(c["order_sd_median_one_shape"], "0.05", 2)
    printed(load("allen_order_spread.json")["summary"]["pooled_all"]["order_sd_median"], "0.05", 2)
    # language models
    q = load("heldout_qwen05_world_knowledge32.json")["axes"]["world_knowledge32"]["layers"]
    oa = [l["delta_orderavg"] for l in q]
    assert len(q) == 25 and all(v > 0 for v in oa) and "positive at all 25" in README
    printed(oa[0], "+0.014", 3); printed(oa[-1], "+0.112", 3)
    assert all(l["last_token_arms"]["matched"]["delta_orderavg_median"] > 0 for l in q[13:])
    second = {}
    for tag, name in (("step0", "stage1-step0-tokens0B"), ("step10k", "stage1-step10000-tokens21B"), ("main", "main")):
        L = load(f"heldout_olmo2_1b_{name}.json")["axes"]["world_knowledge32"]["layers"]
        second[tag] = float(np.mean([l["last_token_arms"]["matched"]["delta_orderavg_median"] for l in L[len(L) // 2 + 1:]]))
        if tag == "main":
            assert len(L) == 17 and all(l["delta_orderavg"] > 0 and l["p_two_orderavg"] <= 0.05 for l in L)
    printed(second["step0"], "-0.013", 3); printed(second["step10k"], "+0.032", 3); printed(second["main"], "+0.047", 3)
    m = load("heldout_qwen05_mmlu8_answer.json")["axes"]["mmlu8_answer"]["layers"][1:]
    u = load("heldout_qwen05_mmlu8.json")["axes"]["mmlu8_full"]["layers"]
    assert sum(l["delta_orderavg"] > 0 and l["p_two_orderavg"] <= 0.05 for l in m) == 22 and "at 22\nof the 24 layers" in README
    printed(np.mean([l["delta_orderavg"] for l in m]), "+0.094", 3)
    printed(np.mean([l["delta_orderavg"] for l in u[1:]]), "+0.193", 3)


def test_allen_finite_n_identity_row():
    v = load("allen_finite_n_identity.json")["verdict"]
    assert v["n"] == 167 and v["sign_agree_W2"] == 163 and "all 167 populations, the shift's sign in 163" in README
    s = load("allen_order_spread.json")["summary"]["pooled_all"]
    assert s["n"] == 167
