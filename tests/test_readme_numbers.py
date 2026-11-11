"""Every number README.md and PAPER.md print, recomputed from its artifact and checked at their rounding; every file they name
exists. Run: pytest -q tests"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_canonical")
README = "\n".join(open(os.path.join(ROOT, f)).read() for f in ("README.md", "REFERENCE.md", "PAPER.md"))


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
    # absolute links to this repository (the README renders on PyPI too) name repository files: check those files as well
    text = README.replace("\\*", "*")
    for prefix in ("https://raw.githubusercontent.com/adilamin89/structure-in-the-zoom/main/",
                   "https://github.com/adilamin89/structure-in-the-zoom/blob/main/"):
        text = text.replace(prefix, "")
    names = set(re.findall(r"([A-Za-z0-9_\-*/]+(?:\.[A-Za-z0-9_\-*]+)*\.(?:json|py|sh|png|gif))", text))   # dotted stems: pythia-2.8b
    missing = []
    for n in names:
        if n.startswith(("http", "raw/", "//")) or n in ("X.npy", "r.json", "lt.json", "lm.json", "v1.json", "pairs.json", "my.json",
                                                        "r.png", "lt.png", "v1.png", "strata.json", "gt3.json", "gt3.png",
                                                        "heads.json", "lm.png", "toy.json", "toy.png", "s_meta.json", "o.json"):   # command-line examples
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
    assert f"the AUC is {sp['mean']:.3f} ± {sp['sd']:.3f}" in README


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


def test_speed_paragraph():
    """The README's runtimes and agreement, read back from scripts_canonical/rung_speed.py's artifact."""
    d = load("rung_speed.json")
    g, l, lp = d["gt3"], d["lm"], d["lm_paper"]
    text = " ".join(README.split())
    assert g["p_values_identical"] and l["p_values_identical"]
    assert f"takes {g['cpu_s_new']:.0f} s of one laptop core's time in 1.11, against {g['cpu_s_old'] / 60:.0f} minutes in 1.10" in text
    assert f"returns the same numbers to {g['max_rel_diff']:.0e} with the same permutation p" in text
    assert f"takes {l['cpu_s_new']:.0f} s of one laptop core's time in 1.11, model encoding included ({l['cpu_s_old'] / 60:.0f} minutes in 1.10)" in text
    assert f"in their row-by-row arithmetic, about {lp['cpu_s_new'] / 60:.0f} minutes" in text
def test_within_arm_sentence():
    s = load("meanfree_shift_exact.json")["score"]
    lo, hi = s["meanfree_shift_range"]
    assert s["n_cells"] == 32 and s["R11a_both_nulls_p_le_0.05"] == 32 and lo > 0
    assert f"positive shift on all 32 label-blind halves, {lo:+.2f} to {hi:+.2f}" in " ".join(README.split())


def test_front_page_numbers():
    """The front page's numbers that are not also in REFERENCE.md or PAPER.md, read back from their artifacts in README.md alone."""
    front = " ".join(open(os.path.join(ROOT, "README.md")).read().split())
    w = load("worked_example_gt3_rung.json")
    assert (w["n_samples"], w["n_features"]) == (4442, 11311)
    assert "One recording of 11,311 mouse V1 neurons gives three" in front and "4,442 trials by 11,311 neurons in eight drift directions" in front
    import rung as r
    tm = load("readme_timing.json")                              # scripts_canonical/readme_timing.py, one core each
    assert tm["rung"].endswith(r.__version__) and tm["workers"] == 1
    assert f"Appendix M's command for GT3 (200 permutations) takes {tm['gt3']['cpu_s']:.0f} CPU-seconds of one laptop core" in front
    assert f"it takes {tm['llm']['cpu_s']:.0f} CPU-seconds of one laptop core ({tm['cpu']}), model encoding included" in front
    ok = [r for r in load("stringer_theta_decomposition.json")["rows"] if r.get("status") == "ok"]
    lo, hi = min(r["theta_obs"] for r in ok), max(r["theta_obs"] for r in ok)
    assert len(ok) == 18 and f"Random-stimulus exponents, {lo:.2f} to {hi:.2f} across all 18 imaging recordings, sit on their floors" in front
    assert abs(np.mean([r["shift_obs"] for r in ok])) < 0.01                       # 'sit on their floors': mean shift -0.002
    assert f"sit on their floors (mean shift {np.mean([r['shift_obs'] for r in ok]):+.3f})" in front
    sp = next(r for r in load("neuron_zoom_spatial.json")["rows"] if r["name"].startswith("gratings_drifting_GT3"))
    assert f"so does adding neighbouring neurons ({sp['theta_spatial']:.3f} against a floor of {sp['theta_floor']:.3f} in the recording above)" in front
    assert abs(sp["theta_spatial"] - sp["theta_floor"]) < 0.01
    g3 = load("readme_gt3.json")
    assert (f"memory and time grow with the square of the number of samples (the Gram alone is 8n² bytes, "
            f"{8 * g3['n_samples'] ** 2 / 1e9:.2f} GB for GT3's {g3['n_samples']:,})") in front
    # Allen: the pooled arm's positives (Figure 1), the residualized arm's negatives and always-positive baseline, and the two terms'
    # signs at the rungs' sample counts (W2: overall and among the negatives) and in mean field (W0)
    pooled = [v["delta_dir8"] for s in load("allen_expansion_all_sessions.json")["results"] if s.get("status") == "ok" for v in s["areas"].values()]
    rows = load("allen_finite_n_identity.json")["rows"]
    dm = np.array([q["delta_measured"] for q in rows]); cm = np.array([q["classmean_pred"] for q in rows])
    ag = {k: np.sign(np.array([q[k] for q in rows]) + cm) == np.sign(dm) for k in ("W0", "W2")}
    neg = int((dm < 0).sum())
    assert len(rows) == len(pooled) == 167
    assert f"the two terms, computed on the same trials at the rungs' own sample counts, get the sign in {int(ag['W2'].sum())} of {len(rows)} Allen " \
           f"Neuropixels populations with each direction's temporal-frequency tuning removed, including " \
           f"{int((ag['W2'] & (dm < 0)).sum())} of the {neg} negative ones (always guessing positive gets {len(rows) - neg})" in front
    # XY: the two terms' sign in every lattice and spin-wave ensemble (40 nematic lattices and spin waves, 32 polar)
    xy = load("xy_twoterm.json")
    pol = [v for v in load("xy_polar_twoterm.json")["summary"].values() if isinstance(v, dict)]       # one entry per temperature
    n_pol, ok_pol = sum(v["n"] for v in pol), sum(v["sign_agree_delta"] for v in pol)
    assert xy["sign_agree"] == xy["n"] and ok_pol == n_pol and xy["n"] + n_pol == 72
    assert "the two terms give its sign in all 72 lattice and spin-wave ensembles" in front
    sp = load("rung_heads_ioi.json")["default_spread"]["AUC_z"]
    io = load("llm_ioi.json")
    assert f"at AUC {sp['mean']:.2f} (the mean of ten seeds), beside {io['AUC_probe_acc']:.2f} for a trained linear classifier and " \
           f"{io['AUC_classmean_frac']:.2f} for the heads' class-mean variance share" in front
    rep = next(r for r in load("rung_acceptance_replay.json")["rows"] if r["name"].startswith("GT3"))
    s = rep["default_20_seeds"]
    gt3 = next(r for r in load("orientation_zoom.json")["results"] if r["name"].startswith("gratings_drifting_GT3"))
    assert f"default floor of 20 random ladders, shift {rep['default']['delta']:+.3f}" in front
    assert f"+{rep['default']['delta']:.3f} to {gt3['delta']:+.3f} under different draws ({gt3['delta']:+.3f} in Figure 1)" in front
    assert f"`--paper-floor` replays Table 9's {rep['replay']['delta']:+.3f}" in front and abs(rep["replay"]["delta"] - rep["paper"]["delta"]) < 1e-5
    assert f"the mean over 20 seeds of the default floor, {s['mean']:+.3f} ± {s['sd']:.3f}" in front


def test_front_page_reading():
    """The front page prints `rung summarize` for its run on the paper's recording twice: cut to the exponent's, the declared order's
    and the split's lines (the skipped lines marked by a vertical ellipsis) in the quickstart, and in full (wrapped) in the details
    below it; its glosses, the report's alt text and the clouds animation's alt text read their numbers from the same run
    (data_canonical/readme_gt3.json, rung 1.14.0: the declared order, floor and split equal 1.13's; the order-averaged line reads every
    class order)."""
    import rung as r
    g = load("readme_gt3.json")
    assert g["rung_version"] == r.__version__ and (g["n_samples"], g["n_features"], g["n_classes"]) == (4442, 11311, 8)
    assert g["orderavg_method"] == "exact" and g["n_orders_averaged"] == 40320
    front = open(os.path.join(ROOT, "README.md")).read()
    flat = " ".join(front.split())
    blocks = front.split("$ rung summarize gt3.json\n")
    assert len(blocks) == 3
    cut, full = (b.split("```", 1)[0] for b in blocks[1:])
    L = r.summarize_data(g, verbose=False)["lines"]
    i = next(i for i, l in enumerate(L) if l.startswith("split (Section 5)"))
    assert " ".join(full.split()) == " ".join(" ".join(L).split())
    assert " ".join(cut.split()) == " ".join(" ".join([L[0], L[1], "\u22ee", L[i]]).split())
    fs = g["decompose"]["delta_split_free"]; cm = fs["Tb"] + fs["Cb"] + fs["Cx"]
    assert abs(fs["D"] + cm - g["delta"]) < 1e-9
    R = g["decompose"]["rungs"]
    assert [q["classes"] for q in R] == [1, 2, 3, 4, 6, 8] and "the rungs hold the trials of the first 1, 2, 3, 4, 6 and all 8 directions" in flat
    assert (f"random subsets of the same sizes give {g['theta_floor']:.3f} of the exponent {g['theta_obs']:.3f}, and adding whole "
            f"directions class by class gives the other {g['delta']:+.3f}") in flat
    assert f"`z = +{g['z']:.1f}`: the shift sits {g['z']:.0f} standard deviations above the same ladder with the labels permuted" in flat
    assert abs(g["pr_obs"][-1] - g["pr_floor"][-1]) < 1e-9 * g["pr_obs"][-1]
    assert (f"One direction's trials span {g['pr_obs'][0]:.0f} dimensions where random trials from all eight span "
            f"{g['pr_floor'][0]:.0f}; both ladders end at the full set's {g['pr_obs'][-1]:.0f}, so the direction ladder climbs further") in flat
    assert f"carry the shift (the within-class part D, {fs['D']:+.3f}), and the class means pull it down (the class-mean part Tb + Cb + Cx, {cm:+.3f})" in flat
    assert f"default floor of 20 random ladders, shift {g['delta']:+.3f}" in flat and g["n_floor_draws"] == 20
    # the clouds animation's ladders: PR(W) from the rung traces, the floor's from the rung's D (as the generator draws them)
    tr = np.array([q["traces_obs"] for q in R]); ldw = np.log(tr[:, 1] ** 2 / tr[:, 3]); ldw_f = ldw - np.array([q["D"] for q in R])
    lo, lf = np.array([q["log_pr_obs"] for q in R]), np.array([q["log_pr_floor"] for q in R])
    span = lambda a: f"{np.exp(a[0]):.0f} to {np.exp(a[-1]):.0f}"
    assert (f"fluctuations alone climbs from {span(ldw)} on the direction ladder against {span(ldw_f)} on its floor, a within-class "
            f"part of {fs['D']:+.3f}; with the class means the whole rung climbs from {span(lo)} against {span(lf)}, the shift of "
            f"{g['delta']:+.3f}; class means {cm:+.3f}") in flat
    x = np.log([q["size"] for q in R]); slope = lambda y: float(np.polyfit(x, y, 1)[0])
    assert abs(slope(ldw - ldw_f) - fs["D"]) < 1e-9 and abs(slope(lo - lf) - g["delta"]) < 1e-9


def test_front_page_hero():
    """The first screen's animation (media/hero.gif) and its alt text: the ladders and slopes of the front page's run, the exponent
    as floor plus shift, the shift as its two parts (the generator, make_hero_anim.py, asserts the same on every frame it draws)."""
    g = load("readme_gt3.json")
    front = " ".join(open(os.path.join(ROOT, "README.md")).read().split())
    assert os.path.exists(os.path.join(ROOT, "media", "hero.gif"))
    fs = g["decompose"]["delta_split_free"]; cm = fs["Tb"] + fs["Cb"] + fs["Cx"]
    x = np.log(g["rung_sizes"]); slope = lambda y: float(np.polyfit(x, np.log(y), 1)[0])
    assert abs(slope(g["pr_obs"]) - g["theta_obs"]) < 1e-9 and abs(slope(g["pr_floor"]) - g["theta_floor"]) < 1e-9
    assert abs(g["pr_obs"][-1] - g["pr_floor"][-1]) < 1e-9 * g["pr_obs"][-1]          # 'meets it at the full set'
    assert abs(g["theta_obs"] - g["theta_floor"] - g["delta"]) < 1e-9 and abs(fs["D"] + cm - g["delta"]) < 1e-9
    assert (f"climbs from {g['pr_obs'][0]:.0f} to {g['pr_obs'][-1]:.0f} dimensions, while its floor, random trials with the labels "
            f"ignored, starts at {g['pr_floor'][0]:.0f} and meets it at the full set: slopes {g['theta_obs']:.3f} and "
            f"{g['theta_floor']:.3f}. The exponent {g['theta_obs']:.3f} is the floor {g['theta_floor']:.3f} plus a shift of "
            f"{g['delta']:+.3f}, which the fluctuations within each direction ({fs['D']:+.3f}) and the class means ({cm:+.3f}) add up to") in front


def test_front_page_toy():
    """The quickstart toy, run exactly as the README prints it (its Python block, then `rung data X.npy labels.npy --decompose`),
    prints the reading the README shows, and runs in seconds (CPU time; wall time in a sandbox can be capped)."""
    import resource
    import subprocess
    import sys
    import tempfile
    import rung as r
    front = open(os.path.join(ROOT, "README.md")).read()
    py = front.split("```python\n", 1)[1].split("```", 1)[0]
    assert "rung data X.npy labels.npy --decompose --out toy.json\nrung summarize toy.json" in front
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "PYTHONPATH": ROOT}
        t0 = resource.getrusage(resource.RUSAGE_CHILDREN)
        subprocess.run([sys.executable, "-c", py], cwd=tmp, check=True, env=env)
        subprocess.run([sys.executable, "-m", "rung", "data", "X.npy", "labels.npy", "--decompose", "--out", "toy.json"], cwd=tmp,
                       check=True, env=env, capture_output=True)
        t1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        toy = json.load(open(os.path.join(tmp, "toy.json")))
    cpu = t1.ru_utime + t1.ru_stime - t0.ru_utime - t0.ru_stime
    assert cpu < 10, cpu
    L = r.summarize_data(toy, verbose=False)["lines"]
    sp = next(l for l in L if l.startswith("split (Section 5)"))
    block = front.split("rung summarize toy.json", 1)[1].split("```text\n", 1)[1].split("```", 1)[0]
    assert " ".join(block.split()) == " ".join(" ".join([L[0], "\u22ee", sp]).split())
    fs = toy["decompose"]["delta_split_free"]
    cm, sh = fs["Tb"] + fs["Cb"] + fs["Cx"], toy["decompose"]["shuffle"]["delta_split"]
    cm_shuf = sh["Tb"] + sh["Cb"] + sh["Cx"]
    assert abs(cm - cm_shuf) < 0.01                                                   # 'matches what the same split gives with the labels shuffled'
    assert (f"of the shift {toy['delta']:+.3f}, the within-class part D is {fs['D']:+.3f}, and the class-mean part, {cm:+.3f}, matches "
            f"what the same split gives with the labels shuffled ({cm_shuf:+.3f})") in " ".join(front.split())
    assert [q["classes"] for q in toy["decompose"]["rungs"]] == [1, 2, 3, 4, 6, 8] and "the samples of the first 1, 2, 3, 4, 6 and all 8 classes" in front
    assert fs["D"] > 0.8 * toy["delta"]                       # 'the reading says so': the toy's classes differ only in their fluctuations


def test_front_page_lm():
    """The GPT-2 animation (media/lm_heads.gif) and its alt text: the battery's counts, the run drawn (the paper's seeds, which replay
    llm_ioi.json) and its AUC, the boxed heads; the text's AUCs are test_front_page_numbers'."""
    hi, io = load("rung_heads_ioi.json"), load("llm_ioi.json")
    front = " ".join(open(os.path.join(ROOT, "README.md")).read().split())
    assert os.path.exists(os.path.join(ROOT, "media", "lm_heads.gif"))
    assert hi["paper_seeds"]["replays_llm_ioi"] and len(hi["heads_paper_seeds"]) == 144 and len(io["positive_set"]) == 11
    assert io["n_prompts"] == len(io["prompts"]) == 128 and io["frac_logit_diff_positive"] == 1.0
    assert sorted(hi["paper_seeds"]["rank_negative"]) == ["10.7", "11.10"]
    names = json.load(open(os.path.join(ROOT, "axes", "ioi_names.json")))
    assert list(names)[io["labels"][0]] == "Mary" and len(names) == 8
    assert (f"Three of the 128 prompts, the first '{io['prompts'][0]}', each with the name to be said; the right name's logit beats the "
            "other name's on all 128") in front
    assert "the two negative name movers dashed, 3rd and 4th by z" in front and hi["paper_seeds"]["rank_negative"] == {"10.7": 3, "11.10": 4}
    assert f"in the run drawn the solid-boxed heads outrank the others at AUC {hi['paper_seeds']['AUC_z']:.2f}" in front
    ranks = [q for sd in hi["default_seeds"] for q in sd["rank_negative"].values()]
    assert len(hi["default_seeds"]) == 10 and max(ranks) == 7
    assert "The two negative name movers, which push against the answer, rank in the top seven of the 144 heads in all ten seeds" in front


def test_front_page_xy_sentence():
    """'positive while polar order holds, negative from T = 0.9 to 1.1 as it melts, near zero above' (the J1/J2 = 0.42 curve, L = 32,
    the XY animation's data): positive with its 95% interval above zero up to 0.85, below zero from 0.9 to 1.1, and within 0.01 of zero
    from 1.2 (the paper prints two decimals)."""
    pts = load("lattice_ground_truth_score.json")["xy"]["points"]
    T = sorted(float(k.split("_T")[1]) for k in pts if k.startswith("L32_J0.42_"))
    P = {t: pts[f"L32_J0.42_T{t}"] for t in T}
    assert all(P[t]["equilibrated"] for t in T)
    assert all(P[t]["ci95"][0] > 0 for t in T if t <= 0.85)
    assert [t for t in T if P[t]["ci95"][1] < 0 and abs(P[t]["delta_director"]) >= 0.01] == [0.9, 1.0, 1.1]
    assert all(abs(P[t]["delta_director"]) < 0.01 for t in T if t >= 1.2)
    assert "positive while polar order holds, negative from T = 0.9 to 1.1 as it melts, near zero above" in " ".join(
        open(os.path.join(ROOT, "README.md")).read().split())


def test_front_page_macaque():
    """The macaque sentence: two arrays positive, one negative at its printed value, the two terms' sign in all three."""
    front = " ".join(open(os.path.join(ROOT, "README.md")).read().split())
    mac = load("macaque_pvc11_twoterm.json")["rows"]
    neg = [q["measured_delta"] for q in mac if q["measured_delta"] < 0]
    assert len(mac) == 3 and len(neg) == 1 and all(np.sign(q["delta_pred"]) == np.sign(q["measured_delta"]) for q in mac)
    assert (f"With one average fluctuation overlap for all pairs of classes (mean field) they also give the signs of all three macaque V1 arrays from a third laboratory (Smith and Kohn), one of them negative "
            f"({neg[0]:.2f})") in front


def test_front_page_layers():
    """The Pythia layers paragraph and animation (media/lm_layers.gif): topics above their permutation null at every layer of
    Pythia-2.8B with the order-averaged shift at the embedding and the last layer; TruthfulQA's count past the embedding with every
    question ending on a question mark; BLiMP below its permutation null and inside the within-pair swap null (llm_score L4)."""
    front = " ".join(open(os.path.join(ROOT, "README.md")).read().split())
    assert os.path.exists(os.path.join(ROOT, "media", "lm_layers.gif"))
    bat = load("llm_battery_pythia-2.8b-deduped.json")
    tl, bl = bat["batteries"]["world_knowledge32"]["layers"], bat["batteries"]["blimp_B"]["layers"]
    t_n = sum(r["perm"]["p_one"] <= 0.05 for r in tl)
    q = load(os.path.join("llm_benchmark", "llm_benchmark_score.json"))["pythia-2.8b-deduped/tqa_category_26"]
    L4 = load("llm_score.json")["L4"]["pythia-2.8b-deduped"]
    below = sum(r["perm"]["p_two"] <= 0.05 and r["dbar50_run37"] < r["perm"]["null_mean"] for r in bl)
    assert len(tl) == t_n == 33 and set(q["distinct_last_tokens"]) <= {1, 2} and L4["swap_absorbs"] == 33
    assert (f"eight topics lift all {t_n} layers above their permutation null, more so late (order-averaged shift "
            f"{tl[0]['dbar50_run37']:+.2f} at the embedding, {tl[-1]['dbar50_run37']:+.2f} at the last layer), and TruthfulQA's "
            f"categories lift {q['B1_past_embedding'].split('/')[0]} of the 32 layers past the embedding although every question ends "
            f"on a question mark") in front
    assert (f"BLiMP's grammaticality labels sit far below their permutation null at {below} of 33 layers and inside the null that "
            f"swaps labels within each sentence pair at all {L4['swap_absorbs']}") in front


def test_language_model_prompt_clouds():
    """The prompt-clouds paragraph and animation (README v4): Pythia-2.8B topics against rung's shape surrogate, every number from
    data_canonical/lm_shape_surrogate_pythia-2.8b.json (scripts_canonical/lm_shape_surrogate.py)."""
    import rung as r
    front = " ".join(open(os.path.join(ROOT, "README.md")).read().split())
    d = load("lm_shape_surrogate_pythia-2.8b.json")
    cov, sm, l24 = d["frames"]["cov"], d["summary"]["cov"], d["layer24"]
    r24 = cov[24]
    assert d["rung"] == r.__version__ and d["n_per_class"] == 32 and len(cov) == 33 == sm["n_layers"]
    assert sm["net_D_positive"] == 33 == d["summary"]["z"]["net_D_positive"]          # in both frames
    assert max(sm["surrogate_share_of_shift_max"], d["summary"]["z"]["surrogate_share_of_shift_max"]) < 0.5
    assert (f"layer 24's shift falls from {r24['delta']:+.2f} to {r24['surrogate_delta']:+.2f}. At all 33 layers, the embedding included, the split "
            f"net of the surrogate (in the declared order, rung's default) puts what the topics' own fluctuations add at {sm['net_D_min']:+.3f} "
            f"to {sm['net_D_max']:+.3f}, and at 32 prompts per topic the class means add almost nothing beyond the surrogate "
            f"({sm['net_class_means_min']:+.3f} to {sm['net_class_means_max']:+.3f})") in front
    dims = r24["class_dim_split"]
    n_below = sum(v > d["n_per_class"] for v in dims)
    assert f"{ {7: 'Seven'}[n_below]} of the eight topics hold fewer prompts (32) than their estimated fluctuation dimension ({min(dims):.0f} to {max(dims):.0f} at layer 24, from split halves)" in front
    assert f"shift of {r24['delta']:+.2f}; a control that keeps every topic's centre" in front and f"one fluctuation shape has a mean shift of {r24['surrogate_delta']:+.2f}" in front
    assert (round(100 * l24["plane_share_of_class_means"]), round(100 * l24["plane_share_of_all"])) == (44, 7)
    assert os.path.exists(os.path.join(ROOT, "media", "lm_clouds.gif")) and os.path.exists(os.path.join(ROOT, "scripts_canonical", "lm_shape_surrogate.py"))


def test_what_to_do_next_quotes_rung():
    """The README's 'What to do next' table quotes rung's own printed words; each must still be in rung.py."""
    import rung as r
    src = open(r.__file__).read()
    front = open(os.path.join(ROOT, "README.md")).read()
    table = front.split("### What to do next", 1)[1].split("### From Python", 1)[0]
    quoted = [q for row in table.splitlines() if row.startswith("| `") for q in re.findall(r"`([^`]+)`", row.split("|")[1])]
    assert len(quoted) >= 10
    for q in quoted:
        for part in q.split("..."):
            part = part.strip()
            assert not part or part in src, part
    for opt in re.findall(r"`(--[a-z-]+)", table):
        assert f'"{opt}"' in src, opt


def test_install_line_and_python_block(tmp_path):
    """The install line names this release's tag, and the From Python block runs as printed on the quickstart's toy."""
    import contextlib
    import io
    import rung as r
    front = open(os.path.join(ROOT, "README.md")).read()
    assert f'pip install "rung @ git+https://github.com/adilamin89/structure-in-the-zoom@v{r.__version__}"' in front
    toy = front.split("```python\n", 1)[1].split("```", 1)[0]
    block = front.split("### From Python", 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        exec(toy, {})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(block, {})
    finally:
        os.chdir(cwd)
    out = buf.getvalue()
    assert out.startswith("theta_obs = +0.511 = floor +0.271 + shift +0.240") and "split (Section 5): within-class part D" in out
    assert "`decompose(X, y, seed=0, n_shuffle=5)` repeats the command line's split" in " ".join(front.split())
    X, y = np.load(tmp_path / "X.npy"), np.load(tmp_path / "labels.npy")
    printed_d = re.search(r"within-class part D \+([0-9.]+) \+ class-mean part", front.split("rung summarize toy.json", 1)[1]).group(1)
    assert f"{r.decompose(X, y, seed=0, n_shuffle=5)['delta_split_free']['D']:.3f}" == printed_d


def test_rung_114_lines():
    """README v5 (rung 1.14): rule 8's raw and net split at Pythia-2.8B's layer 24 from the surrogate artifact; GT3's declared and
    order-averaged shifts over eight wall positions from data_canonical/continuous_gratings.json; the ORCID iD and CITATION.cff."""
    import rung as r
    front = " ".join(open(os.path.join(ROOT, "README.md")).read().split())
    l24 = load("lm_shape_surrogate_pythia-2.8b.json")["frames"]["cov"][24]
    assert f"the raw split gives the within-class part {l24['D']:+.3f} and the net split {l24['net_D']:+.3f}" in front
    g3 = next(q for q in load("continuous_gratings.json")["rows"] if q["name"].startswith("gratings_drifting_GT3"))["origins_raw"]
    s = g3["summary"]
    assert g3["n_origins"] == 8 and not g3["residualize"]
    assert (f"on GT3 the declared shift runs {s['delta']['min']:+.2f} to {s['delta']['max']:+.2f} over the wall positions while the "
            f"order-averaged shift stays within {s['delta_orderavg']['min']:+.2f} to {s['delta_orderavg']['max']:+.2f}") in front
    assert "[ORCID 0000-0001-6570-7718](https://orcid.org/0000-0001-6570-7718)" in front
    cff = open(os.path.join(ROOT, "CITATION.cff")).read()
    assert f"version: {r.__version__}" in cff and cff.count('orcid: "https://orcid.org/0000-0001-6570-7718"') == 2


def test_rung_114_reference_numbers():
    """REFERENCE.md's 1.14 sentences: the exact order average on GT3 against its random-order check (readme_gt3.json), the union
    count at eight classes, and the default net split at Pythia-2.8B's layer 24 (lm_shape_surrogate_pythia-2.8b.json)."""
    import rung as r
    ref = " ".join(open(os.path.join(ROOT, "REFERENCE.md")).read().split())
    g = load("readme_gt3.json")
    assert len(r._rung_subsets(8, (1, 2, 3, 4, 6, 8))[0]) == 191 and g["n_orders_averaged"] == 40320
    assert "average over the 40,320 orders of eight classes is a weighted sum over the 191 class unions" in ref
    assert (f"on GT3 the exact value is {g['delta_orderavg']:+.3f}, the random orders gave {g['delta_orderavg_random']:+.3f} +- "
            f"{g['delta_orderavg_random_se']:.3f}") in ref
    l24 = load("lm_shape_surrogate_pythia-2.8b.json")["frames"]["cov"][24]
    assert f"raw within-class part is {l24['D']:+.3f} and the net one {l24['net_D']:+.3f}" in ref
