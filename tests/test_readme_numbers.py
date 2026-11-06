"""Every number README.md prints, recomputed from its artifact and checked at the README's rounding; every file the README names
exists. Run: pytest -q tests"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_canonical")
README = open(os.path.join(ROOT, "README.md")).read()


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
    names = set(re.findall(r"([A-Za-z0-9_\-*/]+\.(?:json|py|sh|png))", README))
    missing = []
    for n in names:
        if n.startswith(("http", "raw/")) or n in ("X.npy", "r.json", "lt.json", "lm.json", "v1.json", "pairs.json", "my.json",
                                                        "r.png", "lt.png", "v1.png", "strata.json"):   # command-line examples
            continue
        hits = [p for base in ("", "scripts_canonical", "data_canonical", "figures_canonical", "tests")
                for p in glob.glob(os.path.join(ROOT, base, n))]
        if not hits:
            missing.append(n)
    assert not missing, sorted(missing)
