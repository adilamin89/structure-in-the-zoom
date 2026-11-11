"""The extended version's new numbers, read back from the artifacts in v3/ (the text rounds them as printed here)."""
import json
from pathlib import Path

import numpy as np

V3 = Path(__file__).resolve().parent.parent / "v3"


def load(name):
    return json.load(open(V3 / name))


def rng(rows, key):
    v = [r[key] for r in rows]
    return round(min(v), 3), round(max(v), 3)


def test_lm_split_orderavg():
    o = load("lm_split_orderavg.json")
    assert o["n_orders"] == 20 and o["n_bias"] == 10
    cov, z = o["frames"]["cov"], o["frames"]["z"]
    assert rng(cov, "net_D") == (0.030, 0.067) and rng(z, "net_D") == (0.029, 0.066)
    assert all(r["net_D"] > 0 for r in cov + z) and len(cov) == len(z) == 33
    assert min(r["net_D"] / (r["net_D_sd"] / np.sqrt(20)) for r in cov + z) >= 3.4
    assert rng(cov, "net_cm") == (-0.004, 0.007) and rng(z, "net_cm") == (-0.008, 0.004)
    dims = [d for r in cov for d in r["class_dim_split"]]
    assert (round(min(dims)), round(max(dims)), round(float(np.median(dims)))) == (23, 289, 111)


def test_lm_pooled_control_and_declared_split():
    p = load("lm_pooled_control_recentered.json")["frames"]
    assert rng(p["cov"], "delta_orderavg") == (0.021, 0.093) and rng(p["cov"], "pooled_orderavg") == (-0.006, 0.019)
    assert rng(p["z"], "delta_orderavg") == (0.025, 0.073) and rng(p["z"], "pooled_orderavg") == (-0.006, 0.007)
    b = load("lm_split_bias.json")["frames"]
    assert rng(b["cov"], "net_D") == (0.039, 0.110) and rng(b["z"], "net_D") == (0.034, 0.069)
    assert rng(b["cov"], "excess_D") == (0.038, 0.109) and rng(b["z"], "excess_D") == (0.037, 0.058)


def test_cyclic_starts():
    rows = load("start_probe_all8.json")["rows"]
    starts = [v["delta"] for r in rows.values() for v in r["starts"].values()]
    assert len(starts) == 128 and all(s > 0 for s in starts)
    sds = [r["summary"]["starts_sd"] for r in rows.values()]
    assert (round(min(sds), 2), round(max(sds), 2)) == (0.04, 0.08)
    g = rows["Drifting GT3"]["summary"]
    assert round(g["paper_walls_declared"], 3) == 0.231 and (round(g["starts_min"], 3), round(g["starts_max"], 3)) == (0.229, 0.465)
    assert (round(g["walls_declared_min"], 2), round(g["walls_declared_max"], 2)) == (0.23, 0.42)
    assert (round(g["walls_orderavg_min"], 2), round(g["walls_orderavg_max"], 2)) == (0.29, 0.33)
    assert max(r["summary"]["walls_orderavg_sd"] for r in rows.values()) <= 0.015


def test_freedman_lane_one_half_and_macaque():
    f = load("fl_one_half.json")
    assert round(f["unrestricted"]["p_one"], 2) == 0.04 and round(f["epoch"]["p_one"], 2) == 0.08
    m = {r["monkey"]: r for r in load("a3_macaque_finite_n.json")}
    assert round(m[2]["total_fin"], 3) == -0.346 and round(m[2]["delta_200"], 3) == -0.357
    assert (round(m[1]["total_fin"], 3), round(m[3]["total_fin"], 3)) == (0.087, 0.243)
    assert (round(m[1]["delta_200"], 3), round(m[3]["delta_200"], 3)) == (0.152, 0.249)


def test_comparison_invariances():
    for name in ("g1_invariance.json", "g1_invariance_ms0.15.json"):      # exact invariances; D rises when fluctuations rotate apart
        g = load(name)
        assert g["A1"]["pass"] and g["A3"]["pass"]
    v = load("g1_invariance_ms0.15.json")["values"]
    for k in ("mft_radius", "mft_dimension", "beta"):                  # what the rotation keeps; CKA carries PR(C), which it moves
        assert abs(v["T2"][k] - v["base"][k]) <= 1e-3 * abs(v["base"][k])
    r = load("g1_invariance.json")["values"]                             # well-separated classes: RSA does not move
    assert abs(r["T2"]["rsa"] - r["base"]["rsa"]) < 1e-9
    assert (round(v["base"]["D_orderavg"], 3), round(v["T2"]["D_orderavg"], 3)) == (-0.034, 0.051)
    assert (round(v["base"]["sim_capacity"], 2), round(v["T2"]["sim_capacity"], 2)) == (0.37, 0.19)
    assert round(v["base"]["mft_capacity"], 3) == round(v["T2"]["mft_capacity"], 3) == 0.245


def test_nongauss_floor_log():
    lines = (V3 / "nongauss_floor_check.log").read_text().splitlines()
    for line in lines:
        sim, formula = float(line.split("PR ratio sim ")[1].split()[0]), float(line.split("formula ")[1].split()[0])
        assert abs(sim - formula) / sim < 0.01
