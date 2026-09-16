"""Tests for rung (numpy only; no network). Run: pytest -q tests"""
import json, os, subprocess, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rung as tz


def _gaussian_classes(n=640, d=96, C=8, sep=0.8, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(C), n // C)
    means = rng.normal(size=(C, d)) * sep
    return means[labels] + rng.normal(size=(n, d)), labels


def test_decomposition_is_an_identity():
    X, labels = _gaussian_classes()
    r = tz.zoom(X, labels, n_perm=0, k_orders=5, n_floor_draws=5)
    assert abs(r["theta_obs"] - (r["theta_floor"] + r["delta"])) < 1e-12
    assert len(r["pr_obs"]) == len(r["rung_sizes"]) == len(r["pr_floor"])


def test_structured_labels_certify_and_shuffled_do_not():
    X, labels = _gaussian_classes()
    r = tz.zoom(X, labels, n_perm=100, k_orders=10, n_floor_draws=5, seed=1)
    assert r["p_two"] <= 0.05 and abs(r["z"]) > 3
    rng = np.random.default_rng(3)
    r0 = tz.zoom(X, rng.permutation(labels), n_perm=100, k_orders=10, n_floor_draws=5, seed=1)
    assert r0["p_two"] > 0.05


def test_strata_null_runs_and_is_reported():
    X, labels = _gaussian_classes()
    strata = np.arange(len(labels)) % 4
    r = tz.zoom(X, labels, n_perm=50, k_orders=5, n_floor_draws=5, strata=strata)
    assert "strat_p_two" in r and 0 < r["strat_p_two"] <= 1


def test_cli_data_writes_json_plot_and_summary(tmp_path):
    X, labels = _gaussian_classes(n=320, d=48)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "labels.npy", labels)
    out, png = tmp_path / "r.json", tmp_path / "r.png"
    cmd = [sys.executable, os.path.join(os.path.dirname(tz.__file__), "rung.py"), "data",
           str(tmp_path / "X.npy"), str(tmp_path / "labels.npy"), "--n-perm", "30", "--k-orders", "5",
           "--out", str(out), "--plot", str(png)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    d = json.load(open(out)); assert "delta" in d and png.exists() and png.stat().st_size > 1000
    rep = tz.summarize_data(d, verbose=False)
    assert "certified" in rep["verdict"] and rep["lines"]


def test_build_axis_offline_with_strata():
    rng = np.random.default_rng(0)
    rows = [{"text": f"sentence number {i} about topic {i % 10} " + "x" * 20, "label": f"L{i % 10}", "topic": f"t{i % 3}"}
            for i in range(400)]
    axis, strata = tz.build_axis(rows, "text", "label", n_classes=8, n_per_class=16, strata_field="topic", seed=0)
    assert len(axis) == 8 and all(len(v) == 16 for v in axis.values())
    assert set(strata) == set(axis) and all(len(v) == 16 for v in strata.values())
    with pytest.raises(ValueError):
        tz.build_axis(rows[:40], "text", "label", n_classes=8, n_per_class=16)


def test_theta_zoom_alias_is_rung():
    import theta_zoom
    assert theta_zoom.zoom is tz.zoom and theta_zoom.main is tz.main


def test_spectrum_flags_a_rogue_dimension_and_standardize_repairs_it():
    X, labels = _gaussian_classes()
    Xr = X.copy(); Xr[:, 0] += 200.0 * np.random.default_rng(5).normal(size=len(X))   # one rogue dimension
    sp = tz.spectrum(Xr)
    assert sp["top1_eig_frac"] > 0.9 and sp["d_eff_full"] < 2
    blind = tz.zoom(Xr, labels, n_perm=0, k_orders=3, n_floor_draws=3)
    fixed = tz.zoom(Xr, labels, n_perm=0, k_orders=3, n_floor_draws=3, standardize=True)
    assert abs(fixed["delta"]) > abs(blind["delta"])


def _circle_classes(n_per=60, d=64, harmonic=1, seed=0):
    """Eight classes on a circle with means at harmonic 1 (a dipole code: every
    class has its own mean) or harmonic 2 (an even code: a class and its antipode
    share a mean), isotropic within-class noise."""
    rng = np.random.default_rng(seed)
    C = 8
    labels = np.repeat(np.arange(C), n_per)
    u1, u2 = rng.normal(size=d), rng.normal(size=d)
    ang = 2 * np.pi * np.arange(C) / C * harmonic
    means = 2.5 * (np.cos(ang)[:, None] * u1 + np.sin(ang)[:, None] * u2) / np.sqrt(d)
    return means[labels] + rng.normal(size=(len(labels), d)), labels


def test_deficit_ladder_and_the_stall_test():
    Xd, labels = _circle_classes(harmonic=1)
    Xe, _ = _circle_classes(harmonic=2)
    pairs = {c: (c + 4) % 8 for c in range(8)}
    rd = tz.zoom(Xd, labels, n_perm=0, k_orders=3, n_floor_draws=8, antipode=pairs)
    re = tz.zoom(Xe, labels, n_perm=0, k_orders=3, n_floor_draws=8, antipode=pairs)
    for r in (rd, re):
        assert r["rung_classes"] == [1, 2, 3, 4, 6, 8] and len(r["deficit"]) == 6
        assert abs(r["deficit"][-1]) < 1e-9          # the top rung is the floor's own set
        assert r["late_rung_classes"] == 4 and r["antipode_first_half_has_no_pair"]
    # the even code has every class mean by four classes, so its climb is complete
    # there; the dipole code still has a deficit at four classes
    assert re["late_fraction"] < rd["late_fraction"]
    assert abs(re["deficit"][3]) < abs(rd["deficit"][3])
    with pytest.warns(UserWarning):
        tz.zoom(Xd, labels, n_perm=0, k_orders=2, n_floor_draws=3, antipode={0: 1, 1: 0})


def test_split_by_runs_matched_subsets(tmp_path):
    X, labels = _gaussian_classes(n=320, d=48)
    score = np.arange(48, dtype=float)
    sp = tz.split_by(X, labels, score, n_perm=0, k_orders=3, n_floor_draws=4)
    assert sp["n_features_per_subset"] == 16 and set(sp["subsets"]) == {"top", "bottom", "random"}
    assert all(r["n_features"] == 16 for r in sp["subsets"].values())
    assert sp["subsets"]["top"]["median_score"] > sp["subsets"]["bottom"]["median_score"]
    with pytest.raises(ValueError):
        tz.split_by(X, labels, score[:10], n_perm=0)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "labels.npy", labels); np.save(tmp_path / "score.npy", score)
    out = tmp_path / "r.json"
    cmd = [sys.executable, os.path.join(os.path.dirname(tz.__file__), "rung.py"), "data",
           str(tmp_path / "X.npy"), str(tmp_path / "labels.npy"), "--n-perm", "10", "--k-orders", "3",
           "--split-by", str(tmp_path / "score.npy"), "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    d = json.load(open(out)); assert "split_by" in d and "deficit" in d and "late_fraction" in d
    assert "share of the climb" in "\n".join(tz.summarize_data(d, verbose=False)["lines"])


def test_decompose_is_exact_and_separates_private_from_shared():
    rng = np.random.default_rng(0); n_c, N, K = 30, 200, 8
    lab = np.repeat(np.arange(K), n_c); W = rng.standard_normal((N, 5))
    shared = np.concatenate([rng.standard_normal((n_c, 5)) @ W.T + 0.3 * rng.standard_normal((n_c, N)) for _ in range(K)])
    private = np.concatenate([rng.standard_normal((n_c, 5)) @ rng.standard_normal((N, 5)).T + 0.3 * rng.standard_normal((n_c, N)) for _ in range(K)])
    scaled = np.concatenate([(0.5 + 0.1 * c) * (rng.standard_normal((n_c, 5)) @ W.T + 0.3 * rng.standard_normal((n_c, N))) for c in range(K)])
    rs, rp, rc = (tz.decompose(X, lab, n_null=5, n_shuffle=3) for X in (shared, private, scaled))
    for r in (rs, rp, rc):
        assert r["max_sum_check"] < 1e-9 and r["max_regroup_check"] < 1e-9 and abs(r["delta_split_check"]) < 1e-9
        assert r["max_pr_identity_rel"] < 1e-4 and len(r["rungs"]) == 6 and "at_four" in r
    # private modes: a large dimension term and a private slope near the block-mixture value; shared modes: small
    # (the sampling baseline at thirty trials per class in 200 dimensions is removed by the permutations, to noise)
    assert rp["excess"]["delta_split"]["D"] > 0.4 and abs(rs["excess"]["delta_split"]["D"]) < 0.2
    assert rp["excess"]["delta_split"]["D"] > 3 * abs(rs["excess"]["delta_split"]["D"])
    assert rp["excess"]["private_slope_dim"] > 0.4 and abs(rs["excess"]["private_slope_dim"]) < 0.25
    # a class-dependent scale moves A and P by the same amount with opposite signs and leaves D near zero
    assert abs(rc["excess"]["delta_split"]["S"]) > 0.5 and abs(rc["excess"]["delta_split"]["D"]) < 0.1
    assert abs(rc["excess"]["delta_split"]["A"] - rc["excess"]["delta_split"]["S"] - rc["excess"]["delta_split"]["Tb"]) < 1e-9


def test_cli_decompose_reports_the_split(tmp_path):
    X, labels = _gaussian_classes(n=320, d=48)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "labels.npy", labels)
    out = tmp_path / "r.json"
    cmd = [sys.executable, os.path.join(os.path.dirname(tz.__file__), "rung.py"), "data",
           str(tmp_path / "X.npy"), str(tmp_path / "labels.npy"), "--n-perm", "0", "--k-orders", "2",
           "--decompose", "--n-shuffle", "2", "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "regrouped: D" in res.stdout and "net of 2 label permutations" in res.stdout
    d = json.load(open(out)); assert "decompose" in d and len(d["decompose"]["rungs"]) == 6 and "excess" in d["decompose"]


def test_sectors_recovers_a_planted_kernel_exactly():
    # eight classes on a circle; class-mean vectors built from four orthonormal,
    # zero-mean feature directions so that their correlation is exactly
    # a + q1 cos(d) + q2 cos(2d): the DFT must return q1, q2 and nothing else
    d, K = 40, 8
    rng = np.random.default_rng(0)
    B = rng.normal(size=(d, 4)); B -= B.mean(axis=0, keepdims=True)   # zero mean across features
    Q, _ = np.linalg.qr(B)
    q1, q2 = 0.3, 0.7
    th = 2 * np.pi * np.arange(K) / K
    M = (np.sqrt(q1) * (np.cos(th)[:, None] * Q[:, 0] + np.sin(th)[:, None] * Q[:, 1])
         + np.sqrt(q2) * (np.cos(2 * th)[:, None] * Q[:, 2] + np.sin(2 * th)[:, None] * Q[:, 3]))
    labels = np.repeat(np.arange(K), 5)
    X = M[labels] + 1e-9 * rng.normal(size=(len(labels), d))
    sc = tz.sectors(X, labels)
    c = sc["coefficients"]
    assert abs(c["q1"] - q1) < 1e-6 and abs(c["q2"] - q2) < 1e-6
    assert abs(c["q3"]) < 1e-6 and abs(c["q4"]) < 1e-6 and abs(c["a"]) < 1e-6
    assert abs(sc["balance"] - q2 / q1) < 1e-5 and abs(sc["A_even"] - q2) < 1e-6 and abs(sc["A_odd"] - q1) < 1e-6
    assert set(sc["named"]) == {"a", "c1", "b2", "c3", "b4"}
    # adjacent classes cohere at q1 cos 45 + q2 cos 90, antipodal at -q1 + q2: the
    # quadrupole-dominant kernel makes the antipodal pair the more coherent one
    assert abs(sc["C_adjacent"] - (q1 * np.cos(np.pi / 4))) < 1e-6
    assert abs(sc["C_antipodal"] - (q2 - q1)) < 1e-6 and sc["order_prediction"] == "antipodal"
    with pytest.raises(ValueError):
        tz.sectors(X[labels < 2], labels[labels < 2])


def test_declared_order_presets_and_the_antipodal_ladder():
    classes = np.arange(8)
    assert tz.declared_order(classes) == list(range(8)) == tz.declared_order(classes, "sequential")
    assert tz.declared_order(classes, "antipodal") == [0, 4, 1, 5, 2, 6, 3, 7]
    assert tz.declared_order(np.array([10, 20, 30]), [30, 10, 20]) == [2, 0, 1]
    with pytest.raises(ValueError):
        tz.declared_order(np.arange(7), "antipodal")
    with pytest.raises(ValueError):
        tz.declared_order(classes, [0, 1, 2])
    Xd, labels = _circle_classes(harmonic=1)
    rs = tz.zoom(Xd, labels, n_perm=0, k_orders=3, n_floor_draws=6, order="sequential")
    ra = tz.zoom(Xd, labels, n_perm=0, k_orders=3, n_floor_draws=6, order="antipodal")
    assert rs["order"] == list(range(8)) and ra["order"] == [0, 4, 1, 5, 2, 6, 3, 7]
    # the declared-order shift is a path property and changes with the order; the
    # order-averaged statistic describes the partition and does not
    assert abs(rs["delta"] - ra["delta"]) > 1e-3
    assert abs(rs["delta_orderavg"] - ra["delta_orderavg"]) < 1e-9
    # entry coherence on the two planted codes: a dipole code's adjacent classes
    # cohere more than its antipodes (sequential first); an even code's antipodes
    # share a mean (antipodal first)
    assert tz.sectors(Xd, labels)["order_prediction"] == "sequential"
    Xe, _ = _circle_classes(harmonic=2)
    assert tz.sectors(Xe, labels)["order_prediction"] == "antipodal"


def test_circular_shift_null_separates_drift_from_alignment():
    # a slowly drifting population: frames in time order, labels = eight time
    # blocks. Drift makes the label permutation an easy null (every block is a
    # contiguous piece of a slow trajectory) while rolling the label sequence
    # keeps the blocks contiguous, so the shift must NOT survive the shift null
    rng = np.random.default_rng(2)
    n, d = 480, 48
    t = np.linspace(0, 1, n)[:, None]
    X = 3.0 * np.sin(2 * np.pi * t * rng.normal(size=(1, d)) * 0.5) + rng.normal(size=(n, d))
    labels = np.repeat(np.arange(8), n // 8)
    r = tz.zoom(X, labels, n_perm=60, k_orders=3, n_floor_draws=5, shift_null=True)
    assert "shift_p_two" in r and "shift_z" in r
    assert r["p_two"] < 0.05 and r["shift_p_two"] > 0.05
    rep = tz.summarize_data(r, verbose=False)
    assert rep["verdict"]["survives_circular_shift"] is False


def test_every_axis_sidecar_matches_its_axis():
    """Every shipped <axis>.strata.json has the classes and prompt counts of its axis, so the second null can read it."""
    import glob, json, os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sidecars = glob.glob(os.path.join(here, "axes", "*.strata.json"))
    assert len(sidecars) >= 4
    for side in sidecars:
        axis = json.load(open(side[: -len(".strata.json")] + ".json"))
        strata = json.load(open(side))
        assert set(axis) == set(strata), side
        for c in axis:
            assert len(axis[c]) == len(strata[c]), (side, c)
    pairs = json.load(open(os.path.join(here, "axes", "blimp_grammaticality.strata.json")))
    good = [c for c in pairs if c.endswith("_good")]
    for g in good:
        assert pairs[g] == pairs[g[: -len("_good")] + "_bad"], g
