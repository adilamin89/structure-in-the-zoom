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
