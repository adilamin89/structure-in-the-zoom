"""Tests for theta_zoom (numpy only; no network). Run: pytest -q tests"""
import json, os, subprocess, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import theta_zoom as tz


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
    cmd = [sys.executable, os.path.join(os.path.dirname(tz.__file__), "theta_zoom.py"), "data",
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
