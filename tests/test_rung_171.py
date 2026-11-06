"""Regression tests for the rung 1.7.1 correctness fixes (numpy only)."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rung as r


def test_order_average_does_not_depend_on_the_declared_order():
    # unequal classes: each order has its own rung sizes and must be read against its own floor
    X = np.random.default_rng(2).normal(size=(18, 5))
    y = np.repeat(np.arange(3), [3, 6, 9])
    # the declared order only decides which size sequence's floor comes from the declared stream, so the two order
    # averages differ by floor-draw noise alone (before 1.7.1: -0.057 against +0.075 at 20 draws)
    kw = dict(n_perm=0, k_orders=20, floor_seed=1, order_seed=2, perm_seed=3, n_floor_draws=400)
    a = r.zoom(X, y, **kw)
    b = r.zoom(X, y, order=[2, 1, 0], **kw)
    assert a["rung_sizes"] != b["rung_sizes"] and a["n_floor_sequences"] > 1
    assert a["delta_orderavg"] == pytest.approx(b["delta_orderavg"], abs=0.01)


def test_equal_classes_reuse_one_floor():
    X = np.random.default_rng(0).normal(size=(64, 6)); y = np.repeat(np.arange(4), 16)
    assert r.zoom(X, y, n_perm=0, k_orders=5)["n_floor_sequences"] == 1


def test_decompose_splits_the_same_ladder_as_zoom():
    X = np.random.default_rng(3).normal(size=(60, 6)); y = np.repeat(np.arange(5), 12)
    a = r.zoom(X, y, n_perm=0, k_orders=2)
    b = r.decompose(X, y, n_null=20, seed=0)
    assert a["rung_classes"] == [t["classes"] for t in b["rungs"]] == [1, 2, 3, 4, 5]
    r._print_decomposition(b)


def test_undefined_statistics_get_no_p_value():
    X = np.random.default_rng(4).normal(size=(36, 4)); y = np.repeat(np.arange(3), 12)
    a = r.zoom(X, y, n_perm=39, k_orders=0, time_blocks=np.arange(36) // 6)
    assert np.isnan(a["delta_orderavg"]) and np.isnan(a["p_two_orderavg"]) and np.isnan(a["tb_p_two_orderavg"])
    with pytest.raises(ValueError):
        r.zoom(X, y, n_floor_draws=0)
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        r.zoom(X, y, n_perm=0)


def test_identity_null_gives_p_one():
    # time blocks that each hold one class: every allowed permutation is the identity
    X = np.random.default_rng(0).normal(size=(36, 6)); y = np.repeat(np.arange(3), 12)
    a = r.zoom(X, y, n_perm=39, k_orders=2, k_null_orders=2, time_blocks=y, seed=1)
    assert a["tb_p_two_orderavg"] == 1.0 and a["tb_p_two"] == 1.0


def test_feature_offsets_do_not_change_any_statistic():
    X = np.random.default_rng(4).normal(size=(36, 4)); y = np.repeat(np.arange(3), 12)
    for off in (1e7, 1e8):
        assert r.zoom(X + off, y, n_perm=0, k_orders=1, n_floor_draws=2)["delta"] == pytest.approx(
            r.zoom(X, y, n_perm=0, k_orders=1, n_floor_draws=2)["delta"], abs=1e-6)
        assert r.decompose(X + off, y, n_null=2, seed=0)["delta"] == pytest.approx(
            r.decompose(X, y, n_null=2, seed=0)["delta"], abs=1e-6)


def test_bias_null_keeps_every_class_variance():
    X = np.array([[-1., -1.], [1., 1.], [9., -1.], [11., 1.], [19., -1.], [21., 1.]])
    y = np.repeat(np.arange(3), 2)
    for s in range(5):
        Y, ly = r.bias_null(X, y, seed=s)
        assert all(Y[ly == c].var(0).sum() > 0 for c in range(3))


def test_split_by_rejects_empty_subsets():
    X = np.random.default_rng(4).normal(size=(36, 2)); y = np.repeat(np.arange(3), 12)
    with pytest.raises(ValueError):
        r.split_by(X, y, [0, 1], n_perm=0, k_orders=1)


def test_one_standardization_for_shift_and_split():
    X = np.random.default_rng(0).normal(size=(36, 4)) * [1., 1., 1., 1e-10]; y = np.repeat(np.arange(3), 12)
    Xz = r._zscore(X)
    assert r.zoom(X, y, n_perm=0, k_orders=1, standardize=True)["delta"] == pytest.approx(
        r.zoom(Xz, y, n_perm=0, k_orders=1)["delta"], abs=1e-9)


def test_bh_count_is_reported_as_computed():
    import json, tempfile
    ps = [.04] + [1.] * 9
    L = [dict(delta=.1, delta_orderavg=.1, p_two=p, p_two_orderavg=p) for p in ps]
    f = os.path.join(tempfile.mkdtemp(), "bh.json"); json.dump(dict(n_perm=24, axes={"x": dict(layers=L)}), open(f, "w"))
    a = r.summarize(f, verbose=False)["axes"]["x"]
    assert a["certified_declared"] == [1, r._bh_count(ps)] == [1, 0]


def test_principal_angles_full_space_and_caps():
    X = np.random.default_rng(0).normal(size=(12, 2)); y = np.repeat(np.arange(3), 4)
    assert r.principal_angle_dissimilarity(X, y, k=5) == pytest.approx(0, abs=1e-12)
    assert r.cov_floors(X, y, k=5)["floor"] == 0.0
    X = np.random.default_rng(0).normal(size=(90, 5)); y = np.repeat(np.arange(3), 30)
    assert r.cov_floors(X, y, k=5)["full_space"]
    with pytest.raises(ValueError):
        r.cov_floors(np.ones((36, 4)), np.repeat(np.arange(3), 12))


def test_declared_order_rejects_unknown_classes():
    with pytest.raises(ValueError):
        r.declared_order(np.array([0, 2, 4]), [-1, 1, 3])


def test_degenerate_class_means_read_as_degenerate():
    base = np.array([[-1., 0.], [1., 0.], [0., -1.], [0., 1.]])
    X = np.tile(base, (9, 1)); y = np.repeat(np.arange(3), 12)
    s = r.sign_law(X, y)
    assert s["degenerate"] == "no class-mean variance" and s["c_full"] == 0.0
    r.decompose(X, y, n_null=2)
    with pytest.raises(ValueError):
        r.decompose(np.ones((36, 4)), y)
