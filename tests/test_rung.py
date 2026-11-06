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


def test_deficit_ladder_and_the_share_left_after_half_the_circle():
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
    assert "within-class part D" in res.stdout and "class-mean part (Tb + Cb + Cx)" in res.stdout
    assert "net of 2 label permutations" in res.stdout and "full-set class-mean remainder c" in res.stdout
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
    assert abs(sc["C_antipodal"] - (q2 - q1)) < 1e-6 and "order_prediction" not in sc
    assert "at least three classes" in tz.sectors(X[labels < 2], labels[labels < 2])["undefined"]


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
    # class-mean coherence on the two planted codes: a dipole code's adjacent
    # classes cohere more than its antipodes; an even code's antipodes share a mean
    sd = tz.sectors(Xd, labels)
    assert sd["C_adjacent"] > sd["C_antipodal"]
    Xe, _ = _circle_classes(harmonic=2)
    se = tz.sectors(Xe, labels)
    assert se["C_antipodal"] > se["C_adjacent"]


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


# ---------------- 1.6.0 ----------------

def test_bias_null_keeps_means_sizes_variance_and_nets_out_on_class_free_fluctuations():
    rng = np.random.default_rng(5)
    C, m, d = 8, 60, 40
    labels = np.repeat(np.arange(C), m)
    A = rng.normal(size=(d, d)) / np.sqrt(d)                       # one fluctuation shape for every class
    X = rng.normal(size=(C, d))[labels] * 1.5 + rng.normal(size=(C * m, d)) @ A
    Y, ly = tz.bias_null(X, labels, "shape", seed=0)
    for c in range(C):
        assert np.allclose(Y[ly == c].mean(0), X[labels == c].mean(0), atol=1e-9)
        assert abs(Y[ly == c].var(0).sum() - X[labels == c].var(0).sum()) < 1e-6 * X[labels == c].var(0).sum()
    assert np.array_equal(np.bincount(ly), np.bincount(labels))
    r = tz.decompose(X, labels, n_null=10, n_bias=3)
    assert abs(r["net_of_bias"]["delta_split_free"]["D"]) < 0.03


def test_sign_law_is_exact():
    rng = np.random.default_rng(7)
    for _ in range(200):
        C, m, d = int(rng.integers(3, 9)), int(rng.integers(8, 30)), int(rng.integers(4, 60))
        labels = np.repeat(np.arange(C), m)
        X = rng.normal(size=(C, d))[labels] * rng.uniform(0.1, 3) + rng.normal(size=(C * m, d)) * rng.uniform(0.2, 2, size=d)
        sl = tz.sign_law(X, labels)
        b1, w1, b2, w2, x = tz.traces(X.astype(np.float64), labels)
        assert abs(sl["c_full"] - np.log((1 + b1 / w1) ** 2 / (1 + (b2 + 2 * x) / w2))) < 1e-4
        assert sl["class_mean_negative"] == sl["law_says_negative"]


def test_time_blocks_and_same_class_runs():
    import warnings
    X, labels = _gaussian_classes(n=320, d=32)
    runs = np.repeat(np.tile(np.arange(8), 8), 5)                 # runs of five, eight passes: clustered, not grouped
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tz.zoom(X, runs, n_perm=0, k_orders=3, n_floor_draws=3)
        assert any("cluster in time" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        tz.zoom(X, np.random.default_rng(0).permutation(runs), n_perm=0, k_orders=3, n_floor_draws=3)
        tz.zoom(X, np.sort(runs), n_perm=0, k_orders=3, n_floor_draws=3)            # grouped by class: no warning
        assert not any("cluster in time" in str(x.message) for x in w)
    r = tz.zoom(X, runs, n_perm=30, k_orders=3, n_floor_draws=3, time_blocks=tz.block_ids(len(runs), 40))
    assert "tb_p_two" in r and "tb_p_two_orderavg" in r and r["n_time_blocks"] == 8


def test_pair_preserving_orders_match_the_contrast_design():
    import itertools
    groups = [[0, 1], [2, 3], [4, 5]]
    orders, total = tz.pair_preserving_orders(groups)
    ref = [sum(([2 * c + mm, 2 * c + 1 - mm] for c, mm in zip(p, mem)), [])
           for p in itertools.permutations(range(3)) for mem in itertools.product((0, 1), repeat=3)]
    assert total == 48 and sorted(map(tuple, orders)) == sorted(map(tuple, ref))
    rng = np.random.default_rng(1)
    labels = np.repeat(np.arange(6), 20)
    strata = np.array([f"c{c // 2}_p{k}" for c in range(6) for k in range(20)])   # pair ids shared by partner classes
    X = rng.normal(size=(120, 24)) + rng.normal(size=(6, 24))[labels]
    r = tz.zoom(X, labels, n_perm=20, k_orders=5, n_floor_draws=3, strata=strata)
    assert r["pair_groups"] == groups and r["n_pair_orders"] == 48 and "pairavg_p_two" in r


def test_last_token_report_and_matched_arms():
    labels = np.repeat(np.arange(4), 12)
    last = np.concatenate([np.arange(12), np.arange(12) % 3, np.arange(12) + 50, np.full(12, 7)])
    rep = tz.last_token_report(last, labels)
    assert rep["distinct_per_class"] == [12, 3, 12, 1] and rep["warnings"] and not rep["one_token_everywhere"]
    assert tz.last_token_report(np.zeros(48, int), labels)["one_token_everywhere"]
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(4), 16)
    last = np.tile(np.arange(8), 8)                                # eight distinct last tokens in every class
    X = rng.normal(size=(64, 20)) + rng.normal(size=(4, 20))[labels]
    arms = tz.last_token_arms(X, labels, last, n_sub=3, n_perm=10, k_orders=3)
    assert arms["m_per_class"] == 8 and "matched" in arms and "random" in arms


def test_frames_disagree_flags_a_dominant_unit():
    rng = np.random.default_rng(4)
    C, m, d = 8, 40, 60
    labels = np.repeat(np.arange(C), m)
    X = rng.normal(size=(C * m, d))
    X[:, :10] *= (1 + labels[:, None] % 3)                         # class-dependent fluctuation sizes: positive shift
    X[:, -1] = 300 * (labels % 2) + rng.normal(size=C * m)         # one huge unit carrying two class means: negative
    kw = dict(n_perm=40, k_orders=10, n_floor_draws=5, seed=0)
    rl, rz = tz.zoom(X, labels, **kw), tz.zoom(X, labels, standardize=True, **kw)
    assert rl["delta_orderavg"] < rl["null_mean_orderavg"] and rz["delta_orderavg"] > rz["null_mean_orderavg"]
    assert tz.frames_disagree(rl, rz)


def test_valid_from_intervals_overlap_rule():
    starts, stops = np.array([0.0, 2.0, 4.0, 6.0]), np.array([1.0, 3.0, 5.0, 7.0])
    v = tz.valid_from_intervals(starts, stops, [[1.0, 2.0], [4.5, 4.6], [6.9, 9.0]])
    # touching an interval's ends is not an overlap; inside or partial overlaps are invalid
    assert v.tolist() == [True, True, False, False]
    assert tz.valid_from_intervals(starts, stops, np.zeros((0, 2))).all()


def test_cli_validity_mask_equals_running_on_the_kept_samples(tmp_path):
    X, labels = _gaussian_classes(n=320, d=48)
    rng = np.random.default_rng(5)
    valid = rng.random(len(labels)) > 0.25
    t0 = np.arange(len(labels)) * 3.0
    times = np.stack([t0, t0 + 2.0], 1)
    bad = times[~valid] + np.array([0.5, -0.5])                      # one invalid interval inside each dropped window
    for name, arr in (("X", X), ("labels", labels), ("valid", valid.astype(int)), ("times", times), ("bad", bad),
                      ("Xk", X[valid]), ("labk", labels[valid])):
        np.save(tmp_path / f"{name}.npy", arr)
    script = os.path.join(os.path.dirname(tz.__file__), "rung.py")
    base = ["--n-perm", "20", "--k-orders", "3", "--seed", "4"]
    outs = {}
    for tag, args in (("mask", ["X.npy", "labels.npy", "--valid", "valid.npy"]),
                      ("intervals", ["X.npy", "labels.npy", "--sample-times", "times.npy", "--invalid-intervals", "bad.npy"]),
                      ("subset", ["Xk.npy", "labk.npy"])):
        out = tmp_path / f"{tag}.json"
        res = subprocess.run([sys.executable, script, "data", *[str(tmp_path / a) if a.endswith(".npy") else a for a in args], *base,
                              "--out", str(out)], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        outs[tag] = json.load(open(out))
    for tag in ("mask", "intervals"):
        assert outs[tag]["delta"] == outs["subset"]["delta"] and outs[tag]["p_two"] == outs["subset"]["p_two"]
        assert outs[tag]["n_samples_valid"] == int(valid.sum()) and outs[tag]["n_samples_total"] == len(labels)


def test_gamma_nodes_carry_the_gain_moments():
    for kappa in (1.0, 1.006, 1.5, 3.7):
        t, w = tz._gamma_nodes(kappa)
        assert np.isfinite(t).all() and abs(w @ t - 1) < 1e-9 and abs(w @ t ** 2 - max(kappa, 1.0)) < 1e-6


def test_principal_angle_dissimilarity_by_definition():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 30)); labels = np.repeat(np.arange(4), 20)
    R = [X[labels == c] - X[labels == c].mean(0) for c in range(4)]
    U = [np.linalg.svd(r, full_matrices=False)[2][:3].T for r in R]
    ref = 1 - np.mean([np.sum((U[a].T @ U[b]) ** 2) / 3 for a in range(4) for b in range(a + 1, 4)])
    assert abs(tz.principal_angle_dissimilarity(X, labels, k=3) - ref) < 1e-12


def test_cov_floor_law_predicts_the_label_blind_floor():
    # identical classes from an elliptical population: the observed value is all floor; the law with the gain lands near the
    # permutation floor, the spectrum alone well below it
    rng = np.random.default_rng(1)
    p, n_per = 120, 50
    lam = np.arange(1, p + 1) ** -1.0
    tau = rng.gamma(0.5, 2.0, 8 * n_per)                       # E tau = 1, E tau^2 = 3
    X = rng.standard_normal((8 * n_per, p)) * np.sqrt(lam) * np.sqrt(tau)[:, None]
    labels = np.repeat(np.arange(8), n_per)
    r = tz.cov_floors(X, labels, n_perm=30, seed=0)
    assert abs(r["floor"] - r["floor_permutation"]) / r["floor_permutation"] < 0.08
    assert r["floor_spectrum_only"] < r["floor"]
    assert abs(r["floor"] - r["floor_permutation"]) < abs(r["floor_spectrum_only"] - r["floor_permutation"])
