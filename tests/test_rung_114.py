"""Tests for rung 1.14: the order-averaged shift over every class order, exact (a linear functional of the rung subsets' log PR);
the split read net of the shape surrogate when the sampling check fires."""
import itertools
import math

import numpy as np
import pytest

import rung as r


def _toy(seed=0, counts=None, d=48):
    g = np.random.default_rng(seed)
    counts = counts or [40] * 8
    y = np.repeat(np.arange(len(counts)), counts)
    X = g.standard_normal((len(y), d))
    X *= 1 + 2 * (y[:, None] == np.arange(d) // (d // len(counts)))
    return X, y


def _brute_orderavg(X, y, bc, floor_logs):
    """Every order's shift through the 1.13 ladder code (_union_ladders on all K! orders), each read against the floor at its own
    rung sizes from floor_logs {size: mean log PR}: the definition the exact path must equal."""
    Xc = X - X.mean(0); Xc = Xc / (Xc.std() + 1e-9)
    K = Xc @ Xc.T
    nc = int(y.max()) + 1
    orders = np.array(list(itertools.permutations(range(nc))))
    sl, _, m, _ = r._union_ladders(r._class_blocks(K, y, nc), orders, bc)
    fl = np.array([r._slope(row, np.exp([floor_logs[int(round(s))] for s in row])) for row in m])
    d = sl - fl
    return float(d.mean()), float(d.std())


@pytest.mark.parametrize("counts", [[30, 30, 30, 30, 30], [25, 31, 40, 28, 35, 22]])
def test_exact_order_average_equals_every_order_through_the_ladder_code(counts):
    X, y = _toy(1, counts, d=36)
    res = r.zoom(X, y, n_perm=0, k_orders=10, order_average="exact")
    assert res["orderavg_method"] == "exact" and res["n_orders_averaged"] == math.factorial(len(counts))
    bc = tuple(res["rung_classes"])
    fl = {int(k): v for k, v in res["orderavg_floor_logs"].items()}
    mean, sd = _brute_orderavg(X, y, bc, fl)
    assert abs(res["delta_orderavg"] - mean) < 1e-10
    assert abs(res["delta_orderavg_sd"] - sd) < 1e-10


def test_equal_classes_order_average_is_the_slope_of_the_subset_mean_log_pr():
    X, y = _toy(2)
    res = r.zoom(X, y, n_perm=0, k_orders=0, order_average="exact")
    Xc = X - X.mean(0); Xc = Xc / (Xc.std() + 1e-9)
    K = Xc @ Xc.T
    mem = [np.where(y == c)[0] for c in range(8)]
    ys = [np.mean([np.log(r._subset_pr(K, np.concatenate([mem[i] for i in S]))) for S in itertools.combinations(range(8), c)])
          for c in res["rung_classes"]]
    want = r._lsq_slope(np.log(res["rung_sizes"]), ys) - res["theta_floor"]
    assert abs(res["delta_orderavg"] - want) < 1e-10
    assert res["delta_orderavg_se"] == res["theta_floor_se"]          # equal classes: the declared floor, its own SE


def test_exact_agrees_with_random_orders_within_their_monte_carlo_se_over_seeds():
    X, y = _toy(3)
    z = []
    for s in range(60):
        res = r.zoom(X, y, n_perm=0, k_orders=30, seed=s, order_average="exact")
        z.append((res["delta_orderavg_random"] - res["delta_orderavg"]) / res["delta_orderavg_random_se"])
    z = np.array(z)
    assert abs(z.mean()) < 3 / np.sqrt(len(z)) * 1.5 and 0.6 < z.std() < 1.4


def test_unequal_classes_exact_agrees_with_random_orders_within_their_combined_se():
    X, y = _toy(4, [30, 44, 36, 52, 40, 33, 47, 38])
    z = []
    for s in range(30):
        res = r.zoom(X, y, n_perm=0, k_orders=30, seed=s, order_average="exact")
        se = np.hypot(res["delta_orderavg_random_se"], res["delta_orderavg_se"])
        z.append((res["delta_orderavg_random"] - res["delta_orderavg"]) / se)
    assert abs(np.mean(z)) < 0.6 and np.std(z) < 1.5


def test_exact_order_average_is_bit_identical_across_threads_and_keeps_the_declared_null():
    X, y = _toy(5, [30, 44, 36, 52, 40, 33, 47, 38])
    st = np.repeat(np.arange(4), len(y) // 4 + 1)[: len(y)]
    kw = dict(n_perm=40, k_orders=12, strata=st, time_blocks=r.block_ids(len(y), 40))
    a = r.zoom(X, y, workers=1, order_average="exact", **kw)
    b = r.zoom(X, y, workers=4, order_average="exact", **kw)
    for k in a:
        assert np.array_equal(np.asarray(a[k], dtype=object), np.asarray(b[k], dtype=object)), k
    old = r.zoom(X, y, workers=1, order_average="random", **kw)
    for k in ("delta", "p_two", "z", "null_mean", "null_sd", "strat_p_two", "tb_p_two", "theta_floor"):
        assert a[k] == old[k], k                                       # the declared order and its draws are 1.13.1's
    assert a["delta_orderavg_random"] == old["delta_orderavg"]


def test_exact_null_reads_each_relabelling_over_every_order():
    X, y = _toy(6)
    res = r.zoom(X, y, n_perm=30, k_orders=5, keep_null=True, order_average="exact")
    assert np.isfinite(res["p_two_orderavg"]) and res["p_two_orderavg"] <= 1 / 31 + 1e-12
    g = np.random.default_rng(0)
    z0 = r.zoom(X, y[g.permutation(len(y))], n_perm=30, k_orders=5, order_average="exact")
    assert abs(z0["delta_orderavg"]) < 4 * res["null_sd_orderavg"] + 0.05


def test_auto_order_average():
    X, y = _toy(7)
    assert r.zoom(X, y, n_perm=0, k_orders=5)["orderavg_method"] == "exact"
    assert r.zoom(X, y, n_perm=0, k_orders=5, fast=False)["orderavg_method"] == "exact"    # the same statistic in either arithmetic
    assert r.zoom(X, y, n_perm=0, k_orders=5, floor_by_rung=True)["orderavg_method"] == "random"   # the paper's replays
    assert r.zoom(X, y, n_perm=0, k_orders=5, perm_seed=3700, order_seed=3701)["orderavg_method"] == "random"
    assert np.isnan(r.zoom(X, y, n_perm=0, k_orders=0)["delta_orderavg"])                   # k_orders 0: no order average
    X11, y11 = _toy(8, [12] * 11, d=44)
    assert r.zoom(X11, y11, n_perm=0, k_orders=5)["orderavg_method"] == "random"           # 11! orders: random


def test_summarize_names_the_exact_order_average():
    X, y = _toy(9)
    lines = r.summarize_data(r.zoom(X, y, n_perm=20, k_orders=10), verbose=False)["lines"]
    assert any("over all 40,320 class orders" in l for l in lines)


# ---- the split net of the shape surrogate when the sampling check fires ----

def test_auto_bias_null_fires_only_below_a_class_dimension():
    X, y = _toy(0)                                                      # 40 per class, class dimensions ~20: no fire
    d = r.decompose(X, y)
    assert d["bias_auto"]["fired"] is False and "net_of_bias" not in d
    assert d["delta_split_free"] == r.decompose(X, y, n_bias=0)["delta_split_free"]
    g = np.random.default_rng(1)
    y2 = np.repeat(np.arange(6), 16)
    X2 = g.standard_normal((96, 120)) * np.linspace(1.0, 0.5, 120)     # 16 per class against class dimensions near 60
    d2 = r.decompose(X2, y2)
    assert d2["bias_auto"]["fired"] is True and d2["bias_null"]["n"] == 10
    ref = r.decompose(X2, y2, n_bias=10)
    assert d2["net_of_bias"] == ref["net_of_bias"] and d2["delta_split_free"] == ref["delta_split_free"]
    lines = r.summarize_data({**r.zoom(X2, y2, n_perm=0, k_orders=0), "decompose": d2}, verbose=False)["lines"]
    first = next(l for l in lines if l.startswith("split"))
    assert first.startswith("split (Section 5), net of the shape surrogate")


# ---- the drift check ----

def _drifting(seed=0, drift=0.0, n_per=40):
    """Rows in time order, classes presented in random order; each class stretches its own features by a factor that grows with time
    by `drift` (the shift rises with that factor: 0.18 at 2, 0.31 at 4, order-averaged, 20 per class)."""
    g = np.random.default_rng(seed)
    y = g.permutation(np.repeat(np.arange(8), n_per))
    t = np.linspace(0, 1, len(y))
    X = g.standard_normal((len(y), 48))
    X *= 1 + (1.0 + drift * t)[:, None] * (y[:, None] == np.arange(48) // 6)
    return X, y, t


def test_drift_check_flags_a_planted_drift_and_reads_times():
    X, y, t = _drifting(0, drift=3.0)
    dr = r.drift_check(X, y, n_splits=12, n_floor_draws=6)
    # the order-averaged shift carries the drift; the declared shift is one path, whose spread at 20 per class hides it
    assert dr["stats"]["orderavg"]["late_minus_early"] > 0 and "orderavg" in dr["flagged"]
    assert len(dr["stats"]["shift"]["interleaved_differences"]) == 12 and np.isfinite(dr["stats"]["shift"]["alternate_difference"])
    g = np.random.default_rng(1).permutation(len(y))            # the same rows stored out of time order, with their times
    dr2 = r.drift_check(X[g], y[g], times=t[g], n_splits=12, n_floor_draws=6)
    assert dr2["order"] == "sample times"
    for k in ("early", "late"):
        assert abs(dr2["stats"]["orderavg"][k] - dr["stats"]["orderavg"][k]) < 1e-9
    lines = r._drift_lines(dr)
    assert "FLAGGED" in lines[0] and "time-blocked null" in lines[0]


def test_drift_check_does_not_flag_a_stationary_session():
    flags = 0
    for s in range(4):
        X, y, _ = _drifting(10 + s, drift=0.0)
        flags += len(r.drift_check(X, y, n_splits=12, n_floor_draws=6, seed=s)["flagged"])
    assert flags <= 2                                             # 12 statistic-reads at about 5% each


def test_command_line_runs_the_drift_check_with_time_blocks(tmp_path):
    import json
    import subprocess
    import sys
    X, y, _ = _drifting(2, drift=3.0)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "y.npy", y)
    subprocess.run([sys.executable, "-m", "rung", "data", "X.npy", "y.npy", "--n-perm", "10", "--k-orders", "4", "--floor-draws", "4",
                    "--time-blocks", "40", "--drift-splits", "6", "--out", "o.json"], cwd=tmp_path, capture_output=True, text=True, check=True)
    d = json.load(open(tmp_path / "o.json"))
    assert d["drift"]["n_splits"] == 6 and d["drift"]["order"] == "row order"
    assert any(l.startswith("drift check") for l in r.summarize_data(d, verbose=False)["lines"])


# ---- classes from a continuous variable ----

def _paper_bins(phi, nb):
    # scripts_canonical/cr_order_and_withinbin_controls.bins
    return np.clip(np.digitize(phi % (2 * np.pi), np.linspace(0, 2 * np.pi, nb + 1)) - 1, 0, nb - 1)


def _paper_remove_withinbin(X, lab, phi, n_sub_per_class, n_class):
    # scripts_canonical/cr_order_and_withinbin_controls.remove_withinbin, in float64
    sub = _paper_bins(phi, n_class * n_sub_per_class)
    Y = X.copy()
    for b in np.unique(sub):
        m = sub == b; nb = m.sum()
        if nb < 2:
            continue
        S = X[m].sum(0)
        Y[m] = X[m] - (S - X[m]) / (nb - 1)
    for c in np.unique(lab):
        m = lab == c; Y[m] += X[m].mean(0)
    return Y


def _tuned(seed=0, n=640, d=40):
    """Directions uniform on the circle; each feature tuned to direction (a cosine at its own preferred angle), plus noise."""
    g = np.random.default_rng(seed)
    phi = g.uniform(0, 2 * np.pi, n)
    pref = g.uniform(0, 2 * np.pi, d)
    X = 1.5 * np.cos(phi[:, None] - pref[None, :]) + g.standard_normal((n, d))
    return X, phi


def test_width_cells_replay_the_paper_bins_and_residualization():
    X, phi = _tuned(0)
    lab = r.cells_from_continuous(phi, 8, "width", cyclic=True)
    assert np.array_equal(lab, _paper_bins(phi, 8))
    sub = lab * 48 + r.cells_from_continuous(phi, 48, "width", cyclic=True)
    Y, share = r.residualize_within_cells(X, lab, sub)
    assert np.allclose(Y, _paper_remove_withinbin(X, lab, phi, 6, 8), atol=1e-12)
    # tuning inside a 45-degree cell: 1.5^2 x (1 - sinc^2(pi/8)) / 2 = 0.057 per feature against noise 1, a share of 0.054
    assert abs(share - 0.054) < 0.02
    Xn = np.random.default_rng(1).standard_normal(X.shape)          # no tuning: near zero
    assert abs(r.residualize_within_cells(Xn, lab, sub)[1]) < 0.015


def test_count_cells_and_guard():
    phi = np.random.default_rng(2).exponential(1.0, 800)
    lab = r.cells_from_continuous(phi, 8, "count")
    assert np.bincount(lab).tolist() == [100] * 8
    lw = r.cells_from_continuous(phi, 8, "width")
    assert np.bincount(lw)[0] > 300                                 # equal width on a skewed variable: crowded low cells
    g = r.cells_from_continuous(phi, 8, "count", guard=0.1)
    assert (g == -1).sum() == 160 and set(np.unique(g[g >= 0])) == set(range(8))
    with pytest.raises(ValueError):
        r.cells_from_continuous(phi, 8, origin=0.3)                 # a linear axis has one cut


def test_turning_the_walls_a_full_cell_leaves_the_order_average_and_rotates_the_declared_start():
    X, phi = _tuned(3)
    w = 2 * np.pi / 8
    a = r.continuous(X, phi, 8, cyclic=True, n_origins=1, n_perm=0, k_orders=5, split=False)
    b = r.continuous(X, phi, 8, cyclic=True, origin=w, n_origins=1, n_perm=0, k_orders=5, split=False)
    la, lb = r.cells_from_continuous(phi, 8, cyclic=True), r.cells_from_continuous(phi, 8, cyclic=True, origin=w)
    assert np.array_equal((lb + 1) % 8, la)                         # the same partition, labels rotated by one
    # the exact order average depends on the partition alone; the declared order starts one cell later
    assert abs(a["summary"]["delta_orderavg"]["first"] - b["summary"]["delta_orderavg"]["first"]) < 1e-10
    c = r.continuous(X, phi, 8, cyclic=True, n_perm=0, k_orders=5)
    assert c["n_origins"] == 8 and abs(c["step"] - w / 8) < 1e-15 and len(c["origins"]) == 8
    assert c["summary"]["delta"]["sd"] > 0 and "within" in c["summary"]


def test_command_line_continuous(tmp_path):
    import json
    import subprocess
    import sys
    X, phi = _tuned(4)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "phi.npy", phi)
    subprocess.run([sys.executable, "-m", "rung", "data", "X.npy", "--continuous", "phi.npy", "--cyclic", "--origins", "4",
                    "--within-cell", "residualize", "--n-perm", "10", "--k-orders", "4", "--floor-draws", "4", "--decompose",
                    "--out", "o.json"], cwd=tmp_path, capture_output=True, text=True, check=True)
    d = json.load(open(tmp_path / "o.json"))
    assert d["continuous"]["n_origins"] == 4 and d["continuous"]["residualize"] and d["n_classes"] == 8
    lines = r.summarize_data(d, verbose=False)["lines"]
    assert any(l.startswith("continuous variable cut into 8 equal-width arcs") for l in lines)
    assert any(l.startswith("within-cell tuning:") for l in lines)


def test_one_version_string():
    import pathlib
    root = pathlib.Path(r.__file__).parent
    if not (root / "pyproject.toml").exists():
        pytest.skip("rung installed without its source tree")
    assert r.__version__ == "1.14.0" and 'version = "1.14.0"' in (root / "pyproject.toml").read_text()
    readme = (root / "README.md").read_text()
    assert "rung-1.14.0" in readme and "@v1.14.0" in readme and "1.13.1" not in readme.split("## The approach")[0]
    nb = (root / "examples" / "quickstart.ipynb").read_text()
    assert "@v1.14.0" in nb and "rung 1.14.0" in nb


def test_head_scores_read_every_order():
    g = np.random.default_rng(11)
    y = np.repeat(np.arange(6), 20)
    H = g.standard_normal((1, 3, len(y), 24))
    H[0, 0] *= 1 + 2 * (y[:, None] == np.arange(24) // 4)          # one head organized by the label
    ex = r.head_scores(H, y, [0], n_perm=10, k_orders=3000, split=False)
    rd = r.head_scores(H, y, [0], n_perm=10, k_orders=3000, split=False, order_average="random")
    for a, b in zip(ex, rd):                                       # the same floor draws: the difference is order sampling alone
        assert a["orderavg_method"] == "exact" and b["orderavg_method"] == "random"
        assert abs(a["delta_orderavg"] - b["delta_orderavg"]) < 0.01
    assert r.head_scores(H, y, [0], n_perm=4, k_orders=5, split=False, paper_seeds=True)[0]["orderavg_method"] == "random"


def test_drift_docstring_numbers_come_from_the_calibration():
    import json
    import os
    path = os.path.join(os.path.dirname(r.__file__), "data_canonical", "drift_calibration.json")
    if not os.path.exists(path):
        pytest.skip("rung installed without its data")
    c = json.load(open(path))
    doc = " ".join(r.drift_check.__doc__.split())
    st, dr = c["stationary"], c["drift3"]
    rates = ", ".join(f"{100 * st[k]['flag_rate']:.1f}%" for k in ("shift", "orderavg", "meanfree")).replace(", 6.0%", " and 6.0%")
    assert f"flagged {rates} of the time" in doc
    assert "(z SD " + ", ".join(f"{st[k]['z_sd']:.2f}" for k in ("shift", "orderavg", "meanfree")) + ")" in doc
    assert f"flagged {round(50 * dr['orderavg']['flag_rate'])} of 50 (mean z {dr['orderavg']['z_mean']:.1f})" in doc
    assert dr["shift"]["flag_rate"] == dr["meanfree"]["flag_rate"] == 0.18 and "statistics 18%" in doc


def test_review_fixes_code():
    """The DeepSeek V4.1 Flash code review's reproductions (t12 review ledger): each now reads or stops cleanly."""
    import contextlib
    import io
    X, phi = _tuned(5, n=240, d=20)
    c = r.continuous(X, phi, 6, cyclic=True, n_origins=3, k_orders=0, split=False, n_perm=0)    # no order average asked for
    lines = r._continuous_lines(c)
    assert "declared-order shift" in lines[1] and "order-averaged" not in lines[1]
    with pytest.raises(ValueError):
        r.continuous(X, phi, 6, cyclic=True, n_origins=0, n_perm=0, split=False)
    with pytest.raises(ValueError):
        r.cells_from_continuous(phi, 4, cyclic=True, period=0.0)
    g = np.random.default_rng(0)
    y = np.repeat(np.arange(6), [40, 40, 40, 3, 3, 3])                                         # three classes too small to read
    ba = r.decompose(g.standard_normal((len(y), 5)), y)["bias_auto"]
    assert ba["fired"] and ba["classes_too_small"] == 3 and ba["classes_at_or_below"] >= 3
    y2 = np.repeat(np.arange(6), 16)
    X2 = g.standard_normal((96, 120)) * np.linspace(1.0, 0.5, 120)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r._print_decomposition(r.decompose(X2, y2))
    assert buf.getvalue().startswith("decompose (Section 5), net of the shape surrogate")
    dr = {"order": "row order", "n_splits": 20, "n_splits_used": 17, "flag_z": 2.0, "flagged": [],
          "stats": {"shift": {"early": 0.1, "late": 0.1, "late_minus_early": 0.0, "spread": 0.01, "z": 0.0, "flag": False,
                              "alternate_difference": 0.0}}}
    assert "over 17 random interleaved splits of 20" in r._drift_lines(dr)[0]


def test_review_fixes_claims():
    """The Codex claims audit's code items (t12 review ledger)."""
    Y, _ = r.residualize_within_cells(np.array([[0.0], [2.0], [8.0], [1.0], [3.0]]), np.array([0, 0, 0, 1, 1]),
                                      np.array([0, 0, 1, 2, 2]))
    assert abs(Y[:3].mean() - 10 / 3) < 1e-12 and abs(Y[3:].mean() - 2.0) < 1e-12          # a singleton sub-cell keeps the class mean
    X, y = _toy(0)
    res = {**r.zoom(X, y, n_perm=0, k_orders=0), "decompose": r.decompose(X, y, n_bias=3)}   # asked for, not fired
    first = next(l for l in r.summarize_data(res, verbose=False)["lines"] if l.startswith("split"))
    assert first.startswith("split (Section 5), net of the shape surrogate") and "as requested (--n-bias)" in first
    Xd, yd, _ = _drifting(3, drift=0.0)
    assert "too few valid interleaved splits" in r._drift_lines(r.drift_check(Xd, yd, n_splits=1, n_floor_draws=4))[0]
    rev = list(range(7, -1, -1))
    dr = r.drift_check(Xd, yd, n_splits=2, n_floor_draws=4, order=rev)
    early = np.arange(len(yd)) < len(yd) // 2
    z = r.zoom(Xd[early], yd[early], n_perm=0, k_orders=0, n_floor_draws=4, order=rev, order_average="exact")
    assert abs(dr["stats"]["shift"]["early"] - z["delta"]) < 1e-12


def test_measured_drift_rate_in_the_readme():
    import json
    import os
    root = os.path.dirname(r.__file__)
    if not os.path.exists(os.path.join(root, "README.md")):
        pytest.skip("rung installed without its source tree")
    st = json.load(open(os.path.join(root, "data_canonical", "drift_calibration.json")))["stationary"]
    rates = [100 * st[k]["flag_rate"] for k in ("shift", "orderavg", "meanfree")]
    txt = f"{min(rates):.1f} to {max(rates):.1f}% per statistic"
    assert txt in " ".join(open(os.path.join(root, "README.md")).read().split()) and txt in open(r.__file__).read()


def test_many_classes_few_rows_give_an_undefined_split_not_a_crash():
    """LB1's report (Nov 9): 48 classes x 16 rows raised ZeroDivisionError, a 16-row floor draw holding only singleton classes."""
    X = np.random.default_rng(0).normal(size=(48 * 16, 64)); y = np.repeat(np.arange(48), 16)
    d = r.decompose(X, y, n_null=20, seed=42, n_bias=0)
    assert "no two samples of one class" in d["undefined"] and np.isnan(d["delta_split_free"]["D"])
    assert "undefined" in r.summarize_data({**r.zoom(X, y, n_perm=0, k_orders=0), "decompose": d}, verbose=False)["lines"][-2]
    X40 = np.random.default_rng(0).normal(size=(40 * 16, 64))
    assert "undefined" not in r.decompose(X40, np.repeat(np.arange(40), 16), n_null=20, seed=42, n_bias=0)
