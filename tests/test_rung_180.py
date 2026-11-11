"""Tests for rung 1.8.0: the floor's standard error, the stratified null of the order-averaged shift, the readings, the
candidate screen, the two-term account, the feature ladder, block coarse-graining, the loaders and the attention-head scores.
The synthetic tests run everywhere (numpy only); the regressions against the paper's artifacts run when the raw recordings
are present under raw/ (set RUNG_DATA_TESTS=1), the GPT-2 head replay when torch, transformers and the cached model are
there too (RUNG_SLOW_TESTS=1)."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rung as r  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data_canonical")
RAW = os.path.join(ROOT, "raw")
data_tests = pytest.mark.skipif(os.environ.get("RUNG_DATA_TESTS") != "1", reason="raw recordings: set RUNG_DATA_TESTS=1")


def _structured(n_per=60, K=8, d=40, seed=0, private=4, shared=0.0):
    """K classes whose fluctuations live in partly private subspaces (a positive within-class part)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(d, d))
    X, y = [], []
    for c in range(K):
        A = shared * base[:, :private] + rng.normal(size=(d, private))
        X.append(rng.normal(size=(n_per, private)) @ A.T + 0.3 * rng.normal(size=(n_per, d)) + 0.5 * rng.normal(size=d))
        y.append(np.full(n_per, c))
    return np.concatenate(X), np.concatenate(y)


# ---------------------------------------------------------------- zoom additions
def test_floor_standard_error_is_the_sd_of_single_draw_slopes():
    X, y = _structured()
    a = r.zoom(X, y, n_perm=0, k_orders=0, n_floor_draws=12, seed=3)
    assert np.isfinite(a["theta_floor_se"]) and a["theta_floor_se"] > 0
    assert abs(a["theta_floor_se"] - a["theta_floor_sd"] / np.sqrt(12)) < 1e-12


def test_floor_by_rung_changes_only_the_draw_order():
    X, y = _structured()
    a = r.zoom(X, y, n_perm=0, k_orders=0, n_floor_draws=1, floor_seed=5)
    b = r.zoom(X, y, n_perm=0, k_orders=0, n_floor_draws=1, floor_seed=5, floor_by_rung=True)
    assert abs(a["theta_floor"] - b["theta_floor"]) < 1e-12        # one draw: both orders consume the stream alike
    c = r.zoom(X, y, n_perm=0, k_orders=0, n_floor_draws=6, floor_seed=5, floor_by_rung=True)
    assert abs(c["theta_obs"] - a["theta_obs"]) < 1e-12              # the declared ladder never depends on the floor


def test_stratified_null_of_the_order_averaged_shift():
    X, y = _structured(n_per=24)
    st = np.tile(np.arange(4), len(y) // 4)
    a = r.zoom(X, y, n_perm=20, k_orders=10, k_null_orders=5, strata=st, seed=1)
    for k in ("strat_p_two_orderavg", "strat_z_orderavg", "strat_null_mean_orderavg"):
        assert k in a and np.isfinite(a[k])


def test_summary_says_neither_null_separates_when_neither_does():
    res = {"theta_obs": 0.1, "theta_floor": 0.1, "delta": 0.0, "n_classes": 8, "rung_sizes": [1, 2, 3], "n_perm": 100,
           "z": 0.2, "p_two": 0.4, "strat_p_two": 0.5, "n_floor_draws": 20, "theta_floor_se": 0.01}
    txt = " ".join(r.summarize_data(res, verbose=False)["lines"])
    assert "not separated from the label-permutation null to begin with" in txt and "ABSORBED" not in txt
    res["p_two"] = 0.01
    assert "ABSORBED" in " ".join(r.summarize_data(res, verbose=False)["lines"])


def _res(**kw):
    base = {"theta_obs": 0.2, "theta_floor": 0.1, "delta": 0.1, "n_classes": 8, "rung_sizes": [1, 2, 3], "n_perm": 500,
            "z": 0.9, "p_two": 0.375, "n_floor_draws": 20, "theta_floor_se": 0.01,
            "delta_orderavg": 0.16, "delta_orderavg_sd": 0.03, "p_two_orderavg": 0.002}
    base.update(kw)
    return base


def test_second_nulls_give_a_verdict_per_statistic():
    # the declared shift unseparated, the order-averaged one separated: each null says so per statistic, and the
    # order-averaged shift is never called a product of drift when it survives the time blocks (X1 issue 14)
    res = _res(strat_p_two=0.355, strat_p_two_orderavg=0.002, tb_p_two=0.679, tb_p_two_orderavg=0.002, n_time_blocks=13)
    out = r.summarize_data(res, verbose=False)
    txt = " ".join(out["lines"])
    assert "the declared shift is not separated from the label-permutation null to begin with" in txt
    assert "the order-averaged shift survives within-stratum relabelling" in txt
    assert "the order-averaged shift survives: drift and slow state at the blocks' scale do not account for it" in txt
    assert "drift or slow state could produce" not in txt and "neither null" not in txt
    assert out["verdict"]["beyond_drift_orderavg"] is True and out["verdict"]["label_linked_beyond_strata_orderavg"] is True
    res = _res(p_two=0.002, tb_p_two=0.3, n_time_blocks=13, tb_p_two_orderavg=0.4)
    txt = " ".join(r.summarize_data(res, verbose=False)["lines"])
    assert "the declared shift does not survive: drift or slow state could produce it" in txt
    assert "the order-averaged shift does not survive" in txt


def test_summary_applies_rule_3_when_the_order_spread_is_large():
    # within two order-SDs of zero (X1 issue 15)
    txt = " ".join(r.summarize_data(_res(delta=0.085, delta_orderavg=0.164, delta_orderavg_sd=0.239), verbose=False)["lines"])
    assert "within two SDs over class orders" in txt and "README rule 3" in txt
    # far from the orders' average although far from zero: an atypical path
    txt = " ".join(r.summarize_data(_res(delta=0.582, p_two=0.002, delta_orderavg=0.024, delta_orderavg_sd=0.229),
                                    verbose=False)["lines"])
    assert "atypical path" in txt
    # GT3-like: the declared shift sits inside the order spread and away from zero: no rule-3 line
    txt = " ".join(r.summarize_data(_res(delta=0.231, p_two=0.005, delta_orderavg=0.335, delta_orderavg_sd=0.083),
                                    verbose=False)["lines"])
    assert "README rule 3" not in txt


def _llm_json(tmp_path, layers):
    path = tmp_path / "lm.json"
    path.write_text(json.dumps({"model": "m", "n_perm": 500, "axes": {"ax": {"layers": layers}}}))
    return str(path)


def _layer(i, oa, p, oa_z, null_sd=0.02, null_sd_z=0.02):
    return {"layer": i, "delta": oa, "p_two": p, "delta_orderavg": oa, "p_two_orderavg": p, "null_mean_orderavg": 0.0,
            "null_sd_orderavg": null_sd, "delta_orderavg_z": oa_z, "null_mean_orderavg_z": 0.0, "null_sd_orderavg_z": null_sd_z}


def test_llm_frames_line_separates_agree_disagree_and_one_frame(tmp_path):
    # X1 issue 13: linear -0.151 beyond its null, z-scored +0.010 inside its null is 'one frame', never 'agree'
    L = [_layer(0, 0.10, 0.002, 0.09), _layer(1, -0.151, 0.002, 0.010), _layer(2, 0.10, 0.002, -0.08)]
    rep = r.summarize(_llm_json(tmp_path, L), verbose=False)
    txt = " ".join(rep["axes"]["ax"]["reading"])
    assert "agree in sign at 1/3 layers, disagree at 1" in txt and "one frame sits inside its null at 1 (layers 1)" in txt
    assert rep["axes"]["ax"]["frames"] == {"agree": 1, "disagree": 1, "one": 1, "neither": 0}


def test_llm_certified_layers_are_counted_by_sign(tmp_path):
    # X1 issue 16: an untrained model whose certified layers are mostly negative is not read as a lift
    L = [_layer(i, -0.02, 0.002, -0.02) for i in range(7)] + [_layer(7 + i, 0.02, 0.002, 0.02) for i in range(2)] \
        + [_layer(9 + i, 0.001, 0.5, 0.001) for i in range(8)]
    rep = r.summarize(_llm_json(tmp_path, L), verbose=False)
    txt = rep["axes"]["ax"]["reading"][0]
    assert "with mixed signs" in txt and "positive at 2/17" in txt and "negative at 7/17" in txt
    assert rep["axes"]["ax"]["certified_orderavg_by_sign"]["negative"][0] == 7
    L = [_layer(i, 0.05, 0.002, 0.05) for i in range(10)]
    txt = r.summarize(_llm_json(tmp_path, L), verbose=False)["axes"]["ax"]["reading"][0]
    assert txt.startswith("label-linked at most depths, the declared classes lifting the exponent")


def test_class_pr_is_each_class_own_dimension():
    rng = np.random.default_rng(0)
    X = np.concatenate([rng.normal(size=(200, 30)) * np.r_[np.ones(k), np.zeros(30 - k)] for k in (2, 10)])
    y = np.repeat([0, 1], 200)
    cp = r.class_pr(X, y)
    assert 1.5 < cp[0] < 2.2 and 8 < cp[1] < 10.5
    d = r.decompose(*_structured(), n_null=4)
    assert set(d["class_pr"]) == set(range(8))


def test_antipode_wording_follows_appendix_m():
    src = open(os.path.join(ROOT, "rung.py")).read() + open(os.path.join(ROOT, "README.md")).read()
    assert "stall" not in src.lower().replace("install", "")
    assert "the share of the climb left after half the circle" in src


def test_decompose_reading_uses_the_papers_names(capsys):
    X, y = _structured()
    d = r.decompose(X, y, n_null=4)
    r._print_decomposition(d)
    out = capsys.readouterr().out
    f = d["delta_split_free"]
    assert "within-class part D" in out and "class-mean part (Tb + Cb + Cx)" in out and "full-set class-mean remainder c" in out
    assert abs(f["D"] + f["Tb"] + f["Cb"] + f["Cx"] - d["delta"]) < 1e-9


# ---------------------------------------------------------------- candidates
def test_candidates_ranks_the_organizing_label_first():
    X, y = _structured(n_per=40, seed=2)
    rnd = np.random.default_rng(9).permutation(y)
    res = r.candidates(X, {"planted": y, "random": rnd}, n_perm=60, seed=0)
    assert res["pick"] == "planted"
    assert res["screen"]["planted"]["z"] > 4 and abs(res["screen"]["random"]["z"]) < 3
    assert res["screen"]["planted"]["p_one"] == pytest.approx(1 / 61)


def test_candidates_on_halves_discovers_and_replicates():
    X, y = _structured(n_per=60, seed=4)
    rnd = np.random.default_rng(1).permutation(y)
    res = r.candidates(X, {"planted": y, "random": rnd}, halves="oddeven", n_perm=60)
    assert res["per_candidate"]["planted"]["replicated"] == [True, True]
    assert not any(res["per_candidate"]["random"]["replicated"])
    assert all(p["pick"] == "planted" and p["confirmed"] for p in res["picks"])


def test_candidates_warns_on_strata_with_nothing_to_permute():
    X, y = _structured(n_per=12, seed=5)
    fine = np.arange(len(y))                                            # one row per stratum
    with pytest.warns(UserWarning, match="little to permute"):
        r.candidates(X, {"a": y, "b": np.random.default_rng(0).permutation(y)}, strata=fine, n_perm=5)


# ---------------------------------------------------------------- the two-term account
def test_finite_identity_is_eq2_for_one_gaussian_class():
    d, n = 30, 50
    mom = {"t": np.array([float(d)]), "T": np.array([[float(d)]]), "m4": np.array([d * d + 2.0 * d])}
    assert r.pr_finite_identity([n], mom) == pytest.approx(d * (n - 1) / (d + n), rel=1e-12)


def test_sigma_reads_shared_and_private_fluctuations():
    rng = np.random.default_rng(0)
    d, K, n = 24, 6, 200
    shared = rng.normal(size=(d, 4))
    Xs = np.concatenate([rng.normal(size=(n, 4)) @ shared.T for _ in range(K)])
    Xp = np.concatenate([rng.normal(size=(n, 4)) @ np.eye(d)[:, 4 * c:4 * c + 4].T for c in range(K)])
    y = np.repeat(np.arange(K), n)
    assert r.split_half_sigma(Xs, y)[0] > 0.9
    assert r.split_half_sigma(Xp, y)[0] < 0.05


def test_account_on_identical_and_on_private_classes():
    rng = np.random.default_rng(1)
    d, K, n = 24, 6, 300
    shared = rng.normal(size=(d, 4))
    y = np.repeat(np.arange(K), n)
    Xs = np.concatenate([rng.normal(size=(n, 4)) @ shared.T + 0.05 * rng.normal(size=(n, d)) for _ in range(K)])
    a = r.account(Xs, y, n_null=6, n_comp=40, heldout=False)
    assert abs(a["within_pred"]["mean_field"]) < 0.1 and abs(a["measured"]["within"]) < 0.15
    Xp = np.concatenate([rng.normal(size=(n, 4)) @ np.eye(d)[:, 4 * c:4 * c + 4].T for c in range(K)])
    b = r.account(Xp, y, n_null=6, n_comp=40, heldout=False)
    assert b["within_pred"]["mean_field"] > 0.9 and b["measured"]["within"] > 0.5
    assert abs(b["within_pred"]["finite"] - b["measured"]["within"]) < 0.1     # the identity at the rungs' counts tracks it


def test_classmean_term_is_zero_without_class_means():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(240, 10)); y = np.repeat(np.arange(8), 30)
    for c in range(8):
        X[y == c] -= X[y == c].mean(0)
    slope, c_full, _ = r._classmean_term_rungs(X, y, (1, 2, 3, 4, 6, 8))
    assert abs(c_full) < 1e-9 and abs(slope) < 1e-9


# ---------------------------------------------------------------- feature ladder and blocks
def test_feature_ladder_in_random_order_sits_at_its_floor():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 600)) @ rng.normal(size=(600, 600)) * 0.1 + rng.normal(size=(200, 600))
    f = r.feature_ladder(X, order=rng.permutation(600), sizes=(50, 100, 200, 400), n_floor_draws=20)
    assert abs(f["delta"]) < 0.05 and abs(f["delta_control"]) < 0.05


def test_feature_ladder_reads_spatial_groups():
    # 60 latents, each shared by 10 neighbouring features: a neighbour rung of n features holds about n/10 latents, so its
    # PR keeps climbing, while random subsets hold most of the 60 latents by n = 100 and saturate: the neighbour ladder
    # starts low and climbs, a positive shift (as a class ladder does when classes bring new directions)
    rng = np.random.default_rng(4)
    n, groups, per = 300, 60, 10
    lat = rng.normal(size=(n, groups))
    X = np.repeat(lat, per, axis=1) + 0.3 * rng.normal(size=(n, groups * per))
    pos = np.repeat(np.arange(groups), per)[:, None] + 0.01 * rng.normal(size=(groups * per, 1))
    f = r.feature_ladder(X, positions=pos, sizes=(20, 50, 100, 200, 400), n_anchors=20, n_floor_draws=20)
    assert f["delta"] > 0.3 and abs(f["delta_control"]) < 0.05
    assert f["pr_obs"][0] < f["pr_floor"][0]                         # the first rung holds few latents


def test_blocks_keep_the_shift_when_sorted_and_lose_it_when_random():
    rng = np.random.default_rng(5)
    K, n_per, d = 8, 60, 512
    pref = rng.uniform(0, 2 * np.pi, d)
    ang = 2 * np.pi * np.arange(K) / K
    y = np.repeat(np.arange(K), n_per)
    tun = np.exp(2 * np.cos(ang[y][:, None] - pref[None]))                         # direction tuning
    X = tun * (1 + 0.6 * rng.normal(size=(len(y), d))) + 0.3 * rng.normal(size=(len(y), d))
    b = r.blocks(X, y, r.preferred_phase(X, y, 1), sizes=(1, 32), k_orders=0)
    big = b["rows"][1]
    assert big["sorted"]["delta"] > big["random"]["delta"] + 0.05


def test_block_features_averages_consecutive_groups():
    X = np.arange(24, dtype=float).reshape(2, 12)
    B = r.block_features(X, 4, order=np.arange(12))
    assert B.shape == (2, 3) and np.allclose(B[0], [1.5, 5.5, 9.5])


# ---------------------------------------------------------------- loaders
def test_npz_and_mat_loaders(tmp_path):
    a = np.arange(12.0).reshape(3, 4)
    np.savez(tmp_path / "z.npz", X=a, y=np.arange(3))
    assert np.array_equal(r._load_array(str(tmp_path / "z.npz") + ":X"), a)
    with pytest.raises(SystemExit, match="holds 2 arrays"):
        r._load_array(str(tmp_path / "z.npz"))
    from scipy.io import savemat
    savemat(tmp_path / "m.mat", {"resp": a})
    assert np.array_equal(r._load_array(str(tmp_path / "m.mat")), a)
    assert np.array_equal(r._load_array(str(tmp_path / "m.mat") + ":resp"), a)


def test_nwb_counts_on_a_small_file(tmp_path):
    h5py = pytest.importorskip("h5py")
    f = h5py.File(tmp_path / "s.nwb", "w")
    u = f.create_group("units")
    u["spike_times"] = np.array([0.1, 0.2, 1.1, 2.5, 0.15, 1.6, 1.7, 2.2])     # unit 0: 4 spikes, unit 1: 4 spikes
    u["spike_times_index"] = np.array([4, 8])
    u["id"] = np.array([10, 11])
    u["location"] = np.array([b"VISp", b"VISl"])
    t = f.create_group("intervals/trials")
    t["start_time"] = np.array([0.0, 1.0, 2.0])
    t["stop_time"] = np.array([0.5, 1.8, 3.0])
    t["direction"] = np.array([90.0, 0.0, np.nan])
    f.close()
    out = r.nwb_counts(str(tmp_path / "s.nwb"), "trials", "direction")
    assert out["X"].tolist() == [[2, 1], [1, 2]] and out["labels"].tolist() == [1, 0] and out["label_values"] == [0.0, 90.0]
    only = r.nwb_counts(str(tmp_path / "s.nwb"), "trials", "direction", units_where=("location", "VISp"))
    assert only["X"].tolist() == [[2], [1]] and only["unit_ids"].tolist() == [10]
    w = r.nwb_counts(str(tmp_path / "s.nwb"), "trials", "direction", window=(0.0, 0.12))
    assert w["X"].tolist() == [[1, 0], [1, 0]]


# ---------------------------------------------------------------- command line
def _cli(*args):
    return subprocess.run([sys.executable, os.path.join(ROOT, "rung.py"), *args], capture_output=True, text=True)


def test_cli_candidates_account_rungs_and_features(tmp_path):
    X, y = _structured(n_per=40, seed=6)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "dir.npy", y)
    np.save(tmp_path / "rnd.npy", np.random.default_rng(0).permutation(y))
    res = _cli("data", str(tmp_path / "X.npy"), str(tmp_path / "dir.npy"), "--n-perm", "20", "--k-orders", "4",
               "--rungs", "1,2,4,8", "--account", "--no-heldout", "--candidates", str(tmp_path / "rnd.npy"),
               "--out", str(tmp_path / "r.json"))
    assert res.returncode == 0, res.stderr
    d = json.load(open(tmp_path / "r.json"))
    assert d["rung_classes"] == [1, 2, 4, 8] and "account" in d and d["candidates"]["pick"] == "dir"
    assert "two terms, mean field" in res.stdout
    s = _cli("summarize", str(tmp_path / "r.json"))
    assert s.returncode == 0 and "two-term account" in s.stdout and "candidates ranked by z" in s.stdout
    np.save(tmp_path / "order.npy", np.arange(X.shape[1]))
    f = _cli("features", str(tmp_path / "X.npy"), "--order", str(tmp_path / "order.npy"), "--sizes", "5,10,20,40",
             "--floor-draws", "5", "--out", str(tmp_path / "f.json"), "--plot", str(tmp_path / "f.png"))
    assert f.returncode == 0, f.stderr
    assert os.path.getsize(tmp_path / "f.png") > 1000 and "feature ladder" in f.stdout


def test_cli_help_shows_defaults():
    res = _cli("data", "--help")
    assert res.returncode == 0 and "(default: 500)" in res.stdout and "--paper-floor" in res.stdout


def test_head_projection_map_and_scores_on_a_planted_head():
    assert {"gpt2", "gpt_neox", "llama", "gpt_neo"} <= set(r._HEAD_PROJ)
    rng = np.random.default_rng(7)
    y = np.repeat(np.arange(8), 16)
    H = rng.normal(size=(1, 3, len(y), 32))
    for c in range(8):                                                     # head 1 fluctuates differently per class
        H[0, 1, y == c, 4 * c:4 * c + 4] += 3 * rng.normal(size=(16, 4))
    rows = r.head_scores(H, y, [0], n_perm=20, k_orders=10, split=True)
    assert max(rows, key=lambda q: q["z"])["head"] == "0.1" and "within" in rows[1]


# ---------------------------------------------------------------- regressions against the paper's artifacts
def _raw(name):
    p = os.path.join(RAW, name)
    if not os.path.exists(p):
        pytest.skip(f"raw/{name} not present")
    return p


@data_tests
def test_feature_ladder_reproduces_table_4_gt3():
    d = np.load(_raw("gratings_drifting_GT3_2019_04_05_1.npy"), allow_pickle=True).item()
    f = r.feature_ladder(np.asarray(d["sresp"], np.float64).T, positions=np.array([s["med"] for s in d["stat"]], float))
    art = next(q for q in json.load(open(os.path.join(DATA, "neuron_zoom_spatial.json")))["rows"] if q["name"].startswith("gratings_drifting_GT3"))
    assert f["theta_obs"] == pytest.approx(art["theta_spatial"], abs=1e-9)
    assert f["theta_floor"] == pytest.approx(art["theta_floor"], abs=1e-9)
    assert f["delta_control"] == pytest.approx(art["delta_control_rand_vs_rand"], abs=1e-9)


@data_tests
def test_paper_floor_replays_the_allen_and_macaque_artifacts():
    sys.path.insert(0, os.path.join(ROOT, "scripts_canonical"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("E", os.path.join(ROOT, "scripts_canonical", "allen_engine.py"))
    E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    _raw("allen_counts")
    S = E.load_session(715093703)
    V = E.pop_view(S, "VISp", E.populations(S)["VISp"])
    a = r.zoom(V["X"], V["lab"], n_perm=0, k_orders=0, n_floor_draws=10, floor_seed=42, floor_by_rung=True)
    art = json.load(open(os.path.join(DATA, "allen_sessions", "session_715093703.json")))["areas"]["VISp"]["pooled"]
    assert a["delta"] == pytest.approx(art["delta"], abs=1e-12)


@data_tests
def test_candidates_reproduce_the_allen_axis_screen():
    sys.path.insert(0, os.path.join(ROOT, "scripts_canonical"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("E", os.path.join(ROOT, "scripts_canonical", "allen_engine.py"))
    E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
    _raw("allen_running")
    sid = 715093703
    S = E.load_session(sid)
    run = np.load(os.path.join(RAW, "allen_running", f"session_{sid}.npz"))
    V = E.pop_view(S, "VISp", E.populations(S)["VISp"], extra={"speed": run["speed"]})
    sp = V["speed"].astype(float); o = np.argsort(sp, kind="stable"); tert = np.empty(len(sp), int); tert[o] = (np.arange(len(sp)) * 3) // len(sp)
    rnd = np.empty(len(sp), int); rnd[np.random.default_rng(0).permutation(len(sp))] = np.arange(len(sp)) % 8
    L = {"direction": V["lab"], "tf": V["tf"], "running": tert, "random": rnd}
    st = {"direction": L["tf"] * 3 + L["running"], "tf": L["direction"] * 3 + L["running"],
          "running": L["direction"] * 5 + L["tf"], "random": L["direction"] * 5 + L["tf"]}
    res = r.candidates(V["X"], L, strata=st, halves=E.halves(V["cell"]), n_perm=200, n_floor_draws=10, floor_seed=42)
    art = next(q for q in json.load(open(os.path.join(DATA, "allen_axis_screen.json")))["rows"] if q["session"] == sid)
    for h in (0, 1):
        for k in L:
            assert res["halves"][h][k]["z"] == pytest.approx(art[f"half{h}"][k]["z"], abs=1e-9)
            assert res["halves"][h][k]["delta"] == pytest.approx(art[f"half{h}"][k]["delta"], abs=1e-9)


@data_tests
def test_account_reproduces_the_gt3_two_term_artifact():
    d = np.load(_raw("gratings_drifting_GT3_2019_04_05_1.npy"), allow_pickle=True).item()
    X = np.ascontiguousarray(np.asarray(d["sresp"], np.float32).T); X /= X.std() + 1e-9
    phi = np.asarray(d["istim"], float)
    bins = lambda nb: np.clip(np.digitize(phi % (2 * np.pi), np.linspace(0, 2 * np.pi, nb + 1)) - 1, 0, nb - 1)
    lab, sub48 = bins(8), bins(48)
    h = np.zeros(len(phi), int)
    for b in np.unique(sub48):
        idx = np.where(sub48 == b)[0]; h[idx[1::2]] = 1
    a = r.account(X.astype(np.float64), lab, halves=h, n_null=10, seed=42, floor_by_rung=True, heldout=False)
    tt = next(q for q in json.load(open(os.path.join(DATA, "twoterm_imaging.json")))["rows"] if q["name"] == "gratings_drifting_GT3_2019_04_05_1")["B2"]
    # the generator built sigma's residuals in float32 (rung: float64), so sigma and the mean-field law agree to float32 rounding;
    # Eq. 3's term was computed in float64 there and replays to 1e-9
    assert a["sigma"] == pytest.approx(tt["sigma"], abs=1e-6)
    assert a["within_pred"]["mean_field"] == pytest.approx(tt["within_pred"], abs=1e-6)
    assert a["class_mean_pred"] == pytest.approx(tt["classmean_pred"], abs=1e-9)
    assert a["measured"]["within"] == pytest.approx(tt["t9_within"], abs=1e-5)       # the table's Gram is float32


@pytest.mark.skipif(os.environ.get("RUNG_SLOW_TESTS") != "1", reason="GPT-2 head replay: set RUNG_SLOW_TESTS=1")
def test_heads_paper_seeds_replay_appendix_i():
    pytest.importorskip("torch"); pytest.importorskip("transformers")
    res = r.llm_heads("openai-community/gpt2", os.path.join(ROOT, "axes", "ioi_names.json"), paper_seeds=True,
                      known=["9.6", "9.9", "10.0", "10.10", "10.6", "10.2", "10.1", "11.2", "9.7", "9.0", "11.9"])
    art = json.load(open(os.path.join(DATA, "llm_ioi.json")))
    assert res["AUC_z"] == pytest.approx(art["AUC_z"], abs=1e-9)
    assert res["AUC_classmean_frac"] == pytest.approx(art["AUC_classmean_frac"], abs=1e-9)
