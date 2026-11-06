"""rung 1.10: head-score AUCs beside their random-set null and the write-norm baseline; rung heads for any per-unit stack;
the untrained comparator (random weights from the config, checked to differ from the pretrained ones); the report's baseline."""
import json
import subprocess
import sys

import numpy as np
import pytest

import rung as r


def test_auc_null_matches_auc_and_the_mann_whitney_width():
    rng = np.random.default_rng(0)
    scores = np.round(rng.standard_normal(144), 1)                         # rounded: ties
    pos = np.zeros(144, bool); pos[rng.choice(144, 7, replace=False)] = True
    a = r._auc_null(scores, pos, n_draws=20000)
    assert a["auc"] == pytest.approx(r._auc(scores, pos), abs=1e-12)
    assert a["null_mean"] == pytest.approx(0.5, abs=0.005)
    assert a["null_sd"] == pytest.approx(np.sqrt((144 + 1) / (12 * 7 * 137)), abs=0.005)   # 0.112 for 7 of 144
    scores[pos] = 10.0                                                      # the known set on top
    b = r._auc_null(scores, pos, n_draws=2000)
    assert b["auc"] == 1.0 and b["p"] == pytest.approx(1 / 2001)


def _stack(seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(6), 10)
    H = rng.standard_normal((2, 4, 60, 12))
    H[1, 2] *= 1 + 2 * (y[:, None] == np.arange(12)[None, :] % 6)        # unit 1.2 writes class-specific fluctuations
    return H, y


def test_heads_command_scores_any_stack_with_nulls_ranks_and_norm(tmp_path):
    H, y = _stack()
    np.save(tmp_path / "H.npy", H); np.save(tmp_path / "y.npy", y)
    subprocess.run([sys.executable, "-m", "rung", "heads", "H.npy", "y.npy", "--layer-names", "4,9", "--known", "9.2",
                    "--n-perm", "10", "--k-orders", "4", "--floor-draws", "3", "--out", "h.json"], cwd=tmp_path, check=True,
                   capture_output=True, text=True)
    d = json.load(open(tmp_path / "h.json"))
    assert [h["head"] for h in d["heads"]][:5] == ["4.0", "4.1", "4.2", "4.3", "9.0"] and d["rank_z"] == {"9.2": 1}
    assert all("norm" in h for h in d["heads"]) and "AUC_norm" in d
    assert set(d["AUC_z_null"]) == {"null_mean", "null_sd", "p"} and d["AUC_z"] == 1.0


def test_report_shows_the_baseline_runs_aucs(tmp_path):
    H, y = _stack()
    rows = r.head_scores(H, y, [0, 1], n_perm=10, k_orders=4, n_floor_draws=3, split=False)
    meta = {"model": "toy", "axis": "toy", "classes": list(range(6)), "n_prompts": 60, "n_perm": 10, "k_orders": 4}
    trained = r._heads_result(rows, ["1.2"], meta, n_null_sets=500)
    base = r._heads_result(r.head_scores(np.random.default_rng(9).standard_normal(H.shape), y, [0, 1], n_perm=10, k_orders=4,
                                         n_floor_draws=3, split=False), ["1.2"], dict(meta, untrained_seed=1), n_null_sets=500)
    (tmp_path / "t.json").write_text(json.dumps(trained)); (tmp_path / "b.json").write_text(json.dumps(base))
    page = open(r.report(str(tmp_path / "t.json"), baseline=str(tmp_path / "b.json"))).read()
    assert f"baseline run {base['AUC_z']:.3f}" in page and "random 1-sets" in page and "the write norm alone reaches AUC" in page
    assert "<h2>Top heads by z</h2>" in page and "<th>within-class part</th>" in page and "can cancel" in page and "class-mean share tracks" in page


def test_untrained_weights_differ_from_the_pretrained_and_repeat_by_seed(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch"); transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(n_layer=2, n_head=2, n_embd=16, vocab_size=50, n_positions=32)
    torch.manual_seed(123)
    transformers.GPT2LMHeadModel(cfg).save_pretrained(tmp_path / "tiny")                # stands in for a pretrained model
    ref = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "tiny").get_input_embeddings().weight
    load = lambda seed: r._from_pretrained(transformers.AutoModelForCausalLM, str(tmp_path / "tiny"), torch.float32, untrained=seed)
    w1, w1b, w2 = (load(s_).get_input_embeddings().weight for s_ in (1, 1, 2))
    assert not torch.equal(w1, ref) and torch.equal(w1, w1b) and not torch.equal(w1, w2)
    # an initialization that silently keeps the pretrained weights must be caught
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_config",
                        lambda c: transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "tiny"))
    with pytest.raises(RuntimeError, match="random initialization did not run"):
        load(1)


def _constant_classes(n_flat, noise=1.0, seed=1):
    labels = np.repeat(np.arange(8), 16)
    X = np.repeat(np.eye(8), 16, axis=0)[:, :8].astype(float)
    X[labels >= n_flat] += noise * np.random.default_rng(seed).standard_normal(((labels >= n_flat).sum(), 8))
    return X, labels


@pytest.mark.parametrize("n_flat,noise", [(7, 1.0), (7, 1e-3), (8, 0.0)])
def test_constant_classes_mark_the_split_and_account_undefined(n_flat, noise):
    # a layer read at a token every carrier of a class shares: classes with no within-class variance. The split and the
    # overlap law are undefined: the result says so with its reason and no number, never a crash or a silent NaN
    X, labels = _constant_classes(n_flat, noise)
    for res in (r.decompose(X, labels), r.account(X, labels)):
        assert "no within-class variance" in res["undefined"]
    assert np.isnan(r.decompose(X, labels)["delta_split_free"]["D"])


def test_a_constant_class_after_varied_ones_still_splits_but_the_account_is_undefined():
    X, labels = _constant_classes(4)                          # classes 0-3 constant, 4-7 varied
    order = [4, 5, 6, 7, 0, 1, 2, 3]                          # varied classes first: every rung has within-class variance
    assert np.isfinite(r.decompose(X, labels, order=order)["delta_split_free"]["D"])
    assert "in a split half" in r.account(X, labels, order=order)["undefined"]     # the split is defined; the overlap law is not
    assert "no within-class variance" in r.account(X, labels)["undefined"]        # declared order starts on a constant class


def test_two_classes_give_no_reading_and_no_crash():
    rng = np.random.default_rng(0)
    X, y = rng.standard_normal((40, 6)), np.repeat([0, 1], 20)
    z = r.zoom(X, y, n_perm=5)
    assert "at least three rungs" in z["degenerate"] and np.isnan(z["delta"])
    assert r.summarize_data(z, verbose=False)["lines"][0].startswith("no reading:")
    assert "at least three classes" in r.account(X, y)["undefined"] and "undefined" in r.sectors(X, y)


@pytest.mark.parametrize("case", ["7flat", "allflat", "2class"])
def test_the_command_line_finishes_on_degenerate_data(tmp_path, case):
    labels = np.repeat(np.arange(8), 16)
    X = np.repeat(np.eye(8), 16, axis=0)[:, :8].astype(float)
    if case == "7flat":
        X[labels == 7] += np.random.default_rng(1).standard_normal((16, 8))
    if case == "2class":
        X, labels = np.random.default_rng(2).standard_normal((128, 8)), labels % 2
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "y.npy", labels)
    run = lambda *a: subprocess.run([sys.executable, "-m", "rung", *a], cwd=tmp_path, capture_output=True, text=True)
    p = run("data", "X.npy", "y.npy", "--n-perm", "10", "--k-orders", "3", "--floor-draws", "3", "--decompose", "--account",
            "--sectors", "--cov-floors", "--n-bias", "2", "--out", "r.json", "--plot", "r.png")
    assert p.returncode == 0, p.stderr[-400:]
    assert run("summarize", "r.json").returncode == 0 and run("report", "r.json").returncode == 0


def test_flagged_heads_and_the_one_line_error(tmp_path):
    rng = np.random.default_rng(0); y = np.repeat(np.arange(6), 10)
    H = rng.standard_normal((1, 3, 60, 8)); H[0, 1] = 0.0; H[0, 2] = np.eye(8)[y]
    rows = r.head_scores(H, y, [0], n_perm=5, k_orders=3, n_floor_draws=3)
    assert [h.get("flag") for h in rows] == [None, "no variance", "constant within every class"]
    p = subprocess.run([sys.executable, "-m", "rung", "data", "missing.npy", "y.npy"], cwd=tmp_path, capture_output=True, text=True)
    assert p.returncode == 2 and p.stderr.strip().startswith("rung: error:") and "Traceback" not in p.stderr
