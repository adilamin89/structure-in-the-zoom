"""Tests for rung 1.8.1: a permutation p that the budget cannot resolve below 0.05 is read as unresolved, never as
unseparated, absorbed or not surviving."""
import numpy as np

import rung as r


def _res(n_perm, **kw):
    base = {"theta_obs": 0.2, "theta_floor": 0.1, "delta": 0.1, "n_classes": 8, "rung_sizes": [1, 2, 3], "n_perm": n_perm,
            "z": 26.5, "p_two": 1.0 / (n_perm + 1), "n_floor_draws": 20, "theta_floor_se": 0.01}
    base.update(kw)
    return base


def test_unresolvable_budget_says_so():
    # 9 permutations: the smallest p is 0.100, so a huge z must not read as 'not separated'
    out = r.summarize_data(_res(9), verbose=False)
    txt = " ".join(out["lines"])
    assert "not separated" not in txt
    assert "unresolved: 9 permutations cannot give p below 0.100 (no permutation reached the observed value)" in txt
    assert out["verdict"]["certified"] is False and out["verdict"]["p_unresolved"] is True
    # one permutation reached it: still unresolved, and the count is printed
    txt = " ".join(r.summarize_data(_res(9, p_two=0.2), verbose=False)["lines"])
    assert "(1 of 9 reached it)" in txt


def test_resolving_budget_keeps_the_verdicts():
    out = r.summarize_data(_res(19), verbose=False)                   # 1/20 = 0.05: still cannot give p < 0.05
    assert "unresolved: 19 permutations" in " ".join(out["lines"])
    out = r.summarize_data(_res(200, p_two=0.4), verbose=False)       # resolves: an honest 'not separated'
    assert "not separated from the label-permutation null" in " ".join(out["lines"]) and "p_unresolved" not in out["verdict"]
    out = r.summarize_data(_res(200), verbose=False)
    assert "label-linked at p < 0.05" in " ".join(out["lines"])


def test_second_nulls_and_circular_shift_at_an_unresolvable_budget():
    res = _res(9, strat_p_two=0.3, tb_p_two=0.5, n_time_blocks=4, shift_p_two=0.4)
    out = r.summarize_data(res, verbose=False)
    txt = " ".join(out["lines"])
    assert "ABSORBED" not in txt and "does not survive" not in txt and "does NOT survive" not in txt
    assert txt.count("cannot be read at 9 permutations") == 2
    assert out["verdict"]["label_linked_beyond_strata"] is None and out["verdict"]["beyond_drift"] is None
    assert out["verdict"]["survives_circular_shift"] is None


def test_zoom_with_few_permutations_end_to_end():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(8), 32)
    X = rng.standard_normal((256, 32)) * (1 + 2 * (y[:, None] == np.arange(32)[None, :] // 4))
    res = r.zoom(X, y, n_perm=9, k_orders=3, n_floor_draws=3)
    assert res["p_two"] >= 0.1
    assert "unresolved" in " ".join(r.summarize_data(res, verbose=False)["lines"])
