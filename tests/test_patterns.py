"""Compare bounded PrefixSpan and the inference path to independent small examples."""

from collections import Counter
from itertools import combinations
from math import ceil

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from atm.mining.discriminative import adjust, adjust_holdout, effect, fold_results
from atm.mining.prefixspan import SequenceIndex, mine
from atm.mining.stratified import cmh


@pytest.mark.parametrize("max_gap", [0, 1, 3])
@pytest.mark.parametrize("min_support", [0.25, 0.5, 1])
def test_prefixspan_matches_brute_force(max_gap: int, min_support: float) -> None:
    data = [["a", "x", "a", "b", "c"], ["a", "c", "b", "c"], ["x", "b", "a", "c"], ["b", "b", "b", "a"]]
    brute = Counter()
    for sequence in data:
        present = set()
        for n in range(1, 5):
            for indices in combinations(range(len(sequence)), n):
                if all(b - a - 1 <= max_gap for a, b in zip(indices, indices[1:])):
                    present.add(tuple(sequence[idx] for idx in indices))
        brute.update(present)
    expected = {pattern: count for pattern, count in brute.items() if count >= ceil(min_support * len(data))}
    assert mine(data, min_support, 4, max_gap) == expected
    projected = dict(SequenceIndex(data, max_gap).supports([*expected, ("unseen",)]))
    assert {pattern: len(ends) for pattern, ends in projected.items()} == {**expected, ("unseen",): 0}


def test_cmh_matches_hand_computed_score() -> None:
    tables = np.array([[[8, 2], [3, 7]], [[6, 4], [2, 8]], [[3, 0], [0, 0]]])
    odds, p_value, n_used = cmh(tables)
    expected = 10 * 11 / 20 + 10 * 8 / 20
    variance = (10 * 10 * 11 * 9 + 10 * 10 * 8 * 12) / (20**2 * 19)
    assert n_used == 2
    assert odds == pytest.approx((8 * 7 + 6 * 8) / (2 * 3 + 4 * 2))
    assert p_value == pytest.approx(chi2.sf((8 + 6 - expected)**2 / variance, 1))


def test_adjustments_and_failure_orientation() -> None:
    frame = pd.DataFrame({"p": [0.01, 0.025, 0.20]})
    adjust(frame, ("p",))
    assert frame["p_bonf"].tolist() == pytest.approx([0.03, 0.075, 0.60])
    assert frame["p_bh"].tolist() == pytest.approx([0.03, 0.0375, 0.20])
    result = effect(np.array([[8, 2], [3, 7]]))
    assert result["support_diff"] == pytest.approx(0.5)
    assert result["odds_ratio"] == pytest.approx((8 * 7) / (2 * 3))


def test_held_out_changes_cannot_change_discovery() -> None:
    data = pd.DataFrame({
        "instance_id": [f"task-{i}" for i in range(12)], "failed": [0, 1] * 6,
        "tokens": [["edit", "view"], ["test", "edit"]] * 6,
        "difficulty": ["hard", "hard", "medium", "medium", "easy", "easy"] * 2,
        "framework": ["a", "b"] * 6,
    })
    train, test = data.iloc[:6], data.iloc[6:].copy()
    params = {"min_support": 0.25, "max_pattern_len": 3, "max_gap": 1}
    before, _ = fold_results(train, test, 1, params)
    test["failed"] = 1 - test["failed"]
    test["tokens"] = [["future-only-token"]] * len(test)
    after, _ = fold_results(train, test, 1, params)
    for left, right in zip(before, after, strict=True):
        compare = ["pattern", "train_support_fail", "train_support_success", "train_fisher_p"]
        assert [left[k] for k in compare] == [right[k] for k in compare]
        assert "future-only-token" not in right["pattern"]
    assert any(a["support_diff"] != b["support_diff"] for a, b in zip(before, after, strict=True))


def test_holdout_bonferroni_budgets_alpha_equally_despite_unequal_candidate_counts() -> None:
    frame = pd.DataFrame({"fold": [1, 1, 2], "p": [0.01, 0.025, 0.20]})
    adjust_holdout(frame, ("p",), ["fold"])
    assert frame["p_bonf"].tolist() == pytest.approx([0.04, 0.10, 0.40])
    assert frame["p_bh"].tolist() == pytest.approx([0.03, 0.0375, 0.20])
