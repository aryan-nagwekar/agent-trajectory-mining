"""Verify conformal rank, strict thresholds, and cost/stop accounting."""

from fractions import Fraction

import numpy as np
import pandas as pd

from atm.calibration import conformal_threshold, first_stops, summarize
from atm.cohorts import pair_auc


def test_conformal_uses_one_based_rank_and_suppresses_unfeasible_stop() -> None:
    assert conformal_threshold([0.1, 0.9, 0.4, 0.7], Fraction(1, 5)) == (0.9, 4)
    assert conformal_threshold([0.1, 0.9, 0.4, 0.7], Fraction(1, 10)) == (np.inf, 5)


def test_split_conformal_null_stays_near_target() -> None:
    rng = np.random.default_rng(42)
    stopped = 0
    for calibration in rng.uniform(size=(4000, 121)):
        threshold, _ = conformal_threshold(calibration[:-1], Fraction(1, 20))
        stopped += int(calibration[-1] > threshold)
    assert stopped / 4000 <= 0.06


def test_first_stop_and_compute_denominators() -> None:
    population = pd.DataFrame({"run_id": ["a", "b", "c"], "failed": [1, 0, 0], "total_chars": [100, 60, 40],
                               "n_steps": [20, 10, 5]})
    scored = pd.DataFrame({"run_id": ["a", "a", "b", "c"], "checkpoint": [5, 10, 5, 5],
                           "score": [.85, .90, .8, .99], "prefix_chars": [20, 45, 23, 40]}).merge(
        population, on="run_id", validate="many_to_one"
    )
    stopped = first_stops(scored, {5: .8, 10: .8})
    assert list(zip(stopped.run_id, stopped.checkpoint)) == [("a", 5)]
    stats = summarize(population, population.query("n_steps > 5"), stopped)
    assert stats["saved_chars"] == stats["saved_failed_chars"] == 80
    assert stats["total_chars"] == 200 and stats["failed_chars"] == 100
    assert stats["successes"] == 2 and stats["eligible_successes"] == 1 and stats["false_stops"] == 0


def test_conditional_auc_keeps_only_same_task_pairs() -> None:
    sample = pd.DataFrame({"task": ["a", "a", "b", "b", "c"], "failed": [0, 1, 0, 1, 1],
                           "score": [.1, .2, .9, .8, 1.]})
    assert pair_auc(sample, "task") == (0.5, 2)
