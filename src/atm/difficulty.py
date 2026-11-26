"""Describe task difficulty from the observed cross-submission solve rate."""

from fractions import Fraction

import pandas as pd


def task_difficulty(runs: pd.DataFrame, cutoffs: list[str]) -> pd.DataFrame:
    """Estimate difficulty for descriptive stratification only."""
    low, high = map(Fraction, cutoffs)
    if not 0 < low < high < 1:
        raise ValueError("difficulty cutoffs must be increasing within (0, 1)")
    counts = runs.groupby("instance_id", sort=True)["failed"].agg(n_runs="size", n_failed="sum").reset_index()
    counts["n_resolved"] = counts["n_runs"] - counts["n_failed"]
    counts["solve_rate"] = counts["n_resolved"] / counts["n_runs"]
    hard = counts["n_resolved"] * low.denominator <= low.numerator * counts["n_runs"]
    medium = counts["n_resolved"] * high.denominator <= high.numerator * counts["n_runs"]
    counts["difficulty"] = "easy"
    counts.loc[medium, "difficulty"] = "medium"
    counts.loc[hard, "difficulty"] = "hard"
    return counts.drop(columns="n_failed")
