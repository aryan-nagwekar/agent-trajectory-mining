"""Difficulty-adjusted association and the run-length confounding check."""

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.contingency_tables import StratifiedTable

BINS = ("hard", "medium", "easy")


def informative(table: np.ndarray) -> bool:
    """A two-by-two table contributes only when both margins vary."""
    return bool(table.sum(axis=0).min() > 0 and table.sum(axis=1).min() > 0)


def cmh(tables: np.ndarray) -> tuple[float, float, int]:
    """Return the pooled failure odds ratio, nominal CMH p-value, and strata count."""
    usable = [table for table in tables if informative(table)]
    if not usable:
        return np.nan, 1.0, 0
    with np.errstate(divide="ignore", invalid="ignore"):
        model = StratifiedTable(usable, shift_zeros=False)
        return float(model.oddsratio_pooled), float(model.test_null_odds().pvalue), len(usable)


class TaskClusterScore:
    """Score sensitivity with difficulty fixed effects and task-cluster variance."""

    def __init__(self, failed: np.ndarray, strata: np.ndarray, task_ids: np.ndarray):
        self.failed = failed
        self.strata = strata
        self.size = np.bincount(strata, minlength=len(BINS))
        self.task = np.unique(task_ids, return_inverse=True)[1]
        self.n_tasks = len(np.unique(self.task))
        self.residual = failed - np.bincount(
            strata, weights=failed, minlength=len(BINS)
        )[strata] / self.size[strata]

    def p_value(self, present: np.ndarray) -> float:
        if self.n_tasks < 2:
            return 1.0
        within_bin = np.bincount(self.strata, weights=present, minlength=len(BINS))
        centered = present - within_bin[self.strata] / self.size[self.strata]
        score = np.bincount(self.task, weights=centered * self.residual)
        variance = self.n_tasks / (self.n_tasks - 1) * np.square(score - score.mean()).sum()
        return float(2 * norm.sf(abs(score.sum()) / np.sqrt(variance))) if variance > 0 else 1.0


def length_correlations(runs: pd.DataFrame) -> pd.DataFrame:
    """Pearson r (point-biserial) for raw step count versus the failure flag."""
    rows = []
    for name, group in [("overall", runs)] + [(b, runs.loc[runs["difficulty"] == b]) for b in BINS]:
        rows.append({
            "stratum": name, "runs": len(group), "tasks": group["instance_id"].nunique(),
            "failed": int(group["failed"].sum()), "pearson_r": group["n_steps"].corr(group["failed"]),
        })
    centered = runs[["n_steps", "failed"]] - runs.groupby("difficulty")[["n_steps", "failed"]].transform("mean")
    rows.append({
        "stratum": "within_bin_centered", "runs": len(runs), "tasks": runs["instance_id"].nunique(),
        "failed": int(runs["failed"].sum()), "pearson_r": centered["n_steps"].corr(centered["failed"]),
    })
    logging.info("Length correlation overall %.3f, after centering within bins %.3f",
                 rows[0]["pearson_r"], rows[-1]["pearson_r"])
    return pd.DataFrame(rows)
