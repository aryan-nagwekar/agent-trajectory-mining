"""Separate a fixed long-run cohort from the deployment and subgroup views."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from atm.predict import metrics, write_summary


def pair_auc(frame: pd.DataFrame, key: str) -> tuple[float, int]:
    """Concordance within groups, weighted by their positive-negative pair counts."""
    concordance, pairs = 0.0, 0
    for _, rows in frame.groupby(key, sort=False):
        cases = int(rows["failed"].sum())
        count = cases * (len(rows) - cases)
        if count:
            concordance += count * roc_auc_score(rows["failed"], rows["score"])
            pairs += count
    return (concordance / pairs if pairs else np.nan), pairs


def breakdown(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups, contrasts = [], []
    for (checkpoint, fold), subset in data.groupby(["checkpoint", "fold"]):
        for column in ("difficulty", "framework"):
            for name, rows in subset.groupby(column):
                cases = int(rows["failed"].sum())
                groups.append({
                    "checkpoint": checkpoint, "fold": fold, "by": column, "group": name,
                    "runs": len(rows), "failures": cases,
                    "auroc": roc_auc_score(rows["failed"], rows["score"]) if 0 < cases < len(rows) else np.nan,
                })
        overall = roc_auc_score(subset["failed"], subset["score"])
        pairs = int(subset["failed"].sum()) * int((1 - subset["failed"]).sum())
        within_bin, bin_pairs = pair_auc(subset, "difficulty")
        within_task, task_pairs = pair_auc(subset, "instance_id")
        contrasts.append({
            "checkpoint": checkpoint, "fold": fold, "overall_auc": overall, "within_bin_auc": within_bin,
            "within_task_auc": within_task, "same_bin_pair_share": bin_pairs / pairs,
            "same_task_pairs": task_pairs, "cross_bin_auc": (overall * pairs - within_bin * bin_pairs) / (pairs - bin_pairs),
            "overall_minus_within_bin": overall - within_bin,
            "overall_minus_within_task": overall - within_task,
        })
    return pd.DataFrame(groups), pd.DataFrame(contrasts)


def main() -> None:
    d, r = Path("data/processed"), Path("results")
    runs = pd.read_parquet(d / "runs.parquet", columns=["run_id", "n_steps", "instance_id", "framework"])
    difficulty = pd.read_parquet(d / "difficulty.parquet", columns=["instance_id", "difficulty"])
    scores = pd.read_parquet(d / "prefix_predictions_revised.parquet")
    data = scores.loc[scores["partition"] == "test"].merge(runs, on="run_id", validate="many_to_one").merge(
        difficulty, on="instance_id", validate="many_to_one"
    )
    fixed = data.loc[data["n_steps"] >= 40]
    expected = set(runs.loc[runs["n_steps"] >= 40, "run_id"])
    for (model, checkpoint), group in fixed.groupby(["model", "checkpoint"]):
        if len(group) != len(expected) or set(group["run_id"]) != expected:
            raise ValueError(f"fixed cohort changed for {model} at {checkpoint}")
    columns = ["fold", "model", "checkpoint"]
    for cohort, population in [("fixed_cohort", fixed), ("deployment", data)]:
        rows = []
        for (fold, model, t), group in population.groupby(columns):
            rows.append({"fold": fold, "model": model, "checkpoint": t, "runs": len(group),
                         "failures": int(group["failed"].sum()), **metrics(group["failed"], group["score"])})
        fold_metrics = pd.DataFrame(rows)
        fold_metrics.to_csv(r / f"{cohort}_fold_metrics.csv", index=False)
        write_summary(fold_metrics, ["model", "checkpoint"], r / f"{cohort}_metrics.csv")
        prevalence = fold_metrics.loc[fold_metrics["model"] == "lightgbm"].groupby("checkpoint")[
            ["runs", "failures"]
        ].sum()
        prevalence["failure_rate"] = prevalence["failures"] / prevalence["runs"]
        prevalence.to_csv(r / f"{cohort}_prevalence.csv")
    subgroup = data.loc[data["model"].eq("lightgbm") & data["checkpoint"].isin([5, 10, 20])]
    groups, contrasts = breakdown(subgroup)
    groups.to_csv(r / "lightgbm_subgroup_folds.csv", index=False)
    groups.groupby(["by", "group", "checkpoint"]).agg(
        runs=("runs", "sum"), failures=("failures", "sum"), auroc_mean=("auroc", "mean"),
        auroc_std=("auroc", "std"), evaluable_folds=("auroc", "count"),
    ).reset_index().to_csv(r / "lightgbm_subgroups.csv", index=False)
    contrasts.to_csv(r / "difficulty_signal_folds.csv", index=False)
    contrasts.drop(columns="fold").groupby("checkpoint").mean().to_csv(r / "difficulty_signal.csv")
    logging.warning("Fixed cohort includes %d runs at every checkpoint", len(expected))


if __name__ == "__main__":
    main()
