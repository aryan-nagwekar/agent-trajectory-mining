"""Create task-disjoint outer, calibration, and framework holdout splits."""

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

KEYS = ["scheme", "held_framework", "fold"]
PARTITIONS = ("train", "calibration", "test")
COLS = KEYS + ["partition", "run_id", "instance_id", "framework"]


def make_splits(runs: pd.DataFrame, n_splits: int, calibration_fraction: float, seed: int) -> pd.DataFrame:
    """Assign whole tasks first, then apply the held-framework restriction."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration fraction must be between 0 and 1")
    ordered = runs.sort_values(["instance_id", "run_id"]).reset_index(drop=True)
    if ordered["run_id"].duplicated().any():
        raise ValueError("run_id must be unique")
    manifests = []
    outer = GroupKFold(n_splits=n_splits)
    for fold, (fit_idx, test_idx) in enumerate(outer.split(ordered, groups=ordered["instance_id"]), start=1):
        fit, test = ordered.iloc[fit_idx], ordered.iloc[test_idx]
        fit_tasks = sorted(set(fit["instance_id"]))
        n_calibration = math.ceil(len(fit_tasks) * calibration_fraction)
        if n_calibration >= len(fit_tasks):
            raise ValueError("no training tasks remain after calibration")
        rng = np.random.default_rng(np.random.SeedSequence([seed, fold]))
        calibration_tasks = set(rng.choice(fit_tasks, n_calibration, replace=False))
        calibration_mask = fit["instance_id"].isin(calibration_tasks)
        partitions = dict(zip(PARTITIONS, (fit.loc[~calibration_mask], fit.loc[calibration_mask], test)))
        for scheme, held in [("grouped", "")] + [("lofo", f) for f in sorted(ordered["framework"].unique())]:
            for partition, selected in partitions.items():
                if scheme == "lofo":
                    in_framework = selected["framework"] == held
                    selected = selected.loc[in_framework if partition == "test" else ~in_framework]
                if selected.empty:
                    raise ValueError(f"empty {scheme}/{held}/{fold}/{partition} partition")
                metadata = selected[["run_id", "instance_id", "framework"]].assign(
                    scheme=scheme, held_framework=held, fold=fold, partition=partition,
                )
                manifests.append(metadata[COLS])
    manifest = pd.concat(manifests, ignore_index=True)
    validate_splits(manifest, ordered)
    return manifest


def validate_splits(manifest: pd.DataFrame, runs: pd.DataFrame) -> None:
    """Validate task disjointness, exact once-only test coverage, and LOFO restrictions."""
    for (scheme, held, fold), rows in manifest.groupby(KEYS, sort=False):
        if rows["run_id"].duplicated().any():
            raise ValueError(f"repeated run in {scheme}/{held}/{fold}")
        by_task = rows.groupby("instance_id")["partition"].nunique()
        if set(rows["partition"]) != set(PARTITIONS) or by_task.max() != 1:
            raise ValueError(f"task overlap or missing partition in {scheme}/{held}/{fold}")
        if scheme == "lofo":
            if not rows["partition"].eq("test").equals(rows["framework"].eq(held)):
                raise ValueError(f"held framework crosses train/test in {held}/{fold}")
    for (scheme, held), rows in manifest.groupby(KEYS[:2], sort=False):
        test = rows.loc[rows["partition"] == "test"]
        expected = runs if scheme == "grouped" else runs.loc[runs["framework"] == held]
        if len(test) != len(expected) or set(test["run_id"]) != set(expected["run_id"]):
            raise ValueError(f"test coverage is not exactly once per run for {scheme}/{held}")
        if test.groupby("instance_id")["fold"].nunique().max() != 1:
            raise ValueError(f"a task has multiple outer test folds for {scheme}/{held}")


def main() -> None:
    from atm.difficulty import task_difficulty

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    logs = Path("results/logs")
    logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(logs / "splits.log"),
    ])
    params = config["splits"]
    directory = Path("data/processed")
    runs = pd.read_parquet(directory / "runs.parquet")
    manifest = make_splits(runs, params["n_splits"], params["calibration_fraction"], config["seed"])
    difficulty = task_difficulty(runs, params["difficulty_cutoffs"])
    manifest.to_parquet(directory / "splits.parquet", index=False)
    difficulty.to_parquet(directory / "difficulty.parquet", index=False)
    manifest.merge(runs[["run_id", "failed"]], on="run_id", validate="many_to_one").groupby(
        KEYS + ["partition"], sort=True
    ).agg(tasks=("instance_id", "nunique"), runs=("run_id", "size"), failures=("failed", "sum")).reset_index().to_csv(
        "results/fold_sizes.csv", index=False,
    )
    difficulty.groupby("difficulty", sort=False).agg(
        tasks=("instance_id", "size"), runs=("n_runs", "sum"), resolved=("n_resolved", "sum"),
    ).reset_index().to_csv("results/difficulty_summary.csv", index=False)
    logging.info("Wrote %d task-disjoint splits covering %d runs and %d distinct tasks",
                 manifest.groupby(KEYS).ngroups, len(runs), len(difficulty))


if __name__ == "__main__":
    main()
