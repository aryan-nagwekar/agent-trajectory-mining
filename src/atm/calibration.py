"""Replay task-grouped split-conformal early-stop policies."""

import argparse
from fractions import Fraction
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from statsmodels.stats.proportion import proportion_confint


def conformal_threshold(scores, alpha: Fraction) -> tuple[float, int]:
    """Use the one-based successful-run order statistic and an infinite fallback."""
    values = np.asarray(scores, dtype=float)
    if not 0 < alpha < 1 or not np.isfinite(values).all():
        raise ValueError("alpha and scores must be finite with 0 < alpha < 1")
    rank = ceil((len(values) + 1) * (1 - alpha))
    threshold = float(np.partition(values, rank - 1)[rank - 1]) if rank <= len(values) else np.inf
    return threshold, rank


def first_stops(test: pd.DataFrame, thresholds: dict[int, float]) -> pd.DataFrame:
    """Stop each continuing run at its earliest strict threshold crossing."""
    applicable = test.loc[test["checkpoint"].isin(thresholds) & test["n_steps"].gt(test["checkpoint"])].copy()
    hit = applicable.loc[applicable["score"].gt(applicable["checkpoint"].map(thresholds))]
    return hit.sort_values(["checkpoint", "run_id"]).drop_duplicates("run_id")


def summarize(population: pd.DataFrame, eligible: pd.DataFrame, stops: pd.DataFrame) -> dict:
    """Use one common all-run cost denominator and count each stop once."""
    failed_cost = int(population.loc[population["failed"] == 1, "total_chars"].sum())
    after = stops["total_chars"] - stops["prefix_chars"]
    if (after < 0).any():
        raise ValueError("prefix cost exceeds total cost")
    failures = stops["failed"].eq(1)
    return {
        "runs": len(population), "successes": int(population["failed"].eq(0).sum()),
        "failures": int(population["failed"].eq(1).sum()), "eligible_successes": int(eligible["failed"].eq(0).sum()),
        "eligible_failures": int(eligible["failed"].eq(1).sum()), "stopped": len(stops),
        "false_stops": int((~failures).sum()), "failures_caught": int(failures.sum()),
        "total_chars": int(population["total_chars"].sum()), "failed_chars": failed_cost,
        "saved_chars": int(after.sum()), "saved_failed_chars": int(after.loc[failures].sum()),
    }


def intervals(row: pd.Series) -> pd.Series:
    def ratio(a, b):
        return a / b if b else np.nan

    low, high = proportion_confint(int(row["false_stops"]), int(row["successes"]), method="wilson")         if row["successes"] else (np.nan, np.nan)
    elow, ehigh = proportion_confint(int(row["false_stops"]), int(row["eligible_successes"]), method="wilson")         if row["eligible_successes"] else (np.nan, np.nan)
    return pd.Series({
        "compute_saved": ratio(row["saved_chars"], row["total_chars"]),
        "failure_catch_rate": ratio(row["failures_caught"], row["failures"]),
        "eligible_failure_catch_rate": ratio(row["failures_caught"], row["eligible_failures"]),
        "false_stop_rate": ratio(row["false_stops"], row["successes"]), "false_stop_ci_low": low, "false_stop_ci_high": high,
        "eligible_false_stop_rate": ratio(row["false_stops"], row["eligible_successes"]),
        "eligible_false_stop_ci_low": elow, "eligible_false_stop_ci_high": ehigh,
        "wasted_compute_recovered": ratio(row["saved_chars"], row["failed_chars"]),
        "failed_compute_recovered": ratio(row["saved_failed_chars"], row["failed_chars"]),
    })


def run_policies(scores: pd.DataFrame, runs: pd.DataFrame, splits: pd.DataFrame, config: dict) -> tuple:
    predictions = scores.merge(runs[["run_id", "n_steps", "total_chars", "failed"]],
                               on="run_id", validate="many_to_one", suffixes=("", "_observed"))
    if len(predictions) != len(scores) or not predictions["failed"].eq(predictions["failed_observed"]).all():
        raise ValueError("policy scores do not match the original outcomes")
    allowed = splits.loc[splits["partition"].isin(["test", "calibration"]),
                         ["run_id", "fold", "held_framework", "partition"]].fillna({"held_framework": ""})
    if len(predictions.merge(allowed, on=list(allowed), validate="many_to_one")) != len(predictions):
        raise ValueError("policy scores violate the grouped split manifest")
    checkpoints = config["prediction"]["checkpoints"]
    steps = pd.read_parquet("data/processed/steps.parquet", columns=["run_id", "step_idx", "step_chars"])
    steps = steps.sort_values(["run_id", "step_idx"])
    steps["prefix_chars"] = steps.groupby("run_id")["step_chars"].cumsum()
    step_costs = steps.loc[steps["step_idx"].isin([t - 1 for t in checkpoints]), ["run_id", "step_idx", "prefix_chars"]]
    step_costs["checkpoint"] = step_costs.pop("step_idx") + 1
    expected = len(predictions)
    predictions = predictions.merge(step_costs, on=["run_id", "checkpoint"], validate="many_to_one")
    if len(predictions) != expected:
        raise ValueError("missing observed checkpoint costs")

    rows, gates, stop_rows = [], [], []
    for (held, fold, model), frame in predictions.groupby(["held_framework", "fold", "model"]):
        cohort_ids = splits.loc[splits["held_framework"].fillna("").eq(held)
                                & splits["fold"].eq(fold) & splits["partition"].eq("test"), "run_id"]
        population = runs.loc[runs["run_id"].isin(cohort_ids)]
        calibration = frame.loc[frame["partition"].eq("calibration") & frame["failed"].eq(0)
                                & frame["n_steps"].gt(frame["checkpoint"])]
        test = frame.loc[frame["partition"].eq("test")]
        for requested in config["policy"]["alphas"]:
            alpha = Fraction(str(requested))
            for times in [[t] for t in checkpoints] + [checkpoints]:
                policy = "multi" if len(times) > 1 else "single"
                base = {"held_framework": held, "fold": fold, "model": model, "alpha": requested,
                        "policy": policy, "checkpoint": 0 if policy == "multi" else times[0]}
                per_checkpoint = alpha / len(times)
                cuts = {}
                for t in times:
                    values = calibration.loc[calibration["checkpoint"].eq(t), "score"]
                    cuts[t], rank = conformal_threshold(values, per_checkpoint)
                    gates.append({**base, "threshold_checkpoint": t, "calibration_successes": len(values),
                                  "checkpoint_alpha": float(per_checkpoint), "order_rank": rank, "threshold": cuts[t]})
                stopped = first_stops(test, cuts)
                at_risk = population.loc[population["n_steps"] > min(times)]
                rows.append({**base, **summarize(population, at_risk, stopped)})
                if model == config["policy"]["primary_model"]:
                    stop_rows.extend({**base, "run_id": row.run_id, "stop_step": row.checkpoint,
                                      "failed": row.failed, "saved_chars": row.total_chars - row.prefix_chars}
                                     for row in stopped.itertuples())
    frame = pd.DataFrame(rows)
    by = ["held_framework", "model", "alpha", "policy", "checkpoint"]
    aggregate = frame.drop(columns="fold").groupby(by).sum(numeric_only=True).reset_index()
    return frame, pd.concat([aggregate, aggregate.apply(intervals, axis=1)], axis=1), pd.DataFrame(gates), pd.DataFrame(stop_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view", choices=["grouped", "transfer"])
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    d, r = Path("data/processed"), Path("results")
    runs = pd.read_parquet(d / "runs.parquet")
    partitions = pd.read_parquet(d / "splits.parquet")
    partitions = partitions.loc[partitions["scheme"] == ("grouped" if args.view == "grouped" else "lofo")]
    src = "prefix_predictions_revised.parquet" if args.view == "grouped" else "transfer_predictions.parquet"
    scores = pd.read_parquet(d / src)
    output = "early_stop" if args.view == "grouped" else "transfer_policy"
    fold, pooled, thresholds, stops = run_policies(scores, runs, partitions, config)
    fold.to_csv(r / f"{output}_folds.csv", index=False)
    pooled.to_csv(r / f"{output}.csv", index=False)
    thresholds.to_csv(r / f"{output}_thresholds.csv", index=False)
    stops.to_csv(r / f"{output}_decisions.csv", index=False)
    print(f"{output}: {len(pooled)} pooled policies, {len(thresholds)} calibration thresholds")


if __name__ == "__main__":
    main()
