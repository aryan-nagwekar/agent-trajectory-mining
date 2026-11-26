"""Report cohort coverage, action distributions, and run_test outcomes."""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    args = parser.parse_args()
    thresholds = yaml.safe_load(args.config.read_text())["audit"]
    output = Path("results")
    runs = pd.read_parquet("data/processed/runs.parquet")
    steps = pd.read_parquet("data/processed/steps.parquet", columns=["run_id", "action", "token", "outcome"])
    steps = steps.merge(runs[["run_id", "framework"]], on="run_id", validate="many_to_one")
    official = pd.read_csv(output / "normalization_checks.csv")
    submissions = runs.groupby(["framework", "submission", "base_model"], sort=True).agg(
        runs=("run_id", "size"), failures=("failed", "sum"),
    ).reset_index().merge(
        official[["submission", "official_resolve_rate"]], on="submission", validate="one_to_one",
    )
    submissions["observed_resolve_rate"] = 1 - submissions["failures"] / submissions["runs"]
    submissions.to_csv(output / "submission_audit.csv", index=False)
    coverage = submissions.groupby("framework").agg(submissions=("submission", "size"), runs=("runs", "sum"))
    coverage["flag"] = ((coverage["submissions"] < thresholds["min_submissions"])
                        | (coverage["runs"] < thresholds["min_runs"]))
    coverage.to_csv(output / "framework_coverage.csv")

    for column, filename in (("action", "action_distribution.csv"), ("token", "token_distribution.csv")):
        counts = pd.crosstab(steps["framework"], steps[column]).rename_axis(columns=None)
        long = counts.stack().rename("count").reset_index().rename(columns={"level_1": column})
        long["share"] = long["count"] / long["framework"].map(counts.sum(axis=1))
        long.to_csv(output / filename, index=False)

    outcomes = pd.crosstab(steps.loc[steps["action"] == "run_test", "framework"],
                           steps.loc[steps["action"] == "run_test", "outcome"])
    outcomes = outcomes.reindex(index=coverage.index, columns=["pass", "fail", "none", "error", "ok"], fill_value=0)
    outcomes["total"] = outcomes.sum(axis=1)
    for label in ["pass", "fail", "none", "error", "ok"]:
        outcomes[label + "_share"] = outcomes[label] / outcomes["total"]
    outcomes.to_csv(output / "run_test_outcomes.csv")

    actions = pd.read_csv(output / "action_distribution.csv")
    tokens = pd.read_csv(output / "token_distribution.csv")
    peak_action = actions.loc[actions.groupby("framework")["share"].idxmax()]
    peak_token = tokens.loc[tokens.groupby("framework")["share"].idxmax()]
    print(coverage.to_string())
    print(peak_action[["framework", "action", "share"]].to_string(index=False))
    print(peak_token[["framework", "token", "share"]].to_string(index=False))
    print(outcomes[["total", "pass_share", "fail_share", "none_share", "error_share"]].to_string())
    if (peak_action["share"].gt(thresholds["max_token_share"]).any()
            or peak_token["share"].gt(thresholds["max_token_share"]).any()
            or outcomes["total"].eq(0).any()
            or outcomes["none_share"].gt(thresholds["max_test_none_share"]).any()):
        raise SystemExit("Vocabulary audit requires review before Phase 3.")


if __name__ == "__main__":
    main()
