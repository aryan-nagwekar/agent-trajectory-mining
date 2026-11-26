"""Discover patterns on training tasks and evaluate them on held-out tasks."""

import argparse
import hashlib
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from atm.mining.prefixspan import Pattern, SequenceIndex, mine
from atm.mining.stratified import BINS, TaskClusterScore, cmh, informative, length_correlations
from atm.splits import validate_splits


@lru_cache(maxsize=65_536)
def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p-value with zero information mapped to 1."""
    return float(fisher_exact([[a, b], [c, d]]).pvalue)


def candidate_patterns(train: pd.DataFrame, params: dict) -> dict[Pattern, tuple[int, int]]:
    """Mine the union of frequent patterns independently in each training class."""
    options = {key: params[key] for key in ("min_support", "max_pattern_len", "max_gap")}
    failed = mine(train.loc[train["failed"] == 1, "tokens"].tolist(), **options)
    resolved = mine(train.loc[train["failed"] == 0, "tokens"].tolist(), **options)
    # The union is fixed without access to held-out or calibration trajectories.
    index = SequenceIndex(train["tokens"].tolist(), options["max_gap"])
    classes = train["failed"].to_numpy(dtype=np.int64)
    counts = {}
    for pattern, matching in index.supports(sorted(failed.keys() | resolved.keys())):
        failed_count = int(classes[list(matching)].sum())
        counts[pattern] = failed_count, len(matching) - failed_count
    return counts


def odds_summary(table: np.ndarray) -> tuple[float, float, float]:
    """Return the odds ratio and Wald interval with a zero-cell correction if needed."""
    if not informative(table):
        return np.nan, np.nan, np.nan
    cells = table.astype(float).ravel()
    if (cells == 0).any():
        cells += 0.5
    log_odds = np.log(cells[0]) + np.log(cells[3]) - np.log(cells[1]) - np.log(cells[2])
    half_width = 1.959963984540054 * np.sqrt((1 / cells).sum())
    return tuple(float(value) for value in np.exp([log_odds, log_odds - half_width, log_odds + half_width]))


def effect(table: np.ndarray) -> dict:
    """Compute support, effect and nominal p-value with failure as the first row."""
    a, b, c, d = (int(count) for count in table.ravel())
    sf, ss = a / (a + b) if a + b else np.nan, c / (c + d) if c + d else np.nan
    growth = sf / ss if ss > 0 else np.inf if sf > 0 and ss == 0 else np.nan
    odds, lower, upper = odds_summary(table)
    return {
        "fail_present": a, "fail_total": a + b, "success_present": c, "success_total": c + d,
        "support_fail": sf, "support_success": ss, "support_diff": sf - ss, "growth_rate": growth,
        "odds_ratio": odds, "odds_ci_lower": lower, "odds_ci_upper": upper,
        "fisher_p": fisher(a, b, c, d) if informative(table) else 1.0,
    }


def fold_results(train: pd.DataFrame, test: pd.DataFrame, fold: int, params: dict) -> tuple[list[dict], list[dict]]:
    """Score training-defined candidates against unseen tasks and difficulty bins."""
    if set(train["instance_id"]) & set(test["instance_id"]):
        raise ValueError("candidate discovery and assessment tasks overlap")
    candidates = candidate_patterns(train, params)
    index = SequenceIndex(test["tokens"].tolist(), params["max_gap"])
    failed = test["failed"].to_numpy(dtype=np.int64)
    strata = pd.Categorical(test["difficulty"], categories=BINS).codes
    if (strata < 0).any():
        raise ValueError("test rows are missing difficulty bins")
    cluster_score = TaskClusterScore(failed, strata, test["instance_id"].to_numpy())
    label_bin = strata * 2 + (1 - failed)
    bin_totals = np.bincount(label_bin, minlength=2 * len(BINS)).reshape(-1, 2)
    names, codes = np.unique(test["framework"], return_inverse=True)
    label_framework = codes * 2 + (1 - failed)
    framework_totals = np.bincount(label_framework, minlength=2 * len(names)).reshape(-1, 2)
    train_totals = np.array([train["failed"].sum(), (1 - train["failed"]).sum()])
    pooled, frameworks = [], []

    for pattern, matching in index.supports(candidates):
        indices = np.fromiter(matching, dtype=np.intp, count=len(matching))
        present = np.zeros(len(test), dtype=bool)
        present[indices] = True
        per_bin_present = np.bincount(label_bin[indices], minlength=2 * len(BINS)).reshape(-1, 2)
        tables = np.stack((per_bin_present, bin_totals - per_bin_present), axis=2)
        overall = effect(tables.sum(axis=0))
        train_present = np.array(candidates[pattern])
        discovery = effect(np.column_stack((train_present, train_totals - train_present)))
        odds, p_value, n_strata = cmh(tables)
        base = {"fold": fold, "pattern": " → ".join(pattern), "pattern_length": len(pattern)}
        bin_stats = {}
        for name, table in zip(BINS, tables, strict=True):
            stat = effect(table)
            bin_stats.update({f"{name}_{key}": stat[key] for key in (
                "fail_present", "fail_total", "success_present", "success_total", "support_diff", "fisher_p",
            )})
        pooled.append({
            **base, **overall,
            **{f"train_{key}": value for key, value in discovery.items()},
            "cmh_odds_ratio": odds, "cmh_p": p_value, "cmh_informative_bins": n_strata,
            "task_cluster_p": cluster_score.p_value(present), **bin_stats,
        })
        fw_present = np.bincount(label_framework[indices], minlength=2 * len(names)).reshape(-1, 2)
        for framework, count, total in zip(names, fw_present, framework_totals, strict=True):
            frameworks.append({**base, "framework": framework, **effect(np.column_stack((count, total - count)))})
    return pooled, frameworks


def adjust(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    """Control multiplicity across the supplied family of candidate assessments."""
    for p_col in columns:
        values = frame[p_col].fillna(1).to_numpy()
        frame[p_col + "_bonf"] = multipletests(values, method="bonferroni")[1]
        frame[p_col + "_bh"] = multipletests(values, method="fdr_bh")[1]


def adjust_holdout(frame: pd.DataFrame, columns: tuple[str, ...], groups: list[str]) -> None:
    """Spend an equal share of alpha per independent-of-its-own-test discovery family.

    BH is reported across the full panel as a secondary, descriptive adjustment.
    """
    families = list(frame.groupby(groups, sort=False).groups.values())
    adjust(frame, columns)
    for indexes in families:
        for p_col in columns:
            frame.loc[indexes, p_col + "_bonf"] = np.minimum(
                1.0, frame.loc[indexes, p_col].fillna(1) * len(indexes) * len(families),
            )


def top_patterns(frame: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Rank distinct sequences, prioritizing evidence that survives difficulty control."""
    selected = []
    for direction, size in [(1, 30), (-1, 15)]:
        pool = frame.loc[frame["support_diff"] * direction > 0].copy()
        pool["abs_diff"] = pool["support_diff"].abs()
        pool["passes_fisher"] = pool["fisher_p_bonf"] <= alpha
        pool["robust"] = pool["survives_within_bins"] & (pool["task_cluster_p_bonf"] <= alpha)
        pool = pool.sort_values(
            ["robust", "survives_within_bins", "passes_fisher", "abs_diff", "fisher_p_bonf", "fold", "pattern"],
            ascending=[False, False, False, False, True, True, True],
        )
        pool = pool.drop_duplicates("pattern").head(size)
        pool["direction"] = "failure" if direction == 1 else "success"
        pool["rank"] = np.arange(1, len(pool) + 1)
        selected.append(pool.drop(columns=["abs_diff", "passes_fisher", "robust"]))
    return pd.concat(selected, ignore_index=True)


def export_pattern_tiers(frame: pd.DataFrame, alpha: float, destination: Path) -> None:
    """Export one representative held-out result per sequence for each evidence tier."""
    primary = frame.loc[frame["survives_within_bins"]].sort_values(
        ["task_cluster_p_bonf", "cmh_p_bonf", "fisher_p_bonf", "fold", "pattern"]
    )
    for name, rows in (("primary", primary), ("robust", primary.loc[primary["task_cluster_p_bonf"] <= alpha])):
        rows.drop_duplicates("pattern").to_csv(destination / f"patterns_{name}.csv", index=False)


def counts(frame: pd.DataFrame, alpha: float) -> dict:
    """Count significance at the fold-pattern level and unique-sequence level."""
    values = {"tests": len(frame), "unique_candidates": frame["pattern"].nunique()}
    for name, mask in {
        "fisher_bonf": frame["fisher_p_bonf"] <= alpha,
        "fisher_bh": frame["fisher_p_bh"] <= alpha,
        "cmh_bonf": frame["cmh_p_bonf"] <= alpha,
        "cmh_bh": frame["cmh_p_bh"] <= alpha,
        "both_bonf_directional": frame["survives_within_bins"],
        "both_bh_directional": frame["survives_bh"],
        "cluster_bonf": frame["task_cluster_p_bonf"] <= alpha,
        "surviving_cluster_bonf": frame["survives_within_bins"] & (frame["task_cluster_p_bonf"] <= alpha),
    }.items():
        significant = frame.loc[mask]
        values[name] = {
            "fold_pattern_tests": len(significant),
            "unique_patterns": significant["pattern"].nunique(),
            "patterns_in_multiple_folds": int((significant.groupby("pattern")["fold"].nunique() > 1).sum()),
        }
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    params = config["mining"]
    alpha = params["alpha"]
    root = Path("data/processed")
    results = Path("results")
    (results / "logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(results / "logs/mine.log"),
    ])
    files = [root / f"{name}.parquet" for name in ("runs", "collapsed_sequences", "difficulty", "splits")]
    inputs = files + sorted(Path(__file__).parent.glob("*.py")) + [args.config]
    fingerprint = hashlib.sha256()
    for file in inputs:
        fingerprint.update(file.read_bytes())
    marker = results / ".mine-cache"
    outputs = ["patterns_all.csv", "patterns_top.csv", "patterns_robust.csv", "patterns_primary.csv", "patterns_by_framework.csv", "length_confound.csv", "phase4.json"]
    if not args.force and marker.exists() and marker.read_text() == fingerprint.hexdigest() and all(
        (results / filename).exists() for filename in outputs
    ):
        logging.info("Cached Phase 4 results match the input data and settings")
        return
    raw, sequences, difficulty, splits = [pd.read_parquet(path) for path in files]
    validate_splits(splits, raw)
    runs = raw.merge(sequences, on="run_id", validate="one_to_one").merge(
        difficulty[["instance_id", "difficulty"]], on="instance_id", validate="many_to_one",
    ).set_index("run_id")
    if len(runs) != len(raw):
        raise ValueError("missing normalized sequences or difficulty bins")
    all_rows, framework_rows = [], []
    for fold, partition in splits.loc[splits["scheme"] == "grouped"].groupby("fold", sort=True):
        train = runs.loc[partition.loc[partition["partition"] == "train", "run_id"]]
        test = runs.loc[partition.loc[partition["partition"] == "test", "run_id"]]
        logging.info("Fold %d: discovering in %d training runs, assessing %d test runs", fold, len(train), len(test))
        pooled, separate = fold_results(train, test, fold, params)
        all_rows.extend(pooled)
        framework_rows.extend(separate)
        logging.info("Fold %d: %d patterns tested", fold, len(pooled))
    frame = pd.DataFrame(all_rows)
    if frame.empty:
        raise ValueError("no candidates met the training support threshold")
    adjust_holdout(frame, ("fisher_p", "cmh_p", "task_cluster_p"), ["fold"])
    for _, group in frame.groupby("fold", sort=False):
        discovery = group[["train_fisher_p"]].copy()
        adjust(discovery, ("train_fisher_p",))
        for column in ("train_fisher_p_bonf", "train_fisher_p_bh"):
            frame.loc[discovery.index, column] = discovery[column]
    direction = np.sign(frame["support_diff"]) == np.sign(frame["cmh_odds_ratio"] - 1)
    frame["survives_within_bins"] = (frame["fisher_p_bonf"] <= alpha) & (frame["cmh_p_bonf"] <= alpha) & direction
    frame["survives_bh"] = (frame["fisher_p_bh"] <= alpha) & (frame["cmh_p_bh"] <= alpha) & direction
    by_framework = pd.DataFrame(framework_rows)
    adjust_holdout(by_framework, ("fisher_p",), ["fold", "framework"])
    frame.to_csv(results / "patterns_all.csv", index=False)
    top_patterns(frame, alpha).to_csv(results / "patterns_top.csv", index=False)
    export_pattern_tiers(frame, alpha, results)
    by_framework.to_csv(results / "patterns_by_framework.csv", index=False)
    length_correlations(runs).to_csv(results / "length_confound.csv", index=False)
    report = {"total": counts(frame, alpha), "config": config, "experiments_commit": config["experiments_commit"],
              "folds": {str(fold): counts(group, alpha) for fold, group in frame.groupby("fold")}}
    (results / "phase4.json").write_text(json.dumps(report, indent=2) + "\n")
    marker.write_text(fingerprint.hexdigest())
    logging.info("Held-out results: %d Fisher and %d same-direction CMH survivors (Bonferroni)",
                 report["total"]["fisher_bonf"]["unique_patterns"],
                 report["total"]["both_bonf_directional"]["unique_patterns"])


if __name__ == "__main__":
    main()
