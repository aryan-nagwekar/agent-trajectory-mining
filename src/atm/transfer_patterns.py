"""Compare framework-specific, held-out significant sequential patterns."""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from atm.mining.discriminative import adjust_holdout, candidate_patterns, effect
from atm.mining.prefixspan import SequenceIndex


def summarize_overlap(lists: dict[str, set[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (left, first), (right, second) in itertools.combinations(sorted(lists.items()), 2):
        union = first | second
        overlap = first & second
        rows.append({"framework_a": left, "framework_b": right, "a_patterns": len(first),
                     "b_patterns": len(second), "overlap": len(overlap), "union": len(union),
                     "jaccard": len(overlap) / len(union) if union else np.nan,
                     "shared_patterns": " | ".join(sorted(overlap))})
    all_patterns = set().union(*lists.values())
    universality = [{"pattern": p, "n_frameworks": sum(p in group for group in lists.values()),
                     "frameworks": ", ".join(name for name, values in sorted(lists.items()) if p in values)}
                    for p in sorted(all_patterns)]
    return pd.DataFrame(rows), pd.DataFrame(universality, columns=["pattern", "n_frameworks", "frameworks"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())["mining"]
    source = Path("data/processed")
    runs = pd.read_parquet(source / "runs.parquet").merge(
        pd.read_parquet(source / "collapsed_sequences.parquet"), on="run_id", validate="one_to_one"
    ).set_index("run_id")
    manifest = pd.read_parquet(source / "splits.parquet").query("scheme == 'grouped'")
    records = []
    for (framework, fold), rows in manifest.groupby(["framework", "fold"]):
        train = runs.loc[rows.loc[rows["partition"] == "train", "run_id"]]
        test = runs.loc[rows.loc[rows["partition"] == "test", "run_id"]]
        if set(train["instance_id"]) & set(test["instance_id"]):
            raise ValueError("task overlap in framework-specific pattern assessment")
        candidates = candidate_patterns(train, cfg)
        index = SequenceIndex(test["tokens"].tolist(), cfg["max_gap"])
        labels = test["failed"].to_numpy()
        totals = np.array([labels.sum(), (1 - labels).sum()])
        for pattern, projected in index.supports(candidates):
            cases = int(labels[list(projected)].sum())
            present = np.array([cases, len(projected) - cases])
            stats = effect(np.column_stack((present, totals - present)))
            records.append({"framework": framework, "fold": fold, "pattern": " → ".join(pattern), **stats})
    frame = pd.DataFrame(records)
    adjust_holdout(frame, ("fisher_p",), ["fold", "framework"])
    selected, counts = [], []
    for name, rows in frame.groupby("framework"):
        passing = rows.loc[rows["fisher_p_bonf"] <= cfg["alpha"]].copy()
        passing["abs_diff"] = passing["support_diff"].abs()
        distinct = passing.sort_values(["abs_diff", "fisher_p_bonf", "fold", "pattern"],
                                       ascending=[False, True, True, True]).drop_duplicates("pattern")
        selected.append(distinct.head(30).drop(columns="abs_diff"))
        counts.append({"framework": name, "assessments": len(rows), "unique_significant": len(distinct),
                       "retained": min(30, len(distinct))})
    top = pd.concat(selected, ignore_index=True)
    frameworks = {name: set(top.loc[top["framework"] == name, "pattern"]) for name in sorted(runs["framework"].unique())}
    overlap, spread = summarize_overlap(frameworks)
    if not top.empty:
        signs = top.assign(direction=np.sign(top["support_diff"])).groupby("pattern")["direction"].nunique()
        spread["direction_agrees"] = spread["pattern"].map(signs).eq(1)
    results = Path("results")
    frame.to_csv(results / "framework_pattern_candidates.csv", index=False)
    top.to_csv(results / "framework_patterns_top30.csv", index=False)
    pd.DataFrame(counts).to_csv(results / "framework_pattern_counts.csv", index=False)
    overlap.to_csv(results / "framework_pattern_overlap.csv", index=False)
    spread.to_csv(results / "framework_pattern_spread.csv", index=False)
    print(pd.DataFrame(counts).to_string(index=False))


if __name__ == "__main__":
    main()
