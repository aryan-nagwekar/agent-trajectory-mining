"""Sweep exact Jaccard graphs over base-action shingle settings."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from atm.clustering.archetypes import communities, summarize
from atm.clustering.minhash_lsh import Sketcher
from atm.clustering.shingles import exact_jaccard, shingles
from atm.normalize import collapse


def base_sequences() -> pd.Series:
    steps = pd.read_parquet("data/processed/steps.parquet", columns=["run_id", "step_idx", "action"])
    return steps.sort_values(["run_id", "step_idx"]).groupby("run_id")["action"].agg(tuple)


def sweep(runs: pd.DataFrame, base: pd.Series, cfg: dict, seed: int, *, log: bool = True) -> tuple[pd.DataFrame, dict]:
    results, selected = [], {}
    thresholds = sorted(cfg["thresholds"])
    for k in cfg["k_values"]:
        sets = [shingles(collapse(list(base.loc[run_id])), k) for run_id in runs["run_id"]]
        edges = {threshold: set() for threshold in thresholds}
        for i, left in enumerate(sets):
            for j in range(i + 1, len(sets)):
                value = exact_jaccard(left, sets[j])
                for threshold in thresholds:
                    if value < threshold:
                        break
                    edges[threshold].add((i, j))
        for threshold in thresholds:
            group = communities(len(sets), edges[threshold], cfg["min_size"], cfg["max_component"], seed)
            inside = sum(map(len, group))
            largest = max(map(len, group), default=0)
            groups_mix = [runs.iloc[m]["framework"].value_counts().to_dict() for m in group]
            results.append({"k": k, "threshold": threshold, "communities": len(group),
                            "covered_runs": inside, "coverage": inside / len(sets),
                            "largest_community": largest, "largest_share_of_covered": largest / inside if inside else 0,
                            "exact_edges": len(edges[threshold]), "sizes": json.dumps([len(m) for m in group]),
                            "framework_mixes": json.dumps(groups_mix, sort_keys=True)})
            selected[k, threshold] = sets, group
            if log:
                logging.info("k=%d, threshold=%.2f: %d communities, %d covered", k, threshold, len(group), inside)
    return pd.DataFrame(results), selected


def choose(frame: pd.DataFrame, minimum_coverage: float, maximum_community_share: float) -> pd.Series:
    """Prefer high-similarity settings with coverage and multiple coherent groups."""
    options = frame.loc[(frame["coverage"] >= minimum_coverage) & (frame["communities"] >= 2)
                        & (frame["largest_share_of_covered"] <= maximum_community_share)]
    if options.empty:
        raise ValueError("none of the settings meets the descriptive selection criteria")
    return options.sort_values(["threshold", "coverage", "k"], ascending=[False, False, False]).iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    cfg = {**config["clustering"], **config["archetype_sweep"]}
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    runs = pd.read_parquet("data/processed/runs.parquet").query("failed == 1").sort_values("run_id").reset_index(drop=True)
    frame, alternatives = sweep(runs, base_sequences(), cfg, config["seed"])
    frame.to_csv("results/archetype_sweep.csv", index=False)
    selection = choose(frame, cfg["selection_min_coverage"], cfg["selection_max_largest_share"])
    k, threshold = int(selection["k"]), float(selection["threshold"])
    sets, group = alternatives[k, threshold]
    sketcher = Sketcher(config["clustering"]["num_perm"], config["seed"])
    signs = [sketcher(items) for items in sets]
    archetypes, membership = summarize(runs, sets, signs, group)
    archetypes.to_csv("results/archetypes.csv", index=False)
    membership.to_csv("results/archetype_members.csv", index=False)
    lines = ["# Failure archetypes for manual labeling", "",
             f"Base-action k={k}, exact Jaccard ≥ {threshold:.2f}.", ""]
    for row in archetypes.itertuples(index=False):
        lines += [f"## {row.archetype_id}", "", f"Runs in community · {row.size}", "",
                  "- Human name", "- MAST category", "", "Example runs", ""]
        lines += [f"- `{run_id}`" for run_id in json.loads(row.example_run_ids)]
        lines.append("")
    label = Path("results/archetypes_for_labeling.md")
    if label.exists():
        backup = label.with_name("archetypes_for_labeling_phase5.md")
        if not backup.exists():
            backup.write_bytes(label.read_bytes())
    label.write_text("\n".join(lines), encoding="utf-8")
    Path("results/archetype_selection.json").write_text(
        json.dumps({"k": k, "threshold": threshold, "failed_runs": len(runs),
                    "communities": len(group), "covered_runs": len(membership),
                    "criteria": {"min_coverage": cfg["selection_min_coverage"],
                                 "max_largest_share": cfg["selection_max_largest_share"]}}, indent=2) + "\n")
    logging.info("Selected k=%d, threshold=%.2f", k, threshold)


if __name__ == "__main__":
    main()
