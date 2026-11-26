"""Summarize failed-run communities and benchmark similarity retrieval."""

import argparse
import json
import logging
import math
import os
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from atm.clustering.minhash_lsh import Sketcher, approximate_medoid, brute_force_edges, lsh_edges
from atm.clustering.shingles import exact_jaccard, shingles
from atm.normalize import collapse


def communities(n: int, edges: set[tuple[int, int]], min_size: int, max_component: int,
                seed: int) -> list[list[int]]:
    """Use connected components, splitting oversized components with Louvain."""
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(sorted(edges))
    groups = []
    for component in nx.connected_components(graph):
        if len(component) > max_component:
            ordered = graph.subgraph(sorted(component)).copy()
            groups.extend(nx.community.louvain_communities(ordered, seed=seed))
        else:
            groups.append(component)
    return sorted((sorted(group) for group in groups if len(group) >= min_size),
                  key=lambda group: (-len(group), group[0]))


def fit_archetypes(sequences: list, config: dict, seed: int) -> tuple[list, list, list, set]:
    """Fit shingle medoid communities on the supplied trajectories only."""
    shingle_sets = [shingles(tokens, config["k"]) for tokens in sequences]
    sketcher = Sketcher(config["num_perm"], seed)
    signs = [sketcher(items) for items in shingle_sets]
    edges, _ = lsh_edges(shingle_sets, signs, config["threshold"], config["bands"], config["rows"])
    groups = communities(len(sequences), edges, config["min_size"], config["max_component"], seed)
    return shingle_sets, signs, groups, edges


def scaling(sets: list, signs: list, cfg: dict, seed: int) -> pd.DataFrame:
    """Compare the LSH pipeline with exact all-pairs on nested equal-input samples."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sets))
    rows = []
    requested = cfg["benchmark_sizes"]
    sizes = sorted({n for n in requested if n <= len(sets)} | {len(sets)})
    for n in sizes:
        subset = order[:n]
        items = [sets[i] for i in subset]
        signature = [signs[i] for i in subset]
        exact_times, lsh_times, cold_times = [], [], []
        for repeat in range(cfg["benchmark_repeats"]):
            if repeat % 2:
                start = perf_counter()
                hashed, candidates = lsh_edges(items, signature, cfg["threshold"], cfg["bands"], cfg["rows"])
                lsh_times.append(perf_counter() - start)
            start = perf_counter()
            exact = brute_force_edges(items, cfg["threshold"])
            exact_times.append(perf_counter() - start)
            if repeat % 2 == 0:
                start = perf_counter()
                hashed, candidates = lsh_edges(items, signature, cfg["threshold"], cfg["bands"], cfg["rows"])
                lsh_times.append(perf_counter() - start)
            start = perf_counter()
            sketch = Sketcher(cfg["num_perm"], seed)
            cold_edges, cold_candidates = lsh_edges(
                items, [sketch(item) for item in items], cfg["threshold"], cfg["bands"], cfg["rows"]
            )
            cold_times.append(perf_counter() - start)
            if cold_edges != hashed or cold_candidates != candidates:
                raise ValueError("cached and fresh MinHash signatures disagreed")
        if not hashed <= exact:
            raise ValueError("exact verification accepted a false edge")
        rows.append({
            "runs": n, "all_pairs": math.comb(n, 2), "lsh_candidates": candidates,
            "exact_pairs": len(exact), "true_lsh_pairs": len(hashed),
            "lsh_seconds": median(lsh_times), "lsh_cold_seconds": median(cold_times), "brute_seconds": median(exact_times),
            "speedup": median(exact_times) / median(lsh_times),
            "cold_speedup": median(exact_times) / median(cold_times),
            "recall": len(hashed) / len(exact) if exact else np.nan,
        })
        logging.info("n=%d, exact=%d, candidates=%d, recall=%s", n, len(exact), candidates, rows[-1]["recall"])
    return pd.DataFrame(rows)


def summarize(runs: pd.DataFrame, sets: list, signs: list, groups: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    population = Counter(item for items in sets for item in items)
    rows, membership = [], []
    for number, members in enumerate(groups, 1):
        sample = runs.iloc[members]
        medoid = approximate_medoid(signs, members)
        present = Counter(item for i in members for item in sets[i])
        ranked = sorted(present, key=lambda item: (
            -(present[item] / len(members) - population[item] / len(sets)),
            -present[item], item,
        ))[:5]
        examples = sorted(members, key=lambda i: (-exact_jaccard(sets[i], sets[medoid]), i))
        examples.remove(medoid)
        examples = [medoid, *examples[:2]]
        name = f"A{number:02d}"
        mix = sample["framework"].value_counts().to_dict()
        rows.append({
            "archetype_id": name, "size": len(members), "framework_mix": json.dumps(mix, sort_keys=True),
            "mean_steps": sample["n_steps"].mean(), "medoid_run_id": runs.iloc[medoid]["run_id"],
            "top_shingles": json.dumps([
                {"tokens": k, "support": present[k] / len(members),
                 "all_failed_support": population[k] / len(sets)} for k in ranked
            ], ensure_ascii=False),
            "example_run_ids": json.dumps(runs.iloc[examples]["run_id"].tolist()),
        })
        membership.extend({"archetype_id": name, "run_id": row.run_id, "framework": row.framework}
                          for row in sample.itertuples())
    return (pd.DataFrame(rows, columns=["archetype_id", "size", "framework_mix", "mean_steps",
                                       "medoid_run_id", "top_shingles", "example_run_ids"]),
            pd.DataFrame(membership, columns=["archetype_id", "run_id", "framework"]))


def plot(frame: pd.DataFrame, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, ScalarFormatter

    fig, (runtime, recall) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    runtime.loglog(frame["runs"], frame["brute_seconds"], "s--", color="#D55E00", label="Brute force")
    runtime.loglog(frame["runs"], frame["lsh_seconds"], "o-", color="#0072B2", label="LSH with cached sketches")
    runtime.loglog(frame["runs"], frame["lsh_cold_seconds"], "^:", color="#009E73", label="LSH with fresh sketches")
    runtime.set(xlabel="Failed trajectories", ylabel="Median runtime (seconds)")
    runtime.set_xticks([500, 1000, int(frame["runs"].max())])
    runtime.xaxis.set_major_formatter(ScalarFormatter())
    runtime.xaxis.set_minor_locator(NullLocator())
    if frame["recall"].isna().any():
        recall.text(.03, .05, "Recall undefined where no exact pairs exist", transform=recall.transAxes, fontsize=8)
    runtime.legend(frameon=False)
    recall.plot(frame["runs"], frame["recall"], "o-", color="#0072B2")
    recall.axhline(.8, color="#777777", linestyle=":")
    recall.set(xlabel="Failed trajectories", ylabel="Recall vs. exact pairs", ylim=(0, 1.03),
               xlim=(frame["runs"].min() - 65, frame["runs"].max() + 65))
    recall.set_xticks([500, 1000, int(frame["runs"].max())])
    for axis in (runtime, recall):
        axis.grid(alpha=.22)
    destination.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(destination / f"lsh_scaling_failed_runs.{extension}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    cfg = config["clustering"]
    root = Path("results")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(root / "logs/cluster.log"),
    ])
    runs = pd.read_parquet("data/processed/runs.parquet")
    collapsed = pd.read_parquet("data/processed/collapsed_sequences.parquet")
    failed = runs.loc[runs["failed"] == 1].merge(collapsed, on="run_id", validate="one_to_one")
    failed = failed.sort_values("run_id").reset_index(drop=True)
    if len(failed) != runs["failed"].sum():
        raise ValueError("failed trajectory missing its collapsed sequence")
    steps = pd.read_parquet("data/processed/steps.parquet", columns=["run_id", "step_idx", "action"])
    steps = steps.loc[steps["run_id"].isin(failed["run_id"])].sort_values(["run_id", "step_idx"])
    base = steps.groupby("run_id")["action"].agg(list).map(collapse)
    representations = {"tokens": failed["tokens"].tolist(), "actions": base.loc[failed["run_id"]].tolist()}
    start = perf_counter()
    outcomes, sensitivity = {}, []
    for view, sequences in representations.items():
        for k in sorted(set(cfg["sensitivity_k"]) | {cfg["k"]}):
            outcome = fit_archetypes(sequences, {**cfg, "k": k}, config["seed"])
            items, _, groups, edges = outcome
            sensitivity.append({"representation": view, "k": k, "archetypes": len(groups),
                                "clustered_runs": sum(map(len, groups)),
                                "nonempty": sum(map(bool, items)), "edges": len(edges)})
            if (view, k) in {("tokens", cfg["k"]), ("actions", min(cfg["sensitivity_k"]))}:
                outcomes[view, k] = outcome

    default_view = ("tokens", cfg["k"])
    reported = default_view if outcomes[default_view][2] else ("actions", min(cfg["sensitivity_k"]))
    sets, signs, groups, edges = outcomes[reported]
    pipeline_seconds = perf_counter() - start
    stats, assignments = summarize(failed, sets, signs, groups)
    stats.to_csv(root / "archetypes.csv", index=False)
    assignments.to_csv(root / "archetype_members.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(root / "archetype_k_sensitivity.csv", index=False)
    label_file = root / "archetypes_for_labeling.md"
    if not label_file.exists() or label_file.read_text().strip() == "# Failure archetypes for manual labeling":
        lines = ["# Failure archetypes for manual labeling", ""]
        for row in stats.itertuples(index=False):
            lines += [f"## {row.archetype_id}", "", f"Runs in cluster · {row.size}", "", "- Human name",
                      "- MAST category", "", "Example runs", ""]
            lines += [f"- `{item}`" for item in json.loads(row.example_run_ids)]
            lines.append("")
        label_file.write_text("\n".join(lines))
    frame = scaling(sets, signs, cfg, config["seed"])
    if reported != default_view:
        default_sets, default_signs, _, _ = outcomes[default_view]
        scaling(default_sets, default_signs, cfg, config["seed"]).to_csv(root / "lsh_scaling_default.csv", index=False)
    frame.to_csv(root / "lsh_scaling.csv", index=False)
    plot(frame, Path("figures"))
    slope = lambda key: float(np.polyfit(np.log(frame["runs"]), np.log(frame[key]), 1)[0])
    summary = {
        "failed_runs": len(failed), "empty_shingle_sets": sum(not s for s in sets), "archetypes": len(groups),
        "reported_view": {"representation": reported[0], "k": reported[1]},
        "default_archetypes": len(outcomes[default_view][2]),
        "clustered_runs": len(assignments), "unclustered_runs": len(failed) - len(assignments),
        "edges": len(edges), "fit_seconds": pipeline_seconds,
        "lsh_loglog_slope": slope("lsh_seconds"), "lsh_cold_loglog_slope": slope("lsh_cold_seconds"),
        "brute_loglog_slope": slope("brute_seconds"),
        "largest_recall": float(frame.iloc[-1]["recall"]), "config": config,
        "experiments_commit": config["experiments_commit"],
    }
    (root / "phase5.json").write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("%d archetypes covering %d/%d failed runs", len(groups), len(assignments), len(failed))
    if not summary["largest_recall"] >= cfg["min_recall"]:
        raise SystemExit("Largest-subset LSH recall is below the configured gate")


if __name__ == "__main__":
    main()
