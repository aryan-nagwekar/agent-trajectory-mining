"""Generate the Phase 5–6 report from saved benchmark and model outputs."""

import json
from pathlib import Path

import pandas as pd

LABELS = {"steps_lr": "Steps-only LR", "bag_lr": "Bag-of-actions LR", "lightgbm": "LightGBM", "gru": "GRU"}


def table(headers, rows) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(str(value) for value in row) + " |" for row in rows],
    ])


def ci(mean: float, std: float) -> str:
    return f"{mean:.3f} ± {std:.3f}"


def main() -> None:
    results = Path("results")
    p5 = json.loads((results / "phase5.json").read_text())
    p6 = json.loads((results / "phase6.json").read_text())
    p4 = json.loads((results / "phase4.json").read_text())["total"]

    def read(name: str) -> pd.DataFrame:
        return pd.read_csv(results / (name + ".csv"))

    archetypes, scaling = read("archetypes"), read("lsh_scaling")
    models = read("prediction_metrics")
    ablation = read("ablation")
    exclusions = read("prefix_exclusions")
    ref_counts = read("prediction_fold_references")
    model_rows = {(row.checkpoint, row.model): row for row in models.itertuples()}
    checkpoint = p6["shuffled_checkpoint"]
    strongest = models.loc[models["checkpoint"] == checkpoint].sort_values("auroc_mean", ascending=False).iloc[0]
    lsh = scaling.iloc[-1]
    config = p5["config"]["clustering"]
    view = p5["reported_view"]
    markdown = [
        "# Doomed in the First Steps?\n\nPhases 5–6 checkpoint",
        "## Executive summary",
        f"The reported clustering view identified **{p5['archetypes']} archetype** covering "
        f"**{p5['clustered_runs']} of {p5['failed_runs']:,} failed runs**. On all failed runs, LSH recovered "
        f"**{int(lsh.true_lsh_pairs)}/{int(lsh.exact_pairs)}** exact pairs and used {int(lsh.lsh_candidates):,} "
        f"candidate checks out of {int(lsh.all_pairs):,} possible pairs.",
        f"At **t={checkpoint}**, {LABELS[strongest.model]} had the highest mean AUROC "
        f"({ci(strongest.auroc_mean, strongest.auroc_std)}). The shuffled-label LightGBM control at the same "
        f"checkpoint scored **{ci(p6['shuffled_auroc_mean'], p6['shuffled_auroc_std'])} AUROC** over "
        f"{p6['shuffled_repeats']} permutations per fold.",
        f"Phase 4 establishes **{p4['surviving_cluster_bonf']['unique_patterns']} robust patterns**, followed by "
        f"**{p4['both_bonf_directional']['unique_patterns']} primary patterns**. The same tasks repeat across "
        "submissions, so runs within a task are dependent. The task-clustered check addresses that dependence. "
        "BH numbers remain secondary context. [Phase 4 report](phase4_report.md)",
        "## Failure archetypes",
        f"The default outcome-augmented {config['k']}-gram view found {p5['default_archetypes']} eligible communities "
        f"at exact Jaccard ≥ {config['threshold']:.2f} and minimum membership {config['min_size']}. "
        f"The reported sensitivity analysis uses collapsed base actions with k={view['k']} at the same threshold "
        "and size cutoff. Connected components define communities, with seeded Louvain splitting only for oversized components.",
        table(["Community", "Failed runs", "Framework mix", "Mean raw steps"], (
            [r.archetype_id, r.size, ", ".join(f"{name} {count}" for name, count in json.loads(r.framework_mix).items()),
             f"{r.mean_steps:.2f}"] for r in archetypes.itertuples()
        )),
        "[Archetype summaries with over-represented shingles and examples](../results/archetypes.csv) · "
        "[Membership](../results/archetype_members.csv) · [Manual archetype labels](../results/archetypes_for_labeling.md) · "
        "[Representation and k sensitivity](../results/archetype_k_sensitivity.csv)",
        "## MinHash-LSH scaling",
        f"The seeded index uses {config['num_perm']} permutations, {config['bands']} bands and "
        f"{config['rows']} rows per band. It proposes candidates, and exact Jaccard verifies each edge. "
        f"Times are medians of {config['benchmark_repeats']} runs on nested seeded samples. Both methods start with "
        "prepared shingle sets. Fresh LSH includes signature computation, indexing, retrieval, and exact verification. "
        "Cached LSH starts from available signatures. The largest sample uses every failed run in the cohort.",
        table(["Failed runs", "Exact pairs", "LSH candidates", "Cached LSH s", "Fresh LSH s", "All-pairs s", "Recall"], (
            [f"{r.runs:,}", r.exact_pairs, f"{r.lsh_candidates:,}", f"{r.lsh_seconds:.3f}",
             f"{r.lsh_cold_seconds:.3f}", f"{r.brute_seconds:.3f}", "—" if pd.isna(r.recall) else f"{r.recall:.1%}"]
            for r in scaling.itertuples()
        )),
        "Recall is undefined for samples with zero exact matches. Empirical log–log runtime slopes over the measured "
        f"sizes were **{p5['lsh_loglog_slope']:.2f}** with cached sketches, "
        f"**{p5['lsh_cold_loglog_slope']:.2f}** with fresh sketches, and **{p5['brute_loglog_slope']:.2f}** for brute force.",
        "![LSH scaling, runtime and recall](../figures/lsh_scaling_failed_runs.png)",
        "[Reported-view benchmark](../results/lsh_scaling.csv) · [Default-view benchmark](../results/lsh_scaling_default.csv) · "
        "[datasketch documentation](https://ekzhu.com/datasketch/documentation.html)",
        "## Prefix evaluation cohorts · Deployment view",
        "Each model observes steps 0 through t−1. A run is eligible when its recorded length reaches t. "
        "The test columns pool counts across the mutually exclusive outer test tasks. A terminal action can appear "
        "inside an eligible prefix and is observable at that checkpoint. Later checkpoints cover fewer runs, so metric changes across checkpoints also reflect cohort selection.",
        table(["Checkpoint", "Included", "Ended earlier", "Included failures", "Failure rate"], (
            [r.checkpoint, f"{r.included:,}", f"{r.excluded:,}", f"{r.included_failures:,}",
             f"{r.included_failures / r.included:.2%}"] for r in exclusions.itertuples()
        )),
        "[Per-fold and per-partition exclusions](../results/prefix_exclusions_by_fold.csv)",
        "## Held-out model performance",
        f"Cells show the mean ± sample standard deviation across {p6['outer_folds']} outer folds. Failure is the positive class. "
        "PR-AUC uses trapezoidal integration. Average precision is available separately in the data. Steps-only LR uses observed "
        "character and step-token counts, token-availability and visible submission flags. Recorded per-step token counts were absent. "
        "Bag LR uses counts and proportions, LightGBM uses all feature groups, and GRU uses raw prefix tokens. "
        "Hyperparameter and early-stopping decisions use grouped inner training tasks. Calibration scores were reserved for "
        "the later stopping-policy phase. The optional LLM judge was skipped because no API key was supplied.",
        *[
            "### " + label + "\n\n" + table(
                ["t", *LABELS.values()],
                ([t, *(ci(getattr(model_rows[t, key], metric + "_mean"), getattr(model_rows[t, key], metric + "_std"))
                       for key in LABELS)] for t in p6["checkpoints"]),
            )
            for label, metric in (("AUROC", "auroc"), ("PR-AUC", "pr_auc"))
        ],
        "[Model metrics](../results/prediction_metrics.csv) · [Fold-level metrics](../results/prediction_fold_metrics.csv) · "
        "[Selected hyperparameters](../results/prediction_hyperparameters.csv)",
        "## Shuffled-label control",
        f"At t={checkpoint}, LightGBM refit pattern and archetype references on each shuffled training label vector. "
        f"Over {p6['shuffled_repeats']} independent permutations in each outer fold, mean AUROC was "
        f"**{ci(p6['shuffled_auroc_mean'], p6['shuffled_auroc_std'])}** and mean PR-AUC was "
        f"**{ci(p6['shuffled_pr_auc_mean'], p6['shuffled_pr_auc_std'])}** on the original held-out outcomes.",
        "[Control by fold and permutation](../results/shuffled_label_control.csv)",
        "## LightGBM ablation",
        f"Each row drops one group and keeps the full model's inner-selected tree settings. Differences are paired "
        f"within outer folds and take ablated minus full performance. Pattern indicators use up to "
        f"{p6['config']['prediction']['top_patterns']} sequences ranked by absolute training support difference, "
        "independent of significance. A fold without a qualifying training archetype supplies a constant similarity feature. Dropping a constant group reuses the full-model score.",
        table(["Fold", "Patterns", "Training archetypes", "Inner-training archetypes"], (
            [r.fold, r.outer_patterns, r.outer_archetypes, r.inner_archetypes] for r in ref_counts.itertuples()
        )),
        table(["Checkpoint", "Dropped", "AUROC mean ± SD", "PR-AUC mean ± SD", "Δ AUROC", "Δ PR-AUC"], (
            [r.checkpoint, r.dropped, ci(r.auroc_mean, r.auroc_std), ci(r.pr_auc_mean, r.pr_auc_std),
             f"{r.delta_auroc_mean:+.4f}", f"{r.delta_pr_auc_mean:+.4f}"] for r in ablation.itertuples()
        )),
        f"{int(ref_counts['outer_archetypes'].eq(0).sum())} of {len(ref_counts)} outer training folds "
        "had zero qualifying archetypes. The archetype ablation therefore had zero measured change in those folds.",
        "[Ablations](../results/ablation.csv) · [Fold-level ablations](../results/ablation_folds.csv) · "
        "[Training-fold references](../results/prediction_fold_references.csv)",
    ]
    path = Path("report/phase6_report.md")
    path.write_text("\n\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
