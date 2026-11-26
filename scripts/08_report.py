"""Render the fixed-cohort, archetype, policy, and transfer checkpoint."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path("results")
MODELS = {"steps_lr": "Steps-only LR", "bag_lr": "Bag LR", "lightgbm": "LightGBM", "gru": "GRU"}
COLORS = ["#CC79A7", "#009E73", "#0072B2", "#D55E00"]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / (name + ".csv"))


def pct(value: float) -> str:
    return f"{value:.2%}" if pd.notna(value) else "—"


def mean_sd(avg: float, sd: float) -> str:
    return f"{avg:.3f} ± {sd:.3f}" if pd.notna(sd) else f"{avg:.3f}"


def table(columns: list[str], rows) -> str:
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *["| " + " | ".join(str(x).replace("|", r"\|") for x in row) + " |" for row in rows],
    ])


def curves(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    for ax, metric, label in zip(axes, ("auroc", "pr_auc"), ("AUROC", "PR-AUC"), strict=True):
        for (name, legend), color in zip(MODELS.items(), COLORS, strict=True):
            df = frame.loc[frame["model"].eq(name)].sort_values("checkpoint")
            t, mean, sd = df["checkpoint"], df[metric + "_mean"], df[metric + "_std"]
            ax.plot(t, mean, "o-", color=color, label=legend)
            ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=.12)
        ax.set(xlabel="Observed steps", ylabel=label, xticks=sorted(frame["checkpoint"].unique()))
        ax.grid(alpha=.22)
    axes[0].legend(frameon=False, fontsize=8)
    for extension in ("png", "pdf"):
        fig.savefig(f"figures/fixed_cohort.{extension}", dpi=300)
    plt.close(fig)


def model_tables(frame: pd.DataFrame, prevalence: pd.DataFrame) -> list[str]:
    out = []
    cells = frame.set_index(["checkpoint", "model"])
    sizes = prevalence.set_index("checkpoint")
    for metric, title in [("auroc", "AUROC"), ("pr_auc", "PR-AUC")]:
        out.extend([
            f"### {title}",
            table(["t", "Runs", "Failure rate", *MODELS.values()], (
                [t, f"{int(sizes.loc[t,'runs']):,}", pct(sizes.loc[t, "failure_rate"]),
                 *(mean_sd(cells.loc[(t, key), metric + "_mean"], cells.loc[(t, key), metric + "_std"]) for key in MODELS)]
                for t in sorted(frame["checkpoint"].unique())
            )),
        ])
    return out


def policy_cells(frame: pd.DataFrame, transfer: bool = False) -> str:
    header = ["Held-out framework"] if transfer else []
    return table(header + ["α", "Policy", "Compute saved", "Failures caught", "All-run false stops [Wilson 95% CI]",
                           "Eligible false stops [Wilson 95% CI]", "Failed-run compute saved"], (
        ([x.held_framework] if transfer else []) + [
            f"{x.alpha:.2f}", f"t={x.checkpoint}" if x.policy == "single" else "All checkpoints",
            pct(x.compute_saved), f"{x.failures_caught:,} ({pct(x.failure_catch_rate)})",
            f"{x.false_stops:,} · {pct(x.false_stop_rate)} [{pct(x.false_stop_ci_low)}, {pct(x.false_stop_ci_high)}]",
            f"{pct(x.eligible_false_stop_rate)} [{pct(x.eligible_false_stop_ci_low)}, {pct(x.eligible_false_stop_ci_high)}]",
            pct(x.failed_compute_recovered),
        ] for x in frame.itertuples()
    ))


def main() -> None:
    fixed, fixed_prev = load("fixed_cohort_metrics"), load("fixed_cohort_prevalence")
    deploy, prev = load("deployment_metrics"), load("deployment_prevalence")
    groups, signal = load("lightgbm_subgroups"), load("difficulty_signal")
    sweep, archetypes, benchmark = load("archetype_sweep"), load("archetypes"), load("lsh_window_scaling")
    ablation, fit_settings = load("archetype_ablation_refit"), load("archetype_refit_settings")
    early = load("early_stop").query("model == 'lightgbm'")
    transfer_auc = load("transfer_metrics")
    transfer_stop = load("transfer_policy").query("model == 'lightgbm'")
    counts, overlap, spread = (load(n) for n in ("framework_pattern_counts", "framework_pattern_overlap", "framework_pattern_spread"))
    selected = json.loads((RESULTS / "archetype_selection.json").read_text())
    window = json.loads((RESULTS / "lsh_window_config.json").read_text())
    curves(fixed)
    exactness = benchmark.iloc[-1]
    multi = early.query("policy == 'multi'").set_index("alpha")
    first = fixed_prev.iloc[0]
    t20 = signal.loc[signal["checkpoint"].eq(20)].iloc[0]
    universal = spread.loc[spread["n_frameworks"] == len(counts)]
    blocks = [
        "# Doomed in the First Steps?\n\nPhase 8 checkpoint",
        "## Executive summary",
        f"The fixed cohort follows **{int(first.runs):,} runs** from 5 to 40 observed steps, with a constant **{pct(first.failure_rate)}** "
        "failure rate. This is the how-early comparison. The deployment view includes every run that reaches each checkpoint.",
        f"At t=20, LightGBM AUROC was **{t20.overall_auc:.3f}** overall and **{t20.within_task_auc:.3f}** when comparing successes "
        "and failures on the same task. Task difficulty alone cannot account for that within-task discrimination.",
        f"The archetype sweep selected **{selected['communities']} communities covering {selected['covered_runs']} failed runs**. "
        f"Window LSH recalled **{pct(exactness.recall)}** on all {window['window_count']:,} windows.",
        f"In grouped evaluation, the multi-checkpoint LightGBM policy saved **{pct(multi.loc[.05,'compute_saved'])}** of character cost "
        f"at α=0.05 and **{pct(multi.loc[.10,'compute_saved'])}** at α=0.10. Realized false-stop rates were "
        f"**{pct(multi.loc[.05,'false_stop_rate'])}** and **{pct(multi.loc[.10,'false_stop_rate'])}**.",
        "## Fixed cohort · How early is the signal?",
        "Evaluation uses the same runs with at least 40 recorded steps at every checkpoint. Existing models were evaluated on this "
        "retrospectively selected cohort. Training used the ordinary checkpoint populations. Cells report mean ± sample SD across "
        "five outer test folds. PR-AUC uses trapezoidal integration. The LightGBM rows use revised training-only archetype features.",
        "![Fixed-cohort AUROC and PR-AUC curves](../figures/fixed_cohort.png)",
        *model_tables(fixed, fixed_prev),
        "## Deployment view · Runs that reach each checkpoint",
        "The failure rate changes with the set of runs still reaching each checkpoint. This view describes deployment eligibility "
        "and supports the policy replay below.",
        *model_tables(deploy, prev),
        "[Fixed-cohort metrics](../results/fixed_cohort_metrics.csv) · [Deployment metrics](../results/deployment_metrics.csv) · "
        "[Fixed-cohort prevalence](../results/fixed_cohort_prevalence.csv) · [Deployment prevalence](../results/deployment_prevalence.csv)",
        "## Where LightGBM discriminates",
        "Within-bin and per-framework AUROCs use the revised model on grouped held-out tasks. The same-task diagnostic forms "
        "positive-negative pairs only within individual tasks and weights by the number of such pairs in each fold.",
        table(["Grouping", "Group", "t", "Runs", "AUROC mean ± SD"], (
            [x.by, x.group, x.checkpoint, f"{x.runs:,}", mean_sd(x.auroc_mean, x.auroc_std)] for x in groups.itertuples()
        )),
        table(["t", "Overall AUC", "Same-bin AUC", "Cross-bin AUC", "Same-task AUC", "Same-task pairs per fold"], (
            [x.checkpoint, f"{x.overall_auc:.3f}", f"{x.within_bin_auc:.3f}", f"{x.cross_bin_auc:.3f}",
             f"{x.within_task_auc:.3f}", f"{x.same_task_pairs:,.1f}"] for x in signal.itertuples()
        )),
        "The same-task AUC exceeds the overall AUC at each measured checkpoint, and same-bin discrimination exceeds "
        "cross-bin discrimination. The data therefore show substantial discrimination beyond task-difficulty detection. "
        "They do not identify a causal share attributable to difficulty. Same-task estimates use tasks with both outcomes, "
        "and differences between frameworks and models can remain within those pairs. Difficulty bins are retrospective.",
        "[Subgroup metrics](../results/lightgbm_subgroups.csv) · [Pair decomposition](../results/difficulty_signal.csv)",
        "## Archetype sweep and refit",
        "The sweep uses exact base-action Jaccard graphs on collapsed sequences. It keeps communities of at least 20 runs. "
        "The descriptive selection prioritizes the highest threshold that covers at least 5% of failed runs, yields at least "
        "two qualifying communities, and puts at most 85% of covered runs in the largest community. "
        "Prediction repeats the rule within each training partition, so its settings can differ.",
        table(["k", "Jaccard cutoff", "Communities ≥20", "Failed runs covered", "Coverage", "Largest community"], (
            [x.k, f"{x.threshold:.2f}", x.communities, x.covered_runs, pct(x.coverage), x.largest_community]
            for x in sweep.itertuples()
        )),
        table(["Community", "Size", "Framework mix"], (
            [x.archetype_id, x.size, ", ".join(f"{name} {n}" for name, n in json.loads(x.framework_mix).items())]
            for x in archetypes.itertuples()
        )),
        "The paired LightGBM refit uses the Phase 6 settings and the same non-archetype features on both sides. "
        "Deltas are without minus with archetypes.",
        table(["t", "With archetypes AUROC", "Without archetypes AUROC", "Δ AUROC", "Δ PR-AUC"], (
            [x.checkpoint, mean_sd(x.full_auroc_mean, x.full_auroc_std),
             mean_sd(x.without_auroc_mean, x.without_auroc_std), f"{x.delta_auroc_mean:+.4f}", f"{x.delta_pr_auc_mean:+.4f}"]
            for x in ablation.itertuples()
        )),
        "[Sweep](../results/archetype_sweep.csv) · [Archetypes](../results/archetypes.csv) · "
        "[Manual labels](../results/archetypes_for_labeling.md) · [Training-fold settings](../results/archetype_refit_settings.csv) · "
        "[Refit ablation](../results/archetype_ablation_refit.csv)",
        "## Sliding-window scaling",
        f"The corpus contains {window['window_count']:,} full 20-step windows at stride 5, so there are no 40,000 real windows. "
        f"Seeded nested samples use base-action k={window['k']} and Jaccard ≥ {window['threshold']:.2f}. "
        "LSH timings include sketching, indexing, and exact verification. Identical shingle sets are indexed once while "
        "candidate and match counts retain window-pair multiplicities. The compiled exact baseline enumerates the window "
        "pairs with valid size and intersection bounds. Benchmarks start with prepared shingle sets.",
        table(["Windows", "Candidates", "Exact pairs", "LSH s", "Exact s", "Recall"], (
            [f"{x.windows:,}", f"{x.lsh_candidates:,}", f"{int(x.exact_matches):,}" if pd.notna(x.exact_matches) else "—",
             f"{x.lsh_seconds:.3f}", f"{x.exact_seconds:.3f}" if pd.notna(x.exact_seconds) else "—", pct(x.recall)]
            for x in benchmark.itertuples()
        )),
        f"Exact search completed at every available size within the {window['brute_force_budget_seconds'] // 60}-minute budget. "
        "At the larger sample sizes, the compiled exact search was faster despite LSH checking a much smaller candidate set.",
        "![Sliding-window runtime and recall](../figures/lsh_scaling.png)",
        "[Window scaling data](../results/lsh_window_scaling.csv)",
        "## Phase 7 · Split-conformal policy replay",
        "For each fold and checkpoint, the threshold is the ceiling of (n+1)(1−α) as a one-based order statistic of "
        "scores from calibration runs that eventually succeeded and are still continuing. A rank beyond the sample gives "
        "an infinite threshold. The policy stops only on a strict exceedance and only when another step remains. "
        "The multi-checkpoint policy allocates α/6 to each checkpoint and takes the first crossing.",
        "Character cost is used uniformly because per-step token costs are unavailable. Savings are remaining observed "
        "characters divided by total characters across all test runs. Failed-run compute saved uses only savings from "
        "failed runs over total failed-run cost. False-stop rates report both all successful runs and the subset eligible "
        "for the policy. Wilson intervals are nominal run-level intervals. Repeated tasks limit independence, and distribution "
        "shift limits conformal exchangeability.",
        policy_cells(early),
        "[All model policies](../results/early_stop.csv) · [Fold results](../results/early_stop_folds.csv) · "
        "[Calibration thresholds](../results/early_stop_thresholds.csv) · [Conformal tutorial](https://arxiv.org/abs/2107.07511)",
        "## Phase 8 · Framework transfer on unseen tasks",
        "Every framework is evaluated on five outer task folds. Training, reference mining, model tuning, and calibration "
        "use the two other frameworks on disjoint tasks. The table reports LightGBM transfer. The full file includes all four models.",
        table(["Held-out framework", "t", "AUROC mean ± SD", "PR-AUC mean ± SD"], (
            [x.held_framework, x.checkpoint, mean_sd(x.auroc_mean, x.auroc_std), mean_sd(x.pr_auc_mean, x.pr_auc_std)]
            for x in transfer_auc.query("model == 'lightgbm'").itertuples()
        )),
        "### Transfer policy at t=10, t=20, and all checkpoints",
        policy_cells(transfer_stop.loc[transfer_stop["checkpoint"].isin([0, 10, 20])], transfer=True),
        "[All transfer metrics](../results/transfer_metrics.csv) · [Transfer policies](../results/transfer_policy.csv) · "
        "[Transfer thresholds](../results/transfer_policy_thresholds.csv)",
        "### Pattern overlap",
        "For each framework, patterns are mined from its training tasks and scored on its held-out tasks. The comparison "
        "takes up to 30 distinct Fisher-Bonferroni significant patterns per framework, ranked by absolute support difference. "
        "The overlap is descriptive and uses each framework's observed list length.",
        table(["Framework", "Fold-pattern assessments", "Distinct significant", "Retained"], (
            [x.framework, f"{x.assessments:,}", x.unique_significant, x.retained] for x in counts.itertuples()
        )),
        table(["Framework pair", "Shared", "Union", "Jaccard"], (
            [f"{x.framework_a} / {x.framework_b}", x.overlap, x.union,
             f"{x.jaccard:.3f}" if pd.notna(x.jaccard) else "—"] for x in overlap.itertuples()
        )),
        f"Universal patterns in these retained lists · **{len(universal)}**.",
        "[Framework pattern lists](../results/framework_patterns_top30.csv) · [Overlap details](../results/framework_pattern_overlap.csv) · "
        "[Pattern spread](../results/framework_pattern_spread.csv)",
        "## Verification",
        "[Test results](../results/phase78_tests.txt) · [Deviations and choices](../DEVIATIONS.md)",
    ]
    Path("report/phase8_report.md").write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
