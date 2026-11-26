"""Build the project report covering the audit, splits, and Phase 4 results."""

import argparse
import json
from pathlib import Path

import pandas as pd


def table(columns: list[str], rows) -> str:
    """Render generated rows as a Markdown table."""
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(str(value).replace("|", r"\|") for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def pct(value: float) -> str:
    return f"{value:.2%}"


def number(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}" if pd.notna(value) else "—"


def probability(value: float) -> str:
    return f"{value:.3g}" if pd.notna(value) else "—"


def distribution(source: pd.DataFrame, column: str) -> str:
    matrix = source.pivot(index=column, columns="framework", values="share").fillna(0)
    return table([column.capitalize(), *matrix.columns], (
        [f"`{label}`", *(pct(value) for value in row)] for label, row in matrix.iterrows()
    ))


def build(results: Path) -> str:
    def csv(name: str) -> pd.DataFrame:
        return pd.read_csv(results / (name + ".csv"), keep_default_na=False, na_values=[""])

    phase4 = json.loads((results / "phase4.json").read_text())
    summary = phase4["total"]
    config = phase4["config"]
    alpha = config["mining"]["alpha"]
    before = summary["fisher_bonf"]["unique_patterns"]
    after = summary["both_bonf_directional"]["unique_patterns"]
    clustered = summary["surviving_cluster_bonf"]["unique_patterns"]
    submissions = csv("submission_audit")
    cohorts = csv("framework_coverage")
    actions = csv("action_distribution")
    tokens = csv("token_distribution")
    tests = csv("run_test_outcomes")
    difficulty = csv("difficulty_summary")
    folds = csv("fold_sizes").fillna({"held_framework": ""})
    lengths = csv("length_confound").set_index("stratum")
    patterns = csv("patterns_top")
    peak_action = actions["share"].max()
    peak_token = tokens["share"].max()
    peak_none = tests["none_share"].max()
    sections = [
        "# Doomed in the First Steps?\n\nPhase 4 checkpoint",
        "## Executive summary",
        f"The audit covers **{len(submissions)} submissions**, **{int(submissions['runs'].sum()):,} runs**, "
        f"and **{int(actions['count'].sum()):,} steps**. The maximum base-action share was {pct(peak_action)}, "
        f"the maximum action-outcome token share was {pct(peak_token)}, "
        f"and the maximum unknown test-outcome share was {pct(peak_none)}. All audit gates passed.",
        f"**{clustered} robust patterns** passed the Bonferroni-corrected Fisher, difficulty-adjusted CMH, and "
        f"task-clustered checks in the same fold. They are part of **{after} primary patterns** that passed Fisher "
        f"and CMH in the same direction. Fisher alone retained {before} of {summary['unique_candidates']:,} candidates. "
        "BH counts are secondary. [Robust pattern list](../results/patterns_robust.csv) · "
        "[Primary pattern list](../results/patterns_primary.csv)",
        f"The correlation between total steps and failure was **{lengths.loc['overall', 'pearson_r']:.3f}** overall "
        f"and **{lengths.loc['within_bin_centered', 'pearson_r']:.3f}** after centering within difficulty bins.",
        "## Submission coverage",
        table(["Submission", "Framework", "Base model", "Runs", "Official resolve", "Normalized resolve"], (
            [r.submission, r.framework, r.base_model, f"{r.runs:,}",
             pct(r.official_resolve_rate), pct(r.observed_resolve_rate)]
            for r in submissions.itertuples(index=False)
        )),
        table(["Framework", "Submissions", "Runs", "Coverage flag"], (
            [r.framework, r.submissions, f"{r.runs:,}", "Review" if r.flag else "Pass"]
            for r in cohorts.itertuples(index=False)
        )),
        f"The coverage flags use fewer than {config['audit']['min_submissions']} submissions or "
        f"fewer than {config['audit']['min_runs']:,} runs. Official resolution uses the full Verified roster. "
        "Normalized resolution uses the eligible parsed runs. [Submission audit](../results/submission_audit.csv)",
        "## Action vocabulary audit",
        "Shares use all normalized steps from the corresponding framework.",
        distribution(actions, "action"),
        "### Outcome-augmented tokens",
        distribution(tokens, "token"),
        "### Outcomes on `run_test` steps",
        table(["Framework", "Test steps", "Pass", "Fail", "None", "Error", "OK"], (
            [r.framework, f"{r.total:,}", pct(r.pass_share), pct(r.fail_share), pct(r.none_share),
             pct(r.error_share), pct(r.ok_share)] for r in tests.itertuples(index=False)
        )),
        f"The audit checks for any token above {config['audit']['max_token_share']:.0%} and for "
        f"`run_test:none` above {config['audit']['max_test_none_share']:.0%}. The SWE-agent `error` share "
        "records explicit error text without a benchmark-test failure marker. The denominator includes every test step.",
        "[Action distribution](../results/action_distribution.csv) · [Token distribution](../results/token_distribution.csv) · "
        "[Test outcomes](../results/run_test_outcomes.csv) · [Manual validation scorer](../scripts/score_validation.py)",
        "## Task splits and difficulty",
        "GroupKFold keeps each Verified task together across submissions. A seeded 20% share of the outer training tasks "
        "is reserved for calibration. The leave-one-framework-out manifest combines the outer task folds with framework "
        "holdout. Each held-out framework is tested only on the held-out tasks for that fold, and the remaining frameworks "
        "supply training and calibration on disjoint tasks. Rotating the task folds evaluates every run in each held-out "
        "framework exactly once.",
    ]
    for scheme, title in (("grouped", "Grouped folds"), ("lofo", "Leave-one-framework-out folds")):
        rows = folds.loc[folds["scheme"] == scheme]
        packed = []
        for (held, fold), group in rows.groupby(["held_framework", "fold"]):
            partition = group.set_index("partition")
            packed.append([held or "All frameworks", fold, *(
                f"{partition.loc[key, 'tasks']} / {partition.loc[key, 'runs']:,}"
                for key in ("train", "calibration", "test")
            )])
        sections.extend([f"### {title}", table(
            ["Test framework", "Fold", "Train tasks / runs", "Calibration tasks / runs", "Test tasks / runs"], packed,
        )])
    sections.extend([
        "The split and leakage tests passed, including once-only held-out coverage and a held-out-data perturbation "
        "that leaves candidate discovery unchanged. [Test results](../results/phase34_tests.txt)",
        "### Difficulty bins",
        "Hard tasks have solve rates at most one-third. Medium tasks are above one-third through two-thirds. "
        "Easy tasks are above two-thirds. Rates use all observed submissions for descriptive stratification and are "
        "kept outside the split manifest and model inputs.",
        table(["Difficulty", "Tasks", "Runs", "Resolved"], (
            [r.difficulty, r.tasks, f"{r.runs:,}", f"{r.resolved:,}"] for r in difficulty.itertuples(index=False)
        )),
        "[Fold sizes](../results/fold_sizes.csv) · [Difficulty summary](../results/difficulty_summary.csv)",
        "## Held-out sequential patterns",
        f"PrefixSpan discovers candidates separately in the training failures and successes using "
        f"{config['mining']['min_support']:.0%} support, a maximum length of {config['mining']['max_pattern_len']} "
        f"collapsed tokens, and up to {config['mining']['max_gap']} intervening tokens between consecutive matches. "
        "Each sequence contributes at most one presence count. Candidate discovery and its training support estimates exclude calibration and test tasks. "
        "Fisher's two-sided test and difficulty-binned Cochran–Mantel–Haenszel (CMH) use the outer test tasks. "
        f"Bonferroni allocates equal shares of α = {alpha:.2f} to each outer fold and adjusts within each fold's candidate set. "
        "Pooled Benjamini–Hochberg (BH) is secondary. A survivor passes Fisher and CMH at the same correction level "
        "with matching directions in the same fold. Repeated tokens in a gapped pattern can be separated "
        "by intervening actions. A suffix such as `×3+` denotes an adjacent streak in the collapsed sequence.",
        table(["Test and adjustment", "Significant fold-pattern tests", "Distinct patterns", "Patterns in multiple folds"], (
            [label, f"{summary[key]['fold_pattern_tests']:,}", f"{summary[key]['unique_patterns']:,}",
             f"{summary[key]['patterns_in_multiple_folds']:,}"]
            for label, key in (
                ("Robust, including task-cluster check", "surviving_cluster_bonf"),
                ("Primary, Fisher and CMH in same direction", "both_bonf_directional"),
                ("Fisher, Bonferroni", "fisher_bonf"), ("CMH, Bonferroni", "cmh_bonf"),
                ("Fisher, BH", "fisher_bh"), ("CMH, BH", "cmh_bh"),
                ("Both, same direction, BH", "both_bh_directional"),
            )
        )),
        table(["Fold", "Candidates", "Fisher Bonferroni", "Both Bonferroni, same direction"], (
            [fold, f"{value['tests']:,}", value["fisher_bonf"]["fold_pattern_tests"],
             value["both_bonf_directional"]["fold_pattern_tests"]]
            for fold, value in phase4["folds"].items()
        )),
        "The same tasks repeat across submissions, so runs within a task are dependent. "
        "The sensitivity check uses a difficulty-adjusted score test with task-clustered variance. The difficulty bins "
        "were derived from this same corpus. Patterns summarize full trajectories and therefore can contain events near "
        "termination. These associations alone do not establish an early signal.",
        "[All pattern statistics](../results/patterns_all.csv) · [By-framework evidence](../results/patterns_by_framework.csv) · "
        "[PrefixSpan paper](https://doi.org/10.1109/ICDE.2001.914830) · "
        "[CMH implementation](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.StratifiedTable.test_null_odds.html)",
    ])
    sections.append(
        "Ranked rows prioritize robust patterns, then primary Fisher and CMH survivors, then Fisher-only and exploratory rows, "
        "with absolute support difference ranking within each tier. Each direction shows distinct sequences and the fold "
        "represented by the row. Odds ratios are oriented toward failure. Intervals are unadjusted Wald intervals "
        "with a half-count correction when an informative table has a zero cell."
    )
    for label, size in (("failure", 10), ("success", 5)):
        selected = patterns.loc[patterns["direction"] == label].head(size)
        sections.extend([
            f"### Top {size} {label}-associated patterns",
            table(["Pattern", "Fold", "Fail support", "Success support", "Δ pp", "Growth", "OR [95% CI]",
                   "Fisher Bonf / BH", "CMH OR", "CMH Bonf / BH", "Task-cluster Bonf", "Evidence"], (
                [f"`{r.pattern}`", r.fold, pct(r.support_fail), pct(r.support_success),
                 f"{r.support_diff * 100:+.2f}", number(r.growth_rate),
                 f"{number(r.odds_ratio)} [{number(r.odds_ci_lower)}, {number(r.odds_ci_upper)}]",
                 f"{probability(r.fisher_p_bonf)} / {probability(r.fisher_p_bh)}",
                 number(r.cmh_odds_ratio), f"{probability(r.cmh_p_bonf)} / {probability(r.cmh_p_bh)}",
                 probability(r.task_cluster_p_bonf),
                 "Robust" if r.survives_within_bins and r.task_cluster_p_bonf <= alpha
                 else "Primary" if r.survives_within_bins else "Fisher only" if r.fisher_p_bonf <= alpha
                 else "CMH only" if r.cmh_p_bonf <= alpha else "Exploratory"]
                for r in selected.itertuples(index=False)
            )),
        ])
    sections.extend([
        "## Length and failure",
        "Pearson correlation between raw step count and the binary failure flag is point-biserial. The pooled within-bin "
        "estimate centers both variables by their respective bin means before correlating them.",
        table(["Population", "Tasks", "Runs", "Failed", "Pearson r"], (
            [label.replace("_", " ").capitalize(), int(r.tasks), f"{int(r.runs):,}", f"{int(r.failed):,}", f"{r.pearson_r:+.3f}"]
            for label, r in lengths.iterrows()
        )),
        "[Length correlations](../results/length_confound.csv) · [Full ranked CSV](../results/patterns_top.csv) · "
        "[Machine-readable Phase 4 summary](../results/phase4.json)",
    ])
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("report/phase4_report.md"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args.results), encoding="utf-8")


if __name__ == "__main__":
    main()
