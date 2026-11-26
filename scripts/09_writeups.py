"""Refresh the README and timed slide outline from metrics.json."""

import json
from pathlib import Path

import pandas as pd


def pct(value: float) -> str:
    return f"{value:.2%}"


def slides(m: dict, n: dict, single: dict, example: str) -> str:
    entries = [
        ("What the first steps tell us", "Aryan", 30, "01_pipeline",
         f"Fixed-cohort LightGBM reaches {m['FixedFive']} AUROC at step 5.",
         [f"At 5, 20, and 40 steps, fixed-cohort AUROC is {m['FixedFive']}, {m['FixedTwenty']}, and {m['FixedForty']}.",
          f"Same-task AUROC is {m['SameTask']} at step 20, so difficulty alone cannot explain the signal.",
          f"Multi-step stopping at α=0.10 saves {m['MultiSave']}, catches {m['MultiCatch']} of failures, and falsely stops {m['MultiFalse']} [{m['MultiLo']}, {m['MultiHi']}].",
          f"Held-framework AUROCs are {m['TransferOpen']}, {m['TransferSWE']}, and {m['TransferMini']} at step 20. OpenHands reaches {m['TransferFalse']} false stops and retained pattern overlap is zero."]),
        ("Why monitor execution logs?", "Aryan", 40, "01_pipeline",
         "Execution logs give us a monitoring surface even when model activations are unavailable.",
         ["Describe the cost of continuing a run that will eventually fail evaluation.",
          "Explain that the outcome label arrives after the whole trajectory.",
          "An early-stop detector trades saved work against falsely aborting a run that would resolve."]),
        ("Three research questions", "Aryan", 40, "01_pipeline",
         "We study whole-run patterns, early prediction and stopping, and cross-framework transfer.",
         ["Explain the difference between descriptive patterns and prefix prediction.",
          "Show how grouped tasks separate training, calibration, and testing.",
          "Preview the stricter framework holdout, which also reserves unseen tasks."]),
        ("From logs to action tokens", "Aryan", 45, "02_dataset",
         f"{n['StudyRuns']} labeled runs from {n['StudySubmissions']} submissions form the corpus.",
         [f"Show a real normalized step: {example}.",
          "SWE-agent has native actions, OpenHands pairs call and response identifiers, and mini-SWE-agent embeds commands and codes in messages.",
          "The human validation sample is pending."]),
        ("Which patterns survive?", "Niyam", 50, "04_patterns",
         f"{n['RobustPatterns']} robust patterns lead the evidence, followed by {n['PrimaryPatterns']} primary patterns.",
         ["PrefixSpan grows frequent prefixes on training tasks and is assessed on distinct tasks.",
          "Primary means Bonferroni Fisher and same-direction CMH; robust also passes the task-clustered score test.",
          "Submissions repeat tasks, so the cluster check addresses dependence. BH is secondary."]),
        ("Difficulty changes the length result", "Niyam", 45, "03_length_confound",
         f"The length–failure correlation falls from {m['LengthOverall']} overall to {m['LengthWithin']} after centering within bins.",
         ["The pooled length relationship combines different kinds of tasks.",
          "Bin-centered correlation shows a much smaller association.",
          "This is a descriptive check; it does not assign a causal share."]),
        ("Archetypes cover a subset of failures", "Niyam", 50, "05_archetypes",
         f"Base-action trigrams at cutoff {n['ArchetypeCutoff']} yield {n['ArchetypeCount']} communities covering {n['ArchetypeRuns']} failed runs.",
         ["Walk through the sizes and framework composition.",
          "Point to the representative action strips and explain why names remain pending.",
          "Archetype similarities are recomputed within each training fold."]),
        ("LSH cuts candidate work", "Niyam", 55, "06_lsh_pairs",
         f"At {n['WindowCount']} windows, LSH checks {n['CandidatePairs']} candidate pairs versus {n['ExactPairs']} all pairs.",
         [f"Recall is {m['WindowRecall']}. Pair-count slopes are {m['CandidateSlope']} for LSH and {m['ExactPairSlope']} for all pairs.",
          f"Compiled exact search takes {m['WindowExactSeconds']} s and LSH takes {m['WindowLSHSeconds']} s at the largest size.",
          "The intended LSH advantage is asymptotic under sparse collisions. The observed near-quadratic slopes do not establish that advantage here."]),
        ("How early is the signal?", "Navneet", 60, "07_fixed_cohort",
         f"Fixed-cohort LightGBM AUROC rises from {m['FixedFive']} at step 5 to {m['FixedTwenty']} at 20 and {m['FixedForty']} at 40.",
         [f"The same {n['FixedRuns']} runs appear at every checkpoint, with a {m['FixedFailure']} failure rate.",
          f"Same-task pair AUROC at step 20 is {m['SameTask']}; show figure 07b for that diagnostic.",
          f"Highest fixed-cohort AUROC at t=10 is {m['BestTenModel']} ({m['BestTenAUC']}) and at t=20 is {m['BestTwentyModel']} ({m['BestTwentyAUC']}).",
          "Failure prevalence changes with eligibility in deployment, so the fixed cohort is the how-early comparison."]),
        ("How much can stopping save?", "Navneet", 60, "08_compute_saved",
         f"At α=0.10 the multi-checkpoint rule saves {m['MultiSave']} and catches {m['MultiCatch']} of failures.",
         [f"False stops were {m['MultiFalse']} among successful runs with Wilson interval [{m['MultiLo']}, {m['MultiHi']}].",
          "The Bonferroni rule used less than its false-stop budget. Its savings are a conservative floor for feasible threshold relaxation on this replay cohort.",
          f"Gross-savings best singles: {single['0.05']['raw']} at α=0.05 and {single['0.10']['raw']} at α=0.10.",
          f"With the observed eligible-success rate within target, the best singles are {single['0.05']['eligible']} and {single['0.10']['eligible']}, respectively."]),
        ("Complex features add little", "Navneet", 40, "09_ablation",
         "Mined pattern indicators and archetype similarities show no consistent incremental predictive gain.",
         ["Distinguish the full-trace inferential pattern tier from training-ranked prediction features.",
          "Only a small robust pattern subset survives difficulty adjustment and the task-clustered check.",
          "Simple temporal actions are informative even when these more elaborate feature groups add little."]),
        ("Transfer weakens discrimination and calibration", "Navneet", 65, "10_transfer",
         f"Held-framework AUROC at step 20 is {m['TransferOpen']} OpenHands, {m['TransferSWE']} SWE-agent, and {m['TransferMini']} mini-SWE-agent.",
         [f"OpenHands falsely stops {m['TransferFalse']} at α=0.10 in the step-20 single-policy replay.",
          "All retained framework-specific significant-pattern lists have zero pairwise overlap.",
          "Different frameworks use different tools and model mixtures. Task-disjoint transfer tests a stricter setting than within-corpus accuracy."]),
        ("Read one real replay", "Navneet", 60, "11_replay",
         "The demo shows the saved model's live prefix score and the first policy crossing.",
         ["Choose one of the suggested held-out runs and move the observed-step slider.",
          "Explain the blue score markers and orange calibration thresholds.",
          "Show the original recorded result after the last observed step."]),
        ("What the replay can establish", "Navneet", 35, "07b_same_task_pairs",
         "The study measures association and retrospective character savings under an observed logging process.",
         ["Character cost proxies tokens and infrastructure costs.",
          "Same-task comparisons show discrimination beyond task difficulty but cannot identify a causal share.",
          "Manual vocabulary agreement and live stopping validation remain open."]),
        ("Takeaways", "Navneet", 45, "07_fixed_cohort",
         f"Logs carry early signal in this corpus, but framework transfer weakens both discrimination and calibration.",
         [f"Recap fixed-cohort AUROC at steps 5, 20, and 40: {m['FixedFive']}, {m['FixedTwenty']}, {m['FixedForty']}.",
          f"Same-task AUROC is {m['SameTask']} at step 20.",
          f"The α=0.10 policy saves {m['MultiSave']}, catches {m['MultiCatch']}, and falsely stops {m['MultiFalse']} [{m['MultiLo']}, {m['MultiHi']}].",
          f"Transfer AUROCs at step 20 are {m['TransferOpen']}, {m['TransferSWE']}, and {m['TransferMini']}. OpenHands false stops reach {m['TransferFalse']} and pattern overlap is zero."]),
    ]
    output = ["# Doomed in the First Steps?", "", "Twelve-minute talk · Aryan Nagwekar, Niyam Momaya, and Navneet Deswal", ""]
    for number,(title,speaker,seconds,figure,takeaway,notes) in enumerate(entries,1):
        output += [f"## {number}. {title}", "", f"{speaker} · {seconds} seconds",
                   "", takeaway, "", f"[Figure](../figures/{figure}.png)", "", "Speaker notes", "",
                   *[f"- {note}" for note in notes], ""]
    return "\n".join(output)


def main() -> None:
    metrics = json.loads(Path("results/metrics.json").read_text())
    m, n = metrics["display"], metrics["numbers"]
    steps = pd.read_parquet("data/processed/steps.parquet", columns=["raw_command","token","action"])
    sample = steps.loc[steps["action"].eq("list") & steps["raw_command"].fillna("").str.fullmatch(r"ls [-a-z]{1,8}")]
    item = sample.iloc[0] if len(sample) else steps.iloc[0]
    example = f'`{str(item.raw_command)[:55]}` → `{item.token}`'
    single = {}
    for alpha, row in metrics["policies"].items():
        def format_single(item):
            return f"step {item['checkpoint']} ({pct(item['compute_saved'])} saved, {pct(item['eligible_false_stop_rate'])} eligible false stops)" if item else "no policy met the observed eligibility target"
        single[alpha] = {"raw": format_single(row["max_savings_single"]), "eligible": format_single(row["eligible_target_single"])}
    target = Path("slides/slides_outline.md")
    target.parent.mkdir(exist_ok=True)
    target.write_text(slides(m,n,single,example), encoding="utf-8")
    summary = [
        "| Result | Value |", "| --- | --- |",
        f"| Fixed-cohort LightGBM AUROC at t=5 / 20 / 40 | {m['FixedFive']} / {m['FixedTwenty']} / {m['FixedForty']} |",
        f"| Same-task pair AUROC at t=20 | {m['SameTask']} |",
        f"| Multi-checkpoint policy, α=0.10 | {m['MultiSave']} character cost saved, {m['MultiCatch']} failures caught, {m['MultiFalse']} false stops [{m['MultiLo']}, {m['MultiHi']}] |",
        f"| Held-framework AUROC at t=20 | OpenHands {m['TransferOpen']}, SWE-agent {m['TransferSWE']}, mini-SWE-agent {m['TransferMini']} |",
        f"| Transfer failure | OpenHands {m['TransferFalse']} false stops at t=20 and α=0.10. Zero retained pattern overlap |",
        f"| Highest fixed-cohort AUROC at t=10 / 20 | {m['BestTenModel']} {m['BestTenAUC']} / {m['BestTwentyModel']} {m['BestTwentyAUC']} |",
        f"| Multi-checkpoint policy, α=0.05 | {m['LowSave']} character cost saved and {m['LowFalse']} false stops |",
        f"| Highest-savings singles at α=0.05 / 0.10 | {single['0.05']['raw']} / {single['0.10']['raw']} |",
        f"| Best singles within the observed eligible budget at α=0.05 / 0.10 | {single['0.05']['eligible']} / {single['0.10']['eligible']} |",
        "",
        "Realized multi-checkpoint false stops stayed below α. The measured savings are a conservative floor for feasible threshold relaxation on this replay cohort. Pattern indicators and archetype similarity show no consistent incremental prediction gain. The signal persists within difficulty bins and within same-task pairs.",
        "",
        f"At {n['WindowCount']} windows, LSH checks {n['CandidatePairs']} of {n['ExactPairs']} possible pairs at {m['WindowRecall']} recall. Candidate versus exact-pair slopes are {m['CandidateSlope']} and {m['ExactPairSlope']}. Runtime slopes are {m['LSHTimeSlope']} and {m['ExactTimeSlope']}. Compiled exact search takes {m['WindowExactSeconds']} s versus {m['WindowLSHSeconds']} s for LSH. The intended asymptotic benefit depends on sparse candidate growth.",
        "",
        "[Numeric source](results/metrics.json) · [Phase 8 tables](report/phase8_report.md)",
    ]
    path = Path("README.md")
    content = path.read_text()
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    if start in content and end in content:
        first = content.index(start) + len(start)
        last = content.index(end)
        content = content[:first] + "\n\n" + "\n".join(summary) + "\n\n" + content[last:]
    else:
        content += "\n## Results\n\n" + start + "\n\n" + "\n".join(summary) + "\n\n" + end + "\n"
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
