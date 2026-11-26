"""Assemble the report's numeric source, task bootstraps, and booktabs tables."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

ROOT = Path("results")
CHECKPOINTS = (5, 10, 15, 20, 30, 40)
MODELS = {"steps_lr": "Steps-only LR", "bag_lr": "Bag LR", "lightgbm": "LightGBM", "gru": "GRU"}
FRAMEWORKS = ("OpenHands", "SWE-agent", "mini-SWE-agent")


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / f"{name}.csv")


def std(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def pct(value: float) -> str:
    return f"{value:.2%}"


def selected_policies(policies: pd.DataFrame) -> dict:
    """Expose both gross-savings maxima and the maxima with eligible false stops at target."""
    source = policies.loc[policies["model"].eq("lightgbm")]
    result = {}
    for alpha, group in source.groupby("alpha"):
        singles = group.loc[group["policy"].eq("single")].sort_values(
            ["compute_saved", "checkpoint"], ascending=[False, True]
        )
        eligible = singles.loc[singles["eligible_false_stop_rate"] <= alpha]
        multi = group.loc[group["policy"].eq("multi")].iloc[0]
        result[f"{alpha:.2f}"] = {
            "multi": multi.to_dict(),
            "max_savings_single": singles.iloc[0].to_dict(),
            "eligible_target_single": eligible.iloc[0].to_dict() if len(eligible) else None,
        }
    return result


def bootstrap(config: dict, trials: int) -> dict:
    """Resample whole test tasks within each outer fold and retain all their runs."""
    d = Path("data/processed")
    runs = pd.read_parquet(d / "runs.parquet", columns=["run_id", "instance_id", "n_steps"])
    scores = pd.read_parquet(d / "prefix_predictions_revised.parquet")
    scores = scores.loc[scores["partition"].eq("test") & scores["model"].eq("lightgbm")].merge(
        runs, on="run_id", validate="many_to_one"
    ).query("n_steps >= 40")
    # Use the same independent fold resamples at all three checkpoints for paired interpretation.
    result = {}
    for t in (5, 20, 40):
        rng = np.random.default_rng(config["seed"])
        by_fold = []
        for _, rows in scores.loc[scores["checkpoint"] == t].sort_values("run_id").groupby("fold"):
            _, groups = np.unique(rows["instance_id"], return_inverse=True)
            n_tasks = int(groups.max()) + 1
            labels, probs = rows["failed"].to_numpy(), rows["score"].to_numpy()
            fold_values = []
            for _ in range(trials):
                weights = np.bincount(rng.integers(0, n_tasks, size=n_tasks), minlength=n_tasks)[groups]
                if set(labels[weights > 0]) != {0, 1}:
                    fold_values.append([np.nan, np.nan])
                    continue
                roc = roc_auc_score(labels, probs, sample_weight=weights)
                precision, recall, _ = precision_recall_curve(labels, probs, sample_weight=weights)
                fold_values.append([roc, auc(recall, precision)])
            by_fold.append(fold_values)
        draws = np.asarray(by_fold).mean(axis=0)
        complete = draws[np.isfinite(draws).all(axis=1)]
        if len(complete) < .8 * trials:
            raise ValueError(f"too few complete task-bootstrap replicates for t={t}")
        low, high = np.quantile(complete, [0.025, 0.975], axis=0)
        result[str(t)] = {"auroc": [float(low[0]), float(high[0])], "pr_auc": [float(low[1]), float(high[1])],
                          "replicates_used": len(complete)}
    return result


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return json_clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_table(name: str, table: pd.DataFrame) -> None:
    prefix = ROOT / "tables" / name
    table.to_csv(prefix.with_suffix(".csv"), index=False)
    tex = table.to_latex(index=False, escape=True, column_format="l" + "r" * (len(table.columns) - 1))
    # Explicit escapes retain portable pdfLaTeX and Tectonic support.
    tex = tex.replace("±", r"$\pm$").replace("→", r"$\to$").replace("×", r"$\times$")
    prefix.with_suffix(".tex").write_text(tex, encoding="utf-8")


def make_tables(policies: pd.DataFrame) -> None:
    ROOT.joinpath("tables").mkdir(exist_ok=True)
    submissions = read("submission_audit").assign(
        Resolve=lambda frame: frame["official_resolve_rate"].map(pct)
    ).rename(columns={"framework": "Framework", "base_model": "Base model", "runs": "Usable runs"})
    write_table("dataset", submissions[["Framework", "Base model", "Usable runs", "Resolve"]])
    for key, source in [("fixed", "fixed_cohort"), ("deployment", "deployment")]:
        frame = read(f"{source}_metrics")
        prevalence = read(f"{source}_prevalence").set_index("checkpoint")
        for metric, label in [("auroc", "AUROC"), ("pr_auc", "PR-AUC")]:
            pivot = frame.assign(value=frame.apply(lambda x: std(x[metric + "_mean"], x[metric + "_std"]), axis=1)).pivot(
                index="checkpoint", columns="model", values="value"
            ).reindex(columns=MODELS).rename(columns=MODELS)
            pivot.insert(0, "Failure rate", prevalence["failure_rate"].map(pct))
            write_table(f"{key}_{metric}", pivot.reset_index().rename(columns={"checkpoint": "Steps"}))
    selected = policies.loc[policies["model"].eq("lightgbm")].sort_values(["alpha", "checkpoint"])
    table = [{
        "Alpha": f"{row.alpha:.2f}", "Steps": "All" if row.checkpoint == 0 else row.checkpoint,
        "Cost saved": pct(row.compute_saved), "Failed runs caught": f"{row.failures_caught} ({pct(row.failure_catch_rate)})",
        "False stop [Wilson CI]": f"{pct(row.false_stop_rate)} [{pct(row.false_stop_ci_low)}, {pct(row.false_stop_ci_high)}]",
    } for row in selected.itertuples()]
    write_table("early_stop", pd.DataFrame(table))
    robust, primary = read("patterns_robust"), read("patterns_primary")
    for label, frame in [("robust", robust), ("primary", primary)]:
        patterns = [{
            "Sequence": row.pattern, "Fold": row.fold, "Fail": pct(row.support_fail), "Success": pct(row.support_success),
            "OR": f"{row.odds_ratio:.2f}", "Fisher": f"{row.fisher_p_bonf:.3g}", "CMH": f"{row.cmh_p_bonf:.3g}",
            "Cluster": f"{row.task_cluster_p_bonf:.3g}",
        } for row in frame.itertuples()]
        write_table(f"patterns_{label}", pd.DataFrame(patterns, columns=[
            "Sequence", "Fold", "Fail", "Success", "OR", "Fisher", "CMH", "Cluster",
        ]))
    transfer = read("transfer_metrics").query("model == 'lightgbm'").copy()
    transfer["AUROC"] = transfer.apply(lambda x: std(x.auroc_mean, x.auroc_std), axis=1)
    table = transfer.pivot(index="held_framework", columns="checkpoint", values="AUROC").reset_index()
    write_table("transfer", table.rename(columns={"held_framework": "Held framework"}))
    patterns = read("patterns_top")
    write_table("patterns_ranked", pd.DataFrame([{
        "Class": x.direction, "Rank": x.rank, "Sequence": x.pattern, "Fold": x.fold,
        "Fail": pct(x.support_fail), "Success": pct(x.support_success),
        "OR": f"{x.odds_ratio:.2f}", "Fisher": f"{x.fisher_p_bonf:.3g}",
        "CMH": f"{x.cmh_p_bonf:.3g}", "Cluster": f"{x.task_cluster_p_bonf:.3g}",
    } for x in patterns.itertuples()]))
    subgroup_ablation = read("ablation").query("dropped != 'archetypes'")
    revised_ablation = read("archetype_ablation_refit").assign(dropped="archetypes (refit)")
    write_table("ablation", pd.DataFrame([{
        "Steps": x.checkpoint, "Removed": x.dropped,
        "Delta AUROC": f"{x.delta_auroc_mean:+.4f}", "Delta PR-AUC": f"{x.delta_pr_auc_mean:+.4f}",
    } for x in pd.concat([subgroup_ablation, revised_ablation], ignore_index=True).sort_values(
        ["checkpoint", "dropped"]
    ).itertuples()]))
    sweep = read("archetype_sweep")
    write_table("archetype_sweep", sweep.assign(Coverage=sweep["coverage"].map(pct))[[
        "k", "threshold", "communities", "covered_runs", "Coverage",
    ]].rename(columns={"threshold": "Jaccard", "communities": "Communities", "covered_runs": "Covered runs"}))
    lsh = read("lsh_window_scaling")
    write_table("lsh_pairs", pd.DataFrame([{
        "Windows": f"{x.windows:,}", "Candidate pairs": f"{x.lsh_candidates:,}", "All pairs": f"{x.all_pairs:,}",
        "Recall": pct(x.recall) if pd.notna(x.recall) else "N/A",
        "LSH s": f"{x.lsh_seconds:.2f}", "Exact s": f"{x.exact_seconds:.2f}" if pd.notna(x.exact_seconds) else "N/A",
    } for x in lsh.itertuples()]))
    subgroup = read("lightgbm_subgroups")
    subgroup["AUROC"] = subgroup.apply(lambda x: std(x.auroc_mean, x.auroc_std), axis=1)
    write_table("subgroups", subgroup.pivot(index=["by", "group"], columns="checkpoint", values="AUROC")
                .reset_index().rename(columns={"by": "Type", "group": "Group"}))


def make_macros(payload: dict, config: dict) -> None:
    display = payload["display"]
    mac = dict(display)
    for key, value in payload["numbers"].items():
        if key not in mac:
            mac[key] = str(value)
    for word, t in (("Five", 5), ("Twenty", 20), ("Forty", 40)):
        mac["Fixed" + word + "Boot"] = "--".join(
            f"{x:.3f}" for x in payload["fixed_cohort_bootstrap"][str(t)]["auroc"]
        )
    for name, alpha in (("Low", "0.05"), ("High", "0.10")):
        selection = payload["policies"][alpha]
        for label, key in (("Raw", "max_savings_single"), ("Eligible", "eligible_target_single")):
            single = selection[key]
            prefix = f"Single{name}{label}"
            mac[prefix + "T"] = str(single["checkpoint"]) if single else "N/A"
            mac[prefix + "Save"] = pct(single["compute_saved"]) if single else "N/A"
            mac[prefix + "False"] = pct(single["eligible_false_stop_rate"]) if single else "N/A"
    mac.update({
        "StudyWidth": str(config["window_benchmark"]["width"]),
        "StudyStride": str(config["window_benchmark"]["stride"]),
        "CurvePoints": ", ".join(map(str, CHECKPOINTS)), "FoldCount": str(config["splits"]["n_splits"]),
        "PatternSupport": f'{config["mining"]["min_support"]:.0%}', "PatternGap": str(config["mining"]["max_gap"]),
        "PatternLength": str(config["mining"]["max_pattern_len"]),
        "StepFive": str(CHECKPOINTS[0]), "StepTen": str(CHECKPOINTS[1]), "StepTwenty": str(CHECKPOINTS[3]), "StepForty": str(CHECKPOINTS[-1]),
        "AlphaLow": f'{config["policy"]["alphas"][0]:.2f}', "AlphaHigh": f'{config["policy"]["alphas"][-1]:.2f}', "One": "1",
        "ArchetypeMin": str(config["clustering"]["min_size"]), "PatternK": str(config["prediction"]["top_patterns"]),
        "HashCount": str(config["clustering"]["num_perm"]), "WindowBands": str(config["window_benchmark"]["bands"]),
        "WindowRows": str(config["window_benchmark"]["rows"]),
    })
    def escape(value: str) -> str:
        return value.replace("%", r"\%").replace("±", r"$\pm$").replace("[", "{[}").replace("]", "{]}")
    (ROOT / "macros.tex").write_text("% Generated by atm.final_results.\n" + "".join(
        chr(92) + "newcommand{" + chr(92) + name + "}{" + escape(value) + "}\n" for name, value in mac.items()
    ), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    fixed = read("fixed_cohort_metrics").query("model == 'lightgbm'").set_index("checkpoint")
    prevalence = read("fixed_cohort_prevalence").iloc[0]
    deploy = read("deployment_prevalence").set_index("checkpoint")
    lengths = read("length_confound").set_index("stratum")
    difficulty = read("difficulty_signal").set_index("checkpoint")
    fixed_all = read("fixed_cohort_metrics")
    fixed_best = fixed_all.sort_values(["checkpoint","auroc_mean"], ascending=[True,False]).drop_duplicates("checkpoint").set_index("checkpoint")
    transfer = read("transfer_metrics").query("model == 'lightgbm'").set_index(["held_framework", "checkpoint"])
    transfer_policy = read("transfer_policy").query("held_framework == 'OpenHands' and model == 'lightgbm' and alpha == 0.10 and checkpoint == 20").iloc[0]
    phase4 = json.loads((ROOT / "phase4.json").read_text())["total"]
    clustering = json.loads((ROOT / "archetype_selection.json").read_text())
    phase6 = json.loads((ROOT / "phase6.json").read_text())
    phase2 = json.loads((ROOT / "phase2.json").read_text())
    lsh = read("lsh_window_scaling")
    final_lsh = lsh.iloc[-1]
    slopes = {col: float(np.polyfit(np.log(lsh["windows"]), np.log(lsh[col]), 1)[0])
              for col in ("lsh_candidates", "all_pairs", "lsh_seconds", "exact_seconds")}
    policy = read("early_stop")
    chosen = selected_policies(policy)
    low, high = [chosen[f"{alpha:.2f}"]["multi"] for alpha in (.05, .10)]
    display = {
        "FixedFive": std(fixed.loc[5, "auroc_mean"], fixed.loc[5, "auroc_std"]),
        "FixedTwenty": std(fixed.loc[20, "auroc_mean"], fixed.loc[20, "auroc_std"]),
        "FixedForty": std(fixed.loc[40, "auroc_mean"], fixed.loc[40, "auroc_std"]),
        "FixedTen": std(fixed.loc[10, "auroc_mean"], fixed.loc[10, "auroc_std"]),
        "BestTenModel": MODELS[fixed_best.loc[10, "model"]],
        "BestTwentyModel": MODELS[fixed_best.loc[20, "model"]],
        "BestTenAUC": std(fixed_best.loc[10,"auroc_mean"], fixed_best.loc[10,"auroc_std"]),
        "BestTwentyAUC": std(fixed_best.loc[20,"auroc_mean"], fixed_best.loc[20,"auroc_std"]),
        "FixedFailure": pct(prevalence.failure_rate), "SameTask": f'{difficulty.loc[20,"within_task_auc"]:.3f}',
        "SameBin": f'{difficulty.loc[20,"within_bin_auc"]:.3f}', "AllPairAUC": f'{difficulty.loc[20,"overall_auc"]:.3f}',
        "MultiSave": pct(high["compute_saved"]), "MultiCatch": pct(high["failure_catch_rate"]),
        "MultiFalse": pct(high["false_stop_rate"]), "MultiLo": pct(high["false_stop_ci_low"]),
        "MultiHi": pct(high["false_stop_ci_high"]), "LowSave": pct(low["compute_saved"]),
        "LowFalse": pct(low["false_stop_rate"]), "TransferOpen": f'{transfer.loc[("OpenHands",20),"auroc_mean"]:.3f}',
        "TransferSWE": f'{transfer.loc[("SWE-agent",20),"auroc_mean"]:.3f}',
        "TransferMini": f'{transfer.loc[("mini-SWE-agent",20),"auroc_mean"]:.3f}',
        "TransferFalse": pct(transfer_policy.false_stop_rate),
        "TransferLo": pct(transfer_policy.false_stop_ci_low), "TransferHi": pct(transfer_policy.false_stop_ci_high),
        "WindowRecall": pct(final_lsh.recall), "WindowLSHSeconds": f"{final_lsh.lsh_seconds:.2f}",
        "WindowExactSeconds": f"{final_lsh.exact_seconds:.2f}", "CandidateSlope": f'{slopes["lsh_candidates"]:.2f}',
        "ExactPairSlope": f'{slopes["all_pairs"]:.2f}', "LSHTimeSlope": f'{slopes["lsh_seconds"]:.2f}',
        "ExactTimeSlope": f'{slopes["exact_seconds"]:.2f}', "LengthOverall": f'{lengths.loc["overall","pearson_r"]:.3f}',
        "LengthWithin": f'{lengths.loc["within_bin_centered","pearson_r"]:.3f}',
        "ShuffleAUROC": std(phase6["shuffled_auroc_mean"], phase6["shuffled_auroc_std"]),
        "DeploymentFive": pct(deploy.loc[5,"failure_rate"]), "DeploymentForty": pct(deploy.loc[40,"failure_rate"]),
    }
    numbers = {
        "StudyRuns": f'{phase2["runs"]:,}', "StudySteps": f'{phase2["steps"]:,}', "StudySubmissions": phase2["submissions"],
        "StudyTasks": pd.read_parquet("data/processed/runs.parquet", columns=["instance_id"])["instance_id"].nunique(),
        "FixedRuns": f"{int(prevalence.runs):,}", "FixedFailures": f"{int(prevalence.failures):,}",
        "RobustPatterns": phase4["surviving_cluster_bonf"]["unique_patterns"],
        "PrimaryPatterns": phase4["both_bonf_directional"]["unique_patterns"], "FisherPatterns": phase4["fisher_bonf"]["unique_patterns"],
        "CandidatePatterns": f'{phase4["unique_candidates"]:,}', "Assessments": f'{phase4["tests"]:,}',
        "ArchetypeCount": clustering["communities"], "ArchetypeRuns": clustering["covered_runs"],
        "AllFailed": f'{clustering["failed_runs"]:,}', "ArchetypeCutoff": f'{clustering["threshold"]:.2f}',
        "ArchetypeGram": clustering["k"], "WindowCount": f"{int(final_lsh.windows):,}",
        "CandidatePairs": f"{int(final_lsh.lsh_candidates):,}", "ExactPairs": f"{int(final_lsh.all_pairs):,}",
        "CatchCount": high["failures_caught"], "FalseCount": high["false_stops"],
        "LOFOCount": pd.read_csv(ROOT / "transfer_metrics.csv")["held_framework"].nunique() * config["splits"]["n_splits"],
        "PatternOverlap": int(read("framework_pattern_overlap")["overlap"].sum()),
        "ValidationSize": config["phase2"]["validation_size"],
    }
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    payload = {"display": display, "numbers": numbers, "fixed_cohort_bootstrap": bootstrap(config, config.get("presentation", {}).get("bootstrap_replicates", 1000)),
               "fixed_lightgbm": fixed.to_dict(orient="index"), "deployment_prevalence": deploy.to_dict(orient="index"),
               "policies": chosen, "lsh_loglog_slopes": slopes,
               "config": config, "project_git_commit": git.stdout.strip() if git.returncode == 0 else None,
               "experiments_commit": config["experiments_commit"]}
    (ROOT / "metrics.json").write_text(json.dumps(json_clean(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    make_macros(payload, config)
    make_tables(policy)


if __name__ == "__main__":
    main()
