"""Fit and assess the Phase 6 models on grouped outer test tasks."""

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from atm.clustering.minhash_lsh import Sketcher
from atm.features import GROUPS, FeatureSpace, Prefix, bag_matrix, build_prefix, fit_reference, load_traces, matrix
from atm.normalize import collapse
from atm.models import gru
from atm.models.baselines import choose_c, choose_lightgbm, lightgbm, logistic
from atm.splits import validate_splits


def metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    """Evaluate failure as the positive class; report trapezoidal PR-AUC and AP."""
    precision, recall, _ = precision_recall_curve(labels, scores)
    return {"auroc": roc_auc_score(labels, scores), "pr_auc": auc(recall, precision),
            "average_precision": average_precision_score(labels, scores)}


@dataclass
class Eligible:
    ids: list[str]
    rows: list[Prefix]
    labels: np.ndarray
    tasks: np.ndarray


def eligible(ids: list[str], runs: pd.DataFrame, prefix_cache: dict[str, Prefix]) -> Eligible:
    included = [run_id for run_id in ids if run_id in prefix_cache]
    records = runs.loc[included]
    return Eligible(included, [prefix_cache[key] for key in included],
                    records["failed"].to_numpy(dtype=np.float32), records["instance_id"].to_numpy())


def inner_lr(train: Eligible, cfg: dict, seed: int) -> tuple[float, float]:
    """Tune C on grouped inner partitions with independent bag vocabularies."""
    cost = np.array([row.cost for row in train.rows])
    cost_folds, bag_folds = [], []
    for fit, valid in GroupKFold(n_splits=cfg["inner_folds"]).split(cost, train.labels, train.tasks):
        fit_rows, valid_rows = [train.rows[i] for i in fit], [train.rows[i] for i in valid]
        vocabulary = tuple(sorted({token for row in fit_rows for token in row.tokens}))
        cost_folds.append((cost[fit], train.labels[fit], cost[valid], train.labels[valid]))
        bag_folds.append((bag_matrix(fit_rows, vocabulary), train.labels[fit],
                          bag_matrix(valid_rows, vocabulary), train.labels[valid]))
    return tuple(choose_c(folds, cfg["logistic_c"], seed) for folds in (cost_folds, bag_folds))


def write_summary(frame: pd.DataFrame, groups: list[str], target: Path) -> None:
    aggregate = frame.groupby(groups, sort=True)[["auroc", "pr_auc", "average_precision"]].agg(["mean", "std"])
    aggregate.columns = [f"{metric}_{stat}" for metric, stat in aggregate.columns]
    aggregate.reset_index().to_csv(target, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    cfg, seed = config["prediction"], config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    root, results = Path("data/processed"), Path("results")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(results / "logs/predict.log"),
    ])
    phase5 = json.loads((results / "phase5.json").read_text())
    if phase5["largest_recall"] < config["clustering"]["min_recall"]:
        raise ValueError("Phase 5 recall gate must pass before prediction")
    if phase5["config"]["clustering"] != config["clustering"]:
        raise ValueError("clustering settings differ from the completed Phase 5 benchmark")

    raw = pd.read_parquet(root / "runs.parquet")
    sequences = pd.read_parquet(root / "collapsed_sequences.parquet")
    splits = pd.read_parquet(root / "splits.parquet")
    validate_splits(splits, raw)
    runs = raw.merge(sequences, on="run_id", validate="one_to_one").set_index("run_id")
    traces = load_traces(root / "steps.parquet")
    if set(runs.index) != set(traces) or len(raw) != len(traces):
        raise ValueError("normalized run IDs differ between steps and sequence inputs")
    if not all(len(traces[key].tokens) == n for key, n in runs["n_steps"].items()):
        raise ValueError("mismatch between normalized step counts and trace lengths")
    runs["actions"] = pd.Series({key: collapse(list(trace.actions)) for key, trace in traces.items()})
    sketcher = Sketcher(config["clustering"]["num_perm"], seed)
    views = tuple((view, k) for view, k in cfg["archetype_views"])
    logging.info("Preparing per-run prefixes at %s", cfg["checkpoints"])
    prefixes = {
        t: {key: build_prefix(trace, t, sketcher, views)
            for key, trace in traces.items() if len(trace.tokens) >= t}
        for t in cfg["checkpoints"]
    }
    options = (cfg["top_ngrams"], config["mining"]["max_gap"])
    fits, ablations, exclusion, control, chosen, references, predictions = [], [], [], [], [], [], []
    saved = results / "models"
    saved.mkdir(exist_ok=True)
    for fold, manifest in splits.loc[splits["scheme"] == "grouped"].groupby("fold"):
        fold = int(fold)
        partitions = {name: group["run_id"].tolist() for name, group in manifest.groupby("partition")}
        outer_train = runs.loc[partitions["train"]]
        outer_reference = fit_reference(outer_train, config)
        inner = GroupShuffleSplit(n_splits=1, test_size=cfg["inner_validation"], random_state=seed)
        fit_idx, valid_idx = next(inner.split(outer_train, groups=outer_train["instance_id"]))
        inner_train, inner_valid = outer_train.iloc[fit_idx], outer_train.iloc[valid_idx]
        inner_reference = fit_reference(inner_train, config)
        references.append({"fold": fold, "outer_patterns": len(outer_reference.patterns),
                           "outer_archetypes": len(outer_reference.archetypes),
                           "inner_patterns": len(inner_reference.patterns),
                           "inner_archetypes": len(inner_reference.archetypes),
                           "pattern_ranking": json.dumps(outer_reference.patterns, ensure_ascii=False)})
        for t in cfg["checkpoints"]:
            cache = prefixes[t]
            samples = {name: eligible(ids, runs, cache) for name, ids in partitions.items()}
            train, test, calibrate = (samples[name] for name in ("train", "test", "calibration"))
            fit_, valid = eligible(inner_train.index.tolist(), runs, cache), eligible(inner_valid.index.tolist(), runs, cache)
            for name, sample in samples.items():
                exclusion.append({"fold": fold, "checkpoint": t, "partition": name, "included": len(sample.ids),
                                  "excluded": len(partitions[name]) - len(sample.ids),
                                  "included_failures": int(sample.labels.sum())})
            outer_fs = FeatureSpace.fit(train.rows, outer_reference, *options)
            inner_fs = FeatureSpace.fit(fit_.rows, inner_reference, *options)
            train_x, test_x, cal_x = [outer_fs.transform(item.rows) for item in (train, test, calibrate)]
            inner_x, valid_x = [inner_fs.transform(item.rows) for item in (fit_, valid)]
            c_cost, c_bag = inner_lr(train, cfg, seed)
            light = choose_lightgbm(matrix(inner_x), fit_.labels, matrix(valid_x), valid.labels, cfg, seed)
            _, epochs = gru.fit(gru.encode(fit_.rows, inner_fs.vocabulary), fit_.labels, len(inner_fs.vocabulary),
                                cfg, seed, cfg["gru_epochs"],
                                validation=(gru.encode(valid.rows, inner_fs.vocabulary), valid.labels))
            chosen.append({"fold": fold, "checkpoint": t, "steps_c": c_cost, "bag_c": c_bag,
                           "num_leaves": light[0], "learning_rate": light[1], "n_estimators": light[2], "gru_epochs": epochs,
                           "inner_fit_tasks": len(set(fit_.tasks)), "inner_validation_tasks": len(set(valid.tasks))})
            steps_model = logistic(c_cost, seed).fit(train_x["cost"], train.labels)
            bag_model = logistic(c_bag, seed).fit(train_x["bag"], train.labels)
            trees = lightgbm(light, seed, cfg["cpu_threads"]).fit(matrix(train_x), train.labels)
            recurrent, _ = gru.fit(gru.encode(train.rows, outer_fs.vocabulary), train.labels,
                                   len(outer_fs.vocabulary), cfg, seed, epochs)
            path = saved / f"fold_{fold}_t{t}"
            path.mkdir(exist_ok=True)
            joblib.dump({"features": outer_fs, "lightgbm": trees, "steps_lr": steps_model, "bag_lr": bag_model,
                         "checkpoint": t, "sketch_config": config["clustering"], "seed": seed}, path / "tabular.joblib")
            torch.save({"state_dict": recurrent.state_dict(), "vocabulary": outer_fs.vocabulary,
                        "checkpoint": t, "config": cfg}, path / "gru.pt")
            for partition, item, groups in (("test", test, test_x), ("calibration", calibrate, cal_x)):
                scores = {
                    "steps_lr": steps_model.predict_proba(groups["cost"])[:, 1],
                    "bag_lr": bag_model.predict_proba(groups["bag"])[:, 1],
                    "lightgbm": trees.booster_.predict(matrix(groups)),
                    "gru": gru.predict(recurrent, gru.encode(item.rows, outer_fs.vocabulary), cfg["gru_batch"]),
                }
                for name, scores_ in scores.items():
                    if partition == "test":
                        fits.append({"fold": fold, "checkpoint": t, "model": name, "runs": len(item.ids),
                                     **metrics(item.labels, scores_)})
                    predictions.append(pd.DataFrame({
                        "run_id": item.ids, "fold": fold, "checkpoint": t, "partition": partition,
                        "model": name, "failed": item.labels.astype(np.int8), "score": scores_,
                    }))
            full = metrics(test.labels, trees.booster_.predict(matrix(test_x)))
            ablations.append({"fold": fold, "checkpoint": t, "dropped": "none", **full,
                              "delta_auroc": 0.0, "delta_pr_auc": 0.0})
            for group in GROUPS:
                use = tuple(name for name in GROUPS if name != group)
                if not train_x[group].shape[1] or np.all(train_x[group] == train_x[group][0]):
                    stats = full
                else:
                    model = lightgbm(light, seed, cfg["cpu_threads"]).fit(matrix(train_x, use), train.labels)
                    stats = metrics(test.labels, model.booster_.predict(matrix(test_x, use)))
                ablations.append({"fold": fold, "checkpoint": t, "dropped": group, **stats,
                                  "delta_auroc": stats["auroc"] - full["auroc"],
                                  "delta_pr_auc": stats["pr_auc"] - full["pr_auc"]})
            if t == cfg["shuffled_checkpoint"]:
                for repeat in range(1, cfg["shuffled_repeats"] + 1):
                    rng = np.random.default_rng(np.random.SeedSequence([seed, fold, repeat]))
                    randomized = outer_train.copy()
                    randomized["failed"] = rng.permutation(randomized["failed"].to_numpy())
                    shuffled_ref = fit_reference(randomized, config)
                    space = FeatureSpace.fit(train.rows, shuffled_ref, *options)
                    labels = randomized.loc[train.ids, "failed"].to_numpy()
                    control_cfg = tuple(cfg["lightgbm_candidates"][0][:2]) + (cfg["shuffled_trees"],)
                    model = lightgbm(control_cfg, seed, cfg["cpu_threads"]).fit(
                        matrix(space.transform(train.rows)), labels
                    )
                    control.append({"fold": fold, "checkpoint": t, "repeat": repeat,
                                    **metrics(test.labels, model.booster_.predict(matrix(space.transform(test.rows))))})
            logging.info("Fold %d t=%d, %d held-out runs, LightGBM AUROC %.4f, GRU epochs %d",
                         fold, t, len(test.ids), full["auroc"], epochs)

    fold_metrics = pd.DataFrame(fits)
    fold_ablation = pd.DataFrame(ablations)
    fold_metrics.to_csv(results / "prediction_fold_metrics.csv", index=False)
    write_summary(fold_metrics, ["model", "checkpoint"], results / "prediction_metrics.csv")
    fold_ablation.to_csv(results / "ablation_folds.csv", index=False)
    ablation_metrics = fold_ablation.groupby(["checkpoint", "dropped"], sort=True)[
        ["auroc", "pr_auc", "delta_auroc", "delta_pr_auc"]
    ].agg(["mean", "std"])
    ablation_metrics.columns = ["_".join(col) for col in ablation_metrics.columns]
    ablation_metrics.reset_index().to_csv(results / "ablation.csv", index=False)
    exclusions = pd.DataFrame(exclusion)
    exclusions.to_csv(results / "prefix_exclusions_by_fold.csv", index=False)
    heldout = exclusions.loc[exclusions["partition"] == "test"].groupby("checkpoint", sort=True)[
        ["included", "excluded", "included_failures"]
    ].sum().reset_index()
    heldout.to_csv(results / "prefix_exclusions.csv", index=False)
    shuffles = pd.DataFrame(control)
    shuffles.to_csv(results / "shuffled_label_control.csv", index=False)
    pd.DataFrame(chosen).to_csv(results / "prediction_hyperparameters.csv", index=False)
    pd.DataFrame(references).to_csv(results / "prediction_fold_references.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_parquet(root / "prefix_predictions.parquet", index=False)
    summary = {
        "outer_folds": fold_metrics["fold"].nunique(), "checkpoints": cfg["checkpoints"],
        "llm_judge": "skipped; no API key was supplied",
        "shuffled_checkpoint": cfg["shuffled_checkpoint"], "shuffled_repeats": cfg["shuffled_repeats"],
        "shuffled_auroc_mean": shuffles["auroc"].mean(), "shuffled_auroc_std": shuffles["auroc"].std(),
        "shuffled_pr_auc_mean": shuffles["pr_auc"].mean(), "shuffled_pr_auc_std": shuffles["pr_auc"].std(),
        "config": config, "experiments_commit": config["experiments_commit"],
    }
    (results / "phase6.json").write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("Shuffled control AUROC %.4f ± %.4f at t=%d", summary["shuffled_auroc_mean"],
                 summary["shuffled_auroc_std"], cfg["shuffled_checkpoint"])
    low, high = cfg.get("shuffled_acceptance", [0.45, 0.55])
    if not low <= summary["shuffled_auroc_mean"] <= high:
        raise SystemExit("Shuffled-label mean AUROC is outside the acceptance range")


if __name__ == "__main__":
    main()
