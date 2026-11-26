"""Refit archetype-similarity features and evaluate task-disjoint framework transfer."""

import argparse
import json
import logging
import random
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import GroupShuffleSplit

from atm.clustering.minhash_lsh import Sketcher, approximate_medoid
from atm.clustering.sweep import choose, sweep
from atm.features import GROUPS, FeatureSpace, Reference, build_prefix, load_traces, matrix
from atm.mining.discriminative import candidate_patterns
from atm.models import gru
from atm.models.baselines import choose_lightgbm, lightgbm, logistic
from atm.predict import eligible, inner_lr, metrics, write_summary
from atm.splits import validate_splits


def fit_medoids(training: pd.DataFrame, base: pd.Series, config: dict) -> tuple[tuple, dict]:
    """Select settings from the training failures, using the descriptive rule."""
    source = training.loc[training["failed"] == 1].sort_index()
    failures = source.reset_index()[["run_id", "framework"]]
    cfg = {**config["clustering"], **config["archetype_sweep"]}
    frame, alternatives = sweep(failures, base, cfg, config["seed"], log=False)
    try:
        chosen = choose(frame, cfg["selection_min_coverage"], cfg["selection_max_largest_share"])
        rule = "coverage_and_diversity"
    except ValueError:
        pool = frame.loc[frame["communities"] > 0].sort_values(["threshold", "coverage", "k"], ascending=False)
        if pool.empty:
            return (), {"k": None, "threshold": None, "communities": 0, "covered_runs": 0, "rule": "no_eligible_community"}
        chosen, rule = pool.iloc[0], "highest_threshold_with_community"
    k, cutoff = int(chosen["k"]), float(chosen["threshold"])
    sets, groups = alternatives[k, cutoff]
    sketcher = Sketcher(config["clustering"]["num_perm"], config["seed"])
    signatures = [sketcher(items) for items in sets]
    medoids = tuple((("actions", k), signatures[approximate_medoid(signatures, group)].hashvalues) for group in groups)
    return medoids, {"k": k, "threshold": cutoff, "communities": len(groups),
                     "covered_runs": sum(map(len, groups)), "rule": rule}


def train_patterns(frame: pd.DataFrame, config: dict) -> tuple:
    support = candidate_patterns(frame, config["mining"])
    n_fail = int(frame["failed"].sum())
    n_success = len(frame) - n_fail
    if not n_fail or not n_success:
        raise ValueError("both outcomes are required in a training split")
    return tuple(sorted(support, key=lambda p: (
        -abs(support[p][0] / n_fail - support[p][1] / n_success), len(p), p,
    ))[:config["prediction"]["top_patterns"]])


def prepare(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    path = Path("data/processed")
    raw = pd.read_parquet(path / "runs.parquet")
    sequences = pd.read_parquet(path / "collapsed_sequences.parquet")
    splits = pd.read_parquet(path / "splits.parquet")
    validate_splits(splits, raw)
    runs = raw.merge(sequences, on="run_id", validate="one_to_one").set_index("run_id")
    traces = load_traces(path / "steps.parquet")
    if set(traces) != set(runs.index) or not all(len(traces[key].tokens) == n for key, n in runs["n_steps"].items()):
        raise ValueError("normalized traces and run counts disagree")
    base = pd.Series({key: trace.actions for key, trace in traces.items()})
    views = tuple(("actions", k) for k in config["archetype_sweep"]["k_values"])
    sketcher = Sketcher(config["clustering"]["num_perm"], config["seed"])
    prefixes = {
        t: {key: build_prefix(trace, t, sketcher, views) for key, trace in traces.items() if len(trace.tokens) >= t}
        for t in config["prediction"]["checkpoints"]
    }
    return runs, splits, base, prefixes


def score_frame(item, scores: np.ndarray, fold: int, t: int, partition: str, model: str, held: str = "") -> pd.DataFrame:
    return pd.DataFrame({
        "run_id": item.ids, "fold": fold, "checkpoint": t, "partition": partition, "model": model,
        "failed": item.labels.astype(np.int8), "score": scores, "held_framework": held,
    })


def revise(config: dict, runs: pd.DataFrame, splits: pd.DataFrame, base: pd.Series, prefixes: dict) -> None:
    cfg, seed = config["prediction"], config["seed"]
    results = Path("results")
    data = Path("data/processed")
    hyper = pd.read_csv(results / "prediction_hyperparameters.csv").set_index(["fold", "checkpoint"])
    ablation, predicted, settings = [], [], []
    revised = results / "models_revised"
    for fold, manifest in splits.loc[splits["scheme"] == "grouped"].groupby("fold"):
        ids = {key: group["run_id"].tolist() for key, group in manifest.groupby("partition")}
        medoids, choice = fit_medoids(runs.loc[ids["train"]], base, config)
        settings.append({"fold": fold, **choice})
        logging.info("Grouped fold %d medoids: %s", fold, choice)
        for t, cache in prefixes.items():
            train, calibration, test = [eligible(ids[p], runs, cache) for p in ("train", "calibration", "test")]
            previous = joblib.load(results / "models" / f"fold_{fold}_t{t}" / "tabular.joblib")
            space = replace(previous["features"], reference=Reference(previous["features"].reference.patterns, medoids))
            train_x, cal_x, test_x = [space.transform(item.rows) for item in (train, calibration, test)]
            h = hyper.loc[fold, t]
            params = (int(h.num_leaves), float(h.learning_rate), int(h.n_estimators))
            full = lightgbm(params, seed, cfg["cpu_threads"]).fit(matrix(train_x), train.labels)
            keep = tuple(group for group in GROUPS if group != "archetypes")
            reduced = lightgbm(params, seed, cfg["cpu_threads"]).fit(matrix(train_x, keep), train.labels)
            test_scores = full.booster_.predict(matrix(test_x))
            predicted.extend([
                score_frame(test, test_scores, fold, t, "test", "lightgbm"),
                score_frame(calibration, full.booster_.predict(matrix(cal_x)), fold, t, "calibration", "lightgbm"),
            ])
            before = metrics(test.labels, test_scores)
            after = metrics(test.labels, reduced.booster_.predict(matrix(test_x, keep)))
            ablation.append({"fold": fold, "checkpoint": t, "archetypes": len(medoids),
                             **{f"full_{k}": v for k, v in before.items()},
                             **{f"without_{k}": v for k, v in after.items()},
                             **{f"delta_{k}": after[k] - before[k] for k in before}})
            out = revised / f"fold_{fold}_t{t}"
            out.mkdir(parents=True, exist_ok=True)
            joblib.dump({**previous, "features": space, "lightgbm": full, "archetype_choice": choice},
                        out / "tabular.joblib")
    original = pd.read_parquet(data / "prefix_predictions.parquet").assign(held_framework="")
    pd.concat([original.loc[original["model"] != "lightgbm"], *predicted], ignore_index=True).to_parquet(
        data / "prefix_predictions_revised.parquet", index=False,
    )
    frame = pd.DataFrame(ablation)
    frame.to_csv(results / "archetype_ablation_refit_folds.csv", index=False)
    summary = frame.groupby("checkpoint").agg({
        column: ["mean", "std"] for column in frame if column.startswith(("full_", "without_", "delta_"))
    })
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.reset_index().to_csv(results / "archetype_ablation_refit.csv", index=False)
    pd.DataFrame(settings).to_csv(results / "archetype_refit_settings.csv", index=False)


def transfer(config: dict, runs: pd.DataFrame, splits: pd.DataFrame, base: pd.Series, prefixes: dict) -> None:
    cfg, seed = config["prediction"], config["seed"]
    results, data = Path("results"), Path("data/processed")
    fits, predicted, references = [], [], []
    for (held, fold), manifest in splits.loc[splits["scheme"] == "lofo"].groupby(["held_framework", "fold"]):
        partitions = {kind: group["run_id"].tolist() for kind, group in manifest.groupby("partition")}
        train_full = runs.loc[partitions["train"]]
        fit_idx, val_idx = next(GroupShuffleSplit(n_splits=1, test_size=cfg["inner_validation"], random_state=seed)
                                .split(train_full, groups=train_full["instance_id"]))
        inner_train, inner_validation = train_full.iloc[fit_idx], train_full.iloc[val_idx]
        full_med, full_choice = fit_medoids(train_full, base, config)
        inner_med, inner_choice = fit_medoids(inner_train, base, config)
        full_ref = Reference(train_patterns(train_full, config), full_med)
        inner_ref = Reference(train_patterns(inner_train, config), inner_med)
        references.append({"held_framework": held, "fold": fold, **full_choice,
                           "inner_k": inner_choice["k"], "inner_threshold": inner_choice["threshold"],
                           "inner_communities": len(inner_med), "patterns": len(full_ref.patterns)})
        logging.info("LOFO %s fold %d, %d training archetypes", held, fold, len(full_med))
        for t, cache in prefixes.items():
            train, cal, test = [eligible(partitions[kind], runs, cache) for kind in ("train", "calibration", "test")]
            inner_fit, inner_val = [eligible(frame.index.tolist(), runs, cache) for frame in (inner_train, inner_validation)]
            options = (cfg["top_ngrams"], config["mining"]["max_gap"])
            full_space = FeatureSpace.fit(train.rows, full_ref, *options)
            inner_space = FeatureSpace.fit(inner_fit.rows, inner_ref, *options)
            train_x, cal_x, test_x = [full_space.transform(item.rows) for item in (train, cal, test)]
            inner_x, val_x = [inner_space.transform(item.rows) for item in (inner_fit, inner_val)]
            steps_c, bag_c = inner_lr(train, cfg, seed)
            tree_params = choose_lightgbm(matrix(inner_x), inner_fit.labels, matrix(val_x), inner_val.labels, cfg, seed)
            _, epochs = gru.fit(gru.encode(inner_fit.rows, inner_space.vocabulary), inner_fit.labels,
                                len(inner_space.vocabulary), cfg, seed, cfg["gru_epochs"],
                                (gru.encode(inner_val.rows, inner_space.vocabulary), inner_val.labels))
            steps_model = logistic(steps_c, seed).fit(train_x["cost"], train.labels)
            bag_model = logistic(bag_c, seed).fit(train_x["bag"], train.labels)
            tree = lightgbm(tree_params, seed, cfg["cpu_threads"]).fit(matrix(train_x), train.labels)
            seq_model, _ = gru.fit(gru.encode(train.rows, full_space.vocabulary), train.labels,
                                   len(full_space.vocabulary), cfg, seed, epochs)
            # Later policy evaluation uses these scores, independently of threshold calibration.
            for part, sample, groups in [("calibration", cal, cal_x), ("test", test, test_x)]:
                scores = {"steps_lr": steps_model.predict_proba(groups["cost"])[:, 1],
                          "bag_lr": bag_model.predict_proba(groups["bag"])[:, 1],
                          "lightgbm": tree.booster_.predict(matrix(groups)),
                          "gru": gru.predict(seq_model, gru.encode(sample.rows, full_space.vocabulary), cfg["gru_batch"])}
                for model, values in scores.items():
                    predicted.append(score_frame(sample, values, fold, t, part, model, held))
                    if part == "test":
                        fits.append({"held_framework": held, "fold": fold, "checkpoint": t, "model": model,
                                     "runs": len(sample.ids), **metrics(sample.labels, values)})
    frame = pd.DataFrame(fits)
    frame.to_csv(results / "transfer_fold_metrics.csv", index=False)
    write_summary(frame, ["held_framework", "model", "checkpoint"], results / "transfer_metrics.csv")
    pd.concat(predicted, ignore_index=True).to_parquet(data / "transfer_predictions.parquet", index=False)
    pd.DataFrame(references).to_csv(results / "transfer_training_references.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["refit", "transfer"])
    parser.add_argument("--config", type=Path, default=Path("configs/expanded.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["prediction"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(f"results/logs/{args.stage}.log"),
    ])
    task = revise if args.stage == "refit" else transfer
    task(config, *prepare(config))


if __name__ == "__main__":
    main()
