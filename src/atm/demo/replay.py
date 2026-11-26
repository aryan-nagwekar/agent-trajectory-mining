"""Read held-out replay artifacts and score observed prefixes with saved models."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from atm.clustering.minhash_lsh import Sketcher
from atm.features import Trace, build_prefix, matrix


def replay_data(root: Path) -> Path:
    """Use locally reproduced data when its replay inputs are all present."""
    local = root / "data/processed"
    required = ("runs.parquet", "steps.parquet", "prefix_predictions_revised.parquet")
    return local if all((local / name).is_file() for name in required) else root / "demo_assets"


def load_replay_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = replay_data(root)
    runs = pd.read_parquet(base / "runs.parquet", columns=[
        "run_id", "framework", "instance_id", "failed", "n_steps", "total_chars",
    ])
    recorded = pd.read_parquet(base / "prefix_predictions_revised.parquet", filters=[
        ("model", "=", "lightgbm"), ("partition", "=", "test"),
    ])
    if recorded.groupby("run_id")["fold"].nunique().max() != 1:
        raise ValueError("each held-out run must belong to one outer test fold")
    gates = pd.read_csv(root / "results/early_stop_thresholds.csv").query("model == 'lightgbm' and policy == 'multi'")
    return runs, recorded, gates


def replay_runs(runs: pd.DataFrame, recorded: pd.DataFrame, gates: pd.DataFrame) -> list[str]:
    """Choose real examples with failure, success, and contrasting policy decisions."""
    cuts = gates.loc[gates["alpha"].eq(.10), ["fold", "threshold_checkpoint", "threshold"]]
    scores = recorded.merge(cuts, left_on=["fold","checkpoint"], right_on=["fold","threshold_checkpoint"],
                            validate="many_to_one").merge(runs[["run_id", "n_steps"]], on="run_id")
    stopped = set(scores.loc[scores["score"].gt(scores["threshold"]) & scores["n_steps"].gt(scores["checkpoint"]), "run_id"])
    catalog = runs.loc[runs["run_id"].isin(recorded["run_id"])].sort_values(
        ["n_steps", "run_id"]
    ).assign(preferred=lambda x: x["n_steps"].between(41, 75))
    catalog = catalog.sort_values("preferred", ascending=False, kind="stable")
    picks = []
    for failure, prefer_stop in [(1, True), (1, False), (0, False)]:
        remaining = catalog.loc[catalog["failed"].eq(failure) & ~catalog["run_id"].isin(picks)]
        best = remaining.loc[remaining["run_id"].isin(stopped).eq(prefer_stop)]
        if len(remaining):
            picks.append((best if len(best) else remaining).iloc[0]["run_id"])
    return picks


def load_steps(root: Path, run_id: str) -> pd.DataFrame:
    return pd.read_parquet(replay_data(root) / "steps.parquet", filters=[("run_id","=",run_id)]).sort_values("step_idx")


def get_trace(steps: pd.DataFrame) -> Trace:
    return Trace(tuple(steps["token"]), tuple(steps["action"]), tuple(steps["outcome"]), tuple(steps["step_chars"]),
                 tuple(None if pd.isna(value) else int(value) for value in steps["step_tokens"]))


def model_bundle(root: Path, fold: int, checkpoint: int) -> dict:
    return joblib.load(root / "results/models_revised" / f"fold_{fold}_t{checkpoint}" / "tabular.joblib")


def predict_prefix(bundle: dict, trace: Trace, t: int) -> float:
    config = bundle["sketch_config"]
    space = bundle["features"]
    views = tuple(sorted({view for view, _ in space.reference.archetypes}))
    prefix = build_prefix(trace, t, Sketcher(config["num_perm"], bundle["seed"]), views)
    return float(bundle["lightgbm"].booster_.predict(matrix(space.transform([prefix])))[0])


def apply_policy(frame: pd.DataFrame, n_steps: int) -> int | None:
    """Only scores strictly above a threshold while another step remains can stop."""
    accepted = frame.loc[frame["checkpoint"].lt(n_steps) & frame["score"].gt(frame["threshold"])]
    return int(accepted["checkpoint"].min()) if len(accepted) else None


def replay_frame(recorded: pd.DataFrame, gates: pd.DataFrame, run_id: str, alpha: float) -> pd.DataFrame:
    scores = recorded.loc[recorded["run_id"].eq(run_id), ["fold","checkpoint","score"]]
    thresholds = gates.loc[gates["alpha"].eq(alpha), ["fold","threshold_checkpoint","threshold"]]
    return scores.merge(thresholds, left_on=["fold","checkpoint"], right_on=["fold","threshold_checkpoint"],
                        validate="one_to_one").sort_values("checkpoint")
