"""Verify split invariants on a cohort with unbalanced and incomplete tasks."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from atm.difficulty import task_difficulty
from atm.splits import make_splits, validate_splits


def cohort() -> pd.DataFrame:
    return pd.DataFrame([
        {"instance_id": f"task-{task:02d}", "run_id": f"{task}-{framework}-{rep}",
         "framework": framework, "failed": (task + rep) % 2}
        for task in range(15) for framework in ("a", "b", "c")
        for rep in range((task % 2) + 1)
    ])


def test_all_splits_are_task_disjoint_and_cover_test_once() -> None:
    runs = cohort()
    manifest = make_splits(runs, 5, 0.2, 42)
    assert_frame_equal(manifest, make_splits(runs.sample(frac=1, random_state=1), 5, 0.2, 42))
    for (scheme, framework, fold), rows in manifest.groupby(["scheme", "held_framework", "fold"]):
        by_partition = {name: set(group["instance_id"]) for name, group in rows.groupby("partition")}
        assert len(by_partition["train"]) == 9
        assert len(by_partition["calibration"]) == 3
        assert len(by_partition["test"]) == 3
        assert not by_partition["train"] & by_partition["test"]
        assert not by_partition["calibration"] & by_partition["test"]
        assert not by_partition["train"] & by_partition["calibration"]
        if scheme == "lofo":
            assert set(rows.loc[rows["partition"] == "test", "framework"]) == {framework}
            assert framework not in set(rows.loc[rows["partition"] != "test", "framework"])
    for (_, _), group in manifest.loc[manifest["partition"] == "test"].groupby(["scheme", "held_framework"]):
        assert not group["run_id"].duplicated().any()
        assert group.groupby("instance_id")["fold"].nunique().eq(1).all()


def test_rejects_a_task_shifted_from_test_to_train() -> None:
    runs = cohort()
    manifest = make_splits(runs, 5, 0.2, 42)
    idx = manifest.loc[(manifest["scheme"] == "grouped") & manifest["partition"].eq("test")].index[0]
    manifest.loc[idx, "partition"] = "train"
    with pytest.raises(ValueError, match="task overlap"):
        validate_splits(manifest, runs)


def test_difficulty_cutoffs_are_exact_and_bins_are_analysis_only() -> None:
    runs = pd.DataFrame({"instance_id": ["hard"] * 3 + ["medium"] * 3 + ["easy"] * 3,
                         "failed": [0, 1, 1, 0, 0, 1, 0, 0, 0]})
    result = task_difficulty(runs, ["1/3", "2/3"]).set_index("instance_id")
    assert result["difficulty"].to_dict() == {"easy": "easy", "hard": "hard", "medium": "medium"}
    assert "difficulty" not in make_splits(cohort(), 5, 0.2, 42).columns
