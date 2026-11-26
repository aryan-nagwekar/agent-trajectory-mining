"""Check prefix invariance and train-only feature fitting."""

import numpy as np
import pandas as pd

from atm.clustering.minhash_lsh import Sketcher
from atm.features import FeatureSpace, Trace, build_prefix, fit_reference, matrix


def example(actions: tuple[str, ...]) -> Trace:
    return Trace(actions, actions, ("none",) * len(actions), (10,) * len(actions), (None,) * len(actions))


def test_features_do_not_depend_on_the_suffix() -> None:
    sketcher = Sketcher(128, 42)
    views = (("tokens", 4), ("actions", 3))
    start = ("search", "edit", "view", "run_test", "view")
    padded = example(start + ("submit", "create", "other_bash"))
    original = build_prefix(example(start), 5, sketcher, views)
    extended = build_prefix(padded, 5, sketcher, views)
    reference = fit_reference(pd.DataFrame({
        "failed": [0, 0, 1, 1], "tokens": [list(start)] * 4, "actions": [list(start)] * 4,
    }), toy_config())
    space = FeatureSpace.fit([original], reference, 10, 2)
    np.testing.assert_array_equal(matrix(space.transform([original])), matrix(space.transform([extended])))
    assert not extended.cost[-1]


def test_adding_heldout_tokens_leaves_train_refs_and_vocab_unchanged() -> None:
    cfg = toy_config()
    data = pd.DataFrame({
        "failed": [0, 0, 1, 1, 1],
        "tokens": [list("xyzp"), list("xyzp"), list("abcd"), list("abcd"), list("mvt")],
        "actions": [list("xyzp"), list("xyzp"), list("abcd"), list("abcd"), list("mvt")],
    })
    ref = fit_reference(data.iloc[:4], cfg)
    changed = data.copy(deep=True)
    changed.at[4, "tokens"] = ["heldout-only"]
    changed.at[4, "actions"] = ["heldout-only"]
    changed.at[4, "failed"] = 0
    fresh = fit_reference(changed.iloc[:4], cfg)
    assert ref.patterns == fresh.patterns and len(ref.archetypes) == len(fresh.archetypes) == 1
    for (view, sketch), (fresh_view, fresh_sketch) in zip(ref.archetypes, fresh.archetypes, strict=True):
        assert view == fresh_view
        np.testing.assert_array_equal(sketch, fresh_sketch)
    prefix = build_prefix(example(tuple("abcd")), 4, Sketcher(128, 42), (("actions", 3),))
    fitted = FeatureSpace.fit([prefix], ref, 10, 2)
    assert "heldout-only" not in fitted.vocabulary
    assert any("a" in pattern for pattern in ref.patterns)


def toy_config() -> dict:
    return {"seed": 42, "mining": {"min_support": 0.25, "max_pattern_len": 2, "max_gap": 2},
            "clustering": {"k": 4, "num_perm": 128, "threshold": .6, "bands": 32, "rows": 4,
                           "min_size": 2, "max_component": 250},
            "prediction": {"top_patterns": 5, "archetype_views": [("actions", 3)]}}
