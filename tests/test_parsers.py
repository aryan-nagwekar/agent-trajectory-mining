"""Golden action and outcome sequences from six trimmed Phase 1 traces."""

from pathlib import Path

import pytest

from atm.parsers import PARSERS
from atm.vocab import classify

FIXTURES = Path(__file__).parent / "fixtures"
DJANGO = "django__django-11734"
SKLEARN = "scikit-learn__scikit-learn-9288"


@pytest.mark.parametrize(
    "framework,task,expected",
    [
        ("SWE-agent", DJANGO, "view view view view create"),
        ("SWE-agent", SKLEARN, "view view view view create"),
        ("OpenHands", DJANGO, "think view view run_script:error view"),
        ("OpenHands", SKLEARN, "think view view run_script:ok create"),
        ("mini-SWE-agent", DJANGO, "list:ok search:ok view:ok search:ok view:ok"),
        ("mini-SWE-agent", SKLEARN, "search:ok search:ok view:ok search:ok view:ok"),
    ],
)
def test_native_prefix(framework: str, task: str, expected: str) -> None:
    path = FIXTURES / f"{framework.lower()}_{task}.json"
    trajectory = PARSERS[framework](path, task)
    tokens = []
    for step in trajectory.steps:
        action, outcome = classify(step)
        tokens.append(action if outcome == "none" else f"{action}:{outcome}")
    assert tokens == expected.split()


@pytest.mark.parametrize(
    "framework,filename,expected",
    [
        ("SWE-agent", "swe-agent_legacy_pydata__xarray-6992.json",
         "create edit run_script view view"),
        ("OpenHands", "openhands_devstral_pydata__xarray-6992.json",
         "think list view search:ok search"),
    ],
)
def test_expanded_source_variant(framework: str, filename: str, expected: str) -> None:
    trajectory = PARSERS[framework](FIXTURES / filename, "pydata__xarray-6992")
    classified = [classify(step) for step in trajectory.steps]
    assert [action if outcome == "none" else f"{action}:{outcome}"
            for action, outcome in classified] == expected.split()
