"""Rule-level checks for bash commands and outcome tagging."""

import pytest

from atm.normalize import collapse
from atm.parsers.base import Step
from atm.vocab import classify, shell_action


@pytest.mark.parametrize(
    "command,expected",
    [
        ("cd /repo && grep -rn foo .", "search"),
        ("cd /repo\nls -al", "list"),
        ("sed -i 's/a/b/' x.py", "edit"),
        ("sed -n '4,9p' x.py", "view"),
        ("python -m pytest tests/test_x.py", "run_test"),
        ("python reproduce.py", "run_script"),
        ("python -m pip install -e .", "install"),
        ("git checkout -- file", "revert"),
        ("git diff --cached", "vcs"),
        ("ls -R", "search"),
        ("find . -name '*.py' | head -10", "search"),
        ("cat > file.py <<'PY'\nprint(1)\nPY", "create"),
        ("cat file.py", "view"),
        ("bash -lc 'cd /repo && ls'", "list"),
        ("./tests/runtests.py", "run_test"),
        ("python -m pytest --version", "run_script"),
        ("echo MICRO_SWE_AGENT_FINAL_OUTPUT && git add -A", "submit"),
        ("echo MINI_SWE_AGENT_FINAL_OUTPUT && git diff --cached", "submit"),
        ("cd /repo && timeout 5 python -m pytest --pdb test_x.py", "run_test"),
        ("DJANGO_SETTINGS_MODULE=settings django-admin test test_validators", "run_test"),
        ("django-admin startproject test_project", "create"),
        ("bin/test sympy/concrete", "run_test"),
        ("sphinx-build -b html . _build", "run_script"),
        ("echo diagnostics", "other_bash"),
    ],
)
def test_shell_action(command: str, expected: str) -> None:
    assert shell_action(command) == expected


@pytest.mark.parametrize(
    "step,expected",
    [
        (Step("bash", "pytest", "3 failed, 1 passed", exit_code=1), ("run_test", "fail")),
        (Step("bash", "pytest", "1 passed", exit_code=0), ("run_test", "pass")),
        (Step("bash", "pytest", "", exit_code=0), ("run_test", "pass")),
        (Step("bash", "python repro.py", "Traceback (most recent call last):"), ("run_script", "error")),
        (Step("str_replace_editor", "str_replace /file", "Error: ambiguous match"), ("edit", "error")),
        (Step("error_format", "", "Provide a single Bash code block"), ("error_format", "error")),
    ],
)
def test_outcome(step: Step, expected: tuple[str, str]) -> None:
    assert classify(step) == expected


def test_collapse() -> None:
    assert collapse(["view", "view", "view", "edit", "view", "view"]) == ["view×3+", "edit", "view×2"]
