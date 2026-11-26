"""Protect manual validation annotations during normalization reruns."""

from pathlib import Path

from atm.normalize import ValidationSample


def test_existing_annotations_are_preserved(tmp_path: Path) -> None:
    target = tmp_path / "validation.csv"
    original = "raw_command,human_action\nls,list\n"
    target.write_text(original)
    sampler = ValidationSample(42, 1, ["framework"])
    sampler.add("framework", "git diff", "", "vcs", "none")
    assert sampler.write(target, preserve=True) == 1
    assert target.read_text() == original
