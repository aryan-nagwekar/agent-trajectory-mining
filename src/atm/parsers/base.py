"""Native action records shared by the three readers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Step:
    """One observable agent action and its corresponding observation."""

    raw_action_type: str
    raw_command: str
    observation: str = ""
    model_chars: int = 0
    exit_code: int | None = None
    step_tokens: int | None = None


@dataclass
class Trajectory:
    steps: list[Step]
    total_tokens: int | None = None


class Parser(Protocol):
    def __call__(self, path: Path, instance_id: str) -> Trajectory: ...


def message_text(content: str | list[dict] | None) -> str:
    """Extract text from a plain or multipart message."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("unexpected message content type")
    return "\n".join(part["text"] for part in content if part.get("type") == "text")
