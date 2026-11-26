"""Read mini-SWE-agent transcripts and their shell response envelopes."""

import re
from pathlib import Path

import ijson

from atm.parsers.base import Step, Trajectory, message_text

FENCE = re.compile(r"^\s*```(?:bash|sh|shell)[ \t]*\r?\n(.*?)^\s*```[ \t]*$", re.M | re.S | re.I)
RETURN = re.compile(r"<returncode>\s*(-?\d+)\s*</returncode>")
OUTPUT = re.compile(r"<output>(.*?)</output>", re.S)


def parse(path: Path, instance_id: str) -> Trajectory:
    with path.open("rb") as stream:
        identity = next(ijson.items(stream, "instance_id"), None)
        if identity is not None and identity != instance_id:
            raise ValueError(f"embedded instance_id {identity!r} differs from {instance_id!r}")
        stream.seek(0)
        status = next(ijson.items(stream, "info.exit_status"), None)
        if not status:
            raise ValueError("missing mini-SWE-agent exit status")
        stream.seek(0)
        steps = []
        pending = None
        for row in ijson.items(stream, "messages.item"):
            role = row["role"]
            if role == "assistant":
                if pending is not None:
                    raise ValueError("assistant action has no response")
                text = message_text(row.get("content"))
                commands = FENCE.findall(text)
                pending = ("bash", commands[0].strip(), len(text)) if len(commands) == 1 else (
                    "error_format", "", len(text)
                )
            elif role in {"user", "tool"} and pending is not None:
                action_type, command, chars = pending
                observation = message_text(row.get("content"))
                match = RETURN.search(observation)
                output = OUTPUT.search(observation)
                # Preserve warnings and formatting errors outside the output envelope.
                if output and match:
                    observation = (observation[:match.start()] +
                                   observation[match.end():output.start()] +
                                   output[1] + observation[output.end():]).strip()
                steps.append(Step(action_type, command, observation, chars,
                                  int(match[1]) if match else None))
                pending = None
        if pending is not None:
            raise ValueError("last mini-SWE-agent command has no response")
    return Trajectory(steps)
