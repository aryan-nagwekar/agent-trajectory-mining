"""Read the native SWE-agent trajectory export."""

from pathlib import Path

import ijson

from atm.parsers.base import Step, Trajectory


def parse(path: Path, instance_id: str) -> Trajectory:
    steps = []
    identity = None
    stats = {}
    terminal = None
    with path.open("rb") as stream:
        # ijson skips the duplicated conversation history without building it in memory.
        for prefix, event, value in ijson.parse(stream):
            if prefix == "environment" and event == "string":
                identity = value
            elif prefix == "info.exit_status" and event == "string":
                terminal = value
            elif prefix.startswith("info.model_stats.") and event == "number":
                stats[prefix.rsplit(".", 1)[1]] = value
            elif prefix == "trajectory.item" and event == "start_map":
                row = {}
            elif prefix.startswith("trajectory.item.") and event == "string":
                field = prefix.removeprefix("trajectory.item.")
                if field in {"action", "observation", "response", "thought"}:
                    row[field] = value
            elif prefix == "trajectory.item" and event == "end_map":
                command = row["action"].strip()
                observation = row["observation"]
                action = command.split(maxsplit=1)[0] if command else "think"
                steps.append(Step(action, command, observation,
                                  len(row.get("response", row.get("thought", "") + command))))
    if identity not in {None, "swe_main", instance_id}:
        raise ValueError(f"embedded environment {identity!r} differs from {instance_id!r}")
    if not terminal:
        raise ValueError("missing SWE-agent exit status")
    tokens = None
    if {"tokens_sent", "tokens_received"} <= stats.keys():
        tokens = int(stats["tokens_sent"] + stats["tokens_received"])
    return Trajectory(steps, tokens)
