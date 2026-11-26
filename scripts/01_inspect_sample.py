"""Check official outcomes and print native record excerpts for Phase 1."""

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def show(value: Any) -> None:
    """Keep each excerpt on a single line."""
    print(json.dumps(value, ensure_ascii=False))


def clip(value: str, width: int = 125) -> str:
    return value if len(value) <= width else value[:width] + "…"


def preview(framework: str, raw: Any) -> None:
    """Display source fields without assigning action tokens or deriving outcomes."""
    if framework == "SWE-agent":
        print("Root keys:", ", ".join(sorted(raw)))
        rows = raw["trajectory"]
        print("First trajectory row keys:", ", ".join(sorted(rows[0])))
        for row in rows[:5]:
            show({key: clip(row[key]) for key in ("action", "observation")})
        return

    if framework == "OpenHands":
        print("Root: array of messages")
        tool_replies = {row["tool_call_id"]: row for row in raw if row["role"] == "tool"}
        calls = [call for row in raw for call in row.get("tool_calls", [])]
        for call in calls[:5]:
            result = tool_replies[call["id"]]
            show({
                "id": call["id"],
                "name": call["function"]["name"],
                "arguments": clip(call["function"]["arguments"]),
                "tool_call_id": result["tool_call_id"],
                "tool_content": clip(result["content"][0]["text"]),
            })
        return

    print("Root keys:", ", ".join(sorted(raw)))
    messages = raw["messages"]
    turns = [idx for idx, row in enumerate(messages) if row["role"] == "assistant"]
    for idx in turns[:5]:
        assistant = messages[idx]
        reply = messages[idx + 1]
        # Show the fenced command and skip the preceding prose for display.
        _, fence, suffix = assistant["content"].partition("```")
        excerpt = fence + suffix if fence else assistant["content"]
        show({
            assistant["role"]: clip(excerpt),
            reply["role"]: clip(reply["content"][0]["text"]),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--limit", type=int, default=8, help="Trajectories per framework")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")

    config = yaml.safe_load(args.config.read_text())
    print("Experiments revision:", config["experiments_commit"])
    sources = config["submissions"]
    root = Path("data/raw")
    # This per-instance results file enumerates all 500 Verified task IDs.
    mini = next(source for source in sources if source["framework"] == "mini-SWE-agent")
    verified = json.loads((root / mini["submission"] / mini["results_path"]).read_text())

    for source in sources:
        framework = source["framework"]
        directory = root / source["submission"]
        outcomes = json.loads((directory / source["results_path"]).read_text())
        print(f"{framework} | {source['submission']}")
        print(f"Official outcomes from {source['results_path']} (excerpt):")
        if framework == "mini-SWE-agent":
            assert all(type(row["resolved"]) is bool for row in outcomes.values())
            labels = {task: row["resolved"] for task, row in outcomes.items()}
            example_key = next(iter(outcomes))
            show({example_key: outcomes[example_key]})
        else:
            resolved = set(outcomes["resolved"])
            excluded = set(outcomes["no_generation"]) | set(outcomes["no_logs"])
            assert len(resolved) == len(outcomes["resolved"])
            assert (resolved | excluded) <= verified.keys() and not (resolved & excluded)
            labels = {task: task in resolved for task in verified if task not in excluded}
            show({
                "resolved": outcomes["resolved"][:2],
                "no_generation": outcomes["no_generation"],
                "no_logs": outcomes["no_logs"],
            })

        sample = list(source["trajectories"].items())[:args.limit]
        assert sample and len({Path(key).name for _, key in sample}) == len(sample)
        print(f"Official resolved entries: {sum(labels.values())}/{len(verified)} Verified tasks")
        print("Officially flagged no generation or no evaluation logs:", len(verified) - len(labels))
        print("Join results (0=resolved, 1=unresolved):")
        for task, key in sample:
            name = Path(key).name
            assert name.split(".")[0] == task
            raw = json.loads((directory / "trajs" / name).read_text())
            if framework == "SWE-agent":
                assert raw["environment"] == task
            elif framework == "mini-SWE-agent":
                assert raw["instance_id"] == task
            else:
                assert isinstance(raw, list) and raw[0]["role"] == "system"
            print(f"  {task}: {int(not labels[task])}")
            if task in config["example_tasks"]:
                preview(framework, raw)
        print(f"Joined {len(sample)}/{len(sample)} raw files to eligible task IDs\n", flush=True)


if __name__ == "__main__":
    main()
