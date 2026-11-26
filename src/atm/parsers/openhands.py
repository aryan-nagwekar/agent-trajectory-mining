"""Read OpenHands exports including structured calls and text-serialized tools."""

import json
import re
from pathlib import Path

import ijson

from atm.parsers.base import Step, Trajectory, message_text

TEXT_CALL = re.compile(r"<function=([\w.-]+)>(.*?)</function>", re.S)
TEXT_PARAM = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.S)
TEXT_REPLY = re.compile(r"^EXECUTION RESULT of \[([\w.-]+)\]:\s*")


def tool_command(name: str, arguments: dict) -> str:
    command = str(arguments.get("command", ""))
    if name in {"str_replace_editor", "file_editor"}:
        return f"{command} {arguments.get('path', '')}".strip()
    return command


def parse(path: Path, instance_id: str) -> Trajectory:
    events = []
    replies = {}
    identifiers = set()
    text_pending: list[tuple[str, str]] = []
    with path.open("rb") as stream:
        for index, row in enumerate(ijson.items(stream, "item")):
            role = row["role"]
            if role == "assistant":
                text = message_text(row.get("content"))
                calls = row.get("tool_calls") or []
                if not calls:
                    text_calls = TEXT_CALL.findall(text)
                    for position, (name, body) in enumerate(text_calls):
                        call_id = f"text:{index}:{position}"
                        command = tool_command(name, dict(TEXT_PARAM.findall(body)))
                        text_pending.append((name, call_id))
                        events.append((call_id, name, command, len(text) if position == 0 else 0))
                    if not text_calls and text.strip():
                        action = "error_format" if "<function=" in text else "think"
                        events.append((None, action, "", len(text)))
                for position, call in enumerate(calls):
                    call_id = call["id"]
                    if call_id in identifiers:
                        raise ValueError(f"duplicate OpenHands tool call: {call_id}")
                    identifiers.add(call_id)
                    function = call["function"]
                    name = function["name"]
                    arguments = function["arguments"]
                    try:
                        args = json.loads(arguments) if isinstance(arguments, str) else arguments
                        command = tool_command(name, args)
                    except (ValueError, AttributeError):
                        name, command = "error_format", str(arguments)
                    size = len(arguments if isinstance(arguments, str) else json.dumps(arguments, default=str))
                    events.append((call_id, name, command, size + (len(text) if position == 0 else 0)))
            elif role == "tool":
                call_id = row["tool_call_id"]
                if call_id in replies:
                    raise ValueError(f"duplicate OpenHands tool reply: {call_id}")
                replies[call_id] = message_text(row.get("content"))
            elif role == "user" and text_pending:
                text = message_text(row.get("content"))
                match = TEXT_REPLY.match(text)
                if match:
                    matched = next((i for i, (name, _) in enumerate(text_pending) if name == match[1]), None)
                    if matched is not None:
                        _, call_id = text_pending.pop(matched)
                        replies[call_id] = text[match.end():]
    steps = []
    for call_id, name, command, size in events:
        if call_id is not None and call_id not in replies and name not in {"finish", "submit"}:
            raise ValueError(f"missing OpenHands reply for {call_id} ({name})")
        steps.append(Step(name, command, replies.get(call_id, ""), size))
    return Trajectory(steps)
