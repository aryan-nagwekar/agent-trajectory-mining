"""Map observable actions and observations to the shared vocabulary."""

import re
import shlex
from pathlib import PurePosixPath

from atm.parsers.base import Step

SEARCH = {"grep", "rg", "egrep", "fgrep", "find", "fd", "locate", "search_dir", "search_file", "find_file"}
VIEW = {"cat", "open", "view", "head", "tail", "nl", "less", "more", "scroll_up", "scroll_down", "goto", "wc", "pwd", "stat"}
EDIT = {"edit", "str_replace", "insert", "apply_patch", "patch", "rm", "mv", "chmod", "truncate"}
CREATE = {"create", "touch", "mkdir", "cp", "tee"}
REVERT = {"checkout", "restore", "stash", "reset", "revert"}
TEST = {"pytest", "py.test", "tox", "nosetests", "unittest"}
TOOLS = {
    "think": "think", "finish": "submit", "submit": "submit", "error_format": "error_format",
    "read_file": "view", "write_file": "edit", "edit_file": "edit", "run_ipython_cell": "run_script",
    "search": "search", "browser": "search",
}
EDITOR = {"undo_edit": "revert", "write": "edit"}
SHELL_SEPARATOR = {"&&", "||", ";", "|", "\n"}
EXIT_CODE = re.compile(r"\b(?:exit(?:ed with)? (?:status|code)|return code)[: ]+(-?\d+)\b", re.I)
TEST_FAIL = re.compile(r"(?im)^\s*(?:FAILED\b|FAIL:|ERROR:|E\s+.*Error\b)|\b[1-9]\d* (?:failed|errors?)\b")
TEST_PASS = re.compile(r"(?im)^\s*OK\s*$|\b[1-9]\d* passed\b")
OBS_ERROR = re.compile(r"(?im)^\s*(?:Traceback \(most recent call last\):|"
                       r"(?:SyntaxError|IndentationError|ValueError|FileNotFoundError):|"
                       r"ERROR:|Error:|Invalid command\b|Invalid arguments\b|"
                       r"File not found\b|The file .* does not exist\b)")


def command_words(command: str) -> list[str]:
    """Read shell syntax lexically and preserve control and redirect operators."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>\n")
    lexer.whitespace_split = True
    lexer.whitespace = " \t\r"
    try:
        return [token for word in lexer
                for token in (re.findall(r"[^\n]+|\n", word)
                              if re.fullmatch(r"[|&;<>\n]+", word) else [word])]
    except ValueError:
        # Broken quoting is still observable. The shell response says whether it errored.
        return command.split()


def shell_action(command: str) -> str:
    """Classify the leading operation, passing over shell setup commands."""
    words = command_words(command)
    while words:
        first = words[0]
        executable = PurePosixPath(first).name
        if first == "\n" or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", first):
            words = words[1:]
        elif executable in {"cd", "export", "set", "source", ".", "unset"}:
            end = next((i for i, word in enumerate(words) if word in SHELL_SEPARATOR), len(words))
            if end == len(words):
                return "list" if executable == "cd" else "other_bash"
            words = words[end + 1:]
        elif executable in {"timeout", "gtimeout"} and len(words) > 2 and re.fullmatch(
                r"\d+(?:\.\d+)?[smhd]?", words[1]):
            words = words[2:]
        elif executable in {"sudo", "env", "time", "command"}:
            words = words[1:]
        else:
            break
    if not words:
        return "other_bash"

    name = PurePosixPath(words[0]).name
    end = next((i for i, word in enumerate(words) if word in SHELL_SEPARATOR), len(words))
    args = words[1:end]
    operation = args[0] if args else ""
    if name in {"bash", "sh", "zsh"} and len(args) >= 2 and re.fullmatch(r"-[ilx]*c", operation):
        return shell_action(args[1])
    if name == "COMPLETE_TASK" or (name in {"echo", "printf"} and {"COMPLETE_TASK", "MICRO_SWE_AGENT_FINAL_OUTPUT", "MINI_SWE_AGENT_FINAL_OUTPUT"} & set(args)):
        return "submit"
    if name in {"str_replace_editor", "file_editor"}:
        return EDITOR[operation] if operation in EDITOR else shell_action(operation)
    if ">>" in args:
        return "edit"
    if ">" in args or ">|" in args:
        target = args[args.index(">") + 1:] if ">" in args else args[args.index(">|") + 1:]
        if target and target[0] != "/dev/null" and name in {"cat", "echo", "printf", "sed", "tee"}:
            return "create"
    if name in SEARCH or name == "ls" and any(arg == "--recursive" or
            arg.startswith("-") and not arg.startswith("--") and "R" in arg for arg in args):
        return "search"
    if name in {"ls", "tree", "dir"}:
        return "list"
    if name == "sed" and any(arg == "--in-place" or arg.startswith(("-i", "--in-place=")) for arg in args):
        return "edit"
    if name in VIEW or name in {"sed", "awk", "sort", "cut", "uniq", "diff", "realpath", "which"}:
        return "view"
    if name in EDIT:
        return "edit"
    if name in CREATE:
        return "create"
    if name == "git":
        subcommand = next((arg for arg in args if not arg.startswith("-")), "")
        return "revert" if subcommand in REVERT and (subcommand != "checkout" or "--" in args) else "vcs"
    if name.startswith(("pip", "conda", "mamba")) or name in {"apt", "apt-get", "uv", "poetry"}:
        return "install"
    if name in {"django-admin", "django-admin.py"} and operation in {"startproject", "startapp"}:
        return "create"
    if (name in TEST or name in {"runtests.py", "run_tests.py"}
            or name in {"make", "django-admin", "django-admin.py"} and operation in {"test", "tests", "check"}
            or words[0] in {"bin/test", "./bin/test"}):
        return "run_script" if {"--version", "--help", "-h"} & set(args) else "run_test"
    if name.startswith("python") or name in {"pypy", "pypy3"}:
        if len(args) >= 2 and args[0] == "-m" and args[1] == "pip":
            return "install"
        if len(args) >= 2 and args[0] == "-m" and args[1] in TEST:
            return "run_script" if {"--version", "--help", "-h"} & set(args) else "run_test"
        if (any(PurePosixPath(arg).name in {"runtests.py", "run_tests.py"} for arg in args[:2])
                or len(args) >= 2 and args[0].endswith("manage.py") and args[1] == "test"):
            return "run_test"
        inline = command if "<<" in args or "<<-" in args else " ".join(args)
        if re.search(r"\b(?:write_text|write_bytes|writelines)\s*\(", inline):
            return "edit"
        if re.search(r"\bopen\([^,\n]+,\s*['\"][wax][^'\"]*['\"]", inline):
            return "edit"
        return "run_script"
    if (name in {"bash", "sh", "zsh", "ruby", "node", "perl", "make", "xargs",
                 "pylint", "pyreverse", "cython", "sphinx-build", "django-admin"}
            or words[0].startswith("./")):
        return "run_script"
    return TOOLS.get(name, "other_bash")


def classify(step: Step) -> tuple[str, str]:
    """Return the action and coarse observed outcome."""
    raw_type = step.raw_action_type
    action = TOOLS.get(raw_type)
    if action is None:
        command = step.raw_command
        if raw_type in {"str_replace_editor", "file_editor"} and not command.startswith(raw_type + " "):
            command = f"{raw_type} {command}"
        action = shell_action(command)
    status = step.exit_code
    if status is None:
        seen = list(EXIT_CODE.finditer(step.observation))
        if seen:
            status = int(seen[-1][1])
    if action == "error_format":
        return action, "error"
    if action == "run_test":
        if status not in {None, 0} or TEST_FAIL.search(step.observation):
            return action, "fail"
        if status == 0 or TEST_PASS.search(step.observation):
            return action, "pass"
    if status not in {None, 0} or OBS_ERROR.search(step.observation):
        return action, "error"
    return action, "ok" if status == 0 else "none"
