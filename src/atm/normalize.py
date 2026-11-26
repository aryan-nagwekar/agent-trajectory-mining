"""Normalize raw traces, verify coverage, and export Phase 2 evidence."""

import argparse
import csv
import hashlib
import itertools
import json
import logging
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import ijson
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from atm.parsers import PARSERS
from atm.vocab import classify

RUNS = pa.schema([
    ("run_id", pa.string()), ("submission", pa.string()), ("framework", pa.string()),
    ("base_model", pa.string()), ("instance_id", pa.string()), ("repo", pa.string()),
    ("failed", pa.int8()), ("n_steps", pa.int32()), ("total_tokens", pa.int64()),
    ("total_chars", pa.int64()),
])
STEPS = pa.schema([
    ("run_id", pa.string()), ("step_idx", pa.int32()), ("raw_action_type", pa.string()),
    ("raw_command", pa.string()), ("action", pa.string()), ("outcome", pa.string()),
    ("token", pa.string()), ("step_tokens", pa.int64()), ("step_chars", pa.int64()),
])
COLLAPSED = pa.schema([("run_id", pa.string()), ("tokens", pa.list_(pa.string()))])
VALIDATION_FIELDS = [
    "raw_command", "observation_snippet", "predicted_action", "predicted_outcome",
    "human_action", "human_outcome",
]


def collapse(tokens: list[str]) -> list[str]:
    """Preserve singletons and bucket longer runs of identical tokens."""
    result = []
    for token, group in itertools.groupby(tokens):
        count = sum(1 for _ in group)
        suffix = "×3+" if count >= 3 else "×2" if count == 2 else ""
        result.append(token + suffix)
    return result


def labels_for(path: Path, roster: set[str]) -> tuple[dict[str, bool], dict[str, str]]:
    """Resolve benchmark labels and retain each declared exclusion reason."""
    data = json.loads(path.read_bytes())
    if "resolved" not in data:
        if not set(data) <= roster or any(type(row["resolved"]) is not bool for row in data.values()):
            raise ValueError(f"invalid per-instance evaluation results in {path}")
        return {task: row["resolved"] for task, row in data.items()}, {
            task: "no official task outcome" for task in roster - data.keys()
        }
    resolved = set(data["resolved"])
    excluded = {
        task: field for field in ("no_generation", "no_logs") for task in data.get(field, [])
    }
    if (len(resolved) != len(data["resolved"]) or resolved & excluded.keys()
            or not (resolved | excluded.keys()) <= roster):
        raise ValueError(f"invalid resolved or exclusion IDs in {path}")
    return {task: task in resolved for task in roster - excluded.keys()}, excluded


def task_files(path: Path) -> dict[str, Path]:
    """Index the native filename of each cached trajectory."""
    files = {}
    for file in sorted(path.iterdir()):
        suffix = next((suffix for suffix in (".traj.json", ".traj", ".json")
                       if file.name.endswith(suffix)), None)
        if suffix is None:
            continue
        task = file.name.removesuffix(suffix)
        if task in files:
            raise ValueError(f"multiple trajectory files for {task} in {path}")
        files[task] = file
    return files


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None, *, mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def spreadsheet_text(value: str) -> str:
    """Keep command and observation snippets literal when opened in a spreadsheet."""
    return "'" + value if value.lstrip().startswith(("=", "+", "-", "@")) else value


class ValidationSample:
    """Seeded reservoir sampling with the same number of steps per framework."""

    def __init__(self, seed: int, size: int, frameworks: list[str]):
        if not frameworks or size % len(frameworks):
            raise ValueError("validation size must divide evenly across frameworks")
        self.size = size // len(frameworks)
        self.random = random.Random(seed)
        self.counts = Counter()
        self.rows = {framework: [] for framework in frameworks}

    def add(self, framework: str, command: str, observation: str, action: str, outcome: str) -> None:
        candidates = self.rows[framework]
        self.counts[framework] += 1
        index = self.random.randrange(self.counts[framework]) if len(candidates) == self.size else len(candidates)
        if index >= self.size:
            return
        row = dict(zip(VALIDATION_FIELDS, (
            spreadsheet_text(command[:500]), spreadsheet_text(observation[:240]),
            action, outcome, "", "",
        )))
        if index == len(candidates):
            candidates.append(row)
        else:
            candidates[index] = row

    def write(self, path: Path, *, preserve: bool = False) -> int:
        combined = [row for rows in self.rows.values() for row in rows]
        self.random.shuffle(combined)
        try:
            write_csv(path, combined, VALIDATION_FIELDS, mode="x" if preserve else "w")
        except FileExistsError:
            logging.info("Preserving the existing annotation sample at %s", path)
            with path.open(encoding="utf-8", newline="") as stream:
                return sum(1 for _ in csv.DictReader(stream))
        return len(combined)


def normalize(config: dict, cohort: str, limit: int, output: Path, reports: Path) -> bool:
    raw = Path("data/raw")
    sources = config["submissions"]
    roster = set(json.loads((raw / config["verified_roster"]).read_bytes()))
    if len(roster) != config["verified_task_count"]:
        raise ValueError("the Verified roster has unexpected cardinality")
    thresholds = config["phase2"]
    frameworks = sorted({source["framework"] for source in sources})
    if cohort == "expanded" and (len(frameworks) != 3 or not 10 <= len(sources) <= 15):
        raise ValueError("the expanded cohort requires 10–15 submissions across three frameworks")
    sampler = ValidationSample(config["seed"], thresholds["validation_size"], frameworks)
    checks = []
    skipped = []
    other = Counter()
    aggregates = defaultdict(lambda: {"runs": 0, "steps": 0})
    output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=output) as temporary:
        staging = Path(temporary)
        with (pq.ParquetWriter(staging / "runs.parquet", RUNS) as run_writer,
              pq.ParquetWriter(staging / "steps.parquet", STEPS) as step_writer,
              pq.ParquetWriter(staging / "collapsed_sequences.parquet", COLLAPSED) as sequence_writer):
            for source in sources:
                framework = source["framework"]
                submission = source["submission"]
                directory = raw / submission
                labels, declared_excluded = labels_for(directory / source["results_path"], roster)
                files = task_files(directory / "trajs")
                if source.get("trajectories"):
                    work = list(source["trajectories"])
                else:
                    work = sorted(roster)
                if limit:
                    work = work[:limit]
                runs = []
                step_count = parse_errors = missing_count = other_count = 0
                for task in work:
                    if task not in labels or task not in files:
                        reason = declared_excluded.get(task, "missing trajectory")
                        logging.warning("%s %s: %s", submission, task, reason)
                        skipped.append({"submission": submission, "instance_id": task, "reason": reason})
                        missing_count += 1
                        continue
                    try:
                        trajectory = PARSERS[framework](files[task], task)
                        if not trajectory.steps:
                            raise ValueError("empty trajectory")
                    except (OSError, ValueError, KeyError, TypeError, ijson.JSONError) as exc:
                        logging.warning("%s %s: %s", submission, task, exc)
                        skipped.append({"submission": submission, "instance_id": task,
                                        "reason": f"{type(exc).__name__}: {exc}"})
                        parse_errors += 1
                        continue

                    run_id = f"{submission}::{task}"
                    failed = int(not labels[task])
                    step_rows = []
                    for idx, step in enumerate(trajectory.steps):
                        action, outcome = classify(step)
                        if action == "other_bash":
                            other[step.raw_command[:500]] += 1
                            other_count += 1
                        sampler.add(framework, step.raw_command, step.observation, action, outcome)
                        step_rows.append({
                            "run_id": run_id, "step_idx": idx, "raw_action_type": step.raw_action_type,
                            "raw_command": step.raw_command[:500], "action": action, "outcome": outcome,
                            "token": action if outcome == "none" else f"{action}:{outcome}",
                            "step_tokens": step.step_tokens, "step_chars": step.model_chars + len(step.observation),
                        })
                    run = {
                        "run_id": run_id, "submission": submission, "framework": framework,
                        "base_model": source.get("base_model", config.get("base_model")), "instance_id": task,
                        "repo": task.rsplit("-", 1)[0].replace("__", "/"), "failed": failed,
                        "n_steps": len(step_rows), "total_tokens": trajectory.total_tokens,
                        "total_chars": sum(row["step_chars"] for row in step_rows),
                    }
                    runs.append(run)
                    step_count += len(step_rows)
                    group = aggregates[framework, failed]
                    group["runs"] += 1
                    group["steps"] += len(step_rows)
                    step_writer.write_table(pa.Table.from_pylist(step_rows, schema=STEPS))
                    sequence_writer.write_table(pa.Table.from_pylist([{
                        "run_id": run_id, "tokens": collapse([row["token"] for row in step_rows]),
                    }], schema=COLLAPSED))
                if runs:
                    run_writer.write_table(pa.Table.from_pylist(runs, schema=RUNS))
                official_rate = sum(labels.values()) / len(roster)
                observed_rate = sum(run["failed"] for run in runs) / len(runs) if runs else None
                difference = abs(observed_rate - (1 - official_rate)) if observed_rate is not None else None
                parse_rate = parse_errors / max(len(work) - missing_count, 1)
                other_rate = other_count / max(step_count, 1)
                passed = (bool(runs) and parse_rate < thresholds["max_parse_failure"]
                          and other_rate < thresholds["max_other_bash"])
                if cohort == "pilot" and not limit:
                    failed_count = sum(run["failed"] for run in runs)
                    passed &= (len(runs) == len(work) and min(failed_count, len(runs) - failed_count) >= 3)
                if cohort == "expanded" and not limit:
                    passed &= difference is not None and difference <= thresholds["resolve_rate_tolerance"]
                checks.append({
                    "submission": submission, "framework": framework, "base_model": source.get("base_model", config.get("base_model")),
                    "expected": len(work), "runs": len(runs), "steps": step_count, "missing_or_ineligible": missing_count,
                    "parse_failures": parse_errors, "parse_failure_rate": parse_rate, "other_bash": other_count,
                    "other_bash_rate": other_rate, "resolved": sum(run["failed"] == 0 for run in runs),
                    "unresolved": sum(run["failed"] == 1 for run in runs),
                    "official_resolved": sum(labels.values()), "official_denominator": len(roster),
                    "observed_failure_rate": observed_rate, "official_resolve_rate": official_rate,
                    "absolute_failure_rate_difference": difference, "passed": passed,
                })
                logging.info("%s: %d/%d runs, %d steps, %.2f%% parse failures, %.2f%% other_bash",
                             submission, len(runs), len(work), step_count, parse_rate * 100, other_rate * 100)
        passed = all(row["passed"] for row in checks)
        if passed:
            for file in staging.iterdir():
                file.replace(output / file.name)

    write_csv(reports / "normalization_checks.csv", checks)
    write_csv(reports / "normalization_skips.csv", skipped, ["submission", "instance_id", "reason"])
    write_csv(reports / "data_summary.csv", [{
        "framework": framework, "failed": failed, **group,
        "mean_length": group["steps"] / group["runs"],
    } for (framework, failed), group in sorted(aggregates.items())],
        ["framework", "failed", "runs", "steps", "mean_length"])
    other_rows = sorted(other.items(), key=lambda item: (-item[1], item[0]))[:30]
    write_csv(reports / "other_bash_top30.csv", [
        {"raw_command": spreadsheet_text(command), "count": count} for command, count in other_rows
    ], ["raw_command", "count"])
    for command, count in other_rows:
        logging.info("other_bash ×%d: %r", count, command)
    validation_path = Path("data/interim") / ("pilot_validation_sample.csv" if cohort == "pilot" else "validation_sample.csv")
    sample_size = sampler.write(validation_path, preserve=cohort == "expanded") if passed else 0
    summary = {
        "cohort": cohort, "status": "passed" if passed else "failed", "submissions": len(checks),
        "runs": sum(row["runs"] for row in checks), "steps": sum(row["steps"] for row in checks),
        "validation_rows": sample_size,
        "validation_rows_by_framework": {name: len(rows) for name, rows in sampler.rows.items()} if passed else {},
        "experiments_commit": config["experiments_commit"],
        "config": config, "checks": checks,
    }
    (reports / "phase2.json").write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("%s: %s, %d runs, %d steps, %d validation rows", cohort, summary["status"],
                 summary["runs"], summary["steps"], sample_size)
    return passed


def config_fingerprint(args: argparse.Namespace, config: dict) -> str:
    """Hash the config, parser code, and cached inputs used by normalization."""
    key = hashlib.sha256(args.config.read_bytes() + str(args.limit).encode())
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        key.update(path.read_bytes())
    raw = Path("data/raw")
    for source in config["submissions"]:
        directory = raw / source["submission"]
        key.update((directory / source["results_path"]).read_bytes())
        for path in sorted((directory / "trajs").iterdir()):
            stat = path.stat()
            key.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    key.update((raw / config["verified_roster"]).read_bytes())
    return key.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path("configs/default.yaml"), type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Maximum candidates per framework submission")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    config = yaml.safe_load(args.config.read_text())
    cohort = config.get("cohort", "pilot")
    if args.limit:
        cohort = "debug"
    reports = Path("results") / ("pilot" if cohort == "pilot" else "debug" if cohort == "debug" else "")
    output = Path("data/processed") / ("pilot" if cohort == "pilot" else "debug" if cohort == "debug" else "")
    log_dir = Path("results/logs")
    for directory in (reports, log_dir, Path("data/interim")):
        directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", handlers=[
        logging.StreamHandler(), logging.FileHandler(log_dir / f"normalize_{cohort}.log"),
    ])
    fingerprint = config_fingerprint(args, config)
    marker = output / ".normalize-cache"
    ready = all((output / file).is_file() for file in ("steps.parquet", "runs.parquet", "collapsed_sequences.parquet"))
    ready &= all((reports / file).is_file() for file in (
        "phase2.json", "data_summary.csv", "normalization_checks.csv", "normalization_skips.csv",
        "other_bash_top30.csv",
    ))
    if ready and marker.is_file() and marker.read_text() == fingerprint and not args.force:
        logging.info("Normalized %s outputs are cached", cohort)
        return
    if not normalize(config, cohort, args.limit, output, reports):
        raise SystemExit("Phase 2 acceptance checks failed. See normalization_checks.csv.")
    marker.write_text(fingerprint)


if __name__ == "__main__":
    main()
