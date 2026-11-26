"""Reproduce the pipeline with content-based stage checks and an isolated smoke cohort."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from atm.download import EXPERIMENTS, fetch, inventory

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
BASE_CONFIG = PROJECT / "configs/expanded.yaml"


def environment(work: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    env["MPLCONFIGDIR"] = str(work / "results/.mpl")
    torch_lib = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/torch/lib"
    prior = env.get("DYLD_LIBRARY_PATH")
    env["DYLD_LIBRARY_PATH"] = str(torch_lib) + (f":{prior}" if prior else "")
    return env


def execute(work: Path, args: list[str]) -> None:
    subprocess.run(args, cwd=work, env=environment(work), check=True)


def quick_cohort() -> tuple[Path, Path]:
    """Use a reproducible shared task sample in a fresh derived-data workspace."""
    config = yaml.safe_load(BASE_CONFIG.read_text())
    work = PROJECT / "results/quick"
    work.mkdir(parents=True, exist_ok=True)
    if (work / "data/processed/runs.parquet").exists():
        # Another invocation preserves the existing smoke run; new runs use a new numbered directory.
        i = 2
        while (PROJECT / f"results/quick-{i}").exists():
            i += 1
        work = PROJECT / f"results/quick-{i}"
        work.mkdir()
    raw = PROJECT / "data/raw"
    cache = work / "data/raw"
    roster = raw / config["verified_roster"]
    if not roster.exists():
        sub, rel = config["verified_roster"].split("/", 1)
        fetch(f"{EXPERIMENTS}/{config['experiments_commit']}/evaluation/verified/{sub}/{rel}",
              cache / config["verified_roster"], False)
        roster = cache / config["verified_roster"]
    eligible = set(json.loads(roster.read_text()))
    listings = {}
    for source in config["submissions"]:
        name = source["submission"]
        cached = raw / name / "inventory.json"
        entries = json.loads(cached.read_text()) if cached.exists() else inventory(source["trajectory_prefix"])
        listings[name] = entries
        eligible &= {key for key, (_, size) in entries.items() if size > 0}
    tasks = sorted(np.random.default_rng(config["seed"]).choice(sorted(eligible), 50, replace=False))

    def link(source: Path, destination: Path) -> None:
        if not source.is_file() or destination.exists():
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.hardlink_to(source)
        except OSError:
            shutil.copy2(source, destination)

    for source in config["submissions"]:
        name = source["submission"]
        entries = listings[name]
        source["trajectories"] = {str(task): entries[task][0] for task in tasks}
        for filename in ("metadata.yaml", source["results_path"], *[f"trajs/{Path(entries[t][0]).name}" for t in tasks]):
            link(raw / name / filename, cache / name / filename)
    config["cohort"] = "quick"
    config["audit"]["min_runs"] = 100
    config["audit"]["max_test_none_share"] = .50
    config["clustering"].update(min_size=4, max_component=60, threshold=.40, bands=42, rows=3,
                                benchmark_sizes=[50, 100, 200], benchmark_repeats=1)
    config["archetype_sweep"].update(selection_min_coverage=.02, selection_max_largest_share=.99)
    config["prediction"].update(top_patterns=20, top_ngrams=64, gru_epochs=5, gru_patience=2,
                                lightgbm_candidates=[[12,.08,55],[18,.05,70]], lightgbm_patience=8,
                                shuffled_trees=35, shuffled_repeats=2, shuffled_acceptance=[0.0, 1.0])
    config["window_benchmark"]["sizes"] = [500, 1000, 2000]
    config["presentation"]["bootstrap_replicates"] = 60
    config["presentation"]["diagnostic_run"] = True
    path = work / "quick.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return work, path


def digest(paths: list[Path], seed: str) -> str:
    result = hashlib.sha256(seed.encode())
    for item in paths:
        sources = sorted(item.rglob("*")) if item.is_dir() else [item]
        for path in sources:
            if path.is_file() and path.suffix in {".py", ".yaml", ".tex", ".sty", ".bib", ".cpp", ".md"}:
                result.update(path.read_bytes())
    return result.hexdigest()


def compile_report(work: Path) -> None:
    source = work / "report"
    destination = work / "output/pdf"
    destination.mkdir(parents=True, exist_ok=True)
    tectonic = shutil.which("tectonic")
    if tectonic:
        execute(source, [tectonic, "--keep-logs", "--outdir", str(destination), "final_report.tex"])
        return
    latexmk = shutil.which("latexmk")
    if latexmk:
        execute(source, [latexmk, "-pdf", "-halt-on-error", f"-outdir={destination}", "final_report.tex"])
        return
    raise SystemExit("Install Tectonic or latexmk with a TeX distribution to compile the official preprint.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute stage outputs. Public raw cache is reused.")
    args = parser.parse_args()
    if args.report_only:
        compile_report(PROJECT)
        return
    start = time.perf_counter()
    work, config = quick_cohort() if args.quick else (PROJECT, BASE_CONFIG)
    cached = work / "results/.pipeline"
    cached.mkdir(parents=True, exist_ok=True)
    upstream = ""
    atm = SRC / "atm"
    cmd = [sys.executable]
    executed = []

    def stage(name: str, steps: list[list[str]], files: list[str], sources: list[Path]) -> None:
        nonlocal upstream
        fingerprint = digest([config, *sources], upstream + json.dumps(steps))
        marker = cached / f"{name}.sha256"
        current = all((work / path).is_file() for path in files)
        if current and marker.exists() and marker.read_text() == fingerprint and not args.force:
            print(f"{name}: verified cache", flush=True)
        else:
            print(f"{name}: running", flush=True)
            for step in steps:
                force_option = ["--force"] if args.force and name in {"normalize", "mine"} else []
                execute(work, cmd + step + force_option)
            if not all((work / path).is_file() for path in files):
                raise RuntimeError(f"{name} did not produce its required outputs")
            marker.write_text(fingerprint)
            executed.append(name)
        upstream = fingerprint

    def mod(name: str, *positionals: str) -> list[str]:
        return ["-m", "atm." + name, *positionals, "--config", str(config)]

    print(f"Project output · {work}", flush=True)
    execute(work, cmd + mod("download"))
    raw = work / "data/raw"
    raw_stats = [(str(f.relative_to(raw)), f.stat().st_size, f.stat().st_mtime_ns)
                 for f in sorted(raw.glob("*/trajs/*")) if f.is_file()]
    upstream = hashlib.sha256(repr(raw_stats).encode()).hexdigest()
    stage("normalize", [mod("normalize")], ["data/processed/runs.parquet", "data/processed/steps.parquet",
          "data/processed/collapsed_sequences.parquet", "results/phase2.json", "data/interim/validation_sample.csv"],
          [atm / "normalize.py", atm / "parsers", atm / "vocab.py"])
    stage("splits", [mod("audit"), mod("splits")], ["results/submission_audit.csv", "data/processed/splits.parquet",
          "data/processed/difficulty.parquet"], [atm / "splits.py", atm / "audit.py", atm / "difficulty.py"])
    stage("mine", [mod("mining.discriminative")], ["results/patterns_all.csv", "results/patterns_robust.csv",
          "results/patterns_primary.csv", "results/patterns_top.csv", "results/patterns_by_framework.csv", "results/length_confound.csv", "results/phase4.json"],
          [atm / "mining"])
    stage("cluster", [mod("clustering.archetypes")], ["results/phase5.json", "results/lsh_scaling.csv",
          "figures/lsh_scaling_failed_runs.png", "figures/lsh_scaling_failed_runs.pdf"],
          [atm / "clustering/archetypes.py", atm / "clustering/minhash_lsh.py", atm / "clustering/shingles.py"])
    stage("predict", [mod("predict")], ["results/phase6.json", "results/models/fold_5_t40/tabular.joblib",
          "results/ablation.csv", "data/processed/prefix_predictions.parquet"],
          [atm / "predict.py", atm / "models", atm / "features.py"])
    stage("sweep-and-window", [mod("clustering.sweep"), mod("clustering.window_benchmark")],
          ["results/archetype_sweep.csv", "results/archetypes.csv", "results/archetypes_for_labeling.md",
           "results/lsh_window_scaling.csv", "figures/lsh_scaling.png", "figures/lsh_scaling.pdf"],
          [atm / "clustering/sweep.py", atm / "clustering/window_benchmark.py", atm / "clustering/exact_windows.cpp"])
    stage("refit", [mod("revision", "refit"), ["-m","atm.cohorts"]],
          ["data/processed/prefix_predictions_revised.parquet", "results/archetype_ablation_refit.csv",
           "results/models_revised/fold_5_t40/tabular.joblib", "results/fixed_cohort_metrics.csv", "results/deployment_metrics.csv"],
          [atm / "revision.py", atm / "cohorts.py"])
    stage("calibrate", [mod("calibration", "grouped")],
          ["results/early_stop.csv", "results/early_stop_thresholds.csv"], [atm / "calibration.py"])
    stage("transfer", [mod("revision", "transfer"), mod("calibration", "transfer"), mod("transfer_patterns")],
          ["results/transfer_metrics.csv", "results/transfer_policy.csv", "results/framework_pattern_overlap.csv",
           "data/processed/transfer_predictions.parquet"], [atm / "revision.py", atm / "calibration.py", atm / "transfer_patterns.py"])

    if not args.analysis_only:
        if args.quick:
            (work / "report").mkdir(exist_ok=True)
            for name in ("final_report.tex", "neurips_2024.sty", "references.bib"):
                shutil.copy2(PROJECT / "report" / name, work / "report" / name)
            for name in ("README.md", "DEVIATIONS.md"):
                shutil.copy2(PROJECT / name, work / name)
        stage("deliverables", [
            mod("final_results"), ["-m","atm.figures"], [str(PROJECT / "scripts/09_writeups.py")],
            [str(PROJECT / "scripts/04_report.py")], [str(PROJECT / "scripts/06_report.py")],
            [str(PROJECT / "scripts/08_report.py")],
        ], ["results/metrics.json", "results/macros.tex", "results/tables/transfer.tex", "figures/11_replay.png",
            "report/phase8_report.md", "slides/slides_outline.md"],
            [atm / "final_results.py", atm / "figures.py", atm / "demo" / "replay.py", PROJECT / "scripts/09_writeups.py"])
        compile_report(work)
    elapsed = time.perf_counter() - start
    print(f"Completed in {elapsed:.1f} seconds · {work}", flush=True)
    evidence = {"seconds": elapsed, "workspace": str(work), "executed": executed, "quick": args.quick,
                "analysis_only": args.analysis_only}
    target = PROJECT / ("results/quick_run.json" if args.quick else
                        "results/analysis_run.json" if args.analysis_only else "results/all_run.json")
    target.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    main()
