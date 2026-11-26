"""Fetch pinned evaluation files and public SWE-bench trajectory artifacts."""

import argparse
import json
import logging
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree

import requests
import yaml

EXPERIMENTS = "https://raw.githubusercontent.com/SWE-bench/experiments"
TRAJECTORIES = "https://swe-bench-submissions.s3.amazonaws.com"
LOCAL = threading.local()


def session() -> requests.Session:
    """Reuse a connection pool within each download thread."""
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers["Accept-Encoding"] = "identity"
    return LOCAL.session


def fetch(url: str, destination: Path, force: bool, expected_size: int | None = None) -> None:
    """Stream a public file into the cache and publish it after a clean transfer."""
    if destination.is_file() and not force and (
        expected_size is None or destination.stat().st_size == expected_size
    ):
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        temp_path = None
        try:
            with session().get(url, timeout=(15, 90), stream=True) as response:
                response.raise_for_status()
                header = (response.headers.get("Content-Length")
                          if response.headers.get("Content-Encoding", "identity") == "identity" else None)
                size = expected_size if expected_size is not None else int(header) if header else None
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
                    temp_path = Path(temp.name)
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        temp.write(chunk)
            downloaded = temp_path.stat().st_size
            if not downloaded or (size is not None and downloaded != size):
                raise OSError(f"incomplete download ({downloaded} bytes, expected {size})")
            temp_path.replace(destination)
            return
        except (OSError, requests.RequestException) as exc:
            if attempt == 2 or isinstance(exc, requests.HTTPError) and exc.response.status_code in {403, 404}:
                raise
            logging.warning("Retrying %s after %s", destination.name, exc)
            time.sleep(0.5 * (attempt + 1))
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def inventory(prefix: str) -> dict[str, tuple[str, int]]:
    """List the versioned submission prefix, including all S3 pages."""
    query = {"list-type": "2", "prefix": prefix.rstrip("/") + "/"}
    files = {}
    while True:
        response = session().get(TRAJECTORIES, params=query, timeout=(15, 60))
        response.raise_for_status()
        page = ElementTree.fromstring(response.content)
        for entry in page.findall(".//{*}Contents"):
            key = entry.findtext("{*}Key")
            suffix = next((suffix for suffix in (".traj.json", ".traj", ".json") if key.endswith(suffix)), None)
            if suffix is None:
                continue
            task = Path(key).name.removesuffix(suffix)
            if "__" not in task or not task.rsplit("-", 1)[-1].isdigit():
                continue
            if task in files:
                raise ValueError(f"duplicate S3 trajectory for {task} under {prefix}")
            files[task] = (key, int(entry.findtext("{*}Size")))
        continuation = page.findtext("{*}NextContinuationToken")
        if continuation is None:
            return files
        query["continuation-token"] = continuation


def download_source(source: dict, base: str, args: argparse.Namespace, workers: int) -> bool:
    submission = source["submission"]
    target = Path("data/raw") / submission
    for filename in ("metadata.yaml", source["results_path"]):
        fetch(f"{base}/{submission}/{filename}", target / filename, args.force)
    if "trajectories" in source:
        sample = [(key, None) for key in source["trajectories"].values()]
    else:
        index = target / "inventory.json"
        if index.is_file() and not args.force:
            contents = json.loads(index.read_text())
        else:
            contents = inventory(source["trajectory_prefix"])
            if not contents:
                raise ValueError(f"no public trajectory objects for {submission}")
            temporary = index.with_suffix(".tmp")
            temporary.write_text(json.dumps(contents, sort_keys=True) + "\n")
            temporary.replace(index)
        sample = [contents[task] for task in sorted(contents)]
    if args.limit:
        sample = sample[:args.limit]

    def download(item: tuple[str, int | None]) -> bool:
        key, size = item
        try:
            fetch(f"{TRAJECTORIES}/{quote(key)}", target / "trajs" / Path(key).name, args.force, size)
        except (OSError, requests.RequestException) as exc:
            logging.error("%s: %s: %s", submission, key, exc)
            return False
        return True

    with ThreadPoolExecutor(max_workers=workers) as executor:
        successes = 0
        for done, ok in enumerate(executor.map(download, sample), start=1):
            successes += int(ok)
            if done % 100 == 0:
                logging.info("%s: %d/%d trajectories checked", submission, done, len(sample))
    logging.info("%s: %d/%d trajectories ready", submission, successes, len(sample))
    return successes == len(sample)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--limit", type=int, default=0, help="Maximum trajectories per submission, or zero for all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")

    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "data.log")],
    )
    config = yaml.safe_load(args.config.read_text())
    base = f"{EXPERIMENTS}/{config['experiments_commit']}/evaluation/verified"
    ready = [download_source(source, base, args, config.get("download_workers", 4))
             for source in config["submissions"]]
    if not all(ready):
        raise SystemExit("Some trajectory downloads failed. See results/logs/data.log.")


if __name__ == "__main__":
    main()
