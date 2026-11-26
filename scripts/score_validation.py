"""Score human action and outcome labels on the Phase 2 validation sample."""

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, nargs="?", default=Path("data/interim/validation_sample.csv"))
    args = parser.parse_args()
    with args.sample.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for field in ("action", "outcome"):
        labeled = [row for row in rows if row[f"human_{field}"].strip()]
        matches = sum(row[f"human_{field}"].strip().lower() == row[f"predicted_{field}"] for row in labeled)
        print(f"{field}: {matches}/{len(labeled)} correct ({len(labeled)}/{len(rows)} labels filled)")
        if labeled:
            print(f"{field} agreement: {matches / len(labeled):.2%}")
    completed = [row for row in rows if row["human_action"].strip() and row["human_outcome"].strip()]
    if completed:
        joint = sum(all(row[f"human_{field}"].strip().lower() == row[f"predicted_{field}"]
                        for field in ("action", "outcome")) for row in completed)
        print(f"Joint agreement: {joint}/{len(completed)} ({joint / len(completed):.2%})")
    else:
        print("Joint agreement requires human labels.")


if __name__ == "__main__":
    main()
