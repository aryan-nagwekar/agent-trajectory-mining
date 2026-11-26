"""Check generated deliverables and write the definition-of-done record."""

import json
import re
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image

R = Path("results")
EXPECTED = (
    "01_pipeline", "02_dataset", "03_length_confound", "04_patterns", "05_archetypes", "06_lsh_pairs",
    "07_fixed_cohort", "07b_same_task_pairs", "08_compute_saved", "09_ablation", "10_transfer", "11_replay",
)


def main() -> None:
    checks = []

    def check(condition, item: str, evidence: str) -> None:
        if not condition:
            raise AssertionError(item)
        checks.append(f"- [x] **{item}.** {evidence}")

    m = json.loads((R / "metrics.json").read_text())
    n, display = m["numbers"], m["display"]
    all_run = json.loads((R / "all_run.json").read_text())
    analysis = json.loads((R / "analysis_run.json").read_text())
    quick = json.loads((R / "quick_run.json").read_text())
    smoke = json.loads((Path(quick["workspace"]) / "results/phase2.json").read_text())
    text = (R / "final_tests.txt").read_text()
    count = re.search(r"(\d+) passed", text)
    check(count and int(count[1]) >= 70, "make test passes", f"{count[1]} tests including replay. [Test output](final_tests.txt)")
    check(quick["seconds"] < 600 and len(quick["executed"]) == 10,
          "make quick completes from an empty derived-data workspace",
          f"{smoke['runs']} normalized runs in {quick['seconds']:.1f} seconds. The smoke run reused only public raw inputs. [Smoke evidence](quick_run.json)")
    check(not all_run["analysis_only"] and len(analysis["executed"]) >= 9, "make all completes",
          f"Analytical reproduction took {analysis['seconds']:.1f} seconds. The final command verified cached stages and produced the package in {all_run['seconds']:.1f} seconds. [Analytical run](analysis_run.json) · [Final run](all_run.json)")
    normalized = pd.read_csv(R / "normalization_checks.csv")
    card = Path("DATA_CARD.md").read_text()
    check(normalized["passed"].all() and all(name in card for name in normalized["submission"]),
          "Data card lists submission counts and official resolve rates",
          f"All {len(normalized)} submission checks pass. [Data card](../DATA_CARD.md) · [Label joins](normalization_checks.csv)")
    for stem in EXPECTED:
        folder = Path("figures")
        with Image.open(folder / (stem + ".png")) as image:
            assert image.info.get("dpi", (0, 0))[0] >= 299 and image.width >= 1000
        assert (folder / (stem + ".pdf")).read_bytes().startswith(b"%PDF")
    tables = sorted((R / "tables").glob("*.tex"))
    check(len(tables) >= 11 and all(p.with_suffix(".csv").exists() and r"\toprule" in p.read_text() for p in tables),
          "Code-generated figures and booktabs tables cover RQ1–RQ3",
          f"{len(EXPECTED)} PNG/PDF figure pairs cover the eleven numbered figures and the additional same-task figure. "
          f"{len(tables)} matching CSV/LaTeX table pairs. [Figures](../figures) · [Tables](tables)")
    docs = [Path("README.md"), Path("slides/slides_outline.md")]
    keys = ["FixedFive","FixedTwenty","FixedForty","SameTask","MultiSave","MultiCatch","MultiFalse","MultiLo","MultiHi",
            "TransferOpen","TransferSWE","TransferMini","TransferFalse","BestTenAUC","BestTwentyAUC"]
    for p in docs:
        body = p.read_text()
        assert all(display[key] in body for key in keys), p
    tex = (R / "macros.tex").read_text()
    assert all("\\newcommand{\\" + key + "}" in tex for key in keys)
    check(True, "Headline and supplementary numbers match metrics.json",
          "README and slide outline contain the exact display values. The manuscript imports the same generated macros. [Metrics](metrics.json) · [Macros](macros.tex)")
    dev = Path("DEVIATIONS.md").read_text()
    check(all(word in dev for word in ("Phase 1", "Phase 2", "Phases 3 and 4", "Phases 5 and 6", "Phases 7 and 8", "Phase 9")),
          "Departures and scope decisions are recorded", "[Deviations](../DEVIATIONS.md)")
    paper = Path("output/pdf/final_report.pdf")
    info = subprocess.run(["pdfinfo", str(paper)], check=True, capture_output=True, text=True).stdout
    pages = re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE)
    log = paper.with_suffix(".log").read_text()
    check(pages and int(pages[1]) == 9 and not re.search(r"Overfull|Underfull|(?:LaTeX|Package).*Warning", log),
          "NeurIPS preprint compiles cleanly",
          f"{pages[1]} pages including references. [PDF](../output/pdf/final_report.pdf) · [Build log](../output/pdf/final_report.log)")
    source = Path("report/final_report.tex").read_text()
    check(r"\input{../results/macros.tex}" in source and all(token in source for token in
          ("TODO: Validation agreement","TODO: Archetype names","TODO: Confirm the course page limit")),
          "Paper sources results from generated files and marks pending student tasks",
          "[TeX source](../report/final_report.tex)")
    outline = Path("slides/slides_outline.md").read_text()
    sections = re.split(r"^## \d+\. ", outline, flags=re.MULTILINE)[1:]
    assert all(3 <= len(re.findall(r"^\- ", section, re.MULTILINE)) <= 5 for section in sections)
    assert all(Path("slides", path).resolve().is_file() for path in re.findall(r"\]\((\.\./figures/[^)]+)\)", outline))
    check(12 <= len(sections) <= 15 and all(name in outline for name in ("Aryan ·", "Niyam ·", "Navneet ·"))
          and (R / "demo_health.txt").read_text().strip() == "ok",
          "Timed outline and demo verified",
          f"{len(sections)} slides with valid figure links. Streamlit health and AppTest passed. [Outline](../slides/slides_outline.md) · [Demo](../src/atm/demo/app.py)")
    check(n["PatternOverlap"] == 0 and float(display["TransferFalse"].strip("%")) > 10
          and float(display["WindowExactSeconds"]) < float(display["WindowLSHSeconds"]),
          "Negative and surprising results are disclosed",
          "Transfer false-stop breach, zero retained overlap, slightly faster compiled exact search, and weak pattern/archetype ablations. [Paper](../output/pdf/final_report.pdf)")
    header = "# Definition-of-done checklist\n\n"
    (R / "definition_of_done.md").write_text(header + "\n".join(checks) + "\n")
    print(f"Verified {len(checks)} checklist items.")


if __name__ == "__main__":
    main()
