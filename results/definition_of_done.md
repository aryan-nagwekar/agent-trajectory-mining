# Definition-of-done checklist

- [x] **make test passes.** 71 tests including replay. [Test output](final_tests.txt)
- [x] **make quick completes from an empty derived-data workspace.** 498 normalized runs in 113.0 seconds. The smoke run reused only public raw inputs. [Smoke evidence](quick_run.json)
- [x] **make all completes.** Analytical reproduction took 567.9 seconds. The final command verified cached stages and produced the package in 18.7 seconds. [Analytical run](analysis_run.json) · [Final run](all_run.json)
- [x] **Data card lists submission counts and official resolve rates.** All 10 submission checks pass. [Data card](../DATA_CARD.md) · [Label joins](normalization_checks.csv)
- [x] **Code-generated figures and booktabs tables cover RQ1–RQ3.** 12 PNG/PDF figure pairs cover the eleven numbered figures and the additional same-task figure. 14 matching CSV/LaTeX table pairs. [Figures](../figures) · [Tables](tables)
- [x] **Headline and supplementary numbers match metrics.json.** README and slide outline contain the exact display values. The manuscript imports the same generated macros. [Metrics](metrics.json) · [Macros](macros.tex)
- [x] **Departures and scope decisions are recorded.** [Deviations](../DEVIATIONS.md)
- [x] **NeurIPS preprint compiles cleanly.** 9 pages including references. [PDF](../output/pdf/final_report.pdf) · [Build log](../output/pdf/final_report.log)
- [x] **Paper sources results from generated files and marks pending student tasks.** [TeX source](../report/final_report.tex)
- [x] **Timed outline and demo verified.** 15 slides with valid figure links. Streamlit health and AppTest passed. [Outline](../slides/slides_outline.md) · [Demo](../src/atm/demo/app.py)
- [x] **Negative and surprising results are disclosed.** Transfer false-stop breach, zero retained overlap, slightly faster compiled exact search, and weak pattern/archetype ablations. [Paper](../output/pdf/final_report.pdf)
