LOCAL_PYTHON := $(firstword $(wildcard .python/*/bin/python3.11))
PYTHON ?= $(or $(LOCAL_PYTHON),python3.11)
VENV ?= .venv
CONFIG ?= configs/default.yaml
LIMIT ?= 8

.PHONY: setup data

setup:
	"$(PYTHON)" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
	"$(PYTHON)" -m venv "$(VENV)"
	"$(VENV)/bin/python" -m pip install -r requirements.txt
	mkdir -p results
	"$(VENV)/bin/python" -m pip freeze > results/env.txt

data:
	PYTHONPATH=src "$(VENV)/bin/python" -m atm.download --config "$(CONFIG)" --limit "$(LIMIT)"

.PHONY: inspect
inspect:
	"$(VENV)/bin/python" scripts/01_inspect_sample.py --config "$(CONFIG)" --limit "$(LIMIT)"

NORMALIZE_LIMIT ?= 0

.PHONY: normalize test
normalize:
	PYTHONPATH=src "$(VENV)/bin/python" -m atm.normalize --config "$(CONFIG)" --limit "$(NORMALIZE_LIMIT)"

test:
	$(NATIVE_ENV) PYTHONPATH=src "$(VENV)/bin/python" -m pytest -q

ANALYSIS_CONFIG ?= configs/expanded.yaml

.PHONY: audit splits mine
audit:
	PYTHONPATH=src "$(VENV)/bin/python" -m atm.audit --config "$(ANALYSIS_CONFIG)"

splits: audit
	PYTHONPATH=src "$(VENV)/bin/python" -m atm.splits --config "$(ANALYSIS_CONFIG)"

mine: splits
	PYTHONPATH=src "$(VENV)/bin/python" -m atm.mining.discriminative --config "$(ANALYSIS_CONFIG)"

.PHONY: phase4-report
phase4-report: mine
	"$(VENV)/bin/python" scripts/04_report.py

.PHONY: cluster
cluster:
	MPLCONFIGDIR=/tmp/atm-mpl PYTHONPATH=src "$(VENV)/bin/python" -m atm.clustering.archetypes --config "$(ANALYSIS_CONFIG)"

# PyTorch supplies a compatible OpenMP runtime on Apple Silicon.
PYTORCH_LIB := $(firstword $(wildcard $(VENV)/lib/python*/site-packages/torch/lib))
.PHONY: predict
predict:
	DYLD_LIBRARY_PATH="$(abspath $(PYTORCH_LIB)):$(DYLD_LIBRARY_PATH)" MPLCONFIGDIR=/tmp/atm-mpl PYTHONPATH=src "$(VENV)/bin/python" -m atm.predict --config "$(ANALYSIS_CONFIG)"

.PHONY: phase6-report
phase6-report:
	"$(VENV)/bin/python" scripts/06_report.py

NATIVE_ENV = DYLD_LIBRARY_PATH="$(abspath $(PYTORCH_LIB)):$(DYLD_LIBRARY_PATH)" MPLCONFIGDIR=/tmp/atm-mpl

.PHONY: archetype-sweep window-benchmark archetype-refit cohorts calibrate transfer phase8-report
archetype-sweep:
	PYTHONPATH=src "$(VENV)/bin/python" -m atm.clustering.sweep --config "$(ANALYSIS_CONFIG)"

window-benchmark:
	MPLCONFIGDIR=/tmp/atm-mpl PYTHONPATH=src "$(VENV)/bin/python" -m atm.clustering.window_benchmark --config "$(ANALYSIS_CONFIG)"

archetype-refit:
	$(NATIVE_ENV) PYTHONPATH=src "$(VENV)/bin/python" -m atm.revision refit --config "$(ANALYSIS_CONFIG)"

cohorts:
	$(NATIVE_ENV) PYTHONPATH=src "$(VENV)/bin/python" -m atm.cohorts

calibrate:
	PYTHONPATH=src "$(VENV)/bin/python" -m atm.calibration grouped --config "$(ANALYSIS_CONFIG)"

transfer:
	$(NATIVE_ENV) PYTHONPATH=src "$(VENV)/bin/python" -m atm.revision transfer --config "$(ANALYSIS_CONFIG)"
	PYTHONPATH=src "$(VENV)/bin/python" -m atm.calibration transfer --config "$(ANALYSIS_CONFIG)"
	PYTHONPATH=src "$(VENV)/bin/python" -m atm.transfer_patterns --config "$(ANALYSIS_CONFIG)"

phase8-report:
	"$(VENV)/bin/python" scripts/08_report.py

.PHONY: figures tables demo
figures:
	$(NATIVE_ENV) PYTHONPATH=src "$(VENV)/bin/python" -m atm.figures
tables:
	PYTHONPATH=src "$(VENV)/bin/python" -m atm.final_results --config "$(ANALYSIS_CONFIG)"
demo:
	$(NATIVE_ENV) PYTHONPATH=src "$(VENV)/bin/python" -m streamlit run src/atm/demo/app.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false

.PHONY: all quick report
all:
	@test -x "$(VENV)/bin/python" || $(MAKE) setup
	$(NATIVE_ENV) PYTHONPATH=src "$(VENV)/bin/python" scripts/reproduce.py

quick:
	@test -x "$(VENV)/bin/python" || $(MAKE) setup
	$(NATIVE_ENV) PYTHONPATH=src "$(VENV)/bin/python" scripts/reproduce.py --quick

report:
	$(NATIVE_ENV) PYTHONPATH=src "$(VENV)/bin/python" scripts/reproduce.py --report-only
