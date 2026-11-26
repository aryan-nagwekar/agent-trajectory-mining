"""Check the held-out demo and its frozen checkpoint scores."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "data/processed/prefix_predictions_revised.parquet").exists(),
                    reason="trained demo artifacts are produced by make all")
def test_demo_replays_and_changes_alpha() -> None:
    app = AppTest.from_file(str(ROOT / "src/atm/demo/app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].label == "Failure probability"
    default = app.metric[0].value
    app.radio[0].set_value(.05).run()
    assert not app.exception and app.metric[0].value == default
    app.slider[0].set_value(20).run()
    assert not app.exception and app.metric[1].label == "Threshold"
