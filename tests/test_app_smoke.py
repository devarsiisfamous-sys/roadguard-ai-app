"""Headless UI tests with Streamlit's AppTest: every page renders, and the full
load → train → predict workflow works through the real interface."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    "pages/1_Accident_Analytics.py",
    "pages/2_Risk_Prediction.py",
    "pages/3_Model_Laboratory.py",
    "pages/4_Data_Explorer.py",
    "pages/5_About.py",
]
TIMEOUT = 300


def _app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)


def _assert_clean(at: AppTest) -> None:
    assert not at.exception, [e.message for e in at.exception]


@pytest.mark.parametrize("page", [None, *PAGES])
def test_every_page_renders_without_models(page: str | None, models_dir: Path) -> None:
    at = _app().run()
    if page:
        at.switch_page(page).run()
    _assert_clean(at)


def test_risk_page_without_model_explains_next_step(models_dir: Path) -> None:
    at = _app().run()
    at.switch_page("pages/2_Risk_Prediction.py").run()
    _assert_clean(at)
    assert any("No trained model" in i.value for i in at.info)


def test_train_then_predict_workflow(models_dir: Path) -> None:
    at = _app().run()
    at.switch_page("pages/3_Model_Laboratory.py").run()
    _assert_clean(at)
    # Train only the fast models to keep the test quick.
    for key in ("decision_tree", "random_forest", "svm", "xgboost"):
        box = at.checkbox(key=f"train_pick_{key}")
        if not box.disabled:
            box.uncheck()
    submit = next(b for b in at.button if b.label == "Train selected models")
    submit.click().run()
    _assert_clean(at)
    assert not at.error, [e.value for e in at.error]
    assert (models_dir / "logistic_regression" / "pipeline.joblib").exists()
    assert (models_dir / "champion.json").exists()
    assert any("Trained and saved" in s.value for s in at.success)

    at.switch_page("pages/2_Risk_Prediction.py").run()
    _assert_clean(at)
    estimate = next(b for b in at.button if b.label == "Estimate accident risk")
    estimate.click().run()
    _assert_clean(at)
    assert not at.error, [e.value for e in at.error]
    assert any("Estimated accident probability" in m.value for m in at.markdown)

    add = next(b for b in at.button if b.label == "Add to scenario comparison")
    add.click().run()
    _assert_clean(at)
    assert any("Added" in s.value for s in at.success)

    # The other pages still render with a trained model present.
    for page in ("pages/3_Model_Laboratory.py", "pages/1_Accident_Analytics.py", "app.py"):
        at.switch_page(page).run()
        _assert_clean(at)
