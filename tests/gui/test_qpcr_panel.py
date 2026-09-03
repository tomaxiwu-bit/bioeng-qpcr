"""Adversarial workflow tests for the desktop qPCR panel."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from bioeng_qpcr.domain import Assay, Experiment, Sample, Well
from bioeng_qpcr.gui.panels.qpcr_panel import QpcrPanel


def _three_group_experiment() -> Experiment:
    samples = {
        "C1": Sample("C1", group="control"),
        "C2": Sample("C2", group="control"),
        "L1": Sample("L1", group="low-dose"),
        "L2": Sample("L2", group="low-dose"),
        "H1": Sample("H1", group="high-dose"),
        "H2": Sample("H2", group="high-dose"),
    }
    assays = {
        "GOI": Assay("GOI", efficiency=1.0),
        "REF": Assay("REF", efficiency=1.0, is_reference=True),
    }
    wells: list[Well] = []
    position = 1
    for sample_name in samples:
        target_cq = 25.0
        if sample_name.startswith("L"):
            target_cq = 24.0
        elif sample_name.startswith("H"):
            target_cq = 23.0
        if sample_name.endswith("2"):
            target_cq += 0.1
        for target_name, cq in (("GOI", target_cq), ("REF", 20.0)):
            wells.append(
                Well(
                    position=f"A{position:02d}",
                    sample_name=sample_name,
                    target_name=target_name,
                    cq=cq,
                )
            )
            position += 1
    return Experiment("three-groups", wells=wells, samples=samples, assays=assays)


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    widget = QpcrPanel()
    try:
        yield widget
    finally:
        widget.close()
        app.processEvents()


def _select(combo, value: str) -> None:
    index = combo.findData(value)
    assert index >= 0
    combo.setCurrentIndex(index)


def test_group_comparison_is_opt_in_and_never_guessed(panel: QpcrPanel) -> None:
    experiment = _three_group_experiment()
    panel._set_available_groups(experiment)

    assert panel._group_compare_check.isEnabled()
    assert not panel._group_compare_check.isChecked()
    assert panel._selected_group_comparison(experiment, "GOI", ["REF"]) is None


def test_explicit_selection_preserves_requested_pair_with_three_groups(
    panel: QpcrPanel,
) -> None:
    experiment = _three_group_experiment()
    panel._set_available_groups(experiment)
    panel._group_compare_check.setChecked(True)
    _select(panel._control_group_combo, "control")
    _select(panel._treated_group_combo, "high-dose")

    result = panel._selected_group_comparison(experiment, "GOI", ["REF"])

    assert result.control_group == "control"
    assert result.treated_group == "high-dose"
    assert set(result.treated_values) == {"H1", "H2"}
    assert result.fold_difference == pytest.approx(4.0)


def test_comparison_requires_both_explicit_selections(panel: QpcrPanel) -> None:
    experiment = _three_group_experiment()
    panel._set_available_groups(experiment)
    panel._group_compare_check.setChecked(True)
    _select(panel._control_group_combo, "control")

    with pytest.raises(ValueError, match="明确选择"):
        panel._selected_group_comparison(experiment, "GOI", ["REF"])


def test_same_control_and_treated_group_is_rejected(panel: QpcrPanel) -> None:
    experiment = _three_group_experiment()
    panel._set_available_groups(experiment)
    panel._group_compare_check.setChecked(True)
    _select(panel._control_group_combo, "control")
    _select(panel._treated_group_combo, "control")

    with pytest.raises(ValueError, match="必须是不同"):
        panel._selected_group_comparison(experiment, "GOI", ["REF"])


def test_switching_method_disables_group_comparison(panel: QpcrPanel) -> None:
    panel._set_available_groups(_three_group_experiment())
    panel._group_compare_check.setChecked(True)

    panel._method_combo.setCurrentIndex(panel._method_combo.findData("pfaffl"))

    assert not panel._group_compare_check.isEnabled()
    assert not panel._group_compare_check.isChecked()
    assert not panel._control_group_combo.isEnabled()
    assert not panel._treated_group_combo.isEnabled()
