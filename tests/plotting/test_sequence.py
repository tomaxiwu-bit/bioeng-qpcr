"""Tests for sequencing QC figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from bioeng_qpcr.plotting import plot_alignment_coverage, plot_variant_map


def test_alignment_coverage_marks_read_and_variants():
    figure, ax = plot_alignment_coverage(430, 0, 120, [24, 78])
    try:
        assert ax.get_xlim() == pytest.approx((1, 430))
        assert len(ax.collections) >= 3
        assert "120 of 430" in ax.get_title()
    finally:
        plt.close(figure)


def test_alignment_coverage_rejects_invalid_coordinates():
    with pytest.raises(ValueError, match="outside"):
        plot_alignment_coverage(100, 20, 101)


def test_variant_map_is_still_informative_when_no_variants_are_called():
    figure, ax = plot_variant_map(430, [])
    try:
        assert "0 called" in ax.get_title()
        assert any(text.get_text() == "No variants detected" for text in ax.texts)
    finally:
        plt.close(figure)
