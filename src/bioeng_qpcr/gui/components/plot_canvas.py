"""Matplotlib canvas widget for embedding in PySide6.

The backend must be set to ``QtAgg`` **before** ``QApplication`` is created.
The application entry point (:func:`bioeng_qpcr.gui.main_window.main`)
handles this so that the canvas can be imported safely without forcing a
GUI backend on headless environments.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy

# Configure matplotlib to use CJK-compatible fonts so Chinese text in
# axis labels, titles and empty-state placeholders renders correctly.
mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "Microsoft YaHei UI",
    "SimHei",
    "Noto Sans CJK SC",
    "WenQuanYi Micro Hei",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False


class PlotCanvas(FigureCanvas):
    """A matplotlib FigureCanvas embedded in a Qt widget."""

    def __init__(self, parent=None, width: int = 8, height: int = 5, dpi: int = 100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.figure.set_facecolor("#ffffff")
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setStyleSheet("border: none; background: #ffffff;")
        self.show_empty_state()

    def show_empty_state(
        self,
        title: str = "图表预览",
        subtitle: str = "运行分析后将在此显示可视化结果",
    ) -> None:
        """Render a calm, informative placeholder instead of a blank canvas."""
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.set_facecolor("#ffffff")
        axis.axis("off")
        axis.text(
            0.5,
            0.54,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#6e6d68",
            transform=axis.transAxes,
        )
        axis.text(
            0.5,
            0.45,
            subtitle,
            ha="center",
            va="center",
            fontsize=9,
            color="#908e84",
            transform=axis.transAxes,
        )
        self.figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.draw_idle()

    def clear(self) -> None:
        """Reset the canvas to its informative empty state."""
        self.show_empty_state()

    def draw_figure(self, fig: Figure) -> None:
        """Replace the current figure content with an external figure's axes.

        This is a simplified placeholder; for complex plots the caller
        should use :meth:`set_figure` instead.
        """
        self.figure.clear()
        for _ax in fig.axes:
            self.figure.add_subplot(111)
        self.draw()

    def set_figure(self, fig: Figure) -> None:
        """Replace the internal figure and redraw."""
        fig.set_canvas(self)
        self.figure = fig
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        fig.set_size_inches(width / fig.dpi, height / fig.dpi, forward=False)
        try:
            fig.tight_layout(pad=1.2)
        except (RuntimeError, ValueError):
            # Some specialized matplotlib layouts cannot be tightened.
            pass
        self.updateGeometry()
        self.draw_idle()
