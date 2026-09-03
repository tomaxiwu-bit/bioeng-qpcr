"""bioeng-qpcr desktop GUI application.

A PySide6-based desktop interface providing point-and-click access to all
four analysis modules: qPCR analysis, enzyme kinetics, fermentation
kinetics, and FBA metabolic flux analysis.
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> None:
    """Launch the bioeng-qpcr desktop application."""
    from bioeng_qpcr.gui.main_window import main as _main

    _main()
