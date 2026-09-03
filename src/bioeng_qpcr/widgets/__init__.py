"""Interactive ipywidgets widgets for bioeng-qpcr.

This package provides Jupyter-friendly interactive widgets built on top of
the core bioeng-qpcr engine.  ``ipywidgets`` is an optional dependency;
importing this package succeeds without it, but instantiating
:class:`~bioeng_qpcr.widgets.analyzer.QPCRAnalyzer` raises a clear
:class:`ImportError` with installation instructions.
"""

from bioeng_qpcr.widgets.analyzer import QPCRAnalyzer

__all__ = ["QPCRAnalyzer"]
