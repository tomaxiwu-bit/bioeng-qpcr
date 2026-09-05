"""Local web application exposing the auditable qPCR analysis pipeline.

The :mod:`~bioeng_qpcr.web` package wraps the analysis engine behind a
small, Python-only web server that mirrors the CLI workflows (validate,
analyze, curves) in a browser.  It keeps the science in the reusable
:mod:`~bioeng_qpcr.io`, :mod:`~bioeng_qpcr.calculations`,
:mod:`~bioeng_qpcr.validation` and :mod:`~bioeng_qpcr.plotting` layers, so
results are consistent with the CLI, widgets and MCP server.
"""

from __future__ import annotations

from bioeng_qpcr.web.app import create_app, main

__all__ = ["create_app", "main"]