"""Command-line interface for bioeng-qpcr.

This package provides a durable, non-Jupyter command-line interface to the
bioeng-qpcr analysis engine.  The entry point is the :func:`cli` click group
exposed via the ``bioeng-qpcr`` console script (see ``pyproject.toml``).

``cli`` is imported lazily to avoid the ``RuntimeWarning`` that occurs when
``python -m bioeng_qpcr.cli.main`` pre-imports the package's ``__init__``
before executing the ``main`` module.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazily import :func:`cli` on first access."""
    if name == "cli":
        from bioeng_qpcr.cli.main import cli

        return cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["cli"]
