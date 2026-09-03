"""Structured input diagnostics for researcher-facing data ingestion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CSVValidationIssue:
    """One actionable problem found while reading a tidy CSV file."""

    code: str
    message: str
    column: str | None = None
    row: int | None = None
    value: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    def format(self) -> str:
        """Render a concise location-aware diagnostic."""
        location: list[str] = []
        if self.row is not None:
            location.append(f"row {self.row}")
        if self.column:
            location.append(f"column '{self.column}'")
        prefix = ", ".join(location) or "CSV"
        value = f"; received {self.value!r}" if self.value is not None else ""
        return f"{prefix}: {self.message}{value} [{self.code}]"


class CSVValidationError(ValueError):
    """Raised when strict CSV ingestion finds one or more invalid values."""

    def __init__(self, issues: list[CSVValidationIssue]) -> None:
        if not issues:
            raise ValueError("CSVValidationError requires at least one issue")
        self.issues = list(issues)
        preview = "; ".join(issue.format() for issue in self.issues[:5])
        remaining = len(self.issues) - 5
        if remaining > 0:
            preview += f"; and {remaining} more issue(s)"
        super().__init__(
            f"CSV validation failed with {len(self.issues)} issue(s): {preview}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return structured diagnostics suitable for CLI JSON output."""
        return {
            "error": "CSV_VALIDATION_FAILED",
            "issue_count": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }
