"""Auditable batch-analysis result types."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class AnalysisExclusion:
    """A sample intentionally omitted from a batch result, with a reason."""

    sample_name: str
    code: str
    reason: str
    target_name: str | None = None


@dataclass
class BatchAnalysisResult(Generic[T]):
    """Results plus exclusions, warnings, and effective parameters.

    The sequence protocol is provided for convenient migration from the old
    list-returning API, while explicit attributes keep omissions visible.
    """

    results: list[T] = field(default_factory=list)
    exclusions: list[AnalysisExclusion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[T]:
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, index: int) -> T:
        return self.results[index]

    @property
    def is_complete(self) -> bool:
        """Whether every candidate sample produced a result."""
        return not self.exclusions

    def raise_for_exclusions(self) -> None:
        """Raise with a complete summary when any sample was excluded."""
        if not self.exclusions:
            return
        detail = "; ".join(
            f"{item.sample_name}: {item.reason}" for item in self.exclusions
        )
        raise ValueError(
            f"Analysis excluded {len(self.exclusions)} sample(s): {detail}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        serialized_results = [
            asdict(result) if is_dataclass(result) else result
            for result in self.results
        ]
        return {
            "results": serialized_results,
            "exclusions": [asdict(item) for item in self.exclusions],
            "warnings": list(self.warnings),
            "parameters": dict(self.parameters),
            "summary": {
                "result_count": len(self.results),
                "exclusion_count": len(self.exclusions),
                "complete": self.is_complete,
            },
        }
