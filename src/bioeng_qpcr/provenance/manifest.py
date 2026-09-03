"""Provenance manifest: an auditable record of a single analysis run.

The :class:`ProvenanceManifest` is the cornerstone of *auditability* for the
bioeng-qpcr engine.  Every analysis result can be paired with a manifest that
answers four reproducibility questions:

1. **What went in?** -- the SHA-256 content hash of the input experiment.
2. **What software was used?** -- versions of ``bioeng_qpcr`` and its
   scientific dependencies (``numpy``, ``scipy``, ``pandas``).
3. **How was it analysed?** -- the analysis parameters (method, reference
   genes, thresholds, ...).
4. **When did it happen?** -- a UTC creation timestamp.

A manifest is JSON-serialisable via :meth:`ProvenanceManifest.to_json` and can
be reconstructed losslessly with :meth:`ProvenanceManifest.from_dict`.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
from datetime import datetime, timezone
from typing import Any, ClassVar

from bioeng_qpcr.domain.entities import Experiment


class ProvenanceManifest:
    """Record of the provenance of a single qPCR analysis run.

    Parameters
        input_hash:
            SHA-256 hex digest of the input experiment's scientific content
            (typically produced by :meth:`Experiment.content_hash`).
            ``None`` when the manifest is created without an experiment.
        parameters:
            Dictionary of analysis parameters (e.g. quantification method,
            reference gene, Cq threshold).  Defaults to an empty dict.
        package_versions:
            Mapping of package import-name to installed version string.
            When omitted, the versions are left empty and may be captured
            later via :meth:`capture_package_versions`.

    Attributes
        input_hash:
            The recorded input content hash (read-write).
        parameters:
            The recorded analysis parameters (read-write).
        package_versions:
            The recorded package versions (read-write).
        created_at:
            UTC creation timestamp (read-only property).

    Examples
        >>> from bioeng_qpcr.provenance import ProvenanceManifest
        >>> manifest = ProvenanceManifest(
        ...     input_hash="0" * 64,
        ...     parameters={"method": "delta_delta_cq"},
        ... )
        >>> isinstance(manifest.created_at, datetime)
        True
        >>> "input_hash" in manifest.to_dict()
        True
    """

    #: Mapping of *import name* to *distribution name* for every package whose
    #: version is tracked for reproducibility.  ``bioeng_qpcr`` is imported
    #: with an underscore but distributed as ``bioeng-qpcr`` with a hyphen.
    _TRACKED_PACKAGES: ClassVar[dict[str, str]] = {
        "bioeng_qpcr": "bioeng-qpcr",
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
    }

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        input_hash: str | None = None,
        parameters: dict[str, Any] | None = None,
        package_versions: dict[str, str] | None = None,
        raw_input_hash: str | None = None,
        ruleset_versions: dict[str, str] | None = None,
        exclusions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.input_hash: str | None = input_hash
        self.raw_input_hash: str | None = raw_input_hash
        # Copy mutable inputs so external mutation cannot leak into the
        # manifest and so the dangerous "mutable default argument" pitfall is
        # avoided.
        self.parameters: dict[str, Any] = dict(parameters) if parameters else {}
        self.package_versions: dict[str, str] = (
            dict(package_versions) if package_versions else {}
        )
        self.ruleset_versions: dict[str, str] = (
            dict(ruleset_versions) if ruleset_versions else {}
        )
        self.exclusions: list[dict[str, Any]] = [
            dict(item) for item in (exclusions or [])
        ]
        self._created_at: datetime = datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def created_at(self) -> datetime:
        """UTC creation timestamp of this manifest (read-only)."""
        return self._created_at

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_experiment(
        cls,
        experiment: Experiment,
        parameters: dict[str, Any] | None = None,
    ) -> ProvenanceManifest:
        """Build a manifest from an :class:`Experiment`.

        The input hash is derived from the experiment's scientific content via
        :meth:`Experiment.content_hash`, and the versions of all tracked
        packages are captured automatically from the current environment.

        Parameters
            experiment:
                The input experiment whose provenance is being recorded.
            parameters:
                Optional analysis parameters to attach to the manifest.

        Returns
            A new :class:`ProvenanceManifest`.
        """
        input_hash = experiment.content_hash()
        package_versions = cls.capture_package_versions()
        raw_input_hash = experiment.metadata.get("source_sha256")
        from bioeng_qpcr.validation.rules import BUILTIN_RULES

        ruleset_versions = {
            metadata.rule_id: metadata.version for metadata, _func in BUILTIN_RULES
        }
        return cls(
            input_hash=input_hash,
            raw_input_hash=(
                str(raw_input_hash) if raw_input_hash is not None else None
            ),
            parameters=parameters,
            package_versions=package_versions,
            ruleset_versions=ruleset_versions,
        )

    @classmethod
    def capture_package_versions(cls) -> dict[str, str]:
        """Return the installed versions of all tracked packages.

        Resolution prefers distribution metadata
        (:func:`importlib.metadata.version`) and falls back to the module's
        ``__version__`` attribute, which keeps the method working even when the
        package is run from a source checkout without installation.
        """
        versions: dict[str, str] = {}
        for import_name, dist_name in cls._TRACKED_PACKAGES.items():
            version = cls._resolve_version(import_name, dist_name)
            if version is not None:
                versions[import_name] = version
        return versions

    @staticmethod
    def _resolve_version(import_name: str, dist_name: str) -> str | None:
        """Resolve the version of a single package.

        For ``bioeng_qpcr``, the running module version takes precedence so an
        editable checkout records the code actually executing. Other packages
        use distribution metadata first and fall back to ``__version__``.
        Returns ``None`` if the package cannot be located by either mechanism.
        """
        # Editable source trees can temporarily disagree with installed
        # distribution metadata during a version bump. For this package,
        # record the version of the code that is actually executing.
        if import_name == "bioeng_qpcr":
            try:
                module = importlib.import_module(import_name)
            except ImportError:
                module = None
            runtime_version = getattr(module, "__version__", None)
            if runtime_version is not None:
                return str(runtime_version)
        try:
            return importlib.metadata.version(dist_name)
        except importlib.metadata.PackageNotFoundError:
            pass
        try:
            module = importlib.import_module(import_name)
        except ImportError:
            return None
        return getattr(module, "__version__", None)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary representation.

        The ``created_at`` timestamp is rendered as an ISO-8601 string so that
        the entire dictionary can be passed directly to :func:`json.dumps`.
        """
        return {
            "input_hash": self.input_hash,
            "raw_input_hash": self.raw_input_hash,
            "parameters": self.parameters,
            "package_versions": self.package_versions,
            "ruleset_versions": self.ruleset_versions,
            "exclusions": self.exclusions,
            "created_at": self._created_at.isoformat(),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Return a JSON string representation of this manifest.

        A ``default=str`` fallback is used so that non-natively-serialisable
        parameter values (e.g. :class:`~decimal.Decimal` or custom objects)
        are rendered via :func:`str` rather than raising.
        """
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=True,
            default=str,
            ensure_ascii=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceManifest:
        """Reconstruct a :class:`ProvenanceManifest` from a dictionary.

        Accepts the output of :meth:`to_dict` (or a JSON object parsed via
        :func:`json.loads`).  Missing keys default to empty/``None`` so that
        partial records still load without error.  The ``created_at`` string
        is parsed back into a timezone-aware :class:`~datetime.datetime`.

        Raises
            TypeError:
                If ``data`` is not a dictionary.
            ValueError:
                If ``created_at`` is present but not a valid ISO-8601
                timestamp.
        """
        if not isinstance(data, dict):
            raise TypeError(f"from_dict expects a dict, got {type(data).__name__}")

        manifest = cls(
            input_hash=data.get("input_hash"),
            raw_input_hash=data.get("raw_input_hash"),
            parameters=data.get("parameters") or {},
            package_versions=data.get("package_versions") or {},
            ruleset_versions=data.get("ruleset_versions") or {},
            exclusions=data.get("exclusions") or [],
        )

        created_at_str = data.get("created_at")
        if created_at_str:
            try:
                manifest._created_at = datetime.fromisoformat(str(created_at_str))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid created_at timestamp: {created_at_str!r}"
                ) from exc

        return manifest

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable, multi-line summary of the manifest.

        The summary is intentionally plain text so it can be embedded in
        reports, logs, or console output without further formatting.
        """
        lines: list[str] = [
            "Provenance Manifest",
            "===================",
            f"Created at (UTC): {self._created_at.isoformat()}",
            f"Input hash:       {self.input_hash or '(not set)'}",
            f"Raw input hash:   {self.raw_input_hash or '(not set)'}",
            "",
            "Package versions:",
        ]
        if self.package_versions:
            for name in sorted(self.package_versions):
                lines.append(f"  {name} = {self.package_versions[name]}")
        else:
            lines.append("  (none captured)")

        lines.append("")
        lines.append("Validation rules:")
        if self.ruleset_versions:
            for rule_id in sorted(self.ruleset_versions):
                lines.append(f"  {rule_id} = {self.ruleset_versions[rule_id]}")
        else:
            lines.append("  (none captured)")

        lines.append("")
        lines.append("Analysis parameters:")
        if self.parameters:
            for key in sorted(self.parameters):
                lines.append(f"  {key} = {self.parameters[key]}")
        else:
            lines.append("  (none provided)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Return ``True`` when two manifests record identical provenance.

        Equality considers all four fields, including the creation timestamp,
        so that a manifest reconstructed via :meth:`from_dict` compares equal
        to its source only when the timestamp is preserved exactly.
        """
        if not isinstance(other, ProvenanceManifest):
            return NotImplemented
        return (
            self.input_hash == other.input_hash
            and self.parameters == other.parameters
            and self.package_versions == other.package_versions
            and self.raw_input_hash == other.raw_input_hash
            and self.ruleset_versions == other.ruleset_versions
            and self.exclusions == other.exclusions
            and self._created_at == other._created_at
        )

    def __repr__(self) -> str:
        return (
            f"ProvenanceManifest(input_hash={self.input_hash!r}, "
            f"package_versions={self.package_versions!r}, "
            f"created_at={self._created_at.isoformat()!r})"
        )
