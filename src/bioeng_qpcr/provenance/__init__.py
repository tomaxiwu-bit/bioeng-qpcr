"""Provenance tracking for bioeng-qpcr.

A :class:`~bioeng_qpcr.provenance.ProvenanceManifest` captures the full
computational provenance of a single analysis run: the cryptographic hash of
the input experiment, the versions of all software packages involved, the
analysis parameters, and a UTC timestamp.  Together this information lets any
third party reproduce or audit a result.
"""

from bioeng_qpcr.provenance.manifest import ProvenanceManifest

__all__ = ["ProvenanceManifest"]
