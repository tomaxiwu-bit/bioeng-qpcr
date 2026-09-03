"""Flux Balance Analysis (FBA) solving.

Loads a COBRApy metabolic model and solves the FBA optimisation
problem to obtain the optimal flux distribution that maximises (or
minimises) the model's objective function.

The FBA optimisation finds fluxes **v** that satisfy::

    S · v = 0           (steady-state mass balance)
    lb ≤ v ≤ ub          (flux bounds)
    maximise c^T · v     (objective)

where *S* is the stoichiometric matrix, *lb*/*ub* are per-reaction
flux bounds, and *c* is the objective coefficient vector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

__all__ = [
    "FBAResult",
    "ModelSummary",
    "load_model",
    "run_fba",
]

if TYPE_CHECKING:
    from cobra import Model


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSummary:
    """Summary statistics of a metabolic model.

    Attributes
        model_id:
            Identifier of the model.
        n_reactions:
            Number of reactions in the network.
        n_metabolites:
            Number of metabolites in the network.
        n_genes:
            Number of genes in the GPR associations.
        compartments:
            Set of compartment identifiers.
        objective_direction:
            Direction of optimisation (``"max"`` or ``"min"``).
    """

    model_id: str
    n_reactions: int
    n_metabolites: int
    n_genes: int
    compartments: tuple[str, ...]
    objective_direction: str


@dataclass(frozen=True)
class FBAResult:
    """Result of a Flux Balance Analysis.

    Attributes
        objective_value:
            Optimal objective value (e.g. growth rate in h⁻¹).
        status:
            Solver status (``"optimal"``, ``"infeasible"``, etc.).
        fluxes:
            Mapping of reaction ID to flux value.
        model_summary:
            Summary of the model that was solved.
        active_reactions:
            Reactions carrying non-zero flux.
        n_active:
            Number of active reactions.
        n_total:
            Total number of reactions.
    """

    objective_value: float
    status: str
    fluxes: dict[str, float] = field(default_factory=dict)
    model_summary: ModelSummary | None = None
    active_reactions: tuple[str, ...] = ()
    n_active: int = 0
    n_total: int = 0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    source: str | Path,
) -> Model:
    """Load a metabolic model from a file or built-in name.

    Parameters
        source:
            Either a path to an SBML file (``.xml`` / ``.sbml``) or the
            name of a built-in model (e.g. ``"textbook"`` for the
            *E. coli* core model).

    Returns
        A :class:`cobra.Model` instance.

    Raises
        ImportError:
            If COBRApy is not installed.
        FileNotFoundError:
            If *source* is a file path that does not exist.
        ValueError:
            If the model cannot be loaded.
    """
    try:
        import cobra
    except ImportError as exc:
        raise ImportError(
            "COBRApy is required for metabolic flux analysis. "
            "Install it with: pip install cobra"
        ) from exc

    source_str = str(source)
    path = Path(source_str)

    # Built-in model name (e.g. "textbook").
    if not path.exists() and not path.suffix:
        from cobra.io import load_model as _cobra_load
        return _cobra_load(source_str)

    # File path.
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {source_str}")

    suffix = path.suffix.lower()
    if suffix in (".xml", ".sbml"):
        return cobra.io.read_sbml_model(str(path))
    if suffix == ".json":
        return cobra.io.load_json_model(str(path))
    if suffix == ".mat":
        return cobra.io.load_matlab_model(str(path))
    raise ValueError(
        f"Unsupported model format '{suffix}'. "
        "Supported: .xml, .sbml, .json, .mat"
    )


def _build_model_summary(model: Model) -> ModelSummary:
    """Extract summary statistics from a cobra Model."""
    return ModelSummary(
        model_id=model.id or "unknown",
        n_reactions=len(model.reactions),
        n_metabolites=len(model.metabolites),
        n_genes=len(model.genes),
        compartments=tuple(sorted(model.compartments)),
        objective_direction=(
            "max" if model.objective_direction == "max" else "min"
        ),
    )


# ---------------------------------------------------------------------------
# FBA solving
# ---------------------------------------------------------------------------


def run_fba(
    model: Model,
    *,
    objective: str | None = None,
    direction: str | None = None,
) -> FBAResult:
    """Solve Flux Balance Analysis on a metabolic model.

    The input *model* is **never mutated**: when *objective* or
    *direction* differ from the model's current settings the original
    values are saved and restored in a ``finally`` block, so callers can
    safely reuse the model after this function returns.

    Parameters
        model:
            A :class:`cobra.Model` instance.
        objective:
            Optional reaction ID to use as the objective. When omitted
            the model's current objective is used.
        direction:
            Optimisation direction: ``"max"`` or ``"min"``. When omitted
            the model's current direction is used.

    Returns
        :class:`FBAResult` with the optimal flux distribution.

    Raises
        ValueError:
            If *objective* is not a valid reaction ID.
        RuntimeError:
            If the FBA problem is infeasible or unbounded.
    """
    # Validate inputs *before* touching the model.
    if direction is not None:
        direction_lower = direction.lower()
        if direction_lower not in ("max", "min"):
            raise ValueError(
                f"Direction must be 'max' or 'min', got '{direction}'."
            )
    else:
        direction_lower = None

    if objective is not None and objective not in model.reactions:
        raise ValueError(
            f"Objective reaction '{objective}' not found in model."
        )

    # Save original state so we can restore it after optimisation.
    saved_objective = model.objective
    saved_direction = model.objective_direction
    _mutated = False

    try:
        if objective is not None:
            model.objective = model.reactions.get_by_id(objective)
            _mutated = True
        if direction_lower is not None:
            model.objective_direction = direction_lower
            _mutated = True

        solution = model.optimize()

        fluxes: dict[str, float] = {}
        active: list[str] = []
        for rxn_id, flux_val in solution.fluxes.items():
            fv = float(flux_val)
            fluxes[rxn_id] = fv
            if abs(fv) > 1e-10:
                active.append(rxn_id)

        summary = _build_model_summary(model)

        return FBAResult(
            objective_value=float(solution.objective_value),
            status=str(solution.status),
            fluxes=fluxes,
            model_summary=summary,
            active_reactions=tuple(active),
            n_active=len(active),
            n_total=len(model.reactions),
        )
    finally:
        if _mutated:
            model.objective = saved_objective
            model.objective_direction = saved_direction
