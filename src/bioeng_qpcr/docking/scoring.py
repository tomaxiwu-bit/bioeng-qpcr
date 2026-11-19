"""Empirical scoring functions for rigid-body molecular docking.

The docking package treats binding "goodness" as a single scalar *energy-like*
score.  **Lower (more negative) scores mean better predicted binding.**  This
sign convention is used consistently by every layer (search, engine, and
validation) and is documented here so downstream code never has to guess.

The :class:`ScoringFunction` combines five physically-motivated empirical
terms.  Every term is individually weighted by :class:`ScoringWeights`, and
the per-term contributions are always returned alongside the total in a
:class:`ScoreBreakdown` for full auditability:

========================  ================================================
Term                      Physical motivation
========================  ================================================
``vdw``                   Lennard-Jones 12-6 repulsion/attraction with a
                          soft core so two atoms can never diverge to
                          infinity when overlapping.
``electrostatic``         Coulomb interaction with a distance-dependent
                          dielectric (``eps = dielectric * r``) so that
                          charge-charge terms decay softly with distance.
``hbond``                 Donor-H ... acceptor hydrogen-bond potential with
                          an explicit angle term (donor-H-acceptor ~ 180°).
``hydrophobic``           Carbon-carbon contact well rewarding buried
                          hydrophobic surface.
``desolvation``           Penalty for polar atoms left unsatisfied (no
                          complementary polar partner within contact) —
                          a crude buried-area / solvation proxy.
========================  ================================================

All parameters (vdW radii, element partial charges, dielectric constant,
hydrogen-bond geometry, weights) live on the :class:`ScoringFunction` and
:class:`ScoringWeights` objects, so they are fully configurable and
reproducible.  **These are simplified empirical potentials, not a full
force field** (no AMBER/CHARMM terms, no explicit solvation model).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Cut-off distances (angstrom) beyond which pairwise terms are neglected.
_VDW_CUTOFF: float = 12.0
_ELEC_CUTOFF: float = 18.0
_HBOND_CUTOFF: float = 4.0
_HYDROPHOBIC_CUTOFF: float = 6.0
_DESOLV_CUTOFF: float = 5.0

# Soft-core regularisation to keep the LJ potential finite at r -> 0.
_SOFTCORE: float = 0.5

# Hydrogen-bond geometry defaults.
_HBOND_D0: float = 2.0  # ideal H...acceptor distance (angstrom)
_HBOND_W: float = 0.5  # gaussian width
_HBOND_STRENGTH: float = 2.0  # maximal favourable energy (kcal/mol-like)
_HBOND_ANGLE_MIN: float = 0.0  # cos(angle) threshold; 0 -> 90°

# Hydrophobic contact well.
_HYDROPHOBIC_D0: float = 4.2
_HYDROPHOBIC_W: float = 1.2
_HYDROPHOBIC_STRENGTH: float = 0.6

# Desolvation penalty per unsatisfied polar atom.
_DESOLV_PENALTY: float = 1.5

# Per-element Lennard-Jones well depth (epsilon, arbitrary kcal/mol-like) and
# minimum-energy distance for a homonuclear pair (r_min, angstrom).
_ELEMENT_VDW: dict[str, tuple[float, float]] = {
    "H": (0.020, 2.40),
    "C": (0.105, 3.40),
    "N": (0.160, 3.25),
    "O": (0.210, 3.05),
    "S": (0.250, 3.50),
    "P": (0.200, 3.70),
    "F": (0.080, 2.95),
    "CL": (0.276, 3.45),
    "BR": (0.340, 3.60),
    "I": (0.400, 3.80),
    "NA": (0.050, 2.60),
    "MG": (0.050, 2.60),
    "CA": (0.050, 2.80),
    "ZN": (0.050, 2.70),
    "FE": (0.050, 2.70),
}

# Per-element approximate partial charge (e, arbitrary sign/magnitude).
_ELEMENT_CHARGE: dict[str, float] = {
    "H": 0.15,
    "C": 0.00,
    "N": -0.30,
    "O": -0.50,
    "S": -0.20,
    "P": 0.40,
    "F": -0.15,
    "CL": -0.15,
    "BR": -0.15,
    "I": -0.15,
    "NA": 0.50,
    "MG": 0.60,
    "CA": 0.60,
}

_DEFAULT_EPS: float = 0.10
_DEFAULT_RMIN: float = 3.40


def _vdw_params(element: str) -> tuple[float, float]:
    """Return ``(epsilon, r_min)`` for an element, falling back to carbon."""
    return _ELEMENT_VDW.get(element.upper(), (_DEFAULT_EPS, _DEFAULT_RMIN))


def _pair_sigma(elt_a: str, elt_b: str) -> tuple[float, float]:
    """Return ``(sigma, epsilon)`` for a pair using Lorentz-Berthelot mixing."""
    eps_a, rmin_a = _vdw_params(elt_a)
    eps_b, rmin_b = _vdw_params(elt_b)
    sigma = 0.5 * (rmin_a + rmin_b)
    epsilon = float(np.sqrt(eps_a * eps_b))
    return sigma, epsilon


@dataclass(frozen=True)
class ScoringWeights:
    """Relative weights of each scoring term in the total energy.

    Attributes
        vdw:
            Weight of the Lennard-Jones term.
        electrostatic:
            Weight of the Coulomb term.
        hbond:
            Weight of the hydrogen-bond term.
        hydrophobic:
            Weight of the hydrophobic-contact term.
        desolvation:
            Weight of the unsatisfied-polar desolvation penalty.
    """

    vdw: float = 1.0
    electrostatic: float = 0.5
    hbond: float = 1.0
    hydrophobic: float = 0.8
    desolvation: float = 0.5


@dataclass(frozen=True)
class ScoreBreakdown:
    """Per-term and total score of a docked pose.

    The total is always the weighted sum of the five terms; the field is
    provided precomputed so callers can assert the additivity invariant
    directly (auditability requirement).

    Attributes
        vdw:
            Lennard-Jones contribution (weighted).
        electrostatic:
            Coulomb contribution (weighted).
        hbond:
            Hydrogen-bond contribution (weighted).
        hydrophobic:
            Hydrophobic-contact contribution (weighted).
        desolvation:
            Unsatisfied-polar desolvation penalty (weighted).
        total:
            Weighted sum; **lower is better**.
    """

    vdw: float
    electrostatic: float
    hbond: float
    hydrophobic: float
    desolvation: float
    total: float


@dataclass(frozen=True)
class AtomTable:
    """Precomputed per-atom arrays used by the scoring function.

    Building these once (per structure) and reusing them across poses is what
    makes evaluating thousands of rigid transformations affordable.

    Attributes
        coords:
            Atomic coordinates, shape ``(N, 3)``.  For a moving ligand this
            array is updated for every pose.
        elements:
            Element symbol per atom.
        charges:
            Partial charge per atom (derived from element).
        is_acceptor:
            ``True`` for atoms that can accept a hydrogen bond (O, N).
        is_donor_h:
            ``True`` for hydrogen atoms that are donors (bound to O/N).
        donor_vec:
            For donor-H atoms, the unit vector pointing from the H to its
            donor heavy atom; zero elsewhere.  Shape ``(N, 3)``.
        is_hydrophobic:
            ``True`` for non-polar carbon atoms.
        is_polar:
            ``True`` for polar atoms (O, N) used by the desolvation term.
    """

    coords: np.ndarray
    elements: tuple[str, ...]
    charges: np.ndarray
    is_acceptor: np.ndarray
    is_donor_h: np.ndarray
    donor_vec: np.ndarray
    is_hydrophobic: np.ndarray
    is_polar: np.ndarray


def build_atom_table(
    structure_or_coords: object,
    elements: tuple[str, ...] | None = None,
) -> AtomTable:
    """Build an :class:`AtomTable` from a :class:`Structure` or raw arrays.

    When called with a :class:`~bioeng_qpcr.docking.structure.Structure`, the
    element list and coordinates are taken directly.  When called with a raw
    coordinate array, *elements* must be supplied.

    Parameters
        structure_or_coords:
            A :class:`Structure` or an ``(N, 3)`` coordinate array.
        elements:
            Required element tuple when *structure_or_coords* is an array.

    Returns
        The assembled :class:`AtomTable`.

    Raises
        ValueError:
            If raw arrays are supplied without *elements*.
    """
    from bioeng_qpcr.docking.structure import Structure

    if isinstance(structure_or_coords, Structure):
        structure = structure_or_coords
        coords = structure.coordinates
        elements = tuple(a.element for a in structure.atoms)
    else:
        coords = np.asarray(structure_or_coords, dtype=float)
        if elements is None:
            raise ValueError("elements must be provided when passing raw coords.")
        if coords.shape[0] != len(elements):
            raise ValueError("coords and elements length mismatch.")

    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    charges = np.array(
        [_ELEMENT_CHARGE.get(e.upper(), 0.0) for e in elements], dtype=float
    )
    is_acceptor = np.array(
        [e.upper() in ("O", "N") for e in elements], dtype=bool
    )
    is_hydrophobic = np.array(
        [e.upper() == "C" for e in elements], dtype=bool
    )
    is_polar = np.array(
        [e.upper() in ("O", "N") for e in elements], dtype=bool
    )

    # Identify donor hydrogens and their donor heavy-atom direction.
    is_donor_h = np.zeros(n, dtype=bool)
    donor_vec = np.zeros((n, 3), dtype=float)
    # Heavy-atom index lookup for intra-molecular donor detection.
    heavy_idx = [i for i, e in enumerate(elements) if e.upper() != "H"]
    for i, e in enumerate(elements):
        if e.upper() == "H":
            pos = coords[i]
            best_j = -1
            best_d = float("inf")
            for j in heavy_idx:
                d = float(np.linalg.norm(coords[j] - pos))
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0 and best_d < 1.6 and elements[best_j].upper() in ("O", "N"):
                is_donor_h[i] = True
                vec = coords[best_j] - pos
                norm = np.linalg.norm(vec)
                if norm > 0:
                    donor_vec[i] = vec / norm

    return AtomTable(
        coords=coords,
        elements=elements,
        charges=charges,
        is_acceptor=is_acceptor,
        is_donor_h=is_donor_h,
        donor_vec=donor_vec,
        is_hydrophobic=is_hydrophobic,
        is_polar=is_polar,
    )


def transformed_table(table: AtomTable, transform: object) -> AtomTable:
    """Return a copy of *table* with coordinates transformed by *transform*.

    Parameters
        table:
            The source :class:`AtomTable` (e.g. a ligand).
        transform:
            A :class:`~bioeng_qpcr.docking.structure.RigidTransform`.

    Returns
        A new :class:`AtomTable` with updated coordinates; all chemical flags
        are reused unchanged.
    """
    new_coords = transform.apply(table.coords)
    return AtomTable(
        coords=new_coords,
        elements=table.elements,
        charges=table.charges,
        is_acceptor=table.is_acceptor,
        is_donor_h=table.is_donor_h,
        donor_vec=table.donor_vec @ transform.rotation.T,
        is_hydrophobic=table.is_hydrophobic,
        is_polar=table.is_polar,
    )


@dataclass(frozen=True)
class ScoringFunction:
    """Composable empirical scoring function for rigid-body docking.

    Attributes
        weights:
            Term weights (see :class:`ScoringWeights`).
        dielectric:
            Distance-independent prefactor ``k`` in the distance-dependent
            dielectric ``eps = k * r``.  Larger values screen electrostatics.
        vdw_cutoff, elec_cutoff, hbond_cutoff, hydrophobic_cutoff,
        desolv_cutoff:
            Pairwise cut-offs (angstrom) beyond which each term is zeroed.
    """

    weights: ScoringWeights = field(default_factory=ScoringWeights)
    dielectric: float = 4.0
    vdw_cutoff: float = _VDW_CUTOFF
    elec_cutoff: float = _ELEC_CUTOFF
    hbond_cutoff: float = _HBOND_CUTOFF
    hydrophobic_cutoff: float = _HYDROPHOBIC_CUTOFF
    desolv_cutoff: float = _DESOLV_CUTOFF

    # -- individual terms -------------------------------------------------

    def _vdw_term(self, table_a: AtomTable, table_b: AtomTable) -> float:
        """Lennard-Jones 12-6 term with soft core, summed over pairs."""
        ca = table_a.coords
        cb = table_b.coords
        ea = np.array([_pair_sigma(e, e)[0] for e in table_a.elements])
        eb = np.array([_pair_sigma(e, e)[0] for e in table_b.elements])
        diff = ca[:, None, :] - cb[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        # sigma_ij via Lorentz-Berthelot on r_min.
        sigma = 0.5 * (ea[:, None] + eb[None, :])
        dist2_safe = dist2 + _SOFTCORE**2
        inv = (sigma**2) / dist2_safe
        # LJ: 4*eps*(inv^6 - inv^3) using sigma^2/r^2 as (sigma/r)^2.
        eps = np.sqrt(
            np.array([_pair_sigma(e, e)[1] for e in table_a.elements])[:, None]
            * np.array([_pair_sigma(e, e)[1] for e in table_b.elements])[None, :]
        )
        lj = 4.0 * eps * (inv**6 - inv**3)
        mask = dist2 < self.vdw_cutoff**2
        return float(np.sum(lj * mask))

    def _elec_term(self, table_a: AtomTable, table_b: AtomTable) -> float:
        """Coulomb term with distance-dependent dielectric ``eps = k * r``."""
        ca = table_a.coords
        cb = table_b.coords
        qa = table_a.charges
        qb = table_b.charges
        diff = ca[:, None, :] - cb[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2) + 0.25)
        # E = q_i q_j / (k * r^2)  (distance-dependent dielectric).
        denom = self.dielectric * dist * dist
        energy = (qa[:, None] * qb[None, :]) / denom
        mask = dist < self.elec_cutoff
        return float(np.sum(energy * mask))

    def _hbond_term(self, table_a: AtomTable, table_b: AtomTable) -> float:
        """Hydrogen-bond term: donor-H(a) ... acceptor(b) and vice versa."""
        total = 0.0
        total += self._hbond_one_way(table_a, table_b)
        total += self._hbond_one_way(table_b, table_a)
        return total

    def _hbond_one_way(
        self, donor_table: AtomTable, acceptor_table: AtomTable
    ) -> float:
        """Favourable H-bond energy from donor-table donors to acceptor atoms."""
        d_idx = np.where(donor_table.is_donor_h)[0]
        a_idx = np.where(acceptor_table.is_acceptor)[0]
        if d_idx.size == 0 or a_idx.size == 0:
            return 0.0
        hd = donor_table.coords[d_idx]  # donor H positions
        ha = acceptor_table.coords[a_idx]  # acceptor positions
        dv = donor_table.donor_vec[d_idx]  # H -> donor heavy dir

        diff = hd[:, None, :] - ha[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        dist = np.sqrt(dist2)

        # Gaussian distance preference around d0.
        gauss = np.exp(-((dist - _HBOND_D0) ** 2) / (2.0 * _HBOND_W**2))
        # Angle term: the donor-H-acceptor angle should be ~180 deg, i.e. the
        # H->acceptor and H->donor unit vectors are anti-parallel (cos ~ -1).
        vec_ha = -diff  # H -> acceptor
        norms = np.linalg.norm(vec_ha, axis=2, keepdims=True)
        unit_ha = np.divide(vec_ha, norms, out=np.zeros_like(vec_ha), where=norms > 0)
        cosang = np.sum(unit_ha * dv[:, None, :], axis=2)
        # 1 at linear (cosang = -1), 0 at 90 deg (cosang = 0).
        angle_factor = 0.5 * (1.0 - cosang)

        energy = -_HBOND_STRENGTH * gauss * angle_factor
        mask = (dist < self.hbond_cutoff) & (dist > 0.5)
        return float(np.sum(energy * mask))

    def _hydrophobic_term(self, table_a: AtomTable, table_b: AtomTable) -> float:
        """Carbon-carbon contact well around ``hydrophobic_d0``."""
        ca = table_a.coords
        cb = table_b.coords
        is_ha = table_a.is_hydrophobic
        is_hb = table_b.is_hydrophobic
        diff = ca[:, None, :] - cb[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        d0 = _HYDROPHOBIC_D0
        w = _HYDROPHOBIC_W
        # Parabolic well clipped to a finite basin.
        basin = np.abs(dist - d0) < w * 2.5
        well = 1.0 - ((dist - d0) / w) ** 2
        well = np.clip(well, 0.0, 1.0)
        contact = -_HYDROPHOBIC_STRENGTH * well
        mask = basin & is_ha[:, None] & is_hb[None, :]
        return float(np.sum(contact * mask))

    def _desolvation_term(self, table_a: AtomTable, table_b: AtomTable) -> float:
        """Penalty for polar atoms with no complementary polar partner.

        For each polar atom in the ligand (table_b by convention) that lacks
        any polar neighbour in table_a within the desolvation cut-off, add a
        fixed penalty.  This rewards burying/satisfying polar groups.
        """
        polar_b = np.where(table_b.is_polar)[0]
        if polar_b.size == 0:
            return 0.0
        ca = table_a.coords
        cb = table_b.coords
        diff = cb[polar_b][:, None, :] - ca[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        if not np.any(table_a.is_polar):
            # No polar atoms on the receptor side to satisfy the ligand.
            return float(polar_b.size * _DESOLV_PENALTY)
        polar_rec_cols = np.where(table_a.is_polar)[0]
        min_polar = np.min(dist[:, polar_rec_cols], axis=1)
        unsatisfied = min_polar >= self.desolv_cutoff
        return float(np.sum(unsatisfied) * _DESOLV_PENALTY)

    # -- aggregate --------------------------------------------------------

    def _raw_terms(self, rec: AtomTable, lig: AtomTable) -> ScoreBreakdown:
        """Compute weighted per-term energies between receptor and ligand."""
        vdw = self._vdw_term(rec, lig)
        elec = self._elec_term(rec, lig)
        hb = self._hbond_term(rec, lig)
        hp = self._hydrophobic_term(rec, lig)
        des = self._desolvation_term(rec, lig)
        return ScoreBreakdown(
            vdw=self.weights.vdw * vdw,
            electrostatic=self.weights.electrostatic * elec,
            hbond=self.weights.hbond * hb,
            hydrophobic=self.weights.hydrophobic * hp,
            desolvation=self.weights.desolvation * des,
            total=(
                self.weights.vdw * vdw
                + self.weights.electrostatic * elec
                + self.weights.hbond * hb
                + self.weights.hydrophobic * hp
                + self.weights.desolvation * des
            ),
        )

    def score_table(self, receptor: AtomTable, ligand: AtomTable) -> ScoreBreakdown:
        """Score a (transformed) ligand against the receptor.

        Parameters
            receptor:
                Fixed :class:`AtomTable` of the receptor.
            ligand:
                :class:`AtomTable` of the ligand, usually transformed.

        Returns
            A :class:`ScoreBreakdown` (lower total = better).
        """
        return self._raw_terms(receptor, ligand)


def score_pose(
    receptor: object,
    ligand_coords: np.ndarray,
    ligand_elements: tuple[str, ...],
    scoring: ScoringFunction | None = None,
) -> ScoreBreakdown:
    """Convenience wrapper: score *ligand_coords* against a *receptor*.

    Parameters
        receptor:
            A :class:`Structure` or :class:`AtomTable` for the receptor.
        ligand_coords:
            Transformed ligand coordinates, shape ``(N, 3)``.
        ligand_elements:
            Element symbols of the ligand atoms.
        scoring:
            Scoring function to use; a default instance is created when
            ``None``.

    Returns
        A :class:`ScoreBreakdown`.
    """
    if scoring is None:
        scoring = ScoringFunction()
    if not isinstance(receptor, AtomTable):
        rec_table = build_atom_table(receptor)
    else:
        rec_table = receptor
    lig_table = build_atom_table(ligand_coords, ligand_elements)
    return scoring.score_table(rec_table, lig_table)


__all__ = [
    "ScoringWeights",
    "ScoreBreakdown",
    "AtomTable",
    "ScoringFunction",
    "build_atom_table",
    "transformed_table",
    "score_pose",
]
