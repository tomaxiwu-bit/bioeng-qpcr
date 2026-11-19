"""PDB parsing, structure model and geometric utilities for docking.

This module provides a dependency-free (standard library + NumPy/SciPy only)
parser and writer for the Protein Data Bank (PDB) ``ATOM``/``HETATM`` fixed
column format, plus the immutable data containers (:class:`Atom`,
:class:`Residue`, :class:`Structure`) used by the docking pipeline, and a small
set of geometric helpers (centroid, inertia tensor, principal-axis alignment,
rigid transforms and RMSD via the Kabsch optimal-superposition algorithm).

PDB fixed-column handling
--------------------------
The venerable PDB record layout is honoured exactly (1-indexed columns):
``serial`` 7-11, ``atom name`` 13-16, ``altLoc`` 17, ``resName`` 18-20,
``chainID`` 22, ``resSeq`` 23-26, ``iCode`` 27, ``x/y/z`` 31-54
(``8.3`` floats), ``occupancy`` 55-60, ``tempFactor`` 61-66,
``element`` 77-78, ``charge`` 79-80.  Lines that are too short are tolerated
by padding with spaces before slicing, so malformed trailing records never
crash the parser.

Element inference
-----------------
When an ``ATOM``/``HETATM`` record omits the element symbol (columns 77-78
blank, common in legacy files) the element is inferred from the atom name and
residue name via :func:`infer_element`.  The inferred value is stored on the
:class:`Atom`, so a round-trip write/read preserves it.

Scoring-convention note
-----------------------
This module is geometry-only.  Binding "goodness" is expressed as a
*lower* (more negative) total score throughout the docking package; see
``scoring.py`` for the energy terms and the explicit sign convention.

The :class:`DockingBox` dataclass describes a rectangular search region
(center + half-extents) used by the search and engine layers.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

# Records that describe point atoms in the PDB format.
_ATOM_RECORDS = ("ATOM", "HETATM")

# Residue names that are water or common small ions and are therefore stripped
# by default from both receptor and ligand.
_WATER_RESIDUES = ("HOH", "WAT", "H2O", "DOD", "TIP", "TP3")
_ION_ELEMENTS = (
    "NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU", "NI", "CO", "CD",
    "HG", "LI", "BR", "F", "I", "SO4", "PO4",
)

# Set of valid element symbols used both for inference validation and for the
# element column writer.
_VALID_ELEMENTS = {
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL",
    "SI", "P", "S", "CL", "AR", "K", "CA", "SC", "TI", "V", "CR", "MN", "FE",
    "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR",
    "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
    "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND", "PM", "SM",
    "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W",
    "RE", "OS", "IR", "PT", "AU", "HG", "TL", "PB", "BI", "PO", "AT", "RN",
}


def _safe_slice(line: str, start: int, stop: int, default: str = "") -> str:
    """Return ``line[start:stop]`` padded to the requested width.

    Short/ragged PDB lines (e.g. a truncated tail) are padded with spaces so
    column slicing never raises ``IndexError``.
    """
    segment = line[start:stop]
    if len(segment) < (stop - start):
        segment = segment + " " * ((stop - start) - len(segment))
    return segment


@dataclass(frozen=True)
class Atom:
    """A single point atom parsed from a PDB ``ATOM``/``HETATM`` record.

    Attributes
        serial:
            Atom serial number (PDB columns 7-11).
        name:
            Atom name, e.g. ``"CA"``, ``"OG"`` (columns 13-16, stripped).
        altloc:
            Alternate conformation indicator (column 17, may be ``" "``).
        res_name:
            Residue/ligand name (columns 18-20, stripped).
        chain_id:
            Chain identifier (column 22, may be ``" "``).
        res_seq:
            Residue sequence number (columns 23-26).
        ins_code:
            Insertion code (column 27, may be ``" "``).
        xyz:
            Cartesian coordinates ``(x, y, z)`` in angstrom, shape ``(3,)``.
        occupancy:
            Occupancy (columns 55-60); ``1.0`` when missing.
        temp_factor:
            Temperature factor / B-factor (columns 61-66); ``0.0`` if absent.
        element:
            Element symbol (columns 77-78) upper-cased; inferred when blank.
        charge:
            Partial charge field (columns 79-80) as a string, ``""`` if blank.
        is_hetatm:
            ``True`` for ``HETATM`` records (ligands, ions, waters).
    """

    serial: int
    name: str
    altloc: str
    res_name: str
    chain_id: str
    res_seq: int
    ins_code: str
    xyz: np.ndarray
    occupancy: float
    temp_factor: float
    element: str
    charge: str
    is_hetatm: bool

    def __post_init__(self) -> None:
        """Coerce ``xyz`` into a float ``(3,)`` array and validate it."""
        vec = np.asarray(self.xyz, dtype=float).reshape(3)
        object.__setattr__(self, "xyz", vec)
        object.__setattr__(self, "element", self.element.strip().upper())


@dataclass(frozen=True)
class Residue:
    """A covalently-bonded group of :class:`Atom` objects.

    Attributes
        res_name:
            Residue/ligand name.
        chain_id:
            Chain identifier.
        res_seq:
            Residue sequence number.
        ins_code:
            Insertion code.
        atoms:
            Tuple of constituent :class:`Atom` objects, ordered as parsed.
    """

    res_name: str
    chain_id: str
    res_seq: int
    ins_code: str
    atoms: tuple[Atom, ...] = field(default_factory=tuple)

    @property
    def coordinates(self) -> np.ndarray:
        """Return the ``(n_atoms, 3)`` coordinate array for this residue."""
        return np.asarray([a.xyz for a in self.atoms], dtype=float)


@dataclass(frozen=True)
class Structure:
    """A parsed molecular structure (receptor, ligand or whole PDB file).

    Attributes
        atoms:
            All atoms in parse order.
        residues:
            All residues in parse order.
        source:
            Origin path or descriptor; ``"<string>"`` for in-memory parsing.
    """

    atoms: tuple[Atom, ...] = field(default_factory=tuple)
    residues: tuple[Residue, ...] = field(default_factory=tuple)
    source: str = "<unknown>"

    @property
    def coordinates(self) -> np.ndarray:
        """Return the ``(n_atoms, 3)`` coordinate array of all atoms."""
        return np.asarray([a.xyz for a in self.atoms], dtype=float)

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the structure."""
        return len(self.atoms)


def infer_element(atom_name: str, res_name: str = "") -> str:
    """Infer an element symbol from an atom/residue name.

    PDB records frequently omit the explicit element column (77-78).  The
    heuristic used here is:

    1. Strip all digits and leading formatting characters from the atom name.
    2. Take the leading 1-2 alphabetic characters.  If the first character is
       a digit (``"1HB"``), the element is the second character.
    3. Map common one-letter/two-letter elements; fall back to a single
       upper-cased character, or ``"C"`` for organic residues lacking any
       recognisable symbol (e.g. a bare carbon).

    Parameters
        atom_name:
            PDB atom name (columns 13-16).
        res_name:
            PDB residue name, used only as a final fallback hint.

    Returns
        Upper-cased element symbol, guaranteed to be a non-empty string.
    """
    name = atom_name.strip()
    if not name:
        hint = res_name.strip()
        return hint[:1].upper() if hint else "C"

    # Drop digits and any non-alphabetic leading character.
    if name[0].isdigit():
        letters = name[1:]
    else:
        letters = name
    letters = "".join(ch for ch in letters if ch.isalpha())

    if not letters:
        return "C"

    # Two-letter elements first (e.g. "FE" -> Fe, "CL" -> Cl).
    candidate = letters[:2].upper()
    if candidate in _VALID_ELEMENTS:
        return candidate
    candidate = letters[0].upper()
    if candidate in _VALID_ELEMENTS:
        return candidate
    # Fallback: best-effort single character; carbon is the safe organic default.
    return candidate if candidate.isalpha() else "C"


def _parse_atom_line(line: str) -> Atom:
    """Parse one ``ATOM``/``HETATM`` line into an :class:`Atom`."""
    record = line[0:6].strip()
    is_hetatm = record == "HETATM"

    serial = int(_safe_slice(line, 6, 11).strip() or "0")
    name = _safe_slice(line, 12, 16).strip()
    altloc = _safe_slice(line, 16, 17).strip()
    res_name = _safe_slice(line, 17, 20).strip()
    chain_id = _safe_slice(line, 21, 22)
    if chain_id == " ":
        chain_id = ""
    res_seq = int(_safe_slice(line, 22, 26).strip() or "0")
    ins_code = _safe_slice(line, 26, 27).strip()

    x = float(_safe_slice(line, 30, 38).strip() or "0.0")
    y = float(_safe_slice(line, 38, 46).strip() or "0.0")
    z = float(_safe_slice(line, 46, 54).strip() or "0.0")

    occ_str = _safe_slice(line, 54, 60).strip()
    temp_str = _safe_slice(line, 60, 66).strip()
    occupancy = float(occ_str) if occ_str else 1.0
    temp_factor = float(temp_str) if temp_str else 0.0

    element = _safe_slice(line, 76, 78).strip()
    if not element:
        element = infer_element(name, res_name)
    element = element.upper()
    charge = _safe_slice(line, 78, 80).strip()

    return Atom(
        serial=serial,
        name=name,
        altloc=altloc,
        res_name=res_name,
        chain_id=chain_id,
        res_seq=res_seq,
        ins_code=ins_code,
        xyz=np.array([x, y, z], dtype=float),
        occupancy=occupancy,
        temp_factor=temp_factor,
        element=element,
        charge=charge,
        is_hetatm=is_hetatm,
    )


def _parse_pdb_lines(
    lines: Iterable[str], *, source: str, max_atoms: int | None
) -> Structure:
    """Parse PDB lines, optionally stopping before an unsafe atom count."""
    if max_atoms is not None and max_atoms < 1:
        raise ValueError("max_atoms must be at least 1.")

    atoms: list[Atom] = []
    residues: list[Residue] = []
    current_key: tuple[str, int, str] | None = None
    current_atoms: list[Atom] = []

    def flush() -> None:
        if current_atoms:
            first = current_atoms[0]
            residues.append(
                Residue(
                    res_name=first.res_name,
                    chain_id=first.chain_id,
                    res_seq=first.res_seq,
                    ins_code=first.ins_code,
                    atoms=tuple(current_atoms),
                )
            )

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        record = line[0:6].strip()
        if record not in _ATOM_RECORDS:
            continue
        if max_atoms is not None and len(atoms) >= max_atoms:
            raise ValueError(f"PDB contains more than {max_atoms} atoms.")
        atom = _parse_atom_line(line)
        atoms.append(atom)
        key = (atom.chain_id, atom.res_seq, atom.ins_code)
        if current_key is None:
            current_key = key
            current_atoms = [atom]
        elif key == current_key:
            current_atoms.append(atom)
        else:
            flush()
            current_key = key
            current_atoms = [atom]

    flush()
    return Structure(atoms=tuple(atoms), residues=tuple(residues), source=source)


def parse_pdb_string(text: str, *, max_atoms: int | None = None) -> Structure:
    """Parse PDB text, optionally rejecting files above *max_atoms*."""
    return _parse_pdb_lines(
        text.splitlines(), source="<string>", max_atoms=max_atoms
    )


def parse_pdb(path: str, *, max_atoms: int | None = None) -> Structure:
    """Parse a PDB file from disk into a :class:`Structure`.

    Parameters
        path:
            Filesystem path to a ``.pdb`` file.
        max_atoms:
            Optional hard atom limit. Parsing stops before allocating an atom
            above the limit, which is useful at untrusted input boundaries.

    Returns
        A :class:`Structure` carrying the source path.

    Raises
        FileNotFoundError:
            If *path* does not exist.
    """
    with open(path, encoding="utf-8") as handle:
        return _parse_pdb_lines(handle, source=str(path), max_atoms=max_atoms)


def _format_float(value: float, width: int, decimals: int) -> str:
    """Format a float right-justified into *width* columns with *decimals*."""
    return f"{value:>{width}.{decimals}f}"


def _format_element(element: str) -> str:
    """Format the element symbol into the 2-column (77-78) right-justified field."""
    return f"{element.upper():>2}"[:2]


def atom_to_pdb_line(atom: Atom, record: str = "ATOM") -> str:
    """Render a single :class:`Atom` as a fixed-column PDB line.

    The line is exactly 80 characters when a charge is present and 78
    otherwise, matching the canonical PDB fixed-column layout used by
    :func:`parse_pdb_string`, so the output round-trips losslessly.

    Parameters
        atom:
            The atom to serialise.
        record:
            ``"ATOM"`` or ``"HETATM"``; chosen from ``atom.is_hetatm`` when
            the supplied value is ``"ATOM"`` but the atom is a HETATM.

    Returns
        An 80- (or 78-) character PDB record string.
    """
    record = "HETATM" if (record == "ATOM" and atom.is_hetatm) else record[:6]
    name = atom.name
    # Atom name placement: 4-char names occupy columns 13-16, shorter names
    # are left-justified starting at column 14 (standard convention).
    if len(name) >= 4:
        name_field = f"{name[:4]:<4}"
    else:
        name_field = f" {name:<3}"

    line = (
        f"{record:<6}"
        f"{atom.serial:>5d}"
        f" "
        f"{name_field}"
        f"{atom.altloc:<1}"
        f"{atom.res_name:>3}"
        f" "
        f"{atom.chain_id or ' ':1}"
        f"{atom.res_seq:>4d}"
        f"{atom.ins_code or ' ':1}"
        f"   "
        f"{_format_float(float(atom.xyz[0]), 8, 3)}"
        f"{_format_float(float(atom.xyz[1]), 8, 3)}"
        f"{_format_float(float(atom.xyz[2]), 8, 3)}"
        f"{_format_float(atom.occupancy, 6, 2)}"
        f"{_format_float(atom.temp_factor, 6, 2)}"
        f"          "
        f"{_format_element(atom.element)}"
        f"{atom.charge:>2}"
    )
    return line


def write_pdb(structure: Structure, path: str) -> None:
    """Write a :class:`Structure` to a PDB file.

    Parameters
        structure:
            The structure to serialise.
        path:
            Destination filesystem path.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(write_pdb_string(structure))
        if not path.endswith("\n"):
            handle.write("\n")


def write_pdb_string(structure: Structure) -> str:
    """Serialise a :class:`Structure` to a PDB-format string.

    Parameters
        structure:
            The structure to serialise.

    Returns
        PDB text with one ``ATOM``/``HETATM`` line per atom plus a trailing
        ``END`` record.
    """
    lines = [atom_to_pdb_line(atom) for atom in structure.atoms]
    lines.append("END")
    return "\n".join(lines) + "\n"


def identify_receptor_ligand(
    structure: Structure,
    ligand_res_name: str | None = None,
    ligand_chain: str | None = None,
    *,
    keep_waters: bool = False,
    keep_ions: bool = False,
) -> tuple[Structure, Structure]:
    """Split a :class:`Structure` into receptor and ligand sub-structures.

    Default policy: the ligand is the first ``HETATM`` residue that is not a
    water or a common ion.  This can be overridden by passing
    *ligand_res_name* and/or *ligand_chain*, in which case the matching
    residue(s) are taken as the ligand and everything else (minus stripped
    waters/ions unless *keep_waters*/*keep_ions*) as the receptor.

    Parameters
        structure:
            Parsed structure containing both receptor and ligand.
        ligand_res_name:
            Explicit residue name that identifies the ligand (e.g. ``"LIG"``).
            When ``None`` the ligand is auto-detected.
        ligand_chain:
            Explicit chain identifier of the ligand.  When ``None`` any chain
            matches.
        keep_waters:
            If ``True`` do not strip water residues from either group.
        keep_ions:
            If ``True`` do not strip common ion residues from either group.

    Returns
        A ``(receptor, ligand)`` tuple of :class:`Structure` objects.

    Raises
        ValueError:
            If no ligand residue can be identified.
    """
    receptor_atoms: list[Atom] = []
    ligand_atoms: list[Atom] = []

    def is_water(atom: Atom) -> bool:
        return atom.res_name in _WATER_RESIDUES

    def is_ion(atom: Atom) -> bool:
        if keep_ions:
            return False
        if atom.res_name in _ION_ELEMENTS:
            return True
        # Single-atom residues whose element is a known monoatomic ion.
        if len(atom.res_name) <= 2 and atom.element in {
            "NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU", "NI",
            "CO", "CD", "HG", "LI", "BR", "F", "I",
        }:
            return True
        return False

    if ligand_res_name is None and ligand_chain is None:
        # Auto-detect: first HETATM residue that is neither water nor ion.
        ligand_atoms = []
        for atom in structure.atoms:
            if atom.is_hetatm and not is_water(atom) and not is_ion(atom):
                ligand_atoms.append(atom)
        if not ligand_atoms:
            raise ValueError("No ligand residue (non-water, non-ion HETATM) found.")
        ligand_keys = {(a.chain_id, a.res_seq, a.ins_code) for a in ligand_atoms}
        for atom in structure.atoms:
            if (atom.chain_id, atom.res_seq, atom.ins_code) in ligand_keys:
                continue
            if is_water(atom) and not keep_waters:
                continue
            if is_ion(atom) and not keep_ions:
                continue
            receptor_atoms.append(atom)
    else:
        for atom in structure.atoms:
            matches = True
            if ligand_res_name is not None and atom.res_name != ligand_res_name:
                matches = False
            if ligand_chain is not None and atom.chain_id != ligand_chain:
                matches = False
            if matches:
                ligand_atoms.append(atom)
            else:
                if is_water(atom) and not keep_waters:
                    continue
                if is_ion(atom) and not keep_ions:
                    continue
                receptor_atoms.append(atom)

    if not ligand_atoms:
        raise ValueError("No ligand atoms matched the given criteria.")

    ligand_residues = _group_atoms(ligand_atoms)
    receptor_residues = _group_atoms(receptor_atoms)
    return (
        Structure(
            atoms=tuple(receptor_atoms),
            residues=tuple(receptor_residues),
            source=structure.source,
        ),
        Structure(
            atoms=tuple(ligand_atoms),
            residues=tuple(ligand_residues),
            source=structure.source,
        ),
    )


def _group_atoms(atoms: list[Atom]) -> list[Residue]:
    """Group an ordered atom list into residues by chain/seq/ins."""
    residues: list[Residue] = []
    current_key: tuple[str, int, str] | None = None
    current: list[Atom] = []
    for atom in atoms:
        key = (atom.chain_id, atom.res_seq, atom.ins_code)
        if current_key is None:
            current_key = key
            current = [atom]
        elif key == current_key:
            current.append(atom)
        else:
            residues.append(_residue_from(current))
            current_key = key
            current = [atom]
    if current:
        residues.append(_residue_from(current))
    return residues


def _residue_from(atoms: list[Atom]) -> Residue:
    """Build a :class:`Residue` from an atom list (assumed one residue)."""
    first = atoms[0]
    return Residue(
        res_name=first.res_name,
        chain_id=first.chain_id,
        res_seq=first.res_seq,
        ins_code=first.ins_code,
        atoms=tuple(atoms),
    )


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def centroid(coords: np.ndarray) -> np.ndarray:
    """Return the centroid (mean position) of *coords* shape ``(N, 3)``."""
    coords = np.asarray(coords, dtype=float)
    if coords.shape[0] == 0:
        return np.zeros(3, dtype=float)
    return np.mean(coords, axis=0)


def inertia_tensor(coords: np.ndarray, masses: np.ndarray | None = None) -> np.ndarray:
    """Compute the mass-weighted inertia tensor of a point set.

    Parameters
        coords:
            Cartesian coordinates, shape ``(N, 3)``.
        masses:
            Per-atom masses, shape ``(N,)``.  Uniform unit masses when
            ``None``.

    Returns
        The symmetric ``3x3`` inertia tensor.
    """
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    if masses is None:
        masses = np.ones(n, dtype=float)
    else:
        masses = np.asarray(masses, dtype=float)
    com = np.average(coords, axis=0, weights=masses)
    shifted = coords - com
    tensor = np.zeros((3, 3), dtype=float)
    for i in range(n):
        r = shifted[i]
        m = masses[i]
        tensor += m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return tensor


def principal_axes(
    coords: np.ndarray, masses: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return principal moments and axes of the inertia tensor.

    Parameters
        coords:
            Cartesian coordinates, shape ``(N, 3)``.
        masses:
            Optional per-atom masses, shape ``(N,)``.

    Returns
        A tuple ``(moments, axes)`` where ``moments`` is the sorted array of
        principal moments (ascending) and ``axes`` is a ``3x3`` matrix whose
        rows are the unit eigenvectors (principal axes) corresponding to
        ``moments``.
    """
    tensor = inertia_tensor(coords, masses)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    order = np.argsort(eigenvalues)
    moments = eigenvalues[order]
    axes = eigenvectors[:, order].T
    return moments, axes


def align_to_principal_axes(coords: np.ndarray) -> np.ndarray:
    """Rotate *coords* so their inertia principal axes align with XYZ.

    The returned coordinates are centered at the origin and rotated by the
    transpose of the principal-axis matrix.  Useful for canonicalising a
    ligand's orientation before docking.

    Parameters
        coords:
            Cartesian coordinates, shape ``(N, 3)``.

    Returns
        Aligned coordinates, shape ``(N, 3)``.
    """
    coords = np.asarray(coords, dtype=float)
    c = centroid(coords)
    shifted = coords - c
    _, axes = principal_axes(shifted)
    return shifted @ axes.T


def rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute the unaligned root-mean-square deviation between two sets.

    Parameters
        coords_a, coords_b:
            Coordinate arrays of identical shape ``(N, 3)``.

    Returns
        The straight RMSD without any optimal superposition.

    Raises
        ValueError:
            If the two arrays do not share the same shape.
    """
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Coordinate shapes differ: {a.shape} vs {b.shape}.")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Optimal-superposition RMSD via the Kabsch algorithm.

    Finds the rigid rotation that best superimposes *coords_b* onto
    *coords_a* (uniform weights) and returns the resulting RMSD.  Both arrays
    must have shape ``(N, 3)``.

    .. note::
       Superposition is what makes this the right measure for comparing two
       *structures* whose relative orientation is arbitrary (for example a
       model against a crystal structure).  It is the **wrong** measure for
       comparing docking poses: two poses related by a pure rotation describe
       congruent point sets, so their Kabsch RMSD is ~0 however different the
       binding modes are.  Use :meth:`~bioeng_qpcr.docking.search.Pose.rmsd_to`
       for pose comparison, which works in the receptor frame.

    Parameters
        coords_a, coords_b:
            Coordinate arrays of identical shape ``(N, 3)``.

    Returns
        The Kabsch-optimal RMSD (>= 0).  Identical/congruent sets yield ~0.

    Raises
        ValueError:
            If the two arrays do not share the same shape.
    """
    a = np.asarray(coords_a, dtype=float)
    b = np.asarray(coords_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Coordinate shapes differ: {a.shape} vs {b.shape}.")

    p0 = centroid(a)
    q0 = centroid(b)
    p = a - p0
    q = b - q0

    # Covariance matrix H = Q^T P (moving^T reference) so that the induced
    # rotation maps the moving set *onto* the reference set.
    h_matrix = q.T @ p
    u, _, vt = np.linalg.svd(h_matrix)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    correction = np.diag([1.0, 1.0, d])
    rotation = vt.T @ correction @ u.T

    q_aligned = q @ rotation.T + p0
    return rmsd(a, q_aligned)


# ---------------------------------------------------------------------------
# Rigid transforms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RigidTransform:
    """A rigid-body transformation: rotation ``R`` plus translation ``t``.

    A point ``x`` is mapped to ``R @ x + t``.  The rotation matrix is stored
    as a frozen ``3x3`` array so transforms compose predictably and can be
    serialised for auditability.

    Attributes
        rotation:
            Orthogonal ``3x3`` rotation matrix.
        translation:
            Translation vector, shape ``(3,)``.
    """

    rotation: np.ndarray
    translation: np.ndarray

    def __post_init__(self) -> None:
        """Coerce inputs into float arrays of the correct shape."""
        r = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        t = np.asarray(self.translation, dtype=float).reshape(3)
        object.__setattr__(self, "rotation", r)
        object.__setattr__(self, "translation", t)

    def apply(self, coords: np.ndarray) -> np.ndarray:
        """Apply this transform to a set of coordinates.

        Parameters
            coords:
                Points to transform, shape ``(N, 3)``.

        Returns
            Transformed points, shape ``(N, 3)``.
        """
        coords = np.asarray(coords, dtype=float)
        return coords @ self.rotation.T + self.translation


def transform_from_quaternion(
    quaternion: np.ndarray, translation: np.ndarray
) -> RigidTransform:
    """Build a :class:`RigidTransform` from a unit quaternion and translation.

    Quaternion is ``(w, x, y, z)`` (scalar-last convention is *not* used here;
    scalar-first ``w`` is expected).  It is normalised internally.

    Parameters
        quaternion:
            Unit quaternion ``(w, x, y, z)``.
        translation:
            Translation vector, shape ``(3,)``.

    Returns
        The corresponding :class:`RigidTransform`.
    """
    q = np.asarray(quaternion, dtype=float).reshape(4)
    norm = np.linalg.norm(q)
    if norm == 0:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = q / norm
    w, x, y, z = q
    rotation = np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=float,
    )
    return RigidTransform(
        rotation=rotation, translation=np.asarray(translation, dtype=float)
    )


# ---------------------------------------------------------------------------
# Docking box
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DockingBox:
    """Axis-aligned rectangular search region centred on ``center``.

    Attributes
        center:
            Box centre, shape ``(3,)`` (angstrom).
        half_size:
            Half-extent along each axis, shape ``(3,)`` (angstrom).  The full
            box spans ``center ± half_size``.
    """

    center: np.ndarray
    half_size: np.ndarray

    def __post_init__(self) -> None:
        """Coerce inputs into float ``(3,)`` arrays."""
        c = np.asarray(self.center, dtype=float).reshape(3)
        h = np.asarray(self.half_size, dtype=float).reshape(3)
        object.__setattr__(self, "center", c)
        object.__setattr__(self, "half_size", np.abs(h))

    @property
    def extent(self) -> np.ndarray:
        """Full side lengths ``(2 * half_size)`` along each axis."""
        return 2.0 * self.half_size

    def random_point(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a uniformly random point inside the box.

        Parameters
            rng:
                Seeded NumPy random generator.

        Returns
            A random point, shape ``(3,)``.
        """
        u = rng.random(3)
        return self.center + (u - 0.5) * self.extent

    def contains(self, point: np.ndarray, tol: float = 0.0) -> bool:
        """Return ``True`` if *point* lies inside the box (with *tol* slack)."""
        p = np.asarray(point, dtype=float).reshape(3)
        return bool(np.all(np.abs(p - self.center) <= self.half_size + tol))

    def contains_all(self, coords: np.ndarray, tol: float = 0.0) -> bool:
        """Return ``True`` if every row of *coords* lies inside the box."""
        coords = np.asarray(coords, dtype=float)
        if coords.shape[0] == 0:
            return True
        local = np.abs(coords - self.center)
        return bool(np.all(local <= self.half_size + tol))


def make_box_from_structures(
    receptor: Structure,
    ligand: Structure,
    *,
    padding: float = 8.0,
    ligand_center: np.ndarray | None = None,
) -> DockingBox:
    """Construct a :class:`DockingBox` around the receptor/ligand system.

    The box is centered between the receptor centroid and (optionally) an
    explicit ligand reference position, with half-extents equal to the
    receptor's radius plus *padding* on every axis.  This guarantees the
    search volume encloses the receptor and a margin for ligand placement.

    Parameters
        receptor:
            Receptor structure.
        ligand:
            Ligand structure (used to estimate its extent).
        padding:
            Extra half-extent (angstrom) added beyond the receptor radius.
        ligand_center:
            Optional explicit reference centre for the ligand (e.g. its
            original position).  When omitted the ligand centroid is used.

    Returns
        A :class:`DockingBox` enclosing the receptor with margin.
    """
    rec_coords = receptor.coordinates
    rec_center = centroid(rec_coords)
    if rec_coords.shape[0]:
        rec_radius = float(
            np.max(np.linalg.norm(rec_coords - rec_center, axis=1))
        )
    else:
        rec_radius = 0.0
    if ligand_center is None:
        ligand_center = centroid(ligand.coordinates)
    center = 0.5 * (rec_center + ligand_center)
    half = rec_radius + padding
    # Add the ligand's own extent so it fits when centered at the box centre.
    lig_coords = ligand.coordinates
    if lig_coords.shape[0]:
        lig_radius = float(
            np.max(np.linalg.norm(lig_coords - centroid(lig_coords), axis=1))
        )
        half = max(half, lig_radius + padding * 0.5)
    return DockingBox(center=center, half_size=np.full(3, half))


__all__ = [
    "Atom",
    "Residue",
    "Structure",
    "RigidTransform",
    "DockingBox",
    "infer_element",
    "parse_pdb",
    "parse_pdb_string",
    "write_pdb",
    "write_pdb_string",
    "atom_to_pdb_line",
    "identify_receptor_ligand",
    "centroid",
    "inertia_tensor",
    "principal_axes",
    "align_to_principal_axes",
    "rmsd",
    "kabsch_rmsd",
    "transform_from_quaternion",
    "make_box_from_structures",
]
