"""Auditable validation rules for docking results (DCK-00x).

Every rule produces a :class:`ValidationResult` carrying a stable ``rule_id``
of the form ``DCK-001`` (Docking ChecK), a boolean ``passed`` flag, a textual
``severity`` (``"error"`` / ``"warning"`` / ``"info"``), a human-readable
``message``, and a free-form ``evidence`` dict for the audit trail.

The rules cover the failure modes that matter for a *rigid* docking pipeline:

* **DCK-001** -- the search box covers the ligand's original position, so the
  native-like pose was actually reachable.
* **DCK-002** -- the top-ranked pose keeps the ligand inside the box (no
  fly-out during minimisation).
* **DCK-003** -- no severe atomic clash (overlap) in the best pose.
* **DCK-004** -- search diagnostics agree with the search engine; score gaps
  describe the ranking but do not establish convergence or a global optimum.
* **DCK-005** -- ligand not so large relative to the box that sampling is
  hopelessly incomplete.
* **DCK-006** -- the receptor presents an empty binding pocket (free volume
  inside the box), rather than a fully packed cavity.
* **DCK-007** -- receptor and ligand were both identified with at least one
  atom.
* **DCK-008** -- the best pose has at least some favourable (attractive)
  character; a purely repulsive top score signals a failed docking.
* **DCK-009** -- rigid docking preserved every ligand interatomic distance.
* **DCK-010** -- the top pose forms a receptor-ligand contact rather than
  floating in solvent.

Rules are pure functions of a :class:`DockResult` plus optional ligand
metadata (the original, un-docked ligand coordinates), so they can be unit
tested in isolation.  :func:`validate_docking` runs them all and returns the
ordered list of results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bioeng_qpcr.docking.engine import DockResult
from bioeng_qpcr.docking.structure import (
    Structure,
    centroid,
)

# Threshold parameters (all in angstrom / score units).
_CLASH_DISTANCE: float = 1.2
_BOX_SLACK: float = 1.0
_POCKET_OCCUPANCY_MAX: float = 0.85
_LIGAND_FRACTION_MAX: float = 0.6
_RIGID_DISTANCE_TOLERANCE: float = 1e-5
_CONTACT_DISTANCE: float = 5.0


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of a single docking validation rule.

    Attributes
        rule_id:
            Stable rule identifier, e.g. ``"DCK-001"``.
        passed:
            Whether the structure satisfied the rule.
        severity:
            ``"error"``, ``"warning"`` or ``"info"``.
        message:
            Human-readable explanation of the outcome.
        evidence:
            Structured supporting numbers/lists for the audit trail.
    """

    rule_id: str
    passed: bool
    severity: str
    message: str
    evidence: dict = field(default_factory=dict)


def _min_interatomic_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Return the minimum pairwise distance between two coordinate sets."""
    if coords_a.shape[0] == 0 or coords_b.shape[0] == 0:
        return float("inf")
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(np.min(dist2)))


# ---------------------------------------------------------------------------
# Individual rules
# ---------------------------------------------------------------------------


def check_box_covers_ligand(
    result: DockResult, original_ligand: Structure
) -> ValidationResult:
    """DCK-001: search box must cover the ligand's original position."""
    lig_center = centroid(original_ligand.coordinates)
    inside = result.box.contains(lig_center, tol=_BOX_SLACK)
    return ValidationResult(
        rule_id="DCK-001",
        passed=inside,
        severity="error",
        message=(
            "Search box covers the ligand's original position."
            if inside
            else "Search box does NOT cover the ligand's original position; "
            "native-like poses may be unreachable."
        ),
        evidence={
            "ligand_center": lig_center.tolist(),
            "box_center": result.box.center.tolist(),
            "box_half_size": result.box.half_size.tolist(),
        },
    )


def check_top_pose_in_box(result: DockResult) -> ValidationResult:
    """DCK-002: the best pose keeps the ligand inside the box."""
    if not result.poses:
        return ValidationResult(
            rule_id="DCK-002",
            passed=False,
            severity="error",
            message="No poses were produced; cannot verify box containment.",
            evidence={},
        )
    top = result.poses[0]
    inside = result.box.contains_all(top.ligand_coords, tol=_BOX_SLACK)
    return ValidationResult(
        rule_id="DCK-002",
        passed=inside,
        severity="error",
        message=(
            "Top pose ligand is contained within the search box."
            if inside
            else "Top pose ligand escapes the search box (fly-out)."
        ),
        evidence={"n_atoms": int(top.ligand_coords.shape[0])},
    )


def check_no_clash(
    result: DockResult,
    receptor: Structure,
    original_ligand: Structure | None = None,
) -> ValidationResult:
    """DCK-003: no severe atomic overlap in the best pose.

    The top pose's ligand coordinates are tested against the receptor atoms.
    A minimum heavy-atom separation below ``_CLASH_DISTANCE`` indicates a
    steric clash that a physically meaningful pose should not have.
    """
    if not result.poses:
        return ValidationResult(
            rule_id="DCK-003",
            passed=False,
            severity="error",
            message="No poses produced; clash check is not assessable.",
            evidence={},
        )
    rec_coords = receptor.coordinates
    lig_coords = result.poses[0].ligand_coords
    min_d = _min_interatomic_distance(lig_coords, rec_coords)
    no_clash = min_d >= _CLASH_DISTANCE
    return ValidationResult(
        rule_id="DCK-003",
        passed=no_clash,
        severity="error",
        message=(
            f"Min ligand-receptor distance {min_d:.2f} A >= "
            f"clash threshold {_CLASH_DISTANCE:.2f} A."
            if no_clash
            else f"Steric clash: min distance {min_d:.2f} A < "
            f"{_CLASH_DISTANCE:.2f} A in the top pose."
        ),
        evidence={"min_distance": min_d, "threshold": _CLASH_DISTANCE},
    )


def check_sampling_adequacy(result: DockResult) -> ValidationResult:
    """DCK-004: use search diagnostics, never the gap between two poses."""
    grid = result.stats.seed is None
    evidence = {
        "search_method": "grid" if grid else "monte_carlo",
        "search_diagnostic_passed": result.stats.converged,
        "n_poses": len(result.poses),
    }
    if len(result.poses) >= 2:
        evidence["gap"] = result.poses[1].score - result.poses[0].score
    if not result.stats.converged:
        return ValidationResult(
            rule_id="DCK-004",
            passed=False,
            severity="warning",
            message=(
                "网格枚举未完成；请重新运行。"
                if grid
                else "至少一次搜索的末段最优评分仍在改善；请增加每次运行步数，"
                "并比较不同随机种子的结果。评分差不能证明收敛。"
            ),
            evidence=evidence,
        )
    if len(result.poses) < 2:
        return ValidationResult(
            rule_id="DCK-004",
            passed=False,
            severity="warning",
            message="不足两个独立候选姿势，无法比较排名；请检查输出数量与去重设置。",
            evidence=evidence,
        )
    return ValidationResult(
        rule_id="DCK-004",
        passed=True,
        severity="warning",
        message=(
            "本次有限网格枚举已完成；不代表网格足够细或已找到全局最优。"
            if grid
            else "各次搜索末段暂未观察到进一步改善；这是启发式稳定诊断，"
            "不证明全局最优，也不证明评分排名具有生物学意义。"
        ),
        evidence=evidence,
    )


def summarize_docking(result: DockResult, checks) -> dict:
    """One review conclusion shared by browser exports and the desktop panel."""
    checks = tuple(checks)
    failed = [check for check in checks if not check.passed]
    grid = result.stats.seed is None
    search_label = (
        ("网格枚举完成" if result.stats.converged else "网格枚举未完成")
        if grid
        else ("末段暂稳定" if result.stats.converged else "末段仍在改善")
    )
    if any(check.severity == "error" for check in failed) or not result.poses:
        status, label = "blocked", "存在阻断问题"
    elif not checks:
        status, label = "not_audited", "未执行审计"
    elif failed or not result.stats.converged:
        status, label = "review_required", "需要复核"
    else:
        status, label = "exploratory", "仅供探索"
    actions = []
    if not result.stats.converged:
        actions.append(
            "重新完成网格枚举。"
            if grid
            else "增加每次运行步数，并以不同随机种子重复搜索，比较最佳姿势和评分。"
        )
    if not checks:
        actions.append("启用 DCK 审计规则后重新运行；当前结果未经结构与搜索检查。")
    remedies = {
        "DCK-001": "核对搜索盒中心和范围，确认覆盖预期结合位点。",
        "DCK-002": "检查最佳姿势是否超出搜索盒，修正搜索范围后重算。",
        "DCK-003": "检查结构准备和原子重叠，处理严重碰撞后重算。",
        "DCK-004": "候选不足时增加输出姿势数，并检查 RMSD 去重阈值。",
        "DCK-005": "核对配体尺寸与搜索盒范围；增大范围后同时评估搜索预算。",
        "DCK-006": "检查搜索盒是否定位到真实结合位点；体积占有率只是粗略诊断。",
        "DCK-007": "核对受体与配体文件，确认结构解析完整。",
        "DCK-008": "检查结构准备与评分项；正负评分本身不能证明结合或不结合。",
        "DCK-009": "检查配体原子顺序和几何结构，修正后重新运行。",
        "DCK-010": "检查配体是否与预期结合位点接触，排除游离于溶剂中的姿势。",
    }
    for check in failed:
        if check.rule_id == "DCK-004" and len(result.poses) >= 2:
            continue
        if check.rule_id in remedies:
            actions.append(remedies[check.rule_id])
    actions.append(
        "用已知结合结构或活性对照做独立验证；简化刚性评分不能作为实测亲和力，"
        "也不能单独决定购买或淘汰化合物。"
    )
    return {
        "version": 1,
        "status": status,
        "label": label,
        "summary": f"计算已完成；{label}。搜索诊断：{search_label}。",
        "search_label": search_label,
        "reasons": [f"{check.rule_id}：{check.message}" for check in failed],
        "next_steps": actions,
        "interpretation": "搜索稳定或规则通过均不等于已验证的结合能力。",
    }


def check_ligand_not_too_large(
    result: DockResult, original_ligand: Structure
) -> ValidationResult:
    """DCK-005: ligand extent should not exceed a fraction of the box.

    A ligand whose diameter is a large fraction of the box implies the search
    space per orientation is tiny relative to the ligand, so a rigid scan is
    unlikely to find a good pose.  This is a warning, not a hard error.
    """
    lig_coords = original_ligand.coordinates
    if lig_coords.shape[0] == 0:
        return ValidationResult(
            rule_id="DCK-005",
            passed=False,
            severity="warning",
            message="Ligand has no atoms; cannot assess size.",
            evidence={},
        )
    c = centroid(lig_coords)
    lig_radius = float(np.max(np.linalg.norm(lig_coords - c, axis=1)))
    box_radius = float(np.min(result.box.half_size))
    fraction = lig_radius / box_radius if box_radius > 0 else float("inf")
    ok = fraction <= _LIGAND_FRACTION_MAX
    return ValidationResult(
        rule_id="DCK-005",
        passed=ok,
        severity="warning",
        message=(
            f"Ligand radius {lig_radius:.2f} A is {fraction:.0%} of box half "
            f"size (acceptable)."
            if ok
            else f"Ligand radius {lig_radius:.2f} A is {fraction:.0%} of box "
            f"half size; sampling likely incomplete."
        ),
        evidence={
            "ligand_radius": lig_radius,
            "box_half_size": box_radius,
            "fraction": fraction,
        },
    )


def check_binding_pocket(result: DockResult, receptor: Structure) -> ValidationResult:
    """DCK-006: receptor leaves an empty pocket inside the search box.

    A coarse Monte Carlo sampling of points within the box counts how many fall
    within a van-der-Waals probe distance of any receptor atom.  If nearly all
    box volume is occupied the receptor has no room for a ligand (no pocket);
    if almost none is occupied the box is mostly solvent (also suspicious).
    """
    rng = np.random.default_rng(0)
    n_samples = 400
    half = result.box.half_size
    center = result.box.center
    rec_coords = receptor.coordinates
    samples = center + (rng.random((n_samples, 3)) - 0.5) * 2.0 * half
    if rec_coords.shape[0] == 0:
        occupancy = 0.0
    else:
        # Distance from each sample to nearest receptor atom.
        diff = samples[:, None, :] - rec_coords[None, :, :]
        min_d = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        probe = 2.0  # van-der-Waals-ish probe radius
        occupancy = float(np.mean(min_d < probe))
    # Healthy pocket: some free volume but not a near-empty box.
    ok = occupancy < _POCKET_OCCUPANCY_MAX and occupancy > 0.05
    severity = "warning" if not ok else "info"
    return ValidationResult(
        rule_id="DCK-006",
        passed=ok,
        severity=severity,
        message=(
            f"盒内粗略占有率 {occupancy:.0%} 在启发式范围内；不能据此确认真实结合口袋。"
            if ok
            else f"盒内粗略占有率 {occupancy:.0%} 偏低或偏高；请核对结合位点和搜索盒，"
            "该体积指标不能单独判断口袋是否存在。"
        ),
        evidence={"occupancy": occupancy, "probe_radius": 2.0},
    )


def check_partners_identified(
    result: DockResult, receptor: Structure, original_ligand: Structure
) -> ValidationResult:
    """DCK-007: receptor and ligand both carry at least one atom."""
    ok = receptor.n_atoms > 0 and original_ligand.n_atoms > 0
    return ValidationResult(
        rule_id="DCK-007",
        passed=ok,
        severity="error",
        message=(
            "Receptor and ligand were both identified with >= 1 atom."
            if ok
            else "Receptor or ligand is empty; partner identification failed."
        ),
        evidence={
            "receptor_atoms": receptor.n_atoms,
            "ligand_atoms": original_ligand.n_atoms,
        },
    )


def check_favourable_top_pose(result: DockResult) -> ValidationResult:
    """DCK-008: top pose has at least some attractive character.

    A top score that is purely positive (repulsive) means no favourable
    interaction was found — the docking effectively failed to bind the ligand.
    """
    if not result.poses:
        return ValidationResult(
            rule_id="DCK-008",
            passed=False,
            severity="warning",
            message="No poses produced; cannot assess pose favourability.",
            evidence={},
        )
    top_score = result.poses[0].score
    # Define "some attraction" as a non-trivially positive fraction of the
    # attractive terms (vdw + hbond + hydrophobic) being net negative.
    breakdown = result.poses[0].breakdown
    attractive = breakdown.vdw + breakdown.hbond + breakdown.hydrophobic
    ok = attractive < 0.0 or top_score < 0.0
    return ValidationResult(
        rule_id="DCK-008",
        passed=ok,
        severity="warning",
        message=(
            f"最佳姿势的简化评分含吸引项（总分 {top_score:.3f}）；不能据此证明结合。"
            if ok
            else f"最佳姿势的简化评分未见吸引项（总分 {top_score:.3f}）；"
            "请核对结构与评分设置，不能据此证明不结合。"
        ),
        evidence={"top_score": top_score, "attractive_terms": attractive},
    )


def check_rigid_geometry_preserved(
    result: DockResult, original_ligand: Structure
) -> ValidationResult:
    """DCK-009: the reported rigid pose must preserve ligand geometry."""
    if not result.poses:
        return ValidationResult(
            rule_id="DCK-009",
            passed=False,
            severity="error",
            message="No poses produced; ligand geometry is not assessable.",
        )
    original = original_ligand.coordinates
    posed = result.poses[0].ligand_coords
    if original.shape != posed.shape:
        return ValidationResult(
            rule_id="DCK-009",
            passed=False,
            severity="error",
            message="Top pose atom count does not match the input ligand.",
            evidence={
                "input_shape": list(original.shape),
                "pose_shape": list(posed.shape),
            },
        )
    original_distances = np.linalg.norm(
        original[:, None, :] - original[None, :, :], axis=2
    )
    posed_distances = np.linalg.norm(posed[:, None, :] - posed[None, :, :], axis=2)
    max_delta = float(np.max(np.abs(original_distances - posed_distances)))
    passed = max_delta <= _RIGID_DISTANCE_TOLERANCE
    return ValidationResult(
        rule_id="DCK-009",
        passed=passed,
        severity="error",
        message=(
            f"Rigid ligand geometry preserved (max distance change {max_delta:.2g} A)."
            if passed
            else f"Ligand geometry changed by up to {max_delta:.3g} A in a rigid run."
        ),
        evidence={
            "max_pairwise_distance_change": max_delta,
            "tolerance": _RIGID_DISTANCE_TOLERANCE,
        },
    )


def check_interface_contact(
    result: DockResult, receptor: Structure, original_ligand: Structure
) -> ValidationResult:
    """DCK-010: the top pose must contact the receptor with heavy atoms."""
    if not result.poses:
        return ValidationResult(
            rule_id="DCK-010",
            passed=False,
            severity="warning",
            message="No poses produced; interface contact is not assessable.",
        )
    receptor_mask = np.array([atom.element != "H" for atom in receptor.atoms])
    ligand_mask = np.array([atom.element != "H" for atom in original_ligand.atoms])
    receptor_coords = receptor.coordinates[receptor_mask]
    ligand_coords = result.poses[0].ligand_coords[ligand_mask]
    min_distance = _min_interatomic_distance(ligand_coords, receptor_coords)
    passed = min_distance <= _CONTACT_DISTANCE
    return ValidationResult(
        rule_id="DCK-010",
        passed=passed,
        severity="warning",
        message=(
            f"Top pose contacts the receptor at {min_distance:.2f} A."
            if passed
            else f"Top pose is {min_distance:.2f} A from the receptor; "
            "no interface contact."
        ),
        evidence={
            "min_heavy_atom_distance": min_distance,
            "threshold": _CONTACT_DISTANCE,
        },
    )


def validate_docking(
    result: DockResult,
    receptor: Structure,
    ligand: Structure,
) -> tuple[ValidationResult, ...]:
    """Run every DCK-00x rule against a docking result.

    Parameters
        result:
            The :class:`DockResult` to validate.
        receptor:
            The receptor :class:`Structure` (original, un-docked).
        ligand:
            The ligand :class:`Structure` (original, un-docked).

    Returns
        A tuple of :class:`ValidationResult` in stable ``DCK-00x`` order.
    """
    rules = [
        check_box_covers_ligand(result, ligand),
        check_top_pose_in_box(result),
        check_no_clash(result, receptor, ligand),
        check_sampling_adequacy(result),
        check_ligand_not_too_large(result, ligand),
        check_binding_pocket(result, receptor),
        check_partners_identified(result, receptor, ligand),
        check_favourable_top_pose(result),
        check_rigid_geometry_preserved(result, ligand),
        check_interface_contact(result, receptor, ligand),
    ]
    return tuple(rules)


__all__ = [
    "ValidationResult",
    "check_box_covers_ligand",
    "check_top_pose_in_box",
    "check_no_clash",
    "check_sampling_adequacy",
    "summarize_docking",
    "check_ligand_not_too_large",
    "check_binding_pocket",
    "check_partners_identified",
    "check_favourable_top_pose",
    "check_rigid_geometry_preserved",
    "check_interface_contact",
    "validate_docking",
]
