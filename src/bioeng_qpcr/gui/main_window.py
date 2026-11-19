"""Main window with sidebar navigation for bioeng-qpcr desktop app."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from bioeng_qpcr import __version__
from bioeng_qpcr.gui.theme import THEME_QSS

# Lazy imports of panels to avoid heavy startup when cobra is absent.
_PANEL_CLASSES: dict[str, str] = {
    "workbench": "bioeng_qpcr.gui.panels.workbench_panel.WorkbenchPanel",
    "qpcr": "bioeng_qpcr.gui.panels.qpcr_panel.QpcrPanel",
    "enzyme": "bioeng_qpcr.gui.panels.enzyme_panel.EnzymePanel",
    "ferment": "bioeng_qpcr.gui.panels.ferment_panel.FermentPanel",
    "fba": "bioeng_qpcr.gui.panels.fba_panel.FbaPanel",
    "potency": "bioeng_qpcr.gui.panels.potency_panel.PotencyPanel",
}

_NAV_ITEMS = [
    ("workbench", "综合实验工作台", "全部实验模块 · 浏览器与桌面入口"),
    ("qpcr", "qPCR 分析", "MIQE 验证 · 相对定量 · 分组比较"),
    ("enzyme", "酶动力学", "Michaelis-Menten · 抑制判别"),
    ("ferment", "发酵动力学", "Monod 拟合 · 可辨识性"),
    ("fba", "FBA 代谢通量", "FBA 求解 · 网络分析"),
    ("potency", "效价分析", "IC50 / EC50 · 剂量-反应 · 模型选择"),
]


class MainWindow(QMainWindow):
    """Main application window with a dark sidebar and stacked content area."""

    def __init__(self) -> None:
        super().__init__()
        self._panel_load_errors: dict[str, str] = {}
        self.setWindowTitle("bioeng-qpcr · Bioengineering Analysis")
        self.resize(1280, 820)
        self.setMinimumSize(1024, 680)

        central = QWidget()
        central.setObjectName("ContentArea")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # --- Sidebar ---
        sidebar = self._build_sidebar()
        root.addWidget(sidebar)

        # --- Content ---
        self._stack = QStackedWidget()
        self._panels: dict[str, QWidget] = {}
        self._load_panels()
        root.addWidget(self._stack, stretch=1)

        # --- Status bar ---
        self.statusBar().showMessage("就绪 · 请选择数据文件开始分析")
        runtime_status = QLabel("● 本地运行")
        runtime_status.setStyleSheet("color: #8ca06f; padding-right: 10px;")
        runtime_status.setToolTip("数据仅在本机处理，不会自动上传")
        self.statusBar().addPermanentWidget(runtime_status)

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _build_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Brand block
        brand = QFrame()
        brand.setObjectName("BrandBlock")
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(16, 18, 16, 18)
        brand_layout.setSpacing(11)

        mark = QLabel("BQ")
        mark.setObjectName("BrandMark")
        mark.setFixedSize(38, 38)
        brand_layout.addWidget(mark)

        brand_text = QVBoxLayout()
        brand_text.setSpacing(1)
        title = QLabel("bioeng-qpcr")
        title.setObjectName("AppTitle")
        version = QLabel(f"Research Suite  ·  v{__version__}")
        version.setObjectName("AppVersion")
        brand_text.addWidget(title)
        brand_text.addWidget(version)
        brand_layout.addLayout(brand_text, stretch=1)
        layout.addWidget(brand)

        # Section label
        section = QLabel("分析模块")
        section.setObjectName("SectionLabel")
        layout.addWidget(section)

        # Nav buttons
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)

        for key, display_name, hint in _NAV_ITEMS:
            btn = QPushButton(display_name)
            btn.setProperty("navButton", True)
            btn.setProperty("panelKey", key)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(hint)
            btn.clicked.connect(lambda checked=False, k=key: self._switch_panel(k))
            self._nav_group.addButton(btn)
            layout.addWidget(btn)

        # Spacer
        layout.addStretch()

        # Footer
        footer = QLabel("仅在本机处理数据\n项目源码 MIT · 组件许可见说明")
        footer.setObjectName("SidebarFooter")
        footer.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(footer)

        return sidebar

    # ------------------------------------------------------------------
    # Panel loading
    # ------------------------------------------------------------------

    def _load_panels(self) -> None:
        """Import and instantiate panels lazily."""
        import importlib

        for key, dotted_path in _PANEL_CLASSES.items():
            module_path, class_name = dotted_path.rsplit(".", 1)
            try:
                mod = importlib.import_module(module_path)
                panel_cls = getattr(mod, class_name)
                panel = panel_cls()
                panel.status_message.connect(self._on_status)
                if key == "workbench":
                    panel.open_module_requested.connect(
                        self._open_desktop_panel, Qt.ConnectionType.QueuedConnection
                    )
                self._panels[key] = panel
                if key in {"enzyme", "ferment", "fba"}:
                    history_button = QPushButton(
                        "进入统一工作台分析与保存历史 →", panel
                    )
                    history_button.setObjectName("SecondaryButton")
                    history_button.setToolTip("此桌面页保持独立计算；统一存档、恢复和备份请使用浏览器工作台")
                    history_button.clicked.connect(
                        lambda checked=False, k=key: (
                            self._panels["workbench"].open_workspace(k)
                        )
                    )
                    panel._header.layout().addWidget(history_button)
                self._stack.addWidget(panel)
            except Exception as exc:
                # Create a fallback panel if import fails (e.g. missing cobra).
                message = f"{type(exc).__name__}: {exc}"
                self._panel_load_errors[key] = message
                fallback = _ErrorPanel(f"模块加载失败: {message}")
                self._panels[key] = fallback
                self._stack.addWidget(fallback)

        # Select the first panel.
        first_btn = self._nav_group.buttons()[0]
        first_btn.setChecked(True)
        self._stack.setCurrentWidget(self._panels["workbench"])

    @property
    def panel_load_errors(self) -> dict[str, str]:
        """Return panel import failures detected during application startup."""
        return dict(self._panel_load_errors)

    def _switch_panel(self, key: str) -> None:
        panel = self._panels.get(key)
        if panel is not None:
            self._stack.setCurrentWidget(panel)
            for button in self._nav_group.buttons():
                button.setChecked(button.property("panelKey") == key)

    @Slot(str)
    def _open_desktop_panel(self, key: str) -> None:
        self._switch_panel(key)
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _on_status(self, message: str) -> None:
        self.statusBar().showMessage(message)


class _ErrorPanel(QWidget):
    """Simple error placeholder panel."""

    def __init__(self, message: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(message)
        label.setStyleSheet("color: #c43a39; font-size: 16px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)


def _run_scientific_smoke_checks() -> None:
    """Exercise frozen numerical paths that import successfully but load lazily."""
    import numpy as np

    from bioeng_qpcr.calculations.efficiency import (
        fit_efficiency_from_standard_curve,
    )
    from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
    from bioeng_qpcr.enzyme_kinetics import compare_inhibition_models
    from bioeng_qpcr.fermentation_kinetics.monod import (
        fit_monod,
        solve_batch_fermentation,
    )
    from bioeng_qpcr.plotting.enzyme_kinetics import plot_global_inhibition
    from bioeng_qpcr.reference_genes.normfinder import normfinder

    substrate_levels = np.linspace(0.1, 15, 12)
    inhibition_data = [
        (
            i,
            substrate_levels,
            10 * substrate_levels / (2 * (1 + i / 3) + substrate_levels),
        )
        for i in (0.0, 1.0, 5.0)
    ]
    comparison = compare_inhibition_models(inhibition_data)
    if comparison.best_type.value != "competitive":
        raise RuntimeError("Global inhibition model smoke check failed.")
    figure, _ = plot_global_inhibition(comparison.best_fit, inhibition_data)
    import matplotlib.pyplot as plt

    plt.close(figure)

    log_concentration = np.array([1.0, 2.0, 3.0, 4.0])
    cq = np.array([31.7, 28.2, 25.1, 21.6])
    efficiency = fit_efficiency_from_standard_curve(log_concentration, cq)
    if efficiency.slope_ci is None:
        raise RuntimeError("Standard-curve uncertainty smoke check failed.")

    time = np.linspace(0.0, 12.0, 8)
    biomass, substrate = solve_batch_fermentation(time, 0.1, 10.0, 0.5, 2.0, 0.5)
    biomass = biomass + np.linspace(0.0, 0.01, biomass.size)
    substrate = np.maximum(substrate + np.linspace(0.0, -0.02, substrate.size), 0.0)
    monod = fit_monod(time, biomass, substrate)
    if monod.mu_max_ci is None:
        raise RuntimeError("Monod uncertainty smoke check failed.")

    genes = {
        "REF1": [20.0, 20.1, 20.0, 20.1],
        "REF2": [21.0, 21.1, 22.0, 22.1],
        "REF3": [19.0, 19.2, 18.9, 19.1],
    }
    samples = ["C1", "C2", "T1", "T2"]
    experiment = Experiment(
        name="packaged-smoke",
        wells=[
            Well(
                position=f"P{gene_index}{sample_index}",
                sample_name=sample,
                target_name=gene,
                cq=values[sample_index],
            )
            for gene_index, (gene, values) in enumerate(genes.items())
            for sample_index, sample in enumerate(samples)
        ],
        samples={
            sample: Sample(
                name=sample,
                group="control" if sample.startswith("C") else "treated",
            )
            for sample in samples
        },
        assays={
            gene: Assay(name=gene, efficiency=1.0, is_reference=True) for gene in genes
        },
    )
    model = normfinder(experiment)
    if len(model.ranking) != 3 or not np.isfinite(model.common_group_variance):
        raise RuntimeError("NormFinder model smoke check failed.")

    # --- Phase 5 modules -------------------------------------------------
    # Each check exercises the frozen import graph *and* one numerical path,
    # because a panel can import cleanly while its science is broken by a
    # packaging mistake (a missing SciPy submodule, a dropped data file).
    from bioeng_qpcr.docking import identify_receptor_ligand, parse_pdb_string
    from bioeng_qpcr.docking.engine import dock
    from bioeng_qpcr.io import preview_csv
    from bioeng_qpcr.io.dose_response_csv import read_dose_response_csv
    from bioeng_qpcr.potency import fit_dose_response
    from bioeng_qpcr.sequence import compare_to_reference, detect_variants

    _run_potency_smoke_check(fit_dose_response, read_dose_response_csv)
    _run_docking_smoke_check(parse_pdb_string, identify_receptor_ligand, dock)
    _run_sequence_smoke_check(compare_to_reference, detect_variants)
    _run_preview_smoke_check(preview_csv)
    _run_workbench_smoke_checks()


def _run_workbench_smoke_checks() -> None:
    """Verify the frozen web UI, SQLite, public data, analysis and plot assets."""
    import io
    import math
    import tempfile
    from pathlib import Path

    from bioeng_qpcr.web.app import create_app

    with tempfile.TemporaryDirectory(prefix="bioeng-workbench-smoke-") as directory:
        web = create_app()
        web.config.update(TESTING=True, WORKBENCH_DB=Path(directory) / "smoke.sqlite3")
        client = web.test_client()
        archived = client.post(
            "/api/preview",
            data={
                "file": (
                    io.BytesIO(b"position,sample_name,target_name,cq\nA1,S1,G,20\n"),
                    "smoke.csv",
                ),
                "_archive": "1",
            },
        ).get_json()
        if not archived.get("record_id"):
            raise RuntimeError("Packaged local analysis archive failed.")
        backup = client.get(
            f"/api/lab/records/{archived['record_id']}/export"
        ).get_json()
        if client.post("/api/lab/records/import", json=backup).status_code != 201:
            raise RuntimeError("Packaged analysis backup restore failed.")
        if client.get("/lab").status_code != 200:
            raise RuntimeError("Packaged lab module directory is unavailable.")
        lab = client.get("/api/lab/modules").get_json()
        if {module["key"] for module in lab["primary_modules"]} != (
            set(_PANEL_CLASSES) - {"workbench"}
        ):
            raise RuntimeError("Packaged lab module directory is incomplete.")
        if client.get("/workbench").status_code != 200:
            raise RuntimeError("Packaged workbench HTML is unavailable.")
        catalog = client.get("/api/qpcr/templates").get_json()
        if len(catalog["experiments"]) != 5 or len(catalog["themes"]) != 6:
            raise RuntimeError("Packaged workbench template catalog is incomplete.")

        def require(response):
            payload = response.get_json()
            if response.status_code not in (200, 201) or not payload.get("ok"):
                raise RuntimeError(f"Workbench smoke request failed: {payload}")
            return payload["project"]

        def execute(project, action):
            return require(
                client.post(
                    f"/api/qpcr/projects/{project['id']}/{action}",
                    json={"revision": project["revision"]},
                )
            )

        for demo in ("relative-ct", "rdml-dilution"):
            project = require(client.post("/api/qpcr/projects", json={"demo": demo}))
            for step in ("samples", "design", "program"):
                project = execute(project, "steps/" + step)
            project = execute(project, "replay")
            for step in ("curves", "qc"):
                project = execute(project, "steps/" + step)
            if demo == "relative-ct":
                settings = {
                    **project["analysis_settings"],
                    "efficiency_confirmed": True,
                }
                project = require(
                    client.patch(
                        f"/api/qpcr/projects/{project['id']}",
                        json={
                            "revision": project["revision"],
                            "changes": {"analysis_settings": settings},
                        },
                    )
                )
            project = execute(project, "steps/analysis")
            project = execute(project, "steps/report")
            analysis = project["outputs"]["analysis"]
            if demo == "relative-ct":
                if not math.isclose(
                    analysis["rows"][1]["fold_change"],
                    5.6372830205680815,
                    rel_tol=1e-12,
                ):
                    raise RuntimeError(
                        "Packaged relative-expression example disagrees."
                    )
            elif analysis["unit"] != "cop" or analysis["standard_curve"]["slope"] >= 0:
                raise RuntimeError("Packaged standard-curve example is invalid.")
            root = f"/api/qpcr/projects/{project['id']}"
            report = client.get(root + "/export/report.html")
            if (
                report.status_code != 200
                or b"data:image/svg+xml;base64," not in report.data
            ):
                raise RuntimeError("Packaged HTML report or embedded figures failed.")
            kind = "relative" if demo == "relative-ct" else "standard"
            figure = client.get(root + f"/figures/{kind}.png")
            if figure.status_code != 200 or not figure.data.startswith(b"\x89PNG"):
                raise RuntimeError("Packaged PNG figure export failed.")
            backup = client.get(root + "/export/project.json").get_json()
            restored = require(client.post("/api/qpcr/projects/import", json=backup))
            if restored["id"] == project["id"] or "analysis" in restored["completed"]:
                raise RuntimeError("Packaged project restore trusted old conclusions.")
            if demo == "relative-ct":
                settings = {
                    **project["analysis_settings"],
                    "relative_method": "efficiency_corrected",
                    "efficiency_confirmed": False,
                    "efficiencies": {
                        a["id"]: {
                            "value": 1.0,
                            "source": "Self-check assumption; not measured evidence",
                        }
                        for a in project["assays"]
                    },
                }
                project = require(
                    client.patch(
                        root,
                        json={
                            "revision": project["revision"],
                            "changes": {"analysis_settings": settings},
                        },
                    )
                )
                project = execute(project, "steps/analysis")
                value = project["outputs"]["analysis"]["rows"][1]["fold_change"]
                if not math.isclose(value, 5.6372830205680815, rel_tol=1e-12):
                    raise RuntimeError(
                        "Efficiency-corrected quantification smoke check failed."
                    )

        rows = ["compound,concentration,response"]
        for concentration in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            rows.append(f"smoke,{concentration},{5 + 95 / (1 + concentration)}")
        payload = client.post(
            "/api/potency",
            data={
                "file": (io.BytesIO("\n".join(rows).encode()), "smoke.csv"),
                "_archive": "1",
            },
        ).get_json()
        record_id = payload.get("record_id")
        if not record_id:
            raise RuntimeError("Potency review record smoke check failed.")
        plots = client.get(f"/api/lab/records/{record_id}/potency/plots?compound=smoke")
        if plots.status_code != 200 or len(plots.get_json()["plots"]) != 2:
            raise RuntimeError("Stored potency curve rendering smoke check failed.")

        for module in ("enzyme", "ferment", "fba"):
            example = client.get(f"/api/workspace/examples/{module}").get_json()
            entry = example["files"][0]
            import base64

            response = client.post(
                f"/api/{module}",
                data={
                    "file": (
                        io.BytesIO(base64.b64decode(entry["data"])), entry["name"]
                    ),
                    "_archive": "1",
                },
            )
            result = response.get_json()
            if response.status_code != 200 or not result.get("record_id"):
                raise RuntimeError(f"{module} workbench/archive smoke check failed.")
            csv = client.get(f"/api/lab/records/{result['record_id']}/results.csv")
            if csv.status_code != 200 or not csv.data:
                raise RuntimeError(f"{module} complete CSV smoke check failed.")


def _run_potency_smoke_check(fit_dose_response, read_dose_response_csv) -> None:
    """Fit a synthetic dose-response curve and require a finite potency."""
    import tempfile
    from pathlib import Path

    import numpy as np

    rows = ["compound,concentration,response"]
    for index in range(10):
        concentration = 10.0 ** (index - 2)
        response = 4.0 + 96.0 / (1.0 + 10.0 ** ((0.5 - (index - 2)) * -1.1))
        rows.append(f"smoke,{concentration:g},{response:.4f}")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.csv"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        datasets = read_dose_response_csv(path)
    if len(datasets) != 1:
        raise RuntimeError("Dose-response smoke check failed to read the file.")
    result = fit_dose_response(datasets[0])
    if not result.best_fit.converged or not np.isfinite(result.relative_value):
        raise RuntimeError("Potency smoke check failed.")


def _run_docking_smoke_check(parse_pdb_string, identify_receptor_ligand, dock) -> None:
    """Dock a two-atom ligand into a two-atom receptor.

    The structures are built as :class:`Atom` objects and serialised through
    the package's own writer rather than pasted in as fixed-column PDB text.
    That keeps the columns correct by construction (they are easy to get wrong
    by hand), and it round-trips the writer through the parser.
    """
    import numpy as np

    from bioeng_qpcr.docking import Atom, Structure, write_pdb_string

    def _atom(serial, name, res_name, res_seq, xyz, element, *, hetatm=False):
        return Atom(
            serial=serial,
            name=name,
            altloc=" ",
            res_name=res_name,
            chain_id="A",
            res_seq=res_seq,
            ins_code=" ",
            xyz=np.array(xyz, dtype=float),
            occupancy=1.0,
            temp_factor=0.0,
            element=element,
            charge="",
            is_hetatm=hetatm,
        )

    receptor = Structure(
        atoms=(
            _atom(1, "CA", "ALA", 1, (10.0, 0.0, 0.0), "C"),
            _atom(2, "CB", "ALA", 1, (11.5, 0.0, 0.0), "C"),
        ),
        residues=(),
        source="<smoke>",
    )
    ligand = Structure(
        atoms=(
            _atom(3, "C1", "LIG", 101, (0.0, 0.0, 0.0), "C", hetatm=True),
            _atom(4, "O1", "LIG", 101, (1.2, 0.0, 0.0), "O", hetatm=True),
        ),
        residues=(),
        source="<smoke>",
    )

    # Round-trip through the writer so the smoke check also covers parsing.
    receptor = parse_pdb_string(write_pdb_string(receptor))
    ligand = parse_pdb_string(write_pdb_string(ligand))
    if receptor.n_atoms != 2 or ligand.n_atoms != 2:
        raise RuntimeError("Docking smoke check failed to parse the PDB input.")
    result = dock(receptor, ligand, n_runs=1, n_steps=40, seed=1, top_n=2)
    if not result.poses or not np.isfinite(result.poses[0].score):
        raise RuntimeError("Docking smoke check produced no scored pose.")
    from bioeng_qpcr.docking import summarize_docking, validate_docking

    checks = validate_docking(result, receptor, ligand)
    sampling = next(check for check in checks if check.rule_id == "DCK-004")
    if sampling.passed and not result.stats.converged:
        raise RuntimeError("Docking search review contradicts search diagnostics.")
    review = summarize_docking(result, checks)
    if review["version"] != 1 or not review["next_steps"]:
        raise RuntimeError("Docking review lacks actionable guidance.")


def _run_sequence_smoke_check(compare_to_reference, detect_variants) -> None:
    """Align a read carrying one substitution and require it to be found.

    The template is deliberately non-repetitive.  On a periodic template the
    position of a substitution is ambiguous (the read can be shifted by a
    whole period), so the check would be asserting the aligner's tie-breaking
    rather than the variant caller.
    """
    reference = "ATGCCGTTAGCGATCACGTGACCTAGGCTAATCGCTAGCATGCGTACAGGCCTAGCTAGCTAGGCTA"
    index = 40
    alternate = "A" if reference[index] != "A" else "T"
    query = reference[:index] + alternate + reference[index + 1 :]
    alignment = compare_to_reference(query, reference)
    variants = detect_variants(alignment, reference)
    if len(variants) != 1 or variants[0].ref_position != index:
        raise RuntimeError(
            f"Variant smoke check expected one substitution at {index}, got "
            f"{[(v.type.value, v.ref_position) for v in variants]}."
        )


def _run_preview_smoke_check(preview_csv) -> None:
    """Preview a tidy qPCR CSV and require the format to be recognised."""
    import tempfile
    from pathlib import Path

    text = "position,sample_name,target_name,cq\nA1,S1,GAPDH,18.5\nA2,S1,IL6,25.1\n"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.csv"
        path.write_text(text, encoding="utf-8")
        preview = preview_csv(path)
    if preview.schema_key != "qpcr_tidy" or preview.errors:
        raise RuntimeError(
            f"CSV preview smoke check failed: schema={preview.schema_key!r}, "
            f"errors={[issue.message for issue in preview.errors]}"
        )


def main() -> None:
    """Application entry point."""
    smoke_test = "--smoke-test" in sys.argv
    if smoke_test:
        # QApplication must not receive application-specific arguments.
        sys.argv = [arg for arg in sys.argv if arg != "--smoke-test"]

    import matplotlib

    matplotlib.use("QtAgg")  # must precede QApplication creation

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("bioeng-qpcr")
    app.setStyleSheet(THEME_QSS)

    # Clean font defaults
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    window = MainWindow()

    if smoke_test:
        # Validate the frozen import graph without opening an interactive
        # window, then execute numerical paths with lazily loaded SciPy code.
        # Packaging automation checks the process exit code.
        app.processEvents()
        if window.panel_load_errors:
            for panel, error in sorted(window.panel_load_errors.items()):
                print(f"{panel}: {error}", file=sys.stderr)
            window.close()
            app.quit()
            raise SystemExit(1)
        exit_code = 0
        try:
            _run_scientific_smoke_checks()
        except Exception:
            # Windowed PyInstaller apps otherwise block on a hidden error dialog.
            import traceback

            if sys.stderr is not None:
                traceback.print_exc(file=sys.stderr)
            exit_code = 1
        finally:
            window.close()
            app.quit()
        raise SystemExit(exit_code)

    window.show()
    sys.exit(app.exec())
