"""Main window with sidebar navigation for bioeng-qpcr desktop app."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
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
    "qpcr": "bioeng_qpcr.gui.panels.qpcr_panel.QpcrPanel",
    "enzyme": "bioeng_qpcr.gui.panels.enzyme_panel.EnzymePanel",
    "ferment": "bioeng_qpcr.gui.panels.ferment_panel.FermentPanel",
    "fba": "bioeng_qpcr.gui.panels.fba_panel.FbaPanel",
}

_NAV_ITEMS = [
    ("qpcr", "qPCR 分析", "MIQE 验证 · 相对定量 · 分组比较"),
    ("enzyme", "酶动力学", "Michaelis-Menten · 抑制判别"),
    ("ferment", "发酵动力学", "Monod 拟合 · 可辨识性"),
    ("fba", "FBA 代谢通量", "FBA 求解 · 网络分析"),
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
        runtime_status.setStyleSheet("color: #29845a; padding-right: 10px;")
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
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(hint)
            btn.clicked.connect(lambda checked=False, k=key: self._switch_panel(k))
            self._nav_group.addButton(btn)
            layout.addWidget(btn)

        # Spacer
        layout.addStretch()

        # Footer
        footer = QLabel("仅在本机处理数据\n研究用途 · MIT License")
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
                self._panels[key] = panel
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
        self._stack.setCurrentWidget(self._panels["qpcr"])

    @property
    def panel_load_errors(self) -> dict[str, str]:
        """Return panel import failures detected during application startup."""
        return dict(self._panel_load_errors)

    def _switch_panel(self, key: str) -> None:
        panel = self._panels.get(key)
        if panel is not None:
            self._stack.setCurrentWidget(panel)

    def _on_status(self, message: str) -> None:
        self.statusBar().showMessage(message)


class _ErrorPanel(QWidget):
    """Simple error placeholder panel."""

    def __init__(self, message: str) -> None:
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(message)
        label.setStyleSheet("color: #ef4444; font-size: 16px;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)


def _run_scientific_smoke_checks() -> None:
    """Exercise frozen numerical paths that import successfully but load lazily."""
    import numpy as np

    from bioeng_qpcr.calculations.efficiency import (
        fit_efficiency_from_standard_curve,
    )
    from bioeng_qpcr.domain.entities import Assay, Experiment, Sample, Well
    from bioeng_qpcr.fermentation_kinetics.monod import (
        fit_monod,
        solve_batch_fermentation,
    )
    from bioeng_qpcr.reference_genes.normfinder import normfinder

    log_concentration = np.array([1.0, 2.0, 3.0, 4.0])
    cq = np.array([31.7, 28.2, 25.1, 21.6])
    efficiency = fit_efficiency_from_standard_curve(log_concentration, cq)
    if efficiency.slope_ci is None:
        raise RuntimeError("Standard-curve uncertainty smoke check failed.")

    time = np.linspace(0.0, 12.0, 8)
    biomass, substrate = solve_batch_fermentation(
        time, 0.1, 10.0, 0.5, 2.0, 0.5
    )
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
            gene: Assay(name=gene, efficiency=1.0, is_reference=True)
            for gene in genes
        },
    )
    model = normfinder(experiment)
    if len(model.ranking) != 3 or not np.isfinite(model.common_group_variance):
        raise RuntimeError("NormFinder model smoke check failed.")


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
        _run_scientific_smoke_checks()
        window.close()
        app.quit()
        raise SystemExit(0)

    window.show()
    sys.exit(app.exec())
