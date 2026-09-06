"""QSS theme for the bioeng-qpcr desktop application.

Warm Editorial Science palette — adapted from the Claude Design Library.
Replaces the cold blue/navy scheme with warm terracotta accents on a
cream canvas, evoking an academic-journal sensibility while keeping
the interface clean and professional for scientific work.
"""

THEME_QSS = """
/* ===== Foundation ===== */
QWidget {
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 13px;
    color: #1b1b19;
}

QMainWindow, QDialog, #ContentArea {
    background: #faf9f5;
}

QToolTip {
    background: #1b1b19;
    color: #faf9f5;
    border: none;
    border-radius: 6px;
    padding: 7px 9px;
}

/* ===== Sidebar (warm cream instead of dark navy) ===== */
#Sidebar {
    background: #f5f4ee;
    border: none;
    border-right: 1px solid #e3e0d4;
    min-width: 224px;
    max-width: 224px;
}

#BrandBlock {
    background: transparent;
    border-bottom: 1px solid #e3e0d4;
}

#BrandMark {
    background: #d97757;
    color: #faf9f5;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 800;
    qproperty-alignment: AlignCenter;
}

#AppTitle {
    color: #1b1b19;
    font-size: 15px;
    font-weight: 700;
}

#AppVersion, #SidebarMeta {
    color: #908e84;
    font-size: 11px;
}

#SectionLabel {
    color: #6e6d68;
    font-size: 10px;
    font-weight: 700;
    padding: 18px 18px 8px 18px;
}

QPushButton[navButton="true"] {
    min-height: 44px;
    text-align: left;
    padding: 0px 16px 0px 18px;
    margin: 2px 10px;
    border: none;
    border-radius: 9px;
    background: transparent;
    color: #3d3d3a;
    font-size: 13px;
    font-weight: 500;
}

QPushButton[navButton="true"]:hover {
    background: #e9e6dc;
    color: #1b1b19;
}

QPushButton[navButton="true"]:checked {
    background: #e9e6dc;
    color: #c96442;
    font-weight: 650;
}

#SidebarFooter {
    background: #e9e6dc;
    border: 1px solid #e3e0d4;
    border-radius: 10px;
    margin: 12px;
    padding: 10px;
    color: #6e6d68;
    font-size: 11px;
}

/* ===== Panel header ===== */
#PanelEyebrow {
    color: #c96442;
    font-size: 10px;
    font-weight: 800;
}

#PanelTitle {
    color: #1b1b19;
    font-size: 26px;
    font-weight: 500;
    font-family: "Georgia", "Microsoft YaHei UI", serif;
    letter-spacing: -0.5px;
}

#SectionHint {
    color: #6e6d68;
    font-size: 12px;
}

/* ===== Cards ===== */
QFrame[card="true"] {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #d8d6cd;
}

QLabel#CardTitle {
    font-size: 13px;
    font-weight: 700;
    color: #2c2c2b;
}

QLabel#FieldLabel {
    color: #6e6d68;
    font-size: 12px;
    font-weight: 650;
}

/* ===== Inputs ===== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 34px;
    padding: 0px 11px;
    border: 1px solid #d8d6cd;
    border-radius: 8px;
    background: #fbfcfe;
    color: #1b1b19;
    selection-background-color: #d97757;
    selection-color: #faf9f5;
}

QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #b7b5a9;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    background: #ffffff;
    border: 1px solid #d97757;
}

QLineEdit:read-only {
    color: #908e84;
    background: #f5f4ee;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #6e6d68;
    margin-right: 9px;
}

QComboBox QAbstractItemView {
    border: 1px solid #d8d6cd;
    border-radius: 8px;
    background: #ffffff;
    selection-background-color: #f8e3d8;
    selection-color: #b05730;
    padding: 5px;
    outline: none;
}

QCheckBox {
    color: #46443b;
    spacing: 7px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #b7b5a9;
    border-radius: 4px;
    background: #ffffff;
}

QCheckBox::indicator:checked {
    background: #d97757;
    border-color: #d97757;
}

/* ===== Buttons ===== */
QPushButton {
    min-height: 36px;
    padding: 0px 16px;
    border-radius: 8px;
    border: 1px solid transparent;
    font-weight: 600;
}

QPushButton#PrimaryButton {
    background: #d97757;
    color: #faf9f5;
}

QPushButton#PrimaryButton:hover {
    background: #c96442;
}

QPushButton#PrimaryButton:pressed {
    background: #b05730;
}

QPushButton#PrimaryButton:disabled {
    background: #f8e3d8;
    color: #e8a98f;
}

QPushButton#SecondaryButton {
    background: #e9e6dc;
    color: #46443b;
    border-color: #d8d6cd;
}

QPushButton#SecondaryButton:hover {
    background: #e3e0d4;
    border-color: #b7b5a9;
}

QPushButton#QuietButton {
    background: transparent;
    color: #46443b;
    border-color: #d8d6cd;
}

QPushButton#QuietButton:hover {
    background: #f5f4ee;
}

/* ===== Results ===== */
QTextEdit#ResultsText {
    background: #faf9f5;
    border: none;
    border-radius: 8px;
    padding: 14px;
    font-family: "Cascadia Mono", "Consolas", "Microsoft YaHei UI", monospace;
    font-size: 12px;
    color: #2c2c2b;
    selection-background-color: #f8e3d8;
}

QSplitter::handle {
    background: transparent;
}

QSplitter::handle:hover {
    background: #e3e0d4;
}

/* ===== Tables ===== */
QTableWidget {
    border: 1px solid #d8d6cd;
    border-radius: 8px;
    background: #ffffff;
    gridline-color: #edf1f6;
    selection-background-color: #f8e3d8;
    selection-color: #b05730;
}

QHeaderView::section {
    background: #f5f4ee;
    color: #6e6d68;
    font-weight: 650;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #d8d6cd;
}

/* ===== Scrollbars ===== */
QScrollBar:vertical {
    background: transparent;
    width: 7px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #b7b5a9;
    border-radius: 3px;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover {
    background: #908e84;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: transparent;
    height: 7px;
}

QScrollBar::handle:horizontal {
    background: #b7b5a9;
    border-radius: 3px;
    min-width: 28px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ===== Status bar ===== */
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #d8d6cd;
    color: #6e6d68;
    font-size: 11px;
    padding-left: 8px;
}

QStatusBar::item {
    border: none;
}

/* ===== Miscellaneous ===== */
QGroupBox {
    border: 1px solid #d8d6cd;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 650;
    color: #6e6d68;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0px 4px;
}

QProgressBar {
    border: none;
    border-radius: 4px;
    background: #e3e0d4;
    height: 6px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: #d97757;
    border-radius: 4px;
}
"""
