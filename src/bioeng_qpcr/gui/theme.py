"""QSS theme for the bioeng-qpcr desktop application."""

THEME_QSS = """
/* ===== Foundation ===== */
QWidget {
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    font-size: 13px;
    color: #172033;
}

QMainWindow, QDialog, #ContentArea {
    background: #f4f7fb;
}

QToolTip {
    background: #172033;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 7px 9px;
}

/* ===== Sidebar ===== */
#Sidebar {
    background: #101827;
    border: none;
    min-width: 224px;
    max-width: 224px;
}

#BrandBlock {
    background: transparent;
    border-bottom: 1px solid #253149;
}

#BrandMark {
    background: #2f7df4;
    color: #ffffff;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 800;
    qproperty-alignment: AlignCenter;
}

#AppTitle {
    color: #f8fafc;
    font-size: 15px;
    font-weight: 700;
}

#AppVersion, #SidebarMeta {
    color: #8491a8;
    font-size: 11px;
}

#SectionLabel {
    color: #71809b;
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
    color: #aeb9cc;
    font-size: 13px;
    font-weight: 500;
}

QPushButton[navButton="true"]:hover {
    background: #1b2940;
    color: #f8fafc;
}

QPushButton[navButton="true"]:checked {
    background: #203a61;
    color: #75b6ff;
    font-weight: 650;
}

#SidebarFooter {
    background: #172237;
    border: 1px solid #253149;
    border-radius: 10px;
    margin: 12px;
    padding: 10px;
    color: #91a0b8;
    font-size: 11px;
}

/* ===== Panel header ===== */
#PanelEyebrow {
    color: #2f7df4;
    font-size: 10px;
    font-weight: 800;
}

#PanelTitle {
    color: #14213a;
    font-size: 24px;
    font-weight: 750;
}

#SectionHint {
    color: #69778f;
    font-size: 12px;
}

/* ===== Cards ===== */
QFrame[card="true"] {
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #dfe6f0;
}

QLabel#CardTitle {
    font-size: 13px;
    font-weight: 700;
    color: #26344d;
}

QLabel#FieldLabel {
    color: #53627a;
    font-size: 12px;
    font-weight: 650;
}

/* ===== Inputs ===== */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    min-height: 34px;
    padding: 0px 11px;
    border: 1px solid #cfd8e6;
    border-radius: 8px;
    background: #fbfcfe;
    color: #172033;
    selection-background-color: #2f7df4;
    selection-color: #ffffff;
}

QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {
    border-color: #9db0ca;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    background: #ffffff;
    border: 1px solid #2f7df4;
}

QLineEdit:read-only {
    color: #607089;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #6b7a90;
    margin-right: 9px;
}

QComboBox QAbstractItemView {
    border: 1px solid #cfd8e6;
    border-radius: 8px;
    background: #ffffff;
    selection-background-color: #e8f2ff;
    selection-color: #1d5fae;
    padding: 5px;
    outline: none;
}

QCheckBox {
    color: #596981;
    spacing: 7px;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #b9c5d6;
    border-radius: 4px;
    background: #ffffff;
}

QCheckBox::indicator:checked {
    background: #2f7df4;
    border-color: #2f7df4;
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
    background: #2f7df4;
    color: #ffffff;
}

QPushButton#PrimaryButton:hover {
    background: #246ddd;
}

QPushButton#PrimaryButton:pressed {
    background: #1e5bbd;
}

QPushButton#PrimaryButton:disabled {
    background: #b8d2f5;
    color: #edf5ff;
}

QPushButton#SecondaryButton {
    background: #eef3f9;
    color: #3f5069;
    border-color: #d8e1ec;
}

QPushButton#SecondaryButton:hover {
    background: #e3ebf5;
    border-color: #c4d0df;
}

QPushButton#QuietButton {
    background: transparent;
    color: #4d6080;
    border-color: #d8e1ec;
}

QPushButton#QuietButton:hover {
    background: #f4f7fb;
}

/* ===== Results ===== */
QTextEdit#ResultsText {
    background: #f8fafd;
    border: none;
    border-radius: 8px;
    padding: 14px;
    font-family: "Cascadia Mono", "Consolas", "Microsoft YaHei UI", monospace;
    font-size: 12px;
    color: #26344d;
    selection-background-color: #d9eaff;
}

QSplitter::handle {
    background: transparent;
}

QSplitter::handle:hover {
    background: #dce6f2;
}

/* ===== Tables ===== */
QTableWidget {
    border: 1px solid #dfe6f0;
    border-radius: 8px;
    background: #ffffff;
    gridline-color: #edf1f6;
    selection-background-color: #e8f2ff;
    selection-color: #1d5fae;
}

QHeaderView::section {
    background: #f4f7fb;
    color: #52627b;
    font-weight: 650;
    padding: 8px;
    border: none;
    border-bottom: 1px solid #dfe6f0;
}

/* ===== Scrollbars ===== */
QScrollBar:vertical {
    background: transparent;
    width: 7px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #bec9d8;
    border-radius: 3px;
    min-height: 28px;
}

QScrollBar::handle:vertical:hover {
    background: #96a5b9;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background: transparent;
    height: 7px;
}

QScrollBar::handle:horizontal {
    background: #bec9d8;
    border-radius: 3px;
    min-width: 28px;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* ===== Status bar ===== */
QStatusBar {
    background: #ffffff;
    border-top: 1px solid #dfe6f0;
    color: #68778e;
    font-size: 11px;
    padding-left: 8px;
}

QStatusBar::item {
    border: none;
}

/* ===== Miscellaneous ===== */
QGroupBox {
    border: 1px solid #dfe6f0;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 8px;
    font-weight: 650;
    color: #52627b;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0px 4px;
}

QProgressBar {
    border: none;
    border-radius: 4px;
    background: #e2e9f2;
    height: 6px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: #2f7df4;
    border-radius: 4px;
}
"""
