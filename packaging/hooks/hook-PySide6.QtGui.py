"""Collect Qt GUI without unused PDF and virtual-keyboard plugin dependencies."""

from pathlib import Path

from PyInstaller.utils.hooks.qt import add_qt6_dependencies

hiddenimports, binaries, datas = add_qt6_dependencies(__file__)
# These optional plugins pull in QtPdf, QtQuick/QML and GPL-only keyboard code.
# This application uses QWidget/QPainter; image previews are handled by Pillow.
binaries = [
    (source, target)
    for source, target in binaries
    if Path(source).name.lower() not in {"qpdf.dll", "qtvirtualkeyboardplugin.dll"}
]
