"""Desktop entry to the same local Flask workbench, without a second UI engine."""

from __future__ import annotations

import threading

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget


class WorkbenchPanel(QWidget):
    status_message = Signal(str)
    open_module_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._server = None
        self._url = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 36, 40, 36)
        title = QLabel("综合实验工作台")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        description = QLabel(
            "从统一首页选择实验类型，各模块保留独立的分析工具。\n\n"
            "01  qPCR：设计、采集、质控、定量与报告\n"
            "02  酶动力学：参数拟合与抑制判别\n"
            "03  发酵动力学：Monod 拟合与可辨识性\n"
            "04  FBA：代谢通量求解与网络检查\n"
            "05  效价分析：IC50 / EC50 与剂量–反应\n"
            "\n文件预览位于辅助工具；旧版对接与序列工具保留兼容入口。\n\n"
            "在默认浏览器打开；服务只监听本机地址，项目保存到本地 SQLite。\n"
            "首页可直接切换到桌面模块；也可使用左侧导航。"
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        button = QPushButton("打开综合实验工作台 →")
        button.setObjectName("PrimaryButton")
        button.clicked.connect(self._open_workbench)
        layout.addWidget(button)
        self._message = QLabel("")
        self._message.setWordWrap(True)
        layout.addWidget(self._message)
        layout.addStretch()
        QApplication.instance().aboutToQuit.connect(self._stop_server)

    def _open_workbench(self):
        self._open_path("/lab")

    def open_workspace(self, key):
        if key in {"enzyme", "ferment", "fba"}:
            self._open_path(f"/workspace#{key}")

    def _open_path(self, path):
        try:
            if self._server is None:
                from werkzeug.serving import make_server

                from bioeng_qpcr.web.app import create_app

                # Port zero lets the OS reserve a free loopback port atomically.
                web = create_app()
                web.config["DESKTOP_OPEN_MODULE"] = self.open_module_requested.emit
                self._server = make_server("127.0.0.1", 0, web)
                self._url = f"http://127.0.0.1:{self._server.server_port}"
                threading.Thread(target=self._server.serve_forever, daemon=True).start()
            url = self._url + path
            if not QDesktopServices.openUrl(QUrl(url)):
                raise RuntimeError(f"无法启动默认浏览器，请手动打开 {url}")
            self._message.setText(url)
            self.status_message.emit("实验工作台已在本机浏览器打开")
        except Exception as exc:
            self._message.setText(f"工作台启动失败：{exc}")

    def _stop_server(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
