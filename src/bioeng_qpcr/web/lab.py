"""Shared module directory and a narrow, Qt-signal-backed desktop handoff."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template, request

blueprint = Blueprint("lab", __name__)

MODULES = (
    {
        "key": "qpcr",
        "name": "qPCR 实验",
        "tag": "完整工作流",
        "surface": "web",
        "description": "样品与引物、配液板位、上机程序、采集、质控与定量报告。",
        "flow": "设计 → 采集 → 质控 → 定量 → 报告",
        "href": "/workbench",
        "icon": "qP",
    },
    {
        "key": "enzyme",
        "name": "酶动力学",
        "tag": "工作台 + 桌面",
        "surface": "web",
        "description": "Michaelis–Menten 与全局抑制比较、残差、历史与备份。",
        "flow": "反应速率 → 模型比较 → 审阅 → 存档",
        "href": "/workspace#enzyme",
        "desktop": True,
        "icon": "En",
    },
    {
        "key": "ferment",
        "name": "发酵动力学",
        "tag": "工作台 + 桌面",
        "surface": "web",
        "description": "Monod 模型、生长与底物曲线、参数可辨识性和置信区间。",
        "flow": "导入时间序列 → Monod 拟合 → 参数审阅",
        "href": "/workspace#ferment",
        "desktop": True,
        "icon": "Fm",
    },
    {
        "key": "fba",
        "name": "FBA 代谢通量",
        "tag": "工作台 + 桌面",
        "surface": "web",
        "description": "代谢模型求解、通量分布、阻断反应与死端代谢物检查。",
        "flow": "载入模型 → 基线 / 情景 → 求解 → 存档",
        "href": "/workspace#fba",
        "desktop": True,
        "icon": "Fx",
    },
    {
        "key": "potency",
        "name": "效价分析",
        "tag": "浏览器分析",
        "surface": "web",
        "description": "剂量–反应拟合，IC50 / EC50、模型比较与效价审计。",
        "flow": "导入剂量–反应 → 拟合 → 质控 → 图表",
        "href": "/workspace#potency",
        "icon": "IC",
    },
    {
        "key": "docking",
        "name": "分子对接",
        "tag": "浏览器分析",
        "surface": "web",
        "description": "刚性对接、姿势搜索、能量分解与可审阅的候选结果。",
        "flow": "受体 / 配体 → 搜索 → 姿势排序 → 导出",
        "href": "/workspace#docking",
        "icon": "Dk",
    },
    {
        "key": "sequence",
        "name": "测序比对",
        "tag": "浏览器分析",
        "surface": "web",
        "description": "序列与读长比对、变异检测、共识序列及质量检查。",
        "flow": "读长 / 参考序列 → 比对 → 变异与质控",
        "href": "/workspace#sequence",
        "icon": "AT",
    },
    {
        "key": "preview",
        "name": "数据与图片预览",
        "tag": "通用工具",
        "surface": "web",
        "description": "CSV 导入规范与数据表检查，多格式图片、伪彩和对比度预览。",
        "flow": "选择文件 → 结构检查 / 图像预览 → 导入",
        "href": "/workspace#preview_csv",
        "icon": "Tb",
    },
)
DESKTOP_MODULES = {m["key"] for m in MODULES if m.get("desktop")}
AUXILIARY_KEYS = {"docking", "sequence", "preview"}
PRIMARY_MODULES = tuple(m for m in MODULES if m["key"] not in AUXILIARY_KEYS)
AUXILIARY_MODULES = tuple(m for m in MODULES if m["key"] in AUXILIARY_KEYS)


@blueprint.get("/lab")
def home():
    return render_template(
        "lab.html",
        modules=PRIMARY_MODULES,
        auxiliary_modules=AUXILIARY_MODULES,
        desktop_available=callable(current_app.config.get("DESKTOP_OPEN_MODULE")),
    )


@blueprint.get("/api/lab/modules")
def modules():
    return jsonify(
        ok=True,
        modules=MODULES,
        primary_modules=PRIMARY_MODULES,
        auxiliary_modules=AUXILIARY_MODULES,
        desktop_available=callable(current_app.config.get("DESKTOP_OPEN_MODULE")),
    )


@blueprint.post("/api/lab/open-desktop/<key>")
def open_desktop(key):
    if not request.is_json:
        return jsonify(ok=False, error="需要 JSON 请求"), 415
    if key not in DESKTOP_MODULES:
        return jsonify(ok=False, error="未知的桌面实验模块"), 404
    callback = current_app.config.get("DESKTOP_OPEN_MODULE")
    if not callable(callback):
        return jsonify(
            ok=False, error="请在 Windows 程序中打开综合实验工作台，以切换到此桌面模块"
        ), 409
    # The callback emits a Qt signal; widgets are never touched on the HTTP thread.
    callback(key)
    return jsonify(ok=True, module=key, message="已请求切换至桌面模块")
