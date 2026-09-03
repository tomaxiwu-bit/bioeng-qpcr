# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for building the bioeng-qpcr desktop application.

Build commands:
    pip install pyinstaller
    pyinstaller bioeng-qpcr.spec

The resulting executable is placed in ``dist/bioeng-qpcr/``.
"""

import importlib.util
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all

block_cipher = None

# Project root (where this spec file lives)
project_root = Path(SPECPATH).resolve()
src_dir = project_root / "src"

# Pillow contains dynamically imported image plugins and native extension
# modules. Relying on matplotlib's transitive import discovery is fragile: a
# build can start while every plotting-backed panel fails with a missing PIL.
pillow_datas, pillow_binaries, pillow_hiddenimports = collect_all("PIL")

# COBRApy is optional for source installs. When present in the build
# environment, collect the full package instead of a partial import graph.
cobra_datas: list[tuple[str, str]] = []
cobra_binaries: list[tuple[str, str]] = []
cobra_hiddenimports: list[str] = []
if importlib.util.find_spec("cobra") is not None:
    cobra_datas, cobra_binaries, cobra_hiddenimports = collect_all("cobra")

a = Analysis(
    [str(project_root / "src" / "bioeng_qpcr" / "gui" / "main.py")],
    pathex=[str(src_dir)],
    binaries=pillow_binaries + cobra_binaries,
    datas=[
        # Include test fixtures as sample data
        (
            str(project_root / "tests" / "fixtures" / "*.csv"),
            "bioeng_qpcr/fixtures",
        ),
    ] + pillow_datas + cobra_datas,
    hiddenimports=[
        # Core modules
        "bioeng_qpcr",
        "bioeng_qpcr.cli.main",
        "bioeng_qpcr.io.csv_reader",
        "bioeng_qpcr.calculations",
        "bioeng_qpcr.calculations.delta_cq",
        "bioeng_qpcr.calculations.delta_delta_cq",
        "bioeng_qpcr.calculations.pfaffl",
        "bioeng_qpcr.calculations.normalized_quantity",
        "bioeng_qpcr.calculations.efficiency",
        "bioeng_qpcr.calculations.batch",
        "bioeng_qpcr.validation",
        "bioeng_qpcr.validation.engine",
        "bioeng_qpcr.validation.registry",
        "bioeng_qpcr.validation.rules",
        "bioeng_qpcr.reference_genes",
        "bioeng_qpcr.reference_genes.genorm",
        "bioeng_qpcr.reference_genes.normfinder",
        "bioeng_qpcr.reference_genes.consensus",
        "bioeng_qpcr.statistics",
        "bioeng_qpcr.provenance",
        "bioeng_qpcr.reporting",
        "bioeng_qpcr.reporting.research_report",
        "bioeng_qpcr.domain",
        "bioeng_qpcr.domain.entities",
        "bioeng_qpcr.domain.enums",
        "bioeng_qpcr.domain.assumptions",
        # Plotting
        "bioeng_qpcr.plotting",
        "bioeng_qpcr.plotting.style",
        "bioeng_qpcr.plotting.amplification",
        "bioeng_qpcr.plotting.barplots",
        "bioeng_qpcr.plotting.heatmap",
        "bioeng_qpcr.plotting.enzyme_kinetics",
        "bioeng_qpcr.plotting.fermentation_kinetics",
        "bioeng_qpcr.plotting.metabolic_flux",
        # Enzyme kinetics
        "bioeng_qpcr.enzyme_kinetics",
        "bioeng_qpcr.enzyme_kinetics.michaelis_menten",
        "bioeng_qpcr.enzyme_kinetics.inhibition",
        "bioeng_qpcr.enzyme_kinetics.validation",
        # Fermentation kinetics
        "bioeng_qpcr.fermentation_kinetics",
        "bioeng_qpcr.fermentation_kinetics.monod",
        "bioeng_qpcr.fermentation_kinetics.identifiability",
        "bioeng_qpcr.fermentation_kinetics.validation",
        # Metabolic flux (optional - requires cobra)
        "bioeng_qpcr.metabolic_flux",
        "bioeng_qpcr.metabolic_flux.fba",
        "bioeng_qpcr.metabolic_flux.analysis",
        "bioeng_qpcr.metabolic_flux.validation",
        # GUI
        "bioeng_qpcr.gui",
        "bioeng_qpcr.gui.main_window",
        "bioeng_qpcr.gui.theme",
        "bioeng_qpcr.gui.components",
        "bioeng_qpcr.gui.components.base_panel",
        "bioeng_qpcr.gui.components.file_selector",
        "bioeng_qpcr.gui.components.plot_canvas",
        "bioeng_qpcr.gui.panels",
        "bioeng_qpcr.gui.panels.qpcr_panel",
        "bioeng_qpcr.gui.panels.enzyme_panel",
        "bioeng_qpcr.gui.panels.ferment_panel",
        "bioeng_qpcr.gui.panels.fba_panel",
        # Third-party hidden imports
        "matplotlib.backends.backend_qtagg",
        "matplotlib.backends.backend_agg",
        # Pillow is imported by matplotlib at panel-import time.
        "PIL",
        "PIL.Image",
        "PIL.ImageQt",
        "PIL.PngImagePlugin",
    ] + pillow_hiddenimports + cobra_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size.
        # NOTE: Do NOT exclude 'unittest' — numpy/scipy/pandas reference it
        # at runtime and excluding it causes ImportError in the packaged exe.
        "tkinter",
        "pytest",
        "IPython",
        "notebook",
        "jupyter",
        # Exclude unrelated packages found in the user environment
        "yt_dlp",
        "websockets",
        "requests",
        "urllib3",
        "mutagen",
        "brotli",
        "secretstorage",
        "curl_cffi",
        "pyodide",
        "Cryptodome",
        "Crypto",
        "cx_Oracle",
        "oracle",
        # Large unrelated scientific/network packages
        "cv2",
        "opencv",
        "onnxruntime",
        "numba",
        "llvmlite",
        "openai",
        "fastapi",
        "aiohttp",
        "huggingface_hub",
        "google",
        "googleapiclient",
        "google_auth",
        "cryptography",
        "lz4",
        "msgpack",
        "coverage",
        "pythonwin",
        "win32ui",
        "win32com",
        "pywin32",
        "jsonschema",
        "jinja2",
        "email_validator",
        "opentelemetry",
        "opentelemetry_api",
        "attrs",
        "openpyxl",
        "xlrd",
        "xlwt",
        "selenium",
        "PyPDF2",
        "pypdf",
        "reportlab",
        "bokeh",
        "plotly",
        "streamlit",
        "flask",
        "django",
        "sqlalchemy",
        "pymysql",
        "psycopg2",
        "asyncpg",
        "aiomysql",
        "tornado",
        "zmq",
        "pyzmq",
        "IPython",
        "jedi",
        "parso",
        "pygments",
        "click_completion",
        "setuptools",
        "pip",
        "wheel",
        # Large AI/ML packages not needed by bioeng-qpcr
        "torch",
        "torchvision",
        "torchaudio",
        "transformers",
        "tokenizers",
        "tiktoken",
        "pyarrow",
        "sklearn",
        "scikit_learn",
        "skimage",
        "scikit_image",
        "xgboost",
        "lightgbm",
        "catboost",
        "tensorflow",
        "keras",
        "theano",
        "gensim",
        "spacy",
        "nltk",
        "datasets",
        "accelerate",
        "safetensors",
        "tokenizers",
        "huggingface_hub",
        "colorama",
        "cffi",
        "_cffi_backend",
        "pycparser",
        "mypy",
        "mypy_extensions",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="bioeng-qpcr",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI application - no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add: icon="assets/icon.ico" if you have an icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="bioeng-qpcr",
)
