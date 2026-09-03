"""Root pytest configuration.

The project uses a ``src`` layout and may not be installed (``pip install -e .``)
in every environment.  This conftest ensures the ``src`` directory is on
``sys.path`` so that ``import bioeng_qpcr`` resolves during test collection
regardless of installation state.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
