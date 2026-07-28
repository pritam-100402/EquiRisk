"""
tests/conftest.py

Shared pytest setup. Puts the repository root on sys.path so tests can
`import src.*` when run as plain `pytest` from the repo root, without
requiring the package to be pip-installed.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
