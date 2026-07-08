"""Shared pytest fixtures.

pytest's built-in `tmp_path` fixture scans the OS temp directory
(C:\\Users\\...\\AppData\\Local\\Temp\\pytest-of-*), which hits a
PermissionError in this environment. `tmp_path_local` sideswipes that by
creating/cleaning up a scratch directory inside the repo instead.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

_LOCAL_TMP_ROOT = Path(__file__).parent / ".tmp_test_artifacts"


@pytest.fixture
def tmp_path_local():
    _LOCAL_TMP_ROOT.mkdir(exist_ok=True)
    d = Path(tempfile.mkdtemp(dir=_LOCAL_TMP_ROOT))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
