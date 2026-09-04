"""
Shared pytest fixtures.

Tests run against a temporary database so they never touch the pilot analysis
results, and they build tracks synthetically so behaviour logic can be verified
without depending on model weights or footage.
"""

from __future__ import annotations

import os
import sys
import tempfile
from typing import List, Optional

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Point the whole stack at a scratch database *before* modules import config.
_TMP_DB = os.path.join(tempfile.mkdtemp(prefix="visionguard_test_"), "test.db")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ.setdefault("USE_OPEN_VOCAB", "false")  # tests must not need model weights

import config  # noqa: E402

config.DATABASE_PATH = _TMP_DB

from backend.database.db import DatabaseManager, get_connection, init_db  # noqa: E402
from detection.detector import Detection  # noqa: E402
from detection.object_classes import WarehouseEntity  # noqa: E402
from detection.tracker import PersistentTracker, TrackedObject  # noqa: E402

FRAME_W, FRAME_H = 1280.0, 720.0



@pytest.fixture(scope="session", autouse=True)
def _database():
    init_db()
    yield
    try:
        os.remove(_TMP_DB)
    except OSError:
        pass


@pytest.fixture(autouse=True)
def clean_tables():
    """Every test starts from an empty database."""
    with get_connection() as conn:
        conn.execute("DELETE FROM incidents")
        conn.execute("DELETE FROM videos")
        conn.commit()
    yield


