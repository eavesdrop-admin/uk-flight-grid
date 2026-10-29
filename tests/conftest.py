"""Shared fixtures for the Stage A tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def salt() -> bytes:
    return b"\x01" * 32
