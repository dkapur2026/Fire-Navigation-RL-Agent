from pathlib import Path

import pytest

from emberpath.blueprint import Blueprint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def studio() -> Blueprint:
    return Blueprint.from_file(ROOT / "blueprints" / "studio.txt")


@pytest.fixture(scope="session")
def ranch() -> Blueprint:
    return Blueprint.from_file(ROOT / "blueprints" / "ranch.txt")
