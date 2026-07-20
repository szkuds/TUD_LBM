"""Smoke tests: verify the src public API and operator subcategories import cleanly."""

from __future__ import annotations
import importlib.util


def test_public_api_importable() -> None:
    """Core public symbols resolve without ImportError."""
    import src  # noqa: F401
    from src.config import SimulationConfig  # noqa: F401
    from src.pipeline.runner import init_state  # noqa: F401
    from src.pipeline.runner import run  # noqa: F401
    from src.pipeline.setup import build_setup  # noqa: F401
    from src.registry import get_operator_names  # noqa: F401
    from src.simulation_io import SimulationIO  # noqa: F401


def test_operator_subcategories_importable() -> None:
    """Every standard operator subcategory is findable by importlib."""
    subcategories = [
        "collision",
        "streaming",
        "equilibrium",
        "macroscopic",
        "boundary",
        "differential",
        "force",
        "wetting",
        "initialise",
    ]
    for sub in subcategories:
        spec = importlib.util.find_spec(f"src.operators.{sub}")
        assert spec is not None, f"src.operators.{sub} not found"
