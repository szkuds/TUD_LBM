"""Smoke tests: run every example config through the full pipeline.

Each test loads a TOML from ``tests/example_for_test/``, overrides ``nt=20``
so the run is fast, then asserts the simulation completes without NaN.

Marked ``@pytest.mark.slow`` — skipped in fast CI passes, run on schedule
or before a release with ``pytest -m slow``.
"""

from __future__ import annotations
import dataclasses
from pathlib import Path
import jax.numpy as jnp
import pytest
from src.config.adapter_toml import TomlAdapter
from src.pipeline.runner import init_state
from src.pipeline.runner import run
from src.pipeline.setup import build_setup

EXAMPLES_DIR = Path(__file__).resolve().parent / "example_for_test"
_ADAPTER = TomlAdapter()


def _all_example_tomls() -> list[Path]:
    return sorted(EXAMPLES_DIR.glob("*.toml"))


@pytest.mark.slow
@pytest.mark.parametrize("config_path", _all_example_tomls(), ids=lambda p: p.stem)
def test_example_runs_without_error(config_path: Path, tmp_path: Path) -> None:
    """Each example config loads, builds, runs 20 steps, and stays finite.

    The ``nt`` and ``results_dir`` fields are overridden so the test
    completes quickly without writing to the project tree.
    """
    config = _ADAPTER.load(str(config_path))
    config = dataclasses.replace(config, nt=20, results_dir=str(tmp_path))

    setup = build_setup(config)
    state = init_state(setup)
    final_state, _ = run(setup, state)

    assert not jnp.isnan(final_state.f).any(), (
        f"NaN detected in distribution function after 20 steps — config: {config_path.name}"
    )
    assert int(final_state.t) == 20, f"Expected t=20, got t={int(final_state.t)} — config: {config_path.name}"
