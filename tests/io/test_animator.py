"""Tests for plotting animator."""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.config import SimulationConfig
from src.simulation_io.plotting.animator import Animator

if TYPE_CHECKING:
    from pathlib import Path


class _SpyAnalysis:
    name = "spy"

    def __init__(self):
        self.calls: list[int] = []

    def compute(self, files):
        return {"iters": np.arange(len(files)), "values": np.arange(len(files), dtype=float)}

    def render(self, ax, precomputed):
        ax.plot(precomputed["iters"], precomputed["values"])

    def update(self, ax, files):
        self.calls.append(len(files))
        precomputed = self.compute(files)
        self.render(ax, precomputed)


def _write_snapshots(data_dir: Path, n: int = 3) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for step in range(1, n + 1):
        np.savez(
            data_dir / f"timestep_{step}.npz",
            rho=np.ones((6, 6, 1, 1, 1)),
            u=np.zeros((6, 6, 1, 1, 2)),
        )


def test_animator_creates_frames(tmp_path):
    run_dir = tmp_path / "run"
    _write_snapshots(run_dir / "data", n=4)

    cfg = SimulationConfig(plot_fields=[])
    animator = Animator(config=cfg, run_dir=run_dir)
    spy = _SpyAnalysis()
    animator.builder.field_operators.clear()
    animator.builder.analysis_operators.clear()
    animator.builder.analysis_operators.append(spy)

    frames = animator.build_frames()

    assert len(frames) == 4
    assert all(path.exists() for path in frames)


def test_animator_analysis_grows_over_frames(tmp_path):
    run_dir = tmp_path / "run"
    _write_snapshots(run_dir / "data", n=3)

    cfg = SimulationConfig(plot_fields=[])
    animator = Animator(config=cfg, run_dir=run_dir)
    spy = _SpyAnalysis()
    animator.builder.field_operators.clear()
    animator.builder.analysis_operators.clear()
    animator.builder.analysis_operators.append(spy)

    animator.build_frames()

    assert spy.calls == [1, 2, 3]
