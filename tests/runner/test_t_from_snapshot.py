"""Branch coverage for tud_lbm/pipeline/runner.py.

Targets the 1 uncovered line (88.9 → ~100%):
- _t_from_snapshot: non-"timestep_" stem prefix and non-digit suffix paths
  (both return jnp.array(0) but were unreachable in previous tests).
"""

from __future__ import annotations
from types import SimpleNamespace
from tud_lbm.config import SimulationConfig
from tud_lbm.pipeline.runner import _t_from_snapshot


def _cfg(**kwargs) -> SimulationConfig:
    base = {"grid_shape": (8, 8), "tau": 0.8, "nt": 10}
    base.update(kwargs)
    return SimulationConfig(**base)


class TestTFromSnapshot:
    """All branches of _t_from_snapshot."""

    def test_non_init_from_file_returns_zero(self):
        cfg = _cfg(init_type="standard")
        assert int(_t_from_snapshot(cfg)) == 0

    def test_init_from_file_no_init_dir_returns_zero(self):
        # SimulationConfig rejects init_from_file+None; use SimpleNamespace to
        # reach the guard branch directly.
        cfg = SimpleNamespace(init_type="init_from_file", init_dir=None)
        assert int(_t_from_snapshot(cfg)) == 0

    def test_stem_without_timestep_prefix_returns_zero(self, tmp_path):
        npz = tmp_path / "snapshot_1000.npz"
        npz.write_bytes(b"")
        cfg = _cfg(init_type="init_from_file", init_dir=str(npz))
        assert int(_t_from_snapshot(cfg)) == 0

    def test_stem_with_non_digit_suffix_returns_zero(self, tmp_path):
        npz = tmp_path / "timestep_abc.npz"
        npz.write_bytes(b"")
        cfg = _cfg(init_type="init_from_file", init_dir=str(npz))
        assert int(_t_from_snapshot(cfg)) == 0

    def test_valid_timestep_stem_returns_correct_t(self, tmp_path):
        npz = tmp_path / "timestep_500.npz"
        npz.write_bytes(b"")
        cfg = _cfg(init_type="init_from_file", init_dir=str(npz))
        assert int(_t_from_snapshot(cfg)) == 500

    def test_timestep_zero_stem(self, tmp_path):
        npz = tmp_path / "timestep_0.npz"
        npz.write_bytes(b"")
        cfg = _cfg(init_type="init_from_file", init_dir=str(npz))
        assert int(_t_from_snapshot(cfg)) == 0
