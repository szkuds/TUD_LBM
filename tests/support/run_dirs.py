"""Builders for synthetic simulation run directories.

Used by both the IO and CLI test suites. The snapshot values are fully
deterministic so that CSV output can be pinned against a golden fixture.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.config import SimulationConfig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

GRID_NX = 16
GRID_NY = 12

#: Iteration numbers whose gaps match ``save_interval`` exactly.
UNIFORM_ITERATIONS = (5, 10, 15, 20)

#: Iteration numbers whose gaps do NOT match ``save_interval``.
#: Exposes the difference between nominal-interval and actual-gap
#: differentiation of the contact-line position.
NONUNIFORM_ITERATIONS = (5, 10, 30, 90)

SAVE_INTERVAL = 5


def wetting_config(**overrides: object) -> SimulationConfig:
    """Return a deterministic ``multiphase_wetting`` config for fixtures."""
    params: dict[str, object] = {
        "sim_type": "multiphase_wetting",
        "simulation_name": "golden",
        "grid_shape": (GRID_NX, GRID_NY),
        "tau": 0.9,
        "nt": 20,
        "save_interval": SAVE_INTERVAL,
        "eos": "double-well",
        "kappa": 0.02,
        "interface_width": 2,
        "rho_l": 1.0,
        "rho_v": 0.2,
        "gravity_force": {"force_g": 1e-6, "inclination_angle_deg": 30.0},
        "initialisation": {"radii": [0.25], "centres": [[0.5, 0.5]]},
        "wetting_config": {"advancing_ca": 100.0},
        "bc_config": {"bottom": "wetting"},
        "plot_fields": ["density", "ca_theta_vs_x"],
    }
    params.update(overrides)
    return SimulationConfig(**params)  # ty: ignore[invalid-argument-type]


def write_snapshot(
    data_dir: Path,
    step: int,
    index: int,
    *,
    with_contact_metrics: bool = True,
) -> Path:
    """Write one deterministic ``timestep_{step}.npz`` snapshot.

    The density blob shifts one cell in +x per *index* so that centre-of-mass
    and average-position columns vary between snapshots.

    Args:
        data_dir: Destination ``data/`` directory.
        step: Iteration number encoded in the filename.
        index: Zero-based position in the snapshot sequence.
        with_contact_metrics: When ``False``, omit the ``ca_*``/``cll_*`` keys
            so consumers must derive them from the density field.

    Returns:
        Path to the written ``.npz`` file.
    """
    rho = np.full((GRID_NX, GRID_NY, 1, 1, 1), 0.2)
    lo = 4 + index
    rho[lo : lo + 6, 1:8, 0, 0, 0] = 1.0

    u = np.zeros((GRID_NX, GRID_NY, 1, 1, 2))
    u[:, :, 0, 0, 0] = 0.01 + 0.005 * index
    u[:, :, 0, 0, 1] = 0.002

    payload: dict[str, np.ndarray] = {"rho": rho, "u": u}
    if with_contact_metrics:
        payload |= {
            "ca_left": np.array(80.0 + index),
            "ca_right": np.array(95.0 - index),
            "cll_left": np.array(3.0 + 0.5 * index),
            "cll_right": np.array(10.0 + 0.7 * index),
        }

    out = data_dir / f"timestep_{step}.npz"
    # numpy's savez stub declares `allow_pickle: bool`, which collides with **payload.
    np.savez(out, **payload)  # ty: ignore[invalid-argument-type]
    return out


def build_run_dir(
    root: Path,
    *,
    iterations: Sequence[int] = UNIFORM_ITERATIONS,
    config: SimulationConfig | None = None,
    with_contact_metrics: bool = True,
    write_config_toml: bool = True,
) -> Path:
    """Create a synthetic run directory with snapshots and a ``config.toml``.

    Args:
        root: Directory to create the run inside.
        iterations: Iteration numbers to write snapshots for.
        config: Config to serialise; defaults to :func:`wetting_config`.
        with_contact_metrics: Passed through to :func:`write_snapshot`.
        write_config_toml: Whether to serialise ``config.toml`` into the run.

    Returns:
        Path to the created run directory.
    """
    config = config if config is not None else wetting_config()
    run_dir = root / "run"
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for index, step in enumerate(iterations):
        write_snapshot(data_dir, step, index, with_contact_metrics=with_contact_metrics)

    if write_config_toml:
        from src.config.adapter_toml import TomlAdapter

        TomlAdapter().save(config, str(run_dir / "config.toml"))

    return run_dir
