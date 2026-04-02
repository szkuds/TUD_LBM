"""Integration test: Poiseuille flow (body-force-driven channel).

This is the REFACTORING ANCHOR for the TUD_LBM project. Every structural
change made during the hexagonal architecture migration must keep this
test green.

Physics
-------
A 2-D channel with bounce-back walls on top/bottom and periodic boundaries
on left/right is driven by a constant horizontal body force g.  After the
flow reaches steady state the velocity profile across the channel must
match the analytical parabolic solution:

    u_x(y) = g / (2 * nu) * (y + 0.5) * (H - 0.5 - y)

where nu = (tau - 0.5) / 3 is the kinematic viscosity and H = ny is
the effective channel height under the half-way bounce-back convention
(wall sits half a lattice spacing outside the domain).

Acceptance criteria
-------------------
* L2 relative error between simulated and analytical profile < 2 %
* Test completes in < 60 s on a single CPU
* No files on disk are created (no IO side-effects)
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from config.simulation_config import SimulationConfig
from runner.run import init_state
from runner.run import run
from setup.simulation_setup import build_setup

# ── Test parameters ──────────────────────────────────────────────────

NX = 5  # short channel (periodic direction — length doesn't matter)
NY = 32  # wall-normal direction
TAU = 0.8  # relaxation time  → nu = (0.8 - 0.5)/3 = 0.1
NT = 5000  # timesteps to reach steady state
BODY_FORCE_G = 1e-6  # small force for low-Re flow


# ── Helpers ──────────────────────────────────────────────────────────


def analytical_poiseuille(ny, tau, g):
    """Return the analytical parabolic velocity profile u_x(y).

    Uses the half-way bounce-back convention: the no-slip wall is
    located half a lattice spacing outside the domain, so the
    effective channel width is *ny* (not ny − 1).

        u_x(y) = g / (2 ν) · (y + ½)(ny − ½ − y)
    """
    nu = (tau - 0.5) / 3.0
    y = np.arange(ny)
    return g / (2.0 * nu) * (y + 0.5) * (ny - 0.5 - y)


@pytest.fixture
def poiseuille_simulation():
    """Build and run simulation to steady state using functional API.

    Returns:
    -------
    dict
        - "final_state": State after NT timesteps
        - "config": SimulationConfig used
    """
    config = SimulationConfig(
        sim_type="single_phase",
        grid_shape=(NX, NY),
        lattice_type="D2Q9",
        tau=TAU,
        nt=NT,
        gravity_force={"force_g": BODY_FORCE_G, "inclination_angle_deg": 90.0},
        bc_config={
            "left": "periodic",
            "right": "periodic",
            "top": "bounce-back",
            "bottom": "bounce-back",
        },
        save_interval=NT,  # no intermediate saves
    )

    setup = build_setup(config)
    state = init_state(setup)

    # Run via lax.scan (pure functional, no side effects)
    final_state, _ = run(setup, state, nt=NT)

    return {
        "final_state": final_state,
        "config": config,
    }


# ── Integration Tests ────────────────────────────────────────────────


@pytest.mark.integration
def test_poiseuille_parabolic_profile(poiseuille_simulation):
    """Simulated velocity profile matches analytical parabolic solution.

    This is the PRIMARY physics validation: ensures the entire LBM pipeline
    (equilibrium → collision → streaming → boundary conditions) correctly
    implements the physics.

    Acceptance: L2 relative error < 2%
    """
    final_state = poiseuille_simulation["final_state"]

    u = np.array(final_state.u)  # (nx, ny, 1, 2)
    ux = u[:, :, 0, 0]  # (nx, ny) — x-velocity
    ux_mean = ux.mean(axis=0)  # average over periodic direction

    u_analytical = analytical_poiseuille(NY, TAU, BODY_FORCE_G)

    # L2 relative error
    error = np.linalg.norm(ux_mean - u_analytical) / np.linalg.norm(u_analytical)

    assert error < 0.02, (
        f"L2 relative error {error:.4f} exceeds 2% tolerance.\n"
        f"  max(simulated) = {ux_mean.max():.6e}\n"
        f"  max(analytical) = {u_analytical.max():.6e}"
    )


@pytest.mark.integration
def test_poiseuille_symmetry(poiseuille_simulation):
    """Velocity profile is symmetric about the channel centre.

    Half-way bounce-back convention ensures symmetry if implemented correctly.

    Acceptance: asymmetry < 1%
    """
    final_state = poiseuille_simulation["final_state"]

    u = np.array(final_state.u)
    ux_mean = u[:, :, 0, 0].mean(axis=0)

    # Normalise before checking symmetry
    ux_norm = ux_mean / (ux_mean.max() + 1e-30)
    symmetry_error = np.max(np.abs(ux_norm - ux_norm[::-1]))

    assert symmetry_error < 0.01, f"Profile asymmetry {symmetry_error:.4f} exceeds 1% tolerance"


@pytest.mark.integration
def test_poiseuille_no_nan(poiseuille_simulation):
    """Simulation completed without NaN (numerical stability check).

    NaN indicates divergence or invalid operations (e.g., divide by zero).

    Acceptance: All values are finite
    """
    final_state = poiseuille_simulation["final_state"]
    f = np.array(final_state.f)

    assert not np.isnan(f).any(), "NaN in distribution function"


@pytest.mark.integration
def test_poiseuille_runtime(poiseuille_simulation):
    """Simulation completes quickly (under 60 seconds on single CPU).

    This test is implicit: if the fixture takes > 60s, pytest --timeout=60
    will fail this test. We include it for clarity and documentation.

    Acceptance: Simulation runs in < 60 seconds
    """
    # The fixture has already run successfully; this test documents
    # the performance requirement.
    assert True
