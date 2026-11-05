"""The 3D bulk operators: stencils, faces, streaming, source term, initialiser and refusals."""

from __future__ import annotations
from typing import Any
import jax.numpy as jnp
import numpy as np
import pytest
from src.config import SimulationConfig
from src.config.boundary import BC_APPLICATION_ORDER
from src.config.boundary import Edge
from src.config.boundary import boundary_edges
from src.config.boundary import pad_modes
from src.config.boundary import periodic_axes
from src.lattice.lattice import build_lattice
from src.operators.boundary import build_bc
from src.operators.differential import build_differential
from src.operators.differential._pad_utils import _apply_stencil_padding
from src.operators.initialise import build_initialise_fn
from src.operators.source_term import build_source_fn
from src.operators.streaming import build_streaming_fn
from src.pipeline.runner import init_state
from src.pipeline.setup import build_setup
from src.pipeline.step import kernel

_PERIODIC = ("wrap",) * 6
_N = 16
_MULTIPHASE = {"eos": "double-well", "kappa": 0.02, "rho_l": 1.0, "rho_v": 0.1, "interface_width": 3}
_ALL_FACES = BC_APPLICATION_ORDER


@pytest.fixture(scope="module")
def lattice():
    return build_lattice("D3Q19")


def _closed_axis(edge: str, name: str) -> dict[str, str]:
    """*name* on *edge* and its opposite: streaming opens an axis only when both ends are periodic."""
    opposite = {"bottom": "top", "top": "bottom", "left": "right", "right": "left", "front": "back", "back": "front"}
    return {edge: name, opposite[edge]: name}


def _wave(axis: int, n: int = _N) -> tuple[np.ndarray, float]:
    """``sin(k x_axis)`` on a periodic ``n`` cube, with ``k``."""
    k = 2.0 * np.pi / n
    coordinates = np.meshgrid(*(np.arange(n),) * 3, indexing="ij")
    return np.sin(k * coordinates[axis]), k


def _as_field(values: np.ndarray) -> jnp.ndarray:
    return jnp.asarray(values)[..., None, None]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_gradient_has_the_discrete_derivative_of_its_stencil(lattice, axis: int) -> None:
    """A plane wave along one axis feeds three stencil rows with weights summing to 1/6."""
    values, k = _wave(axis)
    gradient, _ = build_differential(lattice, _PERIODIC)

    result = np.asarray(gradient(_as_field(values), {}))

    coordinates = np.meshgrid(*(np.arange(_N),) * 3, indexing="ij")
    expected = np.sin(k) * np.cos(k * coordinates[axis])
    assert result.shape == (_N, _N, _N, 1, 3)
    np.testing.assert_allclose(result[..., 0, axis], expected, atol=1e-12)
    for other in {0, 1, 2} - {axis}:
        np.testing.assert_allclose(result[..., 0, other], 0.0, atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_laplacian_has_the_discrete_second_derivative_of_its_stencil(lattice, axis: int) -> None:
    values, k = _wave(axis)
    _, laplacian = build_differential(lattice, _PERIODIC)

    result = np.asarray(laplacian(_as_field(values), {}))

    assert result.shape == (_N, _N, _N, 1, 1)
    np.testing.assert_allclose(result[..., 0, 0], 2.0 * (np.cos(k) - 1.0) * values, atol=1e-12)


def test_both_ghost_layers_of_z_come_from_its_own_extent() -> None:
    field = jnp.arange(4.0 * 4 * 4).reshape(4, 4, 4)
    modes = ("edge", "edge", "edge", "edge", "wrap", "edge")

    padded = np.asarray(_apply_stencil_padding(field, modes))

    np.testing.assert_array_equal(padded[1:-1, 1:-1, -1], np.asarray(field)[:, :, 0])  # front wraps
    np.testing.assert_array_equal(padded[1:-1, 1:-1, 0], np.asarray(field)[:, :, 0])  # back is edge


def test_the_config_names_six_faces_in_application_order() -> None:
    config = SimulationConfig(lattice_type="D3Q19", grid_shape=(8, 8, 8), boundary_conditions={"front": "bounce-back"})

    assert [edge.edge for edge in boundary_edges(config)] == list(_ALL_FACES)
    assert pad_modes(config) == ("wrap", "wrap", "wrap", "wrap", "edge", "wrap")
    assert periodic_axes(config) == (True, True, False)


def test_a_two_dimensional_config_keeps_its_four_faces() -> None:
    config = SimulationConfig(grid_shape=(8, 8))

    assert [edge.edge for edge in boundary_edges(config)] == ["bottom", "top", "left", "right"]
    assert len(pad_modes(config)) == 4
    assert periodic_axes(config) == (True, True, True)


def test_streaming_zeroes_the_wrapped_z_layer_only_when_z_is_walled(lattice) -> None:
    f = jnp.ones((4, 4, 4, lattice.q, 1))
    upward = int(lattice.front_indices[0])

    walled = np.asarray(build_streaming_fn((True, True, False))(f, lattice))
    periodic = np.asarray(build_streaming_fn((True, True, True))(f, lattice))

    assert walled[:, :, 0, upward, 0].max() == 0.0
    np.testing.assert_array_equal(periodic, np.asarray(f))


@pytest.mark.parametrize("name", ["bounce-back", "symmetry"])
@pytest.mark.parametrize("edge", _ALL_FACES)
def test_a_closed_face_conserves_mass(lattice, name: str, edge: str) -> None:
    config = SimulationConfig(lattice_type="D3Q19", grid_shape=(6, 7, 8), boundary_conditions=_closed_axis(edge, name))
    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.uniform(0.5, 1.5, (6, 7, 8, lattice.q, 1)))

    streamed = build_streaming_fn(periodic_axes(config))(f, lattice)
    result = build_bc(boundary_edges(config), lattice)(streamed, f)

    assert float(jnp.sum(result)) == pytest.approx(float(jnp.sum(f)), rel=1e-13)


@pytest.mark.parametrize("edge", _ALL_FACES)
def test_symmetry_keeps_the_tangential_momentum(lattice, edge: str) -> None:
    normal = {"bottom": 1, "top": 1, "left": 0, "right": 0, "front": 2, "back": 2}[edge]
    tangential = (normal + 1) % 3
    config = SimulationConfig(
        lattice_type="D3Q19", grid_shape=(6, 6, 6), boundary_conditions=_closed_axis(edge, "symmetry")
    )
    velocity = [0.0, 0.0, 0.0]
    velocity[tangential] = 0.05
    f = build_initialise_fn("standard")((6, 6, 6), lattice, velocity=tuple(velocity))

    streamed = build_streaming_fn(periodic_axes(config))(f, lattice)
    result = build_bc(boundary_edges(config), lattice)(streamed, f)

    momentum = jnp.sum(result * lattice.c, axis=(0, 1, 2, 3))
    assert float(momentum[tangential]) == pytest.approx(float(jnp.sum(f * lattice.c, axis=(0, 1, 2, 3))[tangential]))
    assert float(momentum[normal]) == pytest.approx(0.0, abs=1e-12)


def test_the_well_balanced_source_has_the_moments_of_the_force(lattice) -> None:
    """Zero mass, and first moment ``F``: all three components reach the populations."""
    shape = (4, 4, 4, 1, 3)
    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.uniform(-0.05, 0.05, shape))
    force = jnp.asarray(rng.uniform(-1e-3, 1e-3, shape))

    source = build_source_fn("wb")(
        jnp.ones((4, 4, 4, 1, 1)), u, force, lattice, gradient=lambda grid: jnp.zeros((*grid.shape[:4], 3))
    )

    assert source.shape == (4, 4, 4, lattice.q, 1)
    np.testing.assert_allclose(np.asarray(jnp.sum(source, axis=-2)), 0.0, atol=1e-15)
    first = np.asarray(jnp.sum(source * lattice.c, axis=-2, keepdims=True))
    np.testing.assert_allclose(first, np.asarray(force), atol=1e-15)


def test_an_inclusion_is_spherical_on_a_3d_lattice(lattice) -> None:
    f = build_initialise_fn("multiphase_bubbles")(
        (21, 21, 21),
        lattice,
        rho_l=1.0,
        rho_v=0.1,
        interface_width=2,
        centres=[[0.5, 0.5, 0.5]],
        radii=[0.25],
        dispersed="liquid",
    )
    rho = np.asarray(jnp.sum(f, axis=-2))[..., 0]

    assert rho[10, 10, 10] > 0.9
    assert rho[0, 10, 10] < 0.2
    assert rho[10, 16, 10] == pytest.approx(rho[10, 10, 16])
    assert rho[16, 10, 10] == pytest.approx(rho[10, 10, 16])


def test_a_planar_centre_is_refused_on_a_3d_lattice(lattice) -> None:
    init = build_initialise_fn("multiphase_bubbles")
    centres = [[0.5, 0.5]]

    with pytest.raises(ValueError, match=r"three coordinates"):
        init((8, 8, 8), lattice, centres=centres, radii=[0.2])


def test_a_step_commutes_with_a_roll_along_z(lattice) -> None:
    """A fully periodic z axis has no seam: the step and a z roll commute exactly."""
    config = SimulationConfig(
        lattice_type="D3Q19",
        grid_shape=(12, 12, 12),
        tau=1.0,
        nt=4,
        multiphase=dict(_MULTIPHASE),
        init_type="multiphase_bubbles",
        initialisation={"centres": [[0.5, 0.5, 0.25]], "radii": [0.25], "dispersed": "liquid"},
    )
    setup = build_setup(config)
    f = init_state(setup).f

    stepped_then_rolled = jnp.roll(kernel(setup, f, None, {}), 5, axis=2)
    rolled_then_stepped = kernel(setup, jnp.roll(f, 5, axis=2), None, {})

    np.testing.assert_array_equal(np.asarray(stepped_then_rolled), np.asarray(rolled_then_stepped))


@pytest.mark.parametrize(
    ("kwargs", "named"),
    [
        pytest.param({"wetting": {"phi_left": 1.0}, "multiphase": _MULTIPHASE}, "wetting", id="wetting_section"),
        pytest.param({"boundary_conditions": {"bottom": "wetting"}}, "wetting", id="wetting_edge"),
        pytest.param(
            {"hysteresis": {"ca_advancing": 100.0, "ca_receding": 80.0}, "multiphase": _MULTIPHASE},
            "hysteresis",
            id="hysteresis",
        ),
        pytest.param(
            {
                "forces": {
                    "electric_force": {
                        "permittivity_liquid": 80.0,
                        "permittivity_vapour": 1.0,
                        "conductivity_liquid": 0.01,
                        "conductivity_vapour": 0.001,
                    }
                }
            },
            "electric_force",
            id="electric",
        ),
        pytest.param({"collision_scheme": "mrt", "k_diag": (1.0,) * 9}, "MRT", id="mrt"),
        pytest.param({"obstacle": {"center_x": 8, "center_y": 8, "radius": 2}}, "obstacle", id="obstacle"),
    ],
)
def test_what_has_no_3d_form_is_refused_by_name(kwargs: dict[str, Any], named: str) -> None:
    params: dict[str, Any] = {"lattice_type": "D3Q19", "grid_shape": (16, 16, 1), **kwargs}

    with pytest.raises(ValueError, match=named):
        SimulationConfig(**params)


def test_a_two_dimensional_lattice_refuses_a_deep_grid() -> None:
    shape = (8, 8, 8)

    with pytest.raises(ValueError, match="two-dimensional"):
        SimulationConfig(grid_shape=shape)


@pytest.mark.parametrize("edge", [Edge.BOTTOM, Edge.TOP, Edge.LEFT, Edge.RIGHT])
def test_symmetry_keeps_the_tangential_momentum_in_2d(edge: str) -> None:
    """The 2D faces once reversed the tangential velocity of the diagonals at top and bottom."""
    lattice = build_lattice("D2Q9")
    normal = {"bottom": 1, "top": 1, "left": 0, "right": 0}[edge]
    tangential = 1 - normal
    config = SimulationConfig(grid_shape=(6, 7), boundary_conditions=_closed_axis(edge, "symmetry"))
    velocity = [0.0, 0.0]
    velocity[tangential] = 0.05
    f = build_initialise_fn("standard")((6, 7, 1), lattice, velocity=tuple(velocity))

    streamed = build_streaming_fn(periodic_axes(config))(f, lattice)
    result = build_bc(boundary_edges(config), lattice)(streamed, f)

    np.testing.assert_allclose(np.asarray(result), np.asarray(f), atol=1e-15)


def test_a_channel_between_symmetry_planes_keeps_its_flow() -> None:
    config = SimulationConfig(
        grid_shape=(16, 16),
        tau=0.8,
        init_type="standard",
        initialisation={"velocity": (0.05, 0.0, 0.0)},
        boundary_conditions={"bottom": "symmetry", "top": "symmetry"},
    )
    setup = build_setup(config)
    f = init_state(setup).f

    for _ in range(50):
        f = kernel(setup, f, None, {})

    momentum = jnp.sum(f * setup.lattice.c, axis=(0, 1, 2, 3)) / jnp.sum(f)
    assert float(momentum[0]) == pytest.approx(0.05, rel=1e-12)
    assert float(momentum[1]) == pytest.approx(0.0, abs=1e-14)
