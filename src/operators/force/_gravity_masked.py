"""Buoyancy-referenced gravity force module.

Provides a body force weighted by a *band-limited phase indicator*: zero
throughout the light phase, ``drho * g`` throughout the dense phase, and a
linear ramp between the two across the interface. Referencing the light phase
is what makes it usable in a periodic or wall-bounded domain — the ambient
vapour of a droplet run is left force-free, so the whole ambient body is not
accelerated along with the inclusion.

Written this way there is no topology branch. A liquid droplet in vapour is
driven at ``drho * g`` and its ambient at zero; a vapour bubble in liquid is
itself force-free and the *liquid* around it carries ``drho * g``, so the
bubble rises on the resulting pressure gradient with no sign flag anywhere in
the config. Both are the same formula.

Why the force must sit on the dense phase, not the inclusion
------------------------------------------------------------
An early version of this module injected the *net* buoyancy ``+-drho * g``
masked to the dispersed phase, arguing that a force-free continuous phase
builds no hydrostatic gradient and therefore nothing pushes back. That is
self-consistent for a droplet and self-defeating for a bubble: with nothing
pushing back, the light phase has to carry the entire momentum injection in
its own inertia. A body force enters as ``du = F / rho``
(:mod:`src.operators.macroscopic._multiphase`), so masking ``drho * g`` onto
the vapour accelerates it at ``drho / rho_v`` times ``g`` — a factor of 800
for ``rho_l = 12.18``, ``rho_v = 0.015``, against ``drho / rho_l ~ 1`` for the
droplet the formulation was designed around.

Nor does it reach a steady state. Balancing that body force hydrostatically
needs a pressure difference ``drho * g * 2R`` across the inclusion, which for
the parameters above is 82x the vapour's *absolute* pressure. No equilibrium
exists, so the gas evacuates; and because the injected force did not scale
with ``rho``, thinning the gas raised ``F / rho`` further.

The dense phase has no such limit — it sustains a pressure gradient, in
tension if need be — which is why the indicator vanishes on the light branch
and saturates on the dense one, and not the other way round. Buoyancy on a
light inclusion is then not injected at all: it emerges as the reaction to the
surrounding liquid's hydrostatic gradient, which is also what supplies the
added mass that bounds the rise speed.

Why the indicator is banded, and not the plain excess ``rho - rho_v``
---------------------------------------------------------------------
Its predecessor used the unbounded weight ``rho - rho_v`` with ``rho_v`` taken
from the config. Because a body force enters as ``F / rho``, the acceleration
that leaves in the ambient is

    a_ambient = (1 - rho_ref / rho_ambient) * g

so the ambient is only force-free when ``rho_ref`` matches its *actual*
density to within about a percent. It does not: an equilibrated droplet
relaxes away from the prescribed coexistence densities, and for an inclined
wetting run measured here the bulk vapour sat at ``0.0062`` against a
prescribed ``rho_v = 0.001``. The ambient was therefore driven at **88% of g**
— very nearly as hard as the droplet — the net injected force was 6.6% above
``drho * g * A``, and the gas, carrying 1/160 of the mass, ran away until the
run died. The same failure is invisible in a bubble run, where the ambient is
the dense phase and a 1% error in ``rho_ref`` costs only 1% of ``g``.

Nor is a better ``rho_ref`` sufficient on its own: over that run's 50k-step
equilibration the vapour density climbed monotonically from ``0.001`` to
``0.0064`` with no plateau, and kept drifting under gravity. Any single
reference value goes stale.

The band fixes both. ``_PHASE_BAND`` insets the ramp from each measured phase
density by a tenth of the contrast, so the weight is *exactly* zero everywhere
below ``rho_lo + 0.1 * drho`` and *exactly* ``drho`` everywhere above
``rho_hi - 0.1 * drho``. The band only needs classifier-grade accuracy rather
than reference-grade: a vapour density that wanders by an order of magnitude
still lands on zero. For the run above the lower edge sits at ``0.101``
against an ambient of ``0.0062``, a 16x margin.

The ramp is still continuous through the interface, so unlike a hard threshold
at ``rho_mean`` there is no tie-breaking rule and no discontinuity. It is also
antisymmetric about ``rho_mean`` — the band is inset by the same amount at
both ends — so the net force over an inclusion is unchanged at ``drho * g * A``
for the same ``A`` the ``rho_mean`` contour encloses.

Boundary conditions are load-bearing
------------------------------------
The liquid can only stay at rest if it can hold a hydrostatic ramp along every
direction that has a gravity component, and a periodic axis cannot support a
linear pressure ramp. Under an inclined gravity the tangential component is
then balanced by wall shear instead, and the domain fills with Poiseuille flow
— for a 201x101 channel at 50 degrees, ``u_max = 3e-3``, some 230x the
bubble's own buoyant slide speed. Close the tangential axis (``left`` and
``right`` set to ``bounce-back``) and the same component is balanced by a
density ramp of about 1% of ``rho_l`` instead, with the liquid at rest. Do not
use ``symmetry`` for those walls: its correction assumes a periodic x. This
applies to a *bubble* run, where the ambient is the driven phase; a droplet
run's ambient is force-free and needs no such ramp.

Switching gravity on abruptly leaves the fluid one acoustic crossing behind
the force, which launches a standing sound wave that rings for tens of
thousands of steps at an amplitude far above the motion being measured. The
optional ``ramp_steps`` / ``ramp_start_t`` pair brings the force up slowly
enough that the fluid stays quasi-statically balanced and no wave is excited.
``ramp_start_t`` is an *absolute* timestep because ``state.t`` survives
restarts (:func:`src.pipeline.runner._t_from_snapshot` reads it back from the
snapshot filename), so a resumed run must name the step at which gravity was
first applied for every chunk to agree on the same ramp.

Usage::

    # Via registry (preferred)
    from operators.force import build_force_fn

    module = build_force_fn("gravity_masked_force")
    pre = module.build({"force_g": 0.001}, (64, 64), config=config, lattice=lattice)
    force = module.compute(state, pre)

    # Direct (internal / testing)
    from operators.force._gravity_masked import GravityForceModule

    pre = GravityForceModule.build({"force_g": 0.001}, (64, 64), config=config, lattice=lattice)
    force = GravityForceModule.compute(state, pre)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from typing import NamedTuple
from typing import cast
import jax.numpy as jnp
from src.operators.force._gravity import _build_gravity_template
from src.registry import force_model

if TYPE_CHECKING:
    from src.config.simulation_config import SimulationConfig
    from src.pipeline.state import State

# Fraction of the phase contrast by which the indicator's ramp is inset from
# each phase density. Large enough that the bulk phases sit outside the ramp
# even after the coexistence densities drift; small enough that the ramp still
# spans most of the interface. Inset equally at both ends so the indicator
# stays antisymmetric about rho_mean and the net force is unchanged.
_PHASE_BAND = 0.1


class GravityPrecomputed(NamedTuple):
    """Container for gravity precomputed data.

    Attributes:
        template: constant force field, shape (nx, ny, nz, 1, d)
        band_lo: density below which the weight is exactly zero, or None for a
            single-phase run
        band_hi: density above which the weight is exactly ``drho``, or None
        drho: phase density contrast, i.e. the saturated weight, or None
        ramp_steps: length of the linear switch-on ramp, or None for no ramp
        ramp_start_t: absolute timestep at which the ramp begins
    """

    template: jnp.ndarray
    band_lo: float | None
    band_hi: float | None
    drho: float | None
    ramp_steps: float | None = None
    ramp_start_t: float = 0.0


def _ramp_fraction(t: jnp.ndarray, precomputed: GravityPrecomputed) -> jnp.ndarray | float:
    """Fraction of the full force in effect at timestep ``t``.

    ``ramp_steps`` is a build-time value, so an unramped run resolves to the
    Python float ``1.0`` at trace time and puts nothing in the graph.
    """
    if precomputed.ramp_steps is None:
        return 1.0
    return jnp.clip((t - precomputed.ramp_start_t) / precomputed.ramp_steps, 0.0, 1.0)


def _measured_phase_densities(config: SimulationConfig) -> tuple[float, float] | None:
    """Phase densities read off the run's init field, for ``init_from_file``.

    An equilibrated field's coexistence densities differ from the prescribed
    ones, and the buoyancy contrast reported in ``physical_parameters.txt`` is
    already measured off this same file — reusing its reader keeps the contrast
    the force *injects* equal to the one the run's Bond number is *reported*
    with. Imported lazily because ``build`` runs at setup time, outside JIT.
    """
    if config.init_type != "init_from_file":
        return None
    from src.simulation_io.analysis.physical_parameters import measure_init_phase_densities

    return measure_init_phase_densities(config)


def _phase_references(config: SimulationConfig) -> tuple[float, float] | None:
    """Return ``(rho_lo, rho_hi)`` for the indicator, or None for single phase."""
    if config.rho_v is None or config.rho_l is None:
        return None
    return _measured_phase_densities(config) or (float(config.rho_v), float(config.rho_l))


# ══════════════════════════════════════════════════════════════════════
# ForceOperator protocol — registry-backed module
# ══════════════════════════════════════════════════════════════════════


@force_model(name="gravity_masked_force")
class GravityForceModule:
    """Gravity force conforming to :class:`ForceOperator` protocol.

    Stateless — it uses the default no force state hooks.
    """

    @staticmethod
    def build(
        params: dict,
        grid_shape: tuple[int, ...],
        **kwargs: object,
    ) -> GravityPrecomputed:
        """Build a constant gravity-force template and the indicator's band.

        Args:
            params: Config dict from ``[gravity_masked_force]`` TOML section.
                Required key: ``force_g``.
                Optional keys: ``inclination_angle_deg`` (default 0),
                ``ramp_steps`` (default None, meaning the force is applied at
                full strength immediately) and ``ramp_start_t`` (default 0).
            grid_shape: Spatial dimensions ``(nx, ny, nz, ...)``.
            **kwargs: Additional arguments including ``lattice`` (for dimension info).

        Returns:
            A :class:`GravityPrecomputed` holding the constant template, the
            band edges and contrast of the phase indicator (all ``None`` when
            the config carries no phase densities) and the resolved switch-on
            ramp.

        Raises:
            ValueError: If ``ramp_steps`` is set to a non-positive value.
        """
        template = _build_gravity_template(params, grid_shape, **kwargs)

        # ``config`` arrives through ``**kwargs: object``; the concrete type is
        # restored here rather than reading it back with ``getattr`` chains that
        # would leave every field typed ``object``.
        config = cast("SimulationConfig | None", kwargs.get("config"))
        refs = _phase_references(config) if config is not None else None

        ramp_steps = params.get("ramp_steps")
        if ramp_steps is not None and float(ramp_steps) <= 0.0:
            msg = f"'ramp_steps' must be positive, got {ramp_steps!r}. Omit the key for no ramp."
            raise ValueError(msg)

        band_lo = band_hi = drho = None
        if refs is not None:
            rho_lo, rho_hi = refs
            drho = rho_hi - rho_lo
            band_lo = rho_lo + _PHASE_BAND * drho
            band_hi = rho_hi - _PHASE_BAND * drho

        return GravityPrecomputed(
            template=template,
            band_lo=band_lo,
            band_hi=band_hi,
            drho=drho,
            ramp_steps=float(ramp_steps) if ramp_steps is not None else None,
            ramp_start_t=float(params.get("ramp_start_t", 0.0)),
        )

    @staticmethod
    def compute(
        state: State,
        precomputed: GravityPrecomputed,
        **_kwargs: object,
    ) -> jnp.ndarray:
        """Compute the buoyancy-referenced body force (step-time, jittable).

        Args:
            state: Current simulation :class:`State`. Uses ``state.f`` (to
                compute density) and ``state.t`` (for the switch-on ramp).
            precomputed: Gravity template and band from :meth:`build`.

            **_kwargs: Additional arguments (ignored).

        Returns:
            Body force field, shape ``(nx, ny, nz, 1, d)``, weighted by the
            banded phase indicator: ``drho * g`` in the dense phase, exactly
            zero in the light phase.
        """
        rho = jnp.sum(state.f, axis=-2, keepdims=True)

        weight: jnp.ndarray | float
        if precomputed.drho is None:
            # Single-phase runs carry no phase densities, leaving the plain
            # local weight rho*g.
            weight = rho
        elif precomputed.drho <= 0.0:
            # Degenerate rho_l == rho_v: no contrast, hence no buoyancy. Branch
            # on the build-time float so the band's span never divides by zero.
            weight = 0.0
        else:
            assert precomputed.band_lo is not None  # noqa: S101 - set together with drho
            assert precomputed.band_hi is not None  # noqa: S101 - set together with drho
            span = precomputed.band_hi - precomputed.band_lo
            weight = precomputed.drho * jnp.clip((rho - precomputed.band_lo) / span, 0.0, 1.0)

        return -precomputed.template * _ramp_fraction(state.t, precomputed) * weight
