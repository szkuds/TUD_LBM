"""Pressure field plot operators.

Two views of the pressure a run saves in its snapshots — the ``pressure`` field
the macroscopic operator returns every step, next to ``force``:

``pressure``
    The bulk pressure alone: the EOS ``p_0(rho)`` for a multiphase run, the
    lattice ideal gas ``cs^2 * rho`` for a single-phase one. Diagnostic for the
    EOS itself, but in a multiphase run it swings sharply across a diffuse
    interface because the interfacial ``kappa`` terms are missing.

``pressure_total``
    The full normal pressure ``p = p_0 - kappa * (rho * lap(rho) + |grad rho|^2 / 2)``,
    multiphase only. The interfacial terms largely cancel the ``p_0`` swing,
    leaving the Laplace jump between the bulk phases.

A snapshot that carries no ``pressure`` (it is not in the run's ``save_fields``,
or the snapshot predates the field) gets it recomputed from the saved ``rho``
with the same function the macroscopic operator uses, so the panels render for
any run that saved its density.

Both are opt-in: they only render when named in ``plot_fields`` /
``animate_fields``, so they do not change the default four-panel figure.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.registry import plotting_operator
from src.simulation_io.plotting.base import PlotOperator
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    import matplotlib.axes
    from src.config.multiphase_params import MultiphaseParams
    from src.operators.protocols import DifferentialOperator
    from src.operators.protocols import EosOperator


class _BasePressureOperator(PlotOperator):
    """Shared slicing and panel styling for the pressure views."""

    title: str = "Pressure"
    opt_in = True

    _pressure_fn: EosOperator | None = None

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        """Whether the snapshot carries the pressure, or the density it follows from."""
        return "pressure" in data or "rho" in data

    def _bulk_pressure(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Return the bulk pressure in the snapshot's 5-D layout.

        The saved field when the snapshot has one. Otherwise it is recomputed
        from ``rho`` exactly as the macroscopic operator computes it each step:
        the EOS ``p_0(rho)`` for a multiphase run, ``cs^2 * rho`` for a
        single-phase one.
        """
        if "pressure" in data:
            return np.asarray(data["pressure"])

        rho = np.asarray(data["rho"])
        mp = self.config.multiphase_params
        if mp is None:
            from src.operators.macroscopic._single_phase import CS2

            return CS2 * rho

        if self._pressure_fn is None:
            from src.operators.macroscopic.eos import build_pressure_fn

            self._pressure_fn = build_pressure_fn(mp)
        import jax.numpy as jnp

        return np.asarray(self._pressure_fn(jnp.asarray(rho)))

    def _pressure_2d(self, data: dict[str, np.ndarray]) -> np.ndarray:
        """Return the pressure field as a 2-D ``(ny, nx)`` array ready for imshow."""
        raise NotImplementedError

    def __call__(
        self,
        ax: matplotlib.axes.Axes,
        data: dict[str, np.ndarray],
        timestep: int,
    ) -> None:
        """Render the pressure field as a 2-D colour map."""
        pressure = self._pressure_2d(data)
        im = ax.imshow(pressure, origin="lower", aspect="equal", cmap=DEFAULT_STYLE.colormap_pressure)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="p")
        ax.set_title(f"{self.title}  t={timestep}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")


@plotting_operator(name="pressure")
class BulkPressurePlotOperator(_BasePressureOperator):
    """Render the bulk pressure of the snapshot."""

    name = "pressure"
    title = "Bulk pressure"

    def _pressure_2d(self, data: dict[str, np.ndarray]) -> np.ndarray:
        return self._bulk_pressure(data)[:, :, 0, 0, 0].T


@plotting_operator(name="pressure_total")
class TotalPressurePlotOperator(_BasePressureOperator):
    """Render the full normal pressure, including the interfacial ``kappa`` terms."""

    name = "pressure_total"
    title = "Total pressure"

    _mp: MultiphaseParams | None = None
    _diff_ops: tuple[DifferentialOperator, DifferentialOperator] | None = None

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        """Multiphase only: the interfacial terms need ``kappa``."""
        return "rho" in data and self.config.is_multiphase

    def _operators(self) -> tuple[MultiphaseParams, DifferentialOperator, DifferentialOperator]:
        """Return the cached ``(mp, gradient_density, laplacian_density)``.

        Built lazily from the run's own config, so they inherit its
        boundary-condition pad modes and any fixed-wetting ghost-cell
        correction, and constructing the operator stays cheap for runs where
        this panel never renders.
        """
        if self._mp is None or self._diff_ops is None:
            from src.lattice.lattice import build_lattice
            from src.operators.differential import build_diff_ops

            mp = self.config.multiphase_params
            assert mp is not None  # noqa: S101 - is_available admits multiphase configs only
            lattice = build_lattice(self.config.lattice_type)
            _, gradient_density, laplacian_density = build_diff_ops(self.config, mp, lattice)
            self._mp = mp
            self._diff_ops = (gradient_density, laplacian_density)
        return self._mp, *self._diff_ops

    def _pressure_2d(self, data: dict[str, np.ndarray]) -> np.ndarray:
        import jax.numpy as jnp

        mp, gradient_density, laplacian_density = self._operators()

        # Everything stays in the 5-D (nx, ny, nz, ., .) layout the snapshot is
        # saved in; slice to 2-D only at the end.
        rho = jnp.asarray(data["rho"])
        laplacian = np.asarray(laplacian_density(rho))
        gradient = np.asarray(gradient_density(rho))
        grad_sq = np.sum(gradient**2, axis=-1, keepdims=True)

        rho_np = np.asarray(rho)
        # Sign convention matches the force pipeline's mu = mu_0 - kappa * lap(rho).
        pressure = self._bulk_pressure(data) - mp.kappa * (rho_np * laplacian + 0.5 * grad_sq)
        return pressure[:, :, 0, 0, 0].T
