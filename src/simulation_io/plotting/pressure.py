"""Pressure field plot operators.

Two views of the same field, both built on the bulk pressure ``p_0(rho)`` that
the Young-Laplace surface-tension calibration uses:

``pressure``
    The bulk thermodynamic pressure alone. Diagnostic for the EOS itself, but
    it swings sharply across a diffuse interface because the interfacial
    ``kappa`` terms are missing.

``pressure_total``
    The full normal pressure ``p = p_0 - kappa * (rho * lap(rho) + |grad rho|^2 / 2)``.
    The interfacial terms largely cancel the ``p_0`` swing, leaving the Laplace
    jump between the bulk phases.

Both are opt-in: they only render when named in ``plot_fields`` /
``animate_fields``, so they do not change the default four-panel figure.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from src.operators.macroscopic import eos as _eos  # noqa: F401  registers the pressure operators
from src.registry import get_operator_names
from src.registry import plotting_operator
from src.simulation_io.plotting.base import PlotOperator
from src.simulation_io.plotting.figure_config import DEFAULT_STYLE

if TYPE_CHECKING:
    from collections.abc import Callable
    import matplotlib.axes
    from src.operators.macroscopic import MultiphaseParams
    from src.operators.protocols import DifferentialOperator


class _BasePressureOperator(PlotOperator):
    """Shared slicing, EOS wiring and panel styling for the pressure views."""

    title: str = "Pressure"
    opt_in = True

    _mp: MultiphaseParams | None = None
    _pressure_fn: Callable[[np.ndarray], np.ndarray] | None = None

    def is_available(self, data: dict[str, np.ndarray]) -> bool:
        """Whether the snapshot and config can produce a pressure field.

        Guards single-phase runs and any EOS without a bulk-pressure
        implementation, so an unsupported run silently drops the panel rather
        than rendering an error into it. The registry is the source of truth
        for which EOS have one, so a newly added pressure operator becomes
        available here with no edit to this file.
        """
        return "rho" in data and self.config.is_multiphase and self.config.eos in get_operator_names("pressure")

    def _params(self) -> tuple[MultiphaseParams, Callable[[np.ndarray], np.ndarray]]:
        """Return the cached ``(mp, pressure_fn)`` pair, building it on first use.

        Built lazily so constructing an operator stays cheap: ``FigureBuilder``
        instantiates every requested operator up front, including for runs
        where this panel never renders.
        """
        if self._mp is None or self._pressure_fn is None:
            from src.operators.macroscopic import build_multiphase_params
            from src.operators.macroscopic.eos import build_pressure_fn

            self._mp = build_multiphase_params(self.config)
            self._pressure_fn = build_pressure_fn(self._mp)
        return self._mp, self._pressure_fn

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
    """Render the bulk thermodynamic pressure ``p_0(rho)``."""

    name = "pressure"
    title = "Bulk pressure"

    def _pressure_2d(self, data: dict[str, np.ndarray]) -> np.ndarray:
        _, pressure_fn = self._params()
        rho = np.asarray(data["rho"])[:, :, 0, 0, 0]
        return np.asarray(pressure_fn(rho)).T


@plotting_operator(name="pressure_total")
class TotalPressurePlotOperator(_BasePressureOperator):
    """Render the full normal pressure, including the interfacial ``kappa`` terms."""

    name = "pressure_total"
    title = "Total pressure"

    _diff_ops: tuple[DifferentialOperator, DifferentialOperator] | None = None

    def _differentials(self) -> tuple[DifferentialOperator, DifferentialOperator]:
        """Return the cached ``(gradient_density, laplacian_density)`` closures.

        Built from the run's own config, so they inherit its boundary-condition
        pad modes and any fixed-wetting ghost-cell correction.
        """
        if self._diff_ops is None:
            from src.lattice.lattice import build_lattice
            from src.operators.differential import build_diff_ops

            mp, _ = self._params()
            lattice = build_lattice(self.config.lattice_type)
            _, gradient_density, laplacian_density, _, _ = build_diff_ops(self.config, mp, lattice)
            self._diff_ops = (gradient_density, laplacian_density)
        return self._diff_ops

    def _pressure_2d(self, data: dict[str, np.ndarray]) -> np.ndarray:
        import jax.numpy as jnp

        mp, pressure_fn = self._params()
        gradient_density, laplacian_density = self._differentials()

        # The differential closures work on the 5-D (nx, ny, nz, 1, 1) layout
        # the snapshot is already saved in; slice to 2-D only at the end.
        rho = jnp.asarray(data["rho"])
        laplacian = np.asarray(laplacian_density(rho))
        gradient = np.asarray(gradient_density(rho))
        grad_sq = np.sum(gradient**2, axis=-1, keepdims=True)

        rho_np = np.asarray(rho)
        # Sign convention matches the force pipeline's mu = mu_0 - kappa * lap(rho).
        pressure = pressure_fn(rho_np) - mp.kappa * (rho_np * laplacian + 0.5 * grad_sq)
        return pressure[:, :, 0, 0, 0].T
