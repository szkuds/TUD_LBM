from functools import partial
from typing import Tuple, Any

import jax
import jax.numpy as jnp
from jax import jit
import optax

from src.operators import MacroscopicMultiphaseDW, ContactAngle, ContactLineLocation, WettingParameters
from src.core.grid.grid import Grid
from src.core.lattice.lattice import Lattice
from .update_multiphase import UpdateMultiphase


class UpdateMultiphaseHysteresis(UpdateMultiphase):
    """
    UpdateMultiphase with hysteresis optimization for wetting parameters.
    Optimizes both left and right sides and persists updated operators.
    """

    def __init__(
            self,
            grid: Grid,
            lattice: Lattice,
            tau: float,
            kappa: float,
            interface_width: int,
            rho_l: float,
            rho_v: float,
            bc_config: dict = None,
            force_enabled: bool = False,
            collision_scheme: str = "bgk",
            eos: str = "double-well",
            k_diag=None,
            **kwargs
    ):
        super().__init__(grid, lattice, tau, kappa, interface_width, rho_l, rho_v,
                         bc_config, force_enabled, collision_scheme, eos, k_diag, **kwargs)

        # Extract hysteresis parameters from bc_config
        self.hysteresis_params = bc_config['hysteresis_params']
        if 'chemical_step' in bc_config:
            self.chem_step = bc_config['chemical_step']
            self.ca_advancing_pre_step = self.chem_step.get('ca_advancing_pre_step')
            self.ca_advancing_post_step = self.chem_step.get('ca_advancing_post_step')
            self.ca_receding_pre_step = self.chem_step.get('ca_receding_pre_step')
            self.ca_receding_post_step = self.chem_step.get('ca_receding_post_step')
            self.chem_step_location = (self.grid.shape[0] //
                                       int(1 / self.chem_step['chemical_step_location']))

            self.determine_hysteresis_angles = self._determine_hysteresis_angles_chemical_step
        else:
            self.ca_advancing = self.hysteresis_params.get('ca_advancing', 120.0)
            self.ca_receding = self.hysteresis_params.get('ca_receding', 60.0)

            self.determine_hysteresis_angles = self._determine_hysteresis_angles

        # Setup contact angle and line location calculators
        rho_mean = (rho_l + rho_v) / 2
        self.contact_angle = ContactAngle(rho_mean)
        self.contact_line_location = ContactLineLocation(rho_mean)

        # Optimizer setup
        self.learning_rate = self.hysteresis_params.get('learning_rate', 0.01)
        self.max_iterations = self.hysteresis_params.get('max_iterations', 20)
        self.optimiser = optax.adam(self.learning_rate)

    @partial(jit, static_argnums=(0,))
    def __call__(self, f_t: jnp.ndarray, force: jnp.ndarray = None):
        f_tplus1 = super(UpdateMultiphaseHysteresis, self).__call__(f_t, force)

        if self.force_enabled and force is None:
            raise TypeError(
                "When the force is enabled an external force needs to be provided"
            )
        elif self.force_enabled:
            rho_t, _, _ = self.macroscopic(f_t, force)
            rho_tplus1, _, _ = self.macroscopic(f_tplus1, force)
        else:
            rho_t, _, _ = self.macroscopic(f_t)
            rho_tplus1, _, _ = self.macroscopic(f_tplus1)

        # Get the variables from the previous time step (t)
        ca_left_t, ca_right_t = self.contact_angle.compute(rho_t)
        cll_left_t, cll_right_t = self.contact_line_location.compute(rho_t, ca_left_t, ca_right_t)

        # Get the variables from the next time step (tplus1)
        ca_left_tplus1, ca_right_tplus1 = self.contact_angle.compute(rho_tplus1)
        cll_left_tplus1, cll_right_tplus1 = self.contact_line_location.compute(rho_tplus1, ca_left_tplus1,
                                                                               ca_right_tplus1)

        ca_advancing_left, ca_receding_left = self.determine_hysteresis_angles(cll_left_tplus1)
        ca_advancing_right, ca_receding_right = self.determine_hysteresis_angles(cll_right_tplus1)

        # Check for both sides if the ca_tplus1 is within the hysteresis window
        hysteresis_window_left = self._check_hysteresis_window(ca_left_tplus1, ca_advancing_left, ca_receding_left)
        hysteresis_window_right = self._check_hysteresis_window(ca_right_tplus1, ca_advancing_right, ca_receding_right)

        # Prepare common operand for left side (contains all needed data for both branches)
        operand_left = (ca_left_t, cll_left_t, ca_advancing_left, ca_receding_left, f_t, force)
        params_left = jax.lax.cond(
            hysteresis_window_left,
            lambda args: self._cll_pinned((args[0], args[1], args[4], args[5])),  # unpack only what _cll_pinned expects
            lambda args: self._ca_optimisation(args),  # pass full operand to _ca_optimisation
            operand_left
        )

        # Prepare common operand for right side
        operand_right = (ca_right_t, cll_right_t, ca_advancing_right, ca_receding_right, f_t, force)
        params_right = jax.lax.cond(
            hysteresis_window_right,
            lambda args: self._cll_pinned((args[0], args[1], args[4], args[5])),
            lambda args: self._ca_optimisation(args),
            operand_right
        )

        # Merge optimized parameters from both sides
        merged_params = WettingParameters(
            d_rho_left=params_left.d_rho_left,
            d_rho_right=params_right.d_rho_right,
            phi_left=params_left.phi_left,
            phi_right=params_right.phi_right
        )

        return self._evaluate_with_new_wetting_params(f_t, merged_params, force)[4]

    def _cost_fucntion_cll(self, cll_t: jnp.ndarray, cll_tplus1: jnp.ndarray):
        return jnp.abs(cll_t - cll_tplus1)

    def _cost_function_ca(self, ca_target: jnp.array, ca_tplus1: jnp.array):
        return jnp.abs(ca_target - ca_tplus1)

    def _clamp_wetting_params(self, params: WettingParameters) -> WettingParameters:
        """Clamp wetting parameters to physically reasonable ranges."""
        # Clamp d_rho values to reasonable range (e.g., 0 to 0.8)
        d_rho_left = jnp.clip(params.d_rho_left, 0, 0.2)
        d_rho_right = jnp.clip(params.d_rho_right, 0, 0.2)
        # Clamp phi values to reasonable range (e.g., 1 to 1.5)
        phi_left = jnp.clip(params.phi_left, 1.0, 1.5 )
        phi_right = jnp.clip(params.phi_right, 1.0, 1.5)
        return WettingParameters(d_rho_left, d_rho_right, phi_left, phi_right)

    def _get_current_params(self) -> WettingParameters:
        """Extract current wetting parameters from boundary condition config."""
        wetting_params = self.boundary_condition.bc_config['wetting_params']
        return WettingParameters(
            d_rho_left=jnp.array(wetting_params['d_rho_left']),
            d_rho_right=jnp.array(wetting_params['d_rho_right']),
            phi_left=jnp.array(wetting_params['phi_left']),
            phi_right=jnp.array(wetting_params['phi_right'])
        )

    @partial(jit, static_argnums=(0,))
    def _optimize_cll_left(self, initial_params: WettingParameters,
                           cll_t: jnp.ndarray, f_state: jnp.ndarray, force: jnp.ndarray) -> Tuple[
        WettingParameters, jnp.ndarray]:
        """Optimize left side to pin contact line location by testing both phi and d_rho."""

        # Objective for optimizing d_rho_left
        def objective_d_rho(params):
            _, _, cll_left, _, _ = self._evaluate_with_new_wetting_params(f_state, params, force)
            return self._cost_fucntion_cll(cll_t, cll_left)

        # Objective for optimizing phi_left
        def objective_phi(params):
            _, _, cll_left, _, _ = self._evaluate_with_new_wetting_params(f_state, params, force)
            return self._cost_fucntion_cll(cll_t, cll_left)

        opt_state_d_rho = self.optimiser.init(initial_params)
        opt_state_phi = self.optimiser.init(initial_params)

        # Optimization step for d_rho_left
        def step_d_rho(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(objective_d_rho)(params)
            # Only update d_rho_left, zero out other gradients
            grads = WettingParameters(
                d_rho_left=grads.d_rho_left,
                d_rho_right=jnp.zeros_like(grads.d_rho_right),
                phi_left=jnp.zeros_like(grads.phi_left),
                phi_right=jnp.zeros_like(grads.phi_right)
            )
            updates, opt_state = self.optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._clamp_wetting_params(params)
            return (params, opt_state), loss

        # Optimization step for phi_left
        def step_phi(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(objective_phi)(params)
            # Only update phi_left, zero out other gradients
            grads = WettingParameters(
                d_rho_left=jnp.zeros_like(grads.d_rho_left),
                d_rho_right=jnp.zeros_like(grads.d_rho_right),
                phi_left=grads.phi_left,
                phi_right=jnp.zeros_like(grads.phi_right)
            )
            updates, opt_state = self.optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._clamp_wetting_params(params)
            return (params, opt_state), loss

        # Run both optimizations
        (final_params_d_rho, _), losses_d_rho = jax.lax.scan(
            step_d_rho, (initial_params, opt_state_d_rho), jnp.arange(self.max_iterations)
        )
        (final_params_phi, _), losses_phi = jax.lax.scan(
            step_phi, (initial_params, opt_state_phi), jnp.arange(self.max_iterations)
        )

        # Choose the best result (lowest final loss)
        final_loss_d_rho = losses_d_rho[-1]
        final_loss_phi = losses_phi[-1]
        best_params = jax.lax.cond(
            final_loss_d_rho < final_loss_phi,
            lambda: final_params_d_rho,
            lambda: final_params_phi
        )
        best_loss = jnp.minimum(final_loss_d_rho, final_loss_phi)

        return best_params, best_loss

    @partial(jit, static_argnums=(0,))
    def _optimize_cll_right(self, initial_params: WettingParameters,
                            cll_t: jnp.ndarray, f_state: jnp.ndarray, force: jnp.ndarray) -> Tuple[
        WettingParameters, jnp.ndarray]:
        """Optimize right side to pin contact line location by testing both phi and d_rho."""

        # Objective for optimizing d_rho_right
        def objective_d_rho(params):
            _, _, _, cll_right, _ = self._evaluate_with_new_wetting_params(f_state, params, force)
            return self._cost_fucntion_cll(cll_t, cll_right)

        # Objective for optimizing phi_right
        def objective_phi(params):
            _, _, _, cll_right, _ = self._evaluate_with_new_wetting_params(f_state, params, force)
            return self._cost_fucntion_cll(cll_t, cll_right)

        opt_state_d_rho = self.optimiser.init(initial_params)
        opt_state_phi = self.optimiser.init(initial_params)

        # Optimization step for d_rho_right
        def step_d_rho(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(objective_d_rho)(params)
            # Only update d_rho_right, zero out other gradients
            grads = WettingParameters(
                d_rho_left=jnp.zeros_like(grads.d_rho_left),
                d_rho_right=grads.d_rho_right,
                phi_left=jnp.zeros_like(grads.phi_left),
                phi_right=jnp.zeros_like(grads.phi_right)
            )
            updates, opt_state = self.optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._clamp_wetting_params(params)
            return (params, opt_state), loss

        # Optimization step for phi_right
        def step_phi(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(objective_phi)(params)
            # Only update phi_right, zero out other gradients
            grads = WettingParameters(
                d_rho_left=jnp.zeros_like(grads.d_rho_left),
                d_rho_right=jnp.zeros_like(grads.d_rho_right),
                phi_left=jnp.zeros_like(grads.phi_left),
                phi_right=grads.phi_right
            )
            updates, opt_state = self.optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._clamp_wetting_params(params)
            return (params, opt_state), loss

        # Run both optimizations
        (final_params_d_rho, _), losses_d_rho = jax.lax.scan(
            step_d_rho, (initial_params, opt_state_d_rho), jnp.arange(self.max_iterations)
        )
        (final_params_phi, _), losses_phi = jax.lax.scan(
            step_phi, (initial_params, opt_state_phi), jnp.arange(self.max_iterations)
        )

        # Choose the best result (lowest final loss)
        final_loss_d_rho = losses_d_rho[-1]
        final_loss_phi = losses_phi[-1]
        best_params = jax.lax.cond(
            final_loss_d_rho < final_loss_phi,
            lambda: final_params_d_rho,
            lambda: final_params_phi
        )
        best_loss = jnp.minimum(final_loss_d_rho, final_loss_phi)

        return best_params, best_loss

    @partial(jit, static_argnums=(0,))
    def _optimize_ca_left(self, initial_params: WettingParameters,
                          ca_target: jnp.ndarray, f_state: jnp.ndarray, force: jnp.ndarray) -> Tuple[
        WettingParameters, jnp.ndarray]:
        """Optimize left side contact angle by testing both phi and d_rho."""

        # Objective for optimizing d_rho_left
        def objective_d_rho(params):
            ca_left, _, _, _, _ = self._evaluate_with_new_wetting_params(f_state, params, force)
            return self._cost_function_ca(ca_target, ca_left)

        # Objective for optimizing phi_left
        def objective_phi(params):
            ca_left, _, _, _, _ = self._evaluate_with_new_wetting_params(f_state, params, force)
            return self._cost_function_ca(ca_target, ca_left)

        opt_state_d_rho = self.optimiser.init(initial_params)
        opt_state_phi = self.optimiser.init(initial_params)

        # Optimization step for d_rho_left
        def step_d_rho(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(objective_d_rho)(params)
            # Only update d_rho_left, zero out other gradients
            grads = WettingParameters(
                d_rho_left=grads.d_rho_left,
                d_rho_right=jnp.zeros_like(grads.d_rho_right),
                phi_left=jnp.zeros_like(grads.phi_left),
                phi_right=jnp.zeros_like(grads.phi_right)
            )
            updates, opt_state = self.optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._clamp_wetting_params(params)
            return (params, opt_state), loss

        # Optimization step for phi_left
        def step_phi(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(objective_phi)(params)
            # Only update phi_left, zero out other gradients
            grads = WettingParameters(
                d_rho_left=jnp.zeros_like(grads.d_rho_left),
                d_rho_right=jnp.zeros_like(grads.d_rho_right),
                phi_left=grads.phi_left,
                phi_right=jnp.zeros_like(grads.phi_right)
            )
            updates, opt_state = self.optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._clamp_wetting_params(params)
            return (params, opt_state), loss

        # Run both optimizations
        (final_params_d_rho, _), losses_d_rho = jax.lax.scan(
            step_d_rho, (initial_params, opt_state_d_rho), jnp.arange(self.max_iterations)
        )
        (final_params_phi, _), losses_phi = jax.lax.scan(
            step_phi, (initial_params, opt_state_phi), jnp.arange(self.max_iterations)
        )

        # Choose the best result (lowest final loss)
        final_loss_d_rho = losses_d_rho[-1]
        final_loss_phi = losses_phi[-1]
        best_params = jax.lax.cond(
            final_loss_d_rho < final_loss_phi,
            lambda: final_params_d_rho,
            lambda: final_params_phi
        )
        best_loss = jnp.minimum(final_loss_d_rho, final_loss_phi)

        return best_params, best_loss

    @partial(jit, static_argnums=(0,))
    def _optimize_ca_right(self, initial_params: WettingParameters,
                           ca_target: jnp.ndarray, f_state: jnp.ndarray, force: jnp.ndarray) -> Tuple[
        WettingParameters, jnp.ndarray]:
        """Optimize right side contact angle by testing both phi and d_rho."""

        # Objective for optimizing d_rho_right
        def objective_d_rho(params):
            _, ca_right, _, _, _ = self._evaluate_with_new_wetting_params(f_state, params, force)
            return self._cost_function_ca(ca_target, ca_right)

        # Objective for optimizing phi_right
        def objective_phi(params):
            _, ca_right, _, _, _ = self._evaluate_with_new_wetting_params(f_state, params, force)
            return self._cost_function_ca(ca_target, ca_right)

        opt_state_d_rho = self.optimiser.init(initial_params)
        opt_state_phi = self.optimiser.init(initial_params)

        # Optimization step for d_rho_right
        def step_d_rho(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(objective_d_rho)(params)
            # Only update d_rho_right, zero out other gradients
            grads = WettingParameters(
                d_rho_left=jnp.zeros_like(grads.d_rho_left),
                d_rho_right=grads.d_rho_right,
                phi_left=jnp.zeros_like(grads.phi_left),
                phi_right=jnp.zeros_like(grads.phi_right)
            )
            updates, opt_state = self.optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._clamp_wetting_params(params)
            return (params, opt_state), loss

        # Optimization step for phi_right
        def step_phi(carry, _):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(objective_phi)(params)
            # Only update phi_right, zero out other gradients
            grads = WettingParameters(
                d_rho_left=jnp.zeros_like(grads.d_rho_left),
                d_rho_right=jnp.zeros_like(grads.d_rho_right),
                phi_left=jnp.zeros_like(grads.phi_left),
                phi_right=grads.phi_right
            )
            updates, opt_state = self.optimiser.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            params = self._clamp_wetting_params(params)
            return (params, opt_state), loss

        # Run both optimizations
        (final_params_d_rho, _), losses_d_rho = jax.lax.scan(
            step_d_rho, (initial_params, opt_state_d_rho), jnp.arange(self.max_iterations)
        )
        (final_params_phi, _), losses_phi = jax.lax.scan(
            step_phi, (initial_params, opt_state_phi), jnp.arange(self.max_iterations)
        )

        # Choose the best result (lowest final loss)
        final_loss_d_rho = losses_d_rho[-1]
        final_loss_phi = losses_phi[-1]
        best_params = jax.lax.cond(
            final_loss_d_rho < final_loss_phi,
            lambda: final_params_d_rho,
            lambda: final_params_phi
        )
        best_loss = jnp.minimum(final_loss_d_rho, final_loss_phi)

        return best_params, best_loss

    @partial(jit, static_argnums=(0,))
    def _check_hysteresis_window(self, ca_tplus1: jnp.ndarray, ca_advancing, ca_receding):
        return jnp.logical_and(
            ca_tplus1 >= ca_receding,
            ca_tplus1 <= ca_advancing
        )

    def _create_updated_macroscopic(self, params: WettingParameters):
        """Create new macroscopic operator with updated wetting parameters."""
        new_wetting_params = {
            **self.macroscopic.bc_config.get('wetting_params', {}),
            'd_rho_left': params.d_rho_left,
            'd_rho_right': params.d_rho_right,
            'phi_left': params.phi_left,
            'phi_right': params.phi_right,
        }
        new_bc_config = {
            **self.macroscopic.bc_config,
            'wetting_params': new_wetting_params
        }
        new_macroscopic = MacroscopicMultiphaseDW(
            self.grid, self.lattice, self.macroscopic.kappa, new_wetting_params['width'], new_wetting_params['rho_l'],
            new_wetting_params['rho_v'], self.force_enabled, new_bc_config
        )
        return new_macroscopic

    def _run_timestep_with_new_wetting_params(self, f: jnp.ndarray, params: WettingParameters,
                                              force: jnp.ndarray = None) -> tuple[jnp.ndarray, Any]:
        """
        Run a complete LBM timestep with given wetting parameters.

        This method creates temporary operators with updated parameters and executes
        the full collision-streaming-BC cycle, mimicking super().__call__() behavior.

        Args:
            f: Population distribution at current timestep, shape (nx, ny, q, 1)
            params: WettingParameters containing d_rho_left, d_rho_right, phi_left, phi_right
            force: External force field, shape (nx, ny, 1, 2), optional

        Returns:
            jnp.ndarray: Updated population distribution f_{t+1}, shape (nx, ny, q, 1)
        """
        # Create updated operators
        temp_macroscopic = self._create_updated_macroscopic(params)

        # Macroscopic step (same logic as parent Update.__call__)
        if self.force_enabled:
            rho_prev, u, force_tot = temp_macroscopic(f, force=force)
        else:
            rho_prev, u, force_tot = temp_macroscopic(f)
        feq = self.equilibrium(rho_prev, u)
        source = self.source_term(rho_prev, u, force_tot)
        fcol = self.collision(f, feq, source)
        fstream = self.streaming(fcol)
        if self.boundary_condition is not None:
            fbc = self.boundary_condition(fstream, fcol)
            return fbc, temp_macroscopic
        else:
            return fstream, temp_macroscopic

    def _evaluate_with_new_wetting_params(self, f_t: jnp.ndarray, params: WettingParameters, force: jnp.ndarray = None):
        """Evaluate contact angles and line locations with given wetting parameters (both sides)."""
        f_tplus1, temp_macroscopic = self._run_timestep_with_new_wetting_params(f_t, params, force)
        rho_tplus1, _, _ = temp_macroscopic(f_tplus1)
        ca_left, ca_right = self.contact_angle.compute(rho_tplus1)
        cll_left, cll_right = self.contact_line_location.compute(rho_tplus1, ca_left, ca_right)
        return ca_left, ca_right, cll_left, cll_right, f_tplus1

    def _cll_pinned(self, args):
        """When CA is within hysteresis window, pin the contact line location."""
        ca_t, cll_t, f_state, force = args
        # Get current parameters and optimize to maintain CLL
        initial_params = self._get_current_params()

        # Optimize both left and right sides concurrently
        params_left, _ = self._optimize_cll_left(initial_params, cll_t, f_state, force)
        params_right, _ = self._optimize_cll_right(initial_params, cll_t, f_state, force)

        # Merge the optimized parameters (left side from params_left, right side from params_right)
        merged_params = WettingParameters(
            d_rho_left=params_left.d_rho_left,
            d_rho_right=params_right.d_rho_right,
            phi_left=params_left.phi_left,
            phi_right=params_right.phi_right
        )
        return merged_params

    @partial(jit, static_argnums=(0,))
    def _ca_optimisation(self, args):
        """When CA is outside hysteresis window, optimize to reach target CA."""
        ca_t, cll_t, ca_advancing, ca_receding, f_state, force = args

        # Determine target CA (advancing or receding based on current CA)
        # If CA < receding, target is receding; if CA > advancing, target is advancing
        ca_target = jax.lax.cond(
            ca_t < ca_receding,
            lambda: ca_receding,
            lambda: ca_advancing
        )

        # Get current parameters and optimize to reach target CA
        initial_params = self._get_current_params()

        # Optimize both left and right sides concurrently
        params_left, _ = self._optimize_ca_left(initial_params, ca_target, f_state, force)
        params_right, _ = self._optimize_ca_right(initial_params, ca_target, f_state, force)

        # Merge the optimized parameters
        merged_params = WettingParameters(
            d_rho_left=params_left.d_rho_left,
            d_rho_right=params_right.d_rho_right,
            phi_left=params_left.phi_left,
            phi_right=params_right.phi_right
        )
        return merged_params

    def _determine_hysteresis_angles_chemical_step(self, cll):
        ca_advancing, ca_receding = jax.lax.cond(
            cll < self.chem_step_location,
            lambda: self._pre_chem_step(),
            lambda: self._post_chem_step()
        )
        return ca_advancing, ca_receding

    def _pre_chem_step(self):
        return self.ca_advancing_pre_step, self.ca_receding_pre_step

    def _post_chem_step(self):
        return self.ca_advancing_post_step, self.ca_receding_post_step

    def _determine_hysteresis_angles(self, cll):
        return self.ca_advancing, self.ca_receding
