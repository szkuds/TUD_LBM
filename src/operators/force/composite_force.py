# TODO: I want the composite force to pass whatever is required, I think this should be done with kwargs.

import jax.numpy as jnp
from src.operators.force.force import Force


class CompositeForce(Force):
    """
    Combines multiple force fields by superposition.
    Allows gravitational, electrical, and other forces to work together.
    """

    def __init__(self, *forces: Force):
        """
        Initialize composite force with multiple force components.

        Args:
            *forces: Variable number of Force objects to combine
        """
        if not forces:
            raise ValueError("At least one force must be provided")

        self.forces = forces

        # Check if electrical force is present
        self.electric_present = any(
            getattr(force, 'name', None) == 'ElectricalForce'
            for force in forces
        )

        # Initialize with zeros; actual force is computed dynamically
        super().__init__(jnp.zeros_like(forces[0].force))

    def compute_force(
        self, *args, **kwargs
    ) -> jnp.ndarray:
        """
        Sum contributions from all forces.
        Passes all arguments to each force's compute_force method.

        Returns:
            Combined force array of shape (nx, ny, 1, d)
        """
        total_force = None

        for force in self.forces:
            force_contribution = force.compute_force(**kwargs)
            if total_force is None:
                total_force = force_contribution
            else:
                total_force = total_force + force_contribution

        self.force = total_force
        return total_force

    def get_component_by_name(self, forces: tuple, target_name: str):
        for f in forces:
            if f.name == target_name:
                return f
        raise KeyError(f"No force with name {target_name!r}")
