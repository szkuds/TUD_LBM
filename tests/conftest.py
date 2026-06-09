import builtins
import sys
import pytest


@pytest.fixture(scope="session", autouse=True)
def _register_step_operators() -> None:
    """Auto-register step operators for all tests.

    This fixture ensures that the runner modules are imported
    early, so their @update_timestep_operator decorators fire and
    register the step operators.
    """
    from tud_lbm.operators import step as _  # noqa: F401


@pytest.fixture
def mock_optax_missing(monkeypatch):
    """Fixture: Simulate optax package not being installed.

    This fixture removes optax from sys.modules and makes import attempts fail.
    Use when testing behavior WITHOUT optax installed.

    Example:
        def test_something(mock_optax_missing):
            # optax is not available in this test
            ...
    """
    # Remove optax from sys.modules if present
    monkeypatch.setitem(sys.modules, "optax", None)

    # Store original __import__
    original_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        """Mock import that fails for optax."""
        if name == "optax" or name.startswith("optax."):
            msg = f"No module named '{name}'"
            raise ModuleNotFoundError(msg)
        # Call original import for other modules
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    # Cleanup happens automatically (monkeypatch fixture)


@pytest.fixture
def mock_optax_present(monkeypatch):
    """Fixture: Ensure optax package can be imported.

    This fixture ensures optax is available for testing WITH optax installed.
    Use when testing behavior that REQUIRES optax.

    Example:
        def test_hysteresis_works(mock_optax_present):
            # optax is available in this test
            ...
    """
    # Try to import optax to ensure it exists
    try:
        import optax  # noqa: F401

        yield  # Run test
    except ImportError:
        pytest.skip("optax not installed - skipping test that requires it")
