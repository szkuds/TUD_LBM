import sys
import builtins
import pytest


@pytest.fixture
def mock_optax_missing(monkeypatch):
    """
    Fixture: Simulate optax package not being installed.

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
        """Mock import that fails for optax"""
        if name == "optax" or name.startswith("optax."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        # Call original import for other modules
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)
    yield  # Run test
    # Cleanup happens automatically (monkeypatch fixture)


@pytest.fixture
def mock_optax_present(monkeypatch):
    """
    Fixture: Ensure optax package can be imported.

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


@pytest.fixture
def cleanup_imports():
    """
    Fixture: Clean module imports after each test.

    This fixture removes affected modules from sys.modules after each test,
    ensuring tests don't interfere with each other due to import caching.

    Use when tests might cache module state.

    Example:
        def test_something(cleanup_imports):
            # Import tud_lbm modules...
            # After test, sys.modules is cleaned
    """
    yield  # Run test first

    # Cleanup: Remove affected modules from sys.modules cache
    modules_to_clean = [
        "config",
        "config.simulation_config",
        "setup",
        "setup.simulation_setup",
        "setup.lattice",
        "state",
        "state.state",
        "runner",
        "runner.run",
        "runner.step",
        "operators",
    ]

    for module_name in modules_to_clean:
        sys.modules.pop(module_name, None)
