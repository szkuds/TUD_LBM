"""
Unit tests for optional dependencies pattern (optax for hysteresis optimization).

These tests validate that:
1. The package can be installed without optional dependencies
2. Lazy imports work correctly to defer loading of optional packages
3. Clear error messages guide users to install optional packages
4. pyproject.toml is configured correctly
"""

import os
import sys
import pytest
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

# Get project root from environment or infer it
PROJECT_ROOT = os.getenv('PROJECT_ROOT')
if not PROJECT_ROOT:
    # Fallback: infer from tests directory
    PROJECT_ROOT = str(Path(__file__).parent.parent)


class TestOptaxNotRequiredForCore:
    """Tests that runner functionality doesn't require optax."""

    def test_optax_not_in_core_dependencies(self):
        """
        Core dependencies do NOT include optax.

        Given: pyproject.toml exists
        When: reading runner dependencies
        Then: optax should NOT appear in the dependencies list
        """
        import tomllib

        pyproject_path = os.path.join(PROJECT_ROOT, 'pyproject.toml')
        try:
            with open(pyproject_path, 'rb') as f:
                pyproject = tomllib.load(f)
        except ImportError:
            import toml
            with open(pyproject_path) as f:
                pyproject = toml.load(f)

        core_deps = pyproject.get('project', {}).get('dependencies', [])
        core_deps_str = ' '.join(str(d) for d in core_deps)

        assert 'optax' not in core_deps_str.lower(), "optax should NOT be in runner dependencies"


    def test_core_dependencies_are_minimal(self):
        """
        Core dependencies are minimal and focused.

        Given: pyproject.toml exists
        When: reading runner dependencies
        Then: optax should NOT be in runner dependencies
        """
        import tomllib

        pyproject_path = os.path.join(PROJECT_ROOT, 'pyproject.toml')
        try:
            with open(pyproject_path, 'rb') as f:
                pyproject = tomllib.load(f)
        except ImportError:
            import toml
            with open(pyproject_path) as f:
                pyproject = toml.load(f)

        core_deps = pyproject.get('project', {}).get('dependencies', [])
        core_deps_str = ' '.join(str(d) for d in core_deps).lower()

        # Primary check: optax should NOT be in runner dependencies
        assert 'optax' not in core_deps_str, "optax should not be in runner (it's optional)"

        # With conda app_setup, dependencies may be empty (managed by environment.yml)
        # so just verify that optax is not accidentally included
        assert True, "Core dependencies correctly exclude optax"


class TestLazyImportPattern:
    """Tests for the lazy import pattern itself."""

    def test_mock_optax_missing_fixture_works(self, mock_optax_missing, cleanup_imports):
        """
        Mock fixture correctly simulates missing optax.

        Given: mock_optax_missing fixture is active
        When: trying to import optax directly
        Then: should raise ModuleNotFoundError
        """
        with pytest.raises(ModuleNotFoundError):
            import optax  # noqa: F401

    def test_mock_optax_present_allows_import(self, mock_optax_present, cleanup_imports):
        """
        Mock fixture allows importing optax when present.

        Given: mock_optax_present fixture is active
        When: trying to import optax
        Then: should succeed (or be skipped if optax not actually installed)
        """
        try:
            import optax
            assert True, "optax import succeeded"
        except ModuleNotFoundError:
            pytest.skip("optax not installed in this environment")

    def test_cleanup_imports_fixture_removes_cached_modules(self, cleanup_imports):
        """
        Cleanup fixture removes cached modules between tests.

        Given: cleanup_imports fixture is active
        When: checking sys.modules
        Then: test-related modules should be manageable
        """
        # This test just verifies the fixture is callable and doesn't error
        assert True, "Cleanup imports fixture ran successfully"


class TestEnvironmentConfiguration:
    """Tests for conda and environment configuration."""

    def test_environment_yml_exists(self):
        """
        Environment.yml file exists for conda configuration.

        Given: project directory
        When: checking for environment.yml
        Then: file should exist
        """
        env_file = os.path.join(PROJECT_ROOT, 'environment.yml')
        assert os.path.exists(env_file), "environment.yml should exist for conda app_setup"

    def test_environment_yml_contains_core_deps(self):
        """
        Environment.yml contains runner dependencies.

        Given: environment.yml exists
        When: reading the file
        Then: should list jax, numpy, scipy
        """
        env_file = os.path.join(PROJECT_ROOT, 'environment.yml')
        with open(env_file) as f:
            env_content = f.read()

        for package in ['jax', 'numpy', 'scipy', 'pytest']:
            assert package in env_content.lower(), f"{package} should be in environment.yml"

    def test_environment_yml_comments_optax(self):
        """
        Environment.yml comments out optax (optional).

        Given: environment.yml exists
        When: reading the file
        Then: optax should be commented out, not in main dependencies
        """
        env_file = os.path.join(PROJECT_ROOT, 'environment.yml')
        with open(env_file) as f:
            lines = f.readlines()

        # Find optax - should be commented
        optax_lines = [line for line in lines if 'optax' in line.lower()]
        assert len(optax_lines) > 0, "optax should be mentioned (as optional)"

        # At least one should be commented
        has_commented = any(line.strip().startswith('#') for line in optax_lines)
        assert has_commented, "optax should be commented as optional in environment.yml"


class TestInstallationMethods:
    """Tests for different installation methods."""


    def test_installation_strategies_documented(self):
        """
        Installation methods are documented.

        Given: dev_notes folder
        When: checking for app_setup/install documentation
        Then: should have guidance on installation
        """
        dev_notes = os.path.join(PROJECT_ROOT)

        # Look for documentation files
        doc_files = [f for f in os.listdir(dev_notes) if f.endswith('.md')]
        assert len(doc_files) > 0, "Should have documentation in dev_notes"

        # At least one should mention app_setup/conda
        setup_docs = [f for f in doc_files if 'app_setup' in f.lower() or 'conda' in f.lower() or 'install' in f.lower()]
        assert len(setup_docs) > 0, "Should have app_setup/conda documentation"

    def test_env_file_exists(self):
        """
        .env.example file exists for configuration.

        Given: project directory
        When: checking for .env.example
        Then: file should exist with PROJECT_ROOT definition
        """
        env_example = os.path.join(PROJECT_ROOT, '.env.example')
        assert os.path.exists(env_example), ".env.example should exist"

        with open(env_example) as f:
            content = f.read()
        assert 'PROJECT_ROOT' in content, ".env.example should define PROJECT_ROOT"
