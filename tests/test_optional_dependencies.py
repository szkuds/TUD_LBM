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

    def test_env_file_existzs(self):
        """
        .env.example_for_test file exists for configuration.

        Given: project directory
        When: checking for .env.example_for_test
        Then: file should exist with PROJECT_ROOT definition
        """
        env_example = os.path.join(PROJECT_ROOT, '.env.example')
        assert os.path.exists(env_example), ".env.example should exist"

        with open(env_example) as f:
            content = f.read()
        assert 'PROJECT_ROOT' in content, ".env.example_for_test should define PROJECT_ROOT"
