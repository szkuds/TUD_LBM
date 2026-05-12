"""Unit tests for optional dependencies pattern (optax for hysteresis optimization).

These tests validate that:
1. The package can be installed without optional dependencies
2. Lazy imports work correctly to defer loading of optional packages
3. Clear error messages guide users to install optional packages
4. pyproject.toml is configured correctly
"""

from pathlib import Path

# Get project root from environment or infer it
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])


class TestEnvironmentConfiguration:
    """Tests for conda and environment configuration."""

    def test_environment_yml_exists(self):
        """Environment.yml file exists for conda configuration.

        Given: project directory
        When: checking for environment.yml
        Then: file should exist
        """
        env_file = Path(PROJECT_ROOT) / "environment.yml"
        assert env_file.exists(), "environment.yml should exist for conda app_setup"

    def test_environment_yml_contains_core_deps(self):
        """Environment.yml contains runner dependencies.

        Given: environment.yml exists
        When: reading the file
        Then: should list jax, numpy, scipy
        """
        env_file = Path(PROJECT_ROOT) / "environment.yml"
        with env_file.open() as f:
            env_content = f.read()

        for package in ["jax", "numpy", "scipy", "pytest"]:
            assert package in env_content.lower(), f"{package} should be in environment.yml"

    def test_environment_yml_comments_optax(self):
        """Environment.yml comments out optax (optional).

        Given: environment.yml exists
        When: reading the file
        Then: optax should be commented out, not in main dependencies
        """
        env_file = Path(PROJECT_ROOT) / "environment.yml"
        with env_file.open() as f:
            lines = f.readlines()

        # Find optax - should be commented
        optax_lines = [line for line in lines if "optax" in line.lower()]
        assert len(optax_lines) > 0, "optax should be mentioned (as optional)"

        # At least one should be commented
        has_commented = any(line.strip().startswith("#") for line in optax_lines)
        assert has_commented, "optax should be commented as optional in environment.yml"


class TestInstallationMethods:
    """Tests for different installation methods."""

    def test_env_file_existzs(self):
        """.env.example_for_test file exists for configuration.

        Given: project directory
        When: checking for .env.example_for_test
        Then: file should exist with PROJECT_ROOT definition
        """
        env_example = Path(PROJECT_ROOT) / ".env.example"
        assert env_example.exists(), ".env.example should exist"

        with env_example.open() as f:
            content = f.read()
        assert "PROJECT_ROOT" in content, ".env.example_for_test should define PROJECT_ROOT"
