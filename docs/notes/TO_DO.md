# TO DO

- **Debug flags don't reach sweep workers**: `--debug-stability` (like `--debug-wetting`) sets a module-level global in `config_overview`, which does not propagate to `ProcessPoolExecutor` sweep workers when using the `spawn` start method. Per-leg stability logging in parameter sweeps would need env-var pass-through later, for example by checking it in `configure_jax()`.

- **SonarCloud/SonarQube install documentation**

  ```bash
  npx -y skills add giuseppe-trisciuoglio/developer-kit \
    --skill sonarqube-mcp \
    --agent claude-code
  ```
