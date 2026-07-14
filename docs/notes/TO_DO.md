# TO DO

- **Debug flags don't reach sweep workers**: `--debug-stability` (like `--debug-wetting`) sets a module-level global in `config_overview`, which does not propagate to `ProcessPoolExecutor` sweep workers when using the `spawn` start method. Per-leg stability logging in parameter sweeps would need env-var pass-through later, for example by checking it in `configure_jax()`.

- **SonarCloud/SonarQube install documentation**

  ```bash
  npx -y skills add giuseppe-trisciuoglio/developer-kit \
    --skill sonarqube-mcp \
    --agent claude-code
  ```

- **Plotting config structure is scattered**: field/analysis selection lives in `SimulationConfig.plot_fields`/`animate_fields`, cross-run panels are a hardcoded list (`_COMPARISON_PLOT_CONFIGS` in `run_comparison.py`), and colors are a separate dict (`FigureStyle.colors` in `figure_config.py`) — each independently keyed by operator/column name with no shared source of truth or validation that they stay in sync. Worth consolidating into one declarative registry per plot (name, required columns, style, panel vs. comparison) so adding a plot doesn't mean touching four files.
