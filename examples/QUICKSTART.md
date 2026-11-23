# TUD-LBM quick start

A short guide to the examples in this folder. The full guide is
[`docs/quickstart.rst`](../docs/quickstart.rst).

Run every command from the repository root.

## 1. From the command line

```bash
uv run tud-lbm run examples/config_simple.toml             # run it
uv run tud-lbm run examples/config_simple.toml --dry-run   # show the resolved config
uv run tud-lbm run examples/config_simple.toml --override tau=0.8 --override nt=5000
```

Each run writes a timestamped directory under `results_dir`:

```
<results_dir>/<date>/<time>_<simulation_name>/
├── config.toml               # the exact configuration used, with every default
├── physical_parameters.txt   # dimensionless numbers and derived scales
├── simulation.log
├── data/timestep_<t>.npz     # one snapshot per save_interval
└── plots/snapshots/          # one figure per snapshot, when plot_fields is set
```

## 2. From Python

```python
from src import SimulationConfig, build_setup, init_state, observe, run

config = SimulationConfig(grid_shape=(64, 64), tau=0.8, nt=1000)
setup = build_setup(config)
state = init_state(setup)
final_state, trajectory = run(setup, state, nt=config.nt)

rho = observe(setup, final_state).rho  # shape (64, 64, 1, 1, 1)
```

A bare `SimulationConfig` is a single-phase D2Q9 run with BGK collision,
periodic edges and a rest-equilibrium start. The `State` carries only the
populations `f`, the time `t` and the addon substates `ext`; `observe` gives
`rho`, `u`, `force`, `force_ext` and `pressure`.

`trajectory` stacks the state of every step in memory. For a long run, stream
snapshots to disk instead (`example_run_simple.py`).

### Loading a configuration

```python
from src.simulation_io.readers import DictAdapter, TomlAdapter

config = TomlAdapter().load("examples/config_simple.toml")

config = DictAdapter().load(
    {
        "grid_shape": (128, 128),
        "tau": 0.8,
        "nt": 5000,
        "forces": {"gravity_force": {"force_g": 1e-6, "inclination_angle_deg": 90.0}},
    }
)
```

`DictAdapter` also accepts the nested TOML layout
(`{"simulation_type": {...}, "gravity_force": {...}}`).

### Streaming snapshots to disk

```python
from src import build_setup, init_state, run
from src.config.adapter_toml import TomlAdapter
from src.simulation_io.plotting.figure_builder import FigureBuilder
from src.simulation_io.save import SimulationIO

config = TomlAdapter().load("examples/config_simple.toml")
setup = build_setup(config)
state = init_state(setup)

io = SimulationIO(
    base_dir=config.results_dir,
    config=config,
    simulation_name=config.simulation_name,
    output_format=config.output_format,  # "numpy" or "vtk"
)
final_state, _ = run(
    setup,
    state,
    nt=config.nt,
    save_interval=config.save_interval,
    io_handler=io,
    skip_interval=config.skip_interval,
    save_fields=tuple(config.save_fields) if config.save_fields else None,
)
FigureBuilder(config=config, run_dir=io.run_dir).build_all()
```

## 3. Choosing the physics

There is no simulation type. Each section switches on what it describes:

| Section | Adds |
|---|---|
| (none) | single-phase BGK/MRT collide–stream loop with boundary conditions |
| `[multiphase]` | chemical-potential multiphase simulation with the well-balanced LBM (WB-LBM) and a specified EOS. Set by `eos`, `kappa`, `rho_l`, `rho_v`, `interface_width`. |
| `[boundary_conditions]` | a boundary condition per edge; the default boundary condition is periodic. Set by e.g. `bottom = "bounce-back"`. |
| `[obstacle]` | a circular obstacle with a bounce-back boundary condition. Set by `center_x`, `center_y`, `radius`. |
| `[wetting]` + a `"wetting"` edge | wetting boundary condition with a fixed contact angle at that edge. Set by `phi_left`, `phi_right`, `d_rho_left`, `d_rho_right`. |
| `[hysteresis]` | wetting boundary condition with contact angle hysteresis on that edge. Set by `ca_advancing`, `ca_receding`. |
| `[chemical_step]` | an edge with two regions, each with its own wetting boundary condition, with or without contact angle hysteresis. Set by `ca_advancing_pre_step`, `ca_receding_pre_step`, `ca_advancing_post_step`, `ca_receding_post_step`; `chemical_step_location` separates the regions. |
| `[<name>_force]` | a body force (`gravity_force`, `gravity_masked_force`, `electric_force`; at most one gravity force). Set by e.g. `force_g`, `inclination_angle_deg`. |

Without `[hysteresis]`, the pre-step region of a chemical step follows the `[wetting]` values and the post-step region takes the clamp limit toward the post-step contact angles. Every wetting section needs `[multiphase]` and a `"wetting"` edge.

The terms used here are defined in the *Terminology* section of
[`docs/quickstart.rst`](../docs/quickstart.rst).

## 4. Parameter sweeps

A sweep is staged, not run. `tud-lbm sweep stage` writes one completed TOML per
combination of its `--set` axes:

```bash
uv run tud-lbm sweep stage examples/config_simple.toml --set 'tau=[0.6, 0.8, 1.0]'
```

Run the staged files with `tud-lbm run`. A list in a configuration value is
rejected; use `sweep stage` instead.

## 5. Files in this folder

| File | What it shows |
|---|---|
| `minimal_example.py` | a `SimulationConfig` built in Python, in-memory run |
| `from_toml_example.py` | a config loaded from TOML (`config_simple.toml` by default) |
| `example_run_simple.py` + `config_simple.toml` | single-phase run with snapshots and figures |
| `example_run_von_karman.py` + `config_von_karman.toml` | channel flow past a cylinder: velocity inlet, outlet, obstacle |
| `example_run_complex.py` + `config_complex.toml` | droplet on an inclined wetting edge crossing a chemical step (contact angle hysteresis) |
| `config_bubbles.toml` | double-well bubble in a periodic box |
| `config_cs_simple.toml` | Carnahan–Starling bubble in a periodic box |
| `config_MRT_simple.toml` | single-phase MRT collision |
| `benchmarks/*.toml` | inputs for `tud-lbm benchmark` (see `docs/notes/performance.md`) |

Run the TOML-only examples with `uv run tud-lbm run examples/<file>.toml`.
