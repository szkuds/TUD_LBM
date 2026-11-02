# Plan: code-review fixes (review notes of 23 and 26 October 2026)

Follows the feature-slot refactor (`docs/notes/REFACTOR_PLAN.md`), which is currently uncommitted on `fix/bubble_simulation` (65 files). The project rules still apply: structure first, behaviour preserved unless a change is stated as intended, replaced code removed in the same step, tests adapted and run only once a phase is complete.

## 1. Decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Config ownership | Centralised in the `config` package, organised to mirror the operator packages: one module per section (`config/boundary.py`, `config/multiphase.py`, `config/forces.py`, `config/wetting.py`, ...) holding that section's schema, validation and derived objects. Operator packages hold no config knowledge. `SimulationConfig` holds the core fields and one field per section, and runs all validation when it is constructed. |
| 2 | Old saved configs | Converted on load by one isolated legacy adapter (`src/config/legacy.py`), called from one place. Removing it at the next version bump is deleting that file and its one call. |
| 3 | Defaults | Every default value appears once, in `src/config/defaults.py`. The section modules read it; no operator or addon package declares a default. |
| 4 | 3D | Build 3D support (phase 5, scope below). |
| 5 | Equilibrium / source term | Selected by name from the config, not hard-coded by phase in `build_setup`. Which source term belongs to which equilibrium is physics, so an equilibrium declares its paired source term at registration; the default equilibrium per phase is a value in `config/defaults.py`. |
| 6 | Aggregators | The obstacle becomes the first entry of the boundary chain (same `(f_stream, f_col) -> f` signature). Forces keep their own aggregator: they are summed from the state, a different operation. |
| 7 | Protocols | One file, `src/protocols.py` (moved from `operators/protocols.py`), holds every interface in the codebase: the operator protocols, the addon hook types (`GhostFill`, `PostStepHook`, `InitStateHook`, now in `features/contribution.py`), `DimensionlessNumberOperator` (`analysis/physical_parameters/_inputs.py`) and the optax shim `_OptaxLike` (`wetting/_hysteresis.py`). The unused `SimulationRepository` and `ConfigReader`, and the `PlotOperator` Protocol that duplicates the plotting ABC, are deleted. The four ABCs (`PlotOperator`, `AnalysisPlot`, `ConfigAdapter`, `OutputWriter`) move there as whole classes, shared implementation included; the file therefore holds some implementation, accepted. Constraint: `src/protocols.py` imports only the standard library, `jax` and `numpy` at runtime; everything else (matplotlib, `SimulationConfig`, `State`, ...) only under `TYPE_CHECKING` or inside methods, so importing the core never pulls in plotting and no import cycle forms through this file. |
| 8 | `sim_type` | Removed. A run is multiphase when it has a `[multiphase]` section. Old configs carrying `sim_type` are handled by the legacy adapter. |
| 9 | Naming (D1) | Feature becomes **addon** (`src/addons/`, `@addon_operator`, registry kind `addon`); `Contribution` becomes **`Hooks`**. |
| 10 | Differential registry (D2) | Not added. There is one stencil; a registry is added when a second one exists. |
| 11 | Legacy adapter writes (D4) | Yes: an old-layout `config.toml` is converted on load and the converted file is written back to its results folder. |
| 12 | General initialiser (D5) | `multiphase_bubbles` (`operators/initialise/_multiphase_bubbles.py`). |
| 13 | 3D scope (D6) | Single-phase and multiphase bulk runs in 3D. Wetting and electric refuse 3D with a clear error. |
| 14 | One-line builders (D3) | Kept. Each package keeps its named `build_x` entry point typed with its protocol, including `build_plot_operator` and `build_analysis_plot` in plotting; this is the same pattern as every operator package, so it stays consistent. |

## 2. Review notes mapped to the plan

| Note | Where |
|---|---|
| Differential registry | Not added (decision 10) |
| Plotting builders / protocol location | Builders kept (decision 14); all protocols in `src/protocols.py`, duplicates removed, phase 2 (decision 7) |
| `front`/`back` missing from BC application order | Phase 5 |
| `build_boundary_edges` in config; config mirroring operators; chemical step in wetting | Phase 3: `config/boundary.py`, `config/wetting.py` (with hysteresis and chemical step) |
| Variable input params (multiphase) | Phase 3: `config/multiphase.py` holds a per-EOS parameter table |
| `@property`s on `SimulationConfig` | Phase 3 |
| `wetting_defaults` and other constants in one overview | Phase 3 (decision 3) |
| Initialise legacy examples | Phase 4 |
| Two functions in `_multiphase.py` | Phase 2 |
| Central aggregator | Phase 2 (decision 6) |
| `GhostFill` typed with `Any` | Phase 1 |
| Naming of feature / Contribution | Phase 1 (decision 9) |
| `architecture.rst` extension points | Phase 7 |
| Fixed `sim_type` list | Phase 3 (decision 8) |
| Forces named in the config | Phase 3: one `forces` field holding every `[*_force]` section, schemas in `config/forces.py` |
| `lattice` passed to feature builders | Phase 1: removed, both builders ignore it |
| `forces` slot in `Contribution` | Phase 1: removed, no feature fills it |
| Eq/source split by phase limits new operators | Phase 2 (decision 5) |
| Static `rho_l`/`rho_v` in wetting | Phase 6 |
| `State` storing derived fields | Already done: `State` is `(f, t, ext)` and `observe()` derives the rest. No work. |

Found during the review, not in the notes: `config/__init__.py` `from_toml` has its lazy import inside a comment (a literal `\n`); fixed in phase 2.

## 3. Target config design (phase 3)

The config package is the only place that knows about config: section layout, schemas, defaults, validation and the objects derived from a section. It is organised to mirror the operator packages, so finding the config for an operator means opening the module with the same name.

```
src/config/
  simulation_config.py  SimulationConfig: core fields + one field per section; __post_init__ runs every section's validation
  defaults.py           every default value, once
  boundary.py           [boundary_conditions]: schema, validation, edges in application order, pad modes, periodic axes
  obstacle.py           [obstacle]: schema, validation, mask
  multiphase.py         [multiphase]: schema, per-EOS parameter table, MultiphaseParams, phase references
  forces.py             [*_force]: one schema per force, active forces
  wetting.py            [wetting], [hysteresis], [chemical_step]: schemas, cross-section rules, wall parameters, chemical-step wall, restored state
  legacy.py             old-layout upgrade (see below)
  adapter_*.py, array_expansion.py, run_config.py, config_overview.py   unchanged roles
```

**SimulationConfig** holds:
- Core fields: `simulation_name`, `lattice_type`, `grid_shape`, `tau`, time stepping, collision, initialisation, output.
- One field per section, named like its TOML section: `boundary_conditions`, `obstacle`, `multiphase`, `wetting`, `hysteresis`, `chemical_step`, and `forces` (a mapping holding every `[*_force]` section). The old `*_config` names and the per-force fields go away.
- No `@property`s. The derived objects are functions in the section modules, e.g. `config.boundary.edges(cfg)`, `config.wetting.chemical_step_wall(cfg)`, `config.multiphase.params(cfg)`. Setup, analysis (`wetting_overlay`, `setup_figure`) and the CLI call those.

**Schemas and validation.** Each section module defines its schema (required, optional, positive keys) and its cross-section rules (hysteresis needs wetting, the chemical-step edge must be the wetting wall, only one gravity force), reading defaults from `defaults.py`. The force schemas now on the `@force_model` registrations move to `config/forces.py`. The config still asks the registry which names exist (BCs, EOS, forces, initialisers) to validate a configured name; that is the only link from config to the operators.

**What moves where:**

| Today | Becomes |
|---|---|
| `config/boundary_edges.py`, the `boundary_edges`/`pad_modes`/`periodic_axes` properties | `config/boundary.py` |
| `config/obstacle_mask.py`, the `obstacle_mask` property | `config/obstacle.py` |
| `config/multiphase_params.py`, the multiphase fields, `phase_references`, the Carnahan–Starling special case in `_validate_multiphase` | `config/multiphase.py`; `a_eos`…`t_eos` only in the Carnahan–Starling row of the per-EOS table |
| `gravity_force`/`electric_force`/`gravity_masked_force` fields, force metadata on `@force_model`, `active_forces` | `forces` field; `config/forces.py` |
| `config/wetting_defaults.py`, `config/chemical_step.py`, `_HYSTERESIS_DEFAULTS`, `_CHEMICAL_STEP_DEFAULTS`, the wetting/hysteresis/chemical-step validation, `wetting_defaults`/`chemical_step_wall`/`restored_wetting` | `config/wetting.py`; values to `defaults.py` |
| `sim_type`, `is_multiphase`, `is_single_phase` | multiphase means a `[multiphase]` section is present (`config.multiphase.is_multiphase(cfg)`) |

**Sweeps, adapters, overrides.** Sections keep being nested-sweepable fields; with field names equal to section names, the `cli/overrides.py` alias map goes away. The adapter routes every `[*_force]` TOML section into `forces`, and writes it back the same way.

**Extension cost, stated plainly.** Adding a force or an addon with its own TOML section touches its operator/addon file plus the config package (its schema in the section module, its values in `defaults.py`). That is the price of keeping all config in one place.

**Legacy adapter:** `src/config/legacy.py`, `upgrade(raw) -> raw`. It detects the old layout (`sim_type`, `*_config` and per-force keys, the old multiphase keys) and rewrites it to the new one. It is called once, in the adapter's load path, and nothing else imports it. When it converts a file, the loader writes the converted `config.toml` back to the same path (decision 11), so each old results folder is converted once.

## 4. Execution phases

Each phase is completed fully before the next. Order rationale: renames first so files are touched once; config before 3D because the BC application order moves into `config/boundary.py`; the one intended physics change (phase 6) last, so phases 1–5 can be checked against unchanged baselines.

**Phase 0: Baseline**
- Commit the current refactor so this plan starts from a clean tree.
- Capture baselines, one case per path: single-phase, multiphase, static wetting, hysteresis, chemical step, electric, obstacle, and one sweep config.
- Keep one results folder saved with the old config layout, to test the legacy adapter.

**Phase 1: Feature slots**
- Rename feature to addon and `Contribution` to `Hooks` (decision 9) across code, registry kind, docs and the `build_setup` header comment.
- Remove the `forces` slot and the merge code for it in `build_setup`.
- Remove the `lattice` parameter from feature builders.
- Typed ghost fill: a feature's fill receives only its own substate (`GhostFill[S] = (padded, S) -> padded`); the merge layer passes `ext[name]`. Electric's substate becomes a NamedTuple (`ElectricState`) instead of `{"h": ...}`. This also enforces the independence rule for the fill.

**Phase 2: Operator cleanups**
- `macroscopic/_multiphase.py`: one function. The multiphase parameters and EOS are bound at build time, so both macroscopic operators share one signature and the phase branch in `step._moments` goes away.
- Equilibrium and source term selected by config name with registered pairing (decision 5); `build_setup` names no equilibrium or source.
- Obstacle joins the boundary chain (decision 6); `operators/obstacle` keeps only its geometry.
- Protocols (decision 7): move `operators/protocols.py` to `src/protocols.py`; move in the addon hook types, `DimensionlessNumberOperator`, `_OptaxLike` and the four ABCs (`plotting/base.py`, `config/adapter_base.py`'s `ConfigAdapter`, `output_data/base.py`'s `OutputWriter`); delete the emptied modules and the three unused protocols; update every import; check the import constraint by importing `src.pipeline` without matplotlib loaded.
- Fix the `from_toml` comment line.

**Phase 3: Centralised config** (section 3)
1. `config/defaults.py`: move every default into it (wetting scalars, hysteresis, chemical step, force defaults, BC defaults, default equilibrium per phase).
2. Section modules, one at a time, each with its schema, validation and derived functions: `boundary.py`, `obstacle.py`, `multiphase.py`, `forces.py`, `wetting.py`. Delete the old `config/*` modules each one replaces.
3. `SimulationConfig`: one field per section, `__post_init__` calls the section validations; remove `sim_type` and every `@property`.
4. Remove the schema metadata from `@force_model` registrations.
5. Sweeps, adapters, overrides on the new field names; the adapter's `[*_force]` routing.
6. Update readers in setup, addons, analysis and CLI to the section-module functions.
7. Legacy adapter, tested on the old-layout results folder from phase 0.
8. Check that two-level sweeps (`forces.gravity_force.force_g = [...]`) expand; `array_expansion` handles one nesting level today.

**Phase 4: Initialisers**
- For each legacy initialiser (`wetting`, `wetting_drop_top`, `wetting_chem_step`, `multiphase_bubble_top`), show that the general initialiser reproduces its field (array equality on a few parameter sets). The general one is `multiphase_bubbles` (decision 12).
- Remove those that are reproduced; migrate the example configs, tests and `cli/wetting_init.py` to the general initialiser.
- Any that cannot be reproduced stay, with the reason recorded here.
- Outcome: all four removed; none stay. `tests/operators/initialise/test_legacy_equivalence.py` pins array equality against each removed formula on three parameter sets. The three wall droplets need `interface_width` halved in `[initialisation]`, because their tanh was twice as steep. Other grids and densities differ by up to 2e-15, from two rounding causes: a radius fraction times `min(nx, ny)` cannot always hit the old radius, and `avg - amp` can land an ulp below `rho_v`, which `multiphase_bubbles` clamps. No run under `~/TUD_LBM_data` used any of the four, so the legacy adapter maps nothing.

**Phase 5: 3D support** (scope fixed by decision 13)
Today 2D-only: BC application order (4 edges), every BC, streaming periodicity (x, y), padding (4 faces), the gradient and Laplacian stencils (`to_2d`, `grad_core_2d`), the well-balanced source term, the electric force, all wetting code, and the wetting initialisers.
Scope:
- `front`/`back` in the application order and in every BC.
- z-periodicity in streaming.
- 6-face padding and 3D gradient/Laplacian stencils.
- Single-phase and multiphase bulk runs in 3D, including the well-balanced source.
- Wetting, electric and the 2D initialisers refuse 3D with a clear error instead of silently running.
Verification: 3D Poiseuille flow against the analytic profile; a 3D droplet against the Laplace law.

**Phase 6: Phase densities in wetting** (intended behaviour change)
Today `rho_l`/`rho_v` are the config values in three places: the applicator, the `rho_mean` contact-angle threshold, and the hysteresis clamps (`_hysteresis.py`). Options:
- (a) keep the config values;
- (b) the phase densities measured off the initial field (what `phase_references` already does), static for the run;
- (c) measured every step and carried in the wetting substate.
Run (b) and (c) against the phase 0 wetting and hysteresis baselines, show the differences, and decide with you before changing the default.

**Phase 7: Tests, docs, verification**
- Delete tests that only cover replaced structure; adapt the rest to the new design.
- Run everything; compare phases 1–5 against the phase 0 baselines exactly.
- `architecture.rst`: extension-points table gains rows for an addon, a force with its config section, an EOS with its parameters, and a config section, each naming the operator file and the config-package edits; describe the config package layout, `defaults.py` and the legacy adapter.
- `operators.rst`: equilibrium/source pairing, obstacle in the boundary chain.

## 5. Open decisions

None. All decisions are recorded in section 1 (decisions 9–14 settle D1–D6).
