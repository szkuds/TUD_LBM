# Refactor plan: feature-slot architecture

Structure-only refactor. Internal physics (stencils, contact-angle maths, hysteresis optimiser) is untouched.

## 1. Decisions

| # | Decision | Outcome |
|---|---|---|
| 1 | Static vs dynamic wetting | Wall parameters always live in State, even for static runs. One kernel path serves static wetting, hysteresis and trial steps. |
| 2 | Wetting physics | A modification of the differential operators' padded region, implemented as `ghost_fill`. |
| 3 | Coupling mechanism | Fixed, typed slots. Features register like operators (kind `feature`); `build_setup` merges every active feature's contribution into the slots. No resolver, no shared context object. |
| 4 | Step | One skeleton: `kernel` plus `step`. `sim_type` no longer selects step functions; the features present in the config decide. |
| 5 | Coupling cost | Plain operators and features: a new file with its registry decorator (auto-discovered). A feature or force with its own TOML section also needs one field on `SimulationConfig`, as forces do today (removing that cost is a follow-up, see open items). `build_setup` is edited only to add a new kind of slot. |
| 6 | State | Core fields plus `ext: dict[str, pytree]` keyed by feature name. Each feature owns its substate. |
| 7 | Reference pressure | Rule, documented but not implemented in this refactor: pressure comes from rho inside equilibrium and source, never from a force. The zhang2022 files (split off in 4c36d1d to `.claude/split/zhang2022_files`, planned `feat/zhang2022`, including `ViscosityParams`) are out of scope and get rebuilt on the new slots later. |
| 8 | Chemical step | Part of the wetting feature (hysteresis parameters vary with contact-line location). |
| 9 | Config | Structure unchanged. `SimulationConfig` parses and validates; all guarding stays there. Section fields (`wetting_config`, `hysteresis_config`, `chemical_step_config`, `*_force`) and the derived properties read by analysis (`wetting_defaults`, `chemical_step_wall`, `restored_wetting`) stay as they are; features read them from the config. |
| 10 | Scope | Structure only. Internal physics untouched. |
| 11 | Registry | Stays. Only the kinds the feature slots replace are removed: `update_timestep`, `wetting` (hysteresis variants), `extra_state`. Everything else (plotting/analysis discovery, dimensionless numbers, EOS capability checks, force config schemas, CLI listings) keeps using it. |
| 12 | `sim_type` | Kept, with a reduced meaning. All five values still load (saved configs, analysis, CLI and adapters read it). `single_phase` vs `multiphase*` selects the macroscopic operator (and the equilibrium/source pair where relevant). Wetting, hysteresis and chemical step are switched on only by their config sections. |

## 2. Target architecture

```
config.toml -> SimulationConfig (parse and validate) -> build_setup (recipe) -> Setup -> run (scan over step)
```

**Vocabulary**
- **Feature:** a module that adds behaviour touching several parts of the simulation (wetting, electric potential). Returns a `Contribution`, or `None` when its config section is absent.
- **Slot:** one of the four fixed places a feature can contribute to.
- **Plain operator:** a homogeneous, selectable component (BC, EOS, collision, force, initialiser). Needs no coupling.

**Contribution slots**

| Slot | Shape | Wetting example |
|---|---|---|
| `ghost_fill` | `(padded, ext) -> padded` | Rewrites the wall ghost nodes between padding and the stencil. |
| `forces` | tuple of force operators | Electric force. |
| `post_step` | `(setup, prev_state, new_state) -> new_state` | Contact-angle measurement, hysteresis optimiser, electric potential update. |
| `init_state` | `(setup) -> pytree` | Initial wetting parameters and angles. |

**Feature registration.** A feature registers with a decorator under the registry kind `feature` and is auto-loaded like any operator, so the CLI lists it with no extra work. `build_setup` calls `feature.build(config, lattice)` for every registered feature; it returns a `Contribution`, or `None` when the feature's config section is absent. Merge rules: `ghost_fill` functions are composed, `forces` are concatenated, `init_state` results are keyed into `ext` by feature name, and `post_step` hooks run in name order. Features must be independent of each other. If one needs another's output, that is a real coupling and needs its own slot (an edit to `build_setup`).

**Why `post_step` receives both states.** The electric potential is a second lattice Boltzmann equation (Zhang & Kwok 2005, Eqs. 15-18): `h(t)` is collided with `tau_h = 3*sigma + 0.5`, where sigma comes from the density at the same time level, then streamed to `h(t+1)`. The existing plugin does this with `prev_state.f` and `prev_state.h`. The kernel leaves `h` untouched, so `h(t)` would still be in `new_state`, but `f(t)` is gone; taking sigma from `new_state.f` would pair `h(t)` with `sigma(t+1)` and change the result. Wetting needs the other side: it measures contact angles on `new_state.rho` and reads `prev_state.wetting`. One signature serves both. Hooks read forces from `new_state.force_ext`; there is no `**context`.

No `pre_step` slot for now: hysteresis parameters live in `ext` and the kernel reads them. Add the slot only when a feature needs it.

**Differential.** One builder, `build_differential(lattice, pad_modes, ghost_fill=None)`, returning `op(field, ext)`. The plain stencil ignores `ext`. The package never imports `wetting`.

**Kernel and step**
- `kernel(setup, f, force_ext, ext) -> (f, rho, u, force_tot, pressure)`: macroscopic, equilibrium, collision + source, streaming, obstacle, BC, macroscopic. Passes `ext` through to the differential operators; knows no feature by name.
- `step(setup, state)`: compute forces, call kernel, write new State, run the merged `post_step` hooks (features in name order, see feature registration).
- The hysteresis hook calls `setup.kernel` with `ext | {"wetting": candidate}` for its trial steps. Hooks receive `setup` at call time, so there is no circular dependency.
- Hysteresis stays post_step to preserve current behaviour (optimiser runs on post-step populations and force, writes parameters for the next step).

**Registry stays.** It is the catalog of plain operators and is load-bearing: the CLI lists and selects from it (`cli/display.py`, `visualise.py`, `analysis.py`), `SimulationConfig` validates names and reads force metadata from it, and capability checks use kind membership (e.g. an EOS has a `pressure` or `surface_tension` entry or it does not). Decorator registration plus `auto_load_operators` is what makes a new BC, EOS or force a one-file addition. Only the entries that stop being selectable operators leave it (see section 3).

**build_setup documentation requirement.** A header comment states the pattern: plain operators and features are auto-discovered through the registry, a feature fills the four slots, features are merged in name order and must be independent, and `build_setup` is edited only to add a new kind of slot. Each step of the recipe (lattice, feature merge, differential, forces, operators, step) gets a one-line comment saying what it does and which slots it consumes. There is no per-feature line, because there is no feature list.

## 3. Deleted

- Differential: `build_diff_ops`, `build_wetting_gradient_fn`, `build_wetting_laplacian_fn`, `_gradient_wetting.py`, `_laplacian_wetting.py`.
- Step: `step_multiphase_wetting`, `step_multiphase_hysteresis`, `step_multiphase_hysteresis_chemical_step`, `with_default_wetting`, `_make_wetting_ops`.
- Setup: any equilibrium/source special-casing (`isinstance` checks, by-name lookups) and `partial` binding hacks still present on the branch; verify in step 0.
- State: typed `wetting` and `h` fields, extra-state plugin registry, `pipeline/state/_extra_state.py`.
- Registry kinds that stop being selectable operators: `update_timestep` (step variants), `extra_state` (plugins), and the `wetting` hysteresis-variant kind; `simulation_type` pending a check of its consumers. The registry itself, `_loader.py` and the rest of the kinds stay.
- The ~20 `if x is None: raise TypeError` guards across the step modules.

## 4. Execution order

Each step is completed fully, including removal of replaced code, before the next. No tests are run until step 6.

**Step 0: Preparation**
- Read files not yet seen: `_hysteresis.py`, `_applicator.py`, `simulation_config.py`, `runner.py`.
- Capture baseline outputs from the current code, one case per path: single-phase, multiphase, static wetting, hysteresis, chemical step, electric. This is the only run before the refactor.

**Step 1: State and step skeleton**
- Add `ext` to `State`; remove typed `wetting` / `h` fields.
- Write `kernel` and `step`.
- Delete all step variants and `_common.py`.

**Step 2: Features and ghost_fill**
- Define `Contribution`.
- Wetting becomes a feature: applicator -> `ghost_fill`; contact-angle measurement and hysteresis optimiser -> `post_step`; chemical step folded in; `init_state` builds initial parameters.
- Electric force and potential become a feature.
- Collapse `differential` into one builder.

**Step 3: Dropped.** The zhang2022 files are not on this branch. The rule "pressure comes from rho inside equilibrium/source, never from a force" goes into the docs (step 7). zhang2022 is rebuilt on the new slots later.

**Step 4: build_setup and registration**
- Recipe in fixed order: lattice, active features (loop over the registry kind `feature`, merge slots), differential, forces (config forces plus feature forces), collision/equilibrium/streaming/BC/obstacle, step.
- Header and per-step comments as specified above.
- Remove the obsolete registry kinds listed in section 3; the registry otherwise stays as is.

**Step 5: Config**
- `SimulationConfig` keeps parsing and validation (BCs, forces, multiphase, MRT coupling, obstacle, and so on).
- No change to section fields, derived properties, sweeps, overrides or adapters.
- `sim_type` stops selecting step functions; it keeps only the phase meaning (decision 12). The check "hysteresis config requires a hysteresis `sim_type`" goes away, since config sections switch the features on.

**Step 6: Tests and verification**
- Delete tests that only cover the old structure.
- Rewrite the rest for the new design and invariants (slot wiring, `ext` structure, kernel purity).
- Run everything, compare against step 0 baselines, fix only refactor-introduced issues.

**Step 7: Docs and cleanup**
- Update `docs/architecture.rst` and `docs/operators.rst`, including the rule that pressure is resolved from rho inside equilibrium/source and never supplied by a force.
- Check whether `build/lib` is tracked; remove if it is a stale duplicate.

## 5. Open items

- **Hook order:** extra-state plugins currently run alphabetically; the feature merge runs hooks in name order. Verify that wetting measurement and the optimiser (same feature, ordered inside it) keep their present order, and that the current features (wetting, electric) are independent.
- **Follow-up, out of scope (option B):** a generic `sections` mechanism so features and forces declare their TOML section at registration and `SimulationConfig` needs no new field. It touches sweeps (`array_expansion`), `cli/overrides.py`, the adapters, the saved-config round trip and the analysis readers, so it is a separate project.
- **Tooling:** a shell on the user's machine is needed for baselines (step 0) and tests (step 6).
