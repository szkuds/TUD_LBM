# Plan: ty-Compliant Refactor

## Context

`ty` (Astral) type-checking is in active cleanup on `feature/carnahan-starling-eos`. Running `.venv/bin/ty check` currently reports **136 errors across 38 files**. The majority are None-union handling issues: `SimulationSetup` fields are typed `X | None` but used at call sites without guards, so ty emits `call-non-callable`, `unresolved-attribute`, and `invalid-argument-type` errors. A secondary issue is a **functional regression** in `streaming/__init__.py` where a refactor accidentally dropped the `bc_config` closure, silently treating all boundary axes as periodic.

> **Tooling note:** This project uses `select = ["ALL"]` in ruff — `S101` (use-of-assert) is **not** suppressed in source files, only in `tests/`. Bare `assert x is not None` will be flagged by ruff. Use `if x is None: raise TypeError(...)` instead — ty narrows after a `raise`, so this pattern satisfies both tools. Invoke `/astral:ruff` and `/astral:ty` before starting; run both after each file group.

---

## Error Distribution

| File | Count | Root cause |
|---|---|---|
| `operators/wetting/hysteresis/hysteresis.py` | 24 | `config.hysteresis_config \| None`, `mp \| None`, `trial_step_fn \| None` |
| `io/plotting/analysis.py` | 19 | config multiphase fields `\| None` |
| `operators/step/_multiphase_hysteresis.py` | 15 | `setup.gradient_density_wetting \| None`, wrong `WettingState` annotation |
| `operators/step/_common.py` | 12 | all `setup.*_fn \| None`, `setup.forces \| None` |
| `operators/macroscopic/eos/_carnahan_starling.py` | 8 | `mp.a_eos/b_eos/r_eos/t_eos \| None` → `float` |
| `config/simulation_config.py` | 8 | `array_field()` return type, `field()` overload mismatch |
| `pipeline/state/__init__.py` | 7 | `**context: dict[str, Any]` should be `**context: Any` |
| `io/physical_parameters.py` | 6 | config multiphase fields `\| None` |
| `pipeline/parallel_runner.py` | 5 | `SimulationConfig(**dict)` dynamic-unpack inference |
| `operators/step/_multiphase_wetting.py` | 5 | similar to `_common.py` |
| _(remaining ~28 files)_ | ~27 | mixed; same None-guard pattern |

**Error code breakdown:** `invalid-argument-type` (56), `unresolved-attribute` (27), `call-non-callable` (23), `not-subscriptable` (7), `unsupported-operator` (5), `invalid-return-type` (5), `invalid-assignment` (5), `unknown-argument` (2), `no-matching-overload` (2), `invalid-type-form` (2), `not-iterable` (1), `missing-argument` (1).

---

## Fix Strategy

### Pattern A — `if x is None: raise TypeError(...)` type narrowing (dominant fix)

**Do NOT use bare `assert x is not None`** — ruff `S101` (bandit) flags asserts in source files since the project enables `select = ["ALL"]`. Use the raise-based narrowing pattern instead; ty narrows the type after any unconditional `raise`:

```python
# Step functions (JIT context) — raise is also a compile-time check in JAX
if setup.macroscopic_fn is None:
    msg = "macroscopic_fn is required for multiphase pipeline"
    raise TypeError(msg)
# ty now knows setup.macroscopic_fn is not None below this point
rho, u, force_tot = setup.macroscopic_fn(f_t, lattice, ...)
```

For config dict fields, the same pattern applies:
```python
if setup.config.hysteresis_config is None:
    msg = "hysteresis_config must be set for hysteresis step"
    raise TypeError(msg)
hc = setup.config.hysteresis_config
```

> Ruff also enforces `EM101` (no string literal in raise directly) — use a `msg` variable as shown above.

### Pattern B — `# type: ignore` → `# ty: ignore` migration (4 lines)

Replace all old-style mypy suppressions with ty-style ones. Correct error codes may differ — run `ty check` to confirm the exact code before writing the suppression.

### Pattern C — Structural type fixes (targeted)

- `array_field()` in `simulation_config.py`: return type `field` is invalid (it's a function, not a class); fix the mutual-exclusivity of `default`/`default_factory` to avoid `no-matching-overload`.
- `pipeline/state/__init__.py`: `**context: dict[str, Any]` → `**context: Any`.
- `_multiphase_hysteresis.py`: `wetting: State` parameter annotation → `wetting: WettingState`.
- `hysteresis.py` `update_wetting_state`: remove `= None` default on `trial_step_fn` (always provided at every call site).

### Pattern D — Streaming closure regression fix (functional + type)

`build_streaming_fn` must restore the closure over `bc_config`. The current code just casts the raw operator without binding `bc_config`, so all calls use `bc_config=None` (all axes periodic), which is wrong for bounce-back walls. Restore:

```python
def build_streaming_fn(scheme: str = "standard", bc_config: dict | None = None) -> StreamingOperator:
    op = build_operator("stream", scheme)
    _bc = bc_config

    def _stream(f: jnp.ndarray, lattice: Lattice, bc_config: dict | None = None) -> jnp.ndarray:  # noqa: ARG001
        return cast("jnp.ndarray", op(f, lattice, _bc))

    return cast("StreamingOperator", _stream)
```

The `bc_config` parameter on `_stream` satisfies the `StreamingOperator` protocol signature; `# noqa: ARG001` suppresses the unused-argument ruff warning since `_bc` (captured from the outer scope) takes precedence.

---

## File-by-File Changes

### `tud_lbm/operators/streaming/__init__.py` ← has uncommitted changes, fix regression
Restore the `_bc` closure as shown in Pattern D.

### `tud_lbm/operators/step/_common.py`
Add raise-guards at the top of `_multiphase_pipeline`:
```python
if setup.macroscopic_fn is None:
    raise TypeError("macroscopic_fn required")
if setup.multiphase_params is None:
    raise TypeError("multiphase_params required")
if setup.gradient_standard is None:
    raise TypeError("gradient_standard required")
```
Add raise-guards at the top of `_apply_common_step`:
```python
if setup.equilibrium_fn is None:
    raise TypeError("equilibrium_fn required")
if setup.collision_fn is None:
    raise TypeError("collision_fn required")
if setup.streaming_fn is None:
    raise TypeError("streaming_fn required")
if setup.bc_fn is None:
    raise TypeError("bc_fn required")
```
For the forces branch: add `if setup.forces is None: raise TypeError(...)` inside the `if force_tot is not None:` block. Use `msg =` variables to satisfy `EM101`.

### `tud_lbm/operators/step/_multiphase_hysteresis.py`
- Change `_make_wetting_ops(setup, wetting: State)` → `wetting: WettingState`
- Add raise-guards at top of `_make_wetting_ops` for `setup.gradient_density_wetting` and `setup.laplacian_density_wetting`
- Same raise-guards at top of `_trial_step`
- In `_trial_step`: add raise-guard for `setup.config.hysteresis_config` before dict `.get()` call
- In `step_multiphase_hysteresis`: add raise-guard for `state.wetting` before `_make_wetting_ops(setup, state.wetting)`

### `tud_lbm/operators/wetting/hysteresis/hysteresis.py`
- `_update_wetting_state_impl`: add raise-guard for `setup.multiphase_params` before `mp = setup.multiphase_params`
- Same function: add raise-guard for `setup.config.hysteresis_config` before `hc = ...`
- `_get_hysteresis_window_chemical_step`: add raise-guard for `setup.config.chemical_step_config`
- `update_wetting_state`: remove `= None` default on `trial_step_fn` (always provided at call sites)

### `tud_lbm/operators/macroscopic/eos/_carnahan_starling.py`
Add raise-guards at the start of `build_carnahan_starling_eos`, then bind to locals to avoid repeated `| None` narrowing inside the lambda:
```python
if mp.a_eos is None or mp.b_eos is None or mp.r_eos is None or mp.t_eos is None:
    msg = "a_eos, b_eos, r_eos, t_eos are required for Carnahan-Starling EOS"
    raise ValueError(msg)
a, b, r, t = mp.a_eos, mp.b_eos, mp.r_eos, mp.t_eos
return lambda rho: _eos_carnahan_starling(rho, a, b, r, t)
```

### `tud_lbm/config/simulation_config.py`
1. `# type: ignore[arg-type]` (line 66) → `# ty: ignore[arg-type]` (confirm exact code with `ty check`)
2. `# type: ignore[operator]` (lines 78, 85) → `# ty: ignore[operator]`
3. Change `array_field()` return type from `-> field:` to `-> Any:` (add `Any` import if not present)
4. Fix `field()` mutual-exclusivity to avoid `no-matching-overload`:
   ```python
   if default is not dataclasses.MISSING:
       return field(default=default, metadata=metadata, **kwargs)
   if default_factory is not dataclasses.MISSING:
       return field(default_factory=default_factory, metadata=metadata, **kwargs)
   return field(metadata=metadata, **kwargs)
   ```
5. In `build_multiphase_params`: after the validation loop add raise-style narrowing (the loop already raises `ValueError` but ty doesn't see that as narrowing — add explicit `if config.eos is None: raise` etc.)

### `tud_lbm/operators/macroscopic/__init__.py`
- Line 77: remove `# type: ignore[type-var]` and check if ty still flags; replace with `# ty: ignore[<code>]` only if needed

### `tud_lbm/pipeline/state/__init__.py`
- `update_extra_state` signature: `**context: dict[str, Any]` → `**context: Any`

### `tud_lbm/pipeline/parallel_runner.py`
- Read full file and run `ty check` on it to confirm exact error codes; likely `SimulationConfig(**dict)` dynamic-unpack → add `# ty: ignore[missing-argument]` per site or restructure to explicit typed kwargs

### `tud_lbm/operators/step/_multiphase_wetting.py`
- Add raise-guards (same pattern as `_common.py`) for `setup.gradient_density_wetting`, `setup.laplacian_density_wetting`, `setup.wetting_fn`, and `state.wetting`

### `tud_lbm/io/plotting/analysis.py` + `tud_lbm/io/physical_parameters.py`
- Config multiphase fields (`rho_l`, `rho_v`, `kappa`, `interface_width`) are `| None`; add raise-guards (`if config.rho_l is None: raise ValueError(...)`) before arithmetic. These are IO utilities so runtime validation is correct behaviour, not just type-narrowing

---

## Execution Order

For each step: run ruff then ty after editing, fix any new ruff violations before moving on.

1. **Invoke `/astral:ruff` and `/astral:ty`** — confirm baseline counts before touching any file
2. Fix `streaming/__init__.py` (functional regression + type)
3. Fix `# type: ignore` → `# ty: ignore` in `simulation_config.py` and `macroscopic/__init__.py`
4. Fix `simulation_config.py` structural issues (`array_field` return type, `field()` overloads)
5. Fix `pipeline/state/__init__.py` (`**context` annotation)
6. Fix `operators/step/_common.py` (raise-guards)
7. Fix `operators/step/_multiphase_hysteresis.py` (raise-guards + WettingState annotation)
8. Fix `operators/step/_multiphase_wetting.py` (raise-guards)
9. Fix `operators/wetting/hysteresis/hysteresis.py` (raise-guards + trial_step_fn)
10. Fix `operators/macroscopic/eos/_carnahan_starling.py` (raise-guards for CS EOS params)
11. Fix `io/plotting/analysis.py` + `io/physical_parameters.py` (raise-guards for config fields)
12. Fix `pipeline/parallel_runner.py` + remaining files

---

## Verification

```bash
# Baseline (before any changes):
uv run ruff check tud_lbm/          # note current ruff violations
.venv/bin/ty check                  # note current ty error count (136)

# After each file group:
uv run ruff check --fix tud_lbm/    # auto-fix any ruff regressions
.venv/bin/ty check                  # watch error count drop

# Final:
uv run ruff check tud_lbm/          # target: 0 violations
uv run ruff format --check tud_lbm/ # confirm format unchanged
.venv/bin/ty check                  # target: 0 errors
pytest                              # fast unit tests — no regressions
pytest -m slow                      # full end-to-end including streaming BC test
```