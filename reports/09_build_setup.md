# Phase 9 — Wire build_setup

**Date:** 2026-05-07  
**Status:** build_setup factory wired with 4-way step-function dispatch and conditional operator slots.

---

## 0. OBJECTIVE SUMMARY

Complete the step-function selection logic in `build_setup` to dispatch to the correct step operator based on configuration presence. Ensure all operator slots (gradient/laplacian parametric operators, wetting_fn, extra_state_plugins) are built or left None as appropriate for each case.

**Result:** The factory now correctly wires 4 distinct operator configurations:

1. **Plain multiphase** (no wetting)
2. **Multiphase + fixed wetting** (no hysteresis)
3. **Multiphase + hysteresis**
4. **Multiphase + hysteresis + chemical step**

---

## 1. STEP-FUNCTION SELECTION TABLE

| `sim_type` | `wetting_config` | `hysteresis_config` | `chemical_step_config` | Step Function Selected |
|---|---|---|---|---|
| `"multiphase"` | — | — | — | `multiphase` |
| `"multiphase"` | ✓ | — | — | `multiphase_wetting` |
| `"multiphase"` | ✓ | ✓ | — | `multiphase_hysteresis` |
| `"multiphase"` | ✓ | ✓ | ✓ | `multiphase_hysteresis_chemical_step` |
| Other (`"standard"`, etc.) | — | — | — | `{sim_type}` (direct pass-through) |

### Implementation

Located in `tud_lbm/pipeline/setup.py:200–214`:

```python
# ── Step operator selection: 4-way dispatch ──────────────────────
sim_type = config.sim_type
wc = config.wetting_config
hc = config.hysteresis_config
csc = config.chemical_step_config

if sim_type == "multiphase":
    if wc and hc and csc:
        step_fn = build_step_fn("multiphase_hysteresis_chemical_step")
    elif wc and hc:
        step_fn = build_step_fn("multiphase_hysteresis")
    elif wc:
        step_fn = build_step_fn("multiphase_wetting")
    else:
        step_fn = build_step_fn("multiphase")
else:
    step_fn = build_step_fn(sim_type)
```

**Design rationale:**
- **Nested conditions:** Test presence of wetting first (no wetting → use base case).  
  Then test hysteresis (wetting without hysteresis → fixed-wetting).  
  Then test chemical step (hysteresis without chemical step → plain hysteresis).
- **Order matters:** Later configs build upon earlier ones. Chemical step requires hysteresis, which requires wetting.
- **Short-circuit:** Any missing intermediate config causes fallback to earlier step function.

---

## 2. OPERATOR SLOT CONFIGURATION TABLE

| Case | `gradient_density_wetting` | `laplacian_density_wetting` | `extra_state_plugins` | `wetting_fn` |
|---|---|---|---|---|
| **1. Plain multiphase** | `None` | `None` | `()` | `None` |
| **2. + fixed wetting** | `None` | `None` | `(WettingExtraStatePlugin,)` | `None` |
| **3. + hysteresis** | Built | Built | `(WettingExtraStatePlugin,)` | `build_wetting_fn("hysteresis")` |
| **4. + chemical step** | Built | Built | `(WettingExtraStatePlugin,)` | `build_wetting_fn("hysteresis")` |

### Implementation Details

**Parametric operator slots (Lines 195–198):**

```python
gradient_standard, gradient_density, laplacian_density, gradient_density_wetting, laplacian_density_wetting = (
    build_diff_ops(config, mp_params, lattice)
)
```

`build_diff_ops` returns a 5-tuple:
- `gradient_standard`: Always built (used for chemical potential).
- `gradient_density`: Wetting-corrected when wetting is present, else plain.
- `laplacian_density`: Same as gradient_density.
- `gradient_density_wetting`: Built only when **both** `wetting_config` **and** `hysteresis_config` are present. Otherwise `None`.
- `laplacian_density_wetting`: Same condition as gradient_density_wetting.

**Wetting function (Lines 230–235):**

```python
wetting_fn = None
if wc and hc:
    wetting_fn = build_wetting_fn("hysteresis")
```

Only built when both wetting and hysteresis are configured (cases 3 and 4).

**Extra state plugins (Lines 237–242):**

```python
if wc:
    from tud_lbm.operators.wetting._extra_state import WettingExtraStatePlugin

    extra_state_plugins = (WettingExtraStatePlugin(),)
else:
    extra_state_plugins = ()
```

`WettingExtraStatePlugin` is active for **any** wetting case (cases 2, 3, 4). It initializes `wetting` extra state and calls `wetting_fn` during state updates.

---

## 3. WIRING IN SimulationSetup RETURN STATEMENT

All slots are passed into the constructor (Lines 257–279):

```python
return SimulationSetup(
    # ...
    gradient_standard=gradient_standard,
    gradient_density=gradient_density,
    laplacian_density=laplacian_density,
    gradient_density_wetting=gradient_density_wetting,
    laplacian_density_wetting=laplacian_density_wetting,
    step_fn=step_fn,
    wetting_fn=wetting_fn,
    extra_state_plugins=extra_state_plugins,
    # ...
)
```

All optional fields accept `None` and are declared as such in the `SimulationSetup` NamedTuple (lines 124–126):

```python
gradient_density_wetting: DifferentialOperator | None = None
laplacian_density_wetting: DifferentialOperator | None = None
# ...
```

---

## 4. RUNNER.PY CONFIRMATION

Verified: runner.py requires **zero changes**.

**Relevant lines:**

- **Line 166** (streaming I/O mode):
  ```python
  new_state = setup.step_fn(setup, state)
  ```

- **Line 186** (in-memory trajectory mode):
  ```python
  new_state = setup.step_fn(setup, state)
  ```

Both lines call `setup.step_fn` directly without any conditional logic, sim_type checks, or operator-name references. The runner is fully decoupled from operator selection — it only depends on the pre-built `step_fn` in the setup.

---

## 5. DESIGN PRINCIPLES

### Single Point of Configuration
All operator selection logic lives in `build_setup`. Once the setup is created, the runner and all downstream code are agnostic to which step function is active.

### Lazy Evaluation
Operators (especially expensive parametric closures) are only built when needed:
- `gradient_density_wetting` and `laplacian_density_wetting` are only built if hysteresis is active.
- `wetting_fn` is only resolved if hysteresis is active.
- `WettingExtraStatePlugin` is only instantiated if wetting is configured.

### Composability
Each case is a proper superset of the previous:
- **Case 1** → Plain multiphase step, no wetting.
- **Case 2** → Case 1 + wetting extra-state init, but fixed parameters.
- **Case 3** → Case 2 + hysteresis optimizer, parametric operators, trial steps.
- **Case 4** → Case 3 + per-side spatial CA targeting.

### Zero Consumer Duplication
Consumers (runner, plugins, step functions) **never** check `sim_type`, `wetting_config`, `hysteresis_config`, or `chemical_step_config`. They all read pre-built slots from `setup`. If a slot is `None`, it is simply not used.

---

## 6. IMPLEMENTATION CHECKLIST

- [x] Replace step-fn selection with 4-way dispatch
- [x] Verify build_diff_ops returns parametric operators conditionally
- [x] Build wetting_fn only when hysteresis is active
- [x] Build WettingExtraStatePlugin only when wetting is active
- [x] Pass all slots into SimulationSetup return statement
- [x] Confirm runner.py needs no changes
- [x] Verify SimulationSetup declares all slots as optional

---

## 7. TESTING STRATEGY

**Unit tests would verify:**

1. **Case 1 (plain multiphase):**
   - `setup.step_fn.__name__ == "step_multiphase"`
   - `setup.gradient_density_wetting is None`
   - `setup.laplacian_density_wetting is None`
   - `setup.extra_state_plugins == ()`
   - `setup.wetting_fn is None`

2. **Case 2 (+ wetting, no hysteresis):**
   - `setup.step_fn.__name__ == "step_multiphase_wetting"`
   - `setup.gradient_density_wetting is None`
   - `setup.laplacian_density_wetting is None`
   - `len(setup.extra_state_plugins) == 1`
   - `setup.wetting_fn is None`

3. **Case 3 (+ hysteresis):**
   - `setup.step_fn.__name__ == "step_multiphase_hysteresis"`
   - `setup.gradient_density_wetting is not None`
   - `setup.laplacian_density_wetting is not None`
   - `len(setup.extra_state_plugins) == 1`
   - `setup.wetting_fn is not None`

4. **Case 4 (+ chemical step):**
   - `setup.step_fn.__name__ == "step_multiphase_hysteresis_chemical_step"`
   - `setup.gradient_density_wetting is not None`
   - `setup.laplacian_density_wetting is not None`
   - `len(setup.extra_state_plugins) == 1`
   - `setup.wetting_fn is not None`

---

## 8. SUMMARY

Phase 9 completes the wiring layer. The `build_setup` factory now:

1. **Selects the correct step operator** based on 4 cases (multiphase + wetting + hysteresis + chemical step).
2. **Builds operator slots conditionally** so that only required closures and plugins are created.
3. **Passes all slots** into the immutable `SimulationSetup` NamedTuple.
4. **Decouples all consumers** (runner, step functions, plugins) from operator selection logic.

The result is a clean, single-point-of-dispatch factory that scales to future operator variants without requiring changes to runner.py, plugins, or step functions.

