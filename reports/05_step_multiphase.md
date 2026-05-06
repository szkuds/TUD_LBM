# Phase 5 — step_multiphase (Plain)

**Date:** 2026-05-04  
**Status:** Minimal multiphase step function implemented and registered.

---

## 1. OBJECTIVE SUMMARY

Refactor and simplify `step_multiphase` to use the canonical `_multiphase_pipeline` kernel from Phase 4. Remove all wetting-specific logic, helper functions, and legacy code. Implement the minimal case that works for plain multiphase simulations (no wetting, no hysteresis, no conditional operator selection).

**Result:** A clean, maintainable time-stepping function that delegates physics to the kernel and focuses on state management and plugin coordination.

---

## 2. FUNCTION SIGNATURE

### Location: `tud_lbm/operators/step/_multiphase.py:18–60`

```python
@update_timestep_operator(name="multiphase")
def step_multiphase(setup: SimulationSetup, state: State) -> State:
    """Multiphase LBM step — the canonical multiphase time-stepping operator.

    This is the main time-stepping function for multiphase simulations.
    It runs the complete multiphase LBM pipeline once and updates the state.

    The implementation is minimal:
    1. Compute external forces
    2. Run the physics kernel (_multiphase_pipeline)
    3. Update extra state (wetting, etc.) if active
    4. Return updated state

    Args:
        setup: Closed-over :class:`~tud_lbm.pipeline.setup.SimulationSetup`.
        state: Current :class:`~tud_lbm.pipeline.state.state.State`.

    Returns:
        Updated :class:`~tud_lbm.pipeline.state.state.State` after one time step.
    """
```

### Parameters

| Param | Type | Purpose |
|-------|------|---------|
| `setup` | `SimulationSetup` | Closed-over setup object with all operators and params |
| `state` | `State` | Current simulation state (f, rho, u, t, wetting, etc.) |

### Returns

| Type | Purpose |
|------|---------|
| `State` | Updated state after one complete time step |

---

## 3. IMPLEMENTATION

### Code Flow

```python
# Step 1: Compute external forces
force_ext, state = compute_total_force_ext(setup, state, setup.forces)

# Step 2: Run multiphase physics kernel (Phase 4)
f_out, rho, u, force_tot = _multiphase_pipeline(
    setup,
    state.f,
    force_ext,
    setup.gradient_density,      # Pre-configured at setup time
    setup.laplacian_density,      # Pre-configured at setup time
)

# Step 3: Create new state
new_state = state._replace(
    f=f_out,
    rho=rho,
    u=u,
    force=force_tot,
    t=state.t + 1,
)

# Step 4: Update extra state (wetting, electric potential, etc.)
new_state = update_extra_state(setup, state, new_state, force_ext=force_ext)

return new_state
```

### Key Design Decisions

1. **No wetting logic:** The function assumes `setup.gradient_density` and `setup.laplacian_density` are already configured with appropriate wetting parameters (or defaults). No shim building.

2. **No conditional branching:** All operator selection happens at setup time, not at step time. This simplifies the function and improves clarity.

3. **Force integration:** Calls `compute_total_force_ext()` to gather forces from all sources (body forces, interaction forces, etc.).

4. **Extra state plugins:** Delegates wetting updates, electric potential computation, etc. to the plugin system via `update_extra_state()`.

5. **Minimal state construction:** Only updates the fields actually changed by the step: `f`, `rho`, `u`, `force`, and `t`.

---

## 4. CONFIGURATION TRIGGER

### When This Function Is Used

**Config condition:**
```python
sim_type = "multiphase"
```

**Setup process:**
1. `build_setup(config)` is called with `sim_type="multiphase"`
2. Multiphase parameters are built
3. `build_step_fn("multiphase")` looks up `step_multiphase` from registry
4. Returns `step_multiphase` as the step function

**At runtime:**
```python
step_fn = build_step_fn("multiphase")  # Returns step_multiphase
new_state = step_fn(setup, state)      # Calls step_multiphase(setup, state)
```

### Registry Entry

**Decorator:**
```python
@update_timestep_operator(name="multiphase")
```

**Registry key:** `"update_timestep:multiphase"`

**Lookup path:**
```
build_step_fn("multiphase")
  → build_operator("update_timestep", "multiphase")
  → OPERATOR_REGISTRY["update_timestep:multiphase"]
  → returns step_multiphase
```

---

## 5. WHAT IT REPLACES

### Before Phase 5

**Old function (131 lines):**
```python
@update_timestep_operator(name="multiphase")
def step_multiphase(setup, state):
    # 1. External forces
    force_ext, state = compute_total_force_ext(setup, state, setup.forces)
    
    # 2. Old _run_multiphase_pipeline (with wetting/shim logic)
    f_next, rho_next, u_next, force_tot = _run_multiphase_pipeline(
        setup,
        state.f,
        force_ext=force_ext,
        wetting=state.wetting,  # ← Wetting logic inside
    )
    
    # 3. State update
    _new_state = state._replace(...)
    
    # 4. Extra state
    new_state = update_extra_state(setup, state, _new_state, force_ext=force_ext)
    
    return new_state._replace(force=force_tot)
```

**Problems:**
- ~130 lines total in module (including helper functions)
- Wetting shim building inside `_run_multiphase_pipeline`
- Conditional operator selection logic
- Multiple state._replace calls
- Confusing with separate `multiphase_step` trial function

### After Phase 5

**New function (44 lines with docstring):**
```python
@update_timestep_operator(name="multiphase")
def step_multiphase(setup, state):
    # Minimal, focused implementation
    force_ext, state = compute_total_force_ext(setup, state, setup.forces)
    f_out, rho, u, force_tot = _multiphase_pipeline(
        setup, state.f, force_ext,
        setup.gradient_density,
        setup.laplacian_density,
    )
    new_state = state._replace(f=f_out, rho=rho, u=u, force=force_tot, t=state.t + 1)
    new_state = update_extra_state(setup, state, new_state, force_ext=force_ext)
    return new_state
```

**Improvements:**
- ✓ Only 44 lines (entire module is now 64 lines)
- ✓ No wetting shim building (done at setup time via phases 3–4)
- ✓ No conditional branching (single clear path)
- ✓ Single state._replace call (cleaner)
- ✓ Clear delegation to physics kernel
- ✓ Minimal, focused responsibility

---

## 6. FUNCTION CALLS

### External Dependencies

| Function | Module | Purpose |
|----------|--------|---------|
| `compute_total_force_ext()` | `tud_lbm.operators.force` | Gather forces from all sources |
| `_multiphase_pipeline()` | `tud_lbm.operators.step._common` | **NEW (Phase 4)** — canonical physics kernel |
| `update_extra_state()` | `tud_lbm.pipeline.state` | Apply plugin-based state updates |

### Setup Attributes Used

| Attribute | Type | Purpose |
|-----------|------|---------|
| `setup.forces` | `ForceSetup \| None` | Force specification |
| `setup.gradient_density` | `DifferentialOperator` | Density gradient (pre-configured) |
| `setup.laplacian_density` | `DifferentialOperator` | Density Laplacian (pre-configured) |

### State Attributes

| Attribute | Used as | Purpose |
|-----------|---------|---------|
| `state.f` | Input to physics kernel | Population distribution |
| `state.t` | Incremented | Current timestep |
| Other fields | Preserved | rho, u, force, force_ext, h, wetting |

---

## 7. IMPORTS CLEANUP

### Removed Imports

| Import | Reason |
|--------|--------|
| `import jax.numpy as jnp` | Not needed (physics kernel handles it) |
| `from tud_lbm.operators.step._common import _apply_common_step` | Moved to _multiphase_pipeline |
| `from tud_lbm.operators.step._wetting_differential_operators import ...` | **Module deleted** in Phase 2 |
| `from tud_lbm.pipeline.state.state import WettingState` | Wetting handled via extra_state plugins |
| `Type hints: DifferentialOperator, ...` | No longer needed in function body |

### New Imports

| Import | Source |
|--------|--------|
| `from tud_lbm.operators.step._common import _multiphase_pipeline` | Phase 4 kernel |

### Imports Retained

| Import | Purpose |
|--------|---------|
| `from tud_lbm.operators.force import compute_total_force_ext` | Force computation |
| `from tud_lbm.pipeline.state import update_extra_state` | Plugin coordination |
| `from tud_lbm.pipeline.state.state import State` | State type |
| `from tud_lbm.registry import update_timestep_operator` | Registration decorator |

---

## 8. COMPARISON: BEFORE → AFTER

| Aspect | Before | After | Change |
|--------|--------|-------|--------|
| Lines in _multiphase.py | 131 | 64 | **-51% (67 lines deleted)** |
| Lines in step_multiphase | ~30 | 18 | **-40%** |
| Helper functions | 2 | 0 | Removed `_run_multiphase_pipeline`, `multiphase_step` |
| Wetting logic in function | Yes | No | Delegated to setup time |
| Conditional branches | Yes | No | Single clear path |
| state._replace calls | 2 | 1 | Cleaner state updates |
| Imports (local) | 8 | 3 | **-62%** |
| Explicit loop depth | Mixed | Clear | 4-step linear flow |

---

## 9. INTEGRATION WITH FRAMEWORK

### Step Function Protocol

Implements `StepOperator` protocol:
```python
def __call__(setup: SimulationSetup, state: State) -> State: ...
```

### Registry Integration

```
@update_timestep_operator(name="multiphase")
    ↓
Decorator registers in OPERATOR_REGISTRY["update_timestep:multiphase"]
    ↓
build_step_fn("multiphase") looks up and returns function
    ↓
Runner calls step_fn(setup, state) repeatedly in lax.scan
```

### Runner Loop

```python
# In runner.py
def scan_body(state, _t):
    new_state = setup.step_fn(setup, state)  # ← Calls step_multiphase
    return new_state, new_state
    
final_state, trajectory = jax.lax.scan(scan_body, initial_state, jnp.arange(nt))
```

---

## 10. OPERATOR SELECTION AT SETUP TIME

**Key insight:** All operator selection (gradient_density, laplacian_density, gradient_density_raw, etc.) happens in `build_diff_ops()` at setup time, not in the step function.

**Setup time (build_setup → build_diff_ops):**
- Decides: plain, fixed wetting, or hysteresis
- Populates: setup.gradient_density, setup.laplacian_density, setup.gradient_density_raw (if hysteresis)
- Result: Pre-configured operators ready to use

**Step time (step_multiphase):**
- Uses: setup.gradient_density, setup.laplacian_density
- No decisions needed
- Physics kernel receives ready-made operators

**This separation:**
- ✓ Makes step_multiphase simple and fast
- ✓ Avoids branching in tight loop
- ✓ Enables trial steps (hysteresis) to inject different operators

---

## 11. FILES MODIFIED

| File | Lines | Change | Status |
|------|-------|--------|--------|
| `tud_lbm/operators/step/_multiphase.py` | 131 → 64 | Rewrote entire module (-51%) | ✓ Modified |

**Breakdown:**
- Removed `_run_multiphase_pipeline()` (~45 lines)
- Removed `multiphase_step()` helper (~8 lines)
- Removed wetting imports (~4 lines)
- Simplified `step_multiphase()` (~12 lines shorter)

---

## 12. VERIFICATION CHECKLIST

- ✓ Function signature matches specification
- ✓ Registered with `@update_timestep_operator(name="multiphase")`
- ✓ Calls `_multiphase_pipeline()` from Phase 4
- ✓ No wetting-specific logic
- ✓ No conditional branches
- ✓ Updates extra state (plugins)
- ✓ Returns `State`
- ✓ Imports cleaned up
- ✓ Old helper functions removed
- ✓ Minimal and focused

---

## 13. SUMMARY

| Item | Value |
|------|-------|
| Function name | `step_multiphase` |
| Module | `tud_lbm/operators/step/_multiphase.py` |
| Registration | `@update_timestep_operator(name="multiphase")` |
| Config trigger | `sim_type="multiphase"` |
| Lines of code | 18 (+ 26 docstring/comments = 44 total) |
| Complexity | Very simple: 4 operations |
| Core call | `_multiphase_pipeline(setup, state.f, force_ext, grad, lap)` |
| Status | Plain/minimal (no wetting, no hysteresis) |
| Next phase | Phase 6 — wetting-aware variant |

---

## 14. PHASE 5 COMPLETE ✓

**Objective:** Implement and register minimal `step_multiphase` using the canonical physics kernel.

**Deliverables:**
- ✓ `step_multiphase` implemented with clear 4-step logic
- ✓ Registered as `update_timestep:multiphase` in operator registry
- ✓ Calls `_multiphase_pipeline()` for physics (Phase 4 kernel)
- ✓ No wetting logic (all delegated to setup time and plugins)
- ✓ No conditional branches (single clear path)
- ✓ Module reduced to 64 lines (from 131)
- ✓ Imports cleaned up (removed wetting-specific, added kernel)
- ✓ Ready for Phase 6 (hysteresis variant)

**Code quality improvements:**
- **51% reduction** in module size
- **Single responsibility:** Time-stepping with physics kernel + state management
- **Clear delegation:** Physics → kernel, plugins → extra_state
- **Maintainability:** Minimal, focused, easy to understand

**Next step (Phase 6):**
- Create `step_multiphase_wetting()` that builds trial operators from raw slots
- Route hysteresis parameter updates through extra_state plugins

