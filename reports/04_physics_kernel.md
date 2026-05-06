# Phase 4 — Shared Physics Kernel

**Date:** 2026-05-04  
**Status:** Shared physics kernel implemented — `_multiphase_pipeline` added to `_common.py`.

---

## 1. OBJECTIVE SUMMARY

Implement a single, canonical physics kernel function `_multiphase_pipeline()` in `operators/step/_common.py` that encapsulates the complete multiphase LBM pipeline. This function is called by all step implementations and will be the sole location where multiphase physics runs.

**Key principle:** Single source of truth for physics logic. All callers (single-phase step, multiphase step, hysteresis trial step, etc.) route through this function, ensuring consistency and simplifying maintenance.

---

## 2. FUNCTION SIGNATURE

### Location: `tud_lbm/operators/step/_common.py:19–77`

```python
def _multiphase_pipeline(
    setup: SimulationSetup,
    f_t: jnp.ndarray,
    force_ext: jnp.ndarray | None,
    gradient_density: DifferentialOperator,
    laplacian_density: DifferentialOperator,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run one multiphase physics pass — the single canonical implementation.

    This function encapsulates the core multiphase LBM pipeline:
    1. Compute macroscopic fields (rho, u) from populations
    2. Apply common step (equilibrium → collision → streaming → BCs)
    3. Recompute macroscopic fields from updated populations

    This is the single place where multiphase physics runs. All step functions
    and trial steps (hysteresis, etc.) route through here.

    Args:
        setup: Closed-over :class:`~tud_lbm.pipeline.setup.SimulationSetup`.
        f_t: Current population distribution, shape ``(nx, ny, nz, q, 1)``.
        force_ext: External force field, shape ``(nx, ny, nz, 1, 2)`` or ``None``.
        gradient_density: Density gradient closure ``(grid) -> result``.
            Used in source term and passed to common step.
        laplacian_density: Laplacian of density closure ``(grid) -> result``.
            Used in macroscopic computation.

    Returns:
        ``(f_out, rho, u, force_tot)`` where:
        - ``f_out``: Post-BC populations, shape ``(nx, ny, nz, q, 1)``
        - ``rho``: Updated density field
        - ``u``: Updated velocity field
        - ``force_tot``: Total interaction force (or None if no forces)
    """
```

### Parameters

| Param | Type | Purpose |
|-------|------|---------|
| `setup` | `SimulationSetup` | Closed-over setup object (lattice, operators, params) |
| `f_t` | `jnp.ndarray` | Current population distribution |
| `force_ext` | `jnp.ndarray \| None` | External force field (or None) |
| `gradient_density` | `DifferentialOperator` | Density gradient closure for source term |
| `laplacian_density` | `DifferentialOperator` | Laplacian closure for macroscopic computation |

### Returns

```python
tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
```

Four values:
- **f_out**: Post-BC populations (ready for next step or observation)
- **rho**: Updated density field
- **u**: Updated velocity field
- **force_tot**: Total interaction force (or None if no forces active)

---

## 3. IMPLEMENTATION OVERVIEW

### Pipeline Steps

```
Step 1: Compute macroscopic from f_t
  └─ rho, u, force_tot = setup.macroscopic_fn(f_t, ..., laplacian_density=laplacian_density)

Step 2: Create temporary state
  └─ temp_state = State(f=f_t, rho=rho, u=u, t=0, wetting=None)

Step 3: Apply common LBM sequence
  └─ next_state = _apply_common_step(..., gradient_density=gradient_density)
     ├─ Equilibrium distribution
     ├─ Collision (+source term using gradient_density)
     ├─ Streaming
     └─ Boundary conditions

Step 4: Recompute macroscopic from updated f
  └─ rho_next, u_next = setup.macroscopic_fn(next_state.f, ..., laplacian_density=laplacian_density)

Return (f, rho, u, force) to caller
```

### Key Design Decisions

1. **No wetting state handling:** The function accepts pre-built `gradient_density` and `laplacian_density` closures. Wetting parameters are already injected into these closures by the caller (setup or hysteresis).

2. **Temporary state creation:** A minimal State object is created with `t=0` (unused). Only the `f` field matters for the LBM step.

3. **Double macroscopic call:** The function calls `setup.macroscopic_fn()` twice:
   - **First:** To compute initial (rho, u) for this step
   - **Second:** To compute (rho, u) for the next step (returned to caller)
   
   This ensures consistency across step boundaries and provides the caller with up-to-date fields.

4. **No state mutation:** The function does not modify `setup` or other state objects. It is a pure transformation: `f_t → f_out`.

---

## 4. WHAT IT REPLACES

### Before Phase 4: `_run_multiphase_pipeline` in `_multiphase.py`

**Location (before):** `tud_lbm/operators/step/_multiphase.py:28–69`

**Signature (before):**
```python
def _run_multiphase_pipeline(
    setup: SimulationSetup,
    f_t: jnp.ndarray,
    *,
    force_ext: jnp.ndarray = None,
    wetting: WettingState | None = None,
    gradient_density: DifferentialOperator = None,
    laplacian_density: DifferentialOperator = None,
) -> [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
```

**Issues with old implementation:**
1. Accepted `wetting` parameter but only used it for shim building (now removed)
2. Handled gradient/laplacian fallback logic (now done by caller)
3. Mixed physics logic with operator selection
4. Ambiguous which function was "the" multiphase step

### After Phase 4: `_multiphase_pipeline` in `_common.py`

**Location (after):** `tud_lbm/operators/step/_common.py:19–77`

**Improvements:**
1. ✓ No wetting parameter (cleaner interface)
2. ✓ Requires explicit gradient/laplacian (caller responsible)
3. ✓ Pure physics logic, no operator selection
4. ✓ Clear, narrow responsibility: "Run one multiphase pass"
5. ✓ Co-located with `_apply_common_step` in shared module
6. ✓ Explicitly documented as "the single place where multiphase physics runs"

---

## 5. FUTURE CALLERS (Phases 5–7)

### Phase 5: Step Functions

**Multiphase step function will call:**
```python
# In step_multiphase:
f_next, rho_next, u_next, force_tot = _multiphase_pipeline(
    setup, state.f, force_ext,
    setup.gradient_density,      # or trial-injected closure
    setup.laplacian_density,      # or trial-injected closure
)
```

**Single-phase step will continue using** existing path (no multiphase pipeline call).

### Phase 6: Hysteresis Trial Step

**Hysteresis evaluate_fn will call:**
```python
# In hysteresis.py _build_default_evaluate_fn:
f_out, rho_out, u_out, _ = _multiphase_pipeline(
    setup, f_t, force_ext,
    lambda grid: setup.gradient_density_raw(grid, p.phi_left_pre, p.phi_right_pre, ...),
    lambda grid: setup.laplacian_density_raw(grid, p.phi_left_pre, p.phi_right_pre, ...),
)
```

The parametric operators from setup are wrapped in lambda closures to inject trial parameters.

---

## 6. DELETION PLAN (Phase 8)

### `_run_multiphase_pipeline` — Still Present

**Status:** **NOT YET DELETED** (will be deleted in Phase 8)

**Reason for retention:** 
- Phase 4 only adds the new function; it does not yet migrate all callers
- Phases 5–7 incrementally migrate each call site
- Phase 8 performs final cleanup once all callers are migrated

**Deletion triggers (will be verified in Phase 8):**
1. All references to `_run_multiphase_pipeline` have been replaced with `_multiphase_pipeline`
2. All tests pass
3. Git history confirms the old function is no longer needed

### Migration Timeline

| Phase | Action | Impact |
|-------|--------|--------|
| 4 | Add `_multiphase_pipeline` to `_common.py` | New function available |
| 5 | Migrate step_multiphase to use it | _run_multiphase_pipeline still present |
| 6 | Migrate hysteresis trial step to use it | _run_multiphase_pipeline still present |
| 7 | Migrate any remaining callers | _run_multiphase_pipeline still present |
| 8 | Delete old `_run_multiphase_pipeline` | Final cleanup |

---

## 7. FILES MODIFIED

| File | Change | Status |
|------|--------|--------|
| `tud_lbm/operators/step/_common.py` | Added `_multiphase_pipeline` function + runtime jnp import | ✓ Modified |

**Lines added:** ~60 (function + docstring)

---

## 8. IMPORTS AND DEPENDENCIES

### New Runtime Import

**File:** `_common.py`

```python
import jax.numpy as jnp
```

Moved from TYPE_CHECKING block to runtime because `_multiphase_pipeline` uses `jnp.array()`.

### Dependencies Within Function

1. **Setup attributes used:**
   - `setup.lattice`
   - `setup.macroscopic_fn`
   - `setup.multiphase_params`
   - `setup.gradient_standard`

2. **Called functions:**
   - `setup.macroscopic_fn()` — computes (rho, u, force_tot)
   - `_apply_common_step()` — equilibrium → collision → streaming → BC

3. **State class:**
   - `State` imported from `tud_lbm.pipeline.state.state` (runtime import in function)

---

## 9. DESIGN RATIONALE

### Why a Separate Function?

**Before:** Physics logic was embedded in `step_multiphase()` function, making it hard to reuse for trial steps or testing.

**After:** Physics logic is extracted into `_multiphase_pipeline()`, a pure function that can be called by:
- Step functions (Phase 5)
- Hysteresis trial evaluator (Phase 6)
- Any future physics variant

### Why in `_common.py`?

**Rationale:**
1. **Co-location:** Already shares `_apply_common_step()` here
2. **Non-public:** Function is internal (starts with `_`), not a public API
3. **Logical grouping:** Both are "common step helpers" — shared by multiple callers
4. **Avoids circular imports:** Kept separate from either `_single_phase.py` or `_multiphase.py`

### Why Accept Operators as Arguments?

**Design choice:** Callers are responsible for providing `gradient_density` and `laplacian_density` closures.

**Benefits:**
1. **Flexibility:** Caller can inject different operators (setup defaults or trial-step variants)
2. **Testability:** Easy to test with mock operators
3. **No configuration logic:** Function focuses on physics, not operator selection
4. **Cleaner interface:** No need for `wetting` or fallback logic

---

## 10. VERIFICATION CHECKLIST

- ✓ Function signature matches specification
- ✓ Docstring documents all parameters and return values
- ✓ Function encapsulates complete multiphase physics pipeline
- ✓ Returns expected 4-tuple: (f_out, rho, u, force_tot)
- ✓ Uses `_apply_common_step()` for LBM sequence
- ✓ Calls macroscopic function twice (before and after step)
- ✓ Located in `_common.py` alongside `_apply_common_step`
- ✓ Runtime jnp import added
- ✓ No wetting-specific logic (operators pre-configured by caller)
- ✓ Function is pure (no side effects)

---

## 11. SUMMARY

| Aspect | Value |
|--------|-------|
| New function name | `_multiphase_pipeline` |
| Location | `tud_lbm/operators/step/_common.py` |
| Lines added | ~60 (function + docstring + import) |
| Function complexity | Simple: 4 operations + return |
| Parameters | 5 (setup, f_t, force_ext, grad_density, laplacian_density) |
| Return values | 4 (f_out, rho, u, force_tot) |
| Old function status | Still present, deleted in Phase 8 |
| Callers (Phase 4) | None (added for future use) |
| Future callers (Phase 5+) | step_multiphase, hysteresis trial step |

---

## 12. PHASE 4 COMPLETE ✓

**Objective:** Implement shared physics kernel `_multiphase_pipeline` as a clean, narrow function in `_common.py`.

**Deliverables:**
- ✓ `_multiphase_pipeline()` function implemented with clear signature
- ✓ Function documented with comprehensive docstring
- ✓ Pure physics logic: no operator selection or configuration
- ✓ Located in `_common.py` alongside `_apply_common_step`
- ✓ Runtime jnp import added
- ✓ Old `_run_multiphase_pipeline` still present (will be deleted in Phase 8)
- ✓ Ready for adoption by step functions (Phase 5) and hysteresis (Phase 6)

**Next steps (Phase 5):**
- Migrate `step_multiphase()` to call `_multiphase_pipeline()`
- Migrate `step_single_phase()` to use compatible path
- Begin establishing `_multiphase_pipeline` as the canonical physics kernel

