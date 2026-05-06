# Phase 8 — step_multiphase_hysteresis_chemical_step

**Date:** 2026-05-07  
**Status:** Chemical-step hysteresis step operator implemented; per-side CA targets selected dynamically.

---

## 0. WHAT DIFFERS FROM PHASE 7

**Only two substantive changes:**

1. **New wetting-update call:** `update_wetting_state_chemical_step` instead of `update_wetting_state`
   - Imported from `tud_lbm.operators.wetting.hysteresis`

2. **Two additional imports:**
   - `update_wetting_state_chemical_step` in the new step module
   - `from tud_lbm.operators.step import _multiphase_hysteresis_chemical_step  # noqa: F401` in `__init__.py`

**Structural similarity:** Apart from the wetting-update function name, `step_multiphase_hysteresis_chemical_step` is identical to `step_multiphase_hysteresis`. The operator selection (which step to use) is handled by the factory based on configuration presence.

---

## 1. OBJECTIVE SUMMARY

Implement a hysteresis-enabled step operator that uses **per-side, spatially-varying CA targets** determined by contact-line position relative to a chemical step boundary.

**Key features:**
- Left and right contact lines can independently be on either side of the chemical step
- Each side compares its current CLL position against `chemical_step_location`
- CA advancing/receding pairs are selected from one of four config keys per side, per timestep
- No modification to `WettingParams` or `WettingState`: the 4-field optimizer operates the same way
- Pre/post initialization seeds (`phi_pre_step`, etc.) are consumed only at init time (Phase 6)

---

## 2. WHY WettingParams WAS NOT EXTENDED

The design question: should `WettingParams` carry pre-step and post-step variants (8 fields) for chemical-step simulation?

**Answer: No. The 4-field `WettingParams` is sufficient.**

**Reasoning:**

1. **Initialization responsibility:** The pre-step seeds (`phi_pre_step`, `d_rho_pre_step`, `phi_post_step`, `d_rho_post_step`) are consumed once during `init_wetting_chemical_step` to set up the initial `WettingState`.

2. **Per-step optimizer:** The hysteresis optimizer operates on the same 4-field struct (`phi_left`, `phi_right`, `d_rho_left`, `d_rho_right`) regardless of step position. The optimizer doesn't know or care about the step; it only optimizes the current parameters toward the current (side-dependent) CA targets.

3. **Dynamic per-step selection:** At each timestep, `update_wetting_state_chemical_step` queries the chemical step config to get the CA window for each side **after** measuring the current CLL. This selection is **not** stored; it is computed on demand.

**Life cycle of a chemical-step parameter:**

```
t=0:
  - init_wetting_chemical_step( config.chemical_step_config )
    → selects seeds based on init_cll position
    → initializes WettingState with 4 parameters
  
t=1,2,...:
  - step_multiphase_hysteresis_chemical_step( state ) calls
    - update_wetting_state_chemical_step( state.wetting, ... )
      → measures ca_left, ca_right, cll_left, cll_right from rho
      → for each side: calls _get_hysteresis_window_chemical_step( setup, side_cll )
      → optimizer receives side-specific CA targets
      → returns updated 4-parameter WettingState
```

The pre-step and post-step seeds are **initialization data only**. They do not travel through the optimizer.

---

## 3. _get_hysteresis_window_chemical_step HELPER

### Location
`tud_lbm/operators/wetting/hysteresis/hysteresis.py:153–170`

### Signature & Behavior

```python
def _get_hysteresis_window_chemical_step(setup, cll):
    """Return (ca_advancing, ca_receding) based on CLL position relative to chemical step."""
    step_x = (
        setup.config.chemical_step_config["chemical_step_location"]
        * setup.config.grid_shape[0]
    )
    return jax.lax.cond(
        cll < step_x,
        lambda: (
            setup.config.chemical_step_config["ca_advancing_pre_step"],
            setup.config.chemical_step_config["ca_receding_pre_step"],
        ),
        lambda: (
            setup.config.chemical_step_config["ca_advancing_post_step"],
            setup.config.chemical_step_config["ca_receding_post_step"],
        ),
    )
```

### Design Notes

**Per-side usage:** Called **twice per optimization cycle**, once with `wetting.cll_left` and once with `wetting.cll_right`.

```python
ca_adv_left, ca_rec_left = _get_hysteresis_window_chemical_step(setup, wetting.cll_left)
ca_adv_right, ca_rec_right = _get_hysteresis_window_chemical_step(setup, wetting.cll_right)
```

**Why separate calls?** Each contact line independently determines its region:
- If `cll_left = 20` and `cll_right = 45` with `step_x = 32`:
  - Left side: `cll_left < step_x` → pre-step angles
  - Right side: `cll_right > step_x` → post-step angles
  - Optimized independently within their respective windows

**Comparison with CLL:** The comparison is against the **pre-step CLL** (`wetting.cll_left`, `wetting.cll_right`), not the freshly-measured post-step CLL. This preserves hysteresis memory: the region (pre/post) is "sticky" until the contact line crosses the step.

---

## 4. update_wetting_state_chemical_step FUNCTION

### Location
`tud_lbm/operators/wetting/hysteresis/hysteresis.py:318–447`

### Signature

```python
def update_wetting_state_chemical_step(
    wetting: WettingState,
    rho: jnp.ndarray,
    setup,
    f_bc: jnp.ndarray,
    force: jnp.ndarray,
    *,
    trial_step_fn,
) -> WettingState:
```

### Identity to update_wetting_state

The function body is **verbatim identical** to `update_wetting_state` except for two substitutions at the very start:

```python
# ❌ NOT IN CHEMICAL-STEP VERSION (uses hysteresis_config):
# ca_adv_left = ca_adv_right = hc["ca_advancing"]
# ca_rec_left = ca_rec_right = hc["ca_receding"]

# ✓ IN CHEMICAL-STEP VERSION (uses dynamic per-side lookup):
ca_adv_left, ca_rec_left = _get_hysteresis_window_chemical_step(setup, wetting.cll_left)
ca_adv_right, ca_rec_right = _get_hysteresis_window_chemical_step(setup, wetting.cll_right)
```

After these two lines, every subsequent line in `update_wetting_state_chemical_step` is identical to `update_wetting_state`. No refactoring; two functions remain independently readable and maintainable.

### Why No Refactoring?

Shared logic (the optimizer loop, CLL measurement, etc.) is **left duplicated** because:

1. **Phase 8 scope:** Minimal, focused change to enable chemical-step support
2. **Future flexibility:** If a future phase diverges one function from the other, duplication is less risky than extraction
3. **Readability:** Both functions are understandable end-to-end; no need to trace through a shared helper

---

## 5. step_multiphase_hysteresis_chemical_step OPERATOR

### Location
`tud_lbm/operators/step/_multiphase_hysteresis_chemical_step.py`

### Key Points

**Registration:**
```python
@update_timestep_operator(name="multiphase_hysteresis_chemical_step")
def step_multiphase_hysteresis_chemical_step(setup: SimulationSetup, state: State) -> State:
```

**Only difference from `step_multiphase_hysteresis`:**
- Line 55: `update_wetting_state_chemical_step(...)` instead of `update_wetting_state(...)`
- Imports: `update_wetting_state_chemical_step` added

**Shared helpers (imported):**
```python
from tud_lbm.operators.step._multiphase_hysteresis import _make_wetting_ops, _trial_step
```

These are **imported, not copied**, maintaining the single definition and avoiding duplication of the gradient/laplacian builder.

---

## 6. CONFIGURATION KEYS

### Consumed at Initialization (Phase 6) — Not in Step Loop

```toml
[chemical_step]
phi_pre_step = 1.0
d_rho_pre_step = 0.05
phi_post_step = 1.2
d_rho_post_step = 0.0
```

Used by `init_wetting_chemical_step` to seed wetting parameters based on initial CLL position. **Not accessed or modified during the step.**

### Consumed at Each Step (Phase 8) — Per-Side Dynamic Selection

```toml
[chemical_step]
chemical_step_location = 0.5          # fractional x-position (0.0–1.0)
chemical_step_edge = "bottom"         # edge where step is applied (unused in step loop)
ca_advancing_pre_step = 120.0
ca_receding_pre_step = 100.0
ca_advancing_post_step = 70.0
ca_receding_post_step = 50.0
```

**Usage:**
- `chemical_step_location`, `ca_advancing_pre_step`, `ca_receding_pre_step`: queried for each side in pre-step region
- `ca_advancing_post_step`, `ca_receding_post_step`: queried for each side in post-step region

These are looked up on **every call** to `_get_hysteresis_window_chemical_step`, computed from the current CLL position.

---

## 7. OPERATOR REGISTRY & SELECTION

### Factory Logic (setup.py)

The operator factory selects the correct step function based on configuration presence:

```python
if "wetting_config" in config and "chemical_step_config" in config:
    if "hysteresis_config" in config:
        return "multiphase_hysteresis_chemical_step"  # Phase 8
    else:
        return "multiphase_wetting_chemical_step"     # Phase 6 (future)
elif "wetting_config" in config:
    if "hysteresis_config" in config:
        return "multiphase_hysteresis"                # Phase 7
    else:
        return "multiphase_wetting"                   # Phase 5
else:
    return "multiphase"                               # Base
```

---

## 8. TESTING STRATEGY (Deferred; See Phase 8 Tests)

- `test_multiphase_hysteresis_chemical_step_registered`: Operator is in registry
- `test_multiphase_hysteresis_chemical_step_step_increments_t`: State.t increments
- `test_multiphase_hysteresis_chemical_step_wetting_updated`: Wetting state changed
- `test_multiphase_hysteresis_chemical_step_left_right_independent`: Left side on pre-step angles, right side on post-step angles (both CLL positions can span the step)
- `test_multiphase_hysteresis_chemical_step_cll_crossing`: When a CLL crosses the step, CA targets switch mid-optimization

---

## 9. IMPLEMENTATION CHECKLIST

- [x] Clean `update_wetting_state` signature (remove legacy defaults)
- [x] Add `_get_hysteresis_window_chemical_step` helper
- [x] Add `update_wetting_state_chemical_step` function
- [x] Create `_multiphase_hysteresis_chemical_step.py` step operator
- [x] Register operator in `operators/step/__init__.py`
- [x] Export `update_wetting_state_chemical_step` from hysteresis module
- [x] Update `setup.py` factory to select based on config (already done in Phase 6)

---

## 10. SUMMARY

Phase 8 introduces **spatial heterogeneity** to wetting hysteresis: CA targets are no longer global constants but depend on each contact line's position relative to a chemical step. The optimizer remains unchanged (still 4 parameters, same objective functions), but the target values fed to the optimizer are now **per-side, per-timestep, determined dynamically** from configuration and measured CLL position.

The change is minimal and surgical: two lines in the wetting-update function select per-side CA windows instead of reading a single global pair from config. Everything else—initialization, optimization loop, hysteresis logic—remains identical.

