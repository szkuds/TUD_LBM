# Phase 1 — Inventory and Call Graph Report

**Date:** 2026-05-04  
**Status:** Complete code read — no changes made.

---

## 1. MULTIPHASE PIPELINE CALL GRAPH

### 1.1 Entry Point: Single Timestep (Non-wetting)

```
runner.run() [lax.scan body]
    ↓
setup.step_fn(setup, state)  [registry dispatch]
    ↓
step_multiphase(setup, state)  [@update_timestep_operator(name="multiphase")]
    ↓
[1] compute_total_force_ext(setup, state, setup.forces)
    ↓
[2] _run_multiphase_pipeline(setup, state.f, force_ext=force_ext, wetting=None)
    ↓
    • macroscopic_fn(f_t, lattice, multiphase_params, force_ext, ...)
        → returns (rho, u, force_tot)
    ↓
    • temp_state = State(f=f_t, rho=rho, u=u, t=0, wetting=None)
    ↓
    • _apply_common_step(setup, temp_state, rho, u, force_tot)
        ├─ equilibrium_fn(rho, u, lattice)
        ├─ collision_fn(state.f, feq, setup.tau) [no source term]
        ├─ streaming_fn(f_col, lattice)
        └─ bc_fn(f_stream, f_col, setup.bc_masks)
    ↓
    • macroscopic_fn() again on next_state.f
        → returns (rho_next, u_next, _)
    ↓
    Returns: (f_out, rho_next, u_next, force_tot)
    ↓
[3] update_extra_state(setup, state, _new_state, force_ext=force_ext)
    ↓
    • WettingExtraStatePlugin.update_state() [if wetting active]
        ↓
        setup.wetting_fn(prev_state.wetting, new_state.rho, setup, prev_state.f, ...)
            ↓
            update_wetting_state()  [@wetting_operator(name="hysteresis")]
                ↓ [**INNER LOOP BEGINS**]
                [See § 1.2 below]
                ↓ [**INNER LOOP ENDS**]
        ↓
        Returns: updated WettingState
    ↓
Return: new State with t incremented
```

### 1.2 Hysteresis Inner Loop (Multiple Trial Steps)

**Call chain per hysteresis iteration** (inside `jax.lax.while_loop` at hysteresis.py:131–139):

```
update_wetting_state(wetting_t, rho_t_plus1, setup, f_t, evaluate_fn=None, force_ext=None)
    [registered as @wetting_operator(name="hysteresis")]
    
    ↓ Setup phase (pure computations):
    • compute_contact_angle(rho_t_plus1, rho_mean) → (ca_l, ca_r)
    • compute_contact_line_location(...) → (cll_l, cll_r)
    • Infer advancing/receding direction from delta_cll_*
    • Build gradient masks for left/right sides
    
    ↓ Build evaluate_fn if not supplied:
    
    if evaluate_fn is None:
        evaluate_fn = _build_default_evaluate_fn(setup, f_t, force_ext, rho_mean)
    
    ↓ Optimization loop (nested while_loop):
    
    _optimise_single_param(
        objective_fn=left_objective,
        initial_params,
        grad_mask_fn=left_mask,
        optimiser=optax.adam(lr),
        max_iterations,
        loss_tol
    )
        ↓
        [jax.lax.while_loop body, iterates up to max_iterations]
        
        Loop body per iteration:
            • loss, grads = jax.value_and_grad(left_objective)(params)
            • grads = left_mask(grads)  [zero non-left parameters]
            • updates, new_opt_state = optimiser.update(grads, opt_state, params)
            • new_params = optax.apply_updates(params, updates)
            • new_params = _clamp_params(new_params)
            
            Inside left_objective(p):
                ca_l, _, cll_l, _ = evaluate_fn(p)
                cost = (if in_window: _cost_cll(target, cll_l)
                       else: _cost_ca(ca_target_left, ca_l))
    
    ↓ Then optimize right side identically:
    
    _optimise_single_param(
        objective_fn=right_objective,
        initial_params=p1,  [from left optimization]
        grad_mask_fn=right_mask,
        ...
    )
    
    ↓ Return updated WettingState with new parameters
```

### 1.3 Trial Step Inside Hysteresis Evaluate Function

**Called by `jax.value_and_grad(left_objective)(params)` during optimization:**

```
evaluate_fn(params: WettingParams)
    [Built by _build_default_evaluate_fn() at hysteresis.py:407–463]
    
    ↓ Prepare trial wetting state:
    
    temp_wetting = WettingState(
        # legacy scalar fields from params (all use '_pre' value)
        d_rho_left = params.d_rho_left_pre,
        d_rho_right = params.d_rho_right_pre,
        phi_left = params.phi_left_pre,
        phi_right = params.phi_right_pre,
        # per-region pre/post from params
        d_rho_left_pre = params.d_rho_left_pre,
        d_rho_left_post = params.d_rho_left_post,
        ... all 8 pre/post fields ...
        ca_left = 0, ca_right = 0,  [dummy values]
        cll_left = 0, cll_right = 0,  [dummy values]
    )
    
    ↓ Build wetting shims if wetting_config is not None:
    
    if setup.config.wetting_config is not None:
        gradient_density, laplacian_density = _make_wetting_differential_ops(
            setup,
            temp_wetting
        )
    else:
        gradient_density = setup.gradient_density
        laplacian_density = setup.laplacian_density
    
    ↓ Run trial step:
    
    f_out = setup.multiphase_step(
        f_t,
        force_ext=force_ext,
        gradient_density=gradient_density,
        laplacian_density=laplacian_density,
    )
        ↓
        [See § 1.4 — Trial step execution]
    
    ↓ Measure outcomes:
    
    rho_out = jnp.sum(f_out, axis=-2, keepdims=True)
    ca_l, ca_r = compute_contact_angle(rho_out, rho_mean)
    cll_l, cll_r = compute_contact_line_location(rho_out, ca_l, ca_r, rho_mean)
    
    ↓ Return: (ca_l, ca_r, cll_l, cll_r)  [differentiated by jax.value_and_grad]
```

### 1.4 Trial Multiphase Step (Inside Hysteresis Evaluate)

**Executed inside `setup.multiphase_step()` closure bound at setup.py:256–272:**

```
_multiphase_step_bound(
    f_t,
    force_ext=force_ext,
    wetting=None,  [ignored; wetting passed separately as shims]
    gradient_density=gradient_density,  [wetting-corrected shim]
    laplacian_density=laplacian_density  [wetting-corrected shim]
) → f_out
    ↓
    Call: multiphase_step(setup, f_t, ...)  [not step_multiphase!]
    ↓
    _run_multiphase_pipeline(
        setup, f_t,
        force_ext=None,  [trial step has no external forces]
        wetting=None,  [wetting params injected as shims, not WettingState]
        gradient_density=<wetting shim>,  [pre-computed per trial param]
        laplacian_density=<wetting shim>   [pre-computed per trial param]
    )
        ↓
        • macroscopic_fn(f_t, lattice, multiphase_params, None, ...,
                         laplacian_density=<wetting shim>)
            → (rho, u, force_tot=0)
        ↓
        • temp_state = State(f=f_t, rho=rho, u=u, t=0, wetting=None)
        ↓
        • _apply_common_step(
            setup, temp_state, rho, u, 0,
            gradient_density=<wetting shim>  [used in collision source]
          )
            ├─ equilibrium_fn(rho, u, lattice)
            ├─ collision_fn(f_col, feq, tau) [no source term — force_tot is 0]
            ├─ streaming_fn(f_col, lattice)
            └─ bc_fn(f_stream, f_col, bc_masks)
        ↓
        • macroscopic_fn() again
        ↓
        Return: (f_out, rho_out, u_out, 0)
    ↓
    Return: f_out [only population; other outputs discarded]
```

---

## 2. WETTING DIFFERENTIAL OPERATOR SHIMS

### 2.1 Creation Sites

#### Site 1: Main multiphase step (_multiphase.py:35)

```python
if gradient_density is None or laplacian_density is None:
    if wetting is not None and setup.config.wetting_config is not None:
        wet_grad, wet_lap = _make_wetting_differential_ops(setup, wetting)
    else:
        wet_grad = setup.gradient_density
        wet_lap = setup.laplacian_density
    if gradient_density is None:
        gradient_density = wet_grad
    if laplacian_density is None:
        laplacian_density = wet_lap
```

- **Trigger:** `wetting is not None` (i.e., WettingState object provided) AND `setup.config.wetting_config is not None`
- **Inputs:** `setup` (SimulationSetup), `wetting` (WettingState instance)
- **Outputs:** `(wetting_gradient_density, wetting_laplacian_density)` — two closures

#### Site 2: Hysteresis evaluate function (_build_default_evaluate_fn at hysteresis.py:440)

```python
if setup.config.wetting_config is not None:
    gradient_density, laplacian_density = _make_wetting_differential_ops(
        setup,
        temp_wetting,
    )
else:
    gradient_density = setup.gradient_density
    laplacian_density = setup.laplacian_density
```

- **Trigger:** `setup.config.wetting_config is not None`
- **Inputs:** `setup` (SimulationSetup), `temp_wetting` (WettingState with trial parameters)
- **Outputs:** Same as Site 1

### 2.2 Function Implementation

**Location:** `tud_lbm/operators/step/_wetting_differential_operators.py:18–68`

```python
def _make_wetting_differential_ops(
    setup: SimulationSetup,
    wetting_state: WettingState
) → tuple[Callable, Callable]
```

**Closure extraction:**
1. Checks for chemical step config (`csc = setup.config.chemical_step_config`)
2. If chemical step present:
   - Reads `wetting_state.phi_left_pre`, `.phi_left_post`, etc.
   - Splits per-column based on `cols < step_ix` predicate (static)
   - Uses `jnp.where` to select active region per-column
3. If no chemical step:
   - Uses scalar fields: `wetting_state.phi_left`, `.d_rho_left`, etc.

**Returned closures:**
```python
def wetting_gradient_density(grid: jnp.ndarray) → jnp.ndarray:
    return setup.gradient_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)

def wetting_laplacian_density(grid: jnp.ndarray) → jnp.ndarray:
    return setup.laplacian_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)
```

---

## 3. LEGACY WETTINGSTATE FIELD READS

### 3.1 Scalar (Non-per-region) Fields

These fields exist as fallback/compatibility and are directly read at:

- `wetting.d_rho_left` — read at hysteresis.py:280–283 (warm-start)
- `wetting.d_rho_right` — read at hysteresis.py:280–283 (warm-start)
- `wetting.phi_left` — read at hysteresis.py:280–283 (warm-start)
- `wetting.phi_right` — read at hysteresis.py:280–283 (warm-start)

**Also initialized at:**
- `_extra_state.py:62–65` — WettingExtraStatePlugin.init_state()

### 3.2 Per-Region (Pre/Post) Fields

#### Reads in hysteresis.py

**Warm-start phase (lines 265–283):**
- `wetting_t.phi_left_pre`, `.phi_left_post`
- `wetting_t.d_rho_left_pre`, `.d_rho_left_post`
- `wetting_t.phi_right_pre`, `.phi_right_post`
- `wetting_t.d_rho_right_pre`, `.d_rho_right_post`

All wrapped in `jnp.where(active_predicate, value_from_wetting, seed_value)` to implement warm-start with per-region seeding.

**Used in WettingParams bundle (lines 260–285):**
```python
params = WettingParams(
    phi_left_pre=jnp.where(left_pre_active, wetting_t.phi_left_pre, phi_pre_seed),
    ...  # 8 fields total
)
```

**No-chemical-step branch (lines 275–283):**
```python
params = WettingParams(
    phi_left_pre=wetting_t.phi_left_pre,
    phi_left_post=wetting_t.phi_left_post,
    d_rho_left_pre=wetting_t.d_rho_left_pre,
    d_rho_left_post=wetting_t.d_rho_left_post,
    phi_right_pre=wetting_t.phi_right_pre,
    ...  # 4 more fields
)
```

#### Reads in _wetting_differential_operators.py

**Chemical step branch (lines 48–52):**
- `wetting_state.phi_left_pre`, `.phi_left_post`
- `wetting_state.d_rho_left_pre`, `.d_rho_left_post`
- `wetting_state.phi_right_pre`, `.phi_right_post`
- `wetting_state.d_rho_right_pre`, `.d_rho_right_post`

**No-chemical-step branch (lines 54–57):**
- `wetting_state.phi_left`, `.phi_right`
- `wetting_state.d_rho_left`, `.d_rho_right`

### 3.3 Measurement Fields (Outputs)

These fields are **written** at hysteresis.py:369–377 (never read as input):
- `ca_left`, `ca_right` — current contact angles
- `cll_left`, `cll_right` — current contact-line locations
- `step_crossed_left`, `step_crossed_right` — crossing-event flags

---

## 4. OPERATOR REGISTRY — STEP OPERATORS

### 4.1 Registered Update-Timestep Operators (Kind: `"update_timestep"`)

| Name | Module | Function | Signature |
|------|--------|----------|-----------|
| `"single_phase"` | `operators/step/_single_phase.py` | `step_single_phase` | `(setup, state) → state` |
| `"multiphase"` | `operators/step/_multiphase.py` | `step_multiphase` | `(setup, state) → state` |

**Decorators:**
```python
@update_timestep_operator(name="single_phase")  # _single_phase.py:18
@update_timestep_operator(name="multiphase")    # _multiphase.py:93
```

**Lookup path:**
```
runner.run() → setup.step_fn = build_step_fn(config.sim_type)
            → build_operator("update_timestep", scheme)
            → OPERATOR_REGISTRY[f"update_timestep:{scheme}"]
```

### 4.2 Registered Wetting Operators (Kind: `"wetting"`)

| Name | Module | Function | Purpose |
|------|--------|----------|---------|
| `"contact_angle"` | `operators/wetting/_contact_angle.py` | `compute_contact_angle` | Measure CA from density |
| `"contact_line_location"` | `operators/wetting/_contact_line.py` | `compute_contact_line_location` | Measure CLL from density |
| `"hysteresis"` | `operators/wetting/hysteresis/hysteresis.py` | `update_wetting_state` | Main hysteresis loop |
| `"applicator"` | `operators/wetting/_applicator.py` | (class-based) | Legacy applicator |
| `"resolve_wetting_fields"` | `operators/wetting/_resolve_fields.py` | (class-based) | Field resolution |

**Hysteresis lookup path:**
```
setup.wetting_fn = build_wetting_fn("hysteresis")
                 → build_operator("wetting", "hysteresis")
                 → OPERATOR_REGISTRY["wetting:hysteresis"]
                 → update_wetting_state
```

---

## 5. `setup.multiphase_step` BINDING AND REFERENCES

### 5.1 Binding Site

**Location:** `pipeline/setup.py:254–273` (inside `build_setup()`)

```python
if "multiphase" in config.sim_type:
    from tud_lbm.operators.step._multiphase import multiphase_step

    def _multiphase_step_bound(
        f_t: jnp.ndarray,
        *,
        force_ext: jnp.ndarray | None = None,
        wetting: WettingState = None,
        gradient_density: DifferentialOperator = None,
        laplacian_density: DifferentialOperator = None,
    ) -> jnp.ndarray:
        return multiphase_step(
            setup,
            f_t,
            force_ext=force_ext,
            wetting=wetting,
            gradient_density=gradient_density,
            laplacian_density=laplacian_density,
        )

    setup = setup._replace(multiphase_step=_multiphase_step_bound)
```

- **When bound:** Only for multiphase runs (`"multiphase" in config.sim_type`)
- **Captures:** The `setup` object (via closure)
- **Type annotation:** `StepOperator[..., jnp.ndarray] | None` (returns only `f_out`, not full State)

### 5.2 Reference Sites

#### Reference 1: Hysteresis evaluate function

**Location:** `operators/wetting/hysteresis/hysteresis.py:448–453`

```python
if setup.multiphase_step is None:
    msg = "Multiphase hysteresis requires setup.multiphase_step to be configured."
    raise ValueError(msg)

f_out = setup.multiphase_step(
    f_t,
    force_ext=force_ext,
    gradient_density=gradient_density,
    laplacian_density=laplacian_density,
)
```

- Called once per hysteresis trial step (inside optimization loop)
- Provides trial populations for CA/CLL measurement

#### Reference 2: SimulationSetup docstring

**Location:** `pipeline/setup.py:97–99`

```python
multiphase_step: Optional setup-bound multiphase trial-step callable.
    Signature:
    ``(f_t, force_ext=None, wetting=None, gradient_density=None, laplacian_density=None) -> f_out``.
```

No runtime reference; documentation only.

#### Reference 3: Type definition

**Location:** `pipeline/setup.py:134`

```python
multiphase_step: StepOperator[..., jnp.ndarray] | None = None
```

Class attribute definition.

---

## 6. UNEXPECTED FINDINGS

### 6.1 **No Unused Wetting Fields**

All per-region pre/post fields in `WettingState` are actively used:
- Explicitly read in `hysteresis.py` warm-start loop
- Explicitly written at the end of `update_wetting_state`
- Passed through `_make_wetting_differential_ops`

**Implication:** No dead code; all fields serve the chemical-step splitting logic.

### 6.2 **Legacy Scalar Fields Remain as Fallback**

`WettingState` still carries scalar (`d_rho_left`, `phi_left`, etc.) even though per-region variants exist.

- **Lines 62–65 (_extra_state.py):** Scalar fields initialized
- **Lines 58–65 (hysteresis.py, no chem step):** Scalar fields used directly as fallback
- **Lines 280–283 (hysteresis.py, chem step present):** Scalar fields warm-started from `params` pre-values

**Design:** Two-path compatibility—supports both legacy (no per-region) and new (per-region) workflows.

### 6.3 **Wetting Shim Closures Close Over Live Parameters**

In `_make_wetting_differential_ops`, the returned closures capture the **current** `wetting_state` values:

```python
phi_l = jnp.where(cols < step_ix, wetting_state.phi_left_pre, wetting_state.phi_left_post)
# ... (computed once at shim creation)

def wetting_gradient_density(grid: jnp.ndarray) -> jnp.ndarray:
    return setup.gradient_density(grid, phi_l, phi_r, d_rho_l, d_rho_r)
    # phi_l, etc. are closed-over constants (not re-read from state)
```

**Critical for gradients:** `jax.value_and_grad` can only differentiate through parameters passed as arguments or closed over; it **cannot** differentiate through a lookup to an external state object. This design is correct.

### 6.4 **No Explicit Wetting Gradient Masking in Trial Step**

Inside the trial step (hysteresis evaluate function), the collision source term is **not** computed (force_tot is 0), so the `gradient_density` shim is present but not used.

**Path:**
1. `_apply_common_step(..., force_tot=0, gradient_density=<shim>)`
2. Inside: `if force_tot is not None` → condition is FALSE
3. Gradient is never fetched

**Implication:** The wetting shim is created but discarded; only the post-step `compute_contact_angle` / `compute_contact_line_location` are affected by the step. This is correct—the shim must be passed to ensure the macroscopic function uses it for density in the tanh profile.

### 6.5 **Chemical Step Config Influences Three Distinct Sites**

All three sites read `setup.config.chemical_step_config` and apply per-column masking:

1. **_wetting_differential_operators.py:45–57** — Build per-column phi/d_rho arrays
2. **hysteresis.py:200–206** — Detect crossing events
3. **hysteresis.py:216–226** — Select per-side advancing/receding contact angles

All use static `step_ix = int(float(csc["chemical_step_location"]) * nx)` (computed once).

### 6.6 **Extra-State Plugin Architecture Decouples Wetting Update**

The `WettingExtraStatePlugin` sits as an extra-state plugin, not directly in the step function:

- **Plugin init:** `_extra_state.py:33–88`
- **Plugin update:** `_extra_state.py:94–109`

This allows wetting to be optional and modular—if wetting_config is absent, the plugin is not activated.

### 6.7 **Trial Step Does NOT Read Parent Wetting State**

Inside the hysteresis trial step:
- A **temporary** `temp_wetting` is created from trial parameters
- `setup.multiphase_step` is called with `wetting=None` (implicit in closure)
- The parent `wetting_t` is **not** passed to or read by the trial step

This is correct—each trial should be independent and differ only in the injected parameter values.

### 6.8 **Double Macroscopic Evaluation in Main Step**

`_run_multiphase_pipeline` calls the macroscopic function **twice**:
1. At start (lines 46–50) to compute initial (rho, u)
2. After streaming (lines 66–69) to recompute on updated f

**Purpose:** The second call provides the rho/u for the next step (as output). This is not used during the step itself but is returned and stored in State. Trial steps (hysteresis) discard the return value after u,rho.

---

## 7. SUMMARY TABLE

| Concept | Location(s) | Key Aspect |
|---------|------------|-----------|
| **Main entry** | `runner.py:195–205` | `scan_body(state, _t)` calls `setup.step_fn(setup, state)` |
| **Multiphase step** | `_multiphase.py:93–129` | Dispatcher; calls `_run_multiphase_pipeline` |
| **Pipeline impl** | `_multiphase.py:28–69` | Core LBM sequence with optional wetting shims |
| **Common step** | `_common.py:16–58` | Equilibrium → collision → streaming → BC (shared) |
| **Wetting shims** | `_wetting_differential_operators.py:18–68` | Closure generator for live parameter injection |
| **Hysteresis** | `hysteresis.py:138–377` | Main wetting update; optimizes CA/CLL via trial steps |
| **Hysteresis evaluate** | `hysteresis.py:407–463` | Builds temp wetting state and runs trial step |
| **Trial step binding** | `setup.py:254–273` | Creates `_multiphase_step_bound` closure (for `setup.multiphase_step`) |
| **Wetting plugin** | `_extra_state.py:24–109` | Initializes and updates wetting state after each step |
| **Step registry** | `registry.py` + decorators | `update_timestep:single_phase` and `update_timestep:multiphase` |
| **Wetting registry** | `registry.py` + decorators | `wetting:contact_angle`, `wetting:contact_line_location`, `wetting:hysteresis`, etc. |
| **Legacy fields** | `state.py:32–61` | Scalar (`d_rho_left`, `phi_left`, etc.) + per-region pre/post fields |
| **Shim invocation (1)** | `_multiphase.py:35–41` | Calls `_make_wetting_differential_ops` if wetting active |
| **Shim invocation (2)** | `hysteresis.py:440` | Builds trial shim inside evaluate_fn |
| **Chemical step** | Config + three sites | Splits wetting params per-column at `step_ix` |

---

## 8. CALL GRAPH — SIMPLIFIED LOOP STRUCTURE

```
[Outer: lax.scan over nt timesteps]
    │
    └─→ step_multiphase(setup, state)  [per timestep]
            ├─ compute_total_force_ext(...)
            ├─ _run_multiphase_pipeline(..., wetting=state.wetting)
            │   ├─ If wetting is not None and config has wetting:
            │   │   └─ _make_wetting_differential_ops(setup, wetting)  [shim 1]
            │   ├─ macroscopic_fn(..., laplacian_density=<shim or default>)
            │   ├─ _apply_common_step(..., gradient_density=<shim or default>)
            │   └─ macroscopic_fn(...) again
            └─ update_extra_state(...)
                └─ WettingExtraStatePlugin.update_state(...)
                    └─ setup.wetting_fn(...)  [== update_wetting_state]
                        │
                        ├─ compute_contact_angle(rho_t_plus1)
                        ├─ compute_contact_line_location(...)
                        │
                        ├─ [Inner: two sequential while_loop optimizations]
                        │   ├─ _optimise_single_param(left_objective, ...)
                        │   │   └─ [Inner: while_loop per iteration]
                        │   │       ├─ jax.value_and_grad(left_objective)(params)
                        │   │       │   └─ evaluate_fn(params)
                        │   │       │       ├─ _make_wetting_differential_ops(setup, temp_wetting)  [shim 2]
                        │   │       │       ├─ setup.multiphase_step(f_t, ..., gradient_density=<shim>, ...)
                        │   │       │       │   └─ _run_multiphase_pipeline(..., wetting=None, gradient_density=<shim>, ...)
                        │   │       │       │       ├─ macroscopic_fn(..., laplacian_density=<shim>)
                        │   │       │       │       ├─ _apply_common_step(..., gradient_density=<shim>)
                        │   │       │       │       └─ macroscopic_fn(..., laplacian_density=<shim>)
                        │   │       │       ├─ compute_contact_angle(rho_out)
                        │   │       │       └─ compute_contact_line_location(...)
                        │   │       │           → (ca_l, ca_r, cll_l, cll_r)  [differentiated]
                        │   │       ├─ optax.update(grads, opt_state)
                        │   │       └─ optax.apply_updates(...)
                        │   │
                        │   └─ _optimise_single_param(right_objective, ...)  [symmetric to left]
                        │
                        └─ Return updated WettingState
```

---

## 9. FILES EXAMINED

1. `tud_lbm/operators/step/_multiphase.py` — 131 lines ✓
2. `tud_lbm/operators/step/_common.py` — 68 lines ✓
3. `tud_lbm/operators/step/_wetting_differential_operators.py` — 68 lines ✓
4. `tud_lbm/operators/wetting/hysteresis/hysteresis.py` — 465 lines ✓
5. `tud_lbm/pipeline/setup.py` — 276 lines ✓
6. `tud_lbm/pipeline/state/state.py` — 99 lines ✓
7. `tud_lbm/pipeline/runner.py` — 200 lines ✓
8. `tud_lbm/registry.py` — 285 lines ✓
9. `tud_lbm/operators/step/__init__.py` — 53 lines ✓
10. `tud_lbm/operators/step/_single_phase.py` — 42 lines ✓
11. `tud_lbm/operators/wetting/__init__.py` — 54 lines ✓
12. `tud_lbm/operators/wetting/_params.py` — 25 lines ✓
13. `tud_lbm/pipeline/state/__init__.py` — 83 lines ✓
14. `tud_lbm/pipeline/state/_extra_state.py` — 59 lines ✓
15. `tud_lbm/operators/wetting/_extra_state.py` — 113 lines ✓

---

## 10. PHASE 1 COMPLETE ✓

**Objective:** Read and map the current state without modifying any code.

**Deliverables:**
- ✓ Call graph showing all paths through multiphase + hysteresis trial step loop
- ✓ Inventory of all sites reading legacy WettingState fields
- ✓ Inventory of all sites creating wetting shims / differential operator closures
- ✓ Inventory of all sites referencing `setup.multiphase_step` or `_multiphase_step_bound`
- ✓ List of all registered step operators in the operator registry
- ✓ Notes on unexpected findings (chemical step design, closure semantics, legacy field fallback)

**No code was modified.**

