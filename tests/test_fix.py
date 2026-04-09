#!/usr/bin/env python3
"""Test script to verify the wetting step fix works."""
import sys
sys.path.insert(0, '../src')

from config.adapter_toml import TomlAdapter
from setup.simulation_setup import build_setup
from runner.run import init_state

def test_complex_config():
    """Test complex config with wetting_hysteresis."""
    print("=" * 60)
    print("Testing config_complex.toml workflow")
    print("=" * 60)

    try:
        adapter = TomlAdapter()
        cfg = adapter.load('examples/config_complex.toml')
        print(f"✓ Config loaded: sim_type={cfg.sim_type}")

        setup = build_setup(cfg)
        print(f"✓ Setup built: step_fn={setup.step_fn.__name__}")

        state = init_state(setup)
        print(f"✓ State initialized: f.shape={state.f.shape}")

        for i in range(1, 4):
            new_state = setup.step(state)
            print(f"✓ Step {i} completed: t={int(new_state.t)}")
            state = new_state

        print("\n" + "="*60)
        print("✓✓✓ SUCCESS: Example workflow works correctly!")
        print("="*60)
        return True
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_complex_config()
    sys.exit(0 if success else 1)

