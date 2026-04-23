#!/usr/bin/env python
"""Phase 1 checkpoint: Verify tud_lbm structure is importable."""

import sys

def test_imports():
    """Test that tud_lbm modules can be imported."""
    try:
        # Core lattice
        from tud_lbm.lattice import Lattice, build_lattice
        print("✓ tud_lbm.lattice imports work")
        
        # Lattice instantiation
        lat = build_lattice("D2Q9")
        assert lat.d == 2
        assert lat.q == 9
        print(f"  ✓ D2Q9 lattice: d={lat.d}, q={lat.q}")
        
        # Config
        from tud_lbm.config import SimulationConfig
        print("✓ tud_lbm.config imports work")
        
        # Operators
        from tud_lbm.operators import CollisionOperator
        print("✓ tud_lbm.operators imports work")
        
        # Pipeline
        from tud_lbm.pipeline import build_setup, run
        print("✓ tud_lbm.pipeline imports work")
        
        # Registry
        from tud_lbm import registry
        print("✓ tud_lbm.registry imports work")
        
        return True
    
    except Exception as e:
        print(f"✗ Import failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
