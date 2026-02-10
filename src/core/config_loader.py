"""ConfigLoader — handles config building and simulation name auto-detection."""

import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigLoader:
    """Loads config from file path or kwargs, normalizes collision, infers simulation name."""

    @staticmethod
    def load(config_path_or_kwargs: Union[str, Path, Dict[str, Any]], **overrides) -> Dict[str, Any]:
        """
        Load config from:
        - a dict of kwargs
        - a TOML file path (requires Python 3.11+ for tomllib)
        """
        if isinstance(config_path_or_kwargs, dict):
            cfg = dict(config_path_or_kwargs)
        else:
            path = Path(config_path_or_kwargs).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            import tomllib  # Python 3.11+
            with path.open("rb") as f:
                cfg = tomllib.load(f)
        cfg.update(overrides)
        return cfg

    @staticmethod
    def normalise_collision(collision, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise collision kwarg into config dict."""
        if collision is None:
            return kwargs
        if isinstance(collision, str):
            collision_cfg = {"collision_scheme": collision}
        elif isinstance(collision, dict):
            collision_cfg = collision.copy()
        else:
            raise ValueError(
                "collision must be either a string (for BGK) or dict (for MRT config)."
            )
        merged = dict(kwargs)
        merged.update(collision_cfg)
        return merged

    @staticmethod
    def infer_simulation_name(default: Optional[str] = None) -> Optional[str]:
        """Auto-detect simulation name from calling function via stack frame inspection."""
        if default is not None:
            return default
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back if frame else None
            while caller_frame:
                func_name = caller_frame.f_code.co_name
                if func_name != "<module>" and not func_name.startswith("_"):
                    return func_name
                caller_frame = caller_frame.f_back
        finally:
            del frame
        return None

    @staticmethod
    def build_config(
        *,
        simulation_type: str,
        save_interval: int,
        results_dir: str,
        init_type: str,
        init_dir: Optional[str],
        skip_interval: int,
        save_fields: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Build a normalized config dict."""
        return dict(
            simulation_type=simulation_type,
            save_interval=save_interval,
            results_dir=results_dir,
            init_type=init_type,
            init_dir=init_dir,
            skip_interval=skip_interval,
            save_fields=save_fields,
            **kwargs,
        )
