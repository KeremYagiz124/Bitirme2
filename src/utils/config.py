"""Configuration loader for multi-user environment."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Centralized configuration management with relative paths."""

    def __init__(self):
        """Load configuration from YAML files."""
        # Project root (wherever this file is relative to project)
        self.PROJECT_ROOT = Path(__file__).parent.parent.parent
        
        # Load main config
        self.config_path = self.PROJECT_ROOT / "configs" / "config.yaml"
        self._config = self._load_yaml(self.config_path)
        
        # Load user-local config if exists (not tracked by git)
        local_config_path = self.PROJECT_ROOT / "configs" / "config.local.yaml"
        if local_config_path.exists():
            local_config = self._load_yaml(local_config_path)
            self._config.update(local_config)

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def get_path(self, key: str, relative: bool = True) -> Path:
        """
        Get a path from config, resolving relative to project root.
        
        Args:
            key: Config key (e.g., 'data.synthetic_dir')
            relative: If True, return path relative to project root
        
        Returns:
            Resolved Path object
        """
        value = self._get_nested(key)
        
        if not value:
            raise KeyError(f"Config key not found: {key}")
        
        path = Path(value)
        if not path.is_absolute():
            path = self.PROJECT_ROOT / path
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        return path if not relative else path.relative_to(self.PROJECT_ROOT)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with dot notation (e.g., 'model.yolo.model_size')."""
        return self._get_nested(key) or default

    def _get_nested(self, key: str) -> Any:
        """Get nested config value using dot notation."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return None
        
        return value

    def __repr__(self) -> str:
        return f"Config(root={self.PROJECT_ROOT})"


# Global config instance
config = Config()


def get_config() -> Config:
    """Get global config instance."""
    return config