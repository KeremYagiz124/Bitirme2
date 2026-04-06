"""
Multi-user development setup example.

This module demonstrates how to use the Config system
for consistent paths across all team members.
"""

from src.utils.config import config
import json


def example_usage():
    """Show how to use config for multi-user development."""
    
    print("=== Multi-User Config System ===\n")
    
    # 1. Get simple values
    model_size = config.get("model.yolo.model_size")
    print(f"Model size: {model_size}")
    
    # 2. Get paths (automatically resolved relative to project root)
    synthetic_dir = config.get("data.synthetic_dir")
    print(f"Synthetic data dir: {synthetic_dir}")
    
    # 3. Get nested configs
    train_split = config.get("training.split.train")
    print(f"Training split: {train_split}")
    
    # 4. Get defaults if key doesn't exist
    unknown = config.get("unknown.key", default="default_value")
    print(f"Unknown key (with default): {unknown}")
    
    print("\n=== How each team member uses this ===\n")
    
    print("Person A (Windows, C:/):")
    print("  - Uses same command: python run.py")
    print("  - Config system resolves paths automatically")
    print()
    
    print("Person B (Mac, /Users/*):")
    print("  - Uses same command: python run.py")
    print("  - Paths work identically")
    print()
    
    print("Person C (Linux, /home/*):")
    print("  - Uses same command: python run.py") 
    print("  - All paths portable across OS")
    print()
    
    print("=== Local overrides (config.local.yaml) ===\n")
    print("Each team member can create configs/config.local.yaml:")
    print("  - NOT tracked by git (in .gitignore)")
    print("  - Override project defaults")
    print("  - Example: use different GPU/device settings")
    print()
    
    example_local = {
        "model": {
            "yolo": {
                "cache_dir": "/fast_ssd/model_cache"
            }
        },
        "training": {
            "device": "cuda"
        }
    }
    print("Example config.local.yaml:")
    print(json.dumps(example_local, indent=2))


if __name__ == "__main__":
    example_usage()