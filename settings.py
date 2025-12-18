"""
Settings management for the gesture control app.
Persists settings using JSON file.
"""
import json
import os
import shutil
from pathlib import Path


class Settings:
    """Manages application settings with persistence."""
    
    def __init__(self, config_file=None):
        if config_file is None:
            config_dir = Path.home() / ".axis"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "settings.json"
            
            legacy_file = Path.home() / ".gesture_control" / "settings.json"
            if not config_file.exists() and legacy_file.exists():
                try:
                    shutil.copy2(legacy_file, config_file)
                except Exception as e:
                    print(f"Warning: Could not migrate legacy settings: {e}")
        
        self.config_file = Path(config_file)
        self._settings = self._load_settings()
    
    def _load_settings(self):
        """Load settings from JSON file."""
        default_settings = {
            "camera_index": 0,
            "box_scale": 0.65,
            "smooth_factor": 0.35,
            "hover_threshold": 0.32,
            "click_threshold": 0.1,
            "detection_confidence": 0.65,
            "tracking_confidence": 0.55,
            "mirror": True,
            "overlay_visible": False,
            # One Euro Filter settings
            "use_one_euro": True,
            "one_euro_smoothness": 0.5,  # 0=max smoothing, 1=max responsiveness
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    # Merge with defaults to handle missing keys
                    default_settings.update(loaded)
                    return default_settings
            except Exception as e:
                print(f"Warning: Could not load settings: {e}")
                return default_settings
        
        return default_settings
    
    def _save_settings(self):
        """Save settings to JSON file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._settings, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")
    
    def get(self, key, default=None):
        """Get a setting value."""
        return self._settings.get(key, default)
    
    def set(self, key, value):
        """Set a setting value and save."""
        self._settings[key] = value
        self._save_settings()
    
    def update(self, **kwargs):
        """Update multiple settings at once."""
        self._settings.update(kwargs)
        self._save_settings()
    
    def get_all(self):
        """Get all settings as a dictionary."""
        return self._settings.copy()

    def get_one_euro_params(self):
        """
        Convert smoothness slider value to One Euro Filter parameters.

        Smoothness scale (0.0 to 1.0):
        - 0.0 = Maximum smoothing (less jitter, more lag)
        - 0.5 = Balanced (default)
        - 1.0 = Maximum responsiveness (faster reaction, might be jittery)

        Returns:
            tuple: (min_cutoff, beta) parameters for OneEuroFilter
        """
        smoothness = self.get("one_euro_smoothness", 0.5)

        # Map smoothness to min_cutoff (range: 0.5 to 2.0)
        # Lower smoothness = lower cutoff = more smoothing
        min_cutoff = 0.5 + smoothness * 1.5

        # Map smoothness to beta (range: 0.005 to 0.02)
        # Higher smoothness = higher beta = more responsive
        beta = 0.005 + smoothness * 0.015

        return min_cutoff, beta

