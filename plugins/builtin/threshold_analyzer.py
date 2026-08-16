"""
Threshold analyzer plugin - monitors data and triggers actions when thresholds are crossed.
"""

import numpy as np
from typing import Dict, Any, List
from ..base_plugin import BasePlugin, PluginData, PluginResult, TimeSeriesPlugin


class ThresholdAnalyzerPlugin(TimeSeriesPlugin):
    """Plugin that monitors data for threshold crossings and can trigger actions."""
    
    def get_name(self) -> str:
        return "Threshold Analyzer"
    
    def get_description(self) -> str:
        return "Monitors data for threshold crossings and can trigger stage movement or alerts"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def __init__(self):
        super().__init__()
        self.author = "Microscope Controller"
        self.description = self.get_description()
        
        # Configuration
        self.config = {
            "detector_id": "",  # Which detector to monitor
            "threshold_value": 0.5,  # Threshold value
            "threshold_type": "above",  # "above" or "below"
            "consecutive_measurements": 3,  # Number of consecutive measurements to trigger
            "action_on_trigger": "move",  # "move", "alert", or "none"
            "target_axis": "x",  # Axis to move when triggered
            "target_position": 0.0,  # Position to move to
            "reset_after_trigger": True,  # Reset counter after trigger
        }
        
        # Internal state
        self._consecutive_count = 0
        self._last_triggered = False
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize plugin with configuration."""
        if config:
            self.config.update(config)
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> tuple:
        """Validate plugin configuration."""
        required_keys = ["detector_id", "threshold_value", "threshold_type", "consecutive_measurements"]
        for key in required_keys:
            if key not in config:
                return False, f"Missing required configuration key: {key}"
        
        if config["threshold_type"] not in ["above", "below"]:
            return False, "threshold_type must be 'above' or 'below'"
        
        if config["consecutive_measurements"] < 1:
            return False, "consecutive_measurements must be >= 1"
        
        return True, ""
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for UI generation."""
        return {
            "detector_id": {
                "type": "string",
                "default": "",
                "description": "Detector ID to monitor"
            },
            "threshold_value": {
                "type": "float",
                "default": 0.5,
                "description": "Threshold value to trigger on"
            },
            "threshold_type": {
                "type": "enum",
                "options": ["above", "below"],
                "default": "above",
                "description": "Trigger when value goes above or below threshold"
            },
            "consecutive_measurements": {
                "type": "int",
                "min": 1,
                "max": 100,
                "default": 3,
                "description": "Number of consecutive measurements to trigger"
            },
            "action_on_trigger": {
                "type": "enum",
                "options": ["move", "alert", "none"],
                "default": "move",
                "description": "Action to take when threshold is crossed"
            },
            "target_axis": {
                "type": "enum",
                "options": ["x", "y", "z"],
                "default": "x",
                "description": "Axis to move when triggered"
            },
            "target_position": {
                "type": "float",
                "default": 0.0,
                "description": "Position to move to when triggered"
            },
            "reset_after_trigger": {
                "type": "bool",
                "default": True,
                "description": "Reset consecutive counter after trigger"
            }
        }
    
    def get_required_detectors(self) -> List[str]:
        """Return list of detector IDs this plugin requires."""
        detector_id = self.config.get("detector_id", "")
        return [detector_id] if detector_id else []
    
    def get_required_axes(self) -> List[str]:
        """Return list of axis names this plugin can control."""
        action = self.config.get("action_on_trigger", "move")
        if action == "move":
            return [self.config.get("target_axis", "x")]
        return []
    
    def process_data(self, data: PluginData) -> PluginResult:
        """Process measurement data and check for threshold crossings."""
        result = PluginResult()
        
        detector_id = self.config.get("detector_id", "")
        if not detector_id or detector_id not in data.detector_data:
            result.success = False
            result.message = f"Detector {detector_id} not found in data"
            return result
        
        detector_values = data.detector_data[detector_id]
        
        # Get the latest value (assuming time series)
        if len(detector_values) > 0:
            current_value = float(detector_values[-1])
        else:
            result.success = False
            result.message = "No data available"
            return result
        
        # Check threshold condition
        threshold = self.config.get("threshold_value", 0.5)
        threshold_type = self.config.get("threshold_type", "above")
        
        threshold_crossed = False
        if threshold_type == "above":
            threshold_crossed = current_value > threshold
        else:
            threshold_crossed = current_value < threshold
        
        # Update consecutive count
        if threshold_crossed:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0
        
        # Check if we should trigger
        consecutive_needed = self.config.get("consecutive_measurements", 3)
        if self._consecutive_count >= consecutive_needed:
            # Trigger action
            action = self.config.get("action_on_trigger", "move")
            
            if action == "move":
                target_axis = self.config.get("target_axis", "x")
                target_position = self.config.get("target_position", 0.0)
                
                result.move_commands.append({
                    "axis": target_axis,
                    "position": target_position,
                    "relative": False
                })
                
                result.message = f"Threshold crossed ({current_value:.3f} {threshold_type} {threshold}), moving {target_axis} to {target_position:.3f}"
            elif action == "alert":
                result.message = f"THRESHOLD ALERT: {current_value:.3f} {threshold_type} {threshold}"
                result.extracted_features = {
                    "alert": True,
                    "value": current_value,
                    "threshold": threshold,
                    "threshold_type": threshold_type
                }
            else:
                result.message = f"Threshold crossed but no action configured"
            
            # Reset if configured
            if self.config.get("reset_after_trigger", True):
                self._consecutive_count = 0
            
            self._last_triggered = True
        else:
            result.message = f"Value {current_value:.3f}, consecutive count: {self._consecutive_count}/{consecutive_needed}"
            self._last_triggered = False
        
        result.extracted_features.update({
            "current_value": current_value,
            "threshold": threshold,
            "threshold_crossed": threshold_crossed,
            "consecutive_count": self._consecutive_count
        })
        
        return result
    
    def on_measurement_start(self, data: PluginData) -> None:
        """Reset state when measurement starts."""
        self._consecutive_count = 0
        self._last_triggered = False