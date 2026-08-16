"""
Peak finder plugin - finds peaks in 1D/2D data and can move stage to peak positions.
"""

import numpy as np
from typing import Dict, Any, List
from ..base_plugin import BasePlugin, PluginData, PluginResult, DecoderPlugin


class PeakFinderPlugin(DecoderPlugin):
    """Plugin that finds peaks in data and can move stage to peak positions."""
    
    def get_name(self) -> str:
        return "Peak Finder"
    
    def get_description(self) -> str:
        return "Finds peaks in 1D/2D data and can move stage to peak positions after N measurements"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def __init__(self):
        super().__init__()
        self.author = "Microscope Controller"
        self.description = self.get_description()
        
        # Configuration
        self.config = {
            "peak_threshold": 0.5,  # Threshold for peak detection
            "min_peak_distance": 10,  # Minimum distance between peaks
            "measurements_before_move": 5,  # Number of measurements before moving
            "target_axis": "x",  # Axis to move (x, y, or z)
            "move_to_highest_peak": True,  # Move to highest peak
            "move_offset": 0.0,  # Offset from peak position
        }
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize plugin with configuration."""
        if config:
            self.config.update(config)
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> tuple:
        """Validate plugin configuration."""
        required_keys = ["peak_threshold", "measurements_before_move", "target_axis"]
        for key in required_keys:
            if key not in config:
                return False, f"Missing required configuration key: {key}"
        
        if config["target_axis"] not in ["x", "y", "z"]:
            return False, "target_axis must be 'x', 'y', or 'z'"
        
        if config["measurements_before_move"] < 1:
            return False, "measurements_before_move must be >= 1"
        
        return True, ""
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for UI generation."""
        return {
            "peak_threshold": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "description": "Threshold for peak detection (0-1)"
            },
            "min_peak_distance": {
                "type": "int",
                "min": 1,
                "max": 1000,
                "default": 10,
                "description": "Minimum distance between peaks"
            },
            "measurements_before_move": {
                "type": "int",
                "min": 1,
                "max": 1000,
                "default": 5,
                "description": "Number of measurements before moving stage"
            },
            "target_axis": {
                "type": "enum",
                "options": ["x", "y", "z"],
                "default": "x",
                "description": "Axis to move to peak position"
            },
            "move_to_highest_peak": {
                "type": "bool",
                "default": True,
                "description": "Move to highest peak instead of first peak"
            },
            "move_offset": {
                "type": "float",
                "default": 0.0,
                "description": "Offset from peak position"
            }
        }
    
    def get_required_detectors(self) -> List[str]:
        """Return list of detector IDs this plugin requires."""
        return []  # Works with any detector
    
    def get_required_axes(self) -> List[str]:
        """Return list of axis names this plugin can control."""
        return [self.config.get("target_axis", "x")]
    
    def process_data(self, data: PluginData) -> PluginResult:
        """Process measurement data and find peaks."""
        result = PluginResult()
        
        # Get detector data
        if not data.detector_data:
            result.success = False
            result.message = "No detector data available"
            return result
        
        # Process first detector's data
        detector_id = list(data.detector_data.keys())[0]
        detector_values = data.detector_data[detector_id]
        
        # Find peaks in 1D data
        if len(detector_values.shape) == 1:
            peaks = self._find_1d_peaks(detector_values)
        elif len(detector_values.shape) == 2:
            peaks = self._find_2d_peaks(detector_values)
        else:
            result.success = False
            result.message = "Unsupported data dimensionality"
            return result
        
        result.extracted_features = {
            "peaks": peaks,
            "num_peaks": len(peaks),
            "peak_values": [detector_values[p] if len(detector_values.shape) == 1 else detector_values[p[0], p[1]] for p in peaks]
        }
        
        # Check if we should trigger movement
        self._measurement_count += 1
        measurements_needed = self.config.get("measurements_before_move", 5)
        
        if self._measurement_count >= measurements_needed and peaks:
            # Generate movement command
            target_axis = self.config.get("target_axis", "x")
            move_to_highest = self.config.get("move_to_highest_peak", True)
            offset = self.config.get("move_offset", 0.0)
            
            if move_to_highest:
                # Find highest peak
                peak_values = result.extracted_features["peak_values"]
                max_idx = np.argmax(peak_values)
                target_peak = peaks[max_idx]
            else:
                # Use first peak
                target_peak = peaks[0]
            
            # Calculate target position
            if len(detector_values.shape) == 1:
                target_position = float(target_peak) + offset
            else:
                target_position = float(target_peak[0]) + offset  # Use x-coordinate for 2D
            
            result.move_commands.append({
                "axis": target_axis,
                "position": target_position,
                "relative": False
            })
            
            result.message = f"Found {len(peaks)} peaks, moving {target_axis} to {target_position:.3f}"
            
            # Reset measurement count
            self._measurement_count = 0
        else:
            result.message = f"Found {len(peaks)} peaks, waiting for more measurements ({self._measurement_count}/{measurements_needed})"
        
        return result
    
    def _find_1d_peaks(self, data: np.ndarray) -> List[int]:
        """Find peaks in 1D data using simple threshold method."""
        threshold = self.config.get("peak_threshold", 0.5)
        min_distance = self.config.get("min_peak_distance", 10)
        
        # Normalize data
        if np.max(data) > 0:
            normalized = data / np.max(data)
        else:
            normalized = data
        
        # Find peaks above threshold
        peaks = []
        for i in range(1, len(normalized) - 1):
            if (normalized[i] > threshold and 
                normalized[i] > normalized[i-1] and 
                normalized[i] > normalized[i+1]):
                
                # Check minimum distance
                if not peaks or i - peaks[-1] >= min_distance:
                    peaks.append(i)
        
        return peaks
    
    def _find_2d_peaks(self, data: np.ndarray) -> List[tuple]:
        """Find peaks in 2D data."""
        threshold = self.config.get("peak_threshold", 0.5)
        min_distance = self.config.get("min_peak_distance", 10)
        
        # Normalize data
        if np.max(data) > 0:
            normalized = data / np.max(data)
        else:
            normalized = data
        
        # Find local maxima
        peaks = []
        rows, cols = data.shape
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                if (normalized[i, j] > threshold and
                    normalized[i, j] > normalized[i-1, j] and
                    normalized[i, j] > normalized[i+1, j] and
                    normalized[i, j] > normalized[i, j-1] and
                    normalized[i, j] > normalized[i, j+1]):
                    
                    # Check minimum distance from existing peaks
                    valid = True
                    for peak in peaks:
                        dist = np.sqrt((i - peak[0])**2 + (j - peak[1])**2)
                        if dist < min_distance:
                            valid = False
                            break
                    
                    if valid:
                        peaks.append((i, j))
        
        return peaks