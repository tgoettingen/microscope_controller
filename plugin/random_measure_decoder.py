"""
Random Measure Decoder Plugin - implements a workflow for random positioning and measurement.

This plugin:
1. Moves platform to a random position when strip chart starts
2. Measures for 5 seconds
3. Decodes a position (returns position at half distance from center)
4. Moves to the decoded position
5. Can repeat this operation N times (configurable, default 1)
6. Saves measurement data, decoded positions, and stage readout to file
"""

import numpy as np
import random
import time
from typing import Dict, Any, List
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import csv
import json

# Import base plugin classes from the main plugins package
try:
    # Try absolute import first
    from plugins.base_plugin import BasePlugin, PluginData, PluginResult, MovementPlugin
except ImportError:
    try:
        # Try relative import
        from ..plugins.base_plugin import BasePlugin, PluginData, PluginResult, MovementPlugin
    except ImportError:
        # Try adding parent directory to path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        try:
            from plugins.base_plugin import BasePlugin, PluginData, PluginResult, MovementPlugin
        except ImportError:
            # Fallback: define the classes inline as concrete classes (not abstract)
            @dataclass
            class PluginData:
                detector_data: Dict[str, np.ndarray] = field(default_factory=dict)
                positions: Dict[str, float] = field(default_factory=dict)
                timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
                measurement_index: int = 0
                experiment_id: str = ""
                detector_ids: List[str] = field(default_factory=list)
                camera_image: object = None
            
            @dataclass
            class PluginResult:
                success: bool = True
                message: str = ""
                extracted_features: Dict[str, Any] = field(default_factory=dict)
                processed_data: object = None
                move_commands: List[Dict[str, Any]] = field(default_factory=list)
                visualization_data: object = None
                next_measurement_config: Dict[str, Any] = field(default_factory=dict)
            
            class BasePlugin:
                def __init__(self):
                    self.name = self.__class__.__name__
                    self.version = "1.0.0"
                    self.description = ""
                    self.author = ""
                    self.enabled = True
                    self.config: Dict[str, Any] = {}
                    self._measurement_count = 0
                    self._state: Dict[str, Any] = {}
                
                def get_name(self) -> str:
                    return self.name
                
                def get_description(self) -> str:
                    return self.description
                
                def get_version(self) -> str:
                    return self.version
                
                def initialize(self, config: Dict[str, Any] = None) -> bool:
                    if config:
                        self.config.update(config)
                    return True
                
                def process_data(self, data: PluginData) -> PluginResult:
                    result = PluginResult()
                    result.message = "Process data not implemented"
                    return result
                
                def on_measurement_start(self, data: PluginData) -> None:
                    self._measurement_count = 0
                    self._state.clear()
                
                def on_measurement_end(self, data: PluginData, result: PluginResult) -> None:
                    self._measurement_count += 1
                
                def on_experiment_start(self, experiment_config: Dict[str, Any]) -> None:
                    pass
                
                def on_experiment_end(self, experiment_config: Dict[str, Any]) -> None:
                    pass
                
                def should_trigger_movement(self, data: PluginData, result: PluginResult) -> bool:
                    return len(result.move_commands) > 0
                
                def get_movement_commands(self, data: PluginData, result: PluginResult) -> List[Dict[str, Any]]:
                    return result.move_commands
                
                def get_required_detectors(self) -> List[str]:
                    return []
                
                def get_required_axes(self) -> List[str]:
                    return []
                
                def validate_config(self, config: Dict[str, Any]) -> tuple:
                    return True, ""
                
                def get_config_schema(self) -> Dict[str, Any]:
                    return {}
                
                def cleanup(self) -> None:
                    pass
            
            class MovementPlugin(BasePlugin):
                def calculate_next_position(self, data: PluginData, result: PluginResult) -> Dict[str, float]:
                    return {}


class RandomMeasureDecoderPlugin(MovementPlugin):
    """Plugin that performs random positioning, measurement, and decoded movement."""
    
    def get_name(self) -> str:
        return "Random Measure Decoder"
    
    def get_description(self) -> str:
        return "Moves to random position, measures for 5 seconds, decodes position, moves to decoded position, repeats N times"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def __init__(self):
        super().__init__()
        self.author = "Microscope Controller"
        self.description = self.get_description()
        
        # Configuration
        self.config = {
            "enabled": False,  # Disable plugin by default - only enable when experiment specifies it
            "repeat_count": -1,  # Number of times to repeat the operation (-1 = infinite)
            "measurement_duration": 5.0,  # Measurement duration in seconds
            "post_move_delay": 10.0,  # Delay after moving before starting measurement (seconds)
            "post_decode_delay": 10.0,  # Delay after moving to decoded position before next cycle (seconds)
            "random_range_x": 10.0,  # Random position range for X axis
            "random_range_y": 10.0,  # Random position range for Y axis
            "random_range_z": 5.0,  # Random position range for Z axis
            "center_position": None,  # Center position for decoding (None = auto-calculate from stage range)
            "center_x": None,  # X center position for decoding (None = use center_position or auto-calculate)
            "center_y": None,  # Y center position for decoding (None = use center_position or auto-calculate)
            "stage_x_min": None,  # Stage X minimum (for auto-center calculation)
            "stage_x_max": None,  # Stage X maximum (for auto-center calculation)
            "stage_y_min": None,  # Stage Y minimum (for auto-center calculation)
            "stage_y_max": None,  # Stage Y maximum (for auto-center calculation)
            "decoder_offset_factor": 0.5,  # Factor for decoder position (0.5 = half distance)
            "random_seed": None,  # Random seed for reproducibility
            "debug_level": 2,  # Debug level: 0=none, 1=basic, 2=detailed, 3=verbose
            "save_to_file": True,  # Enable saving measurement data to file
            "output_file": "random_measure_decoder_data.csv",  # Output file name
            "output_directory": None  # Output directory (None = experiment output directory)
        }
        
        # Internal state
        self._current_repeat = 0
        self._current_phase = "idle"  # idle, moving, waiting, measuring, decoding, post_decode_wait
        self._target_position = None
        self._start_time = None
        self._measurement_data = []
        self._center_position = None
        self._center_x = None
        self._center_y = None
        
        # Position history for logging
        self._position_history = []  # List of (cycle, random_pos, decoded_pos, stage_readout) tuples
        self._output_file_path = None  # Path to output file
    
    def _debug(self, level: int, message: str):
        """Print debug message based on debug level.
        
        Args:
            level: Minimum debug level required to print (0=always, 1=basic, 2=detailed, 3=verbose)
            message: Message to print
        """
        debug_level = self.config.get("debug_level", 1)
        if level <= debug_level:
            print(f"[RandomMeasureDecoder] {message}")
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize plugin with configuration."""
        if config:
            self.config.update(config)
        
        # Set random seed if provided
        if self.config.get("random_seed") is not None:
            random.seed(self.config["random_seed"])
            np.random.seed(self.config["random_seed"])
        
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> tuple:
        """Validate plugin configuration."""
        required_keys = ["repeat_count", "measurement_duration"]
        for key in required_keys:
            if key not in config:
                return False, f"Missing required configuration key: {key}"
        
        if config["repeat_count"] < -1:
            return False, "repeat_count must be >= -1 (-1 = infinite)"
        
        if config["measurement_duration"] <= 0:
            return False, "measurement_duration must be > 0"
        
        return True, ""
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for UI generation."""
        return {
            "enabled": {
                "type": "bool",
                "default": False,
                "description": "Enable/disable plugin (default: disabled, enable via experiment config)"
            },
            "repeat_count": {
                "type": "int",
                "min": -1,
                "max": 100,
                "default": -1,
                "description": "Number of times to repeat the measurement cycle (-1 = infinite)"
            },
            "measurement_duration": {
                "type": "float",
                "min": 0.1,
                "max": 60.0,
                "default": 2.0,
                "description": "Measurement duration in seconds"
            },
            "post_move_delay": {
                "type": "float",
                "min": 0.0,
                "max": 60.0,
                "default": 3.0,
                "description": "Delay after moving before starting measurement (seconds)"
            },
            "post_decode_delay": {
                "type": "float",
                "min": 0.0,
                "max": 60.0,
                "default": 3.0,
                "description": "Delay after moving to decoded position before next cycle (seconds)"
            },
            "random_range_x": {
                "type": "float",
                "min": 0.0,
                "max": 1000.0,
                "default": 10.0,
                "description": "Random position range for X axis"
            },
            "random_range_y": {
                "type": "float",
                "min": 0.0,
                "max": 1000.0,
                "default": 10.0,
                "description": "Random position range for Y axis"
            },
            "random_range_z": {
                "type": "float",
                "min": 0.0,
                "max": 1000.0,
                "default": 5.0,
                "description": "Random position range for Z axis"
            },
            "center_position": {
                "type": "float",
                "default": None,
                "description": "Center position for decoding (None = auto-calculate from stage range, used for both X and Y)"
            },
            "center_x": {
                "type": "float",
                "default": None,
                "description": "X center position for decoding (None = use center_position or auto-calculate)"
            },
            "center_y": {
                "type": "float",
                "default": None,
                "description": "Y center position for decoding (None = use center_position or auto-calculate)"
            },
            "stage_x_min": {
                "type": "float",
                "default": None,
                "description": "Stage X minimum (for auto-center calculation)"
            },
            "stage_x_max": {
                "type": "float",
                "default": None,
                "description": "Stage X maximum (for auto-center calculation)"
            },
            "stage_y_min": {
                "type": "float",
                "default": None,
                "description": "Stage Y minimum (for auto-center calculation)"
            },
            "stage_y_max": {
                "type": "float",
                "default": None,
                "description": "Stage Y maximum (for auto-center calculation)"
            },
            "decoder_offset_factor": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "description": "Factor for decoder position (0.5 = half distance from center)"
            },
            "random_seed": {
                "type": "int",
                "min": 0,
                "max": 2**31-1,
                "default": None,
                "description": "Random seed for reproducibility (None = random)"
            },
            "debug_level": {
                "type": "int",
                "min": 0,
                "max": 3,
                "default": 1,
                "description": "Debug level: 0=none, 1=basic, 2=detailed, 3=verbose"
            },
            "save_to_file": {
                "type": "bool",
                "default": True,
                "description": "Save measurement data and positions to CSV file"
            },
            "output_file": {
                "type": "str",
                "default": "random_measure_decoder_data.csv",
                "description": "Output CSV file name"
            },
            "output_directory": {
                "type": "str",
                "default": None,
                "description": "Output directory (None = experiment output directory)"
            }
        }
    
    def get_required_detectors(self) -> List[str]:
        """Return list of detector IDs this plugin requires."""
        return []  # Works with any detector
    
    def get_required_axes(self) -> List[str]:
        """Return list of axis names this plugin can control."""
        return ["x", "y", "z"]
    
    def on_experiment_start(self, experiment_config: Dict[str, Any]) -> None:
        """Called when an experiment starts."""
        self._current_repeat = 0
        self._current_phase = "idle"
        self._measurement_data = []
        self._position_history = []
        
        # Initialize output file if enabled
        if self.config.get("save_to_file", True):
            self._initialize_output_file(experiment_config)
        
        # Calculate center positions from stage range if not manually set
        # X center
        if self.config.get("center_x") is not None:
            self._center_x = self.config.get("center_x")
            self._debug(1, f"Using manually set X center position: {self._center_x:.3f}")
        elif self.config.get("center_position") is not None:
            self._center_x = self.config.get("center_position")
            self._debug(1, f"Using general center position for X: {self._center_x:.3f}")
        else:
            stage_x_min = self.config.get("stage_x_min")
            stage_x_max = self.config.get("stage_x_max")
            if stage_x_min is not None and stage_x_max is not None:
                self._center_x = (stage_x_min + stage_x_max) / 2.0
                self._debug(1, f"Auto-calculated X center position: {self._center_x:.3f} (from stage range {stage_x_min:.3f} to {stage_x_max:.3f})")
            else:
                self._center_x = 0.0
                self._debug(1, f"Using default X center position: {self._center_x:.3f} (stage range not provided)")
        
        # Y center
        if self.config.get("center_y") is not None:
            self._center_y = self.config.get("center_y")
            self._debug(1, f"Using manually set Y center position: {self._center_y:.3f}")
        elif self.config.get("center_position") is not None:
            self._center_y = self.config.get("center_position")
            self._debug(1, f"Using general center position for Y: {self._center_y:.3f}")
        else:
            stage_y_min = self.config.get("stage_y_min")
            stage_y_max = self.config.get("stage_y_max")
            if stage_y_min is not None and stage_y_max is not None:
                self._center_y = (stage_y_min + stage_y_max) / 2.0
                self._debug(1, f"Auto-calculated Y center position: {self._center_y:.3f} (from stage range {stage_y_min:.3f} to {stage_y_max:.3f})")
            else:
                self._center_y = 0.0
                self._debug(1, f"Using default Y center position: {self._center_y:.3f} (stage range not provided)")
        
        # Also set general center_position for backward compatibility
        self._center_position = self._center_x  # Use X center as general center
        
        # Start the first cycle
        self._start_new_cycle()
        self._debug(1, "Starting first cycle")
    
    def on_experiment_end(self, experiment_config: Dict[str, Any]) -> None:
        """Called when an experiment ends."""
        self._debug(1, f"Experiment ended after {self._current_repeat} cycles")
        self._current_phase = "idle"
        self._measurement_data = []
        
        # Close output file if enabled
        if self._output_file_path is not None:
            self._close_output_file()
    
    def _initialize_output_file(self, experiment_config: Dict[str, Any]) -> None:
        """Initialize the output CSV file for logging."""
        try:
            # Determine output directory
            output_dir = self.config.get("output_directory")
            if output_dir is None:
                # Try to get from experiment config
                output_dir = experiment_config.get("output_dir")
            
            if output_dir is None:
                # Use current directory
                output_dir = Path.cwd()
            else:
                output_dir = Path(output_dir)
            
            # Create directory if it doesn't exist
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Set output file path
            output_file = self.config.get("output_file", "random_measure_decoder_data.csv")
            self._output_file_path = output_dir / output_file
            
            # Write CSV header
            with open(self._output_file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    'cycle',
                    'timestamp',
                    'phase',
                    'random_x',
                    'random_y',
                    'random_z',
                    'decoded_x',
                    'decoded_y',
                    'stage_readout_x',
                    'stage_readout_y',
                    'stage_readout_z',
                    'detector_id',
                    'measurement_value',
                    'measurement_timestamp'
                ])
            
            self._debug(1, f"Initialized output file: {self._output_file_path}")
            
        except Exception as e:
            self._debug(0, f"Error initializing output file: {e}")
            self._output_file_path = None
    
    def _close_output_file(self) -> None:
        """Close the output file."""
        try:
            if self._output_file_path is not None:
                self._debug(1, f"Closing output file: {self._output_file_path}")
                self._output_file_path = None
        except Exception as e:
            self._debug(0, f"Error closing output file: {e}")
    
    def _log_to_file(self, cycle: int, phase: str, random_pos: Dict, decoded_pos: Dict, 
                     stage_readout: Dict, measurement_data: List = None) -> None:
        """Log data to output CSV file."""
        if not self.config.get("save_to_file", True) or self._output_file_path is None:
            return
        
        try:
            with open(self._output_file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                timestamp = time.time()
                
                # Log position information
                writer.writerow([
                    cycle,
                    timestamp,
                    phase,
                    random_pos.get('x', ''),
                    random_pos.get('y', ''),
                    random_pos.get('z', ''),
                    decoded_pos.get('x', ''),
                    decoded_pos.get('y', ''),
                    stage_readout.get('x', ''),
                    stage_readout.get('y', ''),
                    stage_readout.get('z', ''),
                    '',  # detector_id placeholder
                    '',  # measurement_value placeholder
                    ''   # measurement_timestamp placeholder
                ])
                
                # Log measurement data if available
                if measurement_data:
                    for meas in measurement_data:
                        writer.writerow([
                            cycle,
                            meas.get('timestamp', timestamp),
                            phase,
                            random_pos.get('x', ''),
                            random_pos.get('y', ''),
                            random_pos.get('z', ''),
                            decoded_pos.get('x', ''),
                            decoded_pos.get('y', ''),
                            stage_readout.get('x', ''),
                            stage_readout.get('y', ''),
                            stage_readout.get('z', ''),
                            meas.get('detector_id', ''),
                            meas.get('value', ''),
                            meas.get('timestamp', timestamp)
                        ])
        except Exception as e:
            self._debug(0, f"Error writing to output file: {e}")
    
    def _extract_stage_readout(self, data: PluginData) -> Dict[str, float]:
        """Extract stage readout position from measurement data.
        
        Returns:
            Dictionary with x, y, z positions from stage
        """
        if data and data.positions:
            return {
                'x': data.positions.get('X', 0.0),
                'y': data.positions.get('Y', 0.0),
                'z': data.positions.get('Z', 0.0)
            }
        return {'x': 0.0, 'y': 0.0, 'z': 0.0}
    
    def process_data(self, data: PluginData) -> PluginResult:
        """Process measurement data based on current phase."""
        result = PluginResult()
        
        if self._current_phase == "idle":
            result.message = "Plugin idle, waiting for start"
            self._debug(2, "Phase: idle")
            return result
        
        elif self._current_phase == "waiting":
            # Wait for stage to settle after movement
            if self._start_time and (time.time() - self._start_time) >= self.config["post_move_delay"]:
                # Move to measuring phase
                self._current_phase = "measuring"
                self._start_time = time.time()
                result.message = f"Delay complete, starting measurement ({self.config['measurement_duration']}s)"
                self._debug(2, f"Phase: waiting -> measuring, duration: {self.config['measurement_duration']}s")
            else:
                elapsed = time.time() - self._start_time if self._start_time else 0
                result.message = f"Waiting for stage to settle... ({elapsed:.1f}s / {self.config['post_move_delay']}s)"
                self._debug(3, f"Phase: waiting, elapsed: {elapsed:.1f}s / {self.config['post_move_delay']}s")
        
        elif self._current_phase == "measuring":
            # Collect measurement data
            if data.detector_data:
                for detector_id, values in data.detector_data.items():
                    if len(values) > 0:
                        self._measurement_data.append({
                            "detector_id": detector_id,
                            "value": float(values[-1]),
                            "timestamp": time.time()
                        })
            
            # Check if measurement duration is complete
            if self._start_time and (time.time() - self._start_time) >= self.config["measurement_duration"]:
                # Move to decoding phase
                self._current_phase = "decoding"
                result.message = f"Measurement complete ({len(self._measurement_data)} samples), decoding position"
                self._debug(2, f"Phase: measuring -> decoding, collected {len(self._measurement_data)} samples")
            else:
                elapsed = time.time() - self._start_time if self._start_time else 0
                result.message = f"Measuring... ({elapsed:.1f}s / {self.config['measurement_duration']}s)"
                self._debug(3, f"Phase: measuring, elapsed: {elapsed:.1f}s, samples: {len(self._measurement_data)}")
        
        elif self._current_phase == "decoding":
            # Check if we need to transition to post_decode_wait (after movement commands were generated)
            if self._state.get("ready_for_post_decode_wait", False):
                self._current_phase = "post_decode_wait"
                self._start_time = time.time()
                self._state["ready_for_post_decode_wait"] = False
                result.message = "Transitioning to post-decode wait"
                self._debug(2, f"Phase: decoding -> post_decode_wait, delay: {self.config['post_decode_delay']}s")
                return result
            
            # Decode position from measurement data
            decoded_position = self._decode_position()
            self._debug(2, f"Phase: decoding, decoded position: X={decoded_position['x']:.3f}, Y={decoded_position['y']:.3f}")
            
            result.message = f"Decoded position: X={decoded_position['x']:.3f}, Y={decoded_position['y']:.3f}, moving to position"
            result.extracted_features = {
                "decoded_position": decoded_position,
                "measurement_count": len(self._measurement_data),
                "cycle_number": self._current_repeat + 1
            }
            
            # Log decoded position and stage readout to file
            if self.config.get("save_to_file", True):
                stage_readout = self._extract_stage_readout(data)
                self._log_to_file(
                    cycle=self._current_repeat,
                    phase="decoded_position",
                    random_pos=self._target_position,
                    decoded_pos=decoded_position,
                    stage_readout=stage_readout,
                    measurement_data=self._measurement_data
                )
            
            # Generate movement commands for both X and Y
            result.move_commands.append({
                "axis": "x",
                "position": decoded_position["x"],
                "relative": False
            })
            result.move_commands.append({
                "axis": "y",
                "position": decoded_position["y"],
                "relative": False
            })
            
            self._debug(1, f"Adding decoded movement commands to result: X={decoded_position['x']:.3f}, Y={decoded_position['y']:.3f}")
            self._debug(1, f"Total movement commands in result: {len(result.move_commands)}")
            
            # Stay in decoding phase for this call
            # Set flag to transition on next call (after movement commands are processed)
            self._state["ready_for_post_decode_wait"] = True
            self._debug(2, "Staying in decoding phase, will transition on next call")
        
        elif self._current_phase == "post_decode_wait":
            # Wait after moving to decoded position before next cycle
            if self._start_time and (time.time() - self._start_time) >= self.config["post_decode_delay"]:
                # Move to completion phase
                self._current_phase = "completing"
                result.message = f"Post-decode wait complete, completing cycle"
                self._debug(2, "Phase: post_decode_wait -> completing")
            else:
                elapsed = time.time() - self._start_time if self._start_time else 0
                result.message = f"Waiting after decode... ({elapsed:.1f}s / {self.config['post_decode_delay']}s)"
                self._debug(3, f"Phase: post_decode_wait, elapsed: {elapsed:.1f}s / {self.config['post_decode_delay']}s")
        
        elif self._current_phase == "completing":
            # Check if we should repeat
            self._current_repeat += 1
            repeat_count = self.config["repeat_count"]
            
            self._debug(2, f"Phase: completing, cycle {self._current_repeat}, repeat_count: {repeat_count}")
            
            # If repeat_count is -1, repeat infinitely
            if repeat_count == -1 or self._current_repeat < repeat_count:
                # Start new cycle
                self._start_new_cycle()
                if repeat_count == -1:
                    result.message = f"Starting cycle {self._current_repeat + 1} (infinite mode)"
                    self._debug(1, f"Starting cycle {self._current_repeat + 1} (infinite mode)")
                else:
                    result.message = f"Starting cycle {self._current_repeat + 1}/{repeat_count}"
                    self._debug(1, f"Starting cycle {self._current_repeat + 1}/{repeat_count}")
            else:
                # All cycles complete
                self._current_phase = "idle"
                result.message = f"All {repeat_count} cycles complete"
                result.extracted_features = {
                    "total_cycles": repeat_count,
                    "all_cycles_complete": True
                }
                self._debug(1, f"All {repeat_count} cycles complete")
        
        return result
    
    def _start_new_cycle(self):
        """Start a new measurement cycle."""
        # Use stage range for random position generation if available
        stage_x_min = self.config.get("stage_x_min")
        stage_x_max = self.config.get("stage_x_max")
        stage_y_min = self.config.get("stage_y_min")
        stage_y_max = self.config.get("stage_y_max")
        
        # If stage range is available, use it; otherwise use configured random ranges
        if stage_x_min is not None and stage_x_max is not None:
            random_range_x = stage_x_max - stage_x_min
            x_center = (stage_x_min + stage_x_max) / 2.0
        else:
            random_range_x = self.config.get("random_range_x", 10.0)
            x_center = 0.0
        
        if stage_y_min is not None and stage_y_max is not None:
            random_range_y = stage_y_max - stage_y_min
            y_center = (stage_y_min + stage_y_max) / 2.0
        else:
            random_range_y = self.config.get("random_range_y", 10.0)
            y_center = 0.0
        

        random_range_z = self.config.get("random_range_z", 5.0)
        
        # Generate random position within the valid range
        self._target_position = {
            "x": x_center + random.uniform(-random_range_x / 2, random_range_x / 2),
            "y": y_center + random.uniform(-random_range_y / 2, random_range_y / 2),
            "z": random.uniform(-random_range_z / 2, random_range_z / 2)
        }
        
        # self._target_position["x"] = 0.0
        # self._target_position["y"] = 0.0
        print(f"-----------------------------------------Target position: {self._target_position}")
        
        # Clamp to stage limits if available
        if stage_x_min is not None and stage_x_max is not None:
            self._target_position["x"] = max(stage_x_min, min(stage_x_max, self._target_position["x"]))
        
        if stage_y_min is not None and stage_y_max is not None:
            self._target_position["y"] = max(stage_y_min, min(stage_y_max, self._target_position["y"]))
        
        self._current_phase = "moving"
        self._measurement_data = []
        
        # Store the target position in state for movement command generation
        self._state["target_position"] = self._target_position
        
        self._debug(1, f"Starting new cycle {self._current_repeat + 1}")
        if stage_x_min is not None and stage_x_max is not None:
            self._debug(2, f"Stage range: X[{stage_x_min:.3f}, {stage_x_max:.3f}] Y[{stage_y_min:.3f}, {stage_y_max:.3f}]")
        else:
            self._debug(2, "Stage range: Not available, using configured ranges")
        self._debug(2, f"Target position: X={self._target_position['x']:.3f}, Y={self._target_position['y']:.3f}, Z={self._target_position['z']:.3f}")
        self._debug(2, "Phase: moving")
        
        # Log random position to file
        if self.config.get("save_to_file", True):
            self._log_to_file(
                cycle=self._current_repeat,
                phase="random_position",
                random_pos=self._target_position,
                decoded_pos={'x': '', 'y': ''},
                stage_readout={'x': '', 'y': '', 'z': ''}
            )
    
    def _decode_position(self) -> Dict[str, float]:
        """Decode position from measurement data.
        
        Returns:
            Decoded position (half distance from center to target) for X and Y
        """
        if not self._measurement_data:
            self._debug(2, "No measurement data, returning center position")
            return {"x": self._center_x, "y": self._center_y}
        
        # Calculate average value from measurements
        values = [d["value"] for d in self._measurement_data]
        average_value = np.mean(values) if values else 0.0
        
        # Get the target position we moved to
        target_x = self._target_position.get("x", 0.0)
        target_y = self._target_position.get("y", 0.0)
        
        # Calculate distance from center for both axes
        # Use signed distance to move towards center
        distance_from_center_x = target_x - self._center_x
        distance_from_center_y = target_y - self._center_y
        
        # Decoder returns position at half the distance from center for both axes
        # This moves the position closer to the center
        decoded_x = self._center_x + (distance_from_center_x * self.config.get("decoder_offset_factor", 0.5))
        decoded_y = self._center_y + (distance_from_center_y * self.config.get("decoder_offset_factor", 0.5))
        
        self._debug(2, f"Decoding: target_x={target_x:.3f}, target_y={target_y:.3f}")
        self._debug(2, f"Center X: {self._center_x:.3f}, Center Y: {self._center_y:.3f}")
        self._debug(2, f"Distance X: {distance_from_center_x:.3f}, Distance Y: {distance_from_center_y:.3f}")
        self._debug(2, f"Decoded X: {decoded_x:.3f}, Decoded Y: {decoded_y:.3f}")
        self._debug(3, f"Average measurement value: {average_value:.3f}")
        
        return {"x": decoded_x, "y": decoded_y}
    
    def should_trigger_movement(self, data: PluginData, result: PluginResult) -> bool:
        """Determine if the plugin should trigger stage movement."""
        # Only trigger movement when we have a target position and are in moving phase
        return (self._current_phase == "moving" and 
                "target_position" in self._state and
                len(result.move_commands) == 0)
    
    def get_movement_commands(self, data: PluginData, result: PluginResult) -> List[Dict[str, Any]]:
        """Get movement commands based on current state."""
        commands = []
        
        if self._current_phase == "moving" and "target_position" in self._state:
            target = self._state["target_position"]
            
            # Add movement commands for all axes
            commands.append({
                "axis": "x",
                "position": target["x"],
                "relative": False
            })
            commands.append({
                "axis": "y", 
                "position": target["y"],
                "relative": False
            })
            commands.append({
                "axis": "z",
                "position": target["z"],
                "relative": False
            })
            
            self._debug(2, f"Generating movement commands: X={target['x']:.3f}, Y={target['y']:.3f}, Z={target['z']:.3f}")
            
            # After moving, start waiting phase
            self._current_phase = "waiting"
            self._start_time = time.time()
            self._debug(2, f"Phase: moving -> waiting, delay: {self.config['post_move_delay']}s")
        
        # Check if we need to move to decoded position
        elif self._state.get("move_to_decoded", False) and "decoded_position" in self._state:
            decoded_position = self._state["decoded_position"]
            print(f"[RandomMeasureDecoder] Generating decoded movement command: X={decoded_position:.3f}")
            
            commands.append({
                "axis": "x",
                "position": decoded_position,
                "relative": False
            })
            
            # Clear the flag
            self._state["move_to_decoded"] = False
            
            # After moving to decoded position, start post-decode wait
            self._current_phase = "post_decode_wait"
            self._start_time = time.time()
            print(f"[RandomMeasureDecoder] Phase: decoded movement -> post_decode_wait, delay: {self.config['post_decode_delay']}s")
        
        return commands
    
    def calculate_next_position(self, data: PluginData, result: PluginResult) -> Dict[str, float]:
        """Calculate the next stage position based on current analysis."""
        # This plugin handles movement through phases, not through this method
        return {}