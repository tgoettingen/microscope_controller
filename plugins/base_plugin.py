"""
Base plugin interface for microscope controller plugins.

Plugins can:
1. Process measurement data (images, time series, etc.)
2. Analyze data and extract information
3. Trigger stage movement based on analysis results
4. Provide visualization of results
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class PluginData:
    """Data container for plugin input."""
    # Measurement data
    detector_data: Dict[str, np.ndarray] = field(default_factory=dict)  # detector_id -> data array
    positions: Dict[str, float] = field(default_factory=dict)  # axis_name -> position
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Metadata
    measurement_index: int = 0
    experiment_id: str = ""
    detector_ids: List[str] = field(default_factory=list)
    
    # Optional image data
    camera_image: Optional[np.ndarray] = None


@dataclass
class PluginResult:
    """Result container for plugin output."""
    success: bool = True
    message: str = ""
    
    # Analysis results
    extracted_features: Dict[str, Any] = field(default_factory=dict)
    processed_data: Optional[np.ndarray] = None
    
    # Movement commands
    move_commands: List[Dict[str, Any]] = field(default_factory=list)  # Each dict: {'axis': str, 'position': float, 'relative': bool}
    
    # Visualization data
    visualization_data: Optional[Dict[str, Any]] = None
    
    # Configuration for next measurement
    next_measurement_config: Dict[str, Any] = field(default_factory=dict)


class BasePlugin(ABC):
    """Base class for all microscope controller plugins."""
    
    def __init__(self):
        self.name = self.__class__.__name__
        self.version = "1.0.0"
        self.description = ""
        self.author = ""
        self.enabled = True
        
        # Plugin configuration
        self.config: Dict[str, Any] = {}
        
        # Internal state
        self._measurement_count = 0
        self._state: Dict[str, Any] = {}
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the plugin name."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of what this plugin does."""
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Return the plugin version."""
        pass
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the plugin with configuration.
        
        Args:
            config: Plugin configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        if config:
            self.config.update(config)
        return True
    
    def process_data(self, data: PluginData) -> PluginResult:
        """Process measurement data and return results.
        
        This is the main method that plugins implement to analyze data.
        
        Args:
            data: Input measurement data
            
        Returns:
            PluginResult containing analysis results and any movement commands
        """
        result = PluginResult()
        result.message = "Process data not implemented"
        return result
    
    def on_measurement_start(self, data: PluginData) -> None:
        """Called when a new measurement sequence starts."""
        self._measurement_count = 0
        self._state.clear()
    
    def on_measurement_end(self, data: PluginData, result: PluginResult) -> None:
        """Called when a measurement sequence ends."""
        self._measurement_count += 1
    
    def on_experiment_start(self, experiment_config: Dict[str, Any]) -> None:
        """Called when an experiment starts."""
        pass
    
    def on_experiment_end(self, experiment_config: Dict[str, Any]) -> None:
        """Called when an experiment ends."""
        pass
    
    def should_trigger_movement(self, data: PluginData, result: PluginResult) -> bool:
        """Determine if the plugin should trigger stage movement.
        
        Args:
            data: Input measurement data
            result: Plugin processing result
            
        Returns:
            True if movement should be triggered, False otherwise
        """
        return len(result.move_commands) > 0
    
    def get_movement_commands(self, data: PluginData, result: PluginResult) -> List[Dict[str, Any]]:
        """Get movement commands based on analysis results.
        
        Args:
            data: Input measurement data
            result: Plugin processing result
            
        Returns:
            List of movement command dictionaries
        """
        return result.move_commands
    
    def get_required_detectors(self) -> List[str]:
        """Return list of detector IDs this plugin requires."""
        return []
    
    def get_required_axes(self) -> List[str]:
        """Return list of axis names this plugin can control."""
        return []
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate plugin configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        return True, ""
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for UI generation.
        
        Returns:
            Dictionary describing configuration parameters
        """
        return {}
    
    def cleanup(self) -> None:
        """Clean up resources when plugin is unloaded."""
        pass


class DecoderPlugin(BasePlugin):
    """Base class for decoder plugins that extract information from images/data."""
    
    def process_image(self, image: np.ndarray) -> PluginResult:
        """Process a single image and extract information.
        
        Args:
            image: Input image array
            
        Returns:
            PluginResult with extracted features
        """
        result = PluginResult()
        result.message = "Image processing not implemented"
        return result


class TimeSeriesPlugin(BasePlugin):
    """Base class for time series analysis plugins."""
    
    def process_time_series(self, data: np.ndarray, timestamps: np.ndarray) -> PluginResult:
        """Process time series data and extract information.
        
        Args:
            data: Time series data array
            timestamps: Corresponding timestamps
            
        Returns:
            PluginResult with extracted features
        """
        result = PluginResult()
        result.message = "Time series processing not implemented"
        return result


class MovementPlugin(BasePlugin):
    """Base class for plugins that primarily control stage movement."""
    
    def calculate_next_position(self, data: PluginData, result: PluginResult) -> Dict[str, float]:
        """Calculate the next stage position based on current analysis.
        
        Args:
            data: Input measurement data
            result: Plugin processing result
            
        Returns:
            Dictionary mapping axis names to target positions
        """
        return {}
