"""
Clickable Decoder Plugin - decodes multi-axis scan data with interactive tooltips.

This plugin:
1. Decodes multi-axis scan data using configurable formula
2. Displays the decoded data in an interactive image
3. Shows detailed tooltips on mouse click with location, channels, and decoded value
"""

import numpy as np
from typing import Dict, Any, List
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Import MultiImageDisplay
try:
    from gui.multi_image_display import MultiImageDisplay
except ImportError:
    try:
        from ..gui.multi_image_display import MultiImageDisplay
    except ImportError:
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        try:
            from gui.multi_image_display import MultiImageDisplay
        except ImportError:
            MultiImageDisplay = None

# Import base plugin classes from the main plugins package
try:
    from plugins.base_plugin import BasePlugin, PluginData, PluginResult, DecoderPlugin
except ImportError:
    try:
        from ..plugins.base_plugin import BasePlugin, PluginData, PluginResult, DecoderPlugin
    except ImportError:
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        try:
            from plugins.base_plugin import BasePlugin, PluginData, PluginResult, DecoderPlugin
        except ImportError:
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
            
            class DecoderPlugin(BasePlugin):
                def process_image(self, image: np.ndarray) -> PluginResult:
                    result = PluginResult()
                    result.message = "Image processing not implemented"
                    return result


class ClickableDecoderPlugin(DecoderPlugin):
    """Plugin that decodes multi-axis scan data with interactive clickable tooltips."""
    
    def get_name(self) -> str:
        return "Clickable Decoder"
    
    def get_description(self) -> str:
        return "Decodes multi-axis scan data and displays with interactive tooltips showing location, channels, and decoded values on click"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def __init__(self):
        super().__init__()
        self.author = "Microscope Controller"
        self.description = self.get_description()
        
        # Configuration
        self.config = {
            "enabled": False,
            "decoder_formula": "mean",  # Options: mean, max, min, diff, custom, dual_output
            "custom_formula": "A + B",  # For custom formula
            "channel_names": [],  # Empty means use all available channels
            "decoder_factor": 1.0,
            "display_window_title": "Clickable Decoder Display",
            "auto_show_display": True,
            "colormap": "plasma",
            "tooltip_channels": "all",  # all, selected, or specific channel names
            "dual_output_mode": True,  # Enable dual output mode: detector1*10 and detector2*100
        }
        
        # Internal state
        self._scan_data = []
        self._decoded_data = None
        self._decoded_outputs = {}  # Dictionary to store multiple outputs
        self._scan_dimensions = None
        self._position_history = []
        self._reference_detector_shape = None
        self._channel_data = {}  # Store per-channel data for tooltips
        self._x_positions = []  # X position values
        self._y_positions = []  # Y position values
        
        # Multi-image display
        self._multi_display = None  # MultiImageDisplay instance
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize plugin with configuration."""
        if config:
            self.config.update(config)
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> tuple:
        """Validate plugin configuration."""
        valid_formulas = ["mean", "max", "min", "diff", "custom"]
        if "decoder_formula" in config:
            if config["decoder_formula"] not in valid_formulas:
                return False, f"decoder_formula must be one of {valid_formulas}"
        return True, ""
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for UI generation."""
        return {
            "enabled": {
                "type": "bool",
                "default": False,
                "description": "Enable/disable plugin"
            },
            "decoder_formula": {
                "type": "str",
                "default": "mean",
                "description": "Formula to use: mean, max, min, diff, custom"
            },
            "custom_formula": {
                "type": "str",
                "default": "A + B",
                "description": "Custom formula (use A, B, C for channels)"
            },
            "channel_names": {
                "type": "list",
                "default": [],
                "description": "Channel names to use (empty = all available)"
            },
            "decoder_factor": {
                "type": "float",
                "min": -1000.0,
                "max": 1000.0,
                "default": 1.0,
                "description": "Factor to multiply decoded result by"
            },
            "display_window_title": {
                "type": "str",
                "default": "Clickable Decoder Display",
                "description": "Title for the display window"
            },
            "auto_show_display": {
                "type": "bool",
                "default": True,
                "description": "Automatically show display window when data is processed"
            },
            "colormap": {
                "type": "str",
                "default": "plasma",
                "description": "Colormap for display"
            },
            "tooltip_channels": {
                "type": "str",
                "default": "all",
                "description": "Channels to show in tooltip: all, selected, or specific names"
            },
            "dual_output_mode": {
                "type": "bool",
                "default": False,
                "description": "Enable dual output mode for testing (detector1*10 + detector2*-1)"
            }
        }
    
    def on_experiment_start(self, experiment_config: Dict[str, Any]) -> None:
        """Called when an experiment starts."""
        self._scan_data = []
        self._decoded_data = None
        self._decoded_outputs = {}  # Reset all outputs
        self._position_history = []
        self._scan_dimensions = None
        self._channel_data = {}
        self._x_positions = []
        self._y_positions = []
        if self._multi_display:
            self._multi_display.cleanup()
            self._multi_display = None
        print("[ClickableDecoder] Experiment started - ready to collect scan data")
    
    def on_experiment_end(self, experiment_config: Dict[str, Any]) -> None:
        """Called when an experiment ends."""
        print(f"[ClickableDecoder] Experiment ended - collected {len(self._scan_data)} data points")
        
        if len(self._scan_data) > 0:
            self._process_scan_data()
            
            if self.config.get("auto_show_display", True) and self._decoded_data is not None and self.enabled:
                self._show_display_window()
    
    def process_data(self, data: PluginData) -> PluginResult:
        """Process measurement data and accumulate scan information."""
        result = PluginResult()
        
        # Store reference detector image shape if available
        if data.camera_image is not None and self._reference_detector_shape is None:
            try:
                self._reference_detector_shape = np.asarray(data.camera_image).shape
                print(f"[ClickableDecoder] Stored reference detector image shape: {self._reference_detector_shape}")
            except Exception:
                pass
        
        # Store scan data with position information
        if data.detector_data:
            for detector_id, values in data.detector_data.items():
                if len(values) > 0:
                    self._scan_data.append({
                        "detector_id": detector_id,
                        "value": float(values[-1]),
                        "positions": data.positions.copy(),
                        "measurement_index": data.measurement_index,
                        "timestamp": data.timestamps[-1] if len(data.timestamps) > 0 else None
                    })
            
            result.message = f"Collected scan data point {len(self._scan_data)}"
            result.extracted_features = {
                "data_points_collected": len(self._scan_data),
                "current_position": data.positions
            }
        else:
            result.message = "No detector data available"
        
        return result
    
    def _process_scan_data(self):
        """Process accumulated scan data to create multi-dimensional decoded data."""
        if not self._scan_data:
            print("[ClickableDecoder] No scan data to process")
            return
        
        print(f"[ClickableDecoder] Processing {len(self._scan_data)} scan data points")
        
        # Get unique positions for each axis to determine grid dimensions
        x_positions = sorted(set([d["positions"].get("x", 0) for d in self._scan_data]))
        y_positions = sorted(set([d["positions"].get("y", 0) for d in self._scan_data]))
        
        # Determine scan dimensions
        dim_x = len(x_positions)
        dim_y = len(y_positions)
        
        # Match detector image dimensions if available
        if self._reference_detector_shape is not None:
            try:
                ref_height, ref_width = self._reference_detector_shape[:2]
                if dim_x != ref_width or dim_y != ref_height:
                    from scipy.ndimage import zoom
                    zoom_y = ref_height / dim_y
                    zoom_x = ref_width / dim_x
                    target_dim_x, target_dim_y = ref_width, ref_height
                else:
                    target_dim_x, target_dim_y = dim_x, dim_y
            except Exception as e:
                print(f"[ClickableDecoder] Could not match detector dimensions: {e}")
                target_dim_x, target_dim_y = dim_x, dim_y
        else:
            target_dim_x, target_dim_y = dim_x, dim_y
        
        self._scan_dimensions = {
            "x_positions": x_positions,
            "y_positions": y_positions,
            "dim_x": target_dim_x,
            "dim_y": target_dim_y,
            "original_dim_x": dim_x,
            "original_dim_y": dim_y
        }
        
        print(f"[ClickableDecoder] Final display dimensions: {target_dim_x}x{target_dim_y}")
        
        # Organize data by detector and position
        # Create data structures for each detector
        detector_data = {}
        for data_point in self._scan_data:
            det_id = data_point["detector_id"]
            if det_id not in detector_data:
                detector_data[det_id] = {}
            
            x_pos = data_point["positions"].get("x", 0)
            y_pos = data_point["positions"].get("y", 0)
            
            # Find indices
            try:
                x_idx = x_positions.index(x_pos)
                y_idx = y_positions.index(y_pos)
            except ValueError:
                continue
            
            key = (x_idx, y_idx)
            detector_data[det_id][key] = data_point["value"]
        
        # Apply decoder formula
        formula = self.config.get("decoder_formula", "mean")
        factor = self.config.get("decoder_factor", 1.0)
        
        # Create decoded data array
        decoded_array = np.zeros((dim_y, dim_x))
        
        # Also store channel data for tooltips
        self._channel_data = {}
        
        for det_id in detector_data.keys():
            channel_array = np.zeros((dim_y, dim_x))
            for (x_idx, y_idx), value in detector_data[det_id].items():
                channel_array[y_idx, x_idx] = value
            self._channel_data[det_id] = channel_array
        
        # Apply formula at each position
        for x_idx in range(dim_x):
            for y_idx in range(dim_y):
                key = (x_idx, y_idx)
                
                # Collect values from all channels at this position
                values = []
                for det_id in detector_data.keys():
                    if key in detector_data[det_id]:
                        values.append(detector_data[det_id][key])
                
                if values:
                    # Apply decoder formula
                    if formula == "mean":
                        decoded_value = np.mean(values)
                    elif formula == "max":
                        decoded_value = np.max(values)
                    elif formula == "min":
                        decoded_value = np.min(values)
                    elif formula == "diff":
                        if len(values) >= 2:
                            decoded_value = values[0] - values[1]
                        else:
                            decoded_value = values[0] if values else 0
                    elif formula == "custom":
                        try:
                            # Simple custom formula evaluation
                            import re
                            custom = self.config.get("custom_formula", "A + B")
                            # Replace A, B, C with actual values
                            for i, val in enumerate(values[:3]):  # Support up to 3 channels
                                custom = custom.replace(chr(65 + i), str(val))
                            decoded_value = eval(custom)
                        except Exception:
                            decoded_value = np.mean(values)
                    else:
                        decoded_value = np.mean(values)
                    
                    decoded_array[y_idx, x_idx] = decoded_value * factor
        
        # Interpolate to match target dimensions if needed
        if (dim_x != target_dim_x or dim_y != target_dim_y) and dim_x > 0 and dim_y > 0:
            try:
                from scipy.ndimage import zoom
                zoom_y = target_dim_y / dim_y
                zoom_x = target_dim_x / dim_x
                decoded_array = zoom(decoded_array, (zoom_y, zoom_x), order=1)
                
                # Also interpolate channel data
                for det_id in self._channel_data:
                    self._channel_data[det_id] = zoom(self._channel_data[det_id], (zoom_y, zoom_x), order=1)
                    
                print(f"[ClickableDecoder] Interpolated decoded data to shape: {decoded_array.shape}")
            except ImportError:
                print("[ClickableDecoder] scipy not available - using original dimensions")
            except Exception as e:
                print(f"[ClickableDecoder] Interpolation failed: {e}")
        
        self._decoded_data = decoded_array
        print(f"[ClickableDecoder] Decoded data shape: {decoded_array.shape}")
        print(f"[ClickableDecoder] Decoded data range: {np.nanmin(decoded_array):.3f} to {np.nanmax(decoded_array):.3f}")
        
        # Store outputs in dictionary
        self._decoded_outputs = {"default": decoded_array}
        
        # Generate additional outputs for dual mode (detector1 * 10, detector2 * 100)
        if self.config.get("dual_output_mode", False):
            detector_ids = list(detector_data.keys())
            if len(detector_ids) >= 2:
                # Create first output array (detector1 * 10)
                decoded_array_1 = np.zeros((dim_y, dim_x))
                
                # Create second output array (detector2 * 100)
                decoded_array_2 = np.zeros((dim_y, dim_x))
                
                for x_idx in range(dim_x):
                    for y_idx in range(dim_y):
                        key = (x_idx, y_idx)
                        
                        # Get first detector value * 10
                        val1 = detector_data.get(detector_ids[0], {}).get(key, 0) * 10
                        # Get second detector value * 100
                        val2 = detector_data.get(detector_ids[1], {}).get(key, 0) * 100
                        
                        decoded_array_1[y_idx, x_idx] = val1
                        decoded_array_2[y_idx, x_idx] = val2
                
                # Interpolate first output if needed
                if (dim_x != target_dim_x or dim_y != target_dim_y) and dim_x > 0 and dim_y > 0:
                    try:
                        from scipy.ndimage import zoom
                        zoom_y = target_dim_y / dim_y
                        zoom_x = target_dim_x / dim_x
                        decoded_array_1 = zoom(decoded_array_1, (zoom_y, zoom_x), order=1)
                        decoded_array_2 = zoom(decoded_array_2, (zoom_y, zoom_x), order=1)
                        print(f"[ClickableDecoder] Interpolated outputs to shape: {decoded_array_1.shape}")
                    except Exception as e:
                        print(f"[ClickableDecoder] Output interpolation failed: {e}")
                
                self._decoded_outputs["detector1_x10"] = decoded_array_1
                self._decoded_outputs["detector2_x100"] = decoded_array_2
                
                print(f"[ClickableDecoder] Output 1 (detector1 * 10) shape: {decoded_array_1.shape}, range: {np.nanmin(decoded_array_1):.3f} to {np.nanmax(decoded_array_1):.3f}")
                print(f"[ClickableDecoder] Output 2 (detector2 * 100) shape: {decoded_array_2.shape}, range: {np.nanmin(decoded_array_2):.3f} to {np.nanmax(decoded_array_2):.3f}")
            else:
                print("[ClickableDecoder] Not enough detectors for dual output mode")
    
    def _show_display_window(self):
        """Show popup window with clickable decoded data using MultiImageDisplay."""
        if not self._decoded_outputs:
            print("[ClickableDecoder] No decoded data to display")
            return
        
        if MultiImageDisplay is None:
            print("[ClickableDecoder] MultiImageDisplay not available, falling back to old display")
            self._show_display_window_old()
            return
        
        try:
            # Create MultiImageDisplay instance if not exists
            if self._multi_display is None:
                self._multi_display = MultiImageDisplay()
            
            # Set images
            for name, data in self._decoded_outputs.items():
                self._multi_display.set_image(name, data)
            
            # Set channel data for overlay
            for name, data in self._channel_data.items():
                self._multi_display.set_channel_data(name, data)
            
            # Set coordinate ranges
            if self._scan_dimensions and self._x_positions and self._y_positions:
                x_min = min(self._x_positions)
                x_max = max(self._x_positions)
                y_min = min(self._y_positions)
                y_max = max(self._y_positions)
                self._multi_display.set_coordinate_ranges(x_min, x_max, y_min, y_max)
            
            # Show display
            self._multi_display.show_display()
            
            print("[ClickableDecoder] Display window shown using MultiImageDisplay")
            
        except Exception as e:
            print(f"[ClickableDecoder] Error showing display with MultiImageDisplay: {e}")
            import traceback
            traceback.print_exc()
            # Fall back to old display
            self._show_display_window_old()
    
    def _show_display_window_old(self):
        """Fallback to old display window implementation."""
        # This is the old implementation - kept as fallback
        # In a real implementation, you would keep the old code here
        print("[ClickableDecoder] Using old display implementation (not implemented in this refactor)")
        return
    
    # ============================================================================
    # OLD METHODS - These are no longer used with MultiImageDisplay
    # Kept for reference or potential fallback
    # ============================================================================
    
    def _create_plot_widgets(self):
        """Create plot widgets dynamically based on available outputs (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _create_overlay_plot(self):
        """Create overlay plot widget (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _on_image_clicked(self, event, output_name="default"):
        """Handle mouse click on the image (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _show_context_menu(self, pos, output_name="default"):
        """Show context menu for an image (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _update_controls_for_focused_output(self):
        """Update control panel to show settings of the focused output (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _on_colormap_changed(self, colormap_name):
        """Handle colormap change from combo box (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _on_min_changed(self, value):
        """Handle min level change (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _on_max_changed(self, value):
        """Handle max level change (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _on_scale_changed(self, mode):
        """Handle scale mode change (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _on_gamma_changed(self, gamma):
        """Handle gamma change (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _on_overlay_toggled(self, enabled):
        """Handle overlay toggle (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _set_scale_mode(self, name, mode):
        """Set scale mode for an image (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _set_gamma(self, name, gamma):
        """Set gamma for an image (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _set_colormap(self, name, colormap):
        """Set colormap for an image (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _reset_coordinate_limits(self):
        """Reset coordinate limits to defaults (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _reset_intensity_levels(self):
        """Reset intensity levels to data range (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _update_colormap(self, colormap_name, output_name=None, save_setting=True):
        """Update the colormap for specified output (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _update_manual_levels(self):
        """Update manual intensity levels (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _update_scale_mode(self, mode):
        """Update scale mode (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _update_gamma(self, gamma):
        """Update gamma correction (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _toggle_overlay(self, enabled):
        """Toggle overlay mode (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _on_dual_mode_toggled(self, checked):
        """Handle dual mode toggle (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _update_all_outputs(self):
        """Update all output displays (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _update_channel_overlay(self):
        """Update the channel overlay (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _populate_colormap_combo(self):
        """Populate colormap combo box (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _on_colormap_index_changed(self, index):
        """Handle colormap combo box index change (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _test_colormap(self, name):
        """Test a colormap by trying to apply it (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _apply_simple_rgb_gradient_to_item(self, image_item, rgb: tuple[int, int, int]) -> bool:
        """Apply simple RGB gradient to a specific image item (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _reset_view(self):
        """Reset the view to show the entire image (OLD)."""
        # This method is no longer used - MultiImageDisplay handles this
        pass
    
    def _create_plot_widgets(self):
        """Create plot widgets dynamically based on available outputs."""
        # Determine number of outputs
        num_outputs = len(self._decoded_outputs) if self._decoded_outputs else 1
        if num_outputs == 0:
            num_outputs = 1  # At least one output
        
        # Calculate grid dimensions (try to make it as square as possible)
        cols = int(np.ceil(np.sqrt(num_outputs)))
        rows = int(np.ceil(num_outputs / cols))
        
        # Get output names
        if self._decoded_outputs:
            output_names = list(self._decoded_outputs.keys())
        else:
            output_names = ["default"]
        
        # Create overlay plot widget if overlay mode is enabled and it doesn't exist
        if self._overlay_mode and self._overlay_plot_widget is None:
            self._create_overlay_plot()
        
        # Clear existing widgets from grid
        while self._plots_grid.count():
            item = self._plots_grid.takeAt(0)
            if item.widget():
                # Don't delete the overlay plot widget if we're in overlay mode
                if self._overlay_mode and item.widget() == self._overlay_plot_widget:
                    continue
                item.widget().deleteLater()
        
        # Clear plot widgets and image items
        self._plot_widgets.clear()
        self._image_items.clear()
        
        # If overlay mode is disabled, clean up overlay widget
        if not self._overlay_mode and self._overlay_plot_widget is not None:
            self._overlay_plot_widget.deleteLater()
            self._overlay_plot_widget = None
            self._overlay_image_item = None
        
        # Create plot widgets for each output
        for i, output_name in enumerate(output_names):
            row = i // cols
            col = i % cols
            
            # Create plot widget
            plot_widget = self._pg.PlotWidget()
            plot_widget.setAspectLocked(True)
            plot_widget.setTitle(output_name.replace('_', ' ').title(), color='w', size='10pt')
            plot_widget.setLabel('left', 'Y Position', units='units')
            plot_widget.setLabel('bottom', 'X Position', units='units')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.setMinimumSize(300, 250)
            
            # Create image item
            image_item = self._pg.ImageItem()
            plot_widget.addItem(image_item)
            
            # Enable clicking on the image
            plot_widget.scene().sigMouseClicked.connect(lambda event, name=output_name: self._on_image_clicked(event, name))
            
            # Add context menu
            plot_widget.setContextMenuPolicy(self._QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            plot_widget.customContextMenuRequested.connect(lambda pos, name=output_name: self._show_context_menu(pos, name))
            
            # Store widgets
            self._plot_widgets[output_name] = plot_widget
            self._image_items[output_name] = image_item
            
            # Add to grid
            self._plots_grid.addWidget(plot_widget, row, col)
        
        # Add overlay plot to grid if overlay mode is enabled
        if self._overlay_mode and self._overlay_plot_widget is not None:
            self._plots_grid.addWidget(self._overlay_plot_widget, rows, 0, 1, cols)
            self._overlay_plot_widget.setVisible(True)
        
        # Restore data to the newly created image items
        self._update_all_outputs()
        
        # Re-apply coordinate ranges and levels if available
        if hasattr(self, '_coord_mapping') and self._coord_mapping:
            mapping = self._coord_mapping
            for plot_widget in self._plot_widgets.values():
                plot_widget.setXRange(mapping['x_max'], mapping['x_min'])  # X reversed
                plot_widget.setYRange(mapping['y_min'], mapping['y_max'])  # Y normal
                plot_widget.plotItem.vb.enableAutoRange(enable=False)
        
        # Re-apply per-output settings (colormap, levels, etc.)
        for output_name, settings in self._output_settings.items():
            if output_name in self._image_items:
                # Re-apply colormap (don't save setting since we're just restoring)
                cmap = settings.get("colormap", "plasma")
                self._update_colormap(cmap, output_name, save_setting=False)
                
                # Re-apply levels
                min_level = settings.get("min", 0.0)
                max_level = settings.get("max", 1.0)
                if output_name in self._image_items:
                    self._image_items[output_name].setLevels([min_level, max_level])
        
        print(f"[ClickableDecoder] Created {num_outputs} plot widgets in {rows}x{cols} grid")
    
    def _create_overlay_plot(self):
        """Create a separate plot widget for the channel overlay."""
        if self._overlay_plot_widget is not None:
            return
        
        # Calculate grid dimensions for outputs
        num_outputs = len(self._plot_widgets) if self._plot_widgets else 1
        cols = int(np.ceil(np.sqrt(num_outputs)))
        
        # Create overlay plot widget
        self._overlay_plot_widget = self._pg.PlotWidget()
        self._overlay_plot_widget.setAspectLocked(True)
        self._overlay_plot_widget.setTitle('Channel Overlay (False Color)', color='w', size='10pt')
        self._overlay_plot_widget.setLabel('left', 'Y Position', units='units')
        self._overlay_plot_widget.setLabel('bottom', 'X Position', units='units')
        self._overlay_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._overlay_plot_widget.setMinimumSize(300, 250)
        
        # Create overlay image item
        self._overlay_image_item = self._pg.ImageItem()
        self._overlay_plot_widget.addItem(self._overlay_image_item)
        
        # Enable clicking on the overlay
        self._overlay_plot_widget.scene().sigMouseClicked.connect(self._on_overlay_clicked)
        
        print("[ClickableDecoder] Created overlay plot widget")
    
    def _on_overlay_clicked(self, event):
        """Handle click on overlay plot."""
        print("[ClickableDecoder] Overlay plot clicked")
    
    def _update_controls_for_focused_output(self):
        """Update control panel to show settings of the focused output."""
        if self._focused_output is None or self._focused_output not in self._output_settings:
            return
        
        settings = self._output_settings[self._focused_output]
        
        # Block signals to prevent feedback loops
        self._cmap_combo.blockSignals(True)
        self._min_spin.blockSignals(True)
        self._max_spin.blockSignals(True)
        self._scale_combo.blockSignals(True)
        self._gamma_spin.blockSignals(True)
        self._overlay_cb.blockSignals(True)
        
        try:
            self._cmap_combo.setCurrentText(settings.get("colormap", "plasma"))
            self._min_spin.setValue(settings.get("min", 0.0))
            self._max_spin.setValue(settings.get("max", 1.0))
            self._scale_combo.setCurrentText(settings.get("scale", "Linear"))
            self._gamma_spin.setValue(settings.get("gamma", 1.0))
            self._overlay_cb.setChecked(settings.get("overlay", False))
        finally:
            self._cmap_combo.blockSignals(False)
            self._min_spin.blockSignals(False)
            self._max_spin.blockSignals(False)
            self._scale_combo.blockSignals(False)
            self._gamma_spin.blockSignals(False)
            self._overlay_cb.blockSignals(False)
        
        # Update overlay visibility based on settings
        if settings.get("overlay", False):
            self._update_channel_overlay()
        else:
            if self._overlay_plot_widget:
                self._overlay_plot_widget.setVisible(False)
    
    def _on_image_clicked(self, event, output_name="default"):
        """Handle mouse click on the image to show detailed tooltip and focus the plot."""
        # Focus on this plot
        self._focused_output = output_name
        print(f"[ClickableDecoder] Focused on output: {output_name}")
        
        # Update controls to show settings of focused output
        self._update_controls_for_focused_output()
        
        # Highlight the focused plot
        for name, plot_widget in self._plot_widgets.items():
            if name == output_name:
                plot_widget.setStyleSheet("border: 2px solid yellow;")
            else:
                plot_widget.setStyleSheet("")
        
        try:
            # Get the plot widget that was clicked
            plot_widget = self._plot_widgets.get(output_name)
            if plot_widget is None:
                return
            
            # Get click position in plot coordinates (position values)
            mouse_point = plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            x, y = mouse_point.x(), mouse_point.y()
            
            # Get the data for this output
            if output_name in self._decoded_outputs:
                data = self._decoded_outputs[output_name]
            else:
                data = self._decoded_data
            
            # Convert position coordinates to array indices
            height, width = data.shape
            
            if hasattr(self, '_coord_mapping') and self._coord_mapping:
                mapping = self._coord_mapping
                # Normalize position to [0,1] range
                # X axis is reversed, flip X for array indexing
                x_norm = 1.0 - (x - mapping['x_min']) / (mapping['x_max'] - mapping['x_min']) if mapping['x_max'] != mapping['x_min'] else 0.5
                
                # Y axis is normal
                y_norm = (y - mapping['y_min']) / (mapping['y_max'] - mapping['y_min']) if mapping['y_max'] != mapping['y_min'] else 0.5
                
                # Convert to array indices
                x_idx = int(round(x_norm * (mapping['width'] - 1)))
                y_idx = int(round(y_norm * (mapping['height'] - 1)))
            else:
                # Fallback to direct coordinates if no mapping (X axis reversed)
                x_idx = int(round(width - 1 - x))  # Flip X for array indexing
                y_idx = int(round(y))
            
            # Clamp indices to valid range
            x_idx = max(0, min(width - 1, x_idx))
            y_idx = max(0, min(height - 1, y_idx))
            
            # Get decoded value
            decoded_value = data[y_idx, x_idx]
            
            # Get actual position (use the clicked position directly since plot uses position values)
            x_pos = x
            y_pos = y
            
            # Get channel values at this position
            channel_values = {}
            if self._channel_data:
                for det_id, channel_array in self._channel_data.items():
                    if 0 <= y_idx < channel_array.shape[0] and 0 <= x_idx < channel_array.shape[1]:
                        channel_values[det_id] = channel_array[y_idx, x_idx]
            
            # Build tooltip text
            tooltip_lines = [
                f"<b>Output:</b> {output_name}",
                f"<b>Location:</b> X={x_pos:.2f}, Y={y_pos:.2f}",
                f"<b>Decoded Value:</b> {decoded_value:.6f}",
                "<b>Channel Values:</b>"
            ]
            
            for det_id, value in channel_values.items():
                tooltip_lines.append(f"  {det_id}: {value:.6f}")
            
            tooltip_text = "<br>".join(tooltip_lines)
            
            # Show tooltip at mouse position
            global_pos = plot_widget.mapToGlobal(event.pos().toPoint())
            self._QtWidgets.QToolTip.showText(global_pos, tooltip_text, plot_widget)
            
            # Update status bar
            if hasattr(self, '_status_bar') and self._status_bar:
                status_text = f"Output: {output_name} | Clicked at X={x_pos:.2f}, Y={y_pos:.2f} | Decoded: {decoded_value:.6f} | Channels: {len(channel_values)}"
                self._status_bar.showMessage(status_text)
            
        except Exception as e:
            print(f"[ClickableDecoder] Error handling click: {e}")
    
    def _show_context_menu(self, pos, output_name):
        """Show context menu for plot."""
        context_menu = self._QtWidgets.QMenu()
        
        # Scale mode submenu
        scale_menu = context_menu.addMenu("Scale Mode")
        linear_action = scale_menu.addAction("Linear")
        log_action = scale_menu.addAction("Log")
        
        current_scale = self._scale_combo.currentText()
        if current_scale == "Linear":
            linear_action.setCheckable(True)
            linear_action.setChecked(True)
        else:
            log_action.setCheckable(True)
            log_action.setChecked(True)
        
        linear_action.triggered.connect(lambda: self._scale_combo.setCurrentText("Linear"))
        log_action.triggered.connect(lambda: self._scale_combo.setCurrentText("Log"))
        
        # Gamma submenu
        gamma_menu = context_menu.addMenu("Gamma")
        gamma_1_0 = gamma_menu.addAction("1.0")
        gamma_1_5 = gamma_menu.addAction("1.5")
        gamma_2_0 = gamma_menu.addAction("2.0")
        gamma_0_5 = gamma_menu.addAction("0.5")
        
        current_gamma = self._gamma_spin.value()
        if abs(current_gamma - 1.0) < 0.01:
            gamma_1_0.setCheckable(True)
            gamma_1_0.setChecked(True)
        elif abs(current_gamma - 1.5) < 0.01:
            gamma_1_5.setCheckable(True)
            gamma_1_5.setChecked(True)
        elif abs(current_gamma - 2.0) < 0.01:
            gamma_2_0.setCheckable(True)
            gamma_2_0.setChecked(True)
        elif abs(current_gamma - 0.5) < 0.01:
            gamma_0_5.setCheckable(True)
            gamma_0_5.setChecked(True)
        
        gamma_1_0.triggered.connect(lambda: self._gamma_spin.setValue(1.0))
        gamma_1_5.triggered.connect(lambda: self._gamma_spin.setValue(1.5))
        gamma_2_0.triggered.connect(lambda: self._gamma_spin.setValue(2.0))
        gamma_0_5.triggered.connect(lambda: self._gamma_spin.setValue(0.5))
        
        # Show menu
        plot_widget = self._plot_widgets.get(output_name)
        if plot_widget:
            context_menu.exec(plot_widget.mapToGlobal(pos))
    
    def _populate_colormap_combo(self):
        """Populate colormap combo box with only working colormaps."""
        working_colormaps = []
        
        # Test colormaps from both APIs and only include ones that work
        test_colormaps = [
            'viridis', 'plasma', 'inferno', 'magma', 'cividis', 
            'grey', 'turbo', 'thermal', 'flame', 'hot', 'cool'
        ]
        
        for cmap_name in test_colormaps:
            if self._test_colormap(cmap_name):
                working_colormaps.append(cmap_name)
        
        # Always add simple RGB colormaps (we'll handle them specially)
        simple_rgb = ['red', 'green', 'blue']
        for cmap_name in simple_rgb:
            if cmap_name not in working_colormaps:
                working_colormaps.append(cmap_name)
        
        if working_colormaps:
            self._cmap_combo.clear()
            for cmap in sorted(working_colormaps):
                self._cmap_combo.addItem(cmap)
            print(f"[ClickableDecoder] Populated {len(working_colormaps)} working colormaps: {sorted(working_colormaps)}")
        else:
            # Ultimate fallback - just use viridis
            self._cmap_combo.clear()
            self._cmap_combo.addItem('viridis')
            print("[ClickableDecoder] No working colormaps found, using viridis only")
    
    def _test_colormap(self, colormap_name):
        """Test if a colormap can be applied successfully."""
        if self._pg is None:
            return False
        
        pg = self._pg
        
        # Test with pg.colormap.get (newer API)
        try:
            if hasattr(pg, 'colormap') and hasattr(pg.colormap, 'get'):
                cmap = pg.colormap.get(colormap_name)
                if cmap is not None:
                    return True
        except Exception:
            pass
        
        # Test with Gradients (older API)
        try:
            from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
            if colormap_name in Gradients:
                return True
        except Exception:
            pass
        
        return False
    
    def _on_colormap_index_changed(self, index):
        """Handle colormap change from combo box index."""
        if index >= 0:
            colormap_name = self._cmap_combo.currentText()
            print(f"[ClickableDecoder] Colormap index changed to {index}, name: {colormap_name}")
            self._on_colormap_changed(colormap_name)
    
    def _on_dual_mode_toggled(self, checked):
        """Handle dual mode toggle."""
        self.config["dual_output_mode"] = checked
        
        if checked:
            # Regenerate data with dual output
            print("[ClickableDecoder] Dual mode enabled, regenerating data")
            self.config["dual_output_mode"] = True
            self._process_scan_data()
            # Update display with new data
            self._update_display_with_dual_data()
        else:
            print("[ClickableDecoder] Dual mode disabled")
            # Hide second plot
            self._plot_widget_2.setVisible(False)
            self._image_item_2.setVisible(False)
    
    def _on_dual_mode_toggled(self, checked):
        """Handle dual mode toggle - generates detector1*10 and detector2*100 outputs."""
        self.config["dual_output_mode"] = checked
        
        if checked:
            # Need to regenerate data with dual output
            print("[ClickableDecoder] Dual mode enabled, regenerating data")
            self.config["dual_output_mode"] = True
            self._process_scan_data()
            # Show display with new outputs
            if self._decoded_outputs and self.enabled:
                self._show_display_window()
        else:
            print("[ClickableDecoder] Dual mode disabled, regenerating data")
            self.config["dual_output_mode"] = False
            self._process_scan_data()
            # Show display with new outputs
            if self._decoded_outputs and self.enabled:
                self._show_display_window()
    
    def _update_all_outputs(self):
        """Update all output displays."""
        for output_name, data in self._decoded_outputs.items():
            if output_name in self._image_items:
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
                
                # Ensure levels are valid numbers
                if not np.isfinite(data_min) or not np.isfinite(data_max):
                    data_min = 0.0
                    data_max = 1.0
                elif data_min == data_max:
                    data_min = data_min - 0.5
                    data_max = data_max + 0.5
                
                # Flip X axis to match reversed orientation
                flipped_data = np.flip(data, axis=1)
                self._image_items[output_name].setImage(flipped_data, levels=[data_min, data_max], autoRange=False)
                print(f"[ClickableDecoder] Updated output '{output_name}': range {data_min:.3f} to {data_max:.3f}")
        
        # Apply colormap to all image items (respecting per-output settings, don't save)
        for output_name in self._decoded_outputs.keys():
            if output_name in self._output_settings:
                cmap = self._output_settings[output_name].get("colormap", "plasma")
                self._update_colormap(cmap, output_name, save_setting=False)
            else:
                self._update_colormap(self.config.get("colormap", "plasma"), output_name, save_setting=False)

    def _on_colormap_changed(self, colormap_name):
        """Handle colormap change from combo box."""
        # Update config
        self.config["colormap"] = colormap_name
        print(f"[ClickableDecoder] Config updated with colormap: {colormap_name}")
        # Apply the colormap to focused output (save setting)
        self._update_colormap(colormap_name, save_setting=True)

    def _update_colormap(self, colormap_name, output_name=None, save_setting=True):
        """Update the colormap for specified output or focused output."""
        # If no output specified, use focused output
        if output_name is None:
            output_name = self._focused_output
        
        if output_name is None:
            return
        
        # Save settings for the output (only if requested)
        if save_setting:
            if output_name not in self._output_settings:
                self._output_settings[output_name] = {}
            self._output_settings[output_name]["colormap"] = colormap_name
        
        print(f"[ClickableDecoder] Updating colormap for output '{output_name}' to: {colormap_name}")
        
        if self._pg is None:
            print("[ClickableDecoder] pyqtgraph not available - cannot update colormap")
            return
        
        pg = self._pg
        
        # Handle simple RGB colormaps specially
        simple_rgb = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255)}
        if colormap_name.lower() in simple_rgb:
            if output_name in self._image_items:
                self._apply_simple_rgb_gradient_to_item(self._image_items[output_name], simple_rgb[colormap_name.lower()])
            return
        
        # Try multiple methods to get the colormap
        success = False
        lut = None
        
        # Method 1: pg.colormap.get (newer pyqtgraph versions)
        try:
            if hasattr(pg, 'colormap') and hasattr(pg.colormap, 'get'):
                cmap = pg.colormap.get(colormap_name)
                if cmap is not None:
                    lut = cmap.getLookupTable(0.0, 1.0, 256)
                    if output_name in self._image_items:
                        self._image_items[output_name].setLookupTable(lut)
                        self._image_items[output_name].updateImage()
                    print(f"[ClickableDecoder] Colormap updated successfully via pg.colormap.get")
                    success = True
                else:
                    print(f"[ClickableDecoder] pg.colormap.get returned None for {colormap_name}")
        except Exception as e:
            print(f"[ClickableDecoder] Error using pg.colormap.get: {e}")
        
        # Method 2: pg.colormap module (alternative newer API)
        if not success:
            try:
                if hasattr(pg, 'colormap'):
                    cmap = pg.colormap(colormap_name)
                    if cmap is not None:
                        lut = cmap.getLookupTable(0.0, 1.0, 256)
                        if output_name in self._image_items:
                            self._image_items[output_name].setLookupTable(lut)
                            self._image_items[output_name].updateImage()
                        print(f"[ClickableDecoder] Colormap updated successfully via pg.colormap module")
                        success = True
            except Exception as e:
                print(f"[ClickableDecoder] Error using pg.colormap module: {e}")
        
        # Method 3: Gradients from GradientEditorItem (older API)
        if not success:
            try:
                from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
                if colormap_name in Gradients:
                    gradient_data = Gradients[colormap_name]
                    # Handle different gradient data formats
                    if isinstance(gradient_data, dict):
                        # Newer format with 'ticks' key
                        if 'ticks' in gradient_data:
                            positions = [t[0] for t in gradient_data['ticks']]
                            colors = [pg.mkColor(t[1]) for t in gradient_data['ticks']]
                        else:
                            # Old format
                            positions = []
                            colors = []
                            for key, value in gradient_data.items():
                                try:
                                    positions.append(float(key))
                                    colors.append(pg.mkColor(value))
                                except:
                                    pass
                    else:
                        # List format [(position, color), ...]
                        positions = [p[0] for p in gradient_data]
                        colors = [pg.mkColor(p[1]) for p in gradient_data]
                    
                    if positions and colors:
                        cmap = pg.ColorMap(positions, colors)
                        lut = cmap.getLookupTable()
                        if output_name in self._image_items:
                            self._image_items[output_name].setLookupTable(lut)
                            self._image_items[output_name].updateImage()
                        print(f"[ClickableDecoder] Colormap updated successfully via Gradients")
                        success = True
                    else:
                        print(f"[ClickableDecoder] Could not parse gradient data for {colormap_name}")
                else:
                    print(f"[ClickableDecoder] Colormap {colormap_name} not found in Gradients")
                    print(f"[ClickableDecoder] Available colormaps: {list(Gradients.keys())}")
            except Exception as e2:
                print(f"[ClickableDecoder] Error using Gradients: {e2}")
        
        if not success:
            print(f"[ClickableDecoder] Failed to apply colormap: {colormap_name}")
    
    def _apply_simple_rgb_gradient_to_item(self, image_item, rgb: tuple[int, int, int]) -> bool:
        """Apply simple RGB gradient to a specific image item."""
        if self._pg is None:
            return False
        
        pg = self._pg
        r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

        try:
            cmap = pg.ColorMap(
                [0.0, 1.0],
                [self._QtGui.QColor(0, 0, 0), self._QtGui.QColor(r, g, b)],
            )
            try:
                image_item.setColorMap(cmap)
                image_item.updateImage()
                return True
            except Exception:
                lut = cmap.getLookupTable(0.0, 1.0, 256)
                if image_item is not None and hasattr(image_item, "setLookupTable"):
                    image_item.setLookupTable(lut)
                    image_item.updateImage()
                    return True
        except Exception:
            return False

    def _update_manual_levels(self):
        """Update image levels from manual spin boxes for focused output."""
        if self._focused_output is None:
            return
        
        min_val = self._min_spin.value()
        max_val = self._max_spin.value()
        
        # Save settings for focused output
        if self._focused_output not in self._output_settings:
            self._output_settings[self._focused_output] = {}
        self._output_settings[self._focused_output]["min"] = min_val
        self._output_settings[self._focused_output]["max"] = max_val
        
        # Check if auto mode (value is 0)
        if min_val == 0.0 and max_val == 0.0:
            # Auto mode - recalculate from data for focused output
            if self._focused_output in self._decoded_outputs:
                data = self._decoded_outputs[self._focused_output]
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
                
                if not np.isfinite(data_min) or not np.isfinite(data_max):
                    data_min = 0.0
                    data_max = 1.0
                elif data_min == data_max:
                    data_min = data_min - 0.5
                    data_max = data_max + 0.5
                
                if self._focused_output in self._image_items:
                    self._image_items[self._focused_output].setLevels([data_min, data_max])
        else:
            # Manual mode - use spinbox values for focused output
            if self._focused_output in self._image_items:
                self._image_items[self._focused_output].setLevels([min_val, max_val])
        
        # Force redraw after levels change
        try:
            if self._focused_output in self._image_items:
                self._image_items[self._focused_output].updateImage()
        except Exception:
            pass
    
    def _update_scale_mode(self, mode):
        """Update scale mode (linear/log) for focused output."""
        if self._focused_output is None:
            return
        
        # Save settings for focused output
        if self._focused_output not in self._output_settings:
            self._output_settings[self._focused_output] = {}
        self._output_settings[self._focused_output]["scale"] = mode
        
        # If overlay is active, regenerate it with new scale
        if self._overlay_mode:
            self._update_channel_overlay()
            return
        
        try:
            if self._focused_output in self._decoded_outputs:
                data = self._decoded_outputs[self._focused_output]
                
                if mode == "Log":
                    data_min = np.nanmin(data)
                    if data_min <= 0:
                        offset = abs(data_min) + 1e-10
                        log_data = np.log10(data + offset)
                    else:
                        log_data = np.log10(data)
                    
                    if self._focused_output in self._image_items:
                        # Flip X axis to match reversed orientation
                        flipped_log_data = np.flip(log_data, axis=1)
                        self._image_items[self._focused_output].setImage(flipped_log_data, autoRange=False)
                        log_min = np.nanmin(log_data)
                        log_max = np.nanmax(log_data)
                        self._image_items[self._focused_output].setLevels([log_min, log_max])
                else:
                    if self._focused_output in self._image_items:
                        # Flip X axis to match reversed orientation
                        flipped_data = np.flip(data, axis=1)
                        self._image_items[self._focused_output].setImage(flipped_data, autoRange=False)
                        data_min = np.nanmin(data)
                        data_max = np.nanmax(data)
                        self._image_items[self._focused_output].setLevels([data_min, data_max])
                
                # Re-apply colormap after scale mode change
                colormap_name = self._output_settings.get(self._focused_output, {}).get("colormap", "plasma")
                self._update_colormap(colormap_name)
                
        except Exception as e:
            print(f"[ClickableDecoder] Error updating scale mode: {e}")
    
    def _toggle_overlay(self, enabled):
        """Toggle overlay mode - shows separate plot with false color channel composite."""
        if self._focused_output is None:
            return
        
        # Save settings for focused output
        if self._focused_output not in self._output_settings:
            self._output_settings[self._focused_output] = {}
        self._output_settings[self._focused_output]["overlay"] = enabled
        
        self._overlay_mode = enabled
        
        # Recreate plot widgets (this will handle adding/removing overlay appropriately)
        self._create_plot_widgets()
        
        if enabled and self._overlay_plot_widget:
            # Generate false color composite for overlay
            self._update_channel_overlay()
        
        # Force redraw
        try:
            if self._focused_output in self._image_items:
                self._image_items[self._focused_output].updateImage()
            if self._overlay_image_item:
                self._overlay_image_item.updateImage()
        except Exception:
            pass
    
    def _update_channel_overlay(self):
        """Update the channel overlay with false color composite of detector channels."""
        if not self._overlay_mode or self._overlay_image_item is None:
            return
        
        if not self._channel_data:
            print("[ClickableDecoder] No channel data available for overlay")
            return
        
        try:
            # Get detector IDs
            detector_ids = list(self._channel_data.keys())
            if not detector_ids:
                return
            
            # Get reference shape from first channel
            first_channel = self._channel_data[detector_ids[0]]
            height, width = first_channel.shape
            
            # Create RGB image for the composite
            rgb_image = np.zeros((height, width, 3), dtype=np.float32)
            
            # Assign colors to channels: R, G, B, and cycle if more than 3
            colors = [
                (1.0, 0.0, 0.0),  # Red
                (0.0, 1.0, 0.0),  # Green
                (0.0, 0.0, 1.0),  # Blue
                (1.0, 1.0, 0.0),  # Yellow
                (1.0, 0.0, 1.0),  # Magenta
                (0.0, 1.0, 1.0),  # Cyan
            ]
            
            # Get gamma value from focused output
            gamma = self._output_settings.get(self._focused_output, {}).get("gamma", 1.0) if self._focused_output else 1.0
            if gamma <= 0:
                gamma = 0.1
            
            # Get scale mode from focused output
            scale_mode = self._output_settings.get(self._focused_output, {}).get("scale", "Linear") if self._focused_output else "Linear"
            
            # Normalize each channel and apply color
            for i, det_id in enumerate(detector_ids):
                channel_data = self._channel_data[det_id]
                if channel_data.shape != (height, width):
                    continue
                
                # Apply log scale if needed
                if scale_mode == "Log":
                    channel_min = np.nanmin(channel_data)
                    if channel_min <= 0:
                        offset = abs(channel_min) + 1e-10
                        channel_data = np.log10(channel_data + offset)
                    else:
                        channel_data = np.log10(channel_data)
                
                # Normalize channel to [0, 1]
                channel_min = np.nanmin(channel_data)
                channel_max = np.nanmax(channel_data)
                if channel_max > channel_min:
                    normalized = (channel_data - channel_min) / (channel_max - channel_min)
                else:
                    normalized = np.zeros_like(channel_data)
                
                # Apply gamma correction
                normalized = np.power(normalized, gamma)
                
                # Get color for this channel
                color = colors[i % len(colors)]
                
                # Add to RGB image
                for c in range(3):
                    rgb_image[:, :, c] += normalized * color[c]
            
            # Clip to [0, 1]
            rgb_image = np.clip(rgb_image, 0, 1)
            
            # Convert to uint8 for display
            rgb_image = (rgb_image * 255).astype(np.uint8)
            
            # Flip X axis to match reversed orientation
            rgb_image = np.flip(rgb_image, axis=1)
            
            # Set overlay image (RGB format for pyqtgraph: height, width, 3)
            self._overlay_image_item.setImage(rgb_image, autoRange=False)
            self._overlay_image_item.setVisible(True)
            
            # Set coordinate ranges to match main plots
            if self._plot_widgets:
                first_plot = list(self._plot_widgets.values())[0]
                x_range = first_plot.plotItem.vb.viewRange()[0]
                y_range = first_plot.plotItem.vb.viewRange()[1]
                if x_range[0] != x_range[1] and y_range[0] != y_range[1]:
                    self._overlay_plot_widget.setXRange(x_range[0], x_range[1])
                    self._overlay_plot_widget.setYRange(y_range[0], y_range[1])
                    self._overlay_plot_widget.plotItem.vb.enableAutoRange(enable=False)
            
            print(f"[ClickableDecoder] Updated channel overlay with {len(detector_ids)} channels, gamma={gamma}, scale={scale_mode}")
            
        except Exception as e:
            print(f"[ClickableDecoder] Error updating channel overlay: {e}")
            import traceback
            traceback.print_exc()
    
    def _reset_view(self):
        """Reset the view to show the entire image."""
        for plot_widget in self._plot_widgets.values():
            plot_widget.setAspectLocked(True)
            plot_widget.autoRange()
    
    def _update_gamma(self, gamma):
        """Update gamma correction for focused output."""
        self._gamma = gamma
        print(f"[ClickableDecoder] Updating gamma to: {gamma}")
        
        if self._focused_output is None:
            return
        
        # Save settings for focused output
        if self._focused_output not in self._output_settings:
            self._output_settings[self._focused_output] = {}
        self._output_settings[self._focused_output]["gamma"] = gamma
        
        # If overlay is active, regenerate it with new gamma
        if self._overlay_mode:
            self._update_channel_overlay()
            return
        
        try:
            if gamma <= 0:
                gamma = 0.1
            
            if self._focused_output in self._decoded_outputs:
                data = self._decoded_outputs[self._focused_output]
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
                
                if data_max > data_min:
                    normalized = (data - data_min) / (data_max - data_min)
                    gamma_corrected = np.power(normalized, gamma)
                    corrected_data = gamma_corrected * (data_max - data_min) + data_min
                else:
                    corrected_data = data
                
                if self._focused_output in self._image_items:
                    # Flip X axis to match reversed orientation
                    flipped_corrected_data = np.flip(corrected_data, axis=1)
                    self._image_items[self._focused_output].setImage(flipped_corrected_data, autoRange=False)
            
            # Re-apply colormap after gamma change
            colormap_name = self._output_settings.get(self._focused_output, {}).get("colormap", "plasma")
            self._update_colormap(colormap_name)
            
        except Exception as e:
            print(f"[ClickableDecoder] Error updating gamma: {e}")
    
    def manual_execute_with_data(self, detector_data: Dict[str, np.ndarray], 
                                 position_history: List[Dict[str, float]],
                                 scan_dimensions: Dict[str, Any] = None) -> bool:
        """Manually execute the plugin with provided data."""
        try:
            print("[ClickableDecoder] Manual execution started")
            
            # Reset state
            self._scan_data = []
            self._decoded_data = None
            self._position_history = []
            self._scan_dimensions = None
            self._channel_data = {}
            
            # Set position history
            self._position_history = position_history if position_history else []
            
            # Set scan dimensions if provided
            if scan_dimensions:
                self._scan_dimensions = scan_dimensions
            
            # Convert detector data to scan data format
            for det_id, data in detector_data.items():
                if not isinstance(data, np.ndarray):
                    data = np.array(data)
                    detector_data[det_id] = data
            
            # Find matching positions
            pos_length = min(len(self._position_history), len(next(iter(detector_data.values()))))
            
            for i in range(pos_length):
                for det_id, data in detector_data.items():
                    if i < len(data):
                        data_point = {
                            "detector_id": det_id,
                            "value": float(data[i]),
                            "positions": self._position_history[i] if i < len(self._position_history) else {},
                            "measurement_index": i,
                            "timestamp": i
                        }
                        self._scan_data.append(data_point)
            
            print(f"[ClickableDecoder] Created {len(self._scan_data)} data points from manual execution")
            
            # Process the scan data
            self._process_scan_data()
            
            # Show display if enabled
            if self.config.get("auto_show_display", True) and self._decoded_outputs and self.enabled:
                self._show_display_window()
            
            print("[ClickableDecoder] Manual execution completed successfully")
            return True
            
        except Exception as e:
            print(f"[ClickableDecoder] Manual execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_decoded_data(self) -> np.ndarray:
        """Get the decoded multi-dimensional data."""
        # Return the default output if available, otherwise None
        return self._decoded_outputs.get("default") if self._decoded_outputs else None
    
    def get_channel_data(self) -> Dict[str, np.ndarray]:
        """Get the channel data for tooltips."""
        return self._channel_data
    
    def get_decoded_value_at_position(self, x: float, y: float) -> tuple:
        """Get the decoded value at a specific position.
        
        Args:
            x: X coordinate in scan position units
            y: Y coordinate in scan position units
            
        Returns:
            Tuple of (value, success) where success is True if value was found
        """
        data = self._decoded_outputs.get("default")
        if data is None:
            return 0.0, False
        
        try:
            # Convert scan position to array indices
            if self._scan_dimensions and self._x_positions and self._y_positions:
                # Find closest positions
                x_idx = min(range(len(self._x_positions)), 
                          key=lambda i: abs(self._x_positions[i] - x))
                y_idx = min(range(len(self._y_positions)), 
                          key=lambda i: abs(self._y_positions[i] - y))
                
                # Interpolate if we have the dimensions
                height, width = data.shape
                if self._scan_dimensions.get('dim_x') != width or self._scan_dimensions.get('dim_y') != height:
                    # Data was interpolated, need to map to interpolated coordinates
                    x_min = min(self._x_positions)
                    x_max = max(self._x_positions)
                    y_min = min(self._y_positions)
                    y_max = max(self._y_positions)
                    
                    x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else 0
                    y_norm = (y - y_min) / (y_max - y_min) if y_max != y_min else 0
                    
                    # X axis is reversed, flip X for array indexing
                    x_idx = int(round((1.0 - x_norm) * (width - 1)))
                    y_idx = int(round(y_norm * (height - 1)))
                
                # Clamp to valid range
                x_idx = max(0, min(width - 1, x_idx))
                y_idx = max(0, min(height - 1, y_idx))
                
                # X axis is reversed in display, flip back for data access
                x_idx = width - 1 - x_idx
                
                value = data[y_idx, x_idx]
                return value, True
            else:
                # No scan dimensions, try direct mapping
                height, width = data.shape
                # X axis is reversed, flip X for array indexing
                x_idx = int(round(width - 1 - x))
                y_idx = int(round(y))
                
                if 0 <= x_idx < width and 0 <= y_idx < height:
                    value = data[y_idx, x_idx]
                    return value, True
                else:
                    return 0.0, False
                    
        except Exception as e:
            print(f"[ClickableDecoder] Error getting decoded value at position: {e}")
            return 0.0, False
    
    def cleanup(self) -> None:
        """Clean up resources when plugin is unloaded."""
        if self._multi_display:
            self._multi_display.cleanup()
            self._multi_display = None
        self._scan_data = []
        self._decoded_data = None
        self._decoded_outputs.clear()
        self._position_history = []
        self._channel_data = {}
        self._x_positions = []
        self._y_positions = []