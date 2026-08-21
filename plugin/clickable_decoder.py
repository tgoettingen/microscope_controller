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
            "decoder_formula": "mean",  # Options: mean, max, min, diff, custom
            "custom_formula": "A + B",  # For custom formula
            "channel_names": [],  # Empty means use all available channels
            "decoder_factor": 1.0,
            "display_window_title": "Clickable Decoder Display",
            "auto_show_display": True,
            "colormap": "plasma",
            "tooltip_channels": "all",  # all, selected, or specific channel names
        }
        
        # Internal state
        self._scan_data = []
        self._decoded_data = None
        self._display_window = None
        self._scan_dimensions = None
        self._position_history = []
        self._reference_detector_shape = None
        self._channel_data = {}  # Store per-channel data for tooltips
        
        # GUI imports
        self._QtCore = None
        self._QPointF = None
    
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
            }
        }
    
    def on_experiment_start(self, experiment_config: Dict[str, Any]) -> None:
        """Called when an experiment starts."""
        self._scan_data = []
        self._decoded_data = None
        self._position_history = []
        self._scan_dimensions = None
        self._channel_data = {}
        print("[ClickableDecoder] Experiment started - ready to collect scan data")
    
    def on_experiment_end(self, experiment_config: Dict[str, Any]) -> None:
        """Called when an experiment ends."""
        print(f"[ClickableDecoder] Experiment ended - collected {len(self._scan_data)} data points")
        
        if len(self._scan_data) > 0:
            self._process_scan_data()
            
            if self.config.get("auto_show_display", True) and self._decoded_data is not None:
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
    
    def _show_display_window(self):
        """Show popup window with clickable decoded data."""
        if self._decoded_data is None:
            print("[ClickableDecoder] No decoded data to display")
            return
        
        try:
            from PyQt6 import QtWidgets, QtCore, QtGui
            from PyQt6.QtCore import QRectF, QPointF
            from PyQt6.QtWidgets import QToolTip
            import pyqtgraph as pg
            
            # Create window if it doesn't exist
            if self._display_window is None or not self._display_window.isVisible():
                self._display_window = QtWidgets.QMainWindow()
                self._display_window.setWindowTitle(self.config.get("display_window_title", "Clickable Decoder Display"))
                self._display_window.resize(1000, 800)
                
                # Create central widget
                central_widget = QtWidgets.QWidget()
                self._display_window.setCentralWidget(central_widget)
                layout = QtWidgets.QVBoxLayout(central_widget)
                
                # Create plot widget
                self._plot_widget = pg.PlotWidget()
                self._plot_widget.setAspectLocked(True)
                self._plot_widget.setTitle('Clickable Decoder Display', color='w', size='12pt')
                self._plot_widget.setLabel('left', 'Y Position', units='units')
                self._plot_widget.setLabel('bottom', 'X Position', units='units')
                self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
                
                # Create image item
                self._image_item = pg.ImageItem()
                self._plot_widget.addItem(self._image_item)
                
                # Enable clicking on the image
                self._plot_widget.scene().sigMouseClicked.connect(self._on_image_clicked)
                
                layout.addWidget(self._plot_widget)
                
                # Add control panel
                control_panel = QtWidgets.QWidget()
                control_layout = QtWidgets.QHBoxLayout(control_panel)
                
                # Colormap selector
                control_layout.addWidget(QtWidgets.QLabel("Colormap:"))
                self._cmap_combo = QtWidgets.QComboBox()
                self._populate_colormap_combo()
                self._cmap_combo.setCurrentText(self.config.get("colormap", "plasma"))
                self._cmap_combo.currentTextChanged.connect(self._update_colormap)
                control_layout.addWidget(self._cmap_combo)
                
                # Auto-range checkbox
                self._auto_range_cb = QtWidgets.QCheckBox("Auto Range")
                self._auto_range_cb.setChecked(True)
                self._auto_range_cb.toggled.connect(self._update_levels)
                control_layout.addWidget(self._auto_range_cb)
                
                # Reset view button
                reset_btn = QtWidgets.QPushButton("Reset View")
                reset_btn.clicked.connect(self._reset_view)
                control_layout.addWidget(reset_btn)
                
                control_layout.addStretch()
                layout.addWidget(control_panel)
                
                # Add info label
                self._info_label = QtWidgets.QLabel()
                self._info_label.setStyleSheet("font-size: 10pt; padding: 4px;")
                layout.addWidget(self._info_label)
                
                # Add status bar
                self._status_bar = self._display_window.statusBar()
                self._status_bar.showMessage("Click on image to see detailed information")
                
                # Store imports for later use
                self._QtCore = QtCore
                self._QPointF = QPointF
                
                # Store coordinate mapping
                self._x_positions = None
                self._y_positions = None
                if self._scan_dimensions:
                    self._x_positions = self._scan_dimensions.get('x_positions', [])
                    self._y_positions = self._scan_dimensions.get('y_positions', [])
            
            # Set the decoded data
            data_min = np.nanmin(self._decoded_data)
            data_max = np.nanmax(self._decoded_data)
            
            self._image_item.setImage(self._decoded_data, levels=[data_min, data_max])
            
            # Apply colormap
            self._update_colormap(self.config.get("colormap", "plasma"))
            
            # Set up coordinate system
            if self._scan_dimensions and self._x_positions and self._y_positions:
                x_min = min(self._x_positions)
                x_max = max(self._x_positions)
                y_min = min(self._y_positions)
                y_max = max(self._y_positions)
                
                self._plot_widget.setXRange(x_min, x_max)
                self._plot_widget.setYRange(y_min, y_max)
                
                self._coord_mapping = {
                    'x_min': x_min, 'x_max': x_max,
                    'y_min': y_min, 'y_max': y_max,
                    'width': self._decoded_data.shape[1],
                    'height': self._decoded_data.shape[0]
                }
            else:
                height, width = self._decoded_data.shape
                self._plot_widget.setXRange(0, width)
                self._plot_widget.setYRange(0, height)
                self._coord_mapping = None
            
            # Update info label
            if self._scan_dimensions:
                dim_info = f"Scan dimensions: {self._scan_dimensions['dim_x']}x{self._scan_dimensions['dim_y']}"
                range_info = f"Data range: {data_min:.3f} to {data_max:.3f}"
                formula_info = f"Formula: {self.config.get('decoder_formula', 'mean')}"
                channel_info = f"Channels: {len(self._channel_data)}"
                self._info_label.setText(f"{dim_info} | {range_info} | {formula_info} | {channel_info}")
            
            # Show window
            self._display_window.show()
            self._display_window.raise_()
            self._display_window.activateWindow()
            
            print("[ClickableDecoder] Interactive display window shown with click functionality")
            
        except ImportError:
            print("[ClickableDecoder] PyQt6 or pyqtgraph not available - cannot display window")
        except Exception as e:
            print(f"[ClickableDecoder] Error showing display window: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_image_clicked(self, event):
        """Handle mouse click on the image to show detailed tooltip."""
        try:
            # Get click position in plot coordinates
            mouse_point = self._plot_widget.plotItem.vb.mapSceneToView(event.scenePos())
            x, y = mouse_point.x(), mouse_point.y()
            
            # Convert to array indices
            height, width = self._decoded_data.shape
            
            if hasattr(self, '_coord_mapping') and self._coord_mapping:
                mapping = self._coord_mapping
                x_norm = (x - mapping['x_min']) / (mapping['x_max'] - mapping['x_min']) if mapping['x_max'] != mapping['x_min'] else 0
                y_norm = (y - mapping['y_min']) / (mapping['y_max'] - mapping['y_min']) if mapping['y_max'] != mapping['y_min'] else 0
                
                x_idx = int(round(x_norm * (mapping['width'] - 1)))
                y_idx = int(round(y_norm * (mapping['height'] - 1)))
            else:
                x_idx = int(round(x))
                y_idx = int(round(y))
            
            # Clamp indices to valid range
            x_idx = max(0, min(width - 1, x_idx))
            y_idx = max(0, min(height - 1, y_idx))
            
            # Get decoded value
            decoded_value = self._decoded_data[y_idx, x_idx]
            
            # Get actual position
            x_pos = x_idx
            y_pos = y_idx
            if self._x_positions and 0 <= x_idx < len(self._x_positions):
                x_pos = self._x_positions[x_idx]
            if self._y_positions and 0 <= y_idx < len(self._y_positions):
                y_pos = self._y_positions[y_idx]
            
            # Get channel values at this position
            channel_values = {}
            if self._channel_data:
                for det_id, channel_array in self._channel_data.items():
                    if 0 <= y_idx < channel_array.shape[0] and 0 <= x_idx < channel_array.shape[1]:
                        channel_values[det_id] = channel_array[y_idx, x_idx]
            
            # Build tooltip text
            tooltip_lines = [
                f"<b>Location:</b> X={x_pos:.2f}, Y={y_pos:.2f}",
                f"<b>Decoded Value:</b> {decoded_value:.6f}",
                "<b>Channel Values:</b>"
            ]
            
            for det_id, value in channel_values.items():
                tooltip_lines.append(f"  {det_id}: {value:.6f}")
            
            tooltip_text = "<br>".join(tooltip_lines)
            
            # Show tooltip at mouse position
            global_pos = self._plot_widget.mapToGlobal(event.pos().toPoint())
            QToolTip.showText(global_pos, tooltip_text, self._plot_widget)
            
            # Update status bar
            if hasattr(self, '_status_bar') and self._status_bar:
                status_text = f"Clicked at X={x_pos:.2f}, Y={y_pos:.2f} | Decoded: {decoded_value:.6f} | Channels: {len(channel_values)}"
                self._status_bar.showMessage(status_text)
            
        except Exception as e:
            print(f"[ClickableDecoder] Error handling click: {e}")
    
    def _populate_colormap_combo(self):
        """Populate colormap combo box."""
        try:
            from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
            colormaps = sorted(list(Gradients.keys()))
            self._cmap_combo.clear()
            for cmap in colormaps:
                self._cmap_combo.addItem(cmap)
        except Exception:
            basic_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'grey', 'hot', 'cool']
            self._cmap_combo.clear()
            for cmap in basic_cmaps:
                self._cmap_combo.addItem(cmap)
    
    def _update_colormap(self, colormap_name):
        """Update the colormap."""
        try:
            cmap = pg.colormap.get(colormap_name)
            if cmap is not None:
                lut = cmap.getLookupTable(0.0, 1.0, 256)
                self._image_item.setLookupTable(lut)
        except Exception:
            try:
                from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
                if colormap_name in Gradients:
                    pos = Gradients[colormap_name]
                    positions = [p[0] for p in pos]
                    colors = [pg.mkColor(p[1]) for p in pos]
                    cmap = pg.ColorMap(positions, colors)
                    self._image_item.setLookupTable(cmap.getLookupTable())
            except Exception:
                pass
    
    def _update_levels(self, auto_range):
        """Update image levels."""
        if auto_range:
            data_min = np.nanmin(self._decoded_data)
            data_max = np.nanmax(self._decoded_data)
            self._image_item.setLevels([data_min, data_max])
    
    def _reset_view(self):
        """Reset the view."""
        self._plot_widget.setAspectLocked(True)
        self._plot_widget.autoRange()
    
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
            if self.config.get("auto_show_display", True) and self._decoded_data is not None:
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
        return self._decoded_data
    
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
        if self._decoded_data is None:
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
                height, width = self._decoded_data.shape
                if self._scan_dimensions.get('dim_x') != width or self._scan_dimensions.get('dim_y') != height:
                    # Data was interpolated, need to map to interpolated coordinates
                    x_min = min(self._x_positions)
                    x_max = max(self._x_positions)
                    y_min = min(self._y_positions)
                    y_max = max(self._y_positions)
                    
                    x_norm = (x - x_min) / (x_max - x_min) if x_max != x_min else 0
                    y_norm = (y - y_min) / (y_max - y_min) if y_max != y_min else 0
                    
                    x_idx = int(round(x_norm * (width - 1)))
                    y_idx = int(round(y_norm * (height - 1)))
                
                # Clamp to valid range
                x_idx = max(0, min(width - 1, x_idx))
                y_idx = max(0, min(height - 1, y_idx))
                
                value = self._decoded_data[y_idx, x_idx]
                return value, True
            else:
                # No scan dimensions, try direct mapping
                height, width = self._decoded_data.shape
                x_idx = int(round(x))
                y_idx = int(round(y))
                
                if 0 <= x_idx < width and 0 <= y_idx < height:
                    value = self._decoded_data[y_idx, x_idx]
                    return value, True
                else:
                    return 0.0, False
                    
        except Exception as e:
            print(f"[ClickableDecoder] Error getting decoded value at position: {e}")
            return 0.0, False
    
    def cleanup(self) -> None:
        """Clean up resources when plugin is unloaded."""
        if self._display_window is not None:
            try:
                self._display_window.close()
            except Exception:
                pass
        self._scan_data = []
        self._decoded_data = None
        self._position_history = []
        self._channel_data = {}