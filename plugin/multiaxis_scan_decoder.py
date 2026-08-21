"""
Multi-Axis Scan Decoder Plugin - decodes multi-axis scan data using channel difference.

This plugin:
1. Takes multi-axis scan data from detector channels
2. Decodes using formula: (channel1 - channel2) * factor
3. Displays the decoded multi-dimensional data in a popup window
4. Matches display dimensions to detector image dimensions
"""

import numpy as np
from typing import Dict, Any, List
import sys
from pathlib import Path
from dataclasses import dataclass, field

# Import base plugin classes from the main plugins package
try:
    # Try absolute import first
    from plugins.base_plugin import BasePlugin, PluginData, PluginResult, DecoderPlugin
except ImportError:
    try:
        # Try relative import
        from ..plugins.base_plugin import BasePlugin, PluginData, PluginResult, DecoderPlugin
    except ImportError:
        # Try adding parent directory to path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        try:
            from plugins.base_plugin import BasePlugin, PluginData, PluginResult, DecoderPlugin
        except ImportError:
            # Fallback: define the classes inline
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


class MultiAxisScanDecoderPlugin(DecoderPlugin):
    """Plugin that decodes multi-axis scan data using channel difference."""
    
    def get_name(self) -> str:
        return "Multi-Axis Scan Decoder"
    
    def get_description(self) -> str:
        return "Decodes multi-axis scan data using (channel1 - channel2) * factor and displays results in popup window"
    
    def get_version(self) -> str:
        return "1.0.0"
    
    def __init__(self):
        super().__init__()
        self.author = "Microscope Controller"
        self.description = self.get_description()
        
        # Configuration
        self.config = {
            "enabled": False,  # Disable plugin by default
            "channel1_name": "channel1",  # Name of first channel
            "channel2_name": "channel2",  # Name of second channel
            "decoder_factor": 1.0,  # Factor to multiply the difference by
            "display_window_title": "Multi-Axis Scan Data",  # Title for display window
            "auto_show_display": True,  # Automatically show display window when data is processed
            "colormap": "viridis",  # Colormap for display
        }
        
        # Internal state
        self._scan_data = []  # Store accumulated scan data
        self._decoded_data = None  # Store decoded multi-dimensional data
        self._display_window = None  # Reference to display window
        self._scan_dimensions = None  # Store dimensions of scan data
        self._position_history = []  # Store position history for reconstruction
        self._reference_detector_shape = None  # Store reference detector image shape for dimension matching
        
        # GUI imports for later use
        self._QtCore = None
        self._QPointF = None
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize plugin with configuration."""
        if config:
            self.config.update(config)
        return True
    
    def validate_config(self, config: Dict[str, Any]) -> tuple:
        """Validate plugin configuration."""
        if "decoder_factor" in config:
            try:
                factor = float(config["decoder_factor"])
                if factor == 0:
                    return False, "decoder_factor cannot be zero"
            except (ValueError, TypeError):
                return False, "decoder_factor must be a number"
        
        return True, ""
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for UI generation."""
        return {
            "enabled": {
                "type": "bool",
                "default": False,
                "description": "Enable/disable plugin"
            },
            "channel1_name": {
                "type": "str",
                "default": "channel1",
                "description": "Name of first detector channel to use"
            },
            "channel2_name": {
                "type": "str", 
                "default": "channel2",
                "description": "Name of second detector channel to use"
            },
            "decoder_factor": {
                "type": "float",
                "min": -1000.0,
                "max": 1000.0,
                "default": 1.0,
                "description": "Factor to multiply (channel1 - channel2) by"
            },
            "display_window_title": {
                "type": "str",
                "default": "Multi-Axis Scan Data",
                "description": "Title for the display window"
            },
            "auto_show_display": {
                "type": "bool",
                "default": True,
                "description": "Automatically show display window when data is processed"
            },
            "colormap": {
                "type": "str",
                "default": "viridis",
                "description": "Colormap for display (viridis, plasma, inferno, magma, cividis, etc.)"
            }
        }
    
    def on_experiment_start(self, experiment_config: Dict[str, Any]) -> None:
        """Called when an experiment starts."""
        self._scan_data = []
        self._decoded_data = None
        self._position_history = []
        self._scan_dimensions = None
        print("[MultiAxisScanDecoder] Experiment started - ready to collect scan data")
    
    def manual_execute_with_data(self, detector_data: Dict[str, np.ndarray], 
                                 position_history: List[Dict[str, float]],
                                 scan_dimensions: Dict[str, Any] = None) -> bool:
        """Manually execute the plugin with provided data.
        
        This method allows the plugin to be run manually on existing scan data
        without going through the full experiment lifecycle.
        
        Args:
            detector_data: Dictionary mapping detector IDs to data arrays
            position_history: List of position dictionaries
            scan_dimensions: Optional scan dimensions information
            
        Returns:
            True if execution successful, False otherwise
        """
        try:
            print("[MultiAxisScanDecoder] Manual execution started")
            
            # Debug: Show current config
            print(f"[MultiAxisScanDecoder] Current config: {self.config}")
            
            # Debug: Show available detector data
            print(f"[MultiAxisScanDecoder] Available detector data keys: {list(detector_data.keys())}")
            for det_id, data in detector_data.items():
                # Convert to numpy array if not already
                if not isinstance(data, np.ndarray):
                    data = np.array(data)
                    detector_data[det_id] = data
                print(f"[MultiAxisScanDecoder]   {det_id}: shape={data.shape if hasattr(data, 'shape') else 'N/A'}")
            
            # Reset state
            self._scan_data = []
            self._decoded_data = None
            self._position_history = []
            self._scan_dimensions = None
            
            # Set position history
            self._position_history = position_history if position_history else []
            print(f"[MultiAxisScanDecoder] Position history length: {len(self._position_history)}")
            
            # Set scan dimensions if provided
            if scan_dimensions:
                self._scan_dimensions = scan_dimensions
                print(f"[MultiAxisScanDecoder] Scan dimensions: {scan_dimensions}")
            
            # Convert detector data to scan data format
            channel1_name = self.config.get("channel1_name", "channel1")
            channel2_name = self.config.get("channel2_name", "channel2")
            
            print(f"[MultiAxisScanDecoder] Looking for channels: '{channel1_name}' and '{channel2_name}'")
            
            # Find the channels
            channel1_data = None
            channel2_data = None
            channel1_matched_id = None
            channel2_matched_id = None
            
            for det_id, data in detector_data.items():
                det_id_lower = det_id.lower()
                print(f"[MultiAxisScanDecoder] Checking detector '{det_id}' (lower: '{det_id_lower}')")
                
                # Check for channel1 match
                if channel1_name.lower() in det_id_lower or det_id == channel1_name:
                    channel1_data = data
                    channel1_matched_id = det_id
                    print(f"[MultiAxisScanDecoder] ✓ Matched '{det_id}' to channel1 '{channel1_name}'")
                
                # Check for channel2 match
                elif channel2_name.lower() in det_id_lower or det_id == channel2_name:
                    channel2_data = data
                    channel2_matched_id = det_id
                    print(f"[MultiAxisScanDecoder] ✓ Matched '{det_id}' to channel2 '{channel2_name}'")
            
            print(f"[MultiAxisScanDecoder] Channel matching results:")
            print(f"[MultiAxisScanDecoder]   Channel1 ({channel1_name}): {'FOUND' if channel1_data is not None else 'NOT FOUND'} (matched: {channel1_matched_id})")
            print(f"[MultiAxisScanDecoder]   Channel2 ({channel2_name}): {'FOUND' if channel2_data is not None else 'NOT FOUND'} (matched: {channel2_matched_id})")
            
            if channel1_data is None or channel2_data is None:
                print(f"[MultiAxisScanDecoder] ✗ Could not find required channels: {channel1_name}, {channel2_name}")
                print(f"[MultiAxisScanDecoder] Available detectors: {list(detector_data.keys())}")
                return False
            
            # Ensure both arrays have the same length
            min_length = min(len(channel1_data), len(channel2_data))
            channel1_data = channel1_data[:min_length]
            channel2_data = channel2_data[:min_length]
            
            # Match with position history
            pos_length = min(len(self._position_history), min_length)
            
            # Create scan data points
            for i in range(pos_length):
                data_point = {
                    "channel1": float(channel1_data[i]) if i < len(channel1_data) else 0.0,
                    "channel2": float(channel2_data[i]) if i < len(channel2_data) else 0.0,
                    "positions": self._position_history[i] if i < len(self._position_history) else {},
                    "measurement_index": i,
                    "timestamp": i
                }
                self._scan_data.append(data_point)
            
            print(f"[MultiAxisScanDecoder] Created {len(self._scan_data)} data points from manual execution")
            
            # Process the scan data
            self._process_scan_data()
            
            # Show display if enabled
            if self.config.get("auto_show_display", True) and self._decoded_data is not None:
                self._show_display_window()
            
            print("[MultiAxisScanDecoder] Manual execution completed successfully")
            return True
            
        except Exception as e:
            print(f"[MultiAxisScanDecoder] Manual execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def on_experiment_end(self, experiment_config: Dict[str, Any]) -> None:
        """Called when an experiment ends."""
        print(f"[MultiAxisScanDecoder] Experiment ended - collected {len(self._scan_data)} data points")
        
        # Process accumulated data if we have enough
        if len(self._scan_data) > 0:
            self._process_scan_data()
            
            # Show display if enabled (only if QApplication exists)
            if self.config.get("auto_show_display", True) and self._decoded_data is not None:
                try:
                    from PyQt6.QtWidgets import QApplication
                    if QApplication.instance() is not None:
                        self._show_display_window()
                    else:
                        print("[MultiAxisScanDecoder] QApplication not available - skipping display window")
                except ImportError:
                    print("[MultiAxisScanDecoder] PyQt6 not available - skipping display window")
    
    def process_data(self, data: PluginData) -> PluginResult:
        """Process measurement data and accumulate scan information."""
        result = PluginResult()
        
        # Debug: Show current config and available data
        print(f"[MultiAxisScanDecoder] process_data called")
        print(f"[MultiAxisScanDecoder] Current config channel1_name: {self.config.get('channel1_name', 'default')}")
        print(f"[MultiAxisScanDecoder] Current config channel2_name: {self.config.get('channel2_name', 'default')}")
        print(f"[MultiAxisScanDecoder] Available detector data keys: {list(data.detector_data.keys())}")
        
        # Store reference detector image shape if available
        if data.camera_image is not None and self._reference_detector_shape is None:
            try:
                self._reference_detector_shape = np.asarray(data.camera_image).shape
                print(f"[MultiAxisScanDecoder] Stored reference detector image shape: {self._reference_detector_shape}")
            except Exception:
                pass
        
        # Store scan data with position information
        if data.detector_data:
            channel1_name = self.config.get("channel1_name", "channel1")
            channel2_name = self.config.get("channel2_name", "channel2")
            
            # Get channel values
            channel1_value = None
            channel2_value = None
            
            for detector_id, values in data.detector_data.items():
                if channel1_name in detector_id.lower() or detector_id == channel1_name:
                    if len(values) > 0:
                        channel1_value = float(values[-1])
                elif channel2_name in detector_id.lower() or detector_id == channel2_name:
                    if len(values) > 0:
                        channel2_value = float(values[-1])
            
            # Store data point with position
            data_point = {
                "channel1": channel1_value,
                "channel2": channel2_value,
                "positions": data.positions.copy(),
                "measurement_index": data.measurement_index,
                "timestamp": data.timestamps[-1] if len(data.timestamps) > 0 else None
            }
            
            self._scan_data.append(data_point)
            self._position_history.append(data.positions.copy())
            
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
            print("[MultiAxisScanDecoder] No scan data to process")
            return
        
        print(f"[MultiAxisScanDecoder] Processing {len(self._scan_data)} scan data points")
        
        # Get unique positions for each axis to determine grid dimensions
        x_positions = sorted(set([d["positions"].get("x", 0) for d in self._scan_data]))
        y_positions = sorted(set([d["positions"].get("y", 0) for d in self._scan_data]))
        
        # Determine scan dimensions
        dim_x = len(x_positions)
        dim_y = len(y_positions)
        
        # Match detector image dimensions if available
        if self._reference_detector_shape is not None:
            try:
                # Assume detector shape is (height, width) or similar
                # Use the detector dimensions as the target
                ref_height, ref_width = self._reference_detector_shape[:2]
                
                # Resize our scan grid to match detector dimensions
                # if scan is smaller, we'll interpolate; if larger, we'll downsample
                print(f"[MultiAxisScanDecoder] Matching detector image dimensions: {ref_width}x{ref_height}")
                
                # For now, if dimensions don't match, we'll use interpolation
                if dim_x != ref_width or dim_y != ref_height:
                    print(f"[MultiAxisScanDecoder] Will interpolate from {dim_x}x{dim_y} to {ref_width}x{ref_height}")
                    # Will handle interpolation after filling the array
                    target_dim_x, target_dim_y = ref_width, ref_height
                else:
                    target_dim_x, target_dim_y = dim_x, dim_y
            except Exception as e:
                print(f"[MultiAxisScanDecoder] Could not match detector dimensions: {e}")
                target_dim_x, target_dim_y = dim_x, dim_y
        else:
            target_dim_x, target_dim_y = dim_x, dim_y
            print("[MultiAxisScanDecoder] Using original scan dimensions (no reference detector shape available)")
        
        self._scan_dimensions = {
            "x_positions": x_positions,
            "y_positions": y_positions,
            "dim_x": target_dim_x,
            "dim_y": target_dim_y,
            "original_dim_x": dim_x,
            "original_dim_y": dim_y
        }
        
        print(f"[MultiAxisScanDecoder] Final display dimensions: {target_dim_x}x{target_dim_y}")
        
        # Create decoded data array at original scan resolution
        decoded_array = np.zeros((dim_y, dim_x))
        
        # Fill array with decoded values using (channel1 - channel2) * factor
        factor = self.config.get("decoder_factor", 1.0)
        
        for data_point in self._scan_data:
            x_pos = data_point["positions"].get("x", 0)
            y_pos = data_point["positions"].get("y", 0)
            
            # Find indices
            try:
                x_idx = x_positions.index(x_pos)
                y_idx = y_positions.index(y_pos)
            except ValueError:
                continue  # Skip if position not found in grid
            
            # Calculate decoded value
            channel1 = data_point["channel1"] if data_point["channel1"] is not None else 0.0
            channel2 = data_point["channel2"] if data_point["channel2"] is not None else 0.0
            
            decoded_value = (channel1 - channel2) * factor
            decoded_array[y_idx, x_idx] = decoded_value
        
        # Interpolate to match detector dimensions if needed
        if (dim_x != target_dim_x or dim_y != target_dim_y) and dim_x > 0 and dim_y > 0:
            try:
                from scipy.ndimage import zoom
                # Calculate zoom factors
                zoom_y = target_dim_y / dim_y
                zoom_x = target_dim_x / dim_x
                decoded_array = zoom(decoded_array, (zoom_y, zoom_x), order=1)
                print(f"[MultiAxisScanDecoder] Interpolated decoded data to shape: {decoded_array.shape}")
            except ImportError:
                print("[MultiAxisScanDecoder] scipy not available - using original dimensions")
            except Exception as e:
                print(f"[MultiAxisScanDecoder] Interpolation failed: {e} - using original dimensions")
        
        self._decoded_data = decoded_array
        print(f"[MultiAxisScanDecoder] Decoded data shape: {decoded_array.shape}")
        print(f"[MultiAxisScanDecoder] Decoded data range: {np.nanmin(decoded_array):.3f} to {np.nanmax(decoded_array):.3f}")
    
    def _show_display_window(self):
        """Show popup window with decoded multi-dimensional data."""
        if self._decoded_data is None:
            print("[MultiAxisScanDecoder] No decoded data to display")
            return
        
        try:
            # Import PyQt6 for GUI
            from PyQt6 import QtWidgets, QtCore, QtGui
            from PyQt6.QtCore import QRectF, QPointF
            import pyqtgraph as pg
            
            # Store imports for later use in mouse event handler
            self._QtCore = QtCore
            self._QPointF = QPointF
            
            # Create window if it doesn't exist
            if self._display_window is None or not self._display_window.isVisible():
                self._display_window = QtWidgets.QMainWindow()
                self._display_window.setWindowTitle(self.config.get("display_window_title", "Multi-Axis Scan Data"))
                self._display_window.resize(1000, 800)
                
                # Create central widget
                central_widget = QtWidgets.QWidget()
                self._display_window.setCentralWidget(central_widget)
                layout = QtWidgets.QVBoxLayout(central_widget)
                
                # Create a more interactive plot widget instead of ImageView
                self._plot_widget = pg.PlotWidget()
                self._plot_widget.setAspectLocked(True)
                self._plot_widget.setTitle('Multi-Axis Scan Data', color='w', size='12pt')
                self._plot_widget.setLabel('left', 'Y Position', units='units')
                self._plot_widget.setLabel('bottom', 'X Position', units='units')
                self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
                
                # Create image item
                self._image_item = pg.ImageItem()
                self._plot_widget.addItem(self._image_item)
                
                # Add crosshair cursor for mouse interaction
                self._v_line = pg.InfiniteLine(angle=90, movable=False)
                self._h_line = pg.InfiniteLine(angle=0, movable=False)
                self._plot_widget.addItem(self._v_line, ignoreBounds=True)
                self._plot_widget.addItem(self._h_line, ignoreBounds=True)
                
                # Add hover label for displaying position and value
                self._hover_label = QtWidgets.QLabel()
                self._hover_label.setStyleSheet("""
                    QLabel {
                        background-color: rgba(0, 0, 0, 180);
                        color: white;
                        padding: 4px 8px;
                        border-radius: 4px;
                        font-size: 10pt;
                    }
                """)
                self._hover_label.hide()
                
                # Position the hover label as an overlay
                self._hover_label.setParent(self._plot_widget)
                self._hover_label.raise_()
                
                layout.addWidget(self._plot_widget)
                
                # Add control panel
                control_panel = QtWidgets.QWidget()
                control_layout = QtWidgets.QHBoxLayout(control_panel)
                
                # Colormap selector
                control_layout.addWidget(QtWidgets.QLabel("Colormap:"))
                self._cmap_combo = QtWidgets.QComboBox()
                self._populate_colormap_combo()
                self._cmap_combo.setCurrentText(self.config.get("colormap", "viridis"))
                self._cmap_combo.currentTextChanged.connect(self._update_colormap)
                control_layout.addWidget(self._cmap_combo)
                
                # Auto-levels checkbox
                self._auto_levels_cb = QtWidgets.QCheckBox("Auto Range")
                self._auto_levels_cb.setChecked(True)
                self._auto_levels_cb.toggled.connect(self._update_levels)
                control_layout.addWidget(self._auto_levels_cb)
                
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
                
                # Add status bar for mouse position information
                self._status_bar = self._display_window.statusBar()
                self._status_bar.showMessage("Ready - Move mouse over image to see position and value")
                
                # Enable mouse tracking on the plot widget
                self._plot_widget.setMouseEnabled(True)
                self._plot_widget.setMouseTracking(True)
                
                # Connect mouse movement for hover information using a different approach
                # We'll use a proxy widget to capture mouse events more reliably
                from PyQt6.QtCore import QTimer
                self._mouse_timer = QTimer()
                self._mouse_timer.timeout.connect(self._check_mouse_position)
                self._mouse_timer.start(50)  # Check every 50ms
                
                # Store cursor position
                self._last_mouse_pos = None
                
                # Store scan positions for coordinate mapping
                self._x_positions = None
                self._y_positions = None
                if self._scan_dimensions:
                    self._x_positions = self._scan_dimensions.get('x_positions', [])
                    self._y_positions = self._scan_dimensions.get('y_positions', [])
            
            # Set the decoded data with proper levels and coordinate system
            data_min = np.nanmin(self._decoded_data)
            data_max = np.nanmax(self._decoded_data)
            
            # Set the image with proper coordinate mapping
            # We'll use array indices for display and map to actual positions in hover function
            height, width = self._decoded_data.shape
            
            print(f"[MultiAxisScanDecoder] Setting image data: shape={self._decoded_data.shape}, range=[{data_min:.3f}, {data_max:.3f}]")
            
            # Set image with levels and explicit shape
            self._image_item.setImage(self._decoded_data, levels=[data_min, data_max])
            
            # Set up coordinate system
            if self._scan_dimensions and self._x_positions and self._y_positions:
                # Use actual scan positions for coordinate mapping
                x_min = min(self._x_positions)
                x_max = max(self._x_positions)
                y_min = min(self._y_positions)
                y_max = max(self._y_positions)
                
                print(f"[MultiAxisScanDecoder] Setting coordinate range: X[{x_min:.1f}, {x_max:.1f}], Y[{y_min:.1f}, {y_max:.1f}]")
                
                # Set the plot item range to match scan positions
                self._plot_widget.setXRange(x_min, x_max)
                self._plot_widget.setYRange(y_min, y_max)
                
                # Store coordinate mapping information
                self._coord_mapping = {
                    'x_min': x_min, 'x_max': x_max,
                    'y_min': y_min, 'y_max': y_max,
                    'width': width, 'height': height
                }
            else:
                # Use array indices as coordinates
                self._plot_widget.setXRange(0, width)
                self._plot_widget.setYRange(0, height)
                self._coord_mapping = None
                print(f"[MultiAxisScanDecoder] Using array indices as coordinates: 0-{width}, 0-{height}")
            
            # Apply colormap
            self._update_colormap(self.config.get("colormap", "viridis"))
            
            # Ensure levels are set properly
            self._image_item.setLevels([data_min, data_max])
            
            # Update info label with dimension information
            if self._scan_dimensions:
                dim_info = f"Scan dimensions: {self._scan_dimensions['dim_x']}x{self._scan_dimensions['dim_y']}"
                range_info = f"Data range: {np.nanmin(self._decoded_data):.3f} to {np.nanmax(self._decoded_data):.3f}"
                factor_info = f"Decoder factor: {self.config.get('decoder_factor', 1.0)}"
                channels_info = f"Channels: {self.config.get('channel1_name', 'ch1')} - {self.config.get('channel2_name', 'ch2')}"
                self._info_label.setText(f"{dim_info} | {range_info} | {factor_info} | {channels_info}")
            
            # Show window and bring to front
            self._display_window.show()
            self._display_window.raise_()
            self._display_window.activateWindow()
            
            print("[MultiAxisScanDecoder] Interactive display window shown with mouse hover and zoom capabilities")
            
        except ImportError:
            print("[MultiAxisScanDecoder] PyQt6 or pyqtgraph not available - cannot display window")
            print(f"[MultiAxisScanDecoder] Decoded data shape: {self._decoded_data.shape}")
            print(f"[MultiAxisScanDecoder] Decoded data range: {np.nanmin(self._decoded_data):.3f} to {np.nanmax(self._decoded_data):.3f}")
        except Exception as e:
            print(f"[MultiAxisScanDecoder] Error showing display window: {e}")
            import traceback
            traceback.print_exc()
    
    def _populate_colormap_combo(self):
        """Populate colormap combo box with available colormaps."""
        try:
            from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
            colormaps = sorted(list(Gradients.keys()))
            self._cmap_combo.clear()
            for cmap in colormaps:
                self._cmap_combo.addItem(cmap)
        except Exception:
            # Fallback to basic colormaps
            basic_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'grey', 'hot', 'cool']
            self._cmap_combo.clear()
            for cmap in basic_cmaps:
                self._cmap_combo.addItem(cmap)
    
    def _update_colormap(self, colormap_name):
        """Update the colormap for the image display."""
        try:
            # Try the modern pyqtgraph colormap API first
            try:
                cmap = pg.colormap.get(colormap_name)
                if cmap is not None:
                    lut = cmap.getLookupTable(0.0, 1.0, 256)
                    self._image_item.setLookupTable(lut)
                    return
            except Exception:
                pass
            
            # Fallback to gradient editor method
            try:
                from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
                if colormap_name in Gradients:
                    pos = Gradients[colormap_name]
                    # Extract positions and colors
                    positions = [p[0] for p in pos]
                    colors = [pg.mkColor(p[1]) for p in pos]
                    cmap = pg.ColorMap(positions, colors)
                    self._image_item.setLookupTable(cmap.getLookupTable())
                    return
            except Exception:
                pass
            
            # Final fallback - try direct colormap creation
            try:
                # Create a simple colormap if nothing else works
                cmap = pg.colormap.get('viridis')  # Default fallback
                self._image_item.setLookupTable(cmap.getLookupTable(0.0, 1.0, 256))
            except Exception:
                pass
                
        except Exception:
            pass
    
    def _update_levels(self, auto_range):
        """Update image levels (min/max for display)."""
        if auto_range:
            # Reset to full data range
            data_min = np.nanmin(self._decoded_data)
            data_max = np.nanmax(self._decoded_data)
            self._image_item.setLevels([data_min, data_max])
        else:
            # Keep current levels (no change when unchecked)
            # This allows users to manually adjust levels via other controls if added later
            pass
    
    def _reset_view(self):
        """Reset the view to show the entire image."""
        self._plot_widget.setAspectLocked(True)
        self._plot_widget.autoRange()
    
    def _on_mouse_moved(self, pos):
        """Handle mouse movement to show position and value information."""
        if self._decoded_data is None:
            return
        
        try:
            # Import QPointF if not already available
            if self._QPointF is None:
                from PyQt6.QtCore import QPointF
                self._QPointF = QPointF
            
            # Check if pos is valid
            if pos is None or (pos.x() is None and pos.y() is None):
                self._hover_label.hide()
                return
            
            # Get mouse position in plot coordinates
            mouse_point = self._plot_widget.plotItem.vb.mapSceneToView(pos)
            x, y = mouse_point.x(), mouse_point.y()
            
            # Convert plot coordinates to array indices
            height, width = self._decoded_data.shape
            
            # Map plot coordinates to array indices
            if hasattr(self, '_coord_mapping') and self._coord_mapping:
                # Use coordinate mapping to convert to array indices
                mapping = self._coord_mapping
                x_norm = (x - mapping['x_min']) / (mapping['x_max'] - mapping['x_min']) if mapping['x_max'] != mapping['x_min'] else 0
                y_norm = (y - mapping['y_min']) / (mapping['y_max'] - mapping['y_min']) if mapping['y_max'] != mapping['y_min'] else 0
                
                x_idx = int(round(x_norm * (mapping['width'] - 1)))
                y_idx = int(round(y_norm * (mapping['height'] - 1)))
            else:
                # Direct mapping (plot coordinates = array indices)
                x_idx = int(round(x))
                y_idx = int(round(y))
            
            # Clamp indices to valid range
            x_idx = max(0, min(width - 1, x_idx))
            y_idx = max(0, min(height - 1, y_idx))
            
            # Get the value at this position
            # Array indexing is [row, column] = [y_idx, x_idx]
            value = self._decoded_data[y_idx, x_idx]
            
            # Get actual scan positions if available
            x_pos = x_idx
            y_pos = y_idx
            if self._x_positions and 0 <= x_idx < len(self._x_positions):
                x_pos = self._x_positions[x_idx]
            if self._y_positions and 0 <= y_idx < len(self._y_positions):
                y_pos = self._y_positions[y_idx]
            
            # Update crosshair positions
            self._v_line.setPos(x)
            self._h_line.setPos(y)
            
            # Update hover label with position and value
            hover_text = f"X: {x_pos:.1f}, Y: {y_pos:.1f} | Value: {value:.4f}"
            self._hover_label.setText(hover_text)
            
            # Update status bar with position and value information
            if hasattr(self, '_status_bar') and self._status_bar:
                status_text = f"Position: X={x_pos:.2f}, Y={y_pos:.2f} | Value: {value:.6f}"
                self._status_bar.showMessage(status_text)
            
            # Position the label near the mouse
            label_pos = self._QPointF(pos.x() + 15, pos.y() + 15)
            self._hover_label.move(label_pos.x(), label_pos.y())
            self._hover_label.show()
                
        except Exception as e:
            # Hide label and move crosshairs out of view on error
            self._hover_label.hide()
            try:
                # Move crosshairs far away instead of setting to None
                self._v_line.setPos(1e6)
                self._h_line.setPos(1e6)
            except Exception:
                pass
            
            # Clear status bar on error
            if hasattr(self, '_status_bar') and self._status_bar:
                self._status_bar.showMessage("Ready - Move mouse over image to see position and value")
    
    def _check_mouse_position(self):
        """Periodically check mouse position over the plot widget."""
        try:
            # Get cursor position using Qt's global cursor
            from PyQt6.QtGui import QCursor
            global_pos = QCursor.pos()
            cursor_pos = self._plot_widget.mapFromGlobal(global_pos)
            
            # Check if cursor is within the plot widget bounds
            if (0 <= cursor_pos.x() <= self._plot_widget.width() and 
                0 <= cursor_pos.y() <= self._plot_widget.height()):
                
                # Convert to scene coordinates
                scene_pos = self._plot_widget.mapToScene(cursor_pos)
                
                # Only process if position changed significantly
                if (self._last_mouse_pos is None or 
                    abs(scene_pos.x() - self._last_mouse_pos.x()) > 1 or
                    abs(scene_pos.y() - self._last_mouse_pos.y()) > 1):
                    
                    self._last_mouse_pos = scene_pos
                    self._on_mouse_moved(scene_pos)
            else:
                # Mouse is outside the widget
                self._hover_label.hide()
                self._last_mouse_pos = None
                
        except Exception as e:
            pass
    
    def get_decoded_data(self) -> np.ndarray:
        """Get the decoded multi-dimensional data."""
        return self._decoded_data
    
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
            print(f"[MultiAxisScanDecoder] Error getting decoded value at position: {e}")
            return 0.0, False
    
    def get_scan_dimensions(self) -> Dict[str, Any]:
        """Get the scan dimensions information."""
        return self._scan_dimensions
    
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