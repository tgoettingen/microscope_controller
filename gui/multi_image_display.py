"""
Multi-Image Display Widget

A standalone widget for displaying multiple images with:
- Colormap selection (including RGB colormaps)
- X/Y coordinate range tuning
- Gamma correction
- Log/linear intensity mapping
- False color channel overlay
- Reset limits functionality
- Context menu with binary options
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field


@dataclass
class ImageDisplaySettings:
    """Settings for a single image display."""
    colormap: str = "plasma"
    min_level: float = 0.0
    max_level: float = 1.0
    gamma: float = 1.0
    scale_mode: str = "Linear"  # "Linear" or "Log"
    overlay_enabled: bool = False
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None


class MultiImageDisplay:
    """
    Standalone multi-image display widget.
    
    Features:
    - Display multiple images in an adaptive grid
    - Per-image colormaps (including RGB)
    - X/Y coordinate range adjustment
    - Gamma correction
    - Log/linear intensity mapping
    - False color channel overlay
    - Reset limits
    - Context menu with binary options
    """
    
    def __init__(self, parent=None):
        """Initialize the multi-image display."""
        self._parent = parent
        self._images: Dict[str, np.ndarray] = {}  # Image data by name
        self._channel_data: Dict[str, np.ndarray] = {}  # Channel data for overlay
        self._settings: Dict[str, ImageDisplaySettings] = {}  # Settings per image
        self._overlay_mode = False  # Global overlay mode
        self._focused_image: Optional[str] = None  # Currently focused image
        
        # PyQt imports (lazy loading)
        self._pg = None
        self._QtWidgets = None
        self._QtCore = None
        self._QtGui = None
        
        # Widget references
        self._display_window = None
        self._plots_grid = None
        self._plot_widgets: Dict[str, Any] = {}
        self._image_items: Dict[str, Any] = {}
        self._overlay_plot_widget = None
        self._overlay_image_item = None
        
        # Control widgets
        self._cmap_combo = None
        self._min_spin = None
        self._max_spin = None
        _scale_combo = None
        self._gamma_spin = None
        self._overlay_cb = None
        self._reset_limits_btn = None
        self._reset_levels_btn = None
        self._info_label = None
        self._status_bar = None
        
        # Coordinate mapping
        self._coord_mapping: Optional[Dict[str, Any]] = None
        
        # Available colormaps
        self._colormaps = [
            "plasma", "viridis", "inferno", "magma", "cividis",
            "cool", "hot", "afmhot", "spring", "summer",
            "autumn", "winter", "bone", "copper", "gray",
            "pink", "ocean", "spectral", "jet", "hsv",
            "red", "green", "blue"  # RGB colormaps
        ]
    
    def _import_pyqt(self):
        """Import PyQt6 and pyqtgraph (lazy loading)."""
        if self._pg is not None:
            return
        
        try:
            import pyqtgraph as pg
            from PyQt6 import QtWidgets, QtCore, QtGui
            from PyQt6.QtCore import QPointF
            
            self._pg = pg
            self._QtWidgets = QtWidgets
            self._QtCore = QtCore
            self._QtGui = QtGui
            self._QPointF = QPointF
            
            print("[MultiImageDisplay] PyQt6 and pyqtgraph imported successfully")
        except ImportError as e:
            print(f"[MultiImageDisplay] Failed to import PyQt6/pyqtgraph: {e}")
            raise
    
    def set_image(self, name: str, data: np.ndarray):
        """Set or update an image."""
        self._images[name] = data
        
        # Initialize settings if not exists
        if name not in self._settings:
            self._settings[name] = ImageDisplaySettings()
            
            # Set initial levels from data
            data_min = np.nanmin(data)
            data_max = np.nanmax(data)
            if not np.isfinite(data_min) or not np.isfinite(data_max):
                data_min = 0.0
                data_max = 1.0
            elif data_min == data_max:
                data_min = data_min - 0.5
                data_max = data_max + 0.5
            
            self._settings[name].min_level = data_min
            self._settings[name].max_level = data_max
        
        print(f"[MultiImageDisplay] Set image '{name}': shape {data.shape}, range {self._settings[name].min_level:.3f} to {self._settings[name].max_level:.3f}")
    
    def auto_assign_rgb_colormaps(self):
        """Automatically assign RGB colormaps to the first 3 images."""
        num_images = len(self._images)
        if num_images == 0:
            return
        
        image_names = list(self._images.keys())
        rgb_colormaps = ["red", "green", "blue"]
        
        if num_images >= 3:
            # First three images get RGB
            for i in range(3):
                self._settings[image_names[i]].colormap = rgb_colormaps[i]
                print(f"[MultiImageDisplay] Auto-assigned colormap '{rgb_colormaps[i]}' to image '{image_names[i]}'")
        elif num_images == 2:
            # First two images get red and green
            self._settings[image_names[0]].colormap = "red"
            self._settings[image_names[1]].colormap = "green"
            print(f"[MultiImageDisplay] Auto-assigned colormap 'red' to image '{image_names[0]}'")
            print(f"[MultiImageDisplay] Auto-assigned colormap 'green' to image '{image_names[1]}'")
        # If only 1 image, keep default (plasma)
    
    def set_channel_data(self, name: str, data: np.ndarray):
        """Set channel data for overlay."""
        self._channel_data[name] = data
        print(f"[MultiImageDisplay] Set channel data '{name}': shape {data.shape}")
    
    def set_coordinate_ranges(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """Set coordinate ranges for all images."""
        self._coord_mapping = {
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'x_inverted': True  # X axis reversed
        }
        print(f"[MultiImageDisplay] Set coordinate ranges: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
    
    def show_display(self):
        """Show the display window."""
        self._import_pyqt()
        
        if self._display_window is None:
            self._create_display_window()
        
        # Auto-assign RGB colormaps for multiple images
        self.auto_assign_rgb_colormaps()
        
        self._update_all_images()
        self._display_window.show()
        self._display_window.raise_()
        self._display_window.activateWindow()
    
    def _create_display_window(self):
        """Create the display window with plots and controls."""
        self._display_window = self._QtWidgets.QMainWindow()
        self._display_window.setWindowTitle("Multi-Image Display")
        self._display_window.resize(1200, 800)
        
        # Central widget
        central_widget = self._QtWidgets.QWidget()
        self._display_window.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = self._QtWidgets.QVBoxLayout(central_widget)
        
        # Plots grid
        self._plots_grid = self._QtWidgets.QGridLayout()
        main_layout.addLayout(self._plots_grid, 1)
        
        # Control panel
        control_panel = self._QtWidgets.QWidget()
        control_layout = self._QtWidgets.QHBoxLayout(control_panel)
        main_layout.addWidget(control_panel)
        
        # Colormap selector
        self._cmap_combo = self._QtWidgets.QComboBox()
        self._cmap_combo.addItems(self._colormaps)
        self._cmap_combo.currentTextChanged.connect(self._on_colormap_changed)
        control_layout.addWidget(self._QtWidgets.QLabel("Colormap:"))
        control_layout.addWidget(self._cmap_combo)
        
        # Min level
        self._min_spin = self._QtWidgets.QDoubleSpinBox()
        self._min_spin.setRange(-1e10, 1e10)
        self._min_spin.setDecimals(3)
        self._min_spin.valueChanged.connect(self._on_min_changed)
        control_layout.addWidget(self._QtWidgets.QLabel("Min:"))
        control_layout.addWidget(self._min_spin)
        
        # Max level
        self._max_spin = self._QtWidgets.QDoubleSpinBox()
        self._max_spin.setRange(-1e10, 1e10)
        self._max_spin.setDecimals(3)
        self._max_spin.valueChanged.connect(self._on_max_changed)
        control_layout.addWidget(self._QtWidgets.QLabel("Max:"))
        control_layout.addWidget(self._max_spin)
        
        # Scale mode
        self._scale_combo = self._QtWidgets.QComboBox()
        self._scale_combo.addItems(["Linear", "Log"])
        self._scale_combo.currentTextChanged.connect(self._on_scale_changed)
        control_layout.addWidget(self._QtWidgets.QLabel("Scale:"))
        control_layout.addWidget(self._scale_combo)
        
        # Gamma
        self._gamma_spin = self._QtWidgets.QDoubleSpinBox()
        self._gamma_spin.setRange(0.1, 10.0)
        self._gamma_spin.setSingleStep(0.1)
        self._gamma_spin.setValue(1.0)
        self._gamma_spin.valueChanged.connect(self._on_gamma_changed)
        control_layout.addWidget(self._QtWidgets.QLabel("Gamma:"))
        control_layout.addWidget(self._gamma_spin)
        
        # Overlay checkbox
        self._overlay_cb = self._QtWidgets.QCheckBox("Overlay")
        self._overlay_cb.toggled.connect(self._on_overlay_toggled)
        control_layout.addWidget(self._overlay_cb)
        
        # Reset limits button
        self._reset_limits_btn = self._QtWidgets.QPushButton("Reset Limits")
        self._reset_limits_btn.clicked.connect(self._reset_coordinate_limits)
        control_layout.addWidget(self._reset_limits_btn)
        
        # Reset levels button
        self._reset_levels_btn = self._QtWidgets.QPushButton("Reset Levels")
        self._reset_levels_btn.clicked.connect(self._reset_intensity_levels)
        control_layout.addWidget(self._reset_levels_btn)
        
        # Info label
        self._info_label = self._QtWidgets.QLabel()
        control_layout.addWidget(self._info_label)
        
        # Status bar
        self._status_bar = self._display_window.statusBar()
        
        # Create plot widgets
        self._create_plot_widgets()
        
        # Set initial focused image
        if self._images:
            self._focused_image = list(self._images.keys())[0]
            self._update_controls_for_focused_image()
    
    def _create_plot_widgets(self):
        """Create plot widgets for all images."""
        if not self._images:
            return
        
        # Clear existing widgets
        while self._plots_grid.count():
            item = self._plots_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self._plot_widgets.clear()
        self._image_items.clear()
        
        # Calculate grid dimensions
        num_images = len(self._images)
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))
        
        # Create plot widgets
        for i, name in enumerate(self._images.keys()):
            row = i // cols
            col = i % cols
            
            # Create plot widget
            plot_widget = self._pg.PlotWidget()
            plot_widget.setAspectLocked(True)
            plot_widget.setTitle(name.replace('_', ' ').title(), color='w', size='10pt')
            plot_widget.setLabel('left', 'Y Position', units='units')
            plot_widget.setLabel('bottom', 'X Position', units='units')
            plot_widget.showGrid(x=True, y=True, alpha=0.3)
            plot_widget.setMinimumSize(300, 250)
            
            # Create image item
            image_item = self._pg.ImageItem()
            plot_widget.addItem(image_item)
            
            # Enable clicking
            plot_widget.scene().sigMouseClicked.connect(lambda event, n=name: self._on_image_clicked(event, n))
            
            # Context menu
            plot_widget.setContextMenuPolicy(self._QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            plot_widget.customContextMenuRequested.connect(lambda pos, n=name: self._show_context_menu(pos, n))
            
            # Store widgets
            self._plot_widgets[name] = plot_widget
            self._image_items[name] = image_item
            
            # Add to grid
            self._plots_grid.addWidget(plot_widget, row, col)
        
        # Add overlay plot if enabled
        if self._overlay_mode and self._overlay_plot_widget is None:
            self._create_overlay_plot()
        
        if self._overlay_mode and self._overlay_plot_widget:
            self._plots_grid.addWidget(self._overlay_plot_widget, rows, 0, 1, cols)
            self._overlay_plot_widget.setVisible(True)
        
        print(f"[MultiImageDisplay] Created {num_images} plot widgets in {rows}x{cols} grid")
    
    def _create_overlay_plot(self):
        """Create overlay plot widget."""
        if self._overlay_plot_widget is not None:
            return
        
        self._overlay_plot_widget = self._pg.PlotWidget()
        self._overlay_plot_widget.setAspectLocked(True)
        self._overlay_plot_widget.setTitle('Channel Overlay (False Color)', color='w', size='10pt')
        self._overlay_plot_widget.setLabel('left', 'Y Position', units='units')
        self._overlay_plot_widget.setLabel('bottom', 'X Position', units='units')
        self._overlay_plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._overlay_plot_widget.setMinimumSize(300, 250)
        
        self._overlay_image_item = self._pg.ImageItem()
        self._overlay_plot_widget.addItem(self._overlay_image_item)
        
        self._overlay_plot_widget.scene().sigMouseClicked.connect(self._on_overlay_clicked)
        
        print("[MultiImageDisplay] Created overlay plot widget")
    
    def _update_all_images(self):
        """Update all image displays."""
        for name, data in self._images.items():
            if name in self._image_items:
                settings = self._settings.get(name, ImageDisplaySettings())
                
                # Apply transformations
                display_data = self._apply_transformations(data, settings)
                
                # Set image
                self._image_items[name].setImage(display_data, levels=[settings.min_level, settings.max_level], autoRange=False)
                
                # Apply colormap
                self._apply_colormap(name, settings.colormap)
        
        # Update overlay if enabled
        if self._overlay_mode:
            self._update_overlay()
        
        # Apply coordinate ranges
        self._apply_coordinate_ranges()
    
    def _apply_transformations(self, data: np.ndarray, settings: ImageDisplaySettings) -> np.ndarray:
        """Apply transformations (gamma, log, flip) to data."""
        result = data.copy()
        
        # Flip X axis
        result = np.flip(result, axis=1)
        
        # Apply log scale
        if settings.scale_mode == "Log":
            data_min = np.nanmin(result)
            if data_min <= 0:
                offset = abs(data_min) + 1e-10
                result = np.log10(result + offset)
            else:
                result = np.log10(result)
        
        # Apply gamma
        data_min = np.nanmin(result)
        data_max = np.nanmax(result)
        if data_max > data_min:
            normalized = (result - data_min) / (data_max - data_min)
            result = np.power(normalized, settings.gamma) * (data_max - data_min) + data_min
        
        return result
    
    def _apply_colormap(self, name: str, colormap_name: str):
        """Apply colormap to an image."""
        if name not in self._image_items or self._pg is None:
            return
        
        image_item = self._image_items[name]
        pg = self._pg
        
        # Handle RGB colormaps
        simple_rgb = {'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255)}
        if colormap_name.lower() in simple_rgb:
            self._apply_rgb_gradient(image_item, simple_rgb[colormap_name.lower()])
            return
        
        # Try standard colormaps
        try:
            if hasattr(pg, 'colormap') and hasattr(pg.colormap, 'get'):
                cmap = pg.colormap.get(colormap_name)
                if cmap is not None:
                    lut = cmap.getLookupTable(0.0, 1.0, 256)
                    image_item.setLookupTable(lut)
                    image_item.updateImage()
                    return
        except Exception:
            pass
        
        try:
            if hasattr(pg, 'colormap'):
                cmap = pg.colormap(colormap_name)
                if cmap is not None:
                    lut = cmap.getLookupTable(0.0, 1.0, 256)
                    image_item.setLookupTable(lut)
                    image_item.updateImage()
                    return
        except Exception:
            pass
    
    def _apply_rgb_gradient(self, image_item, rgb: Tuple[int, int, int]):
        """Apply simple RGB gradient to image item."""
        # Create gradient lookup table
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            factor = i / 255.0
            lut[i] = [int(rgb[0] * factor), int(rgb[1] * factor), int(rgb[2] * factor)]
        image_item.setLookupTable(lut)
        image_item.updateImage()
    
    def _apply_coordinate_ranges(self):
        """Apply coordinate ranges to all plots."""
        if not self._coord_mapping:
            return
        
        mapping = self._coord_mapping
        for plot_widget in self._plot_widgets.values():
            plot_widget.setXRange(mapping['x_max'], mapping['x_min'])  # X reversed
            plot_widget.setYRange(mapping['y_min'], mapping['y_max'])  # Y normal
            plot_widget.plotItem.vb.enableAutoRange(enable=False)
        
        if self._overlay_plot_widget:
            self._overlay_plot_widget.setXRange(mapping['x_max'], mapping['x_min'])
            self._overlay_plot_widget.setYRange(mapping['y_min'], mapping['y_max'])
            self._overlay_plot_widget.plotItem.vb.enableAutoRange(enable=False)
    
    def _update_overlay(self):
        """Update the false color channel overlay."""
        if not self._overlay_mode or not self._channel_data or self._overlay_image_item is None:
            return
        
        try:
            # Get channel names
            channel_names = list(self._channel_data.keys())
            if not channel_names:
                return
            
            # Get reference shape
            first_channel = self._channel_data[channel_names[0]]
            height, width = first_channel.shape
            
            # Create RGB composite
            rgb_image = np.zeros((height, width, 3), dtype=np.float32)
            
            # Channel colors
            colors = [
                (1.0, 0.0, 0.0),  # Red
                (0.0, 1.0, 0.0),  # Green
                (0.0, 0.0, 1.0),  # Blue
                (1.0, 1.0, 0.0),  # Yellow
                (1.0, 0.0, 1.0),  # Magenta
                (0.0, 1.0, 1.0),  # Cyan
            ]
            
            # Get gamma from focused image
            gamma = 1.0
            if self._focused_image and self._focused_image in self._settings:
                gamma = self._settings[self._focused_image].gamma
            
            # Get scale mode from focused image
            scale_mode = "Linear"
            if self._focused_image and self._focused_image in self._settings:
                scale_mode = self._settings[self._focused_image].scale_mode
            
            # Normalize and color each channel
            for i, name in enumerate(channel_names):
                channel_data = self._channel_data[name]
                if channel_data.shape != (height, width):
                    continue
                
                # Apply log scale
                if scale_mode == "Log":
                    channel_min = np.nanmin(channel_data)
                    if channel_min <= 0:
                        offset = abs(channel_min) + 1e-10
                        channel_data = np.log10(channel_data + offset)
                    else:
                        channel_data = np.log10(channel_data)
                
                # Normalize
                channel_min = np.nanmin(channel_data)
                channel_max = np.nanmax(channel_data)
                if channel_max > channel_min:
                    normalized = (channel_data - channel_min) / (channel_max - channel_min)
                else:
                    normalized = np.zeros_like(channel_data)
                
                # Apply gamma
                normalized = np.power(normalized, gamma)
                
                # Add color
                color = colors[i % len(colors)]
                for c in range(3):
                    rgb_image[:, :, c] += normalized * color[c]
            
            # Clip and convert
            rgb_image = np.clip(rgb_image, 0, 1)
            rgb_image = (rgb_image * 255).astype(np.uint8)
            rgb_image = np.flip(rgb_image, axis=1)  # Flip X
            
            self._overlay_image_item.setImage(rgb_image, autoRange=False)
            self._overlay_image_item.setVisible(True)
            
            print(f"[MultiImageDisplay] Updated overlay with {len(channel_names)} channels")
            
        except Exception as e:
            print(f"[MultiImageDisplay] Error updating overlay: {e}")
    
    def _on_image_clicked(self, event, name: str):
        """Handle image click - focus the image."""
        self._focused_image = name
        self._update_controls_for_focused_image()
        
        # Highlight focused plot
        for n, plot_widget in self._plot_widgets.items():
            if n == name:
                plot_widget.setStyleSheet("border: 2px solid yellow;")
            else:
                plot_widget.setStyleSheet("")
    
    def _on_overlay_clicked(self, event):
        """Handle overlay click."""
        print("[MultiImageDisplay] Overlay clicked")
    
    def _show_context_menu(self, pos, name: str):
        """Show context menu for an image."""
        if name not in self._settings:
            return
        
        settings = self._settings[name]
        plot_widget = self._plot_widgets[name]
        
        menu = self._QtWidgets.QMenu()
        
        # Scale mode
        scale_menu = menu.addMenu("Scale Mode")
        linear_action = scale_menu.addAction("Linear")
        linear_action.setCheckable(True)
        linear_action.setChecked(settings.scale_mode == "Linear")
        linear_action.triggered.connect(lambda: self._set_scale_mode(name, "Linear"))
        
        log_action = scale_menu.addAction("Log")
        log_action.setCheckable(True)
        log_action.setChecked(settings.scale_mode == "Log")
        log_action.triggered.connect(lambda: self._set_scale_mode(name, "Log"))
        
        # Gamma
        gamma_menu = menu.addMenu("Gamma")
        for gamma in [0.5, 1.0, 1.5, 2.0]:
            action = gamma_menu.addAction(str(gamma))
            action.setCheckable(True)
            action.setChecked(abs(settings.gamma - gamma) < 0.01)
            action.triggered.connect(lambda g=gamma: self._set_gamma(name, g))
        
        # Colormap submenu
        cmap_menu = menu.addMenu("Colormap")
        for cmap in self._colormaps:
            action = cmap_menu.addAction(cmap)
            action.setCheckable(True)
            action.setChecked(settings.colormap == cmap)
            action.triggered.connect(lambda c=cmap: self._set_colormap(name, c))
        
        menu.exec(plot_widget.mapToGlobal(pos))
    
    def _update_controls_for_focused_image(self):
        """Update control panel to show focused image settings."""
        if self._focused_image is None or self._focused_image not in self._settings:
            return
        
        settings = self._settings[self._focused_image]
        
        # Block signals
        self._cmap_combo.blockSignals(True)
        self._min_spin.blockSignals(True)
        self._max_spin.blockSignals(True)
        self._scale_combo.blockSignals(True)
        self._gamma_spin.blockSignals(True)
        self._overlay_cb.blockSignals(True)
        
        try:
            self._cmap_combo.setCurrentText(settings.colormap)
            self._min_spin.setValue(settings.min_level)
            self._max_spin.setValue(settings.max_level)
            self._scale_combo.setCurrentText(settings.scale_mode)
            self._gamma_spin.setValue(settings.gamma)
            self._overlay_cb.setChecked(self._overlay_mode)
        finally:
            self._cmap_combo.blockSignals(False)
            self._min_spin.blockSignals(False)
            self._max_spin.blockSignals(False)
            self._scale_combo.blockSignals(False)
            self._gamma_spin.blockSignals(False)
            self._overlay_cb.blockSignals(False)
    
    # Control callbacks
    def _on_colormap_changed(self, colormap_name: str):
        """Handle colormap change."""
        if self._focused_image and self._focused_image in self._settings:
            self._settings[self._focused_image].colormap = colormap_name
            self._apply_colormap(self._focused_image, colormap_name)
    
    def _on_min_changed(self, value: float):
        """Handle min level change."""
        if self._focused_image and self._focused_image in self._settings:
            self._settings[self._focused_image].min_level = value
            if self._focused_image in self._image_items:
                self._image_items[self._focused_image].setLevels([value, self._settings[self._focused_image].max_level])
    
    def _on_max_changed(self, value: float):
        """Handle max level change."""
        if self._focused_image and self._focused_image in self._settings:
            self._settings[self._focused_image].max_level = value
            if self._focused_image in self._image_items:
                self._image_items[self._focused_image].setLevels([self._settings[self._focused_image].min_level, value])
    
    def _on_scale_changed(self, mode: str):
        """Handle scale mode change."""
        if self._focused_image and self._focused_image in self._settings:
            self._settings[self._focused_image].scale_mode = mode
            self._update_all_images()
    
    def _on_gamma_changed(self, gamma: float):
        """Handle gamma change."""
        if self._focused_image and self._focused_image in self._settings:
            self._settings[self._focused_image].gamma = gamma
            self._update_all_images()
    
    def _on_overlay_toggled(self, enabled: bool):
        """Handle overlay toggle."""
        self._overlay_mode = enabled
        self._create_plot_widgets()
        
        # Always update all images after recreating widgets
        self._update_all_images()
        
        if enabled:
            self._update_overlay()
    
    # Context menu actions
    def _set_scale_mode(self, name: str, mode: str):
        """Set scale mode for an image."""
        if name in self._settings:
            self._settings[name].scale_mode = mode
            self._update_all_images()
    
    def _set_gamma(self, name: str, gamma: float):
        """Set gamma for an image."""
        if name in self._settings:
            self._settings[name].gamma = gamma
            self._update_all_images()
    
    def _set_colormap(self, name: str, colormap: str):
        """Set colormap for an image."""
        if name in self._settings:
            self._settings[name].colormap = colormap
            self._apply_colormap(name, colormap)
    
    # Reset functions
    def _reset_coordinate_limits(self):
        """Reset coordinate limits to defaults."""
        self._coord_mapping = None
        for plot_widget in self._plot_widgets.values():
            plot_widget.enableAutoRange()
        if self._overlay_plot_widget:
            self._overlay_plot_widget.enableAutoRange()
        print("[MultiImageDisplay] Reset coordinate limits")
    
    def _reset_intensity_levels(self):
        """Reset intensity levels to data range."""
        for name, data in self._images.items():
            if name in self._settings:
                data_min = np.nanmin(data)
                data_max = np.nanmax(data)
                if not np.isfinite(data_min) or not np.isfinite(data_max):
                    data_min = 0.0
                    data_max = 1.0
                elif data_min == data_max:
                    data_min = data_min - 0.5
                    data_max = data_max + 0.5
                
                self._settings[name].min_level = data_min
                self._settings[name].max_level = data_max
                
                if name in self._image_items:
                    self._image_items[name].setLevels([data_min, data_max])
        
        if self._focused_image:
            self._update_controls_for_focused_image()
        
        print("[MultiImageDisplay] Reset intensity levels")
    
    def cleanup(self):
        """Clean up resources."""
        if self._display_window:
            self._display_window.close()
        self._images.clear()
        self._channel_data.clear()
        self._settings.clear()
        self._plot_widgets.clear()
        self._image_items.clear()
