from __future__ import annotations

import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import QToolTip
from PyQt6.QtCore import QEvent, QObject
import pyqtgraph as pg


class DetectorClickFilter(QObject):
    """Event filter to capture mouse clicks on detector images."""
    
    def __init__(self, detector_id: str, img_view, panel):
        super().__init__()
        self.detector_id = detector_id
        self.img_view = img_view
        self.panel = panel
    
    def eventFilter(self, obj, event):
        """Filter events to catch mouse clicks and releases."""
        try:
            event_type = event.type()
            
            if event_type == QEvent.Type.MouseButtonPress:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    print(f"[DetectorClickFilter] ✓ Left mouse button press detected on {self.detector_id}")
                    # Get position in widget coordinates
                    pos = event.pos()
                    # Convert to scene coordinates
                    scene_pos = self.img_view.mapToScene(pos)
                    print(f"[DetectorClickFilter] Widget pos: {pos}, Scene pos: {scene_pos}")
                    # Create a simple position object with scenePos attribute
                    class SimpleClickEvent:
                        def __init__(self, scene_pos):
                            self.scenePos = lambda: scene_pos
                        def pos(self):
                            return scene_pos
                    click_event = SimpleClickEvent(scene_pos)
                    # Call the panel's click handler
                    self.panel._on_detector_clicked(self.detector_id, self.img_view, click_event)
                    return True  # Event was handled
            
            elif event_type == QEvent.Type.MouseButtonRelease:
                if event.button() == QtCore.Qt.MouseButton.LeftButton:
                    print(f"[DetectorClickFilter] ✓ Left mouse button release detected on {self.detector_id}")
                    # Call the panel's release handler
                    self.panel._on_detector_released()
                    return True  # Event was handled
            
            elif event_type == QEvent.Type.MouseMove:
                if self.panel._mouse_button_pressed:
                    # Update tooltip position while button is held
                    pos = event.pos()
                    scene_pos = self.img_view.mapToScene(pos)
                    self.panel._update_tooltip_position(scene_pos)
                    return True  # Event was handled
        except Exception as e:
            print(f"[DetectorClickFilter] Error in event filter: {e}")
            import traceback
            traceback.print_exc()
        
        return False  # Let other events pass through


class DetectorImagePanel(QtWidgets.QWidget):
    """Container widget that holds multiple per-detector heatmaps.

    Features:
    - Per-detector colormap (gradient) selection.
    - Optional composite "false-color" overlay view.
    """

    overlay_toggled = QtCore.pyqtSignal(bool)
    overlay_settings_changed = QtCore.pyqtSignal()

    _FALSE_COLOR_RGB = [
        (1.0, 0.0, 0.0),  # red
        (0.0, 1.0, 0.0),  # green
        (0.0, 0.0, 1.0),  # blue
        (1.0, 0.0, 1.0),  # magenta
        (0.0, 1.0, 1.0),  # cyan
        (1.0, 1.0, 0.0),  # yellow
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._settings = QtCore.QSettings("MicroscopeController", "DetectorImagePanel")

        self._detectors: list[str] = []
        self._detector_widgets: dict[str, QtWidgets.QWidget] = {}
        self._detector_views: dict[str, pg.ImageView] = {}
        self._detector_cmap_combos: dict[str, QtWidgets.QComboBox] = {}

        try:
            self._default_gradient = str(self._settings.value("default_gradient", "viridis"))
        except Exception:
            self._default_gradient = "viridis"

        # overlay mapping controls: detector -> RGB channels
        try:
            self._overlay_use_all = bool(int(self._settings.value("overlay_use_all", 1)))
        except Exception:
            self._overlay_use_all = True
        self._overlay_map = {"R": "", "G": "", "B": ""}  # empty == None
        try:
            self._overlay_map["R"] = str(self._settings.value("overlay_map/R", "") or "")
            self._overlay_map["G"] = str(self._settings.value("overlay_map/G", "") or "")
            self._overlay_map["B"] = str(self._settings.value("overlay_map/B", "") or "")
        except Exception:
            pass

        # overlay coloring mode: fixed false colors or per-detector colormap
        try:
            self._overlay_color_mode = str(self._settings.value("overlay_color_mode", "fixed") or "fixed")
        except Exception:
            self._overlay_color_mode = "fixed"
        if self._overlay_color_mode not in ("fixed", "cmap"):
            self._overlay_color_mode = "fixed"
        
        # Multi-axis scan data storage for tooltips
        self._scan_data = {}  # detector_id -> list of (position, value) tuples
        self._decoded_plugins = {}  # Store references to decoder plugins for tooltip values
        self._scan_dimensions = None  # Store scan dimensions for coordinate mapping
        
        # Tooltip display options
        self._show_other_detectors = True  # Show values from other detectors
        self._show_plugin_values = True  # Show decoded values from plugins
        
        # Load tooltip setting from settings
        try:
            tooltip_setting = self._settings.value("tooltip_enabled", True)
            self._tooltip_enabled = bool(tooltip_setting)
        except Exception:
            self._tooltip_enabled = True  # Enable tooltips by default
        
        # Load other detectors display setting
        try:
            other_detectors_setting = self._settings.value("show_other_detectors", True)
            self._show_other_detectors = bool(other_detectors_setting)
        except Exception:
            self._show_other_detectors = True
        
        # Load plugin values display setting
        try:
            plugin_values_setting = self._settings.value("show_plugin_values", True)
            self._show_plugin_values = bool(plugin_values_setting)
        except Exception:
            self._show_plugin_values = True
        
        # Mouse button state tracking for persistent tooltips
        self._mouse_button_pressed = False
        self._current_tooltip_detector = None
        self._current_tooltip_position = None
        self._tooltip_label = None  # Persistent tooltip label

        # --- top controls ---
        self.overlay_cb = QtWidgets.QCheckBox("Overlay (false color)")
        try:
            self.overlay_cb.setChecked(bool(int(self._settings.value("overlay_enabled", 0))))
        except Exception:
            self.overlay_cb.setChecked(False)
        self.overlay_cb.toggled.connect(self._on_overlay_toggled)

        self.default_cmap_combo = QtWidgets.QComboBox()
        self._populate_gradients(self.default_cmap_combo)
        self._set_combo_text(self.default_cmap_combo, self._default_gradient)
        self.default_cmap_combo.currentTextChanged.connect(self._on_default_cmap_changed)

        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top)
        top_layout.setContentsMargins(4, 4, 4, 4)
        top_layout.addWidget(self.overlay_cb)
        top_layout.addSpacing(12)
        top_layout.addWidget(QtWidgets.QLabel("Default colormap:"))
        top_layout.addWidget(self.default_cmap_combo)
        top_layout.addStretch(1)

        # --- overlay controls (only meaningful when overlay enabled) ---
        self.overlay_controls = QtWidgets.QWidget()
        o = QtWidgets.QHBoxLayout(self.overlay_controls)
        o.setContentsMargins(4, 0, 4, 4)

        self.overlay_all_cb = QtWidgets.QCheckBox("Use all detectors")
        try:
            self.overlay_all_cb.setChecked(bool(self._overlay_use_all))
        except Exception:
            self.overlay_all_cb.setChecked(True)
        self.overlay_all_cb.toggled.connect(self._on_overlay_settings_changed)
        o.addWidget(self.overlay_all_cb)
        o.addSpacing(12)

        o.addWidget(QtWidgets.QLabel("Overlay colors:"))
        self.overlay_color_combo = QtWidgets.QComboBox()
        self.overlay_color_combo.addItem("Fixed false color", userData="fixed")
        self.overlay_color_combo.addItem("Detector colormap", userData="cmap")
        try:
            # restore selection
            idx = 0 if self._overlay_color_mode == "fixed" else 1
            self.overlay_color_combo.setCurrentIndex(idx)
        except Exception:
            pass
        self.overlay_color_combo.currentIndexChanged.connect(self._on_overlay_color_mode_changed)
        o.addWidget(self.overlay_color_combo)
        o.addSpacing(12)

        self.overlay_r_combo = QtWidgets.QComboBox()
        self.overlay_g_combo = QtWidgets.QComboBox()
        self.overlay_b_combo = QtWidgets.QComboBox()
        for w, name in [(self.overlay_r_combo, "R"), (self.overlay_g_combo, "G"), (self.overlay_b_combo, "B")]:
            w.currentTextChanged.connect(lambda _t, ch=name: self._on_overlay_channel_changed(ch))

        o.addWidget(QtWidgets.QLabel("R:"))
        o.addWidget(self.overlay_r_combo)
        o.addWidget(QtWidgets.QLabel("G:"))
        o.addWidget(self.overlay_g_combo)
        o.addWidget(QtWidgets.QLabel("B:"))
        o.addWidget(self.overlay_b_combo)
        o.addStretch(1)

        self.overlay_controls.setVisible(False)

        # Bundle the controls into a single widget so callers (e.g. LiveTab)
        # can move them into another panel (Detectors dock) without reaching
        # into layouts.
        self.controls_widget = QtWidgets.QWidget()
        cw = QtWidgets.QVBoxLayout(self.controls_widget)
        cw.setContentsMargins(0, 0, 0, 0)
        cw.setSpacing(0)
        cw.addWidget(top)
        cw.addWidget(self.overlay_controls)

        # --- per-detector containers ---
        self.container = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.container)
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(4, 4, 4, 4)

        # --- overlay composite view ---
        self.overlay_view = pg.ImageView()
        try:
            self.overlay_view.ui.roiBtn.hide()
        except Exception:
            pass
        try:
            self.overlay_view.ui.menuBtn.hide()
        except Exception:
            pass
        try:
            # The histogram isn't meaningful for RGB composites; keep it hidden.
            self.overlay_view.ui.histogram.hide()
        except Exception:
            pass
        try:
            self.overlay_view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            self.overlay_view.customContextMenuRequested.connect(self._show_overlay_context_menu)
        except Exception:
            pass
        self.overlay_view.hide()

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.controls_widget)
        main_layout.addWidget(self.overlay_view)
        main_layout.addWidget(self.container)

        self._refresh_overlay_detector_lists()

        # Apply visibility based on persisted overlay state.
        try:
            self._apply_overlay_visibility(bool(self.overlay_cb.isChecked()))
            self.overlay_controls.setVisible(bool(self.overlay_cb.isChecked()))
        except Exception:
            pass

    def _save_setting(self, key: str, value) -> None:
        try:
            self._settings.setValue(key, value)
        except Exception:
            pass

    def _reset_image_view(self, img_view: pg.ImageView) -> None:
        """Reset an ImageView's view range and displayed levels."""
        try:
            view = img_view.getView()
        except Exception:
            view = None
        try:
            item = img_view.getImageItem()
        except Exception:
            item = None

        try:
            if view is not None:
                view.autoRange()
        except Exception:
            pass

        try:
            if item is not None:
                image = getattr(item, "image", None)
                if image is not None:
                    arr = np.asarray(image)
                    finite = arr[np.isfinite(arr)]
                    if finite.size > 0:
                        item.setLevels(float(np.min(finite)), float(np.max(finite)))
                        return
            if hasattr(img_view, "autoLevels"):
                img_view.autoLevels()
        except Exception:
            pass

    def _show_detector_context_menu(self, detector_id: str, img_view: pg.ImageView, pos) -> None:
        """Show right-click menu for a detector image view."""
        try:
            menu = QtWidgets.QMenu(self)
            
            # Add reset image view action
            reset_action = menu.addAction("Reset Image View")
            
            # Add separator
            menu.addSeparator()
            
            # Add tooltip toggle action
            tooltip_action = menu.addAction("Enable Click Tooltips")
            tooltip_action.setCheckable(True)
            tooltip_action.setChecked(self._tooltip_enabled)
            
            # Add separator
            menu.addSeparator()
            
            # Add show other detectors toggle
            other_detectors_action = menu.addAction("Show Other Detectors")
            other_detectors_action.setCheckable(True)
            other_detectors_action.setChecked(self._show_other_detectors)
            
            # Add show plugin values toggle
            plugin_values_action = menu.addAction("Show Plugin Values")
            plugin_values_action.setCheckable(True)
            plugin_values_action.setChecked(self._show_plugin_values)
            
            # Execute menu
            chosen = menu.exec(img_view.mapToGlobal(pos))
            
            if chosen == reset_action:
                self._reset_image_view(img_view)
            elif chosen == tooltip_action:
                self._tooltip_enabled = tooltip_action.isChecked()
                print(f"[DetectorImagePanel] Tooltips {'enabled' if self._tooltip_enabled else 'disabled'}")
                
                # Hide existing tooltip if disabling
                if not self._tooltip_enabled:
                    self._hide_tooltip()
                
                # Save setting
                self._settings.setValue("tooltip_enabled", self._tooltip_enabled)
                
            elif chosen == other_detectors_action:
                self._show_other_detectors = other_detectors_action.isChecked()
                print(f"[DetectorImagePanel] Other detectors display {'enabled' if self._show_other_detectors else 'disabled'}")
                # Save setting
                self._settings.setValue("show_other_detectors", self._show_other_detectors)
                
            elif chosen == plugin_values_action:
                self._show_plugin_values = plugin_values_action.isChecked()
                print(f"[DetectorImagePanel] Plugin values display {'enabled' if self._show_plugin_values else 'disabled'}")
                # Save setting
                self._settings.setValue("show_plugin_values", self._show_plugin_values)
                
        except Exception as e:
            print(f"[DetectorImagePanel] Error showing context menu: {e}")

    def _show_overlay_context_menu(self, pos) -> None:
        """Show right-click menu for the overlay image view."""
        try:
            menu = QtWidgets.QMenu(self)
            reset_action = menu.addAction("Reset Image View")
            chosen = menu.exec(self.overlay_view.mapToGlobal(pos))
            if chosen == reset_action:
                self._reset_image_view(self.overlay_view)
        except Exception:
            pass
    
    def _on_detector_clicked(self, detector_id: str, img_view: pg.ImageView, pos) -> None:
        """Handle click on detector image to show tooltip with scan information."""
        print(f"[DetectorImagePanel] Click detected on detector: {detector_id}")
        print(f"[DetectorImagePanel] Click position type: {type(pos)}")
        print(f"[DetectorImagePanel] Click position: {pos}")
        print(f"[DetectorImagePanel] Tooltip enabled: {self._tooltip_enabled}")
        
        if not self._tooltip_enabled:
            print("[DetectorImagePanel] Tooltips are disabled, skipping")
            return
        
        # Mark mouse button as pressed
        self._mouse_button_pressed = True
        self._current_tooltip_detector = detector_id
        self._current_tooltip_position = pos
        
        try:
            # Extract scene position from MouseClickEvent if needed
            if hasattr(pos, 'scenePos'):
                scene_pos = pos.scenePos()
                print(f"[DetectorImagePanel] Extracted scenePos: {scene_pos}")
            else:
                scene_pos = pos
                print(f"[DetectorImagePanel] Using pos directly as scene_pos")
            
            # Get click position in image coordinates
            mouse_point = img_view.getView().mapSceneToView(scene_pos)
            view_x, view_y = mouse_point.x(), mouse_point.y()
            print(f"[DetectorImagePanel] View coordinates: X={view_x:.2f}, Y={view_y:.2f}")
            
            # Get the image item's bounds to map to array indices
            array_x, array_y = view_x, view_y  # Default fallback
            scan_x, scan_y = view_x, view_y  # Default fallback
            
            try:
                image_item = img_view.getImageItem()
                if image_item is not None:
                    # Get the image's bounding rect in view coordinates
                    image_rect = image_item.boundingRect()
                    print(f"[DetectorImagePanel] Image bounding rect: {image_rect}")
                    
                    # Map view coordinates to image-relative coordinates
                    img_x = view_x - image_rect.x()
                    img_y = view_y - image_rect.y()
                    
                    # Get image dimensions
                    image = image_item.image
                    if image is not None:
                        if not isinstance(image, np.ndarray):
                            image = np.asarray(image)
                        height, width = image.shape[:2]
                        
                        # Normalize to array indices
                        img_x_norm = img_x / image_rect.width() if image_rect.width() > 0 else 0
                        img_y_norm = img_y / image_rect.height() if image_rect.height() > 0 else 0
                        
                        # Convert to array indices
                        array_x = img_x_norm * (width - 1)
                        array_y = img_y_norm * (height - 1)
                        
                        print(f"[DetectorImagePanel] Array indices: X={array_x:.2f}, Y={array_y:.2f}")
                        
                        # Convert array indices to scan coordinates
                        scan_x, scan_y = self._image_coords_to_scan_coords(array_x, array_y)
                        print(f"[DetectorImagePanel] Converted to scan coordinates: X={scan_x:.2f}, Y={scan_y:.2f}")
            except Exception as e:
                print(f"[DetectorImagePanel] Error mapping coordinates: {e}")
                import traceback
                traceback.print_exc()
            
            # Use scan coordinates for data lookup
            x, y = scan_x, scan_y
            
            # Use array indices for image value extraction
            x_idx = int(round(array_x))
            y_idx = int(round(array_y))
            
            # Try to get image data
            try:
                image_item = img_view.getImageItem()
                print(f"[DetectorImagePanel] Image item: {image_item}")
                if image_item is None:
                    print("[DetectorImagePanel] No image item found")
                    return
                
                image = image_item.image
                print(f"[DetectorImagePanel] Image data: {image}")
                if image is None:
                    print("[DetectorImagePanel] No image data found")
                    return
                
                # Convert to array if needed
                if not isinstance(image, np.ndarray):
                    image = np.asarray(image)
                
                print(f"[DetectorImagePanel] Image shape: {image.shape}, dtype: {image.dtype}")
                
                # Use array indices computed earlier from image coordinates
                height, width = image.shape[:2]
                
                print(f"[DetectorImagePanel] Using array indices: x_idx={x_idx}, y_idx={y_idx}")
                print(f"[DetectorImagePanel] Image dimensions: height={height}, width={width}")
                
                # Clamp to valid range
                x_idx = max(0, min(width - 1, x_idx))
                y_idx = max(0, min(height - 1, y_idx))
                print(f"[DetectorImagePanel] Clamped indices: x_idx={x_idx}, y_idx={y_idx}")
                
                # Get value at this position
                if image.ndim == 2:
                    value = image[y_idx, x_idx]
                elif image.ndim == 3:
                    value = image[y_idx, x_idx, 0]  # Take first channel
                else:
                    value = 0
                
                print(f"[DetectorImagePanel] Extracted value: {value:.6f}")
                
                # Build tooltip information
                tooltip_lines = [
                    f"<b>Detector:</b> {detector_id}",
                    f"<b>Position:</b> X={array_x:.1f}, Y={array_y:.1f} (pixel: {x_idx}, {y_idx})",
                    f"<b>Value:</b> {value:.6f}"
                ]
                
                # For live detector images, only show the pixel value
                # Don't show accumulated scan data "closest points" - that's confusing
                print(f"[DetectorImagePanel] Tooltip for live detector image at pixel ({x_idx}, {y_idx})")
                
                # Show other detector values if enabled and data is available
                if self._show_other_detectors and self._scan_data:
                    other_detectors = [d for d in self._scan_data.keys() if d != detector_id]
                    if other_detectors:
                        tooltip_lines.append("<b>Other Detectors:</b>")
                        for other_det in other_detectors[:3]:  # Show up to 3 other detectors
                            if other_det in self._scan_data and self._scan_data[other_det]:
                                # Find closest scan point based on the clicked position
                                if self._scan_data[other_det]:
                                    closest = min(self._scan_data[other_det], 
                                               key=lambda p: abs(p[0][0] - x) + abs(p[0][1] - y))
                                    closest_pos, closest_val = closest
                                    tooltip_lines.append(f"  {other_det}: {closest_val:.6f}")
                                    print(f"[DetectorImagePanel] Other detector {other_det}: {closest_val:.6f}")
                
                # Show decoded values if enabled and plugins are registered
                if self._show_plugin_values and self._decoded_plugins:
                    tooltip_lines.append("<b>Decoded Values:</b>")
                    print(f"[DetectorImagePanel] Processing {len(self._decoded_plugins)} decoder plugins")
                    for plugin_name, plugin in self._decoded_plugins.items():
                        try:
                            # Get decoded value at this position
                            # Use scan coordinates (x, y) for decoder plugins
                            print(f"[DetectorImagePanel] Querying {plugin_name} at scan coordinates ({x:.1f}, {y:.1f})")
                            decoded_value, success = plugin.get_decoded_value_at_position(x, y)
                            if success:
                                tooltip_lines.append(f"  {plugin_name}: {decoded_value:.6f}")
                                print(f"[DetectorImagePanel] ✓ Decoded value from {plugin_name} at ({x:.1f}, {y:.1f}): {decoded_value:.6f}")
                            else:
                                print(f"[DetectorImagePanel] ✗ Failed to get decoded value from {plugin_name} at ({x:.1f}, {y:.1f})")
                        except Exception as e:
                            print(f"[DetectorImagePanel] ✗ Error getting decoded value from {plugin_name}: {e}")
                            import traceback
                            traceback.print_exc()
                
                # Add decoded values from registered decoder plugins
                if self._decoded_plugins:
                    tooltip_lines.append("<b>Decoded Values:</b>")
                    print(f"[DetectorImagePanel] Processing {len(self._decoded_plugins)} decoder plugins")
                    for plugin_name, plugin in self._decoded_plugins.items():
                        try:
                            # Get decoded value at this position
                            # Use scan coordinates (x, y) for decoder plugins
                            print(f"[DetectorImagePanel] Querying {plugin_name} at scan coordinates ({x:.1f}, {y:.1f})")
                            decoded_value, success = plugin.get_decoded_value_at_position(x, y)
                            if success:
                                tooltip_lines.append(f"  {plugin_name}: {decoded_value:.6f}")
                                print(f"[DetectorImagePanel] ✓ Decoded value from {plugin_name} at ({x:.1f}, {y:.1f}): {decoded_value:.6f}")
                            else:
                                print(f"[DetectorImagePanel] ✗ Failed to get decoded value from {plugin_name} at ({x:.1f}, {y:.1f})")
                        except Exception as e:
                            print(f"[DetectorImagePanel] ✗ Error getting decoded value from {plugin_name}: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    print(f"[DetectorImagePanel] No decoder plugins registered")
                
                tooltip_text = "<br>".join(tooltip_lines)
                print(f"[DetectorImagePanel] Tooltip text length: {len(tooltip_text)}")
                
                # Create or update persistent tooltip label
                if self._tooltip_label is None:
                    self._tooltip_label = QtWidgets.QLabel()
                    self._tooltip_label.setStyleSheet("""
                        QLabel {
                            background-color: rgba(0, 0, 0, 220);
                            color: white;
                            padding: 8px 12px;
                            border-radius: 6px;
                            font-size: 10pt;
                            border: 1px solid rgba(255, 255, 255, 150);
                        }
                    """)
                    self._tooltip_label.setWindowFlags(QtCore.Qt.WindowType.ToolTip | QtCore.Qt.WindowType.FramelessWindowHint)
                    self._tooltip_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentForMouseEvents)
                
                self._tooltip_label.setText(tooltip_text)
                self._tooltip_label.adjustSize()
                
                # Show tooltip at mouse position
                global_pos = img_view.mapToGlobal(scene_pos.toPoint())
                self._tooltip_label.move(global_pos.x(), global_pos.y())
                self._tooltip_label.show()
                self._tooltip_label.raise_()
                
                print("[DetectorImagePanel] Persistent tooltip displayed successfully")
                
            except Exception as e:
                print(f"[DetectorImagePanel] Error processing image data: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to simple tooltip using persistent label
                simple_tooltip = f"<b>Detector:</b> {detector_id}<br><b>Position:</b> X={x:.1f}, Y={y:.1f}"
                if self._tooltip_label is None:
                    self._tooltip_label = QtWidgets.QLabel()
                    self._tooltip_label.setStyleSheet("""
                        QLabel {
                            background-color: rgba(0, 0, 0, 220);
                            color: white;
                            padding: 8px 12px;
                            border-radius: 6px;
                            font-size: 10pt;
                            border: 1px solid rgba(255, 255, 255, 150);
                        }
                    """)
                    self._tooltip_label.setWindowFlags(QtCore.Qt.WindowType.ToolTip | QtCore.Qt.WindowType.FramelessWindowHint)
                    self._tooltip_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentForMouseEvents)
                
                self._tooltip_label.setText(simple_tooltip)
                self._tooltip_label.adjustSize()
                global_pos = img_view.mapToGlobal(scene_pos.toPoint())
                self._tooltip_label.move(global_pos.x(), global_pos.y())
                self._tooltip_label.show()
                self._tooltip_label.raise_()
                
        except Exception as e:
            print(f"[DetectorImagePanel] Error in click handler: {e}")
            import traceback
            traceback.print_exc()
            # Hide tooltip on error
            self._hide_tooltip()
            self._mouse_button_pressed = False
    
    def set_scan_data(self, detector_id: str, scan_data: list) -> None:
        """Store multi-axis scan data for a detector.
        
        Args:
            detector_id: The detector identifier
            scan_data: List of (position_tuple, value) tuples where position_tuple contains (x, y, z)
        """
        self._scan_data[detector_id] = scan_data
    
    def append_scan_data(self, detector_id: str, position: tuple, value: float) -> None:
        """Append a single scan data point for a detector.
        
        Args:
            detector_id: The detector identifier
            position: Position tuple (x, y, z)
            value: The detector value at this position
        """
        if detector_id not in self._scan_data:
            self._scan_data[detector_id] = []
        self._scan_data[detector_id].append((position, value))
    
    def clear_scan_data(self) -> None:
        """Clear all stored scan data."""
        self._scan_data.clear()
        self._scan_dimensions = None
    
    def set_scan_dimensions(self, scan_dimensions: dict) -> None:
        """Store scan dimensions for coordinate mapping.
        
        Args:
            scan_dimensions: Dictionary with 'x_positions', 'y_positions', 'dim_x', 'dim_y'
        """
        self._scan_dimensions = scan_dimensions
        print(f"[DetectorImagePanel] Scan dimensions set: {scan_dimensions}")
    
    def _image_coords_to_scan_coords(self, img_x: float, img_y: float) -> tuple:
        """Convert image coordinates to scan coordinates.
        
        Args:
            img_x: X coordinate in image space (array indices)
            img_y: Y coordinate in image space (array indices)
            
        Returns:
            Tuple of (scan_x, scan_y) in actual scan position units
        """
        if self._scan_dimensions is None:
            return img_x, img_y
        
        try:
            x_positions = self._scan_dimensions.get('x_positions', [])
            y_positions = self._scan_dimensions.get('y_positions', [])
            dim_x = self._scan_dimensions.get('dim_x', len(x_positions))
            dim_y = self._scan_dimensions.get('dim_y', len(y_positions))
            
            if not x_positions or not y_positions:
                return img_x, img_y
            
            # Map image coordinates to scan coordinates
            x_min = min(x_positions)
            x_max = max(x_positions)
            y_min = min(y_positions)
            y_max = max(y_positions)
            
            # Normalize image coordinates to [0, 1]
            x_norm = img_x / (dim_x - 1) if dim_x > 1 else 0
            y_norm = img_y / (dim_y - 1) if dim_y > 1 else 0
            
            # Map to scan coordinate range
            scan_x = x_min + x_norm * (x_max - x_min)
            scan_y = y_min + y_norm * (y_max - y_min)
            
            return scan_x, scan_y
            
        except Exception as e:
            print(f"[DetectorImagePanel] Error converting coordinates: {e}")
            return img_x, img_y
    
    def enable_tooltips(self, enabled: bool) -> None:
        """Enable or disable click tooltips."""
        self._tooltip_enabled = enabled
    
    def register_decoder_plugin(self, plugin_name: str, plugin) -> None:
        """Register a decoder plugin to provide decoded values for tooltips.
        
        Args:
            plugin_name: Name of the decoder plugin
            plugin: The plugin instance with get_decoded_value_at_position method
        """
        self._decoded_plugins[plugin_name] = plugin
        print(f"[DetectorImagePanel] Registered decoder plugin: {plugin_name}")
    
    def unregister_decoder_plugin(self, plugin_name: str) -> None:
        """Unregister a decoder plugin."""
        if plugin_name in self._decoded_plugins:
            del self._decoded_plugins[plugin_name]
            print(f"[DetectorImagePanel] Unregistered decoder plugin: {plugin_name}")
    
    def _on_detector_released(self) -> None:
        """Handle mouse button release to hide tooltip."""
        print("[DetectorImagePanel] Mouse button released - hiding tooltip")
        self._mouse_button_pressed = False
        self._current_tooltip_detector = None
        self._current_tooltip_position = None
        
        self._hide_tooltip()
    
    def _hide_tooltip(self) -> None:
        """Hide the tooltip if it's currently visible."""
        # Hide Qt tooltip
        QtWidgets.QToolTip.hideText()
        
        # Hide persistent tooltip label if exists
        if self._tooltip_label is not None:
            self._tooltip_label.hide()
            self._tooltip_label = None
    
    def _update_tooltip_position(self, scene_pos) -> None:
        """Update tooltip position while mouse button is held."""
        if not self._mouse_button_pressed or self._current_tooltip_detector is None:
            return
        
        try:
            # Get the detector and image view
            detector_id = self._current_tooltip_detector
            if detector_id not in self._detector_views:
                return
            
            img_view = self._detector_views[detector_id]
            
            # Update the persistent tooltip label position
            if self._tooltip_label is not None:
                global_pos = img_view.mapToGlobal(scene_pos.toPoint())
                self._tooltip_label.move(global_pos.x(), global_pos.y())
            
        except Exception as e:
            print(f"[DetectorImagePanel] Error updating tooltip position: {e}")

    # -----------------
    # public helpers
    # -----------------
    def add_detector_view(self, detector_id: str, img_view: pg.ImageView) -> None:
        """Add a per-detector ImageView with a colormap selector."""
        if detector_id in self._detector_views:
            return

        self._detectors.append(detector_id)
        self._detector_views[detector_id] = img_view

        # configure the view (keep UI compact)
        try:
            img_view.ui.roiBtn.hide()
        except Exception:
            pass
        try:
            img_view.ui.menuBtn.hide()
        except Exception:
            pass
        try:
            img_view.ui.histogram.show()
            img_view.ui.histogram.setFixedWidth(36)
        except Exception:
            pass
        try:
            img_view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            img_view.customContextMenuRequested.connect(
                lambda pos, did=detector_id, iv=img_view: self._show_detector_context_menu(did, iv, pos)
            )
        except Exception:
            pass
        
        # Also enable context menu on the container widget
        try:
            container.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            container.customContextMenuRequested.connect(
                lambda pos, did=detector_id, iv=img_view: self._show_detector_context_menu(did, iv, pos)
            )
        except Exception:
            pass
        
        # Add click handler for tooltips using multiple approaches
        print(f"[DetectorImagePanel] Setting up click handlers for {detector_id}")
        
        # Method 1: Scene click signal with proper position extraction
        try:
            def on_scene_clicked(pos, did=detector_id, iv=img_view):
                print(f"[DetectorImagePanel] Scene clicked, pos type: {type(pos)}")
                # Extract scene position from MouseClickEvent
                if hasattr(pos, 'scenePos'):
                    scene_pos = pos.scenePos()
                else:
                    scene_pos = pos
                self._on_detector_clicked(did, iv, scene_pos)
            
            img_view.scene().sigMouseClicked.connect(on_scene_clicked)
            print(f"[DetectorImagePanel] ✓ Connected scene.sigMouseClicked for {detector_id}")
        except Exception as e:
            print(f"[DetectorImagePanel] ✗ Failed scene.sigMouseClicked: {e}")
        
        # Method 2: View scene click signal with proper position extraction
        try:
            def on_view_clicked(pos, did=detector_id, iv=img_view):
                print(f"[DetectorImagePanel] View clicked, pos type: {type(pos)}")
                # Extract scene position from MouseClickEvent
                if hasattr(pos, 'scenePos'):
                    scene_pos = pos.scenePos()
                else:
                    scene_pos = pos
                self._on_detector_clicked(did, iv, scene_pos)
            
            img_view.getView().scene().sigMouseClicked.connect(on_view_clicked)
            print(f"[DetectorImagePanel] ✓ Connected view.scene.sigMouseClicked for {detector_id}")
        except Exception as e:
            print(f"[DetectorImagePanel] ✗ Failed view.scene.sigMouseClicked: {e}")
        
        # Method 3: Install event filter on the ImageView
        try:
            img_view.installEventFilter(DetectorClickFilter(detector_id, img_view, self))
            print(f"[DetectorImagePanel] ✓ Installed event filter for {detector_id}")
        except Exception as e:
            print(f"[DetectorImagePanel] ✗ Failed event filter: {e}")

        # Load per-detector preferred colormap if present; otherwise use default.
        desired = self._default_gradient
        try:
            v = self._settings.value(f"detector_cmap/{detector_id}")
            if v is not None and str(v).strip():
                desired = str(v)
        except Exception:
            pass

        self._apply_gradient_to_imageview(img_view, desired)

        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)

        title = QtWidgets.QLabel(detector_id)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        cmap_row = QtWidgets.QWidget()
        cmap_layout = QtWidgets.QHBoxLayout(cmap_row)
        cmap_layout.setContentsMargins(0, 0, 0, 0)
        cmap_layout.addWidget(QtWidgets.QLabel("Colormap:"))
        cmap_combo = QtWidgets.QComboBox()
        self._populate_gradients(cmap_combo)
        self._set_combo_text(cmap_combo, desired)
        cmap_layout.addWidget(cmap_combo)

        def _on_cmap(name: str, did=detector_id, iv=img_view):
            self._apply_gradient_to_imageview(iv, name)
            self._save_setting(f"detector_cmap/{did}", str(name))

        cmap_combo.currentTextChanged.connect(_on_cmap)

        vbox.addWidget(title)
        vbox.addWidget(cmap_row)
        vbox.addWidget(img_view)

        self._detector_widgets[detector_id] = container
        self._detector_cmap_combos[detector_id] = cmap_combo
        self.layout.addWidget(container)

        self._refresh_overlay_detector_lists()

        # apply current overlay state
        self._apply_overlay_visibility(bool(self.overlay_cb.isChecked()))

    def clear_detectors(self) -> None:
        """Remove all per-detector image views and reset tracking state."""
        for detector_id in list(self._detector_widgets.keys()):
            container = self._detector_widgets.get(detector_id)
            if container is not None:
                try:
                    self.layout.removeWidget(container)
                except Exception:
                    pass
                try:
                    container.setParent(None)
                    container.deleteLater()
                except Exception:
                    pass
        self._detectors.clear()
        self._detector_views.clear()
        self._detector_widgets.clear()
        self._detector_cmap_combos.clear()
        try:
            self._refresh_overlay_detector_lists()
        except Exception:
            pass

    def overlay_enabled(self) -> bool:
        return bool(self.overlay_cb.isChecked())

    def overlay_use_all_detectors(self) -> bool:
        return bool(self._overlay_use_all)

    def overlay_channel_map(self) -> dict[str, str | None]:
        """Return mapping {"R": det_id|None, "G": det_id|None, "B": det_id|None}."""
        out: dict[str, str | None] = {}
        for ch in ("R", "G", "B"):
            val = str(self._overlay_map.get(ch, "") or "").strip()
            out[ch] = val if val else None
        return out

    def overlay_color_mode(self) -> str:
        """Return 'fixed' or 'cmap'.

        Only relevant when overlay is enabled and 'Use all detectors' is on.
        """
        try:
            return str(getattr(self, "_overlay_color_mode", "fixed"))
        except Exception:
            return "fixed"

    def false_color_for(self, detector_id: str) -> tuple[float, float, float]:
        try:
            idx = self._detectors.index(detector_id)
        except ValueError:
            idx = 0
        return self._FALSE_COLOR_RGB[idx % len(self._FALSE_COLOR_RGB)]

    def set_overlay_image(self, rgb_image: np.ndarray) -> None:
        """Set the RGB overlay image (H,W,3) as uint8 or float."""
        try:
            self.overlay_view.setImage(rgb_image, autoLevels=False)
        except Exception:
            try:
                self.overlay_view.setImage(rgb_image)
            except Exception:
                pass

    # -----------------
    # internal
    # -----------------
    def _on_overlay_toggled(self, checked: bool) -> None:
        self._apply_overlay_visibility(bool(checked))
        try:
            self.overlay_controls.setVisible(bool(checked))
        except Exception:
            pass
        self._save_setting("overlay_enabled", 1 if checked else 0)
        try:
            self.overlay_toggled.emit(bool(checked))
        except Exception:
            pass

    def _apply_overlay_visibility(self, overlay: bool) -> None:
        try:
            self.overlay_view.setVisible(bool(overlay))
        except Exception:
            pass
        try:
            self.container.setVisible(not bool(overlay))
        except Exception:
            pass

    def _on_default_cmap_changed(self, name: str) -> None:
        self._default_gradient = str(name)
        self._save_setting("default_gradient", self._default_gradient)
        # Apply to existing views (treat this as a global default)
        for det_id, iv in list(self._detector_views.items()):
            try:
                self._apply_gradient_to_imageview(iv, self._default_gradient)
            except Exception:
                pass
            try:
                combo = self._detector_cmap_combos.get(det_id)
                if combo is not None:
                    self._set_combo_text(combo, self._default_gradient)
            except Exception:
                pass

    def _populate_gradients(self, combo: QtWidgets.QComboBox) -> None:
        names: list[str]
        try:
            from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

            names = sorted(list(Gradients.keys()))
        except Exception:
            names = [
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "grey",
            ]
        # Add simple RGB ramps (requested) even if pyqtgraph doesn't ship them.
        for extra in ["red", "green", "blue"]:
            if extra not in names:
                names.insert(0, extra)
        combo.clear()
        for n in names:
            combo.addItem(n)

    def _apply_simple_rgb_gradient(self, img_view: pg.ImageView, rgb: tuple[int, int, int]) -> bool:
        """Apply a simple black->color ramp; returns True on success."""
        r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

        # Prefer explicit ColorMap API if available.
        try:
            cmap = pg.ColorMap(
                [0.0, 1.0],
                [QtGui.QColor(0, 0, 0), QtGui.QColor(r, g, b)],
            )
            try:
                img_view.setColorMap(cmap)
                return True
            except Exception:
                # older versions might only support imageItem.setLookupTable
                lut = cmap.getLookupTable(0.0, 1.0, 256)
                item = img_view.getImageItem()
                if item is not None and hasattr(item, "setLookupTable"):
                    item.setLookupTable(lut)
                    return True
        except Exception:
            pass

        # Fallback: drive HistogramLUT gradient directly.
        try:
            hist = getattr(getattr(img_view, "ui", None), "histogram", None)
            grad = getattr(hist, "gradient", None)
            if grad is None or not hasattr(grad, "restoreState"):
                return False
            state = {
                "mode": "rgb",
                "ticks": [
                    (0.0, (0, 0, 0, 255)),
                    (1.0, (r, g, b, 255)),
                ],
            }
            grad.restoreState(state)
            return True
        except Exception:
            return False

    def _apply_gradient_to_imageview(self, img_view: pg.ImageView, gradient_name: str) -> None:
        """Apply a predefined colormap/gradient to an ImageView.

        pyqtgraph has had a few API shapes across versions; this tries the
        common paths.
        """
        name = str(gradient_name)

        simple = name.strip().lower()
        if simple in ("red", "green", "blue"):
            rgb = (255, 0, 0) if simple == "red" else (0, 255, 0) if simple == "green" else (0, 0, 255)
            if self._apply_simple_rgb_gradient(img_view, rgb):
                return
        # 1) Newer/explicit helper
        try:
            img_view.setPredefinedGradient(name)
            return
        except Exception:
            pass

        # 2) HistogramLUTWidget gradient preset
        try:
            hist = getattr(getattr(img_view, "ui", None), "histogram", None)
            grad = getattr(hist, "gradient", None)
            if grad is not None and hasattr(grad, "loadPreset"):
                grad.loadPreset(name)
                return
        except Exception:
            pass

        # 3) Restore from Gradients dict
        try:
            from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

            state = Gradients.get(name)
            hist = getattr(getattr(img_view, "ui", None), "histogram", None)
            grad = getattr(hist, "gradient", None)
            if state is not None and grad is not None and hasattr(grad, "restoreState"):
                grad.restoreState(state)
                return
        except Exception:
            pass

    def _refresh_overlay_detector_lists(self) -> None:
        """Refresh overlay detector selectors from current detector list."""
        dets = list(self._detectors)
        items = ["(none)"] + dets
        for combo, ch in [(self.overlay_r_combo, "R"), (self.overlay_g_combo, "G"), (self.overlay_b_combo, "B")]:
            try:
                combo.blockSignals(True)
                combo.clear()
                for it in items:
                    combo.addItem(it)
                # restore selection
                desired = self._overlay_map.get(ch, "") or "(none)"
                if desired and desired != "(none)":
                    idx = combo.findText(desired)
                else:
                    idx = combo.findText("(none)")
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            finally:
                try:
                    combo.blockSignals(False)
                except Exception:
                    pass

        # Enable/disable combos based on "use all"
        try:
            use_all = bool(self.overlay_all_cb.isChecked())
            self.overlay_r_combo.setEnabled(not use_all)
            self.overlay_g_combo.setEnabled(not use_all)
            self.overlay_b_combo.setEnabled(not use_all)
        except Exception:
            pass

    def _on_overlay_settings_changed(self, *_args) -> None:
        try:
            self._overlay_use_all = bool(self.overlay_all_cb.isChecked())
        except Exception:
            self._overlay_use_all = True
        self._save_setting("overlay_use_all", 1 if self._overlay_use_all else 0)
        self._refresh_overlay_detector_lists()
        try:
            self.overlay_settings_changed.emit()
        except Exception:
            pass

    def _on_overlay_color_mode_changed(self, *_args) -> None:
        try:
            mode = self.overlay_color_combo.currentData()
            mode = str(mode) if mode is not None else "fixed"
        except Exception:
            mode = "fixed"
        if mode not in ("fixed", "cmap"):
            mode = "fixed"
        self._overlay_color_mode = mode
        self._save_setting("overlay_color_mode", mode)
        try:
            self.overlay_settings_changed.emit()
        except Exception:
            pass

    def _on_overlay_channel_changed(self, ch: str) -> None:
        if ch not in ("R", "G", "B"):
            return
        combo = {"R": self.overlay_r_combo, "G": self.overlay_g_combo, "B": self.overlay_b_combo}.get(ch)
        if combo is None:
            return
        txt = str(combo.currentText() or "").strip()
        if txt == "(none)":
            txt = ""
        self._overlay_map[ch] = txt
        self._save_setting(f"overlay_map/{ch}", txt)
        try:
            self.overlay_settings_changed.emit()
        except Exception:
            pass

    def _set_combo_text(self, combo: QtWidgets.QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)
    
    def keyPressEvent(self, event):
        """Handle keyboard events - Ctrl+H hides panel, Ctrl+R reloads panel (hide and show)."""
        if event.key() == QtCore.Qt.Key.Key_H and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            # Hide panel when Ctrl+H is pressed
            parent_dock = self.parent()
            while parent_dock and not isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock = parent_dock.parent()
            
            if parent_dock and isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock.setVisible(False)
        elif event.key() == QtCore.Qt.Key.Key_R and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            # Reload panel (hide and show) when Ctrl+R is pressed
            parent_dock = self.parent()
            while parent_dock and not isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock = parent_dock.parent()
            
            if parent_dock and isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock.setVisible(False)
                parent_dock.setVisible(True)
                parent_dock.raise_()
        else:
            # Pass other key events to parent
            super().keyPressEvent(event)
