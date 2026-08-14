from __future__ import annotations

from PyQt6 import QtWidgets, QtCore, QtGui
from devices.base import StageXY, FocusZ
import json
from pathlib import Path


class PositionBlockWidget(QtWidgets.QWidget):
    """2D position visualization widget with click interaction and color-coded positions."""
    
    position_clicked = QtCore.pyqtSignal(float, float)  # x, y coordinates
    position_drag_started = QtCore.pyqtSignal(float, float)  # x, y coordinates
    position_dragged = QtCore.pyqtSignal(float, float)  # x, y coordinates
    position_drag_ended = QtCore.pyqtSignal(float, float)  # x, y coordinates
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_position = (0.5, 0.5)  # Normalized 0-1 (green)
        self.target_position = None  # Normalized 0-1 (red)
        self.old_position = None  # Normalized 0-1 (gray)
        self.x_min = 0.0
        self.x_max = 100.0
        self.y_min = 0.0
        self.y_max = 100.0
        self._is_dragging = False
        self.setMinimumSize(200, 200)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
    
    def set_position(self, x, y):
        """Set current position in real units (green marker)."""
        # Convert to normalized coordinates with offset handling
        if self.x_max > self.x_min:
            x_norm = (x - self.x_min) / (self.x_max - self.x_min)
        else:
            x_norm = 0.5
            
        if self.y_max > self.y_min:
            y_norm = (y - self.y_min) / (self.y_max - self.y_min)
        else:
            y_norm = 0.5
            
        # Clamp to valid range
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        
        # If we have a target position and it matches the new current position, clear target
        if self.target_position is not None:
            if abs(self.target_position[0] - x_norm) < 0.01 and abs(self.target_position[1] - y_norm) < 0.01:
                self.old_position = self.target_position
                self.target_position = None
        
        self.current_position = (x_norm, y_norm)
        self.update()
    
    def set_limits(self, x_min, x_max, y_min, y_max):
        """Set the limits in real units."""
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
    
    def _reset_target(self):
        """Reset target position after successful move."""
        if self.target_position is not None:
            self.old_position = self.target_position
        self.target_position = None
        self._is_dragging = False
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse click to set position."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            width = self.width()
            height = self.height()
            
            # Calculate normalized position from click
            clicked_x = event.position().x() / width
            clicked_y = event.position().y() / height
            
            # Clamp to valid range
            clicked_x = max(0.0, min(1.0, clicked_x))
            clicked_y = max(0.0, min(1.0, clicked_y))
            
            # Convert to real units using limits
            real_x = self.x_min + clicked_x * (self.x_max - self.x_min)
            real_y = self.y_min + clicked_y * (self.y_max - self.y_min)
            
            # Set as target position (red)
            self.target_position = (clicked_x, clicked_y)
            self._is_dragging = True
            
            self.position_clicked.emit(real_x, real_y)
            self.position_drag_started.emit(real_x, real_y)
            self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse drag."""
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            width = self.width()
            height = self.height()
            
            # Calculate normalized position from drag
            dragged_x = event.position().x() / width
            dragged_y = event.position().y() / height
            
            # Clamp to valid range
            dragged_x = max(0.0, min(1.0, dragged_x))
            dragged_y = max(0.0, min(1.0, dragged_y))
            
            # Convert to real units using limits
            real_x = self.x_min + dragged_x * (self.x_max - self.x_min)
            real_y = self.y_min + dragged_y * (self.y_max - self.y_min)
            
            # Update target position during drag
            self.target_position = (dragged_x, dragged_y)
            
            self.position_dragged.emit(real_x, real_y)
            self.update()
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            width = self.width()
            height = self.height()
            
            # Calculate normalized position from release
            released_x = event.position().x() / width
            released_y = event.position().y() / height
            
            # Clamp to valid range
            released_x = max(0.0, min(1.0, released_x))
            released_y = max(0.0, min(1.0, released_y))
            
            # Convert to real units using limits
            real_x = self.x_min + released_x * (self.x_max - self.x_min)
            real_y = self.y_min + released_y * (self.y_max - self.y_min)
            
            self.target_position = (released_x, released_y)
            self._is_dragging = False
            
            self.position_drag_ended.emit(real_x, real_y)
            self.update()
    
    def paintEvent(self, event):
        """Paint the position block with current position indicator and limit borders."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw background
        painter.fillRect(0, 0, width, height, QtGui.QColor(240, 240, 240))
        
        # Draw grid
        painter.setPen(QtGui.QColor(200, 200, 200))
        for i in range(1, 10):
            x = width * i / 10
            y = height * i / 10
            painter.drawLine(int(x), 0, int(x), height)
            painter.drawLine(0, int(y), width, int(y))
        
        # Draw border
        painter.setPen(QtGui.QColor(100, 100, 100))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(0, 0, width - 1, height - 1)
        
        # Draw center cross
        center_x = width / 2
        center_y = height / 2
        painter.setPen(QtGui.QColor(150, 150, 150))
        painter.drawLine(int(center_x) - 10, int(center_y), int(center_x) + 10, int(center_y))
        painter.drawLine(int(center_x), int(center_y) - 10, int(center_x), int(center_y) + 10)
        
        # Draw old position (gray)
        if self.old_position is not None:
            old_x = self.old_position[0] * width
            old_y = self.old_position[1] * height
            
            # Draw old position marker (gray)
            painter.setPen(QtGui.QColor(128, 128, 128))
            painter.setBrush(QtGui.QColor(128, 128, 128))
            painter.drawEllipse(QtCore.QPointF(old_x, old_y), 6, 6)
            
            # Draw position lines to edges (gray)
            painter.setPen(QtGui.QColor(128, 128, 128, 100))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawLine(int(old_x), int(old_y), int(old_x), height)
            painter.drawLine(int(old_x), int(old_y), width, int(old_y))
        
        # Draw target position (red)
        if self.target_position is not None:
            target_x = self.target_position[0] * width
            target_y = self.target_position[1] * height
            
            # Draw target position marker (red)
            painter.setPen(QtGui.QColor(244, 67, 54))
            painter.setBrush(QtGui.QColor(244, 67, 54))
            painter.drawEllipse(QtCore.QPointF(target_x, target_y), 8, 8)
            
            # Draw position lines to edges (red)
            painter.setPen(QtGui.QColor(244, 67, 54, 100))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawLine(int(target_x), int(target_y), int(target_x), height)
            painter.drawLine(int(target_x), int(target_y), width, int(target_y))
        
        # Draw current position (green)
        cur_x = self.current_position[0] * width
        cur_y = self.current_position[1] * height
        
        # Draw current position marker (green)
        painter.setPen(QtGui.QColor(76, 175, 80))
        painter.setBrush(QtGui.QColor(76, 175, 80))
        painter.drawEllipse(QtCore.QPointF(cur_x, cur_y), 8, 8)
        
        # Draw position lines to edges (green)
        painter.setPen(QtGui.QColor(76, 175, 80, 100))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawLine(int(cur_x), int(cur_y), int(cur_x), height)
        painter.drawLine(int(cur_x), int(cur_y), width, int(cur_y))
        
        # Draw limit boundaries (highlight the actual usable area)
        # The entire block represents the limits, so draw a subtle inner frame
        # to indicate the boundary
        painter.setPen(QtGui.QColor(50, 50, 50, 50))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(1, 1, width - 3, height - 3)
        
        # Draw limit labels
        painter.setPen(QtGui.QColor(80, 80, 80))
        painter.setFont(QtGui.QFont("Arial", 8))
        
        # X limits labels
        painter.drawText(5, height - 5, f"{self.x_min:.1f}")
        painter.drawText(width - 40, height - 5, f"{self.x_max:.1f}")
        
        # Y limits labels
        painter.drawText(5, 12, f"{self.y_max:.1f}")
        painter.drawText(5, height - 12, f"{self.y_min:.1f}")
        
        painter.end()


class StageControlTab(QtWidgets.QWidget):
    """Stage control panel as a dockable tab with real units and slider control."""
    
    position_changed = QtCore.pyqtSignal(float, float, float)  # x, y, z in real units
    
    def __init__(self, stage: StageXY = None, focus: FocusZ = None, config_path: str = None, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.focus = focus
        self.config_path = config_path
        self._is_live_mode = False  # Default: live mode OFF
        self._is_dragging = False  # Track dragging state
        self._load_config()
        self._build_ui()
        self._setup_position_timer()
        
    def _load_config(self):
        """Load stage and focus configuration for units and limits."""
        self.stage_config = {
            'x_scale': 1.0,
            'x_offset': 0.0,
            'y_scale': 1.0,
            'y_offset': 0.0,
            'x_min': None,
            'x_max': None,
            'y_min': None,
            'y_max': None,
            'unit': 'mm'
        }
        
        self.focus_config = {
            'scale': 1.0,
            'offset': 0.0,
            'min': None,
            'max': None,
            'unit': 'mm'
        }
        
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load stage config
                if 'stage' in config:
                    stage_cfg = config['stage']
                    if 'scaling' in stage_cfg:
                        self.stage_config['x_scale'] = stage_cfg['scaling'].get('x_scale', 1.0)
                        self.stage_config['x_offset'] = stage_cfg['scaling'].get('x_offset', 0.0)
                        self.stage_config['y_scale'] = stage_cfg['scaling'].get('y_scale', 1.0)
                        self.stage_config['y_offset'] = stage_cfg['scaling'].get('y_offset', 0.0)
                    
                    if 'range' in stage_cfg:
                        self.stage_config['x_min'] = stage_cfg['range'].get('x_min')
                        self.stage_config['x_max'] = stage_cfg['range'].get('x_max')
                        self.stage_config['y_min'] = stage_cfg['range'].get('y_min')
                        self.stage_config['y_max'] = stage_cfg['range'].get('y_max')
                
                # Load focus config
                if 'focus' in config:
                    focus_cfg = config['focus']
                    if 'scaling' in focus_cfg:
                        self.focus_config['scale'] = focus_cfg['scaling'].get('scale', 1.0)
                        self.focus_config['offset'] = focus_cfg['scaling'].get('offset', 0.0)
            
            except Exception as e:
                print(f"Error loading config: {e}")
    
    def _steps_to_real_units(self, steps: float, scale: float, offset: float) -> float:
        """Convert steps to real units."""
        return steps * scale + offset
    
    def _real_units_to_steps(self, real_units: float, scale: float, offset: float) -> float:
        """Convert real units to steps."""
        if scale == 0:
            return 0.0
        return (real_units - offset) / scale
        
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)  # Reduced spacing
        layout.setContentsMargins(8, 8, 8, 8)  # Reduced margins
        
        # Device status indicator
        self.device_status_label = QtWidgets.QLabel("⚠ No devices loaded")
        self.device_status_label.setStyleSheet("background-color: #FFF3E0; color: #E65100; padding: 6px; border-radius: 4px; font-weight: bold;")
        self.device_status_label.setWordWrap(True)
        layout.addWidget(self.device_status_label)
        
        # Live mode switch
        control_row = QtWidgets.QHBoxLayout()
        self.live_switch = QtWidgets.QCheckBox("Live Mode")
        self.live_switch.setChecked(False)  # Default: OFF
        self.live_switch.setStyleSheet("QCheckBox { font-weight: bold; padding: 4px; }")
        self.live_switch.toggled.connect(self._on_live_mode_toggled)
        self.live_switch.setEnabled(False)  # Disabled until devices are loaded
        control_row.addWidget(self.live_switch)
        control_row.addStretch()
        layout.addLayout(control_row)
        
        # 2D Position visualization with sliders around it
        position_group = QtWidgets.QGroupBox("Stage Position")
        position_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        position_layout = QtWidgets.QGridLayout()
        position_layout.setSpacing(4)
        position_layout.setContentsMargins(8, 8, 8, 8)
        
        # Get limits for labels and tooltip
        x_min = self.stage_config['x_min'] if self.stage_config['x_min'] is not None else 0.0
        x_max = self.stage_config['x_max'] if self.stage_config['x_max'] is not None else 100.0
        y_min = self.stage_config['y_min'] if self.stage_config['y_min'] is not None else 0.0
        y_max = self.stage_config['y_max'] if self.stage_config['y_max'] is not None else 100.0
        
        # Position block in center (row 1, col 1)
        self.position_block = PositionBlockWidget()
        self.position_block.set_limits(x_min, x_max, y_min, y_max)
        self.position_block.setMinimumSize(200, 200)  # Match slider dimensions: 200x200
        self.position_block.setMaximumSize(200, 200)  # Fixed size to match sliders
        self.position_block.position_clicked.connect(self._on_position_block_clicked)
        self.position_block.position_drag_started.connect(self._on_drag_started)
        self.position_block.position_dragged.connect(self._on_dragged)
        self.position_block.position_drag_ended.connect(self._on_drag_ended)
        position_layout.addWidget(self.position_block, 1, 1, 1, 1)
        
        # Y slider to the left of the block (row 1, col 0) - vertical, exactly 200px height
        y_slider_container = QtWidgets.QWidget()
        y_slider_layout = QtWidgets.QVBoxLayout(y_slider_container)
        y_slider_layout.setContentsMargins(0, 0, 0, 0)
        y_slider_layout.setSpacing(4)
        
        # Y max label at top
        y_max_label = QtWidgets.QLabel(f"{y_max:.1f}")
        y_max_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        y_max_label.setStyleSheet("font-size: 9px; color: #666;")
        y_slider_layout.addWidget(y_max_label)
        
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setRange(-1e6, 1e6)
        self.y_spin.setDecimals(3)
        self.y_spin.setSuffix(f" {self.stage_config['unit']}")
        self.y_spin.setValue(0.0)
        self.y_spin.setMaximumWidth(100)
        
        self.y_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.y_slider.setRange(0, 1000)
        self.y_slider.setValue(0)
        self.y_slider.setFixedHeight(200)  # Exactly 200px to match position block height
        self.y_slider.setToolTip(f"Y range: {y_min:.1f} to {y_max:.1f} {self.stage_config['unit']}")
        
        y_slider_layout.addWidget(self.y_spin)
        y_slider_layout.addWidget(self.y_slider)
        
        # Y min label at bottom
        y_min_label = QtWidgets.QLabel(f"{y_min:.1f}")
        y_min_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        y_min_label.setStyleSheet("font-size: 9px; color: #666;")
        y_slider_layout.addWidget(y_min_label)
        
        position_layout.addWidget(y_slider_container, 1, 0, 1, 1)
        
        # X slider below the block (row 2, col 1) - horizontal, exactly 200px width
        x_slider_container = QtWidgets.QWidget()
        x_slider_layout = QtWidgets.QVBoxLayout(x_slider_container)
        x_slider_layout.setContentsMargins(0, 0, 0, 0)
        x_slider_layout.setSpacing(4)
        
        self.x_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.x_slider.setRange(0, 1000)
        self.x_slider.setValue(0)
        self.x_slider.setFixedWidth(200)  # Exactly 200px to match position block width
        self.x_slider.setToolTip(f"X range: {x_min:.1f} to {x_max:.1f} {self.stage_config['unit']}")
        
        # X limits labels row
        x_limits_row = QtWidgets.QWidget()
        x_limits_layout = QtWidgets.QHBoxLayout(x_limits_row)
        x_limits_layout.setContentsMargins(0, 0, 0, 0)
        x_limits_layout.setSpacing(0)
        
        x_min_label = QtWidgets.QLabel(f"{x_min:.1f}")
        x_min_label.setStyleSheet("font-size: 9px; color: #666;")
        x_limits_layout.addWidget(x_min_label)
        
        x_limits_layout.addStretch()
        
        x_max_label = QtWidgets.QLabel(f"{x_max:.1f}")
        x_max_label.setStyleSheet("font-size: 9px; color: #666;")
        x_limits_layout.addWidget(x_max_label)
        
        x_slider_layout.addWidget(self.x_slider)
        x_slider_layout.addWidget(x_limits_row)
        
        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setRange(-1e6, 1e6)
        self.x_spin.setDecimals(3)
        self.x_spin.setSuffix(f" {self.stage_config['unit']}")
        self.x_spin.setValue(0.0)
        self.x_spin.setMaximumWidth(100)
        
        x_slider_layout.addWidget(self.x_spin)
        
        position_layout.addWidget(x_slider_container, 2, 1, 1, 1)
        
        # Labels for axes
        y_label = QtWidgets.QLabel("Y:")
        y_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        position_layout.addWidget(y_label, 0, 0, 1, 1)
        
        x_label = QtWidgets.QLabel("X:")
        x_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        position_layout.addWidget(x_label, 2, 0, 1, 1)
        
        position_group.setLayout(position_layout)
        layout.addWidget(position_group)
        
        # Focus Z control in compact form
        focus_group = QtWidgets.QGroupBox("Focus Position")
        focus_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        focus_layout = QtWidgets.QFormLayout()
        focus_layout.setSpacing(4)
        focus_layout.setContentsMargins(8, 8, 8, 8)
        
        # Z position with slider
        z_container = QtWidgets.QWidget()
        z_layout = QtWidgets.QHBoxLayout(z_container)
        z_layout.setContentsMargins(0, 0, 0, 0)
        z_layout.setSpacing(4)
        
        self.z_spin = QtWidgets.QDoubleSpinBox()
        self.z_spin.setRange(-1e6, 1e6)
        self.z_spin.setDecimals(3)
        self.z_spin.setSuffix(f" {self.focus_config['unit']}")
        self.z_spin.setValue(0.0)
        self.z_spin.setMaximumWidth(100)
        
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_slider.setRange(0, 1000)
        self.z_slider.setValue(0)
        self.z_slider.setMinimumWidth(300)  # 3x longer than default
        
        z_layout.addWidget(self.z_spin)
        z_layout.addWidget(self.z_slider)
        
        focus_layout.addRow("Z:", z_container)
        
        focus_group.setLayout(focus_layout)
        layout.addWidget(focus_group)
        
        # Move button - compact
        self.move_btn = QtWidgets.QPushButton("Move")
        self.move_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 6px;")
        self.move_btn.setMaximumHeight(30)
        layout.addWidget(self.move_btn)
        
        # Current position display - all in one line
        pos_group = QtWidgets.QGroupBox("Current Position")
        pos_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        pos_layout = QtWidgets.QHBoxLayout()
        pos_layout.setSpacing(8)
        pos_layout.setContentsMargins(8, 8, 8, 8)
        
        self.current_x_label = QtWidgets.QLabel("X: -")
        self.current_x_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        self.current_y_label = QtWidgets.QLabel("Y: -")
        self.current_y_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        self.current_z_label = QtWidgets.QLabel("Z: -")
        self.current_z_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        
        pos_layout.addWidget(self.current_x_label)
        pos_layout.addWidget(self.current_y_label)
        pos_layout.addWidget(self.current_z_label)
        pos_layout.addStretch()
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Refresh button - compact
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.refresh_btn.setMaximumHeight(25)
        layout.addWidget(self.refresh_btn)
        
        layout.addStretch(1)
        
        # Connect signals
        self.move_btn.clicked.connect(self._move_to_position)
        self.refresh_btn.clicked.connect(self._refresh_position)
        
        # Connect spinbox-slider synchronization
        self.x_spin.valueChanged.connect(self._on_x_spin_changed)
        self.x_slider.valueChanged.connect(self._on_x_slider_changed)
        self.y_spin.valueChanged.connect(self._on_y_spin_changed)
        self.y_slider.valueChanged.connect(self._on_y_slider_changed)
        self.z_spin.valueChanged.connect(self._on_z_spin_changed)
        self.z_slider.valueChanged.connect(self._on_z_slider_changed)
        
        # Set slider limits based on config
        self._update_slider_limits()
    
    def _on_live_mode_toggled(self, checked):
        """Handle live mode toggle."""
        self._is_live_mode = checked
        
        # Check if devices are available
        has_devices = self.stage is not None or self.focus is not None
        
        if checked:
            # Live mode: enable auto-refresh, hide move button
            if has_devices:
                self.position_timer.start(500)
                self.move_btn.setEnabled(False)
                self.move_btn.setText("Live Active")
                self.refresh_btn.setEnabled(False)
            else:
                # No devices, disable live mode
                self.live_switch.setChecked(False)
                QtWidgets.QMessageBox.warning(self, "No Devices", 
                    "Cannot enable Live Mode without loaded devices.")
        else:
            # Normal mode: disable auto-refresh, show move button
            self.position_timer.stop()
            if has_devices:
                self.move_btn.setEnabled(True)
                self.move_btn.setText("Move")
            self.refresh_btn.setEnabled(True)
    
    def _on_drag_started(self, x, y):
        """Handle drag start in live mode."""
        if self._is_live_mode:
            self._is_dragging = True
            # Move stage immediately
            self._move_to_position_xy(x, y)
    
    def _on_dragged(self, x, y):
        """Handle drag movement in live mode."""
        if self._is_live_mode and self._is_dragging:
            # Move stage continuously during drag
            self._move_to_position_xy(x, y)
        elif not self._is_live_mode:
            # In normal mode, just update UI during drag
            self.x_spin.blockSignals(True)
            self.y_spin.blockSignals(True)
            self.x_spin.setValue(x)
            self.y_spin.setValue(y)
            self.x_spin.blockSignals(False)
            self.y_spin.blockSignals(False)
            self._update_sliders_from_spinboxes()
            # Position block target is already updated by the widget itself
    
    def _on_drag_ended(self, x, y):
        """Handle drag end in live mode."""
        if self._is_live_mode:
            self._is_dragging = False
            # Final move to position
            self._move_to_position_xy(x, y)
        elif not self._is_live_mode:
            # In normal mode, drag ended - just ensure UI is updated
            self.x_spin.blockSignals(True)
            self.y_spin.blockSignals(True)
            self.x_spin.setValue(x)
            self.y_spin.setValue(y)
            self.x_spin.blockSignals(False)
            self.y_spin.blockSignals(False)
            self._update_sliders_from_spinboxes()
            # Position block target is already updated by the widget itself
    
    def _move_to_position_xy(self, x, y):
        """Move stage to specific X,Y position."""
        try:
            # Convert real units to steps
            x_steps = self._real_units_to_steps(x, self.stage_config['x_scale'], self.stage_config['x_offset'])
            y_steps = self._real_units_to_steps(y, self.stage_config['y_scale'], self.stage_config['y_offset'])
            
            if self.stage and hasattr(self.stage, 'move_to'):
                self.stage.move_to(x_steps, y_steps)
                
            # Update UI
            self.x_spin.blockSignals(True)
            self.y_spin.blockSignals(True)
            self.x_spin.setValue(x)
            self.y_spin.setValue(y)
            self.x_spin.blockSignals(False)
            self.y_spin.blockSignals(False)
            
            self._update_sliders_from_spinboxes()
            self._auto_refresh_position()
            
        except Exception as e:
            # Silently fail during live mode to avoid blocking
            pass
    
    def _on_position_block_clicked(self, x, y):
        """Handle position block click - update sliders and spinboxes."""
        # Update spinboxes
        self.x_spin.blockSignals(True)
        self.y_spin.blockSignals(True)
        self.x_spin.setValue(x)
        self.y_spin.setValue(y)
        self.x_spin.blockSignals(False)
        self.y_spin.blockSignals(False)
        
        # Update sliders
        self._update_sliders_from_spinboxes()
        
        # Update position block target (already updated by the widget itself)
        # No need to call set_position as widget handles the target position
    
    def _setup_position_timer(self):
        """Set up automatic position reading timer (500ms interval)."""
        self.position_timer = QtCore.QTimer()
        self.position_timer.timeout.connect(self._auto_refresh_position)
        # Don't start timer by default (normal mode = no auto-refresh)
        # Timer will be started when live mode is enabled
    
    def _auto_refresh_position(self):
        """Auto-refresh position without blocking UI."""
        try:
            x_real, y_real, z_real = 0.0, 0.0, 0.0
            
            if self.stage and hasattr(self.stage, 'get_position'):
                pos = self.stage.get_position()
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    x_steps, y_steps = float(pos[0]), float(pos[1])
                else:
                    x_steps = float(pos) if pos is not None else 0.0
                    y_steps = 0.0
                    
                # Convert steps to real units
                x_real = self._steps_to_real_units(x_steps, self.stage_config['x_scale'], self.stage_config['x_offset'])
                y_real = self._steps_to_real_units(y_steps, self.stage_config['y_scale'], self.stage_config['y_offset'])
                    
            if self.focus and hasattr(self.focus, 'get_position'):
                z_steps = float(self.focus.get_position()) if self.focus.get_position() is not None else 0.0
                z_real = self._steps_to_real_units(z_steps, self.focus_config['scale'], self.focus_config['offset'])
                
            self.current_x_label.setText(f"X: {x_real:.3f} {self.stage_config['unit']}")
            self.current_y_label.setText(f"Y: {y_real:.3f} {self.stage_config['unit']}")
            self.current_z_label.setText(f"Z: {z_real:.3f} {self.focus_config['unit']}")
            
            # Update position block visualization
            self.position_block.set_position(x_real, y_real)
            
            # Only update spinboxes and sliders in live mode
            if self._is_live_mode:
                self.x_spin.blockSignals(True)
                self.y_spin.blockSignals(True)
                self.z_spin.blockSignals(True)
                self.x_spin.setValue(x_real)
                self.y_spin.setValue(y_real)
                self.z_spin.setValue(z_real)
                self.x_spin.blockSignals(False)
                self.y_spin.blockSignals(False)
                self.z_spin.blockSignals(False)
                
                self._update_sliders_from_spinboxes()
            
            self.position_changed.emit(x_real, y_real, z_real)
            
        except Exception:
            # Silently fail on auto-refresh to avoid UI blocking
            pass
    
    def _update_slider_limits(self):
        """Update slider limits based on configuration and sync with position block."""
        # Get range values from config
        x_min = self.stage_config['x_min'] if self.stage_config['x_min'] is not None else 0.0
        x_max = self.stage_config['x_max'] if self.stage_config['x_max'] is not None else 100.0
        y_min = self.stage_config['y_min'] if self.stage_config['y_min'] is not None else 0.0
        y_max = self.stage_config['y_max'] if self.stage_config['y_max'] is not None else 100.0
        
        # X slider limits - use 0-1000 range normalized
        self.x_spin.setRange(x_min, x_max)
        self.x_slider.setRange(0, 1000)  # Normalized 0-1000 range
        
        # Y slider limits - use 0-1000 range normalized
        self.y_spin.setRange(y_min, y_max)
        self.y_slider.setRange(0, 1000)  # Normalized 0-1000 range
        
        # Z slider limits - use 0-1000 range normalized
        if self.focus_config['min'] is not None and self.focus_config['max'] is not None:
            self.z_spin.setRange(self.focus_config['min'], self.focus_config['max'])
            self.z_slider.setRange(0, 1000)  # Normalized 0-1000 range
        
        # Sync position block limits with slider limits
        if hasattr(self, 'position_block'):
            self.position_block.set_limits(x_min, x_max, y_min, y_max)
    
    def _on_x_spin_changed(self, value):
        """Handle X spinbox change."""
        # Get limits for normalization
        x_min = self.stage_config['x_min'] if self.stage_config['x_min'] is not None else 0.0
        x_max = self.stage_config['x_max'] if self.stage_config['x_max'] is not None else 100.0
        
        # Convert to normalized 0-1000 range
        x_norm = int(((value - x_min) / (x_max - x_min)) * 1000) if x_max > x_min else 500
        
        self.x_slider.blockSignals(True)
        self.x_slider.setValue(x_norm)
        self.x_slider.blockSignals(False)
        
        # Update position block target if in normal mode
        if not self._is_live_mode:
            x_val = self.x_spin.value()
            y_val = self.y_spin.value()
            # Get current limits for normalization
            y_min = self.stage_config['y_min'] if self.stage_config['y_min'] is not None else 0.0
            y_max = self.stage_config['y_max'] if self.stage_config['y_max'] is not None else 100.0
            # Convert to normalized coordinates for position block
            x_norm_block = (x_val - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            y_norm_block = (y_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
            self.position_block.target_position = (x_norm_block, y_norm_block)
            self.position_block.update()
    
    def _on_x_slider_changed(self, value):
        """Handle X slider change."""
        # Get limits for conversion
        x_min = self.stage_config['x_min'] if self.stage_config['x_min'] is not None else 0.0
        x_max = self.stage_config['x_max'] if self.stage_config['x_max'] is not None else 100.0
        
        # Convert from normalized 0-1000 range to real units
        x_real = x_min + (value / 1000.0) * (x_max - x_min)
        
        self.x_spin.blockSignals(True)
        self.x_spin.setValue(x_real)
        self.x_spin.blockSignals(False)
        
        # Update position block target if in normal mode
        if not self._is_live_mode:
            x_val = self.x_spin.value()
            y_val = self.y_spin.value()
            # Get current limits for normalization
            y_min = self.stage_config['y_min'] if self.stage_config['y_min'] is not None else 0.0
            y_max = self.stage_config['y_max'] if self.stage_config['y_max'] is not None else 100.0
            # Convert to normalized coordinates for position block
            x_norm = (x_val - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            y_norm = (y_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
            self.position_block.target_position = (x_norm, y_norm)
            self.position_block.update()
        
        # In live mode, move stage immediately
        if self._is_live_mode:
            self._move_to_position_xy(self.x_spin.value(), self.y_spin.value())
    
    def _on_y_spin_changed(self, value):
        """Handle Y spinbox change."""
        # Get limits for normalization
        y_min = self.stage_config['y_min'] if self.stage_config['y_min'] is not None else 0.0
        y_max = self.stage_config['y_max'] if self.stage_config['y_max'] is not None else 100.0
        
        # Convert to normalized 0-1000 range
        y_norm = int(((value - y_min) / (y_max - y_min)) * 1000) if y_max > y_min else 500
        
        self.y_slider.blockSignals(True)
        self.y_slider.setValue(y_norm)
        self.y_slider.blockSignals(False)
        
        # Update position block target if in normal mode
        if not self._is_live_mode:
            x_val = self.x_spin.value()
            y_val = self.y_spin.value()
            # Get current limits for normalization
            x_min = self.stage_config['x_min'] if self.stage_config['x_min'] is not None else 0.0
            x_max = self.stage_config['x_max'] if self.stage_config['x_max'] is not None else 100.0
            # Convert to normalized coordinates for position block
            x_norm_block = (x_val - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            y_norm_block = (y_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
            self.position_block.target_position = (x_norm_block, y_norm_block)
            self.position_block.update()
    
    def _on_y_slider_changed(self, value):
        """Handle Y slider change."""
        # Get limits for conversion
        y_min = self.stage_config['y_min'] if self.stage_config['y_min'] is not None else 0.0
        y_max = self.stage_config['y_max'] if self.stage_config['y_max'] is not None else 100.0
        
        # Convert from normalized 0-1000 range to real units
        y_real = y_min + (value / 1000.0) * (y_max - y_min)
        
        self.y_spin.blockSignals(True)
        self.y_spin.setValue(y_real)
        self.y_spin.blockSignals(False)
        
        # Update position block target if in normal mode
        if not self._is_live_mode:
            x_val = self.x_spin.value()
            y_val = self.y_spin.value()
            # Get current limits for normalization
            x_min = self.stage_config['x_min'] if self.stage_config['x_min'] is not None else 0.0
            x_max = self.stage_config['x_max'] if self.stage_config['x_max'] is not None else 100.0
            # Convert to normalized coordinates for position block
            x_norm = (x_val - x_min) / (x_max - x_min) if x_max > x_min else 0.5
            y_norm = (y_val - y_min) / (y_max - y_min) if y_max > y_min else 0.5
            self.position_block.target_position = (x_norm, y_norm)
            self.position_block.update()
        
        # In live mode, move stage immediately
        if self._is_live_mode:
            self._move_to_position_xy(self.x_spin.value(), self.y_spin.value())
    
    def _on_z_spin_changed(self, value):
        """Handle Z spinbox change."""
        # Get limits for normalization
        z_min = self.focus_config['min'] if self.focus_config['min'] is not None else 0.0
        z_max = self.focus_config['max'] if self.focus_config['max'] is not None else 100.0
        
        # Convert to normalized 0-1000 range
        z_norm = int(((value - z_min) / (z_max - z_min)) * 1000) if z_max > z_min else 500
        
        self.z_slider.blockSignals(True)
        self.z_slider.setValue(z_norm)
        self.z_slider.blockSignals(False)
    
    def _on_z_slider_changed(self, value):
        """Handle Z slider change."""
        # Get limits for conversion
        z_min = self.focus_config['min'] if self.focus_config['min'] is not None else 0.0
        z_max = self.focus_config['max'] if self.focus_config['max'] is not None else 100.0
        
        # Convert from normalized 0-1000 range to real units
        z_real = z_min + (value / 1000.0) * (z_max - z_min)
        
        self.z_spin.blockSignals(True)
        self.z_spin.setValue(z_real)
        self.z_spin.blockSignals(False)
    
    def _update_sliders_from_spinboxes(self):
        """Update sliders from current spinbox values using normalized coordinates."""
        x_val = self.x_spin.value()
        y_val = self.y_spin.value()
        z_val = self.z_spin.value()
        
        # Get limits for normalization
        x_min = self.stage_config['x_min'] if self.stage_config['x_min'] is not None else 0.0
        x_max = self.stage_config['x_max'] if self.stage_config['x_max'] is not None else 100.0
        y_min = self.stage_config['y_min'] if self.stage_config['y_min'] is not None else 0.0
        y_max = self.stage_config['y_max'] if self.stage_config['y_max'] is not None else 100.0
        z_min = self.focus_config['min'] if self.focus_config['min'] is not None else 0.0
        z_max = self.focus_config['max'] if self.focus_config['max'] is not None else 100.0
        
        # Convert to normalized 0-1000 range
        x_norm = int(((x_val - x_min) / (x_max - x_min)) * 1000) if x_max > x_min else 500
        y_norm = int(((y_val - y_min) / (y_max - y_min)) * 1000) if y_max > y_min else 500
        z_norm = int(((z_val - z_min) / (z_max - z_min)) * 1000) if z_max > z_min else 500
        
        self.x_slider.blockSignals(True)
        self.y_slider.blockSignals(True)
        self.z_slider.blockSignals(True)
        
        self.x_slider.setValue(x_norm)
        self.y_slider.setValue(y_norm)
        self.z_slider.setValue(z_norm)
        
        self.x_slider.blockSignals(False)
        self.y_slider.blockSignals(False)
        self.z_slider.blockSignals(False)
        
    def _move_to_position(self):
        """Move stage and focus to the specified positions using real units."""
        try:
            # Check if devices are available
            if self.stage is None and self.focus is None:
                QtWidgets.QMessageBox.warning(self, "No Devices", 
                    "No stage or focus devices available. Please load hardware configuration first.")
                return
            
            x_real = self.x_spin.value()
            y_real = self.y_spin.value()
            z_real = self.z_spin.value()
            
            print(f"DEBUG: Moving to X={x_real}, Y={y_real}, Z={z_real}")
            print(f"DEBUG: Stage exists: {self.stage is not None}")
            print(f"DEBUG: Focus exists: {self.focus is not None}")
            
            # Convert real units to steps
            x_steps = self._real_units_to_steps(x_real, self.stage_config['x_scale'], self.stage_config['x_offset'])
            y_steps = self._real_units_to_steps(y_real, self.stage_config['y_scale'], self.stage_config['y_offset'])
            z_steps = self._real_units_to_steps(z_real, self.focus_config['scale'], self.focus_config['offset'])
            
            print(f"DEBUG: Steps - X={x_steps}, Y={y_steps}, Z={z_steps}")
            
            if self.stage and hasattr(self.stage, 'move_to'):
                print(f"DEBUG: Calling stage.move_to({x_steps}, {y_steps})")
                self.stage.move_to(x_steps, y_steps)
                print(f"DEBUG: Stage move completed")
            else:
                print(f"DEBUG: Stage move_to not available")
                
            if self.focus and hasattr(self.focus, 'move_to'):
                print(f"DEBUG: Calling focus.move_to({z_steps})")
                self.focus.move_to(z_steps)
                print(f"DEBUG: Focus move completed")
            else:
                print(f"DEBUG: Focus move_to not available")
                
            # Reset target position after successful move (in normal mode)
            if not self._is_live_mode:
                self.position_block._reset_target()
                
            self._refresh_position()
            
        except Exception as e:
            print(f"DEBUG: Move error: {e}")
            import traceback
            traceback.print_exc()
            QtWidgets.QMessageBox.warning(self, "Move Error", f"Failed to move: {e}")
    
    def _refresh_position(self):
        """Refresh the current position display in real units (manual refresh)."""
        try:
            x_real, y_real, z_real = 0.0, 0.0, 0.0
            
            print(f"DEBUG: Refreshing position...")
            print(f"DEBUG: Stage exists: {self.stage is not None}")
            print(f"DEBUG: Focus exists: {self.focus is not None}")
            
            if self.stage and hasattr(self.stage, 'get_position'):
                pos = self.stage.get_position()
                print(f"DEBUG: Stage position raw: {pos}")
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    x_steps, y_steps = float(pos[0]), float(pos[1])
                else:
                    x_steps = float(pos) if pos is not None else 0.0
                    y_steps = 0.0
                    
                # Convert steps to real units
                x_real = self._steps_to_real_units(x_steps, self.stage_config['x_scale'], self.stage_config['x_offset'])
                y_real = self._steps_to_real_units(y_steps, self.stage_config['y_scale'], self.stage_config['y_offset'])
                print(f"DEBUG: Stage position real: X={x_real}, Y={y_real}")
                    
            if self.focus and hasattr(self.focus, 'get_position'):
                z_steps = float(self.focus.get_position()) if self.focus.get_position() is not None else 0.0
                z_real = self._steps_to_real_units(z_steps, self.focus_config['scale'], self.focus_config['offset'])
                print(f"DEBUG: Focus position real: Z={z_real}")
                
            self.current_x_label.setText(f"X: {x_real:.3f} {self.stage_config['unit']}")
            self.current_y_label.setText(f"Y: {y_real:.3f} {self.stage_config['unit']}")
            self.current_z_label.setText(f"Z: {z_real:.3f} {self.focus_config['unit']}")
            
            # Update position block visualization
            self.position_block.set_position(x_real, y_real)
            
            # Update spinboxes and sliders
            self.x_spin.blockSignals(True)
            self.y_spin.blockSignals(True)
            self.z_spin.blockSignals(True)
            self.x_spin.setValue(x_real)
            self.y_spin.setValue(y_real)
            self.z_spin.setValue(z_real)
            self.x_spin.blockSignals(False)
            self.y_spin.blockSignals(False)
            self.z_spin.blockSignals(False)
            
            self._update_sliders_from_spinboxes()
            
            self.position_changed.emit(x_real, y_real, z_real)
            
        except Exception as e:
            print(f"DEBUG: Refresh error: {e}")
            import traceback
            traceback.print_exc()
            self.current_x_label.setText("Error")
            self.current_y_label.setText("Error")
            self.current_z_label.setText("Error")
    
    def set_stage(self, stage: StageXY):
        """Set the stage device and refresh position."""
        self.stage = stage
        self._update_device_status()
        # Immediately refresh position after loading hardware
        self._refresh_position()
    
    def set_focus(self, focus: FocusZ):
        """Set the focus device and refresh position."""
        self.focus = focus
        self._update_device_status()
        # Immediately refresh position after loading hardware
        self._refresh_position()
    
    def _update_device_status(self):
        """Update device status indicator and enable/disable controls."""
        has_stage = self.stage is not None
        has_focus = self.focus is not None
        
        if has_stage or has_focus:
            devices = []
            if has_stage:
                devices.append("Stage")
            if has_focus:
                devices.append("Focus")
            self.device_status_label.setText(f"✓ Devices loaded: {', '.join(devices)}")
            self.device_status_label.setStyleSheet("background-color: #E8F5E9; color: #2E7D32; padding: 6px; border-radius: 4px; font-weight: bold;")
            self.live_switch.setEnabled(True)
            self.move_btn.setEnabled(not self._is_live_mode)
        else:
            self.device_status_label.setText("⚠ No devices loaded")
            self.device_status_label.setStyleSheet("background-color: #FFF3E0; color: #E65100; padding: 6px; border-radius: 4px; font-weight: bold;")
            self.live_switch.setEnabled(False)
            self.move_btn.setEnabled(False)
    
    def set_config_path(self, config_path: str):
        """Set the config file path and reload configuration."""
        self.config_path = config_path
        self._load_config()
        self._update_slider_limits()
        # Refresh position after config reload to ensure display is correct
        self._refresh_position()
    
    def cleanup(self):
        """Clean up resources when widget is destroyed."""
        if hasattr(self, 'position_timer'):
            self.position_timer.stop()
            self.position_timer.deleteLater()