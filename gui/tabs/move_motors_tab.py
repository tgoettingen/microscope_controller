from __future__ import annotations

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import pyqtSlot, pyqtSignal
from devices.base import StageXY, FocusZ
import json
from pathlib import Path

# Constants for sizing
POSITION_BLOCK_SIZE = 250  # Size of the 2D position visualization widget
SLIDER_RANGE = 1000  # Normalized slider range
AUTO_REFRESH_INTERVAL_MS = 500  # Live mode refresh interval in milliseconds
SLIDER_RANGE = 1000  # Normalized slider range
AUTO_REFRESH_INTERVAL_MS = 500  # Auto-refresh interval in milliseconds


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
        self.aspect_ratio = None  # Aspect ratio constraint (width/height)
        self.y_axis_swapped = False  # Y axis swapped flag
        self.setMinimumSize(POSITION_BLOCK_SIZE, POSITION_BLOCK_SIZE)
        self.setCursor(QtCore.Qt.CursorShape.CrossCursor)
    
    def set_aspect_ratio(self, ratio):
        """Set aspect ratio constraint. None = no constraint."""
        self.aspect_ratio = ratio
        self.update()  # Redraw with new aspect ratio
    
    def set_y_axis_swapped(self, swapped):
        """Set Y axis swapped flag for label display."""
        self.y_axis_swapped = swapped
        self.update()  # Redraw with updated labels
    
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
            
            # Calculate drawing area respecting aspect ratio
            if self.aspect_ratio is not None and self.aspect_ratio > 0:
                container_ratio = width / height
                if container_ratio > self.aspect_ratio:
                    # Container is wider than desired, constrain by height
                    draw_height = height
                    draw_width = draw_height * self.aspect_ratio
                    x_offset = 0
                    y_offset = 0
                else:
                    # Container is taller than desired, constrain by width
                    draw_width = width
                    draw_height = draw_width / self.aspect_ratio
                    x_offset = 0
                    y_offset = 0
            else:
                # No aspect ratio constraint, use full widget
                draw_width = width
                draw_height = height
                x_offset = 0
                y_offset = 0
            
            # Calculate normalized position from click (relative to drawing area)
            clicked_x = (event.position().x() - x_offset) / draw_width
            clicked_y = (event.position().y() - y_offset) / draw_height
            
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
            
            # Calculate drawing area respecting aspect ratio
            if self.aspect_ratio is not None and self.aspect_ratio > 0:
                container_ratio = width / height
                if container_ratio > self.aspect_ratio:
                    # Container is wider than desired, constrain by height
                    draw_height = height
                    draw_width = draw_height * self.aspect_ratio
                    x_offset = 0
                    y_offset = 0
                else:
                    # Container is taller than desired, constrain by width
                    draw_width = width
                    draw_height = draw_width / self.aspect_ratio
                    x_offset = 0
                    y_offset = 0
            else:
                # No aspect ratio constraint, use full widget
                draw_width = width
                draw_height = height
                x_offset = 0
                y_offset = 0
            
            # Calculate normalized position from drag (relative to drawing area)
            dragged_x = (event.position().x() - x_offset) / draw_width
            dragged_y = (event.position().y() - y_offset) / draw_height
            
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
            
            # Calculate drawing area respecting aspect ratio
            if self.aspect_ratio is not None and self.aspect_ratio > 0:
                container_ratio = width / height
                if container_ratio > self.aspect_ratio:
                    # Container is wider than desired, constrain by height
                    draw_height = height
                    draw_width = draw_height * self.aspect_ratio
                    x_offset = 0
                    y_offset = 0
                else:
                    # Container is taller than desired, constrain by width
                    draw_width = width
                    draw_height = draw_width / self.aspect_ratio
                    x_offset = 0
                    y_offset = 0
            else:
                # No aspect ratio constraint, use full widget
                draw_width = width
                draw_height = height
                x_offset = 0
                y_offset = 0
            
            # Calculate normalized position from release (relative to drawing area)
            released_x = (event.position().x() - x_offset) / draw_width
            released_y = (event.position().y() - y_offset) / draw_height
            
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
        
        # Calculate drawing area respecting aspect ratio
        if self.aspect_ratio is not None and self.aspect_ratio > 0:
            # Calculate dimensions that maintain aspect ratio
            container_ratio = width / height
            if container_ratio > self.aspect_ratio:
                # Container is wider than desired, constrain by height
                draw_height = height
                draw_width = draw_height * self.aspect_ratio
                x_offset = 0  # No centering - draw from left
                y_offset = 0
            else:
                # Container is taller than desired, constrain by width
                draw_width = width
                draw_height = draw_width / self.aspect_ratio
                x_offset = 0
                y_offset = 0  # No centering - draw from top
        else:
            # No aspect ratio constraint, use full widget
            draw_width = width
            draw_height = height
            x_offset = 0
            y_offset = 0
        
        # Draw background for the drawing area
        painter.fillRect(int(x_offset), int(y_offset), int(draw_width), int(draw_height), QtGui.QColor(240, 240, 240))
        
        # Draw grid within drawing area
        painter.setPen(QtGui.QColor(200, 200, 200))
        for i in range(1, 10):
            grid_x = x_offset + draw_width * i / 10
            grid_y = y_offset + draw_height * i / 10
            painter.drawLine(int(grid_x), int(y_offset), int(grid_x), int(y_offset + draw_height))
            painter.drawLine(int(x_offset), int(grid_y), int(x_offset + draw_width), int(grid_y))
        
        # Draw border around drawing area
        painter.setPen(QtGui.QColor(100, 100, 100))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(int(x_offset), int(y_offset), int(draw_width - 1), int(draw_height - 1))
        
        # Draw center cross within drawing area
        center_x = x_offset + draw_width / 2
        center_y = y_offset + draw_height / 2
        painter.setPen(QtGui.QColor(150, 150, 150))
        painter.drawLine(int(center_x) - 10, int(center_y), int(center_x) + 10, int(center_y))
        painter.drawLine(int(center_x), int(center_y) - 10, int(center_x), int(center_y) + 10)
        
        # Draw old position (gray)
        if self.old_position is not None:
            old_x = x_offset + self.old_position[0] * draw_width
            old_y = y_offset + self.old_position[1] * draw_height
            
            # Draw old position marker (gray)
            painter.setPen(QtGui.QColor(128, 128, 128))
            painter.setBrush(QtGui.QColor(128, 128, 128))
            painter.drawEllipse(QtCore.QPointF(old_x, old_y), 6, 6)
            
            # Draw position lines to edges (gray)
            painter.setPen(QtGui.QColor(128, 128, 128, 100))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawLine(int(old_x), int(old_y), int(old_x), int(y_offset + draw_height))
            painter.drawLine(int(old_x), int(old_y), int(x_offset + draw_width), int(old_y))
        
        # Draw target position (red)
        if self.target_position is not None:
            target_x = x_offset + self.target_position[0] * draw_width
            target_y = y_offset + self.target_position[1] * draw_height
            
            # Draw target position marker (red)
            painter.setPen(QtGui.QColor(244, 67, 54))
            painter.setBrush(QtGui.QColor(244, 67, 54))
            painter.drawEllipse(QtCore.QPointF(target_x, target_y), 8, 8)
            
            # Draw position lines to edges (red)
            painter.setPen(QtGui.QColor(244, 67, 54, 100))
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.drawLine(int(target_x), int(target_y), int(target_x), int(y_offset + draw_height))
            painter.drawLine(int(target_x), int(target_y), int(x_offset + draw_width), int(target_y))
        
        # Draw current position (green)
        cur_x = x_offset + self.current_position[0] * draw_width
        cur_y = y_offset + self.current_position[1] * draw_height
        
        # Draw current position marker (green)
        painter.setPen(QtGui.QColor(76, 175, 80))
        painter.setBrush(QtGui.QColor(76, 175, 80))
        painter.drawEllipse(QtCore.QPointF(cur_x, cur_y), 8, 8)
        
        # Draw position lines to edges (green)
        painter.setPen(QtGui.QColor(76, 175, 80, 100))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawLine(int(cur_x), int(cur_y), int(cur_x), int(y_offset + draw_height))
        painter.drawLine(int(cur_x), int(cur_y), int(x_offset + draw_width), int(cur_y))
        
        # Draw limit boundaries (highlight the actual usable area)
        painter.setPen(QtGui.QColor(50, 50, 50, 50))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(int(x_offset) + 1, int(y_offset) + 1, int(draw_width - 3), int(draw_height - 3))
        
        # Draw limit labels (Y is inverted: top=min, bottom=max, unless swapped)
        painter.setPen(QtGui.QColor(80, 80, 80))
        painter.setFont(QtGui.QFont("Arial", 8))
        
        # X limits labels
        painter.drawText(int(x_offset) + 5, int(y_offset + draw_height - 5), f"{self.x_min:.1f}")
        painter.drawText(int(x_offset + draw_width - 40), int(y_offset + draw_height - 5), f"{self.x_max:.1f}")
        
        # Y limits labels - handle swapped case
        if self.y_axis_swapped:
            # When swapped: top=max, bottom=min (normal orientation)
            painter.drawText(int(x_offset) + 5, int(y_offset + 12), f"{self.y_min:.1f}")
            painter.drawText(int(x_offset) + 5, int(y_offset + draw_height - 12), f"{self.y_max:.1f}")
        else:
            # Default: top=max, bottom=min (inverted for typical coordinate system)
            painter.drawText(int(x_offset) + 5, int(y_offset + 12), f"{self.y_max:.1f}")
            painter.drawText(int(x_offset) + 5, int(y_offset + draw_height - 12), f"{self.y_min:.1f}")
        
        painter.end()


class StageControlTab(QtWidgets.QWidget):
    """Stage control panel as a dockable tab with real units and slider control."""
    
    position_changed = QtCore.pyqtSignal(float, float, float)  # x, y, z in real units
    _move_complete = QtCore.pyqtSignal(float, float)  # Signal when move operation completes with target position
    _move_error = QtCore.pyqtSignal(str)  # Signal for move errors
    
    def __init__(self, stage: StageXY = None, focus: FocusZ = None, config_path: str = None, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.focus = focus
        self.config_path = config_path
        self._is_live_mode = False  # Default: live mode OFF
        self._is_dragging = False  # Track dragging state
        self._keep_aspect_ratio = True  # Default: keep aspect ratio ON
        self._aspect_ratio = 1.0  # Will be calculated from range
        self._xy_width = POSITION_BLOCK_SIZE  # Dynamic width for XY controls
        self._load_config()
        self._build_ui()
        self._setup_position_timer()
        
        # Connect signals for thread communication
        self._move_complete.connect(self._on_move_complete)
        self._move_error.connect(self._on_move_error_slot)
        
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
        # Stage get_position() returns real units (mm), not actual steps
        # No conversion needed, just apply offset
        real_units = steps + offset
        return real_units
    
    def _real_units_to_steps(self, real_units: float, scale: float, offset: float) -> float:
        """Convert real units to steps."""
        # ScaledStageXY.move_to() expects logical coordinates (real units)
        # and internally converts to steps: rx = x * scale + offset
        # So we pass real units directly
        steps = real_units - offset
        return steps
        
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
        
        # Aspect ratio switch
        self.aspect_ratio_switch = QtWidgets.QCheckBox("Keep Aspect Ratio")
        self.aspect_ratio_switch.setChecked(True)  # Default: ON
        self.aspect_ratio_switch.setStyleSheet("QCheckBox { font-weight: bold; padding: 4px; }")
        self.aspect_ratio_switch.toggled.connect(self._on_aspect_ratio_toggled)
        control_row.addWidget(self.aspect_ratio_switch)
        
        control_row.addStretch()
        layout.addLayout(control_row)
        
        # Main controls layout - XY on left, Z on right
        main_controls_layout = QtWidgets.QHBoxLayout()
        main_controls_layout.setObjectName("main_controls_layout")
        main_controls_layout.setSpacing(10)
        
        # XY Position section - use layout directly, no GroupBox
        xy_container = QtWidgets.QWidget()
        xy_layout = QtWidgets.QVBoxLayout(xy_container)
        xy_layout.setSpacing(2)
        xy_layout.setContentsMargins(0, 0, 0, 0)
        
        # XY label
        xy_label = QtWidgets.QLabel("XY Position")
        xy_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        xy_layout.addWidget(xy_label)
        
        # XY controls grid
        xy_grid = QtWidgets.QGridLayout()
        xy_grid.setSpacing(0)
        xy_grid.setContentsMargins(0, 0, 0, 0)
        
        # Get limits for labels and tooltip
        x_min = self.stage_config['x_min'] if self.stage_config['x_min'] is not None else 0.0
        x_max = self.stage_config['x_max'] if self.stage_config['x_max'] is not None else 100.0
        y_min = self.stage_config['y_min'] if self.stage_config['y_min'] is not None else 0.0
        y_max = self.stage_config['y_max'] if self.stage_config['y_max'] is not None else 100.0
        
        # Calculate aspect ratio from range
        x_range = (x_max - x_min) if (x_max is not None and x_min is not None) else 100.0
        y_range = (y_max - y_min) if (y_max is not None and y_min is not None) else 100.0
        if y_range > 0:
            self._aspect_ratio = x_range / y_range
        else:
            self._aspect_ratio = 1.0
        
        # Calculate initial width based on aspect ratio
        base_size = POSITION_BLOCK_SIZE
        if self._keep_aspect_ratio and self._aspect_ratio is not None:
            if self._aspect_ratio > 1.2:
                self._xy_width = int(base_size * 1.2)
            elif self._aspect_ratio < 0.8:
                self._xy_width = int(base_size * 0.8)
            else:
                self._xy_width = base_size
        else:
            self._xy_width = base_size
        
        # Ensure minimum width
        self._xy_width = max(150, self._xy_width)
        
        # Position block - place directly in grid, no container
        self.position_block = PositionBlockWidget()
        self.position_block.set_limits(x_min, x_max, y_min, y_max)
        self.position_block.setFixedSize(self._xy_width, POSITION_BLOCK_SIZE)
        self.position_block.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        self.position_block.position_clicked.connect(self._on_position_block_clicked)
        self.position_block.position_drag_started.connect(self._on_drag_started)
        self.position_block.position_dragged.connect(self._on_dragged)
        self.position_block.position_drag_ended.connect(self._on_drag_ended)
        
        # Set initial aspect ratio on position block
        self.position_block.set_aspect_ratio(self._aspect_ratio if self._keep_aspect_ratio else None)
        
        # Set Y axis swapped flag based on configuration or default behavior
        # Default to False (standard inverted Y where top=max, bottom=min)
        self.position_block.set_y_axis_swapped(True)
        
        # Add position block directly to grid
        xy_grid.addWidget(self.position_block, 1, 1, 2, 1)  # Row 1-2, col 1
        
        # Print initial sizes
        print(f"Initial setup:")
        print(f"Position Block setFixedSize: {self._xy_width}x{POSITION_BLOCK_SIZE}")
        print(f"X slider setFixedWidth: {self._xy_width}")
        print("-" * 50)
        
        # Row 0: Y label (col 0) and Y spinbox (col 1)
        y_label = QtWidgets.QLabel("Y:")
        y_label.setStyleSheet("font-weight: bold; font-size: 10px;")
        xy_grid.addWidget(y_label, 0, 0, 1, 1)
        
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setRange(-1e6, 1e6)
        self.y_spin.setDecimals(3)
        self.y_spin.setSuffix(" mm")
        self.y_spin.setValue(0.0)
        self.y_spin.setMaximumWidth(120)
        self.y_spin.setStyleSheet("font-size: 10px;")
        xy_grid.addWidget(self.y_spin, 0, 1, 1, 1)
        
        self.current_y_label = QtWidgets.QLabel("-")
        self.current_y_label.setStyleSheet("font-family: monospace; font-weight: bold; font-size: 9px; color: #666;")
        self.current_y_label.setFixedWidth(50)
        xy_grid.addWidget(self.current_y_label, 0, 2, 1, 1)
        
        # Row 1-2: Y slider (col 0, span 2 rows) - vertical, aligned with 2D block height
        self.y_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.y_slider.setRange(0, SLIDER_RANGE)
        self.y_slider.setValue(0)
        self.y_slider.setFixedWidth(20)
        self.y_slider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        self.y_slider.setToolTip(f"Y range: {y_min:.1f} to {y_max:.1f} mm")
        xy_grid.addWidget(self.y_slider, 1, 0, 2, 1)  # Row 1-2, col 0, span 2 rows
        
        # Row 3: X slider (col 1) - wrap in container to enforce fixed width
        self.x_slider_container = QtWidgets.QWidget()
        self.x_slider_container.setFixedWidth(self._xy_width)
        self.x_slider_container.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        x_slider_layout = QtWidgets.QHBoxLayout(self.x_slider_container)
        x_slider_layout.setContentsMargins(0, 0, 0, 0)
        x_slider_layout.setSpacing(0)
        
        self.x_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.x_slider.setRange(0, SLIDER_RANGE)
        self.x_slider.setValue(0)
        self.x_slider.setFixedHeight(20)
        self.x_slider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.x_slider.setToolTip(f"X range: {x_min:.1f} to {x_max:.1f} mm")
        x_slider_layout.addWidget(self.x_slider)
        
        xy_grid.addWidget(self.x_slider_container, 3, 1, 1, 1)
        
        # Row 4: X label (col 0) and X spinbox (col 1)
        x_label = QtWidgets.QLabel("X:")
        x_label.setStyleSheet("font-weight: bold; font-size: 10px;")
        xy_grid.addWidget(x_label, 4, 0, 1, 1)
        
        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setRange(-1e6, 1e6)
        self.x_spin.setDecimals(3)
        self.x_spin.setSuffix(" mm")
        self.x_spin.setValue(0.0)
        self.x_spin.setMaximumWidth(120)
        self.x_spin.setStyleSheet("font-size: 10px;")
        xy_grid.addWidget(self.x_spin, 4, 1, 1, 1)
        
        self.current_x_label = QtWidgets.QLabel("-")
        self.current_x_label.setStyleSheet("font-family: monospace; font-weight: bold; font-size: 9px; color: #666;")
        self.current_x_label.setFixedWidth(50)
        xy_grid.addWidget(self.current_x_label, 4, 2, 1, 1)
        
        # Set column stretches - fixed 3-column layout
        xy_grid.setColumnStretch(0, 0)  # Labels - fixed width
        xy_grid.setColumnStretch(1, 0)  # Spinboxes/slider - fixed width (controlled by setFixedWidth)
        xy_grid.setColumnStretch(2, 0)  # Current values - fixed width
        
        # Set row stretches
        xy_grid.setRowStretch(0, 0)  # Y label/spinbox - fixed height
        xy_grid.setRowStretch(1, 1)  # Y slider + 2D block - expand vertically
        xy_grid.setRowStretch(2, 1)  # 2D block (span) - expand vertically
        xy_grid.setRowStretch(3, 0)  # X slider - fixed height
        xy_grid.setRowStretch(4, 0)  # X label/spinbox - fixed height
        
        xy_layout.addLayout(xy_grid)
        main_controls_layout.addWidget(xy_container)
        
        # Store reference to xy container for later updates
        self.xy_container = xy_container
        
        # Right side: Z Focus control with vertical slider
        z_group = QtWidgets.QGroupBox("Z Focus")
        z_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        z_layout = QtWidgets.QVBoxLayout()
        z_layout.setSpacing(5)
        z_layout.setContentsMargins(8, 8, 8, 8)
        
        # Z controls container
        z_controls = QtWidgets.QWidget()
        z_controls_layout = QtWidgets.QHBoxLayout(z_controls)
        z_controls_layout.setContentsMargins(0, 0, 0, 0)
        z_controls_layout.setSpacing(3)
        
        # Z label and spinbox
        z_label = QtWidgets.QLabel("Z:")
        z_label.setStyleSheet("font-weight: bold; font-size: 10px;")
        
        self.z_spin = QtWidgets.QDoubleSpinBox()
        self.z_spin.setRange(-1e6, 1e6)
        self.z_spin.setDecimals(3)
        self.z_spin.setSuffix(" mm")
        self.z_spin.setValue(0.0)
        self.z_spin.setMaximumWidth(60)
        self.z_spin.setStyleSheet("font-size: 10px;")
        
        z_controls_layout.addWidget(z_label)
        z_controls_layout.addWidget(self.z_spin)
        
        z_layout.addWidget(z_controls)
        
        # Z current position label
        self.current_z_label = QtWidgets.QLabel("Z: -")
        self.current_z_label.setStyleSheet("font-family: monospace; font-weight: bold; font-size: 9px; color: #666;")
        self.current_z_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        z_layout.addWidget(self.current_z_label)
        
        # Z slider - vertical
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Vertical)
        self.z_slider.setRange(0, SLIDER_RANGE)
        self.z_slider.setValue(0)
        self.z_slider.setFixedWidth(20)
        self.z_slider.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)
        z_layout.addWidget(self.z_slider)
        
        z_group.setLayout(z_layout)
        z_group.setMaximumWidth(100)  # Keep Z controls compact
        main_controls_layout.addWidget(z_group)
        
        layout.addLayout(main_controls_layout)
        
        # Action buttons row - compact
        action_layout = QtWidgets.QHBoxLayout()
        action_layout.setSpacing(5)
        
        self.move_btn = QtWidgets.QPushButton("Move")
        self.move_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 4px;")
        self.move_btn.setMaximumHeight(25)
        self.move_btn.setMaximumWidth(60)
        
        self.refresh_btn = QtWidgets.QPushButton("⟳")
        self.refresh_btn.setStyleSheet("padding: 4px; font-size: 12px;")
        self.refresh_btn.setMaximumWidth(30)
        self.refresh_btn.setMaximumHeight(25)
        self.refresh_btn.setToolTip("Refresh Position")
        
        action_layout.addWidget(self.move_btn)
        action_layout.addWidget(self.refresh_btn)
        action_layout.addStretch()
        
        layout.addLayout(action_layout)
        
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
    
    def _update_position_display(self):
        """Update the inline position labels with current values."""
        x_val = self.x_spin.value()
        y_val = self.y_spin.value()
        z_val = self.z_spin.value()
        
        self.current_x_label.setText(f"{x_val:.1f}")
        self.current_y_label.setText(f"{y_val:.1f}")
        self.current_z_label.setText(f"{z_val:.1f}")
    
    def _on_live_mode_toggled(self, checked):
        """Handle live mode toggle."""
        self._is_live_mode = checked
        
        # Check if devices are available
        has_devices = self.stage is not None or self.focus is not None
        
        if checked:
            # Live mode: enable auto-refresh, hide move button
            if has_devices:
                self.position_timer.start(AUTO_REFRESH_INTERVAL_MS)
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
    
    def _on_aspect_ratio_toggled(self, checked):
        """Handle aspect ratio toggle."""
        self._keep_aspect_ratio = checked
        # Update position block aspect ratio
        if hasattr(self, 'position_block'):
            if checked:
                self.position_block.set_aspect_ratio(self._aspect_ratio)
            else:
                self.position_block.set_aspect_ratio(None)  # No aspect ratio constraint
        
        # Recalculate and update size based on aspect ratio
        base_size = POSITION_BLOCK_SIZE
        if self._keep_aspect_ratio and self._aspect_ratio is not None:
            if self._aspect_ratio > 1.2:
                # Wide: increase width
                new_width = int(base_size * 1.2)
                new_height = base_size
            elif self._aspect_ratio < 0.8:
                # Narrow: decrease width
                new_width = int(base_size * 0.8)
                new_height = base_size
            else:
                # Square: keep base size
                new_width = base_size
                new_height = base_size
        else:
            # No aspect ratio: keep base size
            new_width = base_size
            new_height = base_size
        
        # Update position block size
        self.position_block.setFixedSize(new_width, new_height)
        self._xy_width = new_width
        
        # Update X slider container width if it exists
        if hasattr(self, 'x_slider_container'):
            print(f"Updating X slider container width from {self.x_slider_container.width()} to {self._xy_width}")
            self.x_slider_container.setFixedWidth(self._xy_width)
            self.x_slider_container.updateGeometry()
        
        # Trigger layout update
        self.position_block.updateGeometry()
        self.xy_container.updateGeometry()
        
        # Force UI to process layout changes
        QtWidgets.QApplication.processEvents()
        
        # Print info including actual rendered sizes and margins
        print(f"Aspect Ratio Toggled: {checked}")
        print(f"Aspect Ratio Value: {self._aspect_ratio:.2f}")
        print(f"Position Block setFixedSize: {new_width}x{new_height}")
        print(f"Position Block actual size: {self.position_block.width()}x{self.position_block.height()}")
        pb_layout = self.position_block.layout()
        if pb_layout:
            margins = pb_layout.contentsMargins()
            print(f"Position Block layout margins: left={margins.left()}, top={margins.top()}, right={margins.right()}, bottom={margins.bottom()}")
        else:
            print(f"Position Block has no layout")
        print(f"X slider setFixedWidth: {self._xy_width}")
        print(f"X slider actual size: {self.x_slider.width()}x{self.x_slider.height()}")
        if hasattr(self, 'x_slider_container'):
            print(f"X slider container setFixedWidth: {self._xy_width}")
            print(f"X slider container actual size: {self.x_slider_container.width()}x{self.x_slider_container.height()}")
        xs_layout = self.x_slider.layout()
        if xs_layout:
            margins = xs_layout.contentsMargins()
            print(f"X slider layout margins: left={margins.left()}, top={margins.top()}, right={margins.right()}, bottom={margins.bottom()}")
        else:
            print(f"X slider has no layout")
        print("-" * 50)
    
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
        """Move stage to specific X,Y position in a separate thread."""
        try:
            # Convert real units to steps
            x_steps = self._real_units_to_steps(x, self.stage_config['x_scale'], self.stage_config['x_offset'])
            y_steps = self._real_units_to_steps(y, self.stage_config['y_scale'], self.stage_config['y_offset'])
            
            # Move stage in a separate thread to avoid blocking UI
            if self.stage and hasattr(self.stage, 'move_to'):
                import threading
                move_thread = threading.Thread(
                    target=self._move_stage_thread,
                    args=(x_steps, y_steps, x, y),
                    daemon=True
                )
                move_thread.start()
            else:
                # Update UI even if stage not available
                self._update_position_ui(x, y)
                
        except Exception as e:
            print(f"Error moving stage: {e}")
    
    def _move_stage_thread(self, x_steps, y_steps, x_real, y_real):
        """Thread function to move stage."""
        try:
            self.stage.move_to(x_steps, y_steps)
        except Exception as e:
            print(f"Error in stage movement thread: {e}")
            self._move_error.emit(str(e))
        finally:
            # Update UI using signal with target position
            self._move_complete.emit(x_real, y_real)
    
    @pyqtSlot(float, float)
    def _update_position_ui(self, x, y):
        """Update UI after stage movement (called from thread)."""
        try:
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
            print(f"Error updating position UI: {e}")
    
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
                
            self.current_x_label.setText(f"{x_real:.1f}")
            self.current_y_label.setText(f"{y_real:.1f}")
            self.current_z_label.setText(f"{z_real:.1f}")
            
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
        self.y_slider.setRange(0, SLIDER_RANGE)  # Normalized range
        
        # Z slider limits - use normalized range
        if self.focus_config['min'] is not None and self.focus_config['max'] is not None:
            self.z_spin.setRange(self.focus_config['min'], self.focus_config['max'])
            self.z_slider.setRange(0, SLIDER_RANGE)  # Normalized range
        
        # Sync position block limits with slider limits
        if hasattr(self, 'position_block'):
            self.position_block.set_limits(x_min, x_max, y_min, y_max)
    
    def _on_x_spin_changed(self, value):
        """Handle X spinbox change."""
        # Get limits for normalization
        x_min = self.stage_config['x_min'] if self.stage_config['x_min'] is not None else 0.0
        x_max = self.stage_config['x_max'] if self.stage_config['x_max'] is not None else 100.0
        
        # Convert to normalized range
        x_norm = int(((value - x_min) / (x_max - x_min)) * SLIDER_RANGE) if x_max > x_min else SLIDER_RANGE // 2
        
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
        
        # Convert from normalized range to real units
        x_real = x_min + (value / SLIDER_RANGE) * (x_max - x_min)
        
        self.x_spin.blockSignals(True)
        self.x_spin.setValue(x_real)
        self.x_spin.blockSignals(False)
        
        # Update inline position labels
        self._update_position_display()
        
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
        
        # Update inline position labels
        self._update_position_display()
        
        # In live mode, move stage immediately
        if self._is_live_mode:
            self._move_to_position_xy(self.x_spin.value(), self.y_spin.value())
    
    def _on_y_spin_changed(self, value):
        """Handle Y spinbox change."""
        # Get limits for normalization
        y_min = self.stage_config['y_min'] if self.stage_config['y_min'] is not None else 0.0
        y_max = self.stage_config['y_max'] if self.stage_config['y_max'] is not None else 100.0
        
        # Convert to normalized range (reversed for Y axis)
        y_norm = int(((value - y_min) / (y_max - y_min)) * SLIDER_RANGE) if y_max > y_min else SLIDER_RANGE // 2
        y_norm_reversed = SLIDER_RANGE - y_norm  # Reverse for Y axis
        
        self.y_slider.blockSignals(True)
        self.y_slider.setValue(y_norm_reversed)
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
        
        # Convert from normalized range to real units (reversed for Y axis)
        y_norm_reversed = SLIDER_RANGE - value  # Reverse back from slider direction
        y_real = y_min + (y_norm_reversed / SLIDER_RANGE) * (y_max - y_min)
        
        self.y_spin.blockSignals(True)
        self.y_spin.setValue(y_real)
        self.y_spin.blockSignals(False)
        
        # Update inline position labels
        self._update_position_display()
        
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
        
        # Convert to normalized range
        z_norm = int(((value - z_min) / (z_max - z_min)) * SLIDER_RANGE) if z_max > z_min else SLIDER_RANGE // 2
        
        self.z_slider.blockSignals(True)
        self.z_slider.setValue(z_norm)
        self.z_slider.blockSignals(False)
    
    def _on_z_slider_changed(self, value):
        """Handle Z slider change."""
        # Get limits for conversion
        z_min = self.focus_config['min'] if self.focus_config['min'] is not None else 0.0
        z_max = self.focus_config['max'] if self.focus_config['max'] is not None else 100.0
        
        # Convert from normalized range to real units
        z_real = z_min + (value / SLIDER_RANGE) * (z_max - z_min)
        
        self.z_spin.blockSignals(True)
        self.z_spin.setValue(z_real)
        self.z_spin.blockSignals(False)
        
        # Update inline position labels
        self._update_position_display()
    
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
        
        # Convert to normalized range
        x_norm = int(((x_val - x_min) / (x_max - x_min)) * SLIDER_RANGE) if x_max > x_min else SLIDER_RANGE // 2
        y_norm = int(((y_val - y_min) / (y_max - y_min)) * SLIDER_RANGE) if y_max > y_min else SLIDER_RANGE // 2
        y_norm_reversed = SLIDER_RANGE - y_norm  # Reverse for Y axis
        z_norm = int(((z_val - z_min) / (z_max - z_min)) * SLIDER_RANGE) if z_max > z_min else SLIDER_RANGE // 2
        
        self.x_slider.blockSignals(True)
        self.y_slider.blockSignals(True)
        self.z_slider.blockSignals(True)
        
        self.x_slider.setValue(x_norm)
        self.y_slider.setValue(y_norm_reversed)
        self.z_slider.setValue(z_norm)
        
        self.x_slider.blockSignals(False)
        self.y_slider.blockSignals(False)
        self.z_slider.blockSignals(False)
        
    def _move_to_position(self):
        """Move stage and focus to the specified positions using real units in a separate thread."""
        try:
            # Check if devices are available
            if self.stage is None and self.focus is None:
                QtWidgets.QMessageBox.warning(self, "No Devices", 
                    "No stage or focus devices available. Please load hardware configuration first.")
                return
            
            x_real = self.x_spin.value()
            y_real = self.y_spin.value()
            z_real = self.z_spin.value()
            
            # Convert real units to steps
            x_steps = self._real_units_to_steps(x_real, self.stage_config['x_scale'], self.stage_config['x_offset'])
            y_steps = self._real_units_to_steps(y_real, self.stage_config['y_scale'], self.stage_config['y_offset'])
            z_steps = self._real_units_to_steps(z_real, self.focus_config['scale'], self.focus_config['offset'])
            
            # Move in a separate thread to avoid blocking UI
            import threading
            move_thread = threading.Thread(
                target=self._move_stage_focus_thread,
                args=(x_steps, y_steps, z_steps, x_real, y_real, z_real),
                daemon=True
            )
            move_thread.start()
                
        except Exception as e:
            print(f"Error starting move operation: {e}")
            QtWidgets.QMessageBox.warning(self, "Move Error", f"Failed to start move: {e}")
    
    def _move_stage_focus_thread(self, x_steps, y_steps, z_steps, x_real, y_real, z_real):
        """Thread function to move stage and focus."""
        try:
            if self.stage and hasattr(self.stage, 'move_to'):
                self.stage.move_to(x_steps, y_steps)
                
            if self.focus and hasattr(self.focus, 'move_to'):
                self.focus.move_to(z_steps)
                
            # Emit completion signal with target position
            self._move_complete.emit(x_real, y_real)
                
        except Exception as e:
            print(f"Error in move thread: {e}")
            self._move_error.emit(str(e))
    
    @pyqtSlot(float, float)
    def _on_move_complete(self, x_real, y_real):
        """Handle move completion signal."""
        try:
            # Reset target position after successful move (in normal mode)
            if not self._is_live_mode and hasattr(self, 'position_block'):
                self.position_block._reset_target()
            
            # Update position block with the target position
            if hasattr(self, 'position_block'):
                self.position_block.set_position(x_real, y_real)
            
            # Refresh position from hardware to get actual position
            self._refresh_position()
        except Exception as e:
            print(f"Error handling move complete: {e}")
    
    @pyqtSlot(str)
    def _on_move_error_slot(self, error_msg):
        """Handle move error signal."""
        QtWidgets.QMessageBox.warning(self, "Move Error", f"Failed to move: {error_msg}")
    
    @pyqtSlot()
    def _refresh_position(self):
        """Refresh the current position display in real units (manual refresh)."""
        try:
            x_real, y_real, z_real = 0.0, 0.0, 0.0
            
            if self.stage and hasattr(self.stage, 'get_position'):
                pos = self.stage.get_position()
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    x_steps, y_steps = float(pos[0]), float(pos[1])
                    x_real = self._steps_to_real_units(x_steps, self.stage_config['x_scale'], self.stage_config['x_offset'])
                    y_real = self._steps_to_real_units(y_steps, self.stage_config['y_scale'], self.stage_config['y_offset'])
                    
            if self.focus and hasattr(self.focus, 'get_position'):
                z_steps = self.focus.get_position()
                z_real = self._steps_to_real_units(z_steps, self.focus_config['scale'], self.focus_config['offset'])
                
            # Update UI
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
            
            # Update position block visualization
            if hasattr(self, 'position_block'):
                self.position_block.set_position(x_real, y_real)
            
        except Exception as e:
            print(f"Error refreshing position: {e}")
    
    def set_stage(self, stage: StageXY):
        """Set the stage device and refresh position."""
        print(f"DEBUG: set_stage called with stage: {stage}")
        print(f"DEBUG: Stage type: {type(stage)}")
        print(f"DEBUG: Stage id: {id(stage)}")
        print(f"DEBUG: Stage has move_to: {hasattr(stage, 'move_to')}")
        
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
    
    def cleanup(self):
        """Clean up resources when widget is destroyed."""
        if hasattr(self, 'position_timer'):
            self.position_timer.stop()
            self.position_timer.deleteLater()