"""Move Motors Dialog - Manual motor control with sliders and editable fields."""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class MoveMotorsDialog(QtWidgets.QDialog):
    """Dialog for manual motor control with sliders and editable fields."""

    def __init__(self, stage, focus, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._stage = stage
        self._focus = focus
        
        # Mode and state tracking
        self._live_mode = False
        self._is_moving = False
        self._position_timer: QtCore.QTimer | None = None
        
        try:
            self.setWindowTitle("Move Motors")
            self.setMinimumWidth(500)
            self._build_ui()
            
            # Start position update timer for live position display
            self._start_position_timer()
            
            # Load current positions
            self._load_current_positions()
            
        except Exception as e:
            logger.error("Error initializing move motors dialog: %s", e)
            raise

    def _build_ui(self):
        """Build the dialog UI."""
        try:
            layout = QtWidgets.QVBoxLayout(self)
            
            # Mode selection
            mode_layout = QtWidgets.QHBoxLayout()
            mode_label = QtWidgets.QLabel("<b>Mode:</b>")
            mode_layout.addWidget(mode_label)
            
            self._live_mode_checkbox = QtWidgets.QCheckBox("Live Moving Mode")
            self._live_mode_checkbox.setToolTip(
                "When enabled, sliders and spinboxes move motors immediately.\n"
                "When disabled, use 'Move to Position' button to move."
            )
            self._live_mode_checkbox.toggled.connect(self._on_live_mode_toggled)
            mode_layout.addWidget(self._live_mode_checkbox)
            
            mode_layout.addStretch()
            layout.addLayout(mode_layout)
            layout.addSpacing(10)
            
            # Instructions
            self._info_label = QtWidgets.QLabel(
                "<b>Manual Motor Control</b><br>"
                "Use the sliders or enter values to move motors to specific positions."
            )
            self._info_label.setWordWrap(True)
            layout.addWidget(self._info_label)
            layout.addSpacing(10)
            
            # Status indicator
            self._status_label = QtWidgets.QLabel("Status: Ready")
            self._status_label.setStyleSheet("color: green; font-weight: bold;")
            layout.addWidget(self._status_label)
            layout.addSpacing(5)
            
            # Create axis control groups
            self._create_stage_controls(layout)
            self._create_focus_controls(layout)
            
            # Buttons
            button_layout = QtWidgets.QHBoxLayout()
            
            refresh_btn = QtWidgets.QPushButton("Refresh Current Position")
            refresh_btn.clicked.connect(self._load_current_positions)
            button_layout.addWidget(refresh_btn)
            
            button_layout.addStretch()
            
            self._move_btn = QtWidgets.QPushButton("Move to Position")
            self._move_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self._move_btn.clicked.connect(self._move_to_position)
            button_layout.addWidget(self._move_btn)
            
            close_btn = QtWidgets.QPushButton("Close")
            close_btn.clicked.connect(self._close_dialog)
            button_layout.addWidget(close_btn)
            
            layout.addLayout(button_layout)
            
        except Exception as e:
            logger.error("Error building UI: %s", e)
            raise

    def _create_stage_controls(self, parent_layout):
        """Create stage (X/Y) control group."""
        stage_group = QtWidgets.QGroupBox("Stage (X/Y)")
        stage_layout = QtWidgets.QVBoxLayout(stage_group)
        
        # X axis
        x_layout = QtWidgets.QVBoxLayout()
        
        # Current position display (read-only)
        x_current_layout = QtWidgets.QHBoxLayout()
        x_current_label = QtWidgets.QLabel("Current X:")
        x_current_layout.addWidget(x_current_label)
        
        self._x_current_display = QtWidgets.QLineEdit()
        self._x_current_display.setReadOnly(True)
        self._x_current_display.setStyleSheet("background-color: #f0f0f0; color: #666;")
        self._x_current_display.setPlaceholderText("--")
        x_current_layout.addWidget(self._x_current_display)
        x_layout.addLayout(x_current_layout)
        
        # Target position controls
        x_target_layout = QtWidgets.QHBoxLayout()
        x_target_label = QtWidgets.QLabel("Target X:")
        x_target_layout.addWidget(x_target_label)
        
        self._x_spin = QtWidgets.QDoubleSpinBox()
        self._x_spin.setRange(-1e6, 1e6)
        self._x_spin.setDecimals(2)
        self._x_spin.setSingleStep(0.1)
        x_target_layout.addWidget(self._x_spin)
        
        self._x_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._x_slider.setRange(-1000, 1000)  # Will be updated based on actual range
        x_layout.addWidget(self._x_slider)
        x_layout.addLayout(x_target_layout)
        
        # Connect spinbox and slider
        self._x_spin.valueChanged.connect(self._on_x_spin_changed)
        self._x_slider.valueChanged.connect(self._on_x_slider_changed)
        
        stage_layout.addLayout(x_layout)
        
        # Y axis
        y_layout = QtWidgets.QVBoxLayout()
        
        # Current position display (read-only)
        y_current_layout = QtWidgets.QHBoxLayout()
        y_current_label = QtWidgets.QLabel("Current Y:")
        y_current_layout.addWidget(y_current_label)
        
        self._y_current_display = QtWidgets.QLineEdit()
        self._y_current_display.setReadOnly(True)
        self._y_current_display.setStyleSheet("background-color: #f0f0f0; color: #666;")
        self._y_current_display.setPlaceholderText("--")
        y_current_layout.addWidget(self._y_current_display)
        y_layout.addLayout(y_current_layout)
        
        # Target position controls
        y_target_layout = QtWidgets.QHBoxLayout()
        y_target_label = QtWidgets.QLabel("Target Y:")
        y_target_layout.addWidget(y_target_label)
        
        self._y_spin = QtWidgets.QDoubleSpinBox()
        self._y_spin.setRange(-1e6, 1e6)
        self._y_spin.setDecimals(2)
        self._y_spin.setSingleStep(0.1)
        y_target_layout.addWidget(self._y_spin)
        
        self._y_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._y_slider.setRange(-1000, 1000)  # Will be updated based on actual range
        y_layout.addWidget(self._y_slider)
        y_layout.addLayout(y_target_layout)
        
        # Connect spinbox and slider
        self._y_spin.valueChanged.connect(self._on_y_spin_changed)
        self._y_slider.valueChanged.connect(self._on_y_slider_changed)
        
        stage_layout.addLayout(y_layout)
        
        parent_layout.addWidget(stage_group)

    def _create_focus_controls(self, parent_layout):
        """Create focus (Z) control group."""
        if self._focus is None:
            return
            
        focus_group = QtWidgets.QGroupBox("Focus (Z)")
        focus_layout = QtWidgets.QVBoxLayout(focus_group)
        
        # Z axis
        z_layout = QtWidgets.QVBoxLayout()
        
        # Current position display (read-only)
        z_current_layout = QtWidgets.QHBoxLayout()
        z_current_label = QtWidgets.QLabel("Current Z:")
        z_current_layout.addWidget(z_current_label)
        
        self._z_current_display = QtWidgets.QLineEdit()
        self._z_current_display.setReadOnly(True)
        self._z_current_display.setStyleSheet("background-color: #f0f0f0; color: #666;")
        self._z_current_display.setPlaceholderText("--")
        z_current_layout.addWidget(self._z_current_display)
        z_layout.addLayout(z_current_layout)
        
        # Target position controls
        z_target_layout = QtWidgets.QHBoxLayout()
        z_target_label = QtWidgets.QLabel("Target Z:")
        z_target_layout.addWidget(z_target_label)
        
        self._z_spin = QtWidgets.QDoubleSpinBox()
        self._z_spin.setRange(-1e6, 1e6)
        self._z_spin.setDecimals(2)
        self._z_spin.setSingleStep(0.1)
        z_target_layout.addWidget(self._z_spin)
        
        self._z_slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self._z_slider.setRange(-1000, 1000)  # Will be updated based on actual range
        z_layout.addWidget(self._z_slider)
        z_layout.addLayout(z_target_layout)
        
        # Connect spinbox and slider
        self._z_spin.valueChanged.connect(self._on_z_spin_changed)
        self._z_slider.valueChanged.connect(self._on_z_slider_changed)
        
        focus_layout.addLayout(z_layout)
        
        parent_layout.addWidget(focus_group)

    def _start_position_timer(self):
        """Start the position update timer for live position display."""
        if self._position_timer is None:
            self._position_timer = QtCore.QTimer(self)
            self._position_timer.setInterval(200)  # Update every 200ms
            self._position_timer.timeout.connect(self._update_live_positions)
            self._position_timer.start()

    def _stop_position_timer(self):
        """Stop the position update timer."""
        if self._position_timer is not None:
            self._position_timer.stop()

    def _update_live_positions(self):
        """Update position displays with current motor positions."""
        try:
            # Update stage position displays
            if self._stage is not None and hasattr(self._stage, 'get_position'):
                x, y = self._stage.get_position()
                
                # Update current position displays (read-only text boxes)
                self._x_current_display.setText(f"{x:.2f}")
                self._y_current_display.setText(f"{y:.2f}")
            
            # Update focus position displays
            if self._focus is not None and hasattr(self._focus, 'get_position'):
                z = self._focus.get_position()
                
                # Update current position display (read-only text box)
                self._z_current_display.setText(f"{z:.2f}")
                
        except Exception as e:
            logger.warning("Failed to update live positions: %s", e)

    def _on_live_mode_toggled(self, checked: bool):
        """Handle live mode checkbox toggle."""
        self._live_mode = checked
        
        if checked:
            # Enable live moving mode
            self._move_btn.setEnabled(False)
            self._move_btn.setToolTip("Disabled in Live Moving Mode")
            self._info_label.setText(
                "<b>Live Moving Mode</b><br>"
                "Sliders and spinboxes move motors immediately."
            )
            # Enable value change signals for live movement
            self._x_spin.valueChanged.connect(self._on_live_x_changed)
            self._y_spin.valueChanged.connect(self._on_live_y_changed)
            if self._focus is not None:
                self._z_spin.valueChanged.connect(self._on_live_z_changed)
        else:
            # Disable live moving mode
            self._move_btn.setEnabled(not self._is_moving)
            self._move_btn.setToolTip("Click to move motors to specified positions")
            self._info_label.setText(
                "<b>Move to Position Mode</b><br>"
                "Use 'Move to Position' button to move motors."
            )
            # Disconnect live movement signals
            try:
                self._x_spin.valueChanged.disconnect(self._on_live_x_changed)
                self._y_spin.valueChanged.disconnect(self._on_live_y_changed)
                if self._focus is not None:
                    self._z_spin.valueChanged.disconnect(self._on_live_z_changed)
            except Exception:
                pass

    def _on_live_x_changed(self, value):
        """Handle live X position change."""
        if self._live_mode and not self._is_moving:
            self._live_move_stage(value, None)

    def _on_live_y_changed(self, value):
        """Handle live Y position change."""
        if self._live_mode and not self._is_moving:
            self._live_move_stage(None, value)

    def _on_live_z_changed(self, value):
        """Handle live Z position change."""
        if self._live_mode and not self._is_moving and self._focus is not None:
            self._live_move_focus(value)

    def _live_move_stage(self, x: float | None, y: float | None):
        """Move stage immediately in live mode."""
        try:
            if self._stage is None or not hasattr(self._stage, 'get_position'):
                return
                
            current_x, current_y = self._stage.get_position()
            target_x = x if x is not None else current_x
            target_y = y if y is not None else current_y
            
            self._stage.move_to(target_x, target_y)
            logger.info("Live move: Stage to X=%s, Y=%s", target_x, target_y)
            
        except Exception as e:
            logger.warning("Live stage move failed: %s", e)

    def _live_move_focus(self, z: float):
        """Move focus immediately in live mode."""
        try:
            if self._focus is None or not hasattr(self._focus, 'move_to'):
                return
                
            self._focus.move_to(z)
            logger.info("Live move: Focus to Z=%s", z)
            
        except Exception as e:
            logger.warning("Live focus move failed: %s", e)

    def _load_current_positions(self):
        """Load current motor positions into display fields and sync target controls."""
        try:
            # Load stage position
            if self._stage is not None and hasattr(self._stage, 'get_position'):
                x, y = self._stage.get_position()
                self._x_current_display.setText(f"{x:.2f}")
                self._y_current_display.setText(f"{y:.2f}")
                # Also sync target controls to current position
                self._x_spin.setValue(float(x))
                self._y_spin.setValue(float(y))
                self._update_slider_from_spin(self._x_spin, self._x_slider)
                self._update_slider_from_spin(self._y_spin, self._y_slider)
            
            # Load focus position
            if self._focus is not None and hasattr(self._focus, 'get_position'):
                z = self._focus.get_position()
                self._z_current_display.setText(f"{z:.2f}")
                # Also sync target control to current position
                self._z_spin.setValue(float(z))
                self._update_slider_from_spin(self._z_spin, self._z_slider)
                
            self._status_label.setText("Status: Position refreshed")
            self._status_label.setStyleSheet("color: green; font-weight: bold;")
                
        except Exception as e:
            logger.warning("Failed to load current positions: %s", e)
            self._status_label.setText("Status: Error loading position")
            self._status_label.setStyleSheet("color: red; font-weight: bold")
            QtWidgets.QMessageBox.warning(
                self, "Error", f"Could not load current positions:\n{e}"
            )

    def _on_x_spin_changed(self, value):
        """Handle X spinbox value change."""
        if not self._live_mode:
            self._update_slider_from_spin(self._x_spin, self._x_slider)

    def _on_x_slider_changed(self, value):
        """Handle X slider value change."""
        self._update_spin_from_slider(self._x_slider, self._x_spin)
        if self._live_mode and not self._is_moving:
            self._live_move_stage(self._x_spin.value(), None)

    def _on_y_spin_changed(self, value):
        """Handle Y spinbox value change."""
        if not self._live_mode:
            self._update_slider_from_spin(self._y_spin, self._y_slider)

    def _on_y_slider_changed(self, value):
        """Handle Y slider value change."""
        self._update_spin_from_slider(self._y_slider, self._y_spin)
        if self._live_mode and not self._is_moving:
            self._live_move_stage(None, self._y_spin.value())

    def _on_z_spin_changed(self, value):
        """Handle Z spinbox value change."""
        if not self._live_mode:
            self._update_slider_from_spin(self._z_spin, self._z_slider)

    def _on_z_slider_changed(self, value):
        """Handle Z slider value change."""
        self._update_spin_from_slider(self._z_slider, self._z_spin)
        if self._live_mode and not self._is_moving and self._focus is not None:
            self._live_move_focus(self._z_spin.value())

    def _update_slider_from_spin(self, spin: QtWidgets.QDoubleSpinBox, slider: QtWidgets.QSlider):
        """Update slider position based on spinbox value."""
        try:
            value = spin.value()
            slider.blockSignals(True)
            slider.setValue(int(value * 10))  # Scale by 10 for better resolution
            slider.blockSignals(False)
        except Exception:
            pass

    def _update_spin_from_slider(self, slider: QtWidgets.QSlider, spin: QtWidgets.QDoubleSpinBox):
        """Update spinbox value based on slider position."""
        try:
            value = slider.value()
            spin.blockSignals(True)
            spin.setValue(value / 10.0)  # Scale back from slider
            spin.blockSignals(False)
        except Exception:
            pass

    def _move_to_position(self):
        """Move motors to the specified positions."""
        if self._is_moving:
            return  # Already moving
            
        try:
            x = self._x_spin.value()
            y = self._y_spin.value()
            z = self._z_spin.value() if self._focus is not None else None
            
            # Set moving state
            self._is_moving = True
            self._move_btn.setEnabled(False)
            self._move_btn.setText("Moving...")
            self._status_label.setText("Status: Moving to position...")
            self._status_label.setStyleSheet("color: orange; font-weight: bold;")
            
            # Move stage
            if self._stage is not None and hasattr(self._stage, 'move_to'):
                self._stage.move_to(x, y)
                logger.info("Stage moved to X=%s, Y=%s", x, y)
            
            # Move focus
            if self._focus is not None and hasattr(self._focus, 'move_to'):
                self._focus.move_to(z)
                logger.info("Focus moved to Z=%s", z)
            
            # Simulate movement completion (in real implementation, you'd check if movement is complete)
            # For now, we assume immediate completion
            QtCore.QTimer.singleShot(100, self._on_move_complete)
            
        except Exception as e:
            logger.error("Failed to move motors: %s", e)
            self._on_move_error(str(e))

    def _on_move_complete(self):
        """Handle movement completion."""
        try:
            self._is_moving = False
            self._move_btn.setEnabled(not self._live_mode)
            self._move_btn.setText("Move to Position")
            self._status_label.setText("Status: Move complete")
            self._status_label.setStyleSheet("color: green; font-weight: bold;")
            
            x = self._x_spin.value()
            y = self._y_spin.value()
            z = self._z_spin.value() if self._focus is not None else None
            
            QtWidgets.QMessageBox.information(
                self,
                "Move Complete",
                f"Motors moved to:\nX: {x:.2f}\nY: {y:.2f}" + 
                (f"\nZ: {z:.2f}" if self._focus is not None else "")
            )
            
        except Exception as e:
            logger.error("Error in move completion handler: %s", e)
            self._on_move_error(str(e))

    def _on_move_error(self, error_msg: str):
        """Handle movement error."""
        try:
            self._is_moving = False
            self._move_btn.setEnabled(not self._live_mode)
            self._move_btn.setText("Move to Position")
            self._status_label.setText(f"Status: Error - {error_msg}")
            self._status_label.setStyleSheet("color: red; font-weight: bold;")
            
            QtWidgets.QMessageBox.critical(
                self, "Move Error", f"Could not move motors:\n{error_msg}"
            )
        except Exception:
            pass

    def _close_dialog(self):
        """Close dialog and cleanup."""
        self._stop_position_timer()
        self.reject()