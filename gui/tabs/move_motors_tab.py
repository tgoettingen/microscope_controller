from __future__ import annotations

from PyQt6 import QtWidgets, QtCore
from devices.base import StageXY, FocusZ


class MoveMotorsTab(QtWidgets.QWidget):
    """Move motors control panel as a dockable tab."""
    
    position_changed = QtCore.pyqtSignal(float, float, float)  # x, y, z
    
    def __init__(self, stage: StageXY = None, focus: FocusZ = None, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.focus = focus
        self._build_ui()
        
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        
        # Stage X control
        stage_group = QtWidgets.QGroupBox("Stage Position")
        stage_layout = QtWidgets.QFormLayout()
        
        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setRange(-1e6, 1e6)
        self.x_spin.setDecimals(3)
        self.x_spin.setSuffix(" steps")
        self.x_spin.setValue(0.0)
        
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setRange(-1e6, 1e6)
        self.y_spin.setDecimals(3)
        self.y_spin.setSuffix(" steps")
        self.y_spin.setValue(0.0)
        
        stage_layout.addRow("X:", self.x_spin)
        stage_layout.addRow("Y:", self.y_spin)
        stage_group.setLayout(stage_layout)
        layout.addWidget(stage_group)
        
        # Focus Z control
        focus_group = QtWidgets.QGroupBox("Focus Position")
        focus_layout = QtWidgets.QFormLayout()
        
        self.z_spin = QtWidgets.QDoubleSpinBox()
        self.z_spin.setRange(-1e6, 1e6)
        self.z_spin.setDecimals(3)
        self.z_spin.setSuffix(" steps")
        self.z_spin.setValue(0.0)
        
        focus_layout.addRow("Z:", self.z_spin)
        focus_group.setLayout(focus_layout)
        layout.addWidget(focus_group)
        
        # Move button
        self.move_btn = QtWidgets.QPushButton("Move to Position")
        self.move_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        layout.addWidget(self.move_btn)
        
        # Current position display
        pos_group = QtWidgets.QGroupBox("Current Position")
        pos_layout = QtWidgets.QFormLayout()
        
        self.current_x_label = QtWidgets.QLabel("-")
        self.current_y_label = QtWidgets.QLabel("-")
        self.current_z_label = QtWidgets.QLabel("-")
        
        pos_layout.addRow("X:", self.current_x_label)
        pos_layout.addRow("Y:", self.current_y_label)
        pos_layout.addRow("Z:", self.current_z_label)
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)
        
        # Refresh button
        self.refresh_btn = QtWidgets.QPushButton("Refresh Position")
        layout.addWidget(self.refresh_btn)
        
        layout.addStretch(1)
        
        # Connect signals
        self.move_btn.clicked.connect(self._move_to_position)
        self.refresh_btn.clicked.connect(self._refresh_position)
        
    def _move_to_position(self):
        """Move stage and focus to the specified positions."""
        try:
            x = self.x_spin.value()
            y = self.y_spin.value()
            z = self.z_spin.value()
            
            if self.stage and hasattr(self.stage, 'move_to'):
                self.stage.move_to(x, y)
                
            if self.focus and hasattr(self.focus, 'move_to'):
                self.focus.move_to(z)
                
            self._refresh_position()
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Move Error", f"Failed to move: {e}")
    
    def _refresh_position(self):
        """Refresh the current position display."""
        try:
            x, y, z = 0.0, 0.0, 0.0
            
            if self.stage and hasattr(self.stage, 'get_position'):
                pos = self.stage.get_position()
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    x, y = float(pos[0]), float(pos[1])
                else:
                    x = float(pos) if pos is not None else 0.0
                    
            if self.focus and hasattr(self.focus, 'get_position'):
                z = float(self.focus.get_position()) if self.focus.get_position() is not None else 0.0
                
            self.current_x_label.setText(f"{x:.3f}")
            self.current_y_label.setText(f"{y:.3f}")
            self.current_z_label.setText(f"{z:.3f}")
            
            self.position_changed.emit(x, y, z)
            
        except Exception as e:
            self.current_x_label.setText("Error")
            self.current_y_label.setText("Error")
            self.current_z_label.setText("Error")
    
    def set_stage(self, stage: StageXY):
        """Set the stage device."""
        self.stage = stage
        self._refresh_position()
    
    def set_focus(self, focus: FocusZ):
        """Set the focus device."""
        self.focus = focus
        self._refresh_position()