from __future__ import annotations

from PyQt6 import QtWidgets, QtCore
from devices.base import StageXY


class StageCalibrationTab(QtWidgets.QWidget):
    """Stage calibration panel as a dockable tab."""
    
    calibration_saved = QtCore.pyqtSignal(float, float)  # x_scale, y_scale
    
    def __init__(self, stage: StageXY = None, config_path: str = None, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.config_path = config_path
        self._build_ui()
        
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        
        # Quick calibration section
        calib_group = QtWidgets.QGroupBox("Quick Calibration")
        calib_layout = QtWidgets.QFormLayout()
        
        # Reference position
        self.ref_x_spin = QtWidgets.QDoubleSpinBox()
        self.ref_x_spin.setRange(-1e6, 1e6)
        self.ref_x_spin.setDecimals(3)
        self.ref_x_spin.setSuffix(" steps")
        self.ref_x_spin.setValue(0.0)
        
        self.ref_y_spin = QtWidgets.QDoubleSpinBox()
        self.ref_y_spin.setRange(-1e6, 1e6)
        self.ref_y_spin.setDecimals(3)
        self.ref_y_spin.setSuffix(" steps")
        self.ref_y_spin.setValue(0.0)
        
        calib_layout.addRow("Ref X:", self.ref_x_spin)
        calib_layout.addRow("Ref Y:", self.ref_y_spin)
        
        # Current position
        self.cur_x_label = QtWidgets.QLabel("-")
        self.cur_y_label = QtWidgets.QLabel("-")
        
        calib_layout.addRow("Current X:", self.cur_x_label)
        calib_layout.addRow("Current Y:", self.cur_y_label)
        
        # Physical distance moved
        self.phys_x_spin = QtWidgets.QDoubleSpinBox()
        self.phys_x_spin.setRange(0.001, 1000.0)
        self.phys_x_spin.setDecimals(3)
        self.phys_x_spin.setSuffix(" mm")
        self.phys_x_spin.setValue(1.0)
        
        self.phys_y_spin = QtWidgets.QDoubleSpinBox()
        self.phys_y_spin.setRange(0.001, 1000.0)
        self.phys_y_spin.setDecimals(3)
        self.phys_y_spin.setSuffix(" mm")
        self.phys_y_spin.setValue(1.0)
        
        calib_layout.addRow("Physical X:", self.phys_x_spin)
        calib_layout.addRow("Physical Y:", self.phys_y_spin)
        
        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)
        
        # Calculated scale display
        scale_group = QtWidgets.QGroupBox("Calculated Scale")
        scale_layout = QtWidgets.QFormLayout()
        
        self.scale_x_label = QtWidgets.QLabel("-")
        self.scale_y_label = QtWidgets.QLabel("-")
        
        scale_layout.addRow("X Scale:", self.scale_x_label)
        scale_layout.addRow("Y Scale:", self.scale_y_label)
        
        scale_group.setLayout(scale_layout)
        layout.addWidget(scale_group)
        
        # Action buttons
        self.set_ref_btn = QtWidgets.QPushButton("Set Reference Point")
        self.calc_btn = QtWidgets.QPushButton("Calculate Scale")
        self.save_btn = QtWidgets.QPushButton("Save to Config")
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;")
        
        layout.addWidget(self.set_ref_btn)
        layout.addWidget(self.calc_btn)
        layout.addWidget(self.save_btn)
        
        # Refresh button
        self.refresh_btn = QtWidgets.QPushButton("Refresh Position")
        layout.addWidget(self.refresh_btn)
        
        layout.addStretch(1)
        
        # Connect signals
        self.set_ref_btn.clicked.connect(self._set_reference)
        self.calc_btn.clicked.connect(self._calculate_scale)
        self.save_btn.clicked.connect(self._save_calibration)
        self.refresh_btn.clicked.connect(self._refresh_position)
        
        # Initialize
        self._ref_x = 0.0
        self._ref_y = 0.0
        
    def _set_reference(self):
        """Set the current position as reference point."""
        try:
            if self.stage and hasattr(self.stage, 'get_position'):
                pos = self.stage.get_position()
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    self._ref_x = float(pos[0])
                    self._ref_y = float(pos[1])
                    self.ref_x_spin.setValue(self._ref_x)
                    self.ref_y_spin.setValue(self._ref_y)
                else:
                    self._ref_x = float(pos) if pos is not None else 0.0
                    self.ref_x_spin.setValue(self._ref_x)
                    
                QtWidgets.QMessageBox.information(self, "Reference Set", 
                    f"Reference point set to X={self._ref_x:.3f}, Y={self._ref_y:.3f}")
                    
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to set reference: {e}")
    
    def _refresh_position(self):
        """Refresh the current position display."""
        try:
            if self.stage and hasattr(self.stage, 'get_position'):
                pos = self.stage.get_position()
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    x, y = float(pos[0]), float(pos[1])
                else:
                    x = float(pos) if pos is not None else 0.0
                    y = 0.0
                    
                self.cur_x_label.setText(f"{x:.3f}")
                self.cur_y_label.setText(f"{y:.3f}")
                
        except Exception as e:
            self.cur_x_label.setText("Error")
            self.cur_y_label.setText("Error")
    
    def _calculate_scale(self):
        """Calculate the scale factor based on reference and current position."""
        try:
            if self.stage and hasattr(self.stage, 'get_position'):
                pos = self.stage.get_position()
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    cur_x, cur_y = float(pos[0]), float(pos[1])
                else:
                    cur_x = float(pos) if pos is not None else 0.0
                    cur_y = 0.0
                    
                # Calculate delta steps
                delta_x = cur_x - self._ref_x
                delta_y = cur_y - self._ref_y
                
                # Get physical distance
                phys_x = self.phys_x_spin.value()
                phys_y = self.phys_y_spin.value()
                
                # Calculate scale (steps per mm)
                if phys_x > 0:
                    scale_x = delta_x / phys_x
                    self.scale_x_label.setText(f"{scale_x:.3f} steps/mm")
                else:
                    self.scale_x_label.setText("N/A")
                    
                if phys_y > 0:
                    scale_y = delta_y / phys_y
                    self.scale_y_label.setText(f"{scale_y:.3f} steps/mm")
                else:
                    self.scale_y_label.setText("N/A")
                    
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to calculate scale: {e}")
    
    def _save_calibration(self):
        """Save the calculated scale to the config file."""
        try:
            import json
            from pathlib import Path
            
            if not self.config_path:
                QtWidgets.QMessageBox.warning(self, "No Config", "No config file specified")
                return
                
            # Parse scale from labels
            scale_x_text = self.scale_x_label.text()
            scale_y_text = self.scale_y_label.text()
            
            if "N/A" in scale_x_text or "N/A" in scale_y_text:
                QtWidgets.QMessageBox.warning(self, "Invalid Scale", "Please calculate valid scale values first")
                return
                
            try:
                scale_x = float(scale_x_text.split()[0])
                scale_y = float(scale_y_text.split()[0])
            except (ValueError, IndexError):
                QtWidgets.QMessageBox.warning(self, "Parse Error", "Failed to parse scale values")
                return
                
            # Load config
            config_path = Path(self.config_path)
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Update stage scaling
            if 'stage' in config:
                if 'scaling' not in config['stage']:
                    config['stage']['scaling'] = {}
                config['stage']['scaling']['x_scale'] = scale_x
                config['stage']['scaling']['y_scale'] = scale_y
                
            # Save config
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            QtWidgets.QMessageBox.information(self, "Saved", 
                f"Calibration saved:\nX Scale: {scale_x:.3f} steps/mm\nY Scale: {scale_y:.3f} steps/mm")
                
            self.calibration_saved.emit(scale_x, scale_y)
            
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to save calibration: {e}")
    
    def set_stage(self, stage: StageXY):
        """Set the stage device."""
        self.stage = stage
        self._refresh_position()
    
    def set_config_path(self, config_path: str):
        """Set the config file path."""
        self.config_path = config_path