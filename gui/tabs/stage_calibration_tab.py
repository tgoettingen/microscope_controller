from __future__ import annotations

from PyQt6 import QtWidgets, QtCore, QtGui
from devices.base import StageXY


class StageCalibrationTab(QtWidgets.QWidget):
    """User-friendly stage calibration panel with step-by-step workflow."""
    
    calibration_saved = QtCore.pyqtSignal(float, float)  # x_scale, y_scale
    
    def __init__(self, stage: StageXY = None, config_path: str = None, parent=None):
        super().__init__(parent)
        self.stage = stage
        self.config_path = config_path
        self._current_step = 1
        self._build_ui()
        
    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Title and instructions
        title_label = QtWidgets.QLabel("Stage Calibration")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2196F3;")
        layout.addWidget(title_label)
        
        instruction_label = QtWidgets.QLabel("Calibrate stage movement by measuring known distances")
        instruction_label.setStyleSheet("color: #666; font-style: italic;")
        instruction_label.setWordWrap(True)
        layout.addWidget(instruction_label)
        
        # Step indicator
        self._build_step_indicator(layout)
        
        # Step 1: Set Reference
        self.step1_group = QtWidgets.QGroupBox("Step 1: Set Reference Point")
        self.step1_group.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid #2196F3; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        step1_layout = QtWidgets.QVBoxLayout()
        
        ref_info = QtWidgets.QLabel("Move stage to starting position and click Set Reference")
        ref_info.setStyleSheet("color: #555; font-size: 11px;")
        ref_info.setWordWrap(True)
        step1_layout.addWidget(ref_info)
        
        ref_position_layout = QtWidgets.QHBoxLayout()
        self.ref_x_label = QtWidgets.QLabel("X: -")
        self.ref_x_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        self.ref_y_label = QtWidgets.QLabel("Y: -")
        self.ref_y_label.setStyleSheet("font-family: monospace; font-weight: bold;")
        ref_position_layout.addWidget(self.ref_x_label)
        ref_position_layout.addWidget(self.ref_y_label)
        ref_position_layout.addStretch()
        step1_layout.addLayout(ref_position_layout)
        
        self.set_ref_btn = QtWidgets.QPushButton("📍 Set Reference Point")
        self.set_ref_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.set_ref_btn.clicked.connect(self._set_reference)
        step1_layout.addWidget(self.set_ref_btn)
        
        self.step1_group.setLayout(step1_layout)
        layout.addWidget(self.step1_group)
        
        # Step 2: Move and Measure
        self.step2_group = QtWidgets.QGroupBox("Step 2: Move and Measure")
        self.step2_group.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid #ddd; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        step2_layout = QtWidgets.QVBoxLayout()
        
        move_info = QtWidgets.QLabel("Move stage by a known distance and enter the measurement")
        move_info.setStyleSheet("color: #555; font-size: 11px;")
        move_info.setWordWrap(True)
        step2_layout.addWidget(move_info)
        
        # Current position display
        current_pos_layout = QtWidgets.QHBoxLayout()
        current_pos_layout.addWidget(QtWidgets.QLabel("Current Position:"))
        self.cur_x_label = QtWidgets.QLabel("X: -")
        self.cur_x_label.setStyleSheet("font-family: monospace;")
        self.cur_y_label = QtWidgets.QLabel("Y: -")
        self.cur_y_label.setStyleSheet("font-family: monospace;")
        current_pos_layout.addWidget(self.cur_x_label)
        current_pos_layout.addWidget(self.cur_y_label)
        current_pos_layout.addStretch()
        step2_layout.addLayout(current_pos_layout)
        
        # Physical distance input
        distance_layout = QtWidgets.QFormLayout()
        distance_layout.setSpacing(8)
        
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
        
        distance_layout.addRow("Distance X:", self.phys_x_spin)
        distance_layout.addRow("Distance Y:", self.phys_y_spin)
        step2_layout.addLayout(distance_layout)
        
        self.refresh_btn = QtWidgets.QPushButton("🔄 Refresh Position")
        self.refresh_btn.setStyleSheet("background-color: #FF9800; color: white; padding: 6px; border-radius: 4px;")
        self.refresh_btn.clicked.connect(self._refresh_position)
        step2_layout.addWidget(self.refresh_btn)
        
        self.step2_group.setLayout(step2_layout)
        layout.addWidget(self.step2_group)
        
        # Step 3: Calculate and Save
        self.step3_group = QtWidgets.QGroupBox("Step 3: Calculate & Save")
        self.step3_group.setStyleSheet("QGroupBox { font-weight: bold; border: 2px solid #ddd; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }")
        step3_layout = QtWidgets.QVBoxLayout()
        
        # Results display
        results_layout = QtWidgets.QFormLayout()
        results_layout.setSpacing(8)
        
        self.scale_x_label = QtWidgets.QLabel("-")
        self.scale_x_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #4CAF50;")
        self.scale_y_label = QtWidgets.QLabel("-")
        self.scale_y_label.setStyleSheet("font-family: monospace; font-weight: bold; color: #4CAF50;")
        
        results_layout.addRow("X Scale:", self.scale_x_label)
        results_layout.addRow("Y Scale:", self.scale_y_label)
        step3_layout.addLayout(results_layout)
        
        # Action buttons
        button_layout = QtWidgets.QHBoxLayout()
        
        self.calc_btn = QtWidgets.QPushButton("📊 Calculate")
        self.calc_btn.setStyleSheet("background-color: #9C27B0; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.calc_btn.clicked.connect(self._calculate_scale)
        
        self.save_btn = QtWidgets.QPushButton("💾 Save Config")
        self.save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; border-radius: 4px;")
        self.save_btn.clicked.connect(self._save_calibration)
        
        button_layout.addWidget(self.calc_btn)
        button_layout.addWidget(self.save_btn)
        step3_layout.addLayout(button_layout)
        
        self.step3_group.setLayout(step3_layout)
        layout.addWidget(self.step3_group)
        
        # Status bar
        self.status_label = QtWidgets.QLabel("Ready to start calibration")
        self.status_label.setStyleSheet("background-color: #f5f5f5; padding: 8px; border-radius: 4px; color: #666;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        layout.addStretch(1)
        
        # Initialize
        self._ref_x = 0.0
        self._ref_y = 0.0
        self._update_step_styles()
        self._refresh_position()
    
    def _build_step_indicator(self, layout):
        """Build the step indicator widget."""
        steps_layout = QtWidgets.QHBoxLayout()
        steps_layout.setSpacing(4)
        
        for i in range(1, 4):
            step_num = QtWidgets.QLabel(str(i))
            step_num.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            step_num.setFixedSize(24, 24)
            step_num.setStyleSheet(f"""
                QLabel {{
                    background-color: {'#2196F3' if i == 1 else '#ddd'};
                    color: white;
                    border-radius: 12px;
                    font-weight: bold;
                }}
            """)
            step_num.setObjectName(f"step_{i}")
            steps_layout.addWidget(step_num)
            
            if i < 3:
                arrow = QtWidgets.QLabel("→")
                arrow.setStyleSheet("color: #999; font-weight: bold;")
                steps_layout.addWidget(arrow)
        
        steps_layout.addStretch()
        layout.addLayout(steps_layout)
    
    def _update_step_styles(self):
        """Update step indicator and group styles based on current step."""
        for i in range(1, 4):
            step_num = self.findChild(QtWidgets.QLabel, f"step_{i}")
            if step_num:
                if i <= self._current_step:
                    step_num.setStyleSheet("""
                        QLabel {
                            background-color: #2196F3;
                            color: white;
                            border-radius: 12px;
                            font-weight: bold;
                        }
                    """)
                else:
                    step_num.setStyleSheet("""
                        QLabel {
                            background-color: #ddd;
                            color: #999;
                            border-radius: 12px;
                            font-weight: bold;
                        }
                    """)
        
        # Update group border colors
        active_color = "#2196F3"
        inactive_color = "#ddd"
        
        self.step1_group.setStyleSheet(f"QGroupBox {{ font-weight: bold; border: 2px solid {active_color if self._current_step >= 1 else inactive_color}; border-radius: 5px; margin-top: 10px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        self.step2_group.setStyleSheet(f"QGroupBox {{ font-weight: bold; border: 2px solid {active_color if self._current_step >= 2 else inactive_color}; border-radius: 5px; margin-top: 10px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
        self.step3_group.setStyleSheet(f"QGroupBox {{ font-weight: bold; border: 2px solid {active_color if self._current_step >= 3 else inactive_color}; border-radius: 5px; margin-top: 10px; }} QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 3px; }}")
    
    def _set_reference(self):
        """Set the current position as reference point."""
        try:
            if self.stage and hasattr(self.stage, 'get_position'):
                pos = self.stage.get_position()
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    self._ref_x = float(pos[0])
                    self._ref_y = float(pos[1])
                else:
                    self._ref_x = float(pos) if pos is not None else 0.0
                    self._ref_y = 0.0
                    
                self.ref_x_label.setText(f"X: {self._ref_x:.1f}")
                self.ref_y_label.setText(f"Y: {self._ref_y:.1f}")
                
                self._current_step = 2
                self._update_step_styles()
                self.status_label.setText("✓ Reference set! Now move the stage and measure the distance.")
                self.status_label.setStyleSheet("background-color: #E8F5E9; padding: 8px; border-radius: 4px; color: #2E7D32;")
                    
        except Exception as e:
            self.status_label.setText(f"❌ Error setting reference: {e}")
            self.status_label.setStyleSheet("background-color: #FFEBEE; padding: 8px; border-radius: 4px; color: #C62828;")
    
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
                    
                self.cur_x_label.setText(f"X: {x:.1f}")
                self.cur_y_label.setText(f"Y: {y:.1f}")
                
                self.status_label.setText("Position refreshed")
                self.status_label.setStyleSheet("background-color: #E3F2FD; padding: 8px; border-radius: 4px; color: #1565C0;")
                
        except Exception as e:
            self.cur_x_label.setText("X: Error")
            self.cur_y_label.setText("Y: Error")
            self.status_label.setText(f"❌ Error refreshing position: {e}")
            self.status_label.setStyleSheet("background-color: #FFEBEE; padding: 8px; border-radius: 4px; color: #C62828;")
    
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
                    self.scale_x_label.setText(f"{scale_x:.2f} steps/mm")
                else:
                    self.scale_x_label.setText("N/A")
                    
                if phys_y > 0:
                    scale_y = delta_y / phys_y
                    self.scale_y_label.setText(f"{scale_y:.2f} steps/mm")
                else:
                    self.scale_y_label.setText("N/A")
                
                self._current_step = 3
                self._update_step_styles()
                self.status_label.setText("✓ Scale calculated! Review results and save to config.")
                self.status_label.setStyleSheet("background-color: #E8F5E9; padding: 8px; border-radius: 4px; color: #2E7D32;")
                    
        except Exception as e:
            self.status_label.setText(f"❌ Error calculating scale: {e}")
            self.status_label.setStyleSheet("background-color: #FFEBEE; padding: 8px; border-radius: 4px; color: #C62828;")
    
    def _save_calibration(self):
        """Save the calculated scale to the config file."""
        try:
            import json
            from pathlib import Path
            
            if not self.config_path:
                self.status_label.setText("❌ No config file specified")
                self.status_label.setStyleSheet("background-color: #FFEBEE; padding: 8px; border-radius: 4px; color: #C62828;")
                return
                
            # Parse scale from labels
            scale_x_text = self.scale_x_label.text()
            scale_y_text = self.scale_y_label.text()
            
            if "N/A" in scale_x_text or "N/A" in scale_y_text:
                self.status_label.setText("❌ Please calculate valid scale values first")
                self.status_label.setStyleSheet("background-color: #FFEBEE; padding: 8px; border-radius: 4px; color: #C62828;")
                return
                
            try:
                scale_x = float(scale_x_text.split()[0])
                scale_y = float(scale_y_text.split()[0])
            except (ValueError, IndexError):
                self.status_label.setText("❌ Failed to parse scale values")
                self.status_label.setStyleSheet("background-color: #FFEBEE; padding: 8px; border-radius: 4px; color: #C62828;")
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
                
            self.status_label.setText(f"✓ Calibration saved! X: {scale_x:.2f}, Y: {scale_y:.2f} steps/mm")
            self.status_label.setStyleSheet("background-color: #E8F5E9; padding: 8px; border-radius: 4px; color: #2E7D32;")
                
            self.calibration_saved.emit(scale_x, scale_y)
            
        except Exception as e:
            self.status_label.setText(f"❌ Error saving calibration: {e}")
            self.status_label.setStyleSheet("background-color: #FFEBEE; padding: 8px; border-radius: 4px; color: #C62828;")
    
    def set_stage(self, stage: StageXY):
        """Set the stage device."""
        self.stage = stage
        self._refresh_position()
    
    def set_config_path(self, config_path: str):
        """Set the config file path."""
        self.config_path = config_path