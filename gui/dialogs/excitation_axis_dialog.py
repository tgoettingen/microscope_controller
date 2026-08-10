from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QComboBox, QLineEdit

from core.multiaxis import AxisConfig


class ExcitationAxisDialog(QtWidgets.QDialog):
    """Dialog for configuring excitation axis settings in multi-axis scans."""
    
    def __init__(self, parent=None, excitation_name: str | None = None, excitation_devices=None):
        super().__init__(parent)
        self.excitation_name = excitation_name
        self.excitation_devices = excitation_devices if excitation_devices else []
        
        if excitation_name:
            self.setWindowTitle(f"Excitation Axis Settings — {excitation_name}")
        else:
            self.setWindowTitle("Excitation Axis Settings")

        layout = QtWidgets.QFormLayout(self)

        # Excitation device selector (if multiple devices available)
        if len(self.excitation_devices) > 1:
            self.device_combo = QtWidgets.QComboBox()
            for i, device in enumerate(self.excitation_devices):
                device_name = getattr(device, 'name', f"Device {i}")
                device_type = type(device).__name__
                self.device_combo.addItem(f"{device_name} ({device_type})", device_name)
            layout.addRow("Excitation Source", self.device_combo)
        elif excitation_name:
            layout.addRow("Excitation Source", QtWidgets.QLabel(str(excitation_name)))

        # Excitation states configuration
        self.on_off_combo = QtWidgets.QComboBox()
        self.on_off_combo.addItem("ON/OFF (alternating)", "alternating")
        self.on_off_combo.addItem("ON only", "on_only")
        self.on_off_combo.addItem("OFF only", "off_only")
        self.on_off_combo.addItem("Custom sequence", "custom")
        layout.addRow("Excitation Pattern", self.on_off_combo)

        # Custom sequence text field (hidden by default)
        self.custom_sequence_edit = QtWidgets.QLineEdit()
        self.custom_sequence_edit.setPlaceholderText("e.g., True,False,True,False")
        self.custom_sequence_edit.setVisible(False)
        layout.addRow("Custom Sequence", self.custom_sequence_edit)

        # Wait time after state change
        self.wait_spin = QtWidgets.QDoubleSpinBox()
        self.wait_spin.setRange(0.0, 10.0)
        self.wait_spin.setValue(0.1)
        self.wait_spin.setSingleStep(0.01)
        layout.addRow("Wait after state change [s]", self.wait_spin)

        # Channel selection (optional)
        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setRange(0, 7)
        self.channel_spin.setValue(0)
        layout.addRow("Channel (0-7)", self.channel_spin)

        # Connect combo box to show/hide custom sequence
        self.on_off_combo.currentIndexChanged.connect(self._on_pattern_changed)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def _on_pattern_changed(self, index):
        """Show/hide custom sequence field based on pattern selection."""
        pattern = self.on_off_combo.currentData()
        self.custom_sequence_edit.setVisible(pattern == "custom")

    def get_config(self) -> AxisConfig:
        """Get the axis configuration from dialog values."""
        pattern = self.on_off_combo.currentData()
        
        # Determine states based on pattern
        if pattern == "alternating":
            states = [True, False]
        elif pattern == "on_only":
            states = [True]
        elif pattern == "off_only":
            states = [False]
        elif pattern == "custom":
            try:
                # Parse custom sequence like "True,False,True,False"
                state_str = self.custom_sequence_edit.text().strip()
                if state_str:
                    states = [s.strip().lower() in ['true', '1', 'on'] for s in state_str.split(',')]
                else:
                    states = [True, False]  # Fallback
            except Exception:
                states = [True, False]  # Fallback on parse error
        else:
            states = [True, False]

        params = {
            "states": states,
            "wait": self.wait_spin.value(),
            "channel": self.channel_spin.value(),
        }
        
        # Get excitation device name from combo or from constructor
        if hasattr(self, 'device_combo'):
            params["excitation"] = self.device_combo.currentData()
        elif self.excitation_name:
            params["excitation"] = self.excitation_name
            
        return AxisConfig(
            axis_type="Excitation",
            params=params,
        )