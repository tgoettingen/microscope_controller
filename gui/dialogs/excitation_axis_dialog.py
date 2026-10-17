from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QComboBox, QLineEdit

from core.multiaxis import AxisConfig


class ExcitationAxisDialog(QtWidgets.QDialog):
    """Dialog for configuring excitation axis settings in multi-axis scans."""
    
    def __init__(
        self,
        parent=None,
        excitation_name: str | None = None,
        excitation_devices=None,
        config: AxisConfig | None = None,
    ):
        super().__init__(parent)
        if config is not None:
            excitation_name = config.params.get("excitation", excitation_name)
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
            if excitation_name is not None:
                device_index = self.device_combo.findData(excitation_name)
                if device_index >= 0:
                    self.device_combo.setCurrentIndex(device_index)
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
        self.wait_spin.setValue(0.0)
        self.wait_spin.setSingleStep(0.01)
        layout.addRow("Wait after state change [s]", self.wait_spin)

        # Channel selection (optional)
        self.channel_spin = QtWidgets.QSpinBox()
        self.channel_spin.setRange(0, 7)
        self.channel_spin.setValue(0)
        layout.addRow("Channel (0-7)", self.channel_spin)

        if config is not None:
            params = config.params
            states = [bool(state) for state in params.get("states", [True, False])]
            patterns = {
                (True, False): "alternating",
                (True,): "on_only",
                (False,): "off_only",
            }
            pattern = patterns.get(tuple(states), "custom")
            pattern_index = self.on_off_combo.findData(pattern)
            if pattern_index >= 0:
                self.on_off_combo.setCurrentIndex(pattern_index)
            if pattern == "custom":
                self.custom_sequence_edit.setText(
                    ",".join("True" if state else "False" for state in states)
                )
            self.wait_spin.setValue(float(params.get("wait", self.wait_spin.value())))
            self.channel_spin.setValue(int(params.get("channel", self.channel_spin.value())))

        # Connect combo box to show/hide custom sequence
        self.on_off_combo.currentIndexChanged.connect(self._on_pattern_changed)
        self._on_pattern_changed(self.on_off_combo.currentIndex())

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