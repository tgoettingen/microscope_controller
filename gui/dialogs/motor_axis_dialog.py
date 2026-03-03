from PyQt6 import QtWidgets

from core.multiaxis import AxisConfig


class MotorAxisDialog(QtWidgets.QDialog):
    def __init__(self, axis_type: str, parent=None):
        super().__init__(parent)
        self.axis_type = axis_type
        self.setWindowTitle(f"{axis_type} Axis Settings")

        layout = QtWidgets.QFormLayout(self)

        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.wait_spin = QtWidgets.QDoubleSpinBox()

        for s in (self.start_spin, self.end_spin):
            s.setRange(-1e6, 1e6)
        self.step_spin.setRange(0.001, 1e6)
        self.wait_spin.setRange(0.0, 10.0)

        self.start_spin.setValue(0)
        self.end_spin.setValue(1000)
        self.step_spin.setValue(100)
        self.wait_spin.setValue(0.05)

        # multiple motors and mode
        self.motors_edit = QtWidgets.QLineEdit()
        self.motors_edit.setPlaceholderText("Comma-separated device names (e.g. stage,focus)")
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["sequential", "synchronized"]) 

        layout.addRow("Start", self.start_spin)
        layout.addRow("End", self.end_spin)
        layout.addRow("Step", self.step_spin)
        layout.addRow("Wait [s]", self.wait_spin)
        layout.addRow("Motors", self.motors_edit)
        layout.addRow("Motor mode", self.mode_combo)

        # Synchronized wait parameters
        self.sync_timeout_spin = QtWidgets.QDoubleSpinBox()
        self.sync_timeout_spin.setRange(0.01, 3600.0)
        self.sync_timeout_spin.setValue(5.0)
        self.sync_poll_spin = QtWidgets.QDoubleSpinBox()
        self.sync_poll_spin.setRange(0.0001, 10.0)
        self.sync_poll_spin.setDecimals(4)
        self.sync_poll_spin.setValue(0.01)
        self.sync_tol_spin = QtWidgets.QDoubleSpinBox()
        self.sync_tol_spin.setRange(1e-9, 1.0)
        self.sync_tol_spin.setDecimals(6)
        self.sync_tol_spin.setValue(1e-3)

        layout.addRow("Sync timeout [s]", self.sync_timeout_spin)
        layout.addRow("Sync poll [s]", self.sync_poll_spin)
        layout.addRow("Sync tolerance", self.sync_tol_spin)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_config(self) -> AxisConfig:
        motors_raw = self.motors_edit.text().strip()
        motors = [m.strip() for m in motors_raw.split(",") if m.strip()]
        return AxisConfig(
            axis_type=self.axis_type,
            params={
                "start": self.start_spin.value(),
                "end": self.end_spin.value(),
                "step": self.step_spin.value(),
                "wait": self.wait_spin.value(),
                "motors": motors,
                "motor_mode": self.mode_combo.currentText(),
                "sync_timeout": self.sync_timeout_spin.value(),
                "sync_poll": self.sync_poll_spin.value(),
                "sync_tol": self.sync_tol_spin.value(),
            },
        )