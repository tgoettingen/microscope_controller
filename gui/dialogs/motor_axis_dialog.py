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

        layout.addRow("Start", self.start_spin)
        layout.addRow("End", self.end_spin)
        layout.addRow("Step", self.step_spin)
        layout.addRow("Wait [s]", self.wait_spin)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_config(self) -> AxisConfig:
        return AxisConfig(
            axis_type=self.axis_type,
            params={
                "start": self.start_spin.value(),
                "end": self.end_spin.value(),
                "step": self.step_spin.value(),
                "wait": self.wait_spin.value(),
            },
        )