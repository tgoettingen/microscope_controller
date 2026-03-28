from PyQt6 import QtWidgets

from core.multiaxis import AxisConfig


class DetectorAxisDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detector Axis Settings")

        layout = QtWidgets.QFormLayout(self)

        self.wait_spin = QtWidgets.QDoubleSpinBox()
        self.wait_spin.setRange(0.0, 10.0)
        self.wait_spin.setValue(0.01)

        # Detector scaling is taken from the device config JSON.
        # This axis is kept as a timing/no-op axis only.
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
            axis_type="Detector",
            params={
                "wait": self.wait_spin.value(),
            },
        )