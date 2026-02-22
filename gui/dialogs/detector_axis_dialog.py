from PyQt6 import QtWidgets

from core.multiaxis import AxisConfig


class DetectorAxisDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detector Axis Settings")

        layout = QtWidgets.QFormLayout(self)

        self.scale_edit = QtWidgets.QLineEdit("1.0, 2.0")
        self.offset_edit = QtWidgets.QLineEdit("0.0, 0.0")
        self.wait_spin = QtWidgets.QDoubleSpinBox()
        self.wait_spin.setRange(0.0, 10.0)
        self.wait_spin.setValue(0.01)

        layout.addRow("Scales (comma-separated)", self.scale_edit)
        layout.addRow("Offsets (comma-separated)", self.offset_edit)
        layout.addRow("Wait [s]", self.wait_spin)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def get_config(self) -> AxisConfig:
        scales = [float(x.strip()) for x in self.scale_edit.text().split(",") if x.strip()]
        offsets = [float(x.strip()) for x in self.offset_edit.text().split(",") if x.strip()]
        pairs = list(zip(scales, offsets))
        return AxisConfig(
            axis_type="Detector",
            params={
                "scales": pairs,
                "wait": self.wait_spin.value(),
            },
        )