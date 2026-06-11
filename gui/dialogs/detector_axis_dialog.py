from PyQt6 import QtWidgets

from core.multiaxis import AxisConfig


class DetectorAxisDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, detector_name: str | None = None):
        super().__init__(parent)
        self.detector_name = detector_name
        if detector_name:
            self.setWindowTitle(f"Detector Axis Settings — {detector_name}")
        else:
            self.setWindowTitle("Detector Axis Settings")

        layout = QtWidgets.QFormLayout(self)

        if detector_name:
            layout.addRow("Detector", QtWidgets.QLabel(str(detector_name)))

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
        params = {
            "wait": self.wait_spin.value(),
        }
        if self.detector_name:
            params["detector"] = self.detector_name
        return AxisConfig(
            axis_type="Detector",
            params=params,
        )