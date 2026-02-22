from PyQt6 import QtWidgets

from core.multiaxis import AxisConfig


class RoundAxisDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Round Axis Settings")

        layout = QtWidgets.QFormLayout(self)

        self.rounds_spin = QtWidgets.QSpinBox()
        self.rounds_spin.setRange(1, 100000)
        self.rounds_spin.setValue(1)

        layout.addRow("Rounds", self.rounds_spin)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        layout.addRow(btns)

        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

    def get_config(self) -> AxisConfig:
        return AxisConfig(
            axis_type="Round",
            params={"n_rounds": self.rounds_spin.value()},
        )