from PyQt6 import QtWidgets


class DetectorControlPanel(QtWidgets.QWidget):
    """Holds detector visibility/stream controls in a vertical layout."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.group = QtWidgets.QGroupBox("Detectors")
        self.vlayout = QtWidgets.QVBoxLayout(self.group)
        self.vlayout.setContentsMargins(4, 4, 4, 4)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.group)

    def add_control_row(self, widget):
        self.vlayout.addWidget(widget)
