from PyQt6 import QtWidgets


class DetectorControlPanel(QtWidgets.QWidget):
    """Holds detector visibility/stream controls in a vertical layout."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.group = QtWidgets.QGroupBox("Detectors")
        # Foldable/collapsible panel
        try:
            self.group.setCheckable(True)
            self.group.setChecked(True)
        except Exception:
            pass

        # Put controls inside an inner widget so we can hide/show them
        self._content = QtWidgets.QWidget(self.group)
        self.vlayout = QtWidgets.QVBoxLayout(self._content)
        self.vlayout.setContentsMargins(4, 4, 4, 4)

        g_layout = QtWidgets.QVBoxLayout(self.group)
        g_layout.setContentsMargins(6, 12, 6, 6)
        g_layout.addWidget(self._content)

        try:
            self.group.toggled.connect(self._content.setVisible)
        except Exception:
            pass

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.group)

    def add_control_row(self, widget):
        self.vlayout.addWidget(widget)
