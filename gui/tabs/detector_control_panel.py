from PyQt6 import QtWidgets


class DetectorControlPanel(QtWidgets.QWidget):
    """Holds detector visibility/stream controls in a vertical layout."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Keep this panel compact even when placed in a tall dock.
        # The content can scroll instead of forcing the dock to grow.
        self._max_panel_height = 320
        self.group = QtWidgets.QGroupBox("Detectors")
        # Foldable/collapsible panel
        try:
            self.group.setCheckable(True)
            self.group.setChecked(True)
        except Exception:
            pass

        # Put controls inside an inner widget so we can hide/show them
        self._content = QtWidgets.QWidget()
        self.vlayout = QtWidgets.QVBoxLayout(self._content)
        self.vlayout.setContentsMargins(4, 4, 4, 4)
        try:
            self.vlayout.setSpacing(2)
        except Exception:
            pass

        # Make the control list scroll when there are many detectors.
        self._scroll = QtWidgets.QScrollArea(self.group)
        self._scroll.setWidgetResizable(True)
        try:
            self._scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        except Exception:
            pass
        self._scroll.setWidget(self._content)

        g_layout = QtWidgets.QVBoxLayout(self.group)
        g_layout.setContentsMargins(6, 12, 6, 6)
        g_layout.addWidget(self._scroll)

        try:
            self.group.toggled.connect(self._scroll.setVisible)
        except Exception:
            pass

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.group)

        # Size policy: prefer not to expand vertically.
        try:
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
            self.group.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
            self._scroll.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
            self.setMaximumHeight(self._max_panel_height)
            self.group.setMaximumHeight(self._max_panel_height)
            self._scroll.setMaximumHeight(self._max_panel_height)
        except Exception:
            pass

    def add_control_row(self, widget):
        self.vlayout.addWidget(widget)

        # Keep compact height as controls are added dynamically.
        try:
            self.adjustSize()
        except Exception:
            pass
