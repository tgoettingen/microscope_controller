from PyQt6 import QtWidgets
import pyqtgraph as pg


class DetectorImagePanel(QtWidgets.QWidget):
    """Container widget that holds multiple per-detector ImageViews."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.container = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.container)
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(4, 4, 4, 4)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)
