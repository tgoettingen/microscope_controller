from PyQt6 import QtWidgets
import pyqtgraph as pg


class CameraPanel(QtWidgets.QWidget):
    """Simple wrapper providing an ImageView for camera display."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_view = pg.ImageView()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.image_view)
