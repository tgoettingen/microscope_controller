from PyQt6 import QtWidgets
import pyqtgraph as pg


class PlotPanel(QtWidgets.QWidget):
    """Wrapper for a pyqtgraph PlotWidget."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.plot_widget = pg.PlotWidget()
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot_widget)
