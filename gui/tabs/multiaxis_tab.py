from PyQt6 import QtWidgets, QtCore

from core.multiaxis import AxisConfig

# Import dialogs ONLY — these must NOT import gui.tabs.*
from gui.dialogs.motor_axis_dialog import MotorAxisDialog
from gui.dialogs.channel_axis_dialog import ChannelAxisDialog
from gui.dialogs.detector_axis_dialog import DetectorAxisDialog
from gui.dialogs.round_axis_dialog import RoundAxisDialog


class MultiAxisTab(QtWidgets.QWidget):
    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        self.axis_list = QtWidgets.QListWidget()
        layout.addWidget(QtWidgets.QLabel("Defined Axes:"))
        layout.addWidget(self.axis_list)

        btns = QtWidgets.QHBoxLayout()
        self.add_axis_btn = QtWidgets.QPushButton("Add Axis")
        self.remove_axis_btn = QtWidgets.QPushButton("Remove Selected")
        btns.addWidget(self.add_axis_btn)
        btns.addWidget(self.remove_axis_btn)
        layout.addLayout(btns)

        run_btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Run Multi‑Axis")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        run_btns.addWidget(self.start_btn)
        run_btns.addWidget(self.stop_btn)
        layout.addLayout(run_btns)

        layout.addStretch(1)

        self.add_axis_btn.clicked.connect(self._add_axis_dialog)
        self.remove_axis_btn.clicked.connect(self._remove_selected)
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

    def _add_axis_dialog(self):
        dlg = QtWidgets.QInputDialog(self)
        dlg.setComboBoxItems(["X", "Y", "Z", "Channel", "Detector", "Round"])
        dlg.setLabelText("Select axis type:")
        dlg.setWindowTitle("Add Axis")

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        axis_type = dlg.textValue()

        if axis_type in ("X", "Y", "Z"):
            d = MotorAxisDialog(axis_type, self)
        elif axis_type == "Channel":
            d = ChannelAxisDialog(self)
        elif axis_type == "Detector":
            d = DetectorAxisDialog(self)
        elif axis_type == "Round":
            d = RoundAxisDialog(self)
        else:
            return

        if d.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            cfg: AxisConfig = d.get_config()
            item = QtWidgets.QListWidgetItem(cfg.label())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, cfg)
            self.axis_list.addItem(item)

    def _remove_selected(self):
        for item in self.axis_list.selectedItems():
            self.axis_list.takeItem(self.axis_list.row(item))

    def get_axis_configs(self) -> list[AxisConfig]:
        cfgs: list[AxisConfig] = []
        for i in range(self.axis_list.count()):
            item = self.axis_list.item(i)
            cfg = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(cfg, AxisConfig):
                cfgs.append(cfg)
        return cfgs