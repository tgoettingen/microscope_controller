from __future__ import annotations

from PyQt6 import QtWidgets, QtCore

from core.multiaxis import AxisConfig

# Import dialogs ONLY — these must NOT import gui.tabs.*
from gui.dialogs.motor_axis_dialog import MotorAxisDialog
from gui.dialogs.channel_axis_dialog import ChannelAxisDialog
from gui.dialogs.round_axis_dialog import RoundAxisDialog


class MultiViewControlTab(QtWidgets.QWidget):
    """Multi-view camera scan control panel.

    UI/behavior mirrors the Multi-Axis panel (axis list + management + run/stop)
    but is intended for camera capture runs that populate the Multi View Camera
    grid.
    """

    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None, config_path=None):
        super().__init__(parent)
        self._config_path = config_path
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self.axis_list = QtWidgets.QListWidget()
        layout.addWidget(QtWidgets.QLabel("Defined Axes:"))
        layout.addWidget(self.axis_list)

        # Buttons for axis management: add/remove/edit and reorder
        btns = QtWidgets.QHBoxLayout()
        self.add_axis_btn = QtWidgets.QPushButton("Add Axis")
        self.edit_axis_btn = QtWidgets.QPushButton("Edit Selected")
        self.remove_axis_btn = QtWidgets.QPushButton("Remove Selected")
        self.up_axis_btn = QtWidgets.QPushButton("Move Up")
        self.down_axis_btn = QtWidgets.QPushButton("Move Down")
        btns.addWidget(self.add_axis_btn)
        btns.addWidget(self.edit_axis_btn)
        btns.addWidget(self.remove_axis_btn)
        btns.addWidget(self.up_axis_btn)
        btns.addWidget(self.down_axis_btn)
        layout.addLayout(btns)

        run_btns = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Run Multi View")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        run_btns.addWidget(self.start_btn)
        run_btns.addWidget(self.stop_btn)
        layout.addLayout(run_btns)

        layout.addStretch(1)

        self.add_axis_btn.clicked.connect(self._add_axis_dialog)
        self.edit_axis_btn.clicked.connect(self._edit_selected)
        self.remove_axis_btn.clicked.connect(self._remove_selected)
        self.up_axis_btn.clicked.connect(self._move_selected_up)
        self.down_axis_btn.clicked.connect(self._move_selected_down)
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

    def _add_axis_dialog(self) -> None:
        dlg = QtWidgets.QInputDialog(self)
        dlg.setComboBoxItems(["X", "Y", "Z", "Channel", "Round"])
        dlg.setLabelText("Select axis type:")
        dlg.setWindowTitle("Add Axis")

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        axis_type = dlg.textValue()

        if axis_type in ("X", "Y", "Z"):
            d = MotorAxisDialog(axis_type, self, config_path=self._config_path)
        elif axis_type == "Channel":
            d = ChannelAxisDialog(self)
        elif axis_type == "Round":
            d = RoundAxisDialog(self)
        else:
            return

        if d.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            cfg: AxisConfig = d.get_config()
            item = QtWidgets.QListWidgetItem(cfg.label())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, cfg)
            self.axis_list.addItem(item)

    def _edit_selected(self) -> None:
        items = self.axis_list.selectedItems()
        if not items:
            return
        item = items[0]
        cfg = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if cfg is None:
            return

        if cfg.axis_type in ("X", "Y", "Z"):
            dlg = MotorAxisDialog(cfg.axis_type, config=cfg,
                                  config_path=self._config_path)
        elif cfg.axis_type == "Channel":
            dlg = ChannelAxisDialog(self)
        elif cfg.axis_type == "Round":
            dlg = RoundAxisDialog(self)
        else:
            return

        # Populate motor dialog fields if present
        try:
            if hasattr(dlg, "start_spin") and cfg.params:
                dlg.start_spin.setValue(cfg.params.get("start", dlg.start_spin.value()))
                dlg.end_spin.setValue(cfg.params.get("end", dlg.end_spin.value()))
                dlg.step_spin.setValue(cfg.params.get("step", dlg.step_spin.value()))
                dlg.wait_spin.setValue(cfg.params.get("wait", dlg.wait_spin.value()))
                motors = cfg.params.get("motors", [])
                dlg.motors_edit.setText(",".join(motors))
                mode = cfg.params.get("motor_mode", dlg.mode_combo.currentText())
                idx = dlg.mode_combo.findText(mode)
                if idx >= 0:
                    dlg.mode_combo.setCurrentIndex(idx)
                if "pre_pos" in cfg.params and cfg.params["pre_pos"] is not None:
                    dlg.pre_pos_spin.setValue(cfg.params["pre_pos"])
                if "post_pos" in cfg.params and cfg.params["post_pos"] is not None:
                    dlg.post_pos_spin.setValue(cfg.params["post_pos"])
        except Exception:
            pass

        # Populate wait where supported.
        try:
            if hasattr(dlg, "wait_spin") and getattr(cfg, "params", None):
                if "wait" in cfg.params:
                    dlg.wait_spin.setValue(cfg.params.get("wait", dlg.wait_spin.value()))
        except Exception:
            pass

        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_cfg = dlg.get_config()
            item.setText(new_cfg.label())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, new_cfg)

    def _move_selected_up(self) -> None:
        items = self.axis_list.selectedItems()
        if not items:
            return
        row = self.axis_list.row(items[0])
        if row <= 0:
            return
        item = self.axis_list.takeItem(row)
        self.axis_list.insertItem(row - 1, item)
        item.setSelected(True)

    def _move_selected_down(self) -> None:
        items = self.axis_list.selectedItems()
        if not items:
            return
        row = self.axis_list.row(items[0])
        if row >= self.axis_list.count() - 1:
            return
        item = self.axis_list.takeItem(row)
        self.axis_list.insertItem(row + 1, item)
        item.setSelected(True)

    def _remove_selected(self) -> None:
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
