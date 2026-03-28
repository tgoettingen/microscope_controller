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
    # emitted whenever the user changes which detectors are checked
    detectors_changed = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Detector selection area (main detectors available in the system)
        layout.addWidget(QtWidgets.QLabel("Available Detectors:"))
        self.detector_list = QtWidgets.QListWidget()
        self.detector_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        layout.addWidget(self.detector_list)

        # Any checkbox toggle should propagate to whoever consumes "selected detectors".
        try:
            self.detector_list.itemChanged.connect(self._emit_detectors_changed)
        except Exception:
            pass

        self.axis_list = QtWidgets.QListWidget()
        layout.addWidget(QtWidgets.QLabel("Defined Axes:"))
        layout.addWidget(self.axis_list)

        # Default X-axis selector for runs
        xsel_layout = QtWidgets.QHBoxLayout()
        xsel_layout.addWidget(QtWidgets.QLabel("Default X Axis:"))
        self.default_xaxis_combo = QtWidgets.QComboBox()
        self.default_xaxis_combo.addItem("Index")
        xsel_layout.addWidget(self.default_xaxis_combo)
        layout.addLayout(xsel_layout)

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
        self.start_btn = QtWidgets.QPushButton("Run Multi‑Axis")
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

    def set_available_detectors(self, detectors: list[str]):
        """Populate the available detector list with checkable items.

        detectors: list of detector identifiers (strings)
        """
        # Preserve prior check state where possible.
        try:
            previously_checked = set(self.get_selected_detectors())
        except Exception:
            previously_checked = set()

        try:
            self.detector_list.blockSignals(True)
            self.detector_list.clear()
            for d in detectors:
                item = QtWidgets.QListWidgetItem(d)
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(
                    QtCore.Qt.CheckState.Checked
                    if d in previously_checked
                    else QtCore.Qt.CheckState.Unchecked
                )
                item.setData(QtCore.Qt.ItemDataRole.UserRole, d)
                self.detector_list.addItem(item)
        finally:
            try:
                self.detector_list.blockSignals(False)
            except Exception:
                pass

        # Emit once after population.
        try:
            QtCore.QTimer.singleShot(0, self._emit_detectors_changed)
        except Exception:
            pass

    def get_selected_detectors(self) -> list[str]:
        selected = []
        for i in range(self.detector_list.count()):
            item = self.detector_list.item(i)
            if item.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(item.data(QtCore.Qt.ItemDataRole.UserRole))
        return selected

    def set_selected_detectors(self, detector_ids: list[str]):
        """Check the given detector ids in the available detector list."""
        wanted = set(detector_ids or [])
        try:
            self.detector_list.blockSignals(True)
            for i in range(self.detector_list.count()):
                item = self.detector_list.item(i)
                det_id = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if det_id in wanted:
                    item.setCheckState(QtCore.Qt.CheckState.Checked)
                else:
                    item.setCheckState(QtCore.Qt.CheckState.Unchecked)
        finally:
            try:
                self.detector_list.blockSignals(False)
            except Exception:
                pass

        try:
            QtCore.QTimer.singleShot(0, self._emit_detectors_changed)
        except Exception:
            pass

    def _emit_detectors_changed(self, *_args):
        try:
            self.detectors_changed.emit(self.get_selected_detectors())
        except Exception:
            pass

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
            # refresh default x-axis options when axes change
            try:
                self.refresh_default_xaxis_options()
            except Exception:
                pass

    def _edit_selected(self):
        items = self.axis_list.selectedItems()
        if not items:
            return
        item = items[0]
        cfg = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if cfg is None:
            return
        # Launch the appropriate dialog populated with current config
        if cfg.axis_type in ("X", "Y", "Z"):
            dlg = MotorAxisDialog(cfg.axis_type, config=cfg)
        elif cfg.axis_type == "Channel":
            dlg = ChannelAxisDialog(self)
        elif cfg.axis_type == "Detector":
            dlg = DetectorAxisDialog(self)
        elif cfg.axis_type == "Round":
            dlg = RoundAxisDialog(self)
        else:
            return

        # populate motor dialog fields if present
        try:
            if hasattr(dlg, 'start_spin') and cfg.params:
                dlg.start_spin.setValue(cfg.params.get('start', dlg.start_spin.value()))
                dlg.end_spin.setValue(cfg.params.get('end', dlg.end_spin.value()))
                dlg.step_spin.setValue(cfg.params.get('step', dlg.step_spin.value()))
                dlg.wait_spin.setValue(cfg.params.get('wait', dlg.wait_spin.value()))
                motors = cfg.params.get('motors', [])
                dlg.motors_edit.setText(','.join(motors))
                mode = cfg.params.get('motor_mode', dlg.mode_combo.currentText())
                idx = dlg.mode_combo.findText(mode)
                if idx >= 0:
                    dlg.mode_combo.setCurrentIndex(idx)
                # pre/post positions
                if 'pre_pos' in cfg.params and cfg.params['pre_pos'] is not None:
                    dlg.pre_pos_spin.setValue(cfg.params['pre_pos'])
                if 'post_pos' in cfg.params and cfg.params['post_pos'] is not None:
                    dlg.post_pos_spin.setValue(cfg.params['post_pos'])
        except Exception:
            pass

        # Populate detector/channel/round dialog wait values when present.
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

    def _move_selected_up(self):
        items = self.axis_list.selectedItems()
        if not items:
            return
        row = self.axis_list.row(items[0])
        if row <= 0:
            return
        item = self.axis_list.takeItem(row)
        self.axis_list.insertItem(row - 1, item)
        item.setSelected(True)
        try:
            self.refresh_default_xaxis_options()
        except Exception:
            pass

    def _move_selected_down(self):
        items = self.axis_list.selectedItems()
        if not items:
            return
        row = self.axis_list.row(items[0])
        if row >= self.axis_list.count() - 1:
            return
        item = self.axis_list.takeItem(row)
        self.axis_list.insertItem(row + 1, item)
        item.setSelected(True)
        try:
            self.refresh_default_xaxis_options()
        except Exception:
            pass

    def _remove_selected(self):
        for item in self.axis_list.selectedItems():
            self.axis_list.takeItem(self.axis_list.row(item))
        try:
            self.refresh_default_xaxis_options()
        except Exception:
            pass

    def get_axis_configs(self) -> list[AxisConfig]:
        cfgs: list[AxisConfig] = []
        for i in range(self.axis_list.count()):
            item = self.axis_list.item(i)
            cfg = item.data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(cfg, AxisConfig):
                cfgs.append(cfg)
        return cfgs

    def get_default_xaxis(self) -> str:
        try:
            return str(self.default_xaxis_combo.currentText())
        except Exception:
            return "Index"

    def refresh_default_xaxis_options(self):
        # Schedule the actual UI update to avoid nested Qt modifications
        def _do_update():
            seen = set()
            for i in range(self.axis_list.count()):
                item = self.axis_list.item(i)
                cfg = item.data(QtCore.Qt.ItemDataRole.UserRole)
                if isinstance(cfg, AxisConfig):
                    seen.add(cfg.axis_type)
            cur = self.get_default_xaxis()
            try:
                self.default_xaxis_combo.blockSignals(True)
                self.default_xaxis_combo.clear()
                self.default_xaxis_combo.addItem("Index")
                for s in sorted(seen):
                    self.default_xaxis_combo.addItem(s)
                idx = self.default_xaxis_combo.findText(cur)
                if idx >= 0:
                    self.default_xaxis_combo.setCurrentIndex(idx)
            finally:
                try:
                    self.default_xaxis_combo.blockSignals(False)
                except Exception:
                    pass

        QtCore.QTimer.singleShot(0, _do_update)