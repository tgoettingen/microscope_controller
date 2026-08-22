from PyQt6 import QtWidgets, QtCore, QtGui

from core.multiaxis import AxisConfig

# Import dialogs ONLY — these must NOT import gui.tabs.*
from gui.dialogs.motor_axis_dialog import MotorAxisDialog
from gui.dialogs.channel_axis_dialog import ChannelAxisDialog
from gui.dialogs.detector_axis_dialog import DetectorAxisDialog
from gui.dialogs.round_axis_dialog import RoundAxisDialog
from gui.dialogs.group_axis_dialog import GroupAxisDialog
from gui.dialogs.excitation_axis_dialog import ExcitationAxisDialog


class MultiAxisTab(QtWidgets.QWidget):
    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    # emitted whenever the user changes which detectors are checked
    detectors_changed = QtCore.pyqtSignal(list)
    # emitted when per-detector display-offset toggle/value changes
    detector_offset_toggled = QtCore.pyqtSignal(str, bool)
    detector_offset_value_changed = QtCore.pyqtSignal(str, float)
    # emitted when the Default X Axis combo changes
    xaxis_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, config_path=None):
        super().__init__(parent)
        self._config_path = config_path
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)  # Reduced spacing
        layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins

        # Defined axes - compact
        axis_label = QtWidgets.QLabel("Axes:")
        axis_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(axis_label)
        
        self.axis_list = QtWidgets.QListWidget()
        self.axis_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.axis_list.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.axis_list.setMaximumHeight(100)  # Limit height
        layout.addWidget(self.axis_list)

        # Default X-axis selector for runs - compact row
        xsel_layout = QtWidgets.QHBoxLayout()
        xsel_layout.setSpacing(4)
        xsel_layout.addWidget(QtWidgets.QLabel("X Axis:"))
        self.default_xaxis_combo = QtWidgets.QComboBox()
        self.default_xaxis_combo.addItem("Index")
        self.default_xaxis_combo.setMaximumWidth(80)
        self.default_xaxis_combo.currentTextChanged.connect(
            lambda text: self.xaxis_changed.emit(text)
        )
        xsel_layout.addWidget(self.default_xaxis_combo)
        xsel_layout.addStretch()
        layout.addLayout(xsel_layout)

        # Buttons for axis management - compact row
        btns = QtWidgets.QHBoxLayout()
        btns.setSpacing(3)
        
        self.add_axis_btn = QtWidgets.QPushButton("+")
        self.add_axis_btn.setMaximumWidth(30)
        self.add_axis_btn.setToolTip("Add Axis")
        self.group_axis_btn = QtWidgets.QPushButton("Grp")
        self.group_axis_btn.setMaximumWidth(35)
        self.group_axis_btn.setToolTip("Group Axes")
        self.remove_axis_btn = QtWidgets.QPushButton("-")
        self.remove_axis_btn.setMaximumWidth(30)
        self.remove_axis_btn.setToolTip("Remove Selected")
        
        btns.addWidget(self.add_axis_btn)
        btns.addWidget(self.group_axis_btn)
        btns.addWidget(self.remove_axis_btn)
        btns.addStretch()
        layout.addLayout(btns)

        # Run buttons - compact row
        run_btns = QtWidgets.QHBoxLayout()
        run_btns.setSpacing(3)
        
        self.start_btn = QtWidgets.QPushButton("Run")
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 4px;")
        self.start_btn.setMaximumWidth(50)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 4px;")
        self.stop_btn.setMaximumWidth(50)
        
        run_btns.addWidget(self.start_btn)
        run_btns.addWidget(self.stop_btn)
        run_btns.addStretch()
        layout.addLayout(run_btns)

        layout.addStretch(1)

        self.add_axis_btn.clicked.connect(self._add_axis_dialog)
        self.group_axis_btn.clicked.connect(self._group_axis)
        self.remove_axis_btn.clicked.connect(self._remove_selected)
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

        # Enable editing axis by double-clicking the list item
        self.axis_list.itemDoubleClicked.connect(lambda _: self._edit_selected())

        # Ctrl+Up / Ctrl+Down reorders the selected axis
        self.axis_list.keyPressEvent = self._axis_list_key_press

    def set_available_detectors(self, detectors: list[str]):
        """Detector selection is handled by LiveTab's detector control panel.
        
        This method is kept for compatibility but does nothing.
        """
        pass

    def get_selected_detectors(self) -> list[str]:
        """Detector selection is handled by LiveTab's detector control panel.
        
        Returns empty list for compatibility.
        """
        return []

    def get_available_detectors(self) -> list[str]:
        """Detector selection is handled by LiveTab's detector control panel.
        
        Returns empty list for compatibility.
        """
        return []

    def set_selected_detectors(self, detector_ids: list[str]):
        """Detector selection is handled by LiveTab's detector control panel.
        
        This method is kept for compatibility but does nothing.
        """
        pass

    def _emit_detectors_changed(self, *_args):
        """Detector selection is handled by LiveTab's detector control panel.
        
        This method is kept for compatibility but does nothing.
        """
        pass

    def _on_offset_spin_changed(self, detector_id: str, value: float) -> None:
        """Does nothing - detector offset is handled by LiveTab's detector control panel."""
        pass

    def set_detector_offset_state(self, detector_id: str, enabled: bool | None = None, value: float | None = None) -> None:
        """Does nothing - detector offset is handled by LiveTab's detector control panel."""
        pass

    def _add_axis_dialog(self):
        # List each detector individually (e.g. vm2, vm3) so the user can add a
        # detector-specific axis. Fall back to a generic "Detector" entry when
        # no detectors are known yet.
        det_names = self.get_available_detectors()
        items = ["X", "Y", "Z", "Channel", "Excitation"]
        if det_names:
            items += list(det_names)
        else:
            items.append("Detector")
        items.append("Round")

        dlg = QtWidgets.QInputDialog(self)
        dlg.setComboBoxItems(items)
        dlg.setLabelText("Select axis type:")
        dlg.setWindowTitle("Add Axis")

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        axis_type = dlg.textValue()

        if axis_type in ("X", "Y", "Z"):
            d = MotorAxisDialog(axis_type, parent=self, config_path=self._config_path)
        elif axis_type == "Channel":
            d = ChannelAxisDialog(self)
        elif axis_type == "Detector":
            d = DetectorAxisDialog(self)
        elif axis_type in det_names:
            d = DetectorAxisDialog(self, detector_name=axis_type)
        elif axis_type == "Excitation":
            # Try to get excitation devices from config
            excitation_devices = None
            try:
                from core.factory import load_config, build_devices
                cfg = load_config(self._config_path)
                exc_cfg = cfg.get("excitation")
                if exc_cfg:
                    # Build the devices to get the actual device objects
                    _, _, _, _, _, _, excitation = build_devices(self._config_path)
                    excitation_devices = excitation if isinstance(excitation, list) else [excitation]
            except Exception:
                pass
            
            d = ExcitationAxisDialog(self, excitation_devices=excitation_devices)
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
            dlg = MotorAxisDialog(
                cfg.axis_type,
                config=cfg,
                parent=self,
                config_path=self._config_path,
            )
        elif cfg.axis_type == "Channel":
            dlg = ChannelAxisDialog(self)
        elif cfg.axis_type == "Detector":
            dlg = DetectorAxisDialog(self, detector_name=cfg.params.get("detector"))
        elif cfg.axis_type == "Excitation":
            dlg = ExcitationAxisDialog(self)
        elif cfg.axis_type == "Round":
            dlg = RoundAxisDialog(self)
        else:
            return

        # MotorAxisDialog already restores all its fields (including the
        # hardware table) from the config passed to its constructor; only the
        # other dialog types need manual population here.
        try:
            if hasattr(dlg, 'start_spin') and not isinstance(dlg, MotorAxisDialog) and cfg.params:
                dlg.start_spin.setValue(cfg.params.get('start', dlg.start_spin.value()))
                dlg.end_spin.setValue(cfg.params.get('end', dlg.end_spin.value()))
                dlg.step_spin.setValue(cfg.params.get('step', dlg.step_spin.value()))
                dlg.wait_spin.setValue(cfg.params.get('wait', dlg.wait_spin.value()))
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

    def _remove_selected(self):
        for item in self.axis_list.selectedItems():
            self.axis_list.takeItem(self.axis_list.row(item))
        try:
            self.refresh_default_xaxis_options()
        except Exception:
            pass

    def _group_axis(self):
        """Configure how the selected axis is grouped within the scan."""
        items = self.axis_list.selectedItems()
        if not items:
            QtWidgets.QMessageBox.information(
                self, "Group Axis",
                "Select an axis to configure its grouping."
            )
            return
        item = items[0]
        row = self.axis_list.row(item)
        cfg = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if not isinstance(cfg, AxisConfig):
            return

        # All other axes are valid grouping targets (referenced by current row).
        other_axes: list[tuple[int, str]] = []
        for i in range(self.axis_list.count()):
            if i == row:
                continue
            other = self.axis_list.item(i)
            ocfg = other.data(QtCore.Qt.ItemDataRole.UserRole)
            label = ocfg._label_base() if isinstance(ocfg, AxisConfig) else other.text()
            other_axes.append((i, label))

        # Determine the current selection to preselect in the dialog. Collapse
        # and grouping are independent and may both be active.
        collapsed_now = bool(cfg.params.get("collapse_one_step"))
        if cfg.params.get("group_with_prev") and row > 0:
            current = ("axis", row - 1)
        else:
            current = ("none",)

        dlg = GroupAxisDialog(
            cfg._label_base(),
            other_axes,
            parent=self,
            mode=cfg.params.get("group_mode", "sync"),
            length=cfg.params.get("group_length", "longer"),
            current=current,
            collapsed=collapsed_now,
        )
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        choice = dlg.get_choice()
        collapsed = dlg.is_collapsed()
        mode, length = dlg.get_values()

        # Clear any existing grouping/collapse flags first.
        for key in ("group_with_prev", "group_mode", "group_length", "collapse_one_step"):
            cfg.params.pop(key, None)

        if collapsed:
            cfg.params["collapse_one_step"] = True
            cfg.params["group_mode"] = mode
            cfg.params["group_length"] = length

        if choice[0] == "axis":
            target_idx = choice[1]
            target_item = self.axis_list.item(target_idx)
            cfg.params["group_with_prev"] = True
            cfg.params["group_mode"] = mode
            cfg.params["group_length"] = length
            # Move the selected axis to sit directly below its group target so
            # the consecutive-axis grouping logic picks it up.
            taken = self.axis_list.takeItem(row)
            new_target_row = self.axis_list.row(target_item)
            self.axis_list.insertItem(new_target_row + 1, taken)
            taken.setText(cfg.label())
            taken.setData(QtCore.Qt.ItemDataRole.UserRole, cfg)
            self.axis_list.setCurrentItem(taken)
            try:
                self.refresh_default_xaxis_options()
            except Exception:
                pass
            return
        # "none" (and/or collapse-only) → update the item in place.

        item.setText(cfg.label())
        item.setData(QtCore.Qt.ItemDataRole.UserRole, cfg)
        try:
            self.refresh_default_xaxis_options()
        except Exception:
            pass

    def _axis_list_key_press(self, event):
        """Handle Ctrl+Up / Ctrl+Down to reorder the selected axis."""
        mod = event.modifiers()
        key = event.key()
        ctrl = QtCore.Qt.KeyboardModifier.ControlModifier
        if mod & ctrl:
            if key == QtCore.Qt.Key.Key_Up:
                self._move_axis(-1)
                return
            if key == QtCore.Qt.Key.Key_Down:
                self._move_axis(+1)
                return
        # Fall back to default list-widget behaviour for all other keys
        QtWidgets.QListWidget.keyPressEvent(self.axis_list, event)

    def _move_axis(self, direction: int):
        """Move the selected axis up (direction=-1) or down (direction=+1)."""
        items = self.axis_list.selectedItems()
        if not items:
            return
        item = items[0]
        row = self.axis_list.row(item)
        new_row = row + direction
        if new_row < 0 or new_row >= self.axis_list.count():
            return          # already at the boundary

        # Take the item out and re-insert at the new position
        taken = self.axis_list.takeItem(row)
        self.axis_list.insertItem(new_row, taken)
        self.axis_list.setCurrentItem(taken)   # keep it selected after the move

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
    
    def keyPressEvent(self, event):
        """Handle keyboard events - Ctrl+H hides panel, Ctrl+R reloads panel (hide and show)."""
        if event.key() == QtCore.Qt.Key.Key_H and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            # Hide panel when Ctrl+H is pressed
            parent_dock = self.parent()
            while parent_dock and not isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock = parent_dock.parent()
            
            if parent_dock and isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock.setVisible(False)
        elif event.key() == QtCore.Qt.Key.Key_R and event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            # Reload panel (hide and show) when Ctrl+R is pressed
            parent_dock = self.parent()
            while parent_dock and not isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock = parent_dock.parent()
            
            if parent_dock and isinstance(parent_dock, QtWidgets.QDockWidget):
                parent_dock.setVisible(False)
                parent_dock.setVisible(True)
                parent_dock.raise_()
        else:
            # Pass other key events to parent
            super().keyPressEvent(event)