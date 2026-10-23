import json
from copy import deepcopy
from pathlib import Path

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

    _run_button_style = "background-color: #4CAF50; color: white; font-weight: bold; padding: 4px;"
    _running_button_style = "background-color: #8BE6A0; color: white; font-weight: bold; padding: 4px;"

    def __init__(self, parent=None, config_path=None):
        super().__init__(parent)
        self._config_path = config_path
        # Track detectors known to the application and which are selected for
        # multi-axis scans. Default to empty lists; MainWindow will populate
        # these via set_available_detectors().
        self._available_detectors: list[str] = []
        self._selected_detectors: list[str] = []
        self._sequence_experiments: dict[str, dict | None] = {
            "pre": None,
            "post": None,
        }
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(4)  # Reduced spacing
        layout.setContentsMargins(4, 4, 4, 4)  # Reduced margins

        self.axis_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.axis_splitter.setChildrenCollapsible(False)
        layout.addWidget(self.axis_splitter)

        # Defined axes
        axis_panel = QtWidgets.QWidget()
        axis_layout = QtWidgets.QVBoxLayout(axis_panel)
        axis_layout.setSpacing(4)
        axis_layout.setContentsMargins(0, 0, 0, 0)
        axis_label = QtWidgets.QLabel("Axes:")
        axis_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        axis_layout.addWidget(axis_label)
        
        self.axis_list = QtWidgets.QListWidget()
        self.axis_list.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.InternalMove)
        self.axis_list.setDefaultDropAction(QtCore.Qt.DropAction.MoveAction)
        self.axis_list.setMinimumHeight(40)
        axis_layout.addWidget(self.axis_list)
        self.axis_splitter.addWidget(axis_panel)

        controls_panel = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setSpacing(4)
        controls_layout.setContentsMargins(0, 0, 0, 0)

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
        controls_layout.addLayout(xsel_layout)

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
        controls_layout.addLayout(btns)

        # Run buttons - compact row
        run_btns = QtWidgets.QHBoxLayout()
        run_btns.setSpacing(3)
        
        self.start_btn = QtWidgets.QPushButton("Run")
        self.start_btn.setStyleSheet(self._run_button_style)
        self.start_btn.setMaximumWidth(50)
        self._run_button_opacity = QtWidgets.QGraphicsOpacityEffect(self.start_btn)
        self._run_button_opacity.setOpacity(1.0)
        self.start_btn.setGraphicsEffect(self._run_button_opacity)
        self._run_button_animation = QtCore.QPropertyAnimation(
            self._run_button_opacity, b"opacity", self
        )
        self._run_button_animation.setDuration(1000)
        self._run_button_animation.setStartValue(1.0)
        self._run_button_animation.setKeyValueAt(0.5, 0.58)
        self._run_button_animation.setEndValue(1.0)
        self._run_button_animation.setLoopCount(-1)
        self._run_button_animation.setEasingCurve(QtCore.QEasingCurve.Type.InOutSine)
        self.stop_btn = QtWidgets.QPushButton("Stop")
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 4px;")
        self.stop_btn.setMaximumWidth(50)
        
        run_btns.addWidget(self.start_btn)
        run_btns.addWidget(self.stop_btn)
        run_btns.addStretch()
        controls_layout.addLayout(run_btns)

        sequence_form = QtWidgets.QFormLayout()
        sequence_form.setContentsMargins(0, 0, 0, 0)
        sequence_form.setSpacing(3)
        self.pre_experiment_label = QtWidgets.QLabel("Not set")
        self.pre_experiment_label.setToolTip("Optional experiment to run before the main scan")
        self.pre_experiment_btn = QtWidgets.QPushButton("Load…")
        self.pre_experiment_btn.setMaximumWidth(55)
        self.clear_pre_experiment_btn = QtWidgets.QPushButton("Clear")
        self.clear_pre_experiment_btn.setMaximumWidth(45)
        pre_row = QtWidgets.QHBoxLayout()
        pre_row.addWidget(self.pre_experiment_label, 1)
        pre_row.addWidget(self.pre_experiment_btn)
        pre_row.addWidget(self.clear_pre_experiment_btn)
        sequence_form.addRow("Pre:", pre_row)

        self.post_experiment_label = QtWidgets.QLabel("Not set")
        self.post_experiment_label.setToolTip("Optional experiment to run after the main scan")
        self.post_experiment_btn = QtWidgets.QPushButton("Load…")
        self.post_experiment_btn.setMaximumWidth(55)
        self.clear_post_experiment_btn = QtWidgets.QPushButton("Clear")
        self.clear_post_experiment_btn.setMaximumWidth(45)
        post_row = QtWidgets.QHBoxLayout()
        post_row.addWidget(self.post_experiment_label, 1)
        post_row.addWidget(self.post_experiment_btn)
        post_row.addWidget(self.clear_post_experiment_btn)
        sequence_form.addRow("Post:", post_row)
        controls_layout.addLayout(sequence_form)

        controls_layout.addStretch(1)
        self.axis_splitter.addWidget(controls_panel)
        self.axis_splitter.setStretchFactor(0, 1)
        self.axis_splitter.setStretchFactor(1, 0)
        self.axis_splitter.setSizes([120, 140])

        self.add_axis_btn.clicked.connect(self._add_axis_dialog)
        self.group_axis_btn.clicked.connect(self._group_axis)
        self.remove_axis_btn.clicked.connect(self._remove_selected)
        self.start_btn.clicked.connect(self.start_requested.emit)
        self.stop_btn.clicked.connect(self.stop_requested.emit)
        self.pre_experiment_btn.clicked.connect(
            lambda: self._choose_sequence_experiment("pre")
        )
        self.post_experiment_btn.clicked.connect(
            lambda: self._choose_sequence_experiment("post")
        )
        self.clear_pre_experiment_btn.clicked.connect(
            lambda: self.set_sequence_experiment("pre", None)
        )
        self.clear_post_experiment_btn.clicked.connect(
            lambda: self.set_sequence_experiment("post", None)
        )

        # Enable editing axis by double-clicking the list item
        self.axis_list.itemDoubleClicked.connect(lambda _: self._edit_selected())

        # Ctrl/Command+Up / Ctrl/Command+Down reorders the selected axis
        self.axis_list.keyPressEvent = self._axis_list_key_press

    def _choose_sequence_experiment(self, phase: str) -> None:
        default_dir = str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"Load {phase.title()} Experiment",
            default_dir,
            "Experiment (*.json)",
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as experiment_file:
                experiment_data = json.load(experiment_file)
            self.set_sequence_experiment(phase, experiment_data, source=path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                self,
                f"Load {phase.title()} Experiment",
                f"Could not load a valid experiment from {path}:\n{exc}",
            )

    def set_sequence_experiment(
        self,
        phase: str,
        experiment_data: dict | None,
        *,
        source: str = "",
    ) -> None:
        """Set a separately stored experiment that runs before or after the scan."""
        if phase not in self._sequence_experiments:
            raise ValueError(f"Unknown multi-axis sequence phase: {phase}")
        label = getattr(self, f"{phase}_experiment_label")
        if experiment_data is None:
            self._sequence_experiments[phase] = None
            label.setText("Not set")
            label.setToolTip("")
            return

        if not isinstance(experiment_data, dict):
            raise ValueError("Experiment JSON must contain an object.")
        multiaxis = experiment_data.get("multiaxis")
        if not isinstance(multiaxis, dict) or not isinstance(multiaxis.get("axes"), list):
            raise ValueError("Experiment must contain a multiaxis.axes list.")
        if not multiaxis["axes"]:
            raise ValueError("Experiment must define at least one multi-axis.")
        demo_settings = experiment_data.get("demo")
        if demo_settings is not None and not isinstance(demo_settings, dict):
            raise ValueError("Experiment demo settings must be an object.")

        stored_data = {
            "demo": deepcopy(demo_settings or {}),
            "multiaxis": {
                "axes": [],
                "detectors": [],
            },
            "source": source or str(experiment_data.get("source", "")),
        }
        stored_data["multiaxis"]["axes"] = [
            {
                "axis_type": item["axis_type"],
                "params": deepcopy(item["params"]),
            }
            for item in multiaxis["axes"]
            if isinstance(item, dict)
            and isinstance(item.get("axis_type"), str)
            and isinstance(item.get("params"), dict)
        ]
        detectors = multiaxis.get("detectors") or []
        if not isinstance(detectors, list) or any(
            not isinstance(detector_id, str) for detector_id in detectors
        ):
            raise ValueError("Experiment detector selection must be a list of names.")
        stored_data["multiaxis"]["detectors"] = list(detectors)
        if len(stored_data["multiaxis"]["axes"]) != len(multiaxis["axes"]):
            raise ValueError("Experiment contains an invalid multi-axis definition.")

        self._sequence_experiments[phase] = stored_data
        display_name = Path(source).name if source else f"{len(stored_data['multiaxis']['axes'])} axes"
        label.setText(display_name)
        label.setToolTip(source or display_name)

    def get_sequence_experiment(self, phase: str) -> dict | None:
        """Return the separately configured pre- or post-scan experiment."""
        if phase not in self._sequence_experiments:
            raise ValueError(f"Unknown multi-axis sequence phase: {phase}")
        experiment = self._sequence_experiments[phase]
        return deepcopy(experiment) if experiment is not None else None

    def set_scan_running(self, running: bool) -> None:
        """Pulse the green Run button while a multi-axis scan is active."""
        if running:
            self.start_btn.setStyleSheet(self._running_button_style)
            self.start_btn.setToolTip("Multi-axis scan running")
            if self._run_button_animation.state() != QtCore.QAbstractAnimation.State.Running:
                self._run_button_opacity.setOpacity(1.0)
                self._run_button_animation.start()
            return

        self._run_button_animation.stop()
        self._run_button_opacity.setOpacity(1.0)
        self.start_btn.setStyleSheet(self._run_button_style)
        self.start_btn.setToolTip("")

    def set_available_detectors(self, detectors: list[str]):
        """Receive the list of available detector IDs from the application.

        Stores the list and selects all detectors by default so multi-axis
        dialogs can list detectors individually. Emits detectors_changed.
        """
        try:
            self._available_detectors = list(detectors or [])
            # default: select all available detectors
            self._selected_detectors = list(self._available_detectors)
            self.detectors_changed.emit(list(self._selected_detectors))
        except Exception:
            self._available_detectors = []
            self._selected_detectors = []
            try:
                self.detectors_changed.emit([])
            except Exception:
                pass

    def get_selected_detectors(self) -> list[str]:
        """Return the currently selected detectors for multi-axis scans.

        Falls back to all available detectors when none explicitly selected.
        """
        try:
            if getattr(self, "_selected_detectors", None):
                return list(self._selected_detectors)
            if getattr(self, "_available_detectors", None):
                return list(self._available_detectors)
        except Exception:
            pass
        return []

    def get_available_detectors(self) -> list[str]:
        """Return the list of detector ids known to this tab."""
        try:
            return list(getattr(self, "_available_detectors", []) or [])
        except Exception:
            return []

    def set_selected_detectors(self, detector_ids: list[str]):
        """Set the currently selected detector ids and emit detectors_changed."""
        try:
            available = getattr(self, "_available_detectors", []) or []
            # keep only ids that are known
            self._selected_detectors = [d for d in (detector_ids or []) if d in available]
            if not self._selected_detectors and available:
                # if nothing selected, default to all available
                self._selected_detectors = list(available)
            self.detectors_changed.emit(list(self._selected_detectors))
        except Exception:
            try:
                self.detectors_changed.emit([])
            except Exception:
                pass

    def _emit_detectors_changed(self, *_args):
        """Internal helper to re-emit the current selected detectors."""
        try:
            self.detectors_changed.emit(list(self.get_selected_detectors()))
        except Exception:
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
            d = ExcitationAxisDialog(
                self,
                excitation_devices=self._get_excitation_devices(),
            )
        elif axis_type == "Round":
            d = RoundAxisDialog(self)
        else:
            return

        if d.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            cfg: AxisConfig = d.get_config()
            display_name = self._prompt_axis_name(cfg)
            if display_name is None:
                return
            cfg.params["display_name"] = display_name
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
            dlg = ChannelAxisDialog(self, config=cfg)
        elif cfg.axis_type == "Detector":
            dlg = DetectorAxisDialog(self, detector_name=cfg.params.get("detector"))
        elif cfg.axis_type == "Excitation":
            dlg = ExcitationAxisDialog(
                self,
                excitation_devices=self._get_excitation_devices(),
                config=cfg,
            )
        elif cfg.axis_type == "Round":
            dlg = RoundAxisDialog(self, config=cfg)
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

        # Populate detector dialog wait value when present.
        try:
            if isinstance(dlg, DetectorAxisDialog) and cfg.params:
                if "wait" in cfg.params:
                    dlg.wait_spin.setValue(cfg.params.get("wait", dlg.wait_spin.value()))
        except Exception:
            pass

        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_cfg = dlg.get_config()
            for key in (
                "display_name",
                "group_with_prev",
                "group_mode",
                "group_length",
                "collapse_one_step",
            ):
                if key in cfg.params:
                    new_cfg.params[key] = cfg.params[key]
            display_name = self._prompt_axis_name(
                new_cfg,
                default_name=cfg.params.get("display_name"),
            )
            if display_name is None:
                return
            new_cfg.params["display_name"] = display_name
            item.setText(new_cfg.label())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, new_cfg)

    def _prompt_axis_name(
        self,
        cfg: AxisConfig,
        *,
        default_name: str | None = None,
    ) -> str | None:
        """Ask for a user-facing name, choosing an unused default when adding."""
        if default_name is None:
            default_name = cfg.axis_type
            existing_names = set()
            for i in range(self.axis_list.count()):
                axis_config = self.axis_list.item(i).data(
                    QtCore.Qt.ItemDataRole.UserRole
                )
                if isinstance(axis_config, AxisConfig):
                    existing_names.add(
                        str(axis_config.params.get("display_name", "")).strip()
                    )
            candidate = default_name
            suffix = 2
            while candidate in existing_names:
                candidate = f"{default_name} {suffix}"
                suffix += 1
            default_name = candidate

        name, accepted = QtWidgets.QInputDialog.getText(
            self,
            "Name Axis",
            "Axis name:",
            text=str(default_name),
        )
        if not accepted:
            return None
        return name.strip() or str(default_name).strip()

    def _get_excitation_devices(self):
        """Build configured excitation devices for the excitation-axis editor."""
        try:
            from core.factory import build_devices, load_config

            cfg = load_config(self._config_path)
            if not cfg.get("excitation"):
                return None
            *_, excitation = build_devices(self._config_path)
            return excitation if isinstance(excitation, list) else [excitation]
        except Exception:
            return None

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
            label = ocfg.label() if isinstance(ocfg, AxisConfig) else other.text()
            other_axes.append((i, label))

        # Determine the current selection to preselect in the dialog. Collapse
        # and grouping are independent and may both be active.
        collapsed_now = bool(cfg.params.get("collapse_one_step"))
        if cfg.params.get("group_with_prev") and row > 0:
            current = ("axis", row - 1)
        else:
            current = ("none",)

        dlg = GroupAxisDialog(
            cfg.label(),
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
        """Handle Ctrl/Command+Up / Ctrl/Command+Down to reorder the selected axis."""
        mod = event.modifiers()
        key = event.key()
        reorder_modifiers = (
            QtCore.Qt.KeyboardModifier.ControlModifier
            | QtCore.Qt.KeyboardModifier.MetaModifier
        )
        if mod & reorder_modifiers:
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