"""Stage Calibration Wizard — Option B (Move & Measure).

Workflow
--------
Step 1 – Set Reference
    Click "Set Reference Point" to record the current step position of each axis.

Step 2 – Move
    Jog the stage to a new position (by any means — hardware jog, software command, etc.).
    The current position updates live every second.

Step 3 – Enter Distance
    Type the physical distance you actually moved (mm) for X and/or Y.

Step 4 – Confirm
    The dialog computes  scale = Δsteps / Δmm  for each axis, shows a preview,
    and lets you save the result to the device config JSON.

The computed scale has units [steps / mm] — the same convention used by
ScaledStageXY, which converts:  raw_steps = logical_mm * scale + offset

Auto Limit Detection
--------------------
Step A – Auto Detect Limits
    Click "Auto Detect Limits" to automatically find the stage travel limits.
    The stage will move to each direction until it cannot reach further.
    After detection, the working range is set to 95% of the total range
    (2.5% safety margin from each end).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

from PyQt6 import QtWidgets, QtCore, QtGui

logger = logging.getLogger(__name__)


class LimitReachedError(Exception):
    """Custom exception for when stage limit is reached."""
    pass


class CollapsibleGroupBox(QtWidgets.QGroupBox):
    """A QGroupBox that can be collapsed/expanded with a toggle button."""
    
    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None, 
                 initially_collapsed: bool = False):
        super().__init__(title, parent)
        self._is_collapsed = initially_collapsed
        self._content_widget: Optional[QtWidgets.QWidget] = None
        self._toggle_button: Optional[QtWidgets.QPushButton] = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the collapsible UI."""
        # Create a custom title bar with toggle button
        self.setTitle("")  # Clear default title
        self.setFlat(True)
        
        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Title bar with toggle button
        title_bar = QtWidgets.QWidget()
        title_layout = QtWidgets.QHBoxLayout(title_bar)
        title_layout.setContentsMargins(8, 4, 8, 4)
        
        # Title label
        self._title_label = QtWidgets.QLabel(self.title())
        self._title_label.setStyleSheet("font-weight: bold;")
        title_layout.addWidget(self._title_label)
        
        # Toggle button
        self._toggle_button = QtWidgets.QPushButton()
        self._toggle_button.setFixedSize(20, 20)
        self._toggle_button.setText("▼" if not self._is_collapsed else "▶")
        self._toggle_button.setStyleSheet("border: none; font-size: 12px;")
        self._toggle_button.clicked.connect(self._toggle_collapse)
        title_layout.addWidget(self._toggle_button)
        
        main_layout.addWidget(title_bar)
        
        # Content container
        self._content_widget = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(8, 4, 8, 8)
        main_layout.addWidget(self._content_widget)
        
        # Set initial state
        self._update_collapse_state()
    
    def setContentLayout(self, layout: QtWidgets.QLayout):
        """Set the content layout for the collapsible area."""
        # Clear existing layout
        while self._content_layout.count():
            item = self._content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add new layout
        self._content_layout.addLayout(layout)
    
    def _toggle_collapse(self):
        """Toggle the collapsed state."""
        self._is_collapsed = not self._is_collapsed
        self._update_collapse_state()
    
    def _update_collapse_state(self):
        """Update the UI based on collapse state."""
        if self._is_collapsed:
            self._content_widget.setVisible(False)
            self._toggle_button.setText("▶")
        else:
            self._content_widget.setVisible(True)
            self._toggle_button.setText("▼")
    
    def setTitle(self, title: str):
        """Set the title text."""
        super().setTitle(title)
        if hasattr(self, '_title_label'):
            self._title_label.setText(title)
    
    def isCollapsed(self) -> bool:
        """Return whether the group is collapsed."""
        return self._is_collapsed
    
    def setCollapsed(self, collapsed: bool):
        """Set the collapsed state."""
        if self._is_collapsed != collapsed:
            self._is_collapsed = collapsed
            self._update_collapse_state()


class StageCalibrationDialog(QtWidgets.QDialog):
    """Two-page wizard for Move & Measure stage calibration."""

    # Emitted when new scale values have been saved to config.
    calibration_saved = QtCore.pyqtSignal(float, float)  # x_scale, y_scale

    def __init__(
        self,
        stage,                         # live stage object (get_position() required)
        config_path: str | Path,
        parent: Optional[QtWidgets.QWidget] = None,
    ):
        super().__init__(parent)
        self._stage = stage
        self._config_path = Path(config_path)

        self._ref_x: float | None = None   # step position at reference point
        self._ref_y: float | None = None

        # Auto-detection state
        self._auto_detecting = False
        self._auto_detect_cancelled = False
        self._auto_detect_progress = 0.0

        self.setWindowTitle("Stage Calibration — Move & Measure")
        self.setMinimumWidth(480)
        self._build_ui()

        # Poll live position every 500 ms
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._refresh_position)
        self._timer.start()

        self._refresh_position()
        self._populate_current_scaling()   # show existing scale values on open
        self._populate_current_range()     # show existing travel limits on open
        self._update_range_unit_label()    # show which unit the range uses

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)

        # ── Current calibration status banner ─────────────────────────
        xs, xo, ys, yo = self._load_existing_scales_full()
        is_calibrated = (xs != 1.0 or xo != 0.0 or ys != 1.0 or yo != 0.0)
        if is_calibrated:
            status_text = (
                f"<b>Current unit: mm</b> — calibration is active<br>"
                f"<small>X: {xs:.6g} steps/mm, offset {xo:.6g} &nbsp;|&nbsp; "
                f"Y: {ys:.6g} steps/mm, offset {yo:.6g}<br>"
                f"Axis editor shows values in <b>mm</b>. "
                f"After saving here the new scale takes effect on the next run.</small>"
            )
            status_color = "#e6f4ea"   # green
        else:
            status_text = (
                "<b>Current unit: steps</b> — no calibration configured yet<br>"
                "<small>Axis editor shows values in raw motor <b>steps</b>.<br>"
                "Complete this wizard and save to switch axis editor to <b>mm</b>.</small>"
            )
            status_color = "#fff3cd"   # amber
        status_banner = QtWidgets.QLabel(status_text)
        status_banner.setWordWrap(True)
        status_banner.setStyleSheet(
            f"background:{status_color}; border:1px solid #ccc; "
            f"border-radius:4px; padding:6px;"
        )
        root.addWidget(status_banner)
        root.addSpacing(6)

        # ── Instructions (collapsible) ──────────────────────────────────────
        info_grp = CollapsibleGroupBox("Instructions", initially_collapsed=True)
        info_lay = QtWidgets.QVBoxLayout()
        info_grp.setContentLayout(info_lay)
        
        info = QtWidgets.QLabel(
            "<b>How it works:</b><br>"
            "1. Click <i>Set Reference Point</i> to record the current stage position.<br>"
            "2. Move the stage to a new position (use the stage jog controls or hardware).<br>"
            "3. Enter the <b>physical distance</b> you moved (in mm) for each axis.<br>"
            "4. Click <i>Calculate &amp; Save</i> — the scaling factor will be computed and "
            "written to the device config."
        )
        info.setWordWrap(True)
        info_lay.addWidget(info)
        root.addWidget(info_grp)
        root.addSpacing(8)

        # ── Live position display ──────────────────────────────────────
        # Position is always in raw steps (that is what get_position() returns
        # before ScaledStageXY is active, and what we need to compute Δsteps).
        pos_grp = CollapsibleGroupBox("Current Stage Position  [raw steps — as read from hardware]", initially_collapsed=False)
        pos_lay = QtWidgets.QFormLayout()
        pos_grp.setContentLayout(pos_lay)

        self._cur_x_label = QtWidgets.QLabel("—")
        self._cur_y_label = QtWidgets.QLabel("—")
        pos_lay.addRow("X:", self._cur_x_label)
        pos_lay.addRow("Y:", self._cur_y_label)
        root.addWidget(pos_grp)

        # ── Step 1 — Set reference ─────────────────────────────────────
        ref_grp = CollapsibleGroupBox("Step 1 — Set Reference Point  [records current step count]", initially_collapsed=False)
        ref_lay = QtWidgets.QVBoxLayout()
        ref_grp.setContentLayout(ref_lay)

        self._set_ref_btn = QtWidgets.QPushButton("Set Reference Point")
        self._set_ref_btn.setToolTip("Snapshot the current position as the reference (Δ = 0).")
        self._set_ref_btn.clicked.connect(self._on_set_reference)
        ref_lay.addWidget(self._set_ref_btn)

        ref_status_lay = QtWidgets.QFormLayout()
        self._ref_x_label = QtWidgets.QLabel("<i>not set</i>")
        self._ref_y_label = QtWidgets.QLabel("<i>not set</i>")
        ref_status_lay.addRow("Reference X [steps]:", self._ref_x_label)
        ref_status_lay.addRow("Reference Y [steps]:", self._ref_y_label)
        ref_lay.addLayout(ref_status_lay)
        root.addWidget(ref_grp)

        # ── Step 2/3 — Enter physical distance ────────────────────────
        meas_grp = CollapsibleGroupBox(
            "Step 2 — Enter Physical Distance Moved  [measured externally, in mm]", 
            initially_collapsed=False
        )
        meas_lay = QtWidgets.QFormLayout()
        meas_grp.setContentLayout(meas_lay)

        self._dist_x_spin = QtWidgets.QDoubleSpinBox()
        self._dist_x_spin.setRange(0.0, 1e6)
        self._dist_x_spin.setDecimals(4)
        self._dist_x_spin.setSuffix(" mm")
        self._dist_x_spin.setSpecialValueText("(axis not moved)")
        self._dist_x_spin.setValue(0.0)

        self._dist_y_spin = QtWidgets.QDoubleSpinBox()
        self._dist_y_spin.setRange(0.0, 1e6)
        self._dist_y_spin.setDecimals(4)
        self._dist_y_spin.setSuffix(" mm")
        self._dist_y_spin.setSpecialValueText("(axis not moved)")
        self._dist_y_spin.setValue(0.0)

        meas_lay.addRow("Distance X:", self._dist_x_spin)
        meas_lay.addRow("Distance Y:", self._dist_y_spin)

        note = QtWidgets.QLabel(
            "<small>Enter 0 for an axis you did not move — "
            "its current calibration will be kept.</small>"
        )
        note.setWordWrap(True)
        meas_lay.addRow("", note)
        root.addWidget(meas_grp)

        # ── Result preview ────────────────────────────────────────────
        result_grp = CollapsibleGroupBox("Scaling (current → new)", initially_collapsed=True)
        result_lay = QtWidgets.QFormLayout()
        result_grp.setContentLayout(result_lay)

        self._current_x_label = QtWidgets.QLabel("—")
        self._current_y_label = QtWidgets.QLabel("—")
        result_lay.addRow("Current X scale [steps/mm]:", self._current_x_label)
        result_lay.addRow("Current Y scale [steps/mm]:", self._current_y_label)

        result_lay.addRow(QtWidgets.QFrame())   # thin separator line

        self._result_x_label = QtWidgets.QLabel("—")
        self._result_y_label = QtWidgets.QLabel("—")
        result_lay.addRow("New X scale [steps/mm]:", self._result_x_label)
        result_lay.addRow("New Y scale [steps/mm]:", self._result_y_label)
        root.addWidget(result_grp)

        # ── Travel range (soft limits) ────────────────────────────────
        self._range_grp = CollapsibleGroupBox("Travel Range (Soft Limits)", initially_collapsed=True)
        range_vlay = QtWidgets.QVBoxLayout()
        self._range_grp.setContentLayout(range_vlay)

        self._range_enable_chk = QtWidgets.QCheckBox("Enable soft travel limits")
        self._range_enable_chk.setToolTip(
            "When enabled, any stage move that would exceed the configured\n"
            "limits is aborted with a ValueError (scan / jog / pre-pos all affected)."
        )
        self._range_enable_chk.toggled.connect(self._on_range_enable_toggled)
        range_vlay.addWidget(self._range_enable_chk)

        unit_note = QtWidgets.QLabel()
        unit_note.setWordWrap(True)
        self._range_unit_label = unit_note
        range_vlay.addWidget(unit_note)

        range_grid = QtWidgets.QGridLayout()
        # --- helper to build a row with spin + capture button ---
        def _make_row(label_text: str) -> tuple[QtWidgets.QDoubleSpinBox, QtWidgets.QPushButton]:
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(6)
            spin.setButtonSymbols(QtWidgets.QAbstractSpinBox.ButtonSymbols.NoButtons)
            btn = QtWidgets.QPushButton("Use Current")
            btn.setToolTip("Capture the current stage position into this field.")
            return spin, btn

        # Row 0: X
        range_grid.addWidget(QtWidgets.QLabel("<b>X</b>"), 0, 0)
        self._x_min_spin, self._x_min_btn = _make_row("X Min")
        self._x_max_spin, self._x_max_btn = _make_row("X Max")
        range_grid.addWidget(QtWidgets.QLabel("Min"), 0, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        range_grid.addWidget(self._x_min_spin, 0, 2)
        range_grid.addWidget(self._x_min_btn, 0, 3)
        range_grid.addItem(QtWidgets.QSpacerItem(12, 0), 0, 4)
        range_grid.addWidget(QtWidgets.QLabel("Max"), 0, 5, QtCore.Qt.AlignmentFlag.AlignRight)
        range_grid.addWidget(self._x_max_spin, 0, 6)
        range_grid.addWidget(self._x_max_btn, 0, 7)

        # Row 1: Y
        range_grid.addWidget(QtWidgets.QLabel("<b>Y</b>"), 1, 0)
        self._y_min_spin, self._y_min_btn = _make_row("Y Min")
        self._y_max_spin, self._y_max_btn = _make_row("Y Max")
        range_grid.addWidget(QtWidgets.QLabel("Min"), 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)
        range_grid.addWidget(self._y_min_spin, 1, 2)
        range_grid.addWidget(self._y_min_btn, 1, 3)
        range_grid.addItem(QtWidgets.QSpacerItem(12, 0), 1, 4)
        range_grid.addWidget(QtWidgets.QLabel("Max"), 1, 5, QtCore.Qt.AlignmentFlag.AlignRight)
        range_grid.addWidget(self._y_max_spin, 1, 6)
        range_grid.addWidget(self._y_max_btn, 1, 7)
        range_grid.setColumnStretch(2, 1)
        range_grid.setColumnStretch(6, 1)

        range_vlay.addLayout(range_grid)

        hint = QtWidgets.QLabel(
            "<small><b>How to set limits:</b> physically jog the stage to each "
            "extreme (e.g. lower-left corner → press both <i>Use Current</i> "
            "buttons on the Min column), then move to the opposite corner and "
            "capture the Max values. The new limits take effect on the next run.</small>"
        )
        hint.setWordWrap(True)
        range_vlay.addWidget(hint)

        # ── Auto-detect section ─────────────────────────────────────────
        self._auto_detect_grp = CollapsibleGroupBox("Auto-Detect Travel Limits", initially_collapsed=True)
        auto_lay = QtWidgets.QVBoxLayout()
        self._auto_detect_grp.setContentLayout(auto_lay)

        auto_info = QtWidgets.QLabel(
            "<small>Automatically detect stage limits by moving to each direction "
            "until unable to reach further. After detection, working range is set "
            "to 95% of total range (2.5% safety margin from each end).</small>"
        )
        auto_info.setWordWrap(True)
        auto_lay.addWidget(auto_info)

        # Progress bar
        self._auto_detect_progress_bar = QtWidgets.QProgressBar()
        self._auto_detect_progress_bar.setVisible(False)
        self._auto_detect_progress_bar.setRange(0, 100)
        auto_lay.addWidget(self._auto_detect_progress_bar)

        # Status label
        self._auto_detect_status_label = QtWidgets.QLabel("Ready")
        self._auto_detect_status_label.setVisible(False)
        auto_lay.addWidget(self._auto_detect_status_label)

        # Buttons
        auto_btn_lay = QtWidgets.QHBoxLayout()
        self._auto_detect_btn = QtWidgets.QPushButton("Auto Detect Limits")
        self._auto_detect_btn.clicked.connect(self._on_auto_detect_limits)
        auto_btn_lay.addWidget(self._auto_detect_btn)

        self._auto_detect_cancel_btn = QtWidgets.QPushButton("Cancel")
        self._auto_detect_cancel_btn.setVisible(False)
        self._auto_detect_cancel_btn.clicked.connect(self._on_cancel_auto_detect)
        auto_btn_lay.addWidget(self._auto_detect_cancel_btn)

        auto_lay.addLayout(auto_btn_lay)
        range_vlay.addWidget(self._auto_detect_grp)

        self._x_min_btn.clicked.connect(lambda: self._capture_current(self._x_min_spin, "x"))
        self._x_max_btn.clicked.connect(lambda: self._capture_current(self._x_max_spin, "x"))
        self._y_min_btn.clicked.connect(lambda: self._capture_current(self._y_min_spin, "y"))
        self._y_max_btn.clicked.connect(lambda: self._capture_current(self._y_max_spin, "y"))

        root.addWidget(self._range_grp)

        # ── Buttons ───────────────────────────────────────────────────
        btn_lay = QtWidgets.QHBoxLayout()

        self._calc_btn = QtWidgets.QPushButton("Calculate && Preview")
        self._calc_btn.setEnabled(False)
        self._calc_btn.clicked.connect(self._on_calculate)
        btn_lay.addWidget(self._calc_btn)

        self._save_btn = QtWidgets.QPushButton("Save to Config")
        self._save_btn.setEnabled(False)
        self._save_btn.setToolTip("Write the computed scaling factors to the device config JSON.")
        self._save_btn.clicked.connect(self._on_save)
        btn_lay.addWidget(self._save_btn)

        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_lay.addWidget(close_btn)

        root.addLayout(btn_lay)

        # Cached computed scales (set after Calculate)
        self._computed_x: float | None = None
        self._computed_y: float | None = None

    # ------------------------------------------------------------------ #
    # Slots
    # ------------------------------------------------------------------ #

    def _refresh_position(self):
        """Read current stage position and update labels."""
        try:
            x, y = self._stage.get_position()
            self._cur_x_label.setText(f"{x:.2f}")
            self._cur_y_label.setText(f"{y:.2f}")
        except Exception:
            self._cur_x_label.setText("error")
            self._cur_y_label.setText("error")

    def _on_set_reference(self):
        try:
            x, y = self._stage.get_position()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Could not read stage position:\n{exc}"
            )
            return
        self._ref_x = float(x)
        self._ref_y = float(y)
        self._ref_x_label.setText(f"{self._ref_x:.2f} steps")
        self._ref_y_label.setText(f"{self._ref_y:.2f} steps")
        self._calc_btn.setEnabled(True)
        self._save_btn.setEnabled(False)
        self._result_x_label.setText("—")
        self._result_y_label.setText("—")
        self._computed_x = None
        self._computed_y = None
        logger.info("Calibration reference set: x=%s y=%s", self._ref_x, self._ref_y)

    def _on_calculate(self):
        if self._ref_x is None or self._ref_y is None:
            QtWidgets.QMessageBox.warning(self, "No Reference", "Please set a reference point first.")
            return

        try:
            cur_x, cur_y = self._stage.get_position()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not read position:\n{exc}")
            return

        delta_steps_x = float(cur_x) - self._ref_x
        delta_steps_y = float(cur_y) - self._ref_y
        dist_mm_x = self._dist_x_spin.value()
        dist_mm_y = self._dist_y_spin.value()

        # Load existing scales from config as fallback
        existing_x, existing_y = self._load_existing_scales()

        if dist_mm_x > 0:
            if abs(delta_steps_x) < 0.5:
                QtWidgets.QMessageBox.warning(
                    self, "No Movement Detected",
                    "X axis: stage has not moved from the reference point.\n"
                    "Move the stage before calculating."
                )
                return
            self._computed_x = abs(delta_steps_x) / dist_mm_x
        else:
            self._computed_x = existing_x   # keep current

        if dist_mm_y > 0:
            if abs(delta_steps_y) < 0.5:
                QtWidgets.QMessageBox.warning(
                    self, "No Movement Detected",
                    "Y axis: stage has not moved from the reference point.\n"
                    "Move the stage before calculating."
                )
                return
            self._computed_y = abs(delta_steps_y) / dist_mm_y
        else:
            self._computed_y = existing_y   # keep current

        kept_x = dist_mm_x == 0
        kept_y = dist_mm_y == 0

        self._result_x_label.setText(
            f"<b>{self._computed_x:.4f}</b>" +
            (" <small>(unchanged)</small>" if kept_x else
             f" <small>(Δ {delta_steps_x:+.1f} steps / {dist_mm_x} mm)</small>")
        )
        self._result_y_label.setText(
            f"<b>{self._computed_y:.4f}</b>" +
            (" <small>(unchanged)</small>" if kept_y else
             f" <small>(Δ {delta_steps_y:+.1f} steps / {dist_mm_y} mm)</small>")
        )
        self._save_btn.setEnabled(True)
        logger.info(
            "Calibration calculated: x_scale=%s y_scale=%s (delta_steps x=%s y=%s, dist_mm x=%s y=%s)",
            self._computed_x, self._computed_y,
            delta_steps_x, delta_steps_y,
            dist_mm_x, dist_mm_y,
        )

    def _on_save(self):
        """Handle save button click with confirmation dialog."""
        if self._computed_x is None or self._computed_y is None:
            return
        
        # Show save options dialog
        save_option = self._show_save_options_dialog()
        if save_option is None:
            return  # User cancelled
        
        if save_option == "overwrite":
            # Overwrite current config with secondary confirmation
            if not self._confirm_overwrite():
                return
            target_path = self._config_path
        else:
            # Save as new file with name validation
            target_path = self._get_new_config_path()
            if target_path is None:
                return  # User cancelled or invalid filename
        
        # Perform the actual save
        self._perform_save(target_path, save_option)

    def _show_save_options_dialog(self) -> Optional[str]:
        """Show modal dialog with save options (overwrite vs new file).
        
        Returns:
            "overwrite" if user chose to overwrite current config
            "new" if user chose to create new config file
            None if user cancelled
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save Calibration Data")
        dialog.setMinimumWidth(500)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Risk warning banner
        warning_label = QtWidgets.QLabel(
            "<span style='color:#c5221f; font-weight:bold;'>⚠️ WARNING: "
            "Overwriting the current config file will replace all existing "
            "calibration data and cannot be undone!</span>"
        )
        warning_label.setWordWrap(True)
        warning_label.setStyleSheet(
            "background:#fce8e6; border:1px solid #c5221f; "
            "border-radius:4px; padding:8px;"
        )
        layout.addWidget(warning_label)
        layout.addSpacing(12)
        
        # Instructions
        info_label = QtWidgets.QLabel(
            "<b>Choose how to save your calibration data:</b>"
        )
        layout.addWidget(info_label)
        layout.addSpacing(8)
        
        # Radio button group for mutually exclusive options
        button_group = QtWidgets.QButtonGroup(dialog)
        
        # Option 1: Overwrite current config
        overwrite_radio = QtWidgets.QRadioButton(
            f"Overwrite current config file:\n{self._config_path}"
        )
        overwrite_radio.setStyleSheet("font-weight:bold;")
        button_group.addButton(overwrite_radio, 1)
        layout.addWidget(overwrite_radio)
        layout.addSpacing(4)
        
        overwrite_detail = QtWidgets.QLabel(
            "<small>⚠️ This will replace the existing configuration file. "
            "Original data will be lost and cannot be recovered.</small>"
        )
        overwrite_detail.setWordWrap(True)
        overwrite_detail.setStyleSheet("color:#c5221f; padding-left:20px;")
        layout.addWidget(overwrite_detail)
        layout.addSpacing(12)
        
        # Option 2: Save as new file
        new_file_radio = QtWidgets.QRadioButton(
            "Save as a new configuration file"
        )
        new_file_radio.setStyleSheet("font-weight:bold;")
        button_group.addButton(new_file_radio, 2)
        layout.addWidget(new_file_radio)
        layout.addSpacing(4)
        
        new_file_detail = QtWidgets.QLabel(
            "<small>ℹ️ This will create a new config file. "
            "You can specify a custom filename for the new configuration.</small>"
        )
        new_file_detail.setWordWrap(True)
        new_file_detail.setStyleSheet("color:#1a73e8; padding-left:20px;")
        layout.addWidget(new_file_detail)
        layout.addSpacing(16)
        
        # Button box
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Disable OK button until selection is made
        ok_button = button_box.button(QtWidgets.QDialogButtonBox.StandardButton.Ok)
        ok_button.setEnabled(False)
        
        def on_selection_changed():
            ok_button.setEnabled(button_group.checkedButton() is not None)
        
        button_group.buttonClicked.connect(on_selection_changed)
        
        layout.addWidget(button_box)
        
        result = dialog.exec()
        
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            if button_group.checkedButton() == overwrite_radio:
                return "overwrite"
            else:
                return "new"
        else:
            return None

    def _confirm_overwrite(self) -> bool:
        """Show secondary confirmation dialog for overwrite operation.
        
        Returns:
            True if user confirms overwrite, False otherwise
        """
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Overwrite",
            f"<b>Are you sure you want to overwrite the current config file?</b><br><br>"
            f"File: {self._config_path}<br><br>"
            "<span style='color:#c5221f; font-weight:bold;'>"
            "This action cannot be undone. All existing calibration data "
            "in this file will be permanently replaced.</span>",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        return reply == QtWidgets.QMessageBox.StandardButton.Yes

    def _get_new_config_path(self) -> Optional[Path]:
        """Prompt user for new config filename and validate it.
        
        Returns:
            Path to new config file if valid, None if cancelled or invalid
        """
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Save as New Configuration")
        dialog.setMinimumWidth(400)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        # Filename input
        layout.addWidget(QtWidgets.QLabel("<b>Enter new configuration filename:</b>"))
        layout.addSpacing(8)
        
        filename_input = QtWidgets.QLineEdit()
        filename_input.setPlaceholderText("e.g., calibration_config_v2.json")
        # Suggest a default name based on timestamp
        default_name = f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filename_input.setText(default_name)
        filename_input.selectAll()
        layout.addWidget(filename_input)
        layout.addSpacing(8)
        
        # Validation message label
        validation_label = QtWidgets.QLabel()
        validation_label.setWordWrap(True)
        validation_label.setStyleSheet("color:#c5221f;")
        layout.addWidget(validation_label)
        layout.addSpacing(12)
        
        # Button box
        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok |
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        # Validation function
        def validate_filename():
            filename = filename_input.text().strip()
            if not filename:
                validation_label.setText("❌ Filename cannot be empty")
                return False
            
            # Check file extension
            if not filename.endswith('.json'):
                validation_label.setText("❌ Filename must end with .json")
                return False
            
            # Check for invalid characters
            invalid_chars = '<>:"/\\|?*'
            if any(char in filename for char in invalid_chars):
                validation_label.setText(f"❌ Filename contains invalid characters: {invalid_chars}")
                return False
            
            # Check if file already exists
            new_path = self._config_path.parent / filename
            if new_path.exists():
                validation_label.setText(f"❌ File already exists: {filename}")
                return False
            
            validation_label.setText("✓ Filename is valid")
            validation_label.setStyleSheet("color:#1a7f37;")
            return True
        
        filename_input.textChanged.connect(validate_filename)
        
        # Initial validation
        validate_filename()
        
        result = dialog.exec()
        
        if result == QtWidgets.QDialog.DialogCode.Accepted:
            filename = filename_input.text().strip()
            if validate_filename():
                return self._config_path.parent / filename
        return None

    def _perform_save(self, target_path: Path, save_option: str):
        """Perform the actual save operation with logging.
        
        Args:
            target_path: Path where config will be saved
            save_option: "overwrite" or "new"
        """
        start_time = datetime.now()
        
        try:
            cfg = self._read_config()
            stage_cfg = cfg.setdefault("stage", {})
            scaling = stage_cfg.setdefault("scaling", {})
            scaling["x_scale"] = round(self._computed_x, 6)
            scaling["y_scale"] = round(self._computed_y, 6)
            # Preserve existing offsets — calibration only touches scale
            scaling.setdefault("x_offset", 0.0)
            scaling.setdefault("y_offset", 0.0)

            # --- persist travel range from UI ---
            rc = stage_cfg.setdefault("range", {})
            if self._range_enable_chk.isChecked():
                rc["x_min"] = self._nullable_float(self._x_min_spin.value())
                rc["x_max"] = self._nullable_float(self._x_max_spin.value())
                rc["y_min"] = self._nullable_float(self._y_min_spin.value())
                rc["y_max"] = self._nullable_float(self._y_max_spin.value())
            else:
                rc["x_min"] = None
                rc["x_max"] = None
                rc["y_min"] = None
                rc["y_max"] = None

            # Write to target path
            with open(target_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
                
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save Error", f"Could not write config:\n{exc}"
            )
            logger.error(
                "Calibration save failed: option=%s target=%s error=%s",
                save_option, target_path, exc
            )
            return

        # Log successful save operation
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(
            "Calibration saved successfully: "
            "save_option=%s target_path=%s x_scale=%s y_scale=%s "
            "range=%s start_time=%s end_time=%s duration_seconds=%s",
            save_option, target_path,
            self._computed_x, self._computed_y,
            self._range_summary_text(),
            start_time.isoformat(), end_time.isoformat(), duration
        )

        # Emit signal and update UI
        self.calibration_saved.emit(self._computed_x, self._computed_y)
        
        # If we saved to a different file, update our config path reference
        if target_path != self._config_path:
            self._config_path = target_path
        
        self._populate_current_scaling()   # refresh "current" row to show new value
        self._populate_current_range()
        self._update_range_unit_label()
        
        # Show success message
        save_type_text = "Overwritten existing config" if save_option == "overwrite" else "Created new config file"
        QtWidgets.QMessageBox.information(
            self,
            "Saved",
            f"{save_type_text}:\n{target_path}\n\n"
            f"X scale: {self._computed_x:.4f} steps/mm\n"
            f"Y scale: {self._computed_y:.4f} steps/mm\n"
            f"Limits: {self._range_summary_text()}\n\n"
            f"Operation time: {duration:.3f}s\n"
            "The new scaling and limits take effect on the next run "
            "(i.e. when devices are rebuilt)."
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _populate_current_scaling(self) -> None:
        """Fill the 'Current scale' labels from the config file."""
        xs, xo, ys, yo = self._load_existing_scales_full()
        is_calibrated = (xs != 1.0 or xo != 0.0 or ys != 1.0 or yo != 0.0)
        if is_calibrated:
            self._current_x_label.setText(
                f"<b>{xs:.6g}</b> steps/mm"
                + (f"  (offset {xo:.6g})" if xo != 0.0 else "")
            )
            self._current_y_label.setText(
                f"<b>{ys:.6g}</b> steps/mm"
                + (f"  (offset {yo:.6g})" if yo != 0.0 else "")
            )
        else:
            self._current_x_label.setText("<i>1.0 (no calibration — unit is steps)</i>")
            self._current_y_label.setText("<i>1.0 (no calibration — unit is steps)</i>")

    def _load_existing_scales(self) -> tuple[float, float]:
        try:
            cfg = self._read_config()
            sc = cfg.get("stage", {}).get("scaling", {})
            return float(sc.get("x_scale", 1.0)), float(sc.get("y_scale", 1.0))
        except Exception:
            return 1.0, 1.0

    def _load_existing_scales_full(self) -> tuple[float, float, float, float]:
        """Return (x_scale, x_offset, y_scale, y_offset)."""
        try:
            cfg = self._read_config()
            sc = cfg.get("stage", {}).get("scaling", {})
            return (
                float(sc.get("x_scale", 1.0)),
                float(sc.get("x_offset", 0.0)),
                float(sc.get("y_scale", 1.0)),
                float(sc.get("y_offset", 0.0)),
            )
        except Exception:
            return 1.0, 0.0, 1.0, 0.0

    def _read_config(self) -> dict:
        with open(self._config_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_config(self, cfg: dict) -> None:
        with open(self._config_path, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

    # ------------------------------------------------------------------ #
    # Range (travel limits) UI + helpers
    # ------------------------------------------------------------------ #

    def _on_range_enable_toggled(self, checked: bool) -> None:
        """Enable/disable all range input widgets based on the master checkbox."""
        for w in (
            self._x_min_spin, self._x_min_btn,
            self._x_max_spin, self._x_max_btn,
            self._y_min_spin, self._y_min_btn,
            self._y_max_spin, self._y_max_btn,
        ):
            w.setEnabled(checked)

    def _capture_current(self, spin: QtWidgets.QDoubleSpinBox, axis: str) -> None:
        """Capture the current position of ``axis`` into ``spin``."""
        try:
            x, y = self._stage.get_position()
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Could not read stage position:\n{exc}"
            )
            return
        val = float(x if axis.lower() == "x" else y)
        spin.setValue(val)
        logger.info("Range capture: %s = %s -> %s", axis.upper(), val, spin.objectName() or spin)

    @staticmethod
    def _nullable_float(v: float) -> float | None:
        """Return None for sentinel values, else the float itself.

        The input widgets only accept values in [-1e12, 1e12]; anything beyond
        that magnitude (e.g. user cleared / didn't set a bound) is not
        meaningful. We also treat the sentinel minimum / maximum of the
        spinbox as "user didn't set this limit".
        """
        if v is None:
            return None
        f = float(v)
        MAG = 1e12
        if f <= -MAG + 1 or f >= MAG - 1:
            return None
        return f

    def _load_existing_range(self) -> dict:
        """Return the existing range dict from config with values as float|None."""
        empty = {"x_min": None, "x_max": None, "y_min": None, "y_max": None}
        try:
            cfg = self._read_config()
            rc = cfg.get("stage", {}).get("range")
        except Exception:
            rc = None
        if not isinstance(rc, dict):
            return empty
        def _f(v):
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None
        return {
            "x_min": _f(rc.get("x_min")),
            "x_max": _f(rc.get("x_max")),
            "y_min": _f(rc.get("y_min")),
            "y_max": _f(rc.get("y_max")),
        }

    def _populate_current_range(self) -> None:
        """Fill the travel-range UI from the values on disk."""
        r = self._load_existing_range()
        any_set = any(v is not None for v in r.values())
        self._range_enable_chk.blockSignals(True)
        try:
            self._range_enable_chk.setChecked(any_set)
        finally:
            self._range_enable_chk.blockSignals(False)
        self._on_range_enable_toggled(any_set)

        def _apply(spin, v):
            if v is None:
                # Put a sentinel that users will visually notice is "unset".
                spin.setValue(0.0)
            else:
                spin.setValue(float(v))

        _apply(self._x_min_spin, r["x_min"])
        _apply(self._x_max_spin, r["x_max"])
        _apply(self._y_min_spin, r["y_min"])
        _apply(self._y_max_spin, r["y_max"])

    def _update_range_unit_label(self) -> None:
        """Show the user which unit the range fields are expressed in."""
        xs, xo, ys, yo = self._load_existing_scales_full()
        is_calibrated = (xs != 1.0 or xo != 0.0 or ys != 1.0 or yo != 0.0)
        if is_calibrated:
            text = (
                "<span style=\"color:#1a7f37\"><b>Unit: mm</b></span><br>"
                "<small>Range fields are interpreted in <b>millimetres</b> "
                "(same as the axis editor after calibration).</small>"
            )
            suffix = " mm"
        else:
            text = (
                "<span style=\"color:#9a6700\"><b>Unit: steps</b></span><br>"
                "<small>Stage is not calibrated yet — range fields are "
                "interpreted as raw motor <b>steps</b>.</small>"
            )
            suffix = " steps"
        self._range_unit_label.setText(text)
        for sp in (self._x_min_spin, self._x_max_spin, self._y_min_spin, self._y_max_spin):
            sp.setSuffix(suffix)

    def _range_summary_text(self) -> str:
        """Human readable one-line summary of the current range settings."""
        if not self._range_enable_chk.isChecked():
            return "disabled (unlimited)"
        def _fmt(spin, side):
            v = self._nullable_float(spin.value())
            if v is None:
                return f"{side}=unset"
            return f"{side}={v:.6g}"
        xs, xo, ys, yo = self._load_existing_scales_full()
        is_calibrated = (xs != 1.0 or xo != 0.0 or ys != 1.0 or yo != 0.0)
        unit = "mm" if is_calibrated else "steps"
        return (
            f"X [{_fmt(self._x_min_spin, 'min')}, {_fmt(self._x_max_spin, 'max')}] ; "
            f"Y [{_fmt(self._y_min_spin, 'min')}, {_fmt(self._y_max_spin, 'max')}] ({unit})"
        )

    def closeEvent(self, event):
        self._timer.stop()
        # Cancel auto-detection if running
        if self._auto_detecting:
            self._auto_detect_cancelled = True
        super().closeEvent(event)

    def reject(self):
        self._timer.stop()
        # Cancel auto-detection if running
        if self._auto_detecting:
            self._auto_detect_cancelled = True
        super().reject()

    # ------------------------------------------------------------------ #
    # Auto-detect travel limits
    # ------------------------------------------------------------------ #

    def _on_auto_detect_limits(self):
        """Start the automatic limit detection process."""
        if self._auto_detecting:
            return

        # Confirm before starting
        reply = QtWidgets.QMessageBox.question(
            self,
            "Start Auto-Detection",
            "This will move the stage to find its travel limits.\n\n"
            "Make sure the stage has enough space to move freely.\n"
            "The process may take several minutes.\n\n"
            "Do you want to continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        # Start auto-detection in a separate thread
        self._auto_detecting = True
        self._auto_detect_cancelled = False
        self._auto_detect_progress = 0.0

        # Update UI
        self._auto_detect_btn.setEnabled(False)
        self._auto_detect_cancel_btn.setVisible(True)
        self._auto_detect_progress_bar.setVisible(True)
        self._auto_detect_status_label.setVisible(True)
        self._auto_detect_status_label.setText("Starting detection...")

        # Run detection in background thread
        self._auto_detect_thread = QtCore.QThread()
        self._auto_detect_worker = AutoDetectWorker(self._stage)
        self._auto_detect_worker.moveToThread(self._auto_detect_thread)
        self._auto_detect_worker.started.connect(self._on_auto_detect_started)
        self._auto_detect_worker.progress.connect(self._on_auto_detect_progress)
        self._auto_detect_worker.finished.connect(self._on_auto_detect_finished)
        self._auto_detect_worker.error.connect(self._on_auto_detect_error)
        self._auto_detect_thread.started.connect(self._auto_detect_worker.run)
        self._auto_detect_thread.start()

    def _on_cancel_auto_detect(self):
        """Cancel the ongoing auto-detection."""
        if self._auto_detecting:
            self._auto_detect_cancelled = True
            self._auto_detect_status_label.setText("Cancelling...")

    def _on_auto_detect_started(self):
        """Called when auto-detection starts."""
        logger.info("Auto-detection started")
        self._auto_detect_status_label.setText("Detecting limits...")

    def _on_auto_detect_progress(self, progress: float, status: str):
        """Update progress during auto-detection."""
        self._auto_detect_progress = progress
        self._auto_detect_progress_bar.setValue(int(progress))
        self._auto_detect_status_label.setText(status)

    def _on_auto_detect_finished(self, limits: dict):
        """Called when auto-detection completes successfully."""
        self._auto_detecting = False
        self._auto_detect_thread.quit()
        self._auto_detect_thread.wait()

        # Restore UI
        self._auto_detect_btn.setEnabled(True)
        self._auto_detect_cancel_btn.setVisible(False)
        self._auto_detect_progress_bar.setVisible(False)
        self._auto_detect_status_label.setVisible(False)

        # Apply 95% working range
        safe_limits = self._calculate_95_percent_range(limits)

        # Update UI with detected limits
        self._apply_detected_limits(safe_limits)

        logger.info("Auto-detection completed: %s", safe_limits)
        QtWidgets.QMessageBox.information(
            self,
            "Auto-Detection Complete",
            f"Travel limits detected successfully!\n\n"
            f"Working range set to 95% of total range (2.5% safety margin).\n\n"
            f"X: [{safe_limits['x_min']:.2f}, {safe_limits['x_max']:.2f}]\n"
            f"Y: [{safe_limits['y_min']:.2f}, {safe_limits['y_max']:.2f}]"
        )

    def _on_auto_detect_error(self, error_msg: str):
        """Called when auto-detection fails."""
        self._auto_detecting = False
        self._auto_detect_thread.quit()
        self._auto_detect_thread.wait()

        # Restore UI
        self._auto_detect_btn.setEnabled(True)
        self._auto_detect_cancel_btn.setVisible(False)
        self._auto_detect_progress_bar.setVisible(False)
        self._auto_detect_status_label.setVisible(False)

        logger.error("Auto-detection failed: %s", error_msg)
        QtWidgets.QMessageBox.critical(
            self,
            "Auto-Detection Failed",
            f"Auto-detection failed:\n{error_msg}"
        )

    def _calculate_95_percent_range(self, limits: dict) -> dict:
        """Calculate 95% working range from detected limits.

        Args:
            limits: Dict with 'x_min', 'x_max', 'y_min', 'y_max'

        Returns:
            Dict with same keys, but with 2.5% margin from each end
        """
        safe_limits = {}
        for axis in ['x', 'y']:
            min_val = limits[f'{axis}_min']
            max_val = limits[f'{axis}_max']
            total_range = max_val - min_val
            margin = total_range * 0.025  # 2.5% margin
            safe_limits[f'{axis}_min'] = min_val + margin
            safe_limits[f'{axis}_max'] = max_val - margin
        return safe_limits

    def _apply_detected_limits(self, limits: dict):
        """Apply detected limits to the UI range fields."""
        self._range_enable_chk.setChecked(True)
        self._x_min_spin.setValue(limits['x_min'])
        self._x_max_spin.setValue(limits['x_max'])
        self._y_min_spin.setValue(limits['y_min'])
        self._y_max_spin.setValue(limits['y_max'])


class AutoDetectWorker(QtCore.QObject):
    """Worker for auto-detecting stage limits in background thread."""

    started = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(float, str)  # progress (0-100), status message
    finished = QtCore.pyqtSignal(dict)  # detected limits
    error = QtCore.pyqtSignal(str)  # error message

    def __init__(self, stage):
        super().__init__()
        self._stage = stage
        self._cancelled = False

    def run(self):
        """Execute the auto-detection algorithm."""
        try:
            self.started.emit()
            limits = self._detect_limits()
            if not self._cancelled:
                self.finished.emit(limits)
        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))

    def _detect_limits(self) -> dict:
        """Detect stage limits by moving to each direction until failure."""
        # Get current position as starting point
        start_x, start_y = self._stage.get_position()
        logger.info("Starting auto-detection from position: x=%s, y=%s", start_x, start_y)

        limits = {
            'x_min': None,
            'x_max': None,
            'y_min': None,
            'y_max': None
        }

        # Detect X limits
        if not self._cancelled:
            self.progress.emit(10, "Detecting X negative limit...")
            limits['x_min'] = self._detect_axis_limit('x', -1, start_x, start_y)

        if not self._cancelled:
            self.progress.emit(30, "Detecting X positive limit...")
            limits['x_max'] = self._detect_axis_limit('x', 1, start_x, start_y)

        # Detect Y limits
        if not self._cancelled:
            self.progress.emit(50, "Detecting Y negative limit...")
            limits['y_min'] = self._detect_axis_limit('y', -1, start_x, start_y)

        if not self._cancelled:
            self.progress.emit(70, "Detecting Y positive limit...")
            limits['y_max'] = self._detect_axis_limit('y', 1, start_x, start_y)

        if not self._cancelled:
            self.progress.emit(90, "Returning to start position...")
            self._safe_move_to(start_x, start_y)

        self.progress.emit(100, "Detection complete")
        return limits

    def _detect_axis_limit(self, axis: str, direction: int, start_x: float, start_y: float) -> float:
        """Detect limit for a single axis in a given direction using adaptive step size.

        Args:
            axis: 'x' or 'y'
            direction: -1 for negative, 1 for positive
            start_x, start_y: Starting position

        Returns:
            The limit position in steps
        """
        # Adaptive search parameters
        min_step = 10.0      # Minimum step size for precision
        max_step = 5000.0    # Maximum step size for speed
        initial_step = 500.0 # Starting step size
        
        current_x, current_y = self._stage.get_position()
        last_successful_position = current_x if axis == 'x' else current_y
        current_step = initial_step
        
        # Adaptive search: exponentially increase step size until failure, then binary search
        phase = "expanding"  # "expanding" or "refining"
        lower_bound = last_successful_position
        upper_bound = None
        
        max_iterations = 100  # Safety limit
        iteration = 0
        
        while iteration < max_iterations and not self._cancelled:
            iteration += 1
            
            try:
                # Calculate new position based on current phase
                if phase == "expanding":
                    # Exponentially increase step size for speed
                    step = current_step
                    if axis == 'x':
                        new_x = current_x + (direction * step)
                        new_y = current_y
                    else:
                        new_x = current_x
                        new_y = current_y + (direction * step)
                else:
                    # Binary search for precision
                    mid = (lower_bound + upper_bound) / 2
                    if axis == 'x':
                        new_x = mid
                        new_y = current_y
                    else:
                        new_x = current_x
                        new_y = mid

                # Try to move with comprehensive error handling
                self._safe_move_to(new_x, new_y)
                new_x, new_y = self._stage.get_position()
                current_x, current_y = new_x, new_y
                last_successful_position = current_x if axis == 'x' else current_y
                
                if phase == "expanding":
                    # Movement successful - increase step size for next iteration
                    lower_bound = last_successful_position
                    current_step = min(current_step * 1.5, max_step)  # Increase by 50%, cap at max_step
                    
                    # Update progress
                    progress = 10 + (iteration / max_iterations) * 20
                    self.progress.emit(progress, f"Expanding search {axis.upper()} direction {direction:+d}: step {iteration} (step size: {current_step:.0f})")
                else:
                    # Movement successful in refining phase - update lower bound
                    lower_bound = mid
                    
                    # Update progress
                    progress = 50 + (iteration / max_iterations) * 20
                    self.progress.emit(progress, f"Refining {axis.upper()} direction {direction:+d}: iteration {iteration}")

            except (LimitReachedError, Exception) as e:
                # Movement failed - handle based on phase
                error_msg = str(e).lower()
                logger.info("Movement failed at iteration %d (phase %s): %s", iteration, phase, e)
                
                # Check if it's our custom limit error or a regular exception
                is_limit_error = isinstance(e, LimitReachedError) or any(keyword in error_msg for keyword in 
                    ['limit', 'boundary', 'range', 'out of', 'exceed', 'maximum', 'minimum', 'timeout'])
                
                if phase == "expanding":
                    # First failure - switch to binary search mode
                    phase = "refining"
                    upper_bound = last_successful_position + (direction * current_step)
                    current_step = min_step  # Use minimum step for precision
                    logger.info("Switching to refining phase. Bounds: [%s, %s]", lower_bound, upper_bound)
                else:
                    # Failure in refining phase - update upper bound
                    upper_bound = (lower_bound + upper_bound) / 2
                    
                    # Check if we've converged sufficiently
                    if abs(upper_bound - lower_bound) < min_step:
                        logger.info("Converged to within min_step: %s", abs(upper_bound - lower_bound))
                        break
                
                # Small delay to let hardware recover
                time.sleep(0.2)
        
        # Return the last successful position as the limit
        logger.info("Final limit for %s direction %d: %s", axis, direction, last_successful_position)
        return last_successful_position

    def _safe_move_to(self, x: float, y: float):
        """Safely move to position with comprehensive error handling and crash prevention."""
        try:
            # Log the movement attempt
            logger.debug("Attempting move to: x=%s, y=%s", x, y)
            
            # Perform the movement
            self._stage.move_to(x, y)
            
            # Allow time for movement to complete
            time.sleep(0.2)
            
            # Verify the movement was successful by reading position
            try:
                current_x, current_y = self._stage.get_position()
                logger.debug("Current position after move: x=%s, y=%s", current_x, current_y)
            except Exception as pos_error:
                logger.warning("Could not verify position after move: %s", pos_error)
                
        except Exception as e:
            # Comprehensive error classification
            error_msg = str(e).lower()
            logger.warning("Movement failed for x=%s, y=%s: %s", x, y, e)
            
            # Classify error types
            is_limit_error = any(keyword in error_msg for keyword in 
                               ['limit', 'boundary', 'range', 'out of', 'exceed', 'maximum', 'minimum'])
            is_timeout_error = any(keyword in error_msg for keyword in 
                                 ['timeout', 'timed out', 'time out'])
            is_hardware_error = any(keyword in error_msg for keyword in 
                                  ['hardware', 'device', 'connection', 'disconnected', 'not connected'])
            is_movement_error = any(keyword in error_msg for keyword in 
                                  ['move', 'movement', 'position', 'step'])
            
            # Handle different error types
            if is_limit_error:
                # This is expected when hitting limits - treat as limit reached
                logger.info("Limit reached (expected): %s", e)
                raise LimitReachedError(f"Limit reached: {e}")
            elif is_timeout_error:
                # Timeout might indicate a limit or communication issue
                logger.info("Movement timeout (might indicate limit): %s", e)
                raise LimitReachedError(f"Movement timeout (limit): {e}")
            elif is_hardware_error:
                # Hardware errors should be raised for upper level handling
                logger.error("Hardware error during movement: %s", e)
                raise RuntimeError(f"Hardware error: {e}")
            elif is_movement_error:
                # General movement errors - likely limit related
                logger.info("Movement error (likely limit): %s", e)
                raise LimitReachedError(f"Movement error (limit): {e}")
            else:
                # Unknown errors - log and raise
                logger.error("Unknown movement error: %s", e)
                raise RuntimeError(f"Unknown movement error: {e}")
