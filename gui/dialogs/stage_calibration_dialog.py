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
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from PyQt6 import QtWidgets, QtCore

logger = logging.getLogger(__name__)


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

        # ── Instructions ──────────────────────────────────────────────
        info = QtWidgets.QLabel(
            "<b>How it works:</b><br>"
            "1. Click <i>Set Reference Point</i> to record the current stage position.<br>"
            "2. Move the stage to a new position (use the stage jog controls or hardware).<br>"
            "3. Enter the <b>physical distance</b> you moved (in mm) for each axis.<br>"
            "4. Click <i>Calculate &amp; Save</i> — the scaling factor will be computed and "
            "written to the device config."
        )
        info.setWordWrap(True)
        root.addWidget(info)
        root.addSpacing(8)

        # ── Live position display ──────────────────────────────────────
        # Position is always in raw steps (that is what get_position() returns
        # before ScaledStageXY is active, and what we need to compute Δsteps).
        pos_grp = QtWidgets.QGroupBox("Current Stage Position  [raw steps — as read from hardware]")
        pos_lay = QtWidgets.QFormLayout(pos_grp)

        self._cur_x_label = QtWidgets.QLabel("—")
        self._cur_y_label = QtWidgets.QLabel("—")
        pos_lay.addRow("X:", self._cur_x_label)
        pos_lay.addRow("Y:", self._cur_y_label)
        root.addWidget(pos_grp)

        # ── Step 1 — Set reference ─────────────────────────────────────
        ref_grp = QtWidgets.QGroupBox("Step 1 — Set Reference Point  [records current step count]")
        ref_lay = QtWidgets.QVBoxLayout(ref_grp)

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
        meas_grp = QtWidgets.QGroupBox(
            "Step 2 — Enter Physical Distance Moved  [measured externally, in mm]"
        )
        meas_lay = QtWidgets.QFormLayout(meas_grp)

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
        result_grp = QtWidgets.QGroupBox("Scaling (current → new)")
        result_lay = QtWidgets.QFormLayout(result_grp)

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
        if self._computed_x is None or self._computed_y is None:
            return
        try:
            cfg = self._read_config()
            stage_cfg = cfg.setdefault("stage", {})
            scaling = stage_cfg.setdefault("scaling", {})
            scaling["x_scale"] = round(self._computed_x, 6)
            scaling["y_scale"] = round(self._computed_y, 6)
            # Preserve existing offsets — calibration only touches scale
            scaling.setdefault("x_offset", 0.0)
            scaling.setdefault("y_offset", 0.0)
            self._write_config(cfg)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self, "Save Error", f"Could not write config:\n{exc}"
            )
            return

        self.calibration_saved.emit(self._computed_x, self._computed_y)
        self._populate_current_scaling()   # refresh "current" row to show new value
        QtWidgets.QMessageBox.information(
            self,
            "Saved",
            f"Calibration saved to:\n{self._config_path}\n\n"
            f"X scale: {self._computed_x:.4f} steps/mm\n"
            f"Y scale: {self._computed_y:.4f} steps/mm\n\n"
            "The new scaling will take effect the next time devices are built\n"
            "(i.e. when you start the next run)."
        )
        logger.info(
            "Calibration saved: x_scale=%s y_scale=%s -> %s",
            self._computed_x, self._computed_y, self._config_path,
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

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)

    def reject(self):
        self._timer.stop()
        super().reject()
