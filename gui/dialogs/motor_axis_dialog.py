from __future__ import annotations

import json
import re
from pathlib import Path

from PyQt6 import QtWidgets, QtGui, QtCore

from core.multiaxis import AxisConfig
from core.factory import build_devices, load_config


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_stage_scaling(config_path: str | Path | None) -> tuple[float, float, float, float]:
    """Return (x_scale, x_offset, y_scale, y_offset) from the device config.

    Falls back to (1, 0, 1, 0) on any error.
    """
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        sc = cfg.get("stage", {}).get("scaling", {})
        return (
            float(sc.get("x_scale", 1.0)),
            float(sc.get("x_offset", 0.0)),
            float(sc.get("y_scale", 1.0)),
            float(sc.get("y_offset", 0.0)),
        )
    except Exception:
        return 1.0, 0.0, 1.0, 0.0


def _stage_unit(config_path: str | Path | None) -> str:
    """Return 'mm' if a non-trivial calibration is configured, else 'steps'."""
    xs, xo, ys, yo = _read_stage_scaling(config_path)
    if xs != 1.0 or xo != 0.0 or ys != 1.0 or yo != 0.0:
        return "mm"
    return "steps"


# ── Scientific-notation-aware QDoubleSpinBox ──────────────────────────────────

class ScientificSpinBox(QtWidgets.QDoubleSpinBox):
    """QDoubleSpinBox that accepts and displays values in scientific notation.

    - Displays up to 9 significant digits in g-format (e.g. 1.23456789e+03).
    - The user can type either plain decimal (1234.5) or sci-notation (1.2345e3).
    - Step-up/down still works using the configured singleStep().
    """

    _SCI_RE = re.compile(r"^\s*[+-]?\d*\.?\d+([eE][+-]?\d+)?\s*$")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDecimals(9)
        self.setRange(-1e18, 1e18)
        self.setMinimumWidth(150)

    def textFromValue(self, value: float) -> str:
        if value == 0.0:
            return "0"
        return f"{value:.9g}"

    def _strip_affixes(self, text: str) -> str:
        """Remove the spinbox prefix and suffix from a raw text string."""
        t = text
        pfx = self.prefix()
        sfx = self.suffix()
        if pfx and t.startswith(pfx):
            t = t[len(pfx):]
        if sfx and t.endswith(sfx):
            t = t[: len(t) - len(sfx)]
        return t.strip()

    def valueFromText(self, text: str) -> float:
        try:
            return float(self._strip_affixes(text))
        except ValueError:
            return 0.0

    def validate(self, text: str, pos: int):
        t = self._strip_affixes(text)
        # Still being typed: empty, bare sign, or incomplete exponent
        if t in ("", "+", "-", ".", "+.", "-."):
            return QtGui.QValidator.State.Intermediate, text, pos
        if re.search(r"[eE][+-]?$", t):
            return QtGui.QValidator.State.Intermediate, text, pos
        if self._SCI_RE.match(t):
            try:
                v = float(t)
                if self.minimum() <= v <= self.maximum():
                    return QtGui.QValidator.State.Acceptable, text, pos
                return QtGui.QValidator.State.Intermediate, text, pos
            except ValueError:
                pass
        return QtGui.QValidator.State.Intermediate, text, pos

    def fixup(self, text: str) -> str:
        try:
            v = max(self.minimum(), min(self.maximum(), float(self._strip_affixes(text))))
            return self.textFromValue(v)
        except ValueError:
            return self.textFromValue(self.value())


# ── Dialog ────────────────────────────────────────────────────────────────────

class MotorAxisDialog(QtWidgets.QDialog):
    def __init__(self, axis_type: str, config=None, parent=None,
                 config_path: str | Path | None = None):
        super().__init__(parent)
        self.axis_type = axis_type
        self._config_path = config_path
        self._fallback_stage = None
        self._fallback_focus = None
        self.setWindowTitle(f"{axis_type} Axis Settings")

        # Determine unit for X/Y stage axes from the device config
        is_motor_axis = axis_type in ("X", "Y")
        if is_motor_axis:
            self._unit = _stage_unit(config_path)
            xs, xo, ys, yo = _read_stage_scaling(config_path)
        else:
            self._unit = "steps"

        layout = QtWidgets.QFormLayout(self)

        # ── Unit banner (X/Y axes only) ──────────────────────────────────────
        if is_motor_axis:
            if self._unit == "mm":
                xs_str = f"{xs:.6g}"
                ys_str = f"{ys:.6g}"
                banner_text = (
                    f"<b>Unit: mm</b> — stage calibration is active<br>"
                    f"<small>X: {xs_str} steps/mm &nbsp;|&nbsp; Y: {ys_str} steps/mm<br>"
                    f"Enter Start / End / Step in <b>millimetres</b>.</small>"
                )
                banner_color = "#e6f4ea"   # light green
            else:
                banner_text = (
                    "<b>Unit: steps</b> — no calibration configured<br>"
                    "<small>Enter Start / End / Step in raw motor <b>steps</b>.<br>"
                    "Run <i>Action → Calibrate Stage…</i> to enable mm units.</small>"
                )
                banner_color = "#fff3cd"   # amber warning
            banner = QtWidgets.QLabel(banner_text)
            banner.setWordWrap(True)
            banner.setStyleSheet(
                f"background:{banner_color}; border:1px solid #ccc; "
                f"border-radius:4px; padding:6px;"
            )
            layout.addRow(banner)

        # ── Scan range ───────────────────────────────────────────────────────
        self.start_spin = ScientificSpinBox()
        self.end_spin   = ScientificSpinBox()
        self.step_spin  = ScientificSpinBox()

        self.start_spin.setRange(-1e18, 1e18)
        self.end_spin.setRange(-1e18, 1e18)
        self.step_spin.setRange(-1e18, 1e18)

        self.start_spin.setValue(0)
        self.end_spin.setValue(1000)
        self.step_spin.setValue(100)

        # Show unit suffix on the three position spinboxes for X/Y/Z axes
        if axis_type in ("X", "Y", "Z"):
            unit = self._unit if axis_type in ("X", "Y") else "steps"
            suffix = f" {unit}"
            self.start_spin.setSuffix(suffix)
            self.end_spin.setSuffix(suffix)
            self.step_spin.setSuffix(suffix)

        # ── Wait ─────────────────────────────────────────────────────────────
        self.wait_spin = ScientificSpinBox()
        self.wait_spin.setRange(0.0, 3600.0)
        self.wait_spin.setValue(0.05)
        self.wait_spin.setSuffix(" s")

        # ── Motor selection & mode ───────────────────────────────────────────
        self.motors_edit = QtWidgets.QLineEdit()
        self.motors_edit.setPlaceholderText(
            "Comma-separated device names (e.g. stage,focus)"
        )
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["sequential", "synchronized"])

        layout.addRow("Start", self.start_spin)
        layout.addRow("End", self.end_spin)
        layout.addRow("Step", self.step_spin)
        layout.addRow("Wait", self.wait_spin)
        layout.addRow("Motors", self.motors_edit)
        layout.addRow("Motor mode", self.mode_combo)

        # ── Live motor position (read-only, updates every second) ──────────
        self.current_pos_edit = QtWidgets.QLineEdit()
        self.current_pos_edit.setReadOnly(True)
        self.current_pos_edit.setPlaceholderText("N/A")
        self._set_current_pos_style(False)
        layout.addRow("Current position", self.current_pos_edit)

        # ── Synchronized-move parameters ─────────────────────────────────────
        self.sync_timeout_spin = ScientificSpinBox()
        self.sync_timeout_spin.setRange(0.001, 3600.0)
        self.sync_timeout_spin.setValue(5.0)
        self.sync_timeout_spin.setSuffix(" s")

        self.sync_poll_spin = ScientificSpinBox()
        self.sync_poll_spin.setRange(1e-9, 10.0)
        self.sync_poll_spin.setValue(0.01)
        self.sync_poll_spin.setSuffix(" s")

        self.sync_tol_spin = ScientificSpinBox()
        self.sync_tol_spin.setRange(1e-18, 1e6)
        self.sync_tol_spin.setValue(1e-3)

        layout.addRow("Sync timeout", self.sync_timeout_spin)
        layout.addRow("Sync poll", self.sync_poll_spin)
        layout.addRow("Sync tolerance", self.sync_tol_spin)

        # ── Pre/post positions ───────────────────────────────────────────────
        self.pre_pos_spin = ScientificSpinBox()
        self.pre_pos_spin.setRange(-1e18, 1e18)
        self.pre_pos_spin.setValue(0.0)
        self.pre_pos_spin.setSpecialValueText("(none)")

        self.post_pos_spin = ScientificSpinBox()
        self.post_pos_spin.setRange(-1e18, 1e18)
        self.post_pos_spin.setValue(0.0)
        self.post_pos_spin.setSpecialValueText("(none)")

        if axis_type in ("X", "Y", "Z"):
            unit = self._unit if axis_type in ("X", "Y") else "steps"
            self.pre_pos_spin.setSuffix(f" {unit}")
            self.post_pos_spin.setSuffix(f" {unit}")

        layout.addRow("Pre-scan pos", self.pre_pos_spin)
        layout.addRow("Post-scan pos", self.post_pos_spin)

        # ── OK / Cancel ──────────────────────────────────────────────────────
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

        # Restore values from a previous AxisConfig if provided
        if config is not None:
            self._restore(config)

        self._pos_timer = QtCore.QTimer(self)
        self._pos_timer.setInterval(1000)
        self._pos_timer.timeout.connect(self._update_current_position)
        self._pos_timer.start()
        self._update_current_position()

    def _root_main_window(self):
        """Return top-level window to access live stage/focus devices if available."""
        try:
            return self.window()
        except Exception:
            return None

    def _configured_device_type(self, key: str) -> str | None:
        """Return the configured device type for the given config section."""
        try:
            cfg = load_config(self._config_path) if self._config_path else {}
            section = cfg.get(key, {}) if isinstance(cfg, dict) else {}
            if isinstance(section, dict):
                dev_type = section.get("type")
                return str(dev_type) if dev_type is not None else None
        except Exception:
            pass
        return None

    def _fallback_position_device(self, key: str):
        """Build a simulated stage/focus only when no live device exists yet."""
        cache_name = "_fallback_stage" if key == "stage" else "_fallback_focus"
        cached = getattr(self, cache_name, None)
        if cached is not None:
            return cached

        dev_type = self._configured_device_type(key)
        if dev_type != "simulated":
            return None

        try:
            _, stage, focus, *_ = build_devices(self._config_path or "config/default_devices.json")
            device = stage if key == "stage" else focus
            setattr(self, cache_name, device)
            return device
        except Exception:
            return None

    def _read_axis_position(self) -> float | None:
        """Read the current axis position from connected devices."""
        root = self._root_main_window()

        try:
            if self.axis_type in ("X", "Y"):
                stage = getattr(root, "stage", None) if root is not None else None
                if stage is None:
                    stage = self._fallback_position_device("stage")
                if stage is None or not hasattr(stage, "get_position"):
                    return None
                pos = stage.get_position()
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    return float(pos[0] if self.axis_type == "X" else pos[1])
                # Fallback for unusual stage implementations returning a scalar.
                return float(pos)

            if self.axis_type == "Z":
                focus = getattr(root, "focus", None) if root is not None else None
                if focus is None:
                    focus = self._fallback_position_device("focus")
                if focus is None or not hasattr(focus, "get_position"):
                    return None
                return float(focus.get_position())
        except Exception:
            return None

        return None

    def _update_current_position(self) -> None:
        """Refresh the read-only current-position field."""
        try:
            pos = self._read_axis_position()
            if pos is None:
                self.current_pos_edit.setText("N/A")
                self._set_current_pos_style(False)
                return
            unit = self._unit if self.axis_type in ("X", "Y") else "steps"
            self.current_pos_edit.setText(f"{pos:.6g} {unit}")
            self._set_current_pos_style(True)
        except Exception:
            try:
                self.current_pos_edit.setText("N/A")
                self._set_current_pos_style(False)
            except Exception:
                pass

    def _set_current_pos_style(self, live: bool) -> None:
        """Colorize current-position field based on live-data availability."""
        try:
            if live:
                # Soft green when we can read a live position value.
                self.current_pos_edit.setStyleSheet(
                    "QLineEdit { background: #e8f5e9; color: #1b5e20; border: 1px solid #a5d6a7; }"
                )
            else:
                # Muted gray when motor is unavailable.
                self.current_pos_edit.setStyleSheet(
                    "QLineEdit { background: #f3f4f6; color: #6b7280; border: 1px solid #d1d5db; }"
                )
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            if hasattr(self, "_pos_timer") and self._pos_timer is not None:
                self._pos_timer.stop()
        except Exception:
            pass
        super().closeEvent(event)

    # ── Restore from existing config ──────────────────────────────────────────

    def _restore(self, config: AxisConfig) -> None:
        p = config.params
        try: self.start_spin.setValue(p["start"])
        except Exception: pass
        try: self.end_spin.setValue(p["end"])
        except Exception: pass
        try: self.step_spin.setValue(p["step"])
        except Exception: pass
        try: self.wait_spin.setValue(p.get("wait", 0.05))
        except Exception: pass
        try:
            motors = p.get("motors") or []
            self.motors_edit.setText(", ".join(motors))
        except Exception: pass
        try:
            idx = self.mode_combo.findText(p.get("motor_mode", "sequential"))
            if idx >= 0:
                self.mode_combo.setCurrentIndex(idx)
        except Exception: pass
        try: self.sync_timeout_spin.setValue(p.get("sync_timeout", 5.0))
        except Exception: pass
        try: self.sync_poll_spin.setValue(p.get("sync_poll", 0.01))
        except Exception: pass
        try: self.sync_tol_spin.setValue(p.get("sync_tol", 1e-3))
        except Exception: pass
        try:
            v = p.get("pre_pos")
            self.pre_pos_spin.setValue(float(v) if v is not None else 0.0)
        except Exception: pass
        try:
            v = p.get("post_pos")
            self.post_pos_spin.setValue(float(v) if v is not None else 0.0)
        except Exception: pass

    # ── Extract config ────────────────────────────────────────────────────────

    def get_config(self) -> AxisConfig:
        motors_raw = self.motors_edit.text().strip()
        motors = [m.strip() for m in motors_raw.split(",") if m.strip()]
        pre_v  = self.pre_pos_spin.value()
        post_v = self.post_pos_spin.value()
        return AxisConfig(
            axis_type=self.axis_type,
            params={
                "start":        self.start_spin.value(),
                "end":          self.end_spin.value(),
                "step":         self.step_spin.value(),
                "wait":         self.wait_spin.value(),
                "motors":       motors,
                "motor_mode":   self.mode_combo.currentText(),
                "sync_timeout": self.sync_timeout_spin.value(),
                "sync_poll":    self.sync_poll_spin.value(),
                "sync_tol":     self.sync_tol_spin.value(),
                "pre_pos":  None if pre_v  == 0.0 and self.pre_pos_spin.specialValueText()  else pre_v,
                "post_pos": None if post_v == 0.0 and self.post_pos_spin.specialValueText() else post_v,
            },
        )