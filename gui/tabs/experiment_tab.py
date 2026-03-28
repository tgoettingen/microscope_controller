from pathlib import Path

from PyQt6 import QtWidgets, QtCore


class ExperimentTab(QtWidgets.QWidget):
    start_requested = QtCore.pyqtSignal(dict)
    stop_requested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)

        # Legacy fields kept for compatibility with existing experiment
        # save/load code paths, but they are no longer shown in the strip-chart UI.
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["sim", "real"])

        self.n_timepoints_spin = QtWidgets.QSpinBox()
        self.n_timepoints_spin.setRange(1, 10000)
        self.n_timepoints_spin.setValue(3)

        self.interval_spin = QtWidgets.QDoubleSpinBox()
        # Use dot-decimal parsing consistently so values like 0.01 are not
        # interpreted as 1.0 under locales that use ',' as decimal separator.
        c_locale = QtCore.QLocale(QtCore.QLocale.Language.C)
        c_locale.setNumberOptions(QtCore.QLocale.NumberOption.RejectGroupSeparator)
        self.interval_spin.setLocale(c_locale)
        self.interval_spin.setRange(1e-8, 3600.0)
        self.interval_spin.setDecimals(8)
        self.interval_spin.setSingleStep(1e-4)
        self.interval_spin.setValue(0.05)

        self.window_time_spin = QtWidgets.QDoubleSpinBox()
        self.window_time_spin.setRange(0.1, 36000.0)
        self.window_time_spin.setDecimals(2)
        self.window_time_spin.setSingleStep(1.0)
        self.window_time_spin.setValue(5.0)
        self.window_time_spin.setSuffix(" s")

        self.z_start_spin = QtWidgets.QDoubleSpinBox()
        self.z_start_spin.setRange(-10000, 10000)
        self.z_start_spin.setValue(90.0)

        self.z_end_spin = QtWidgets.QDoubleSpinBox()
        self.z_end_spin.setRange(-10000, 10000)
        self.z_end_spin.setValue(110.0)

        self.z_step_spin = QtWidgets.QDoubleSpinBox()
        self.z_step_spin.setRange(0.1, 1000)
        self.z_step_spin.setValue(10.0)

        self.output_dir_edit = QtWidgets.QLineEdit(str(self._default_output_dir()))
        self.browse_btn = QtWidgets.QPushButton("Browse...")

        # Strip-chart controls (detector streaming only)
        timing_row = QtWidgets.QWidget()
        timing_layout = QtWidgets.QHBoxLayout(timing_row)
        timing_layout.setContentsMargins(0, 0, 0, 0)
        timing_layout.setSpacing(8)
        timing_layout.addWidget(QtWidgets.QLabel("Sample interval [s]"), 0)
        timing_layout.addWidget(self.interval_spin, 0)
        timing_layout.addSpacing(10)
        timing_layout.addWidget(QtWidgets.QLabel("Moving window [s]"), 0)
        timing_layout.addWidget(self.window_time_spin, 0)
        timing_layout.addStretch(1)
        form.addRow("Timing", timing_row)
        out_row = QtWidgets.QWidget()
        out_row_layout = QtWidgets.QHBoxLayout(out_row)
        out_row_layout.setContentsMargins(0, 0, 0, 0)
        out_row_layout.setSpacing(6)
        out_row_layout.addWidget(self.output_dir_edit, 1)
        out_row_layout.addWidget(self.browse_btn, 0)
        form.addRow("Output dir", out_row)

        layout.addLayout(form)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.setSpacing(6)
        self.start_btn = QtWidgets.QPushButton("Start Strip Chart")
        self.stop_btn = QtWidgets.QPushButton("Stop Strip Chart")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        layout.addStretch(1)

        self.browse_btn.clicked.connect(self._choose_output_dir)
        self.start_btn.clicked.connect(self._emit_start)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

    def _choose_output_dir(self):
        current = self.output_dir_edit.text().strip() or str(self._default_output_dir())
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory", current
        )
        if d:
            self.output_dir_edit.setText(d)

    def _default_output_dir(self) -> Path:
        """Return the strip-chart default output directory under the workspace."""
        # Prefer the repository root's data/ directory. If resolution fails, fall
        # back to ./data in the current working directory.
        p = None
        try:
            repo_root = Path(__file__).resolve().parents[2]
            p = repo_root / "data"
        except Exception:
            p = None

        if p is None:
            p = Path.cwd() / "data"

        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return p

    def _emit_start(self):
        cfg = {
            # keep legacy keys for compatibility with save/load paths
            "mode": self.mode_combo.currentText(),
            "n_timepoints": self.n_timepoints_spin.value(),
            "interval_s": self.interval_spin.value(),
            "window_time_s": self.window_time_spin.value(),
            "z_start": self.z_start_spin.value(),
            "z_end": self.z_end_spin.value(),
            "z_step": self.z_step_spin.value(),
            "output_dir": self.output_dir_edit.text(),
            # Scaling is defined in config/default_devices.json; do not expose
            # it in the strip-chart UI.
            "det_scale": 1.0,
            "det_offset": 0.0,
        }
        self.start_requested.emit(cfg)