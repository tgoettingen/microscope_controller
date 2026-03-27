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

        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.001, 1000)
        self.scale_spin.setValue(1.0)

        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-1000, 1000)
        self.offset_spin.setValue(0.0)

        # Strip-chart controls (detector streaming only)
        form.addRow("Sample interval [s]", self.interval_spin)
        form.addRow("Moving window [s]", self.window_time_spin)
        form.addRow("Output dir", self.output_dir_edit)
        form.addRow("", self.browse_btn)
        form.addRow("Detector scale", self.scale_spin)
        form.addRow("Detector offset", self.offset_spin)

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
        try:
            repo_root = Path(__file__).resolve().parents[2]
        except Exception:
            repo_root = Path.cwd()
        return repo_root / "data"

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
            "det_scale": self.scale_spin.value(),
            "det_offset": self.offset_spin.value(),
        }
        self.start_requested.emit(cfg)