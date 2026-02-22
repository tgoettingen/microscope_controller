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

        form = QtWidgets.QFormLayout()

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["sim", "real"])

        self.n_timepoints_spin = QtWidgets.QSpinBox()
        self.n_timepoints_spin.setRange(1, 10000)
        self.n_timepoints_spin.setValue(3)

        self.interval_spin = QtWidgets.QDoubleSpinBox()
        self.interval_spin.setRange(0.0, 3600.0)
        self.interval_spin.setValue(2.0)

        self.z_start_spin = QtWidgets.QDoubleSpinBox()
        self.z_start_spin.setRange(-10000, 10000)
        self.z_start_spin.setValue(90.0)

        self.z_end_spin = QtWidgets.QDoubleSpinBox()
        self.z_end_spin.setRange(-10000, 10000)
        self.z_end_spin.setValue(110.0)

        self.z_step_spin = QtWidgets.QDoubleSpinBox()
        self.z_step_spin.setRange(0.1, 1000)
        self.z_step_spin.setValue(10.0)

        self.output_dir_edit = QtWidgets.QLineEdit(str(Path.cwd() / "data"))
        self.browse_btn = QtWidgets.QPushButton("Browse...")

        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.001, 1000)
        self.scale_spin.setValue(1.0)

        self.offset_spin = QtWidgets.QDoubleSpinBox()
        self.offset_spin.setRange(-1000, 1000)
        self.offset_spin.setValue(0.0)

        form.addRow("Mode", self.mode_combo)
        form.addRow("Timepoints", self.n_timepoints_spin)
        form.addRow("Interval [s]", self.interval_spin)
        form.addRow("Z start", self.z_start_spin)
        form.addRow("Z end", self.z_end_spin)
        form.addRow("Z step", self.z_step_spin)
        form.addRow("Output dir", self.output_dir_edit)
        form.addRow("", self.browse_btn)
        form.addRow("Detector scale", self.scale_spin)
        form.addRow("Detector offset", self.offset_spin)

        layout.addLayout(form)

        btn_layout = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start")
        self.stop_btn = QtWidgets.QPushButton("Stop")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)
        layout.addStretch(1)

        self.browse_btn.clicked.connect(self._choose_output_dir)
        self.start_btn.clicked.connect(self._emit_start)
        self.stop_btn.clicked.connect(self.stop_requested.emit)

    def _choose_output_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory", self.output_dir_edit.text()
        )
        if d:
            self.output_dir_edit.setText(d)

    def _emit_start(self):
        cfg = {
            "mode": self.mode_combo.currentText(),
            "n_timepoints": self.n_timepoints_spin.value(),
            "interval_s": self.interval_spin.value(),
            "z_start": self.z_start_spin.value(),
            "z_end": self.z_end_spin.value(),
            "z_step": self.z_step_spin.value(),
            "output_dir": self.output_dir_edit.text(),
            "det_scale": self.scale_spin.value(),
            "det_offset": self.offset_spin.value(),
        }
        self.start_requested.emit(cfg)