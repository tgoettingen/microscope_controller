from __future__ import annotations

from PyQt6 import QtWidgets, QtCore, QtGui


class CameraControlTab(QtWidgets.QWidget):
    """Simple camera control panel: exposure, snapshot, live/stop."""

    exposure_changed = QtCore.pyqtSignal(float)
    frame_rate_changed = QtCore.pyqtSignal(float)
    snapshot_requested = QtCore.pyqtSignal()
    live_toggled = QtCore.pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()

        self.exposure_spin = QtWidgets.QDoubleSpinBox()
        self.exposure_spin.setRange(0.1, 10000.0)
        self.exposure_spin.setDecimals(1)
        self.exposure_spin.setSuffix(" ms")
        self.exposure_spin.setValue(20.0)

        self.fps_spin = QtWidgets.QDoubleSpinBox()
        self.fps_spin.setRange(0.5, 120.0)
        self.fps_spin.setDecimals(1)
        self.fps_spin.setSuffix(" fps")
        self.fps_spin.setValue(10.0)

        form.addRow("Exposure", self.exposure_spin)
        form.addRow("Frame rate", self.fps_spin)
        layout.addLayout(form)

        btns = QtWidgets.QHBoxLayout()
        self.snapshot_btn = QtWidgets.QPushButton("Snapshot")
        self.live_btn = QtWidgets.QPushButton("Live")
        self.live_btn.setCheckable(True)
        btns.addWidget(self.snapshot_btn)
        btns.addWidget(self.live_btn)
        layout.addLayout(btns)
        layout.addStretch(1)

        self.exposure_spin.valueChanged.connect(lambda v: self.exposure_changed.emit(float(v)))
        self.fps_spin.valueChanged.connect(lambda v: self.frame_rate_changed.emit(float(v)))
        self.snapshot_btn.clicked.connect(self.snapshot_requested.emit)
        self.live_btn.toggled.connect(self._on_live_toggled)

    def _on_live_toggled(self, checked: bool) -> None:
        self.live_btn.setText("Stop" if checked else "Live")
        self.live_toggled.emit(bool(checked))

    def set_live_checked(self, live: bool) -> None:
        try:
            self.live_btn.blockSignals(True)
            self.live_btn.setChecked(bool(live))
            self.live_btn.setText("Stop" if live else "Live")
        finally:
            try:
                self.live_btn.blockSignals(False)
            except Exception:
                pass
    
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
