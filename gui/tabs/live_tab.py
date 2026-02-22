import time

import numpy as np
from PyQt6 import QtWidgets, QtCore
import pyqtgraph as pg


class LiveTab(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 1D detector data (classic experiment)
        self._detector_times: list[float] = []
        self._detector_values: list[float] = []
        self._t0 = time.time()

        # Multi-axis detector data
        self.multi_coords: list[tuple[dict, float]] = []

        # view mode: "camera" or "detector"
        self.view_mode = "camera"

        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        # Toggle between camera and detector view
        toggle_layout = QtWidgets.QHBoxLayout()
        self.camera_btn = QtWidgets.QPushButton("Show Camera")
        self.detector_btn = QtWidgets.QPushButton("Show Detector")
        self.camera_btn.setCheckable(True)
        self.detector_btn.setCheckable(True)
        self.camera_btn.setChecked(True)
        toggle_layout.addWidget(self.camera_btn)
        toggle_layout.addWidget(self.detector_btn)
        layout.addLayout(toggle_layout)

        self.camera_btn.clicked.connect(self._set_camera_view)
        self.detector_btn.clicked.connect(self._set_detector_view)

        # Image view (used for camera images and detector heatmaps/volume slices)
        self.image_view = pg.ImageView()
        layout.addWidget(self.image_view, 3)

        # Z slider for 3D detector volumes
        z_layout = QtWidgets.QHBoxLayout()
        z_layout.addWidget(QtWidgets.QLabel("Z slice:"))
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(0)
        self.z_slider.setValue(0)
        self.z_slider.valueChanged.connect(self._update_multiaxis_visualization)
        z_layout.addWidget(self.z_slider)
        layout.addLayout(z_layout)

        # Detector 1D plot (classic or 1D multi-axis)
        self.plot_widget = pg.PlotWidget()
        self.plot_curve = self.plot_widget.plot([], [])
        self.plot_widget.setLabel("left", "Detector", units="a.u.")
        self.plot_widget.setLabel("bottom", "Time / Coord", units="a.u.")
        layout.addWidget(self.plot_widget, 1)

        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._update_plot)
        self.plot_timer.start(100)

    # -----------------------------
    # view mode
    # -----------------------------
    def _set_camera_view(self):
        self.view_mode = "camera"
        self.camera_btn.setChecked(True)
        self.detector_btn.setChecked(False)

    def _set_detector_view(self):
        self.view_mode = "detector"
        self.camera_btn.setChecked(False)
        self.detector_btn.setChecked(True)

    # -----------------------------
    # reset helpers
    # -----------------------------
    def reset_1d_detector(self):
        self._t0 = time.time()
        self._detector_times.clear()
        self._detector_values.clear()

    def reset_multiaxis(self):
        self.multi_coords.clear()
        self.z_slider.setMaximum(0)
        self.z_slider.setValue(0)

    # -----------------------------
    # camera image callback
    # -----------------------------
    @QtCore.pyqtSlot(object, dict)
    def update_image(self, img, meta: dict):
        if self.view_mode == "camera":
            self.image_view.setImage(np.asarray(img).T, autoLevels=True)

    # -----------------------------
    # 1D detector callback
    # -----------------------------
    def add_detector_sample(self, value: float, timestamp: float):
        t_rel = timestamp - self._t0
        self._detector_times.append(t_rel)
        self._detector_values.append(value)

    # -----------------------------
    # multi-axis detector callback
    # -----------------------------
    def add_multiaxis_detector(self, state: dict, value: float):
        self.multi_coords.append((state.copy(), value))
        # update Z slider range if Z present
        if self.multi_coords:
            sample_state, _ = self.multi_coords[0]
            if "Z" in sample_state:
                zs = sorted(set(s["Z"] for s, _ in self.multi_coords))
                self._z_values = zs
                self.z_slider.setMaximum(max(0, len(zs) - 1))

    # -----------------------------
    # periodic plot update
    # -----------------------------
    def _update_plot(self):
        # 1D detector plot
        if self._detector_times:
            self.plot_curve.setData(self._detector_times, self._detector_values)

        # multi-axis visualization
        if self.multi_coords and self.view_mode == "detector":
            self._update_multiaxis_visualization()

    # -----------------------------
    # multi-axis visualization logic
    # -----------------------------
    def _update_multiaxis_visualization(self):
        if not self.multi_coords:
            return

        sample_state, _ = self.multi_coords[0]
        axes = [a for a in ("X", "Y", "Z") if a in sample_state]

        if len(axes) == 1:
            # 1D scan → line plot
            ax = axes[0]
            coords = [s[ax] for s, _ in self.multi_coords]
            vals = [v for _, v in self.multi_coords]
            self.plot_curve.setData(coords, vals)

        elif len(axes) == 2:
            # 2D scan → heatmap
            ax1, ax2 = axes
            xs = sorted(set(s[ax1] for s, _ in self.multi_coords))
            ys = sorted(set(s[ax2] for s, _ in self.multi_coords))

            arr = np.zeros((len(xs), len(ys)))
            for s, v in self.multi_coords:
                i = xs.index(s[ax1])
                j = ys.index(s[ax2])
                arr[i, j] = v

            self.image_view.setImage(arr.T, autoLevels=True)

        elif len(axes) == 3:
            # 3D scan → volume slice via Z slider
            ax1, ax2, ax3 = axes
            xs = sorted(set(s[ax1] for s, _ in self.multi_coords))
            ys = sorted(set(s[ax2] for s, _ in self.multi_coords))
            zs = sorted(set(s[ax3] for s, _ in self.multi_coords))

            arr = np.zeros((len(xs), len(ys), len(zs)))
            for s, v in self.multi_coords:
                i = xs.index(s[ax1])
                j = ys.index(s[ax2])
                k = zs.index(s[ax3])
                arr[i, j, k] = v

            idx = min(max(self.z_slider.value(), 0), len(zs) - 1)
            slice_img = arr[:, :, idx]
            self.image_view.setImage(slice_img.T, autoLevels=True)