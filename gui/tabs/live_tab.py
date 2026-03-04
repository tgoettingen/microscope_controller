import time
import numpy as np
from collections import deque
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


class LiveTab(QtWidgets.QWidget):
    hover_info = QtCore.pyqtSignal(str)
    # emitted when user toggles streaming for a detector: (detector_id, enabled)
    stream_toggled = QtCore.pyqtSignal(str, bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        # per-detector moving window buffers
        self._window_size = 200
        self._detector_buffers: dict[str, deque] = {}
        self._detector_times: dict[str, deque] = {}
        self._detector_curves: dict[str, pg.PlotDataItem] = {}
        self._t0 = time.time()

        # per-detector multi-axis data: det_id -> list[(state, value)]
        self.multi_coords: dict[str, list[tuple[dict, float]]] = {}

        # image display level state
        self._levels = (None, None)  # (min, max) or (None,None) for auto

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
        # connect hover events (ImageView.scene is an attribute, not callable)
        try:
            self.image_view.scene.sigMouseMoved.connect(self._on_image_mouse_move)
        except Exception:
            # fallback: try view's scene
            try:
                self.image_view.getView().scene.sigMouseMoved.connect(self._on_image_mouse_move)
            except Exception:
                pass
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

        # Per-detector controls (visibility + streaming)
        ctl_box = QtWidgets.QGroupBox("Detectors")
        ctl_layout = QtWidgets.QVBoxLayout(ctl_box)
        self.detector_controls_layout = ctl_layout
        layout.addWidget(ctl_box)

        # Controls: moving-window size
        ctl_layout = QtWidgets.QHBoxLayout()
        ctl_layout.addWidget(QtWidgets.QLabel("Window size:"))
        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setMinimum(10)
        self.window_spin.setMaximum(10000)
        self.window_spin.setValue(self._window_size)
        self.window_spin.valueChanged.connect(self.set_window_size)
        ctl_layout.addWidget(self.window_spin)
        layout.addLayout(ctl_layout)

        # keyboard shortcuts for levels
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl++"), self, activated=self.increase_upper)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+-"), self, activated=self.decrease_upper)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+_"), self, activated=self.decrease_lower)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+0"), self, activated=self.reset_levels)

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
        try:
            self.image_view.show()
            self.detector_images_container.hide()
        except Exception:
            pass

    def _set_detector_view(self):
        self.view_mode = "detector"
        self.camera_btn.setChecked(False)
        self.detector_btn.setChecked(True)
        try:
            self.image_view.hide()
            self.detector_images_container.show()
        except Exception:
            pass

    # -----------------------------
    # reset helpers
    # -----------------------------
    def reset_1d_detector(self):
        self._t0 = time.time()
        for k in list(self._detector_times.keys()):
            self._detector_times[k].clear()
            self._detector_buffers[k].clear()

    def register_detector(self, detector_id: str):
        """Ensure UI and buffers exist for a detector id."""
        if detector_id in self._detector_buffers:
            return
        self._detector_buffers[detector_id] = deque(maxlen=self._window_size)
        self._detector_times[detector_id] = deque(maxlen=self._window_size)
        curve = self.plot_widget.plot([], [], pen=pg.mkPen(width=2))
        self._detector_curves[detector_id] = curve

        # Create a small control row: label, visibility checkbox, stream checkbox
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(2, 2, 2, 2)
        lbl = QtWidgets.QLabel(detector_id)
        vis_cb = QtWidgets.QCheckBox("Show")
        vis_cb.setChecked(True)
        stream_cb = QtWidgets.QCheckBox("Stream")
        stream_cb.setChecked(False)
        row_layout.addWidget(lbl)
        row_layout.addWidget(vis_cb)
        row_layout.addWidget(stream_cb)
        row_layout.addStretch(1)
        self.detector_controls_layout.addWidget(row)

        def _on_vis(chk):
            try:
                self._detector_curves[detector_id].setVisible(bool(chk))
            except Exception:
                pass

        def _on_stream(chk):
            self.stream_toggled.emit(detector_id, bool(chk))

        vis_cb.toggled.connect(_on_vis)
        stream_cb.toggled.connect(_on_stream)

        # create a small image view for this detector (for multi-axis heatmaps)
        img_view = pg.ImageView()
        img_view.setPredefinedGradient('viridis')
        # hide ROI and menu controls but show a compact histogram as a colorbar
        try:
            img_view.ui.roiBtn.hide()
        except Exception:
            pass
        try:
            img_view.ui.menuBtn.hide()
        except Exception:
            pass
        try:
            img_view.ui.histogram.show()
            # make histogram narrow to serve as a colorbar
            img_view.ui.histogram.setFixedWidth(36)
        except Exception:
            pass
        img_view.getView().setMinimumWidth(180)
        img_view.getView().setMinimumHeight(160)
        # place label above the image view
        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)
        title = QtWidgets.QLabel(detector_id)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)
        vbox.addWidget(title)
        vbox.addWidget(img_view)
        self._detector_image_views[detector_id] = img_view
        self.detector_images_layout.addWidget(container)

        # ensure data storage
        self.multi_coords.setdefault(detector_id, [])

        # connect hover from this image view to include detector id
        try:
            img_view.scene.sigMouseMoved.connect(lambda pos, did=detector_id, iv=img_view: self._on_detector_image_mouse_move(pos, did, iv))
        except Exception:
            try:
                img_view.getView().scene.sigMouseMoved.connect(lambda pos, did=detector_id, iv=img_view: self._on_detector_image_mouse_move(pos, did, iv))
            except Exception:
                pass
    def reset_multiaxis(self):
        for v in self.multi_coords.values():
            v.clear()
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
    def add_detector_sample(self, detector_id: str, value: float, timestamp: float | None = None):
        """Add a sample for a named detector. timestamp in seconds. Thread-safe caller should use queued calls."""
        if timestamp is None:
            timestamp = time.time()
        t_rel = timestamp - self._t0
        # ensure buffers and UI exist
        if detector_id not in self._detector_buffers:
            self.register_detector(detector_id)

        self._detector_times[detector_id].append(t_rel)
        self._detector_buffers[detector_id].append(value)

    # -----------------------------
    # multi-axis detector callback
    # -----------------------------
    def add_multiaxis_detector(self, detector_id: str, state: dict, value: float):
        """Append a multi-axis sample for a specific detector id."""
        self.multi_coords.setdefault(detector_id, []).append((state.copy(), value))
        # update Z slider range if Z present (from any detector with Z)
        for det_list in self.multi_coords.values():
            if not det_list:
                continue
            sample_state, _ = det_list[0]
            if "Z" in sample_state:
                zs = sorted(set(s["Z"] for s, _ in det_list))
                self._z_values = zs
                self.z_slider.setMaximum(max(0, len(zs) - 1))
                break

    # -----------------------------
    # periodic plot update
    # -----------------------------
    def _update_plot(self):
        # 1D detector plot
        if self._detector_times:
            # update each detector curve
            for det_id, times in self._detector_times.items():
                vals = self._detector_buffers.get(det_id, [])
                curve = self._detector_curves.get(det_id)
                if curve is not None:
                    curve.setData(list(times), list(vals))

        # multi-axis visualization (per-detector)
        if self.multi_coords and self.view_mode == "detector":
            self._update_multiaxis_visualization()

    # -----------------------------
    # multi-axis visualization logic
    # -----------------------------
    def _update_multiaxis_visualization(self):
        if not self.multi_coords:
            return

        # For each detector, compute its own axes and display in its image view
        for det_id, det_list in self.multi_coords.items():
            if not det_list:
                continue
            sample_state, _ = det_list[0]
            axes = [a for a in ("X", "Y", "Z") if a in sample_state]

            # find target image view (fallback to main image_view)
            img_view = self._detector_image_views.get(det_id, self.image_view)

            if len(axes) == 1:
                # 1D scan: plot line on per-detector curve
                ax = axes[0]
                coords = [s[ax] for s, _ in det_list]
                vals = [v for _, v in det_list]
                try:
                    curve = self._detector_curves.get(det_id)
                    if curve is not None:
                        curve.setData(coords, vals)
                    else:
                        self.plot_curve.setData(coords, vals)
                except Exception:
                    pass

            elif len(axes) == 2:
                ax1, ax2 = axes
                xs = sorted(set(s[ax1] for s, _ in det_list))
                ys = sorted(set(s[ax2] for s, _ in det_list))

                arr = np.zeros((len(xs), len(ys)))
                for s, v in det_list:
                    i = xs.index(s[ax1])
                    j = ys.index(s[ax2])
                    arr[i, j] = v

                try:
                    img_view.setImage(arr.T, autoLevels=True)
                    if self._levels[0] is not None and self._levels[1] is not None:
                        img_view.getImageItem().setLevels(self._levels[0], self._levels[1])
                except Exception:
                    pass

            elif len(axes) == 3:
                ax1, ax2, ax3 = axes
                xs = sorted(set(s[ax1] for s, _ in det_list))
                ys = sorted(set(s[ax2] for s, _ in det_list))
                zs = sorted(set(s[ax3] for s, _ in det_list))

                arr = np.zeros((len(xs), len(ys), len(zs)))
                for s, v in det_list:
                    i = xs.index(s[ax1])
                    j = ys.index(s[ax2])
                    k = zs.index(s[ax3])
                    arr[i, j, k] = v

                idx = min(max(self.z_slider.value(), 0), len(zs) - 1)
                slice_img = arr[:, :, idx]
                try:
                    img_view.setImage(slice_img.T, autoLevels=True)
                    if self._levels[0] is not None and self._levels[1] is not None:
                        img_view.getImageItem().setLevels(self._levels[0], self._levels[1])
                except Exception:
                    pass
    def set_window_size(self, n: int):
        self._window_size = int(n)
        # resize existing buffers
        for k in list(self._detector_buffers.keys()):
            old_vals = list(self._detector_buffers[k])
            old_times = list(self._detector_times[k])
            self._detector_buffers[k] = deque(old_vals[-self._window_size :], maxlen=self._window_size)
            self._detector_times[k] = deque(old_times[-self._window_size :], maxlen=self._window_size)

    # -----------------------------
    # image hover and levels
    # -----------------------------
    def _on_image_mouse_move(self, pos):
        vb = self.image_view.getView()
        mouse_point = vb.mapSceneToView(pos)
        x = int(round(mouse_point.x()))
        y = int(round(mouse_point.y()))
        img_item = self.image_view.getImageItem()
        try:
            arr = img_item.image
            if arr is None:
                return
            # arr may be transposed depending on how setImage was called (we sometimes use arr.T).
            # Try both orientations and pick a valid pixel value.
            val = None
            try:
                if 0 <= y < arr.shape[0] and 0 <= x < arr.shape[1]:
                    val = float(arr[y, x])
            except Exception:
                val = None

            if val is None:
                try:
                    if 0 <= x < arr.shape[0] and 0 <= y < arr.shape[1]:
                        val = float(arr[x, y])
                        # swap coordinates for accurate display
                        x, y = y, x
                except Exception:
                    val = None

            if val is None:
                return

            # Include Z slice if present
            z_info = ""
            if hasattr(self, "_z_values") and hasattr(self, "z_slider"):
                try:
                    idx = int(self.z_slider.value())
                    if hasattr(self, "_z_values") and 0 <= idx < len(self._z_values):
                        z_info = f" z={self._z_values[idx]}"
                except Exception:
                    z_info = ""

            self.hover_info.emit(f"x={x} y={y}{z_info} value={val:.3g}")
        except Exception:
            return

    def _on_detector_image_mouse_move(self, pos, detector_id: str, img_view: pg.ImageView):
        # Map mouse position relative to a specific detector image view
        try:
            vb = img_view.getView()
            mouse_point = vb.mapSceneToView(pos)
            x = int(round(mouse_point.x()))
            y = int(round(mouse_point.y()))
            img_item = img_view.getImageItem()
            arr = img_item.image
            if arr is None:
                return
            val = None
            try:
                if 0 <= y < arr.shape[0] and 0 <= x < arr.shape[1]:
                    val = float(arr[y, x])
            except Exception:
                val = None

            if val is None:
                try:
                    if 0 <= x < arr.shape[0] and 0 <= y < arr.shape[1]:
                        val = float(arr[x, y])
                        x, y = y, x
                except Exception:
                    val = None

            if val is None:
                return

            z_info = ""
            if hasattr(self, "_z_values") and hasattr(self, "z_slider"):
                try:
                    idx = int(self.z_slider.value())
                    if hasattr(self, "_z_values") and 0 <= idx < len(self._z_values):
                        z_info = f" z={self._z_values[idx]}"
                except Exception:
                    z_info = ""

            self.hover_info.emit(f"{detector_id}: x={x} y={y}{z_info} value={val:.3g}")
        except Exception:
            return

    def increase_upper(self):
        if self._levels[1] is None:
            item = self.image_view.getImageItem()
            try:
                mn, mx = item.getLevels()
            except Exception:
                return
        else:
            mn, mx = self._levels
        self._levels = (mn, mx * 1.1 if mx is not None else None)
        try:
            self.image_view.getImageItem().setLevels(self._levels[0], self._levels[1])
        except Exception:
            pass

    def decrease_upper(self):
        if self._levels[1] is None:
            item = self.image_view.getImageItem()
            try:
                mn, mx = item.getLevels()
            except Exception:
                return
        else:
            mn, mx = self._levels
        self._levels = (mn, mx * 0.9 if mx is not None else None)
        try:
            self.image_view.getImageItem().setLevels(self._levels[0], self._levels[1])
        except Exception:
            pass

    def decrease_lower(self):
        if self._levels[0] is None:
            item = self.image_view.getImageItem()
            try:
                mn, mx = item.getLevels()
            except Exception:
                return
        else:
            mn, mx = self._levels
        self._levels = (mn * 1.1 if mn is not None else None, mx)
        try:
            self.image_view.getImageItem().setLevels(self._levels[0], self._levels[1])
        except Exception:
            pass

    def reset_levels(self):
        self._levels = (None, None)
        try:
            self.image_view.setLevels(autoLevels=True)
        except Exception:
            try:
                self.image_view.getImageItem().setLevels(None, None)
            except Exception:
                pass