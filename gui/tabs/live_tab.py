import time
import numpy as np
from collections import deque
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from .camera_panel import CameraPanel
from .plot_panel import PlotPanel
from .detector_image_panel import DetectorImagePanel
from .detector_control_panel import DetectorControlPanel


class LiveTab(QtWidgets.QWidget):
    hover_info = QtCore.pyqtSignal(str)
    view_changed = QtCore.pyqtSignal(str)
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

        # per-detector image views for heatmaps (created in _build_ui)
        self._detector_image_views: dict[str, pg.ImageView] = {}
        self.detector_images_container = None
        self.detector_images_layout = None

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
        # Load data button
        self.load_btn = QtWidgets.QPushButton("Load Data")
        toggle_layout.addWidget(self.load_btn)
        layout.addLayout(toggle_layout)

        self.camera_btn.clicked.connect(self._set_camera_view)
        self.detector_btn.clicked.connect(self._set_detector_view)
        self.load_btn.clicked.connect(self._on_load_data)

        # Camera panel (ImageView)
        self.camera_panel = CameraPanel()
        self.image_view = self.camera_panel.image_view
        # connect hover events
        try:
            self.image_view.scene.sigMouseMoved.connect(self._on_image_mouse_move)
        except Exception:
            try:
                self.image_view.getView().scene.sigMouseMoved.connect(self._on_image_mouse_move)
            except Exception:
                pass
        layout.addWidget(self.camera_panel, 3)

        # Detector images panel (per-detector heatmaps)
        self.detector_image_panel = DetectorImagePanel()
        # expose inner container and layout for backward compatibility
        self.detector_images_container = self.detector_image_panel.container
        self.detector_images_layout = self.detector_image_panel.layout
        layout.addWidget(self.detector_image_panel, 3)
        self.detector_image_panel.hide()

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

        # Plot panel (1D detector plot)
        self.plot_panel = PlotPanel()
        self.plot_widget = self.plot_panel.plot_widget
        self.plot_curve = self.plot_widget.plot([], [])
        self.plot_widget.setLabel("left", "Detector", units="a.u.")
        self.plot_widget.setLabel("bottom", "Time / Coord", units="a.u.")
        # X-axis selector for multi-axis plots
        xsel_layout = QtWidgets.QHBoxLayout()
        xsel_layout.addWidget(QtWidgets.QLabel("X Axis:"))
        self.xaxis_combo = QtWidgets.QComboBox()
        self.xaxis_combo.addItem("Index")
        self.xaxis_combo.currentTextChanged.connect(self._update_plot)
        xsel_layout.addWidget(self.xaxis_combo)
        layout.addLayout(xsel_layout)

        layout.addWidget(self.plot_panel, 1)

        # Per-detector controls (visibility + streaming)
        self.detector_control_panel = DetectorControlPanel()
        # expose group and layout for backward compatibility
        self.detector_group = self.detector_control_panel.group
        self.detector_controls_layout = self.detector_control_panel.vlayout
        layout.addWidget(self.detector_control_panel, 0)

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
            # notify main window so docks can be toggled
            try:
                self.view_changed.emit(self.view_mode)
            except Exception:
                pass
        except Exception:
            pass

    def _set_detector_view(self):
        self.view_mode = "detector"
        self.camera_btn.setChecked(False)
        self.detector_btn.setChecked(True)
        try:
            self.image_view.hide()
            self.detector_images_container.show()
            try:
                self.view_changed.emit(self.view_mode)
            except Exception:
                pass
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

        # refresh available x-axis options
        self._refresh_xaxis_options()

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
        # refresh x-axis choices
        self._refresh_xaxis_options()

    def _refresh_xaxis_options(self):
        # Schedule UI updates on the event loop to avoid nested selection-change modifications
        def _do():
            keys = set()
            for det_list in self.multi_coords.values():
                if not det_list:
                    continue
                sample_state, _ = det_list[0]
                try:
                    for k in sample_state.keys():
                        keys.add(k)
                except Exception:
                    continue
            current = self.xaxis_combo.currentText() if hasattr(self, 'xaxis_combo') else 'Index'
            items = ['Index'] + sorted(keys)
            try:
                self.xaxis_combo.blockSignals(True)
                self.xaxis_combo.clear()
                for it in items:
                    self.xaxis_combo.addItem(it)
                idx = self.xaxis_combo.findText(current)
                if idx >= 0:
                    self.xaxis_combo.setCurrentIndex(idx)
            finally:
                try:
                    self.xaxis_combo.blockSignals(False)
                except Exception:
                    pass

        QtCore.QTimer.singleShot(0, _do)

    # -----------------------------
    # Load saved data
    # -----------------------------
    def _on_load_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Data", "", "Data files (*.txt *.npy)")
        if not path:
            return
        try:
            if path.endswith('.npy'):
                arr = np.load(path)
                # expect columns: timestamp, value, x, y, z
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                # ensure at least 2 columns
                times = arr[:, 0]
                vals = arr[:, 1]
                xs = arr[:, 2] if arr.shape[1] > 2 else None
                ys = arr[:, 3] if arr.shape[1] > 3 else None
            else:
                # parse ascii CSV with header
                with open(path, 'r') as f:
                    hdr = f.readline().strip().split(',')
                    cols = {c: i for i, c in enumerate(hdr)}
                    data = []
                    for ln in f:
                        parts = ln.strip().split(',')
                        if len(parts) < 2:
                            continue
                        data.append(parts)
                import numpy as _np
                arr = _np.array(data)
                # try to map columns
                try:
                    times = arr[:, cols.get('timestamp', 0)].astype(float)
                except Exception:
                    times = arr[:, 0].astype(float)
                try:
                    vals = arr[:, cols.get('value', 1)].astype(float)
                except Exception:
                    vals = arr[:, 1].astype(float)
                xs = None
                ys = None
                if 'x' in cols or 'X' in cols:
                    k = 'X' if 'X' in cols else 'x'
                    xs = arr[:, cols[k]].astype(float)
                if 'y' in cols or 'Y' in cols:
                    k = 'Y' if 'Y' in cols else 'y'
                    ys = arr[:, cols[k]].astype(float)

            # display time-series in plot
            try:
                self.plot_widget.clear()
                self.plot_widget.plot(times, vals, pen=pg.mkPen('y', width=2))
            except Exception:
                pass

            # create heatmap if positions are available
            if xs is not None and ys is not None:
                try:
                    # bin positions to a grid and weight by vals
                    nbins = 128
                    xi = np.linspace(np.nanmin(xs), np.nanmax(xs), nbins)
                    yi = np.linspace(np.nanmin(ys), np.nanmax(ys), nbins)
                    H, xedges, yedges = np.histogram2d(xs.astype(float), ys.astype(float), bins=[xi, yi], weights=vals.astype(float))
                    img = np.nan_to_num(H.T)
                    self.image_view.setImage(img, autoLevels=True)
                except Exception:
                    pass
        except Exception:
            QtWidgets.QMessageBox.warning(self, "Load Data", f"Failed to load: {path}")

    def set_xaxis(self, name: str):
        if not hasattr(self, 'xaxis_combo'):
            return
        if name is None:
            return
        idx = self.xaxis_combo.findText(name)
        if idx < 0:
            # add it
            self.xaxis_combo.addItem(name)
            idx = self.xaxis_combo.findText(name)
        if idx >= 0:
            self.xaxis_combo.setCurrentIndex(idx)

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

            # update x-axis options when new data arrives
            self._refresh_xaxis_options()

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

        # If multi-axis numeric plotting is desired, plot detector intensities
        # vs the selected X axis (Index or axis name)
        try:
            xaxis = self.xaxis_combo.currentText() if hasattr(self, 'xaxis_combo') else 'Index'
            if self.multi_coords and xaxis:
                # clear main plot
                self.plot_widget.clear()
                # for each detector, build x and y arrays
                for det_id, det_list in self.multi_coords.items():
                    if not det_list:
                        continue
                    xs = []
                    ys = []
                    for idx, (s, v) in enumerate(det_list):
                        # If axis is Index, use sample index; otherwise try to read from state dict
                        if xaxis == 'Index':
                            xs.append(idx)
                        else:
                            try:
                                # handle Channel or other object values by converting to numeric if possible
                                val = s.get(xaxis)
                                if val is None:
                                    # fallback to index
                                    xs.append(idx)
                                else:
                                    try:
                                        xs.append(float(val))
                                    except Exception:
                                        # non-numeric: use index
                                        xs.append(idx)
                            except Exception:
                                xs.append(idx)
                        ys.append(v)
                    # plot with a labeled curve
                    pen = pg.mkPen(width=2)
                    self.plot_widget.plot(xs, ys, pen=pen, name=det_id)
                # update axis label
                try:
                    self.plot_widget.setLabel('bottom', xaxis)
                except Exception:
                    pass
        except Exception:
            pass

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