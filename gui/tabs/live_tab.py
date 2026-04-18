import time
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import deque
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from .camera_panel import CameraPanel
from .plot_panel import PlotPanel
from .detector_image_panel import DetectorImagePanel
from .detector_control_panel import DetectorControlPanel

try:
    from utils import multichannel_h5
except ImportError:
    multichannel_h5 = None  # type: ignore


class LiveTab(QtWidgets.QWidget):
    hover_info = QtCore.pyqtSignal(str)
    view_changed = QtCore.pyqtSignal(str)
    # emitted when user toggles streaming for a detector: (detector_id, enabled)
    stream_toggled = QtCore.pyqtSignal(str, bool)
    # emitted to post a message to the main window status bar: (message, timeout_ms)
    status_message = QtCore.pyqtSignal(str, int)

    def __init__(self, parent=None):
        super().__init__(parent)

        # per-detector moving window buffers
        self._window_size = 200
        self._detector_buffers: dict[str, deque] = {}
        self._detector_times: dict[str, deque] = {}
        self._detector_curves: dict[str, pg.PlotDataItem] = {}

        # Tracks whether the shared plot is currently being used for strip-chart
        # or for multi-axis numeric plotting.
        self._plot_mode: str = "strip"  # "strip" | "multiaxis"
        self._t0 = time.time()
        
        # Selection filter (None => show all)
        self._selected_detectors_filter: set[str] | None = None
        
        # Keep references to per-detector controls so other UI parts can drive them.
        self._detector_show_cbs: dict[str, QtWidgets.QCheckBox] = {}
        self._detector_stream_cbs: dict[str, QtWidgets.QCheckBox] = {}
        self._detector_control_rows: dict[str, QtWidgets.QWidget] = {}

        # per-detector image views for heatmaps (created in _build_ui)
        self._detector_image_views: dict[str, pg.ImageView] = {}
        # cached last-rendered 2D heatmap per detector (for overlay)
        self._detector_last_images: dict[str, np.ndarray] = {}
        self.detector_images_container = None
        self.detector_images_layout = None

        # per-detector multi-axis data: det_id -> list[(state, value)]
        self.multi_coords: dict[str, list[tuple[dict, float]]] = {}

        # Thread-safe queue: worker thread appends here, GUI timer drains it.
        # deque.append / popleft are GIL-atomic — no lock needed.
        self._multiaxis_queue: deque = deque()

        # tracked Z values for z-slider (avoids O(n²) re-scan on every sample)
        self._z_values_set: set = set()
        self._z_values: list = []

        # multi-axis rendering throttle
        self._multi_dirty = False
        self._last_multi_render = 0.0

        # image display level state
        self._levels = (None, None)  # (min, max) or (None,None) for auto

        # view mode: "camera" or "detector"
        self.view_mode = "camera"

        # Remember last-used layout-preference for the Load dialog
        # True  → keep current layout, False → use data layout
        self._load_keep_layout: bool = True

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

        # "Load Data" tool-button with a drop-down for the layout preference.
        # Using QToolButton lets us attach a small menu arrow without a second widget.
        self.load_btn = QtWidgets.QToolButton()
        self.load_btn.setText("Load Data")
        self.load_btn.setToolButtonStyle(
            QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly
        )
        self.load_btn.setPopupMode(
            QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup
        )
        load_menu = QtWidgets.QMenu(self.load_btn)
        self._load_data_action = load_menu.addAction("Load Data…")
        self._load_data_action.setShortcut("Ctrl+O")
        load_menu.addSeparator()
        self.keep_layout_action = load_menu.addAction("Keep Current Layout When Loading")
        self.keep_layout_action.setCheckable(True)
        self.keep_layout_action.setChecked(self._load_keep_layout)
        self.keep_layout_action.setToolTip(
            "When checked, the current view mode, X axis, Z slice, draw-lines\n"
            "and window size are preserved after loading data.\n"
            "When unchecked, the layout is inferred from the loaded data."
        )
        self.load_btn.setMenu(load_menu)
        toggle_layout.addWidget(self.load_btn)

        # Export heatmap button
        self.export_heatmap_btn = QtWidgets.QPushButton("Export Heatmap")
        toggle_layout.addWidget(self.export_heatmap_btn)

        # Export plot data button
        self.export_plot_btn = QtWidgets.QPushButton("Export Plot Data")
        toggle_layout.addWidget(self.export_plot_btn)
        layout.addLayout(toggle_layout)

        self.camera_btn.clicked.connect(self._set_camera_view)
        self.detector_btn.clicked.connect(self._set_detector_view)
        self.load_btn.clicked.connect(self._on_load_data)
        self._load_data_action.triggered.connect(self._on_load_data)
        self.keep_layout_action.toggled.connect(self._on_keep_layout_toggled)
        self.export_heatmap_btn.clicked.connect(self._on_export_heatmap)
        self.export_plot_btn.clicked.connect(self._on_export_plot_data)

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

        # when user toggles overlay, refresh composite from cached images
        try:
            self.detector_image_panel.overlay_toggled.connect(self._update_false_color_overlay)
        except Exception:
            pass

        # when user changes overlay mapping/settings, refresh composite
        try:
            self.detector_image_panel.overlay_settings_changed.connect(self._update_false_color_overlay)
        except Exception:
            pass

        # when user changes overlay mapping (R/G/B selectors), refresh overlay
        try:
            self.detector_image_panel.overlay_settings_changed.connect(self._update_false_color_overlay)
        except Exception:
            pass

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
        self.plot_widget.addLegend(offset=(10, 10))
        # Hide the legend initially; it is visually noisy when no curves exist.
        # We'll show it lazily once a named curve is added.
        try:
            self._set_legend_visible(False)
        except Exception:
            pass
        # Disable pyqtgraph's built-in export dialog/menu so all exports go
        # through our Save dialog flow with explicit folder+filename choice.
        try:
            self.plot_widget.setMenuEnabled(False)
        except Exception:
            pass
        # Right-click menu on plot area for quick data export.
        try:
            self.plot_widget.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
            self.plot_widget.customContextMenuRequested.connect(self._show_plot_context_menu)
        except Exception:
            pass
        # Color palette for detector curves (cycles for >8 detectors)
        self._det_colors = [
            (86, 180, 233),   # sky blue
            (230, 159, 0),    # orange
            (0, 158, 115),    # green
            (240, 228, 66),   # yellow
            (213, 94, 0),     # vermillion
            (204, 121, 167),  # pink/purple
            (0, 114, 178),    # blue
            (255, 255, 255),  # white
        ]
        self._det_color_idx = 0
        # X-axis selector for multi-axis plots + line toggle
        xsel_layout = QtWidgets.QHBoxLayout()
        xsel_layout.addWidget(QtWidgets.QLabel("X Axis:"))
        self.xaxis_combo = QtWidgets.QComboBox()
        self.xaxis_combo.addItem("Index")
        self.xaxis_combo.currentTextChanged.connect(self._update_plot)
        xsel_layout.addWidget(self.xaxis_combo)
        xsel_layout.addSpacing(12)
        self.draw_lines_cb = QtWidgets.QCheckBox("Lines")
        self.draw_lines_cb.setChecked(True)
        self.draw_lines_cb.toggled.connect(self._on_draw_lines_toggled)
        xsel_layout.addWidget(self.draw_lines_cb)
        xsel_layout.addStretch(1)
        layout.addLayout(xsel_layout)
        self._draw_lines: bool = True

        # Preferred x-axis to select after the next refresh of available axes.
        # This is used to apply the Multi-Axis tab's "Default X Axis" to the
        # Live plot when a run starts.
        self._preferred_plot_xaxis: str | None = None

        layout.addWidget(self.plot_panel, 1)

        # Per-detector controls (visibility + streaming)
        self.detector_control_panel = DetectorControlPanel()
        # expose group and layout for backward compatibility
        self.detector_group = self.detector_control_panel.group
        self.detector_controls_layout = self.detector_control_panel.vlayout
        layout.addWidget(self.detector_control_panel, 0)

        # Put overlay options into the Detectors panel (so they are available
        # even when the Detector Images dock is hidden).
        try:
            controls = getattr(self.detector_image_panel, "controls_widget", None)
            if controls is not None:
                controls.setParent(None)
                self.detector_controls_layout.insertWidget(0, controls)
        except Exception:
            pass

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

    def _set_legend_visible(self, visible: bool) -> None:
        """Show/hide the plot legend.

        The legend is created up-front, but we keep it hidden until at least
        one named curve exists.
        """
        plot_item = None
        try:
            plot_item = self.plot_widget.getPlotItem()
        except Exception:
            try:
                plot_item = getattr(self.plot_widget, "plotItem", None)
            except Exception:
                plot_item = None

        if plot_item is None:
            return

        leg = None
        try:
            leg = getattr(plot_item, "legend", None)
        except Exception:
            leg = None

        if visible:
            if leg is None:
                try:
                    plot_item.addLegend(offset=(10, 10))
                except Exception:
                    try:
                        self.plot_widget.addLegend(offset=(10, 10))
                    except Exception:
                        pass
                try:
                    leg = getattr(plot_item, "legend", None)
                except Exception:
                    leg = None
            try:
                if leg is not None:
                    leg.show()
            except Exception:
                pass
        else:
            try:
                if leg is not None:
                    leg.hide()
            except Exception:
                pass

    def _clear_plot_and_legend(self) -> None:
        """Clear plot items and reset legend entries without duplicating legends."""
        # Prefer operating on the PlotItem directly; PlotWidget forwards attributes
        # but legend handling differs across pyqtgraph versions.
        plot_item = None
        try:
            plot_item = self.plot_widget.getPlotItem()
        except Exception:
            try:
                plot_item = getattr(self.plot_widget, "plotItem", None)
            except Exception:
                plot_item = None

        # Clear plot items
        try:
            self.plot_widget.clear()
        except Exception:
            try:
                if plot_item is not None:
                    plot_item.clear()
            except Exception:
                pass

        # Robust legend reset: pyqtgraph legends can accumulate if a previous one
        # isn't fully removed (they are GraphicsItems attached to the ViewBox).
        # Remove *all* LegendItem instances we can find, then recreate exactly one.
        try:
            if plot_item is not None:
                vb = getattr(plot_item, "vb", None)

                legends: list[object] = []
                try:
                    if vb is not None:
                        # addedItems is where ViewBox tracks items explicitly added
                        for it in list(getattr(vb, "addedItems", []) or []):
                            try:
                                if isinstance(it, pg.LegendItem):
                                    legends.append(it)
                            except Exception:
                                pass
                        # childItems can include anchored widgets/items
                        for it in list(getattr(vb, "childItems", lambda: [])() or []):
                            try:
                                if isinstance(it, pg.LegendItem):
                                    legends.append(it)
                            except Exception:
                                pass
                except Exception:
                    pass

                # Include the PlotItem.legend reference (if any)
                try:
                    leg0 = getattr(plot_item, "legend", None)
                    if leg0 is not None:
                        legends.append(leg0)
                except Exception:
                    pass

                # De-duplicate by object id
                uniq: dict[int, object] = {}
                for legend_item in legends:
                    try:
                        uniq[id(legend_item)] = legend_item
                    except Exception:
                        pass

                for leg in uniq.values():
                    try:
                        if vb is not None and hasattr(vb, "removeItem"):
                            vb.removeItem(leg)
                    except Exception:
                        pass
                    try:
                        sc = leg.scene()
                        if sc is not None:
                            sc.removeItem(leg)
                    except Exception:
                        pass
                    try:
                        # fully detach in case it's still parented somewhere
                        if hasattr(leg, "setParentItem"):
                            leg.setParentItem(None)
                    except Exception:
                        pass

                try:
                    plot_item.legend = None
                except Exception:
                    pass

                try:
                    plot_item.addLegend(offset=(10, 10))
                except Exception:
                    self.plot_widget.addLegend(offset=(10, 10))
            else:
                self.plot_widget.addLegend(offset=(10, 10))
        except Exception:
            pass

        # After a clear/reset there are no curves; keep legend hidden.
        try:
            self._set_legend_visible(False)
        except Exception:
            pass

    # -----------------------------
    # Export heatmap
    # -----------------------------
    def _suggest_heatmap_export_path(self) -> Path:
        """Return default export path: data/heatmap_<timestamp>.tif."""
        # Prefer repository root (two levels above this file: gui/tabs/ -> repo)
        try:
            repo_root = Path(__file__).resolve().parents[2]
        except Exception:
            repo_root = Path.cwd()
        out_dir = repo_root / "data"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return out_dir / f"heatmap_{ts}.tif"

    def _get_current_heatmap_array(self) -> np.ndarray | None:
        """Choose a heatmap array to export.

        Priority:
        - If overlay view is enabled and has an image: export overlay RGB.
        - Else export the first visible detector's last 2D heatmap.
        - Else fall back to the main image_view image (if any).
        """
        # 1) overlay RGB
        try:
            if getattr(self, 'detector_image_panel', None) and self.detector_image_panel.overlay_enabled():
                item = self.detector_image_panel.overlay_view.getImageItem()
                arr = getattr(item, "image", None)
                if arr is not None:
                    a = np.asarray(arr)
                    if a.ndim == 3 and a.shape[2] in (3, 4):
                        return a[:, :, :3]
        except Exception:
            pass

        # 2) cached last 2D heatmaps
        try:
            for det_id in sorted(self._detector_last_images.keys()):
                if not self._is_detector_visible(det_id):
                    continue
                a = np.asarray(self._detector_last_images.get(det_id))
                if a.ndim == 2:
                    return a
        except Exception:
            pass

        # 3) fall back to main image view
        try:
            item = self.image_view.getImageItem()
            arr = getattr(item, "image", None)
            if arr is None:
                return None
            a = np.asarray(arr)
            if a.ndim in (2, 3):
                return a
        except Exception:
            return None

        return None

    def _on_export_heatmap(self):
        data = self._get_current_heatmap_array()
        if data is None:
            QtWidgets.QMessageBox.information(self, "Export Heatmap", "No heatmap image is available to export yet.")
            return

        default_path = self._suggest_heatmap_export_path()
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Heatmap",
            str(default_path),
            "TIFF (*.tif *.tiff);;CSV (*.csv);;HDF5 (*.h5 *.hdf5);;Image (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return

        out_path = Path(path)

        # Infer extension from selected filter when user did not type one.
        suffix = out_path.suffix.lower()
        if not suffix:
            sf = (selected_filter or "").lower()
            if "csv" in sf:
                out_path = out_path.with_suffix(".csv")
            elif "hdf5" in sf:
                out_path = out_path.with_suffix(".h5")
            elif "image" in sf:
                out_path = out_path.with_suffix(".png")
            else:
                out_path = out_path.with_suffix(".tif")
        suffix = out_path.suffix.lower()

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        arr = np.asarray(data)

        # CSV export: preserve numeric data in tabular form.
        if suffix == ".csv":
            try:
                if arr.ndim == 2:
                    np.savetxt(str(out_path), arr, delimiter=",")
                elif arr.ndim == 3:
                    h, w, c = arr.shape
                    flat = arr.reshape(h * w, c)
                    header = ",".join([f"ch{i}" for i in range(c)])
                    np.savetxt(str(out_path), flat, delimiter=",", header=header, comments="")
                else:
                    np.savetxt(str(out_path), arr.reshape(-1), delimiter=",")
                return
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export Heatmap", f"Failed to write CSV.\n\nError: {e}")
                return

        # HDF5 export: keep full array fidelity.
        if suffix in (".h5", ".hdf5"):
            try:
                import h5py

                with h5py.File(str(out_path), "w") as f:
                    f.create_dataset("heatmap", data=arr)
                    try:
                        f.attrs["shape"] = list(arr.shape)
                        f.attrs["dtype"] = str(arr.dtype)
                    except Exception:
                        pass
                return
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export Heatmap", f"Failed to write HDF5.\n\nError: {e}")
                return

        # TIFF export: prefer tifffile (supports float arrays + RGB).
        if suffix in (".tif", ".tiff"):
            try:
                import tifffile  # type: ignore

                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32, copy=False)
                tifffile.imwrite(str(out_path), arr)
                return
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Export Heatmap",
                    "Failed to write TIFF.\n\n"
                    "Tip: install 'tifffile' (pip install tifffile) for robust TIFF export.\n\n"
                    f"Error: {e}",
                )
                return

        # Other image exports (png/jpg/bmp) via QImage.
        try:
            if arr.ndim == 2:
                # Normalize scalar heatmap to 8-bit grayscale.
                arrf = np.asarray(arr, dtype=np.float32)
                finite = np.isfinite(arrf)
                if np.any(finite):
                    vmin = float(np.nanmin(arrf[finite]))
                    vmax = float(np.nanmax(arrf[finite]))
                    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                        arr8 = ((arrf - vmin) / (vmax - vmin) * 255.0).clip(0, 255).astype(np.uint8)
                    else:
                        arr8 = np.zeros(arrf.shape, dtype=np.uint8)
                else:
                    arr8 = np.zeros(arrf.shape, dtype=np.uint8)
                h, w = arr8.shape
                qimg = QtGui.QImage(arr8.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8).copy()
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                rgb = arr[:, :, :3]
                if rgb.dtype != np.uint8:
                    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                h, w, _ = rgb.shape
                qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
            else:
                raise ValueError("Unsupported array shape for image export")

            if not qimg.save(str(out_path)):
                raise RuntimeError("QImage save returned false")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                "Export Heatmap",
                "Failed to write image file.\n\n"
                f"Error: {e}",
            )

    def _suggest_plot_export_path(self) -> Path:
        """Return default plot-data export path: data/plotdata_<timestamp>.csv."""
        try:
            repo_root = Path(__file__).resolve().parents[2]
        except Exception:
            repo_root = Path.cwd()
        out_dir = repo_root / "data"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return out_dir / f"plotdata_{ts}.csv"

    def _collect_plot_series(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """Collect visible plot curves as (name, x, y)."""
        out: list[tuple[str, np.ndarray, np.ndarray]] = []
        try:
            items = self.plot_widget.listDataItems()
        except Exception:
            items = []

        for i, item in enumerate(items):
            try:
                if hasattr(item, "isVisible") and (not item.isVisible()):
                    continue
            except Exception:
                pass

            x_data = getattr(item, "xData", None)
            y_data = getattr(item, "yData", None)
            if x_data is None or y_data is None:
                try:
                    x_data, y_data = item.getData()
                except Exception:
                    continue
            if x_data is None or y_data is None:
                continue

            x = np.asarray(x_data)
            y = np.asarray(y_data)
            if x.size == 0 or y.size == 0:
                continue

            n = min(int(x.size), int(y.size))
            x = np.asarray(x[:n], dtype=float)
            y = np.asarray(y[:n], dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]
            if x.size == 0:
                continue

            name = "curve"
            try:
                nm = item.name()
                if nm:
                    name = str(nm)
            except Exception:
                pass
            if not name:
                name = f"curve_{i + 1}"
            out.append((name, x, y))

        return out

    def _on_export_plot_data(self):
        series = self._collect_plot_series()
        if not series:
            QtWidgets.QMessageBox.information(self, "Export Plot Data", "No plot data is available to export.")
            return

        default_path = self._suggest_plot_export_path()
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Plot Data",
            str(default_path),
            "CSV (*.csv);;HDF5 (*.h5 *.hdf5)",
        )
        if not path:
            return

        out_path = Path(path)
        if not out_path.suffix:
            sf = (selected_filter or "").lower()
            if "hdf5" in sf:
                out_path = out_path.with_suffix(".h5")
            else:
                out_path = out_path.with_suffix(".csv")

        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        suffix = out_path.suffix.lower()
        if suffix == ".csv":
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("curve,x,y\n")
                    for name, x, y in series:
                        for xv, yv in zip(x, y):
                            f.write(f"{name},{float(xv):.12g},{float(yv):.12g}\n")
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export Plot Data", f"Failed to write CSV.\n\nError: {e}")
                return
        elif suffix in (".h5", ".hdf5"):
            try:
                import h5py

                with h5py.File(str(out_path), "w") as f:
                    grp = f.create_group("curves")
                    for idx, (name, x, y) in enumerate(series, start=1):
                        key = f"curve_{idx}"
                        cgrp = grp.create_group(key)
                        cgrp.attrs["name"] = str(name)
                        cgrp.create_dataset("x", data=np.asarray(x, dtype=float))
                        cgrp.create_dataset("y", data=np.asarray(y, dtype=float))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Export Plot Data", f"Failed to write HDF5.\n\nError: {e}")
                return
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Export Plot Data",
                "Unsupported file extension. Use .csv, .h5, or .hdf5.",
            )

    def _show_plot_context_menu(self, pos) -> None:
        """Show right-click context menu on the plot with export actions."""
        try:
            menu = QtWidgets.QMenu(self)
            export_action = menu.addAction("Export Plot Data...")
            chosen = menu.exec(self.plot_widget.mapToGlobal(pos))
            if chosen == export_action:
                self._on_export_plot_data()
        except Exception:
            pass

    # -----------------------------
    # view mode
    # -----------------------------
    def _apply_view_mode(self, mode: str, *, emit_signal: bool = True) -> None:
        """Update view_mode state and button appearance.

        Parameters
        ----------
        mode : "camera" | "detector"
        emit_signal : bool
            When True (the default), ``view_changed`` is emitted so that the
            main window can show/hide the appropriate dock widgets.
            Pass False when calling from data-loading paths where dock layout
            must **not** be disturbed (the caller is responsible for any dock
            management after data is fully loaded).
        """
        self.view_mode = mode
        is_det = (mode == "detector")
        self.camera_btn.setChecked(not is_det)
        self.detector_btn.setChecked(is_det)

        # Only touch widget visibility when NOT inside a dock; when docked the
        # main window handles show/hide via the view_changed signal.
        try:
            det_parent = self.detector_image_panel.parent()
            cam_parent = self.camera_panel.parent()
            in_dock = isinstance(det_parent, QtWidgets.QDockWidget) or isinstance(cam_parent, QtWidgets.QDockWidget)
        except Exception:
            in_dock = False

        if not in_dock:
            try:
                if is_det:
                    self.camera_panel.hide()
                    self.detector_image_panel.show()
                else:
                    self.camera_panel.show()
                    self.detector_image_panel.hide()
            except Exception:
                pass

        if emit_signal:
            try:
                self.view_changed.emit(self.view_mode)
            except Exception:
                pass

    def _set_camera_view(self):
        self._apply_view_mode("camera", emit_signal=True)

    def _set_detector_view(self):
        self._apply_view_mode("detector", emit_signal=True)

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

        # Assign next color from the palette (wrapping)
        color = self._det_colors[self._det_color_idx % len(self._det_colors)]
        self._det_color_idx += 1

        pen = pg.mkPen(color=color, width=2)
        curve = self.plot_widget.plot([], [], pen=pen, name=detector_id)
        self._detector_curves[detector_id] = curve

        # Create a small control row: label (colored), visibility checkbox, stream checkbox
        row = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row)
        row_layout.setContentsMargins(2, 2, 2, 2)
        lbl = QtWidgets.QLabel(detector_id)
        r, g, b = color
        lbl.setStyleSheet(f"color: rgb({r},{g},{b}); font-weight: bold;")
        vis_cb = QtWidgets.QCheckBox("Show")
        vis_cb.setChecked(True)
        stream_cb = QtWidgets.QCheckBox("Stream")
        stream_cb.setChecked(False)
        row_layout.addWidget(lbl)
        row_layout.addWidget(vis_cb)
        row_layout.addWidget(stream_cb)
        row_layout.addStretch(1)
        self.detector_controls_layout.addWidget(row)
        
        self._detector_show_cbs[detector_id] = vis_cb
        self._detector_stream_cbs[detector_id] = stream_cb
        self._detector_control_rows[detector_id] = row

        def _on_vis(chk):
            try:
                self._detector_curves[detector_id].setVisible(bool(chk))
            except Exception:
                pass
            try:
                self.detector_image_panel._detector_widgets[detector_id].setVisible(bool(chk))
            except Exception:
                pass
            # Ensure the multi-axis numeric plot + overlay reflect the new visibility immediately.
            try:
                self._multi_dirty = True
            except Exception:
                pass
            try:
                if getattr(self, 'detector_image_panel', None) and self.detector_image_panel.overlay_enabled():
                    self._update_false_color_overlay()
            except Exception:
                pass

        def _on_stream(chk):
            self.stream_toggled.emit(detector_id, bool(chk))

        vis_cb.toggled.connect(_on_vis)
        stream_cb.toggled.connect(_on_stream)

        # create a small image view for this detector (for multi-axis heatmaps)
        img_view = pg.ImageView()
        try:
            img_view.getView().setMinimumWidth(180)
            img_view.getView().setMinimumHeight(160)
        except Exception:
            pass
        self._detector_image_views[detector_id] = img_view

        # Let the panel create the container + colormap selector.
        try:
            self.detector_image_panel.add_detector_view(detector_id, img_view)
        except Exception:
            # fallback: keep old behavior if something goes wrong
            try:
                self.detector_images_layout.addWidget(img_view)
            except Exception:
                pass

        # ensure data storage
        self.multi_coords.setdefault(detector_id, [])
        
        # Apply global selection filter after the detector exists.
        try:
            self._apply_selection_filter_to_detector(detector_id)
        except Exception:
            pass

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

    def set_stream_enabled(self, detector_id: str, enabled: bool) -> None:
        """Programmatically set the Stream checkbox for a detector."""
        cb = self._detector_stream_cbs.get(detector_id)
        if cb is None:
            return
        try:
            cb.blockSignals(True)
            cb.setChecked(bool(enabled))
        finally:
            try:
                cb.blockSignals(False)
            except Exception:
                pass

    def reset_multiaxis(self):
        for v in self.multi_coords.values():
            v.clear()
        self.multi_coords.clear()
        try:
            self._multiaxis_queue.clear()
        except Exception:
            pass
        self._multi_dirty = False
        self._z_values_set.clear()
        self._z_values = []
        self.z_slider.setMaximum(0)
        self.z_slider.setValue(0)
        try:
            self._detector_last_images.clear()
        except Exception:
            pass

        # Reset plot mode back to strip-chart.
        try:
            self._plot_mode = "strip"
        except Exception:
            pass
        # Clear any run-scoped x-axis preference so strip-chart mode is unaffected.
        self._preferred_plot_xaxis = None
        # refresh x-axis choices
        self._refresh_xaxis_options()

    def prepare_strip_chart_plot(self):
        """Reset shared plot state so strip-chart traces render after multi-axis runs."""
        # Drop multi-axis data/queue so _update_plot no longer overwrites with scan curves.
        self.reset_multiaxis()

        # Multi-axis rendering clears the plot widget, so recreate detector curves.
        try:
            self._clear_plot_and_legend()
            self.plot_widget.setLabel("bottom", "Time [s]")
            self.plot_widget.setLabel("left", "Signal")
        except Exception:
            pass

        for det_id in list(self._detector_curves.keys()):
            try:
                old_curve = self._detector_curves.get(det_id)
                pen = old_curve.opts.get("pen") if old_curve is not None else pg.mkPen(color=(255, 255, 255), width=2)
                self._detector_curves[det_id] = self.plot_widget.plot([], [], pen=pen, name=det_id)
            except Exception:
                pass

    def _map_and_filter_detector_id(self, detector_id: str) -> str | None:
        """Map and filter detector ids according to the global selection.

        This function is safe to call from worker threads because it only
        touches pure-Python state (the selection set) and does not access Qt.

        Returns:
            - A detector id to use (possibly remapped)
            - None if this detector should be ignored for display
        """
        try:
            det_id = str(detector_id)
        except Exception:
            det_id = detector_id

        wanted = self._selected_detectors_filter
        if wanted is None:
            return det_id

        if det_id in wanted:
            return det_id

        # Some upstream paths use a generic id. If exactly one detector is
        # selected, map it so filtering works.
        try:
            if det_id == "detector" and len(wanted) == 1:
                return next(iter(wanted))
        except Exception:
            pass

        return None
        
    def set_selected_detectors(self, detector_ids: list[str] | None):
        """Apply a global detector selection filter.

        - Empty/None means "show all".
        - Non-empty list means show ONLY those detectors.
        """
        wanted = set(detector_ids or [])
        self._selected_detectors_filter = wanted if wanted else None

        # Ensure controls exist for wanted detectors even before samples arrive.
        for det_id in sorted(wanted):
            try:
                if det_id not in self._detector_buffers:
                    self.register_detector(det_id)
            except Exception:
                pass

        # Apply to existing detectors.
        for det_id in list(self._detector_buffers.keys()):
            try:
                self._apply_selection_filter_to_detector(det_id)
            except Exception:
                pass

        # Recompute x-axis options and Z slider range to reflect what is visible.
        try:
            self._refresh_xaxis_options()
        except Exception:
            pass
        try:
            self._recompute_z_values_from_visible_detectors()
        except Exception:
            pass

        # Overlay composition should reflect current visibility.
        try:
            self._update_false_color_overlay()
        except Exception:
            pass

    def _recompute_z_values_from_visible_detectors(self) -> None:
        """Recompute the Z slice list using only currently visible detectors."""
        if not hasattr(self, "z_slider"):
            return
        z_set: set = set()
        try:
            for det_id, det_list in self.multi_coords.items():
                if not self._is_detector_visible(det_id):
                    continue
                for state, _val in det_list:
                    try:
                        if isinstance(state, dict) and "Z" in state:
                            z_set.add(state["Z"])
                    except Exception:
                        continue
        except Exception:
            z_set = set()

        self._z_values_set = set(z_set)
        self._z_values = sorted(self._z_values_set)
        try:
            self.z_slider.setMaximum(max(0, len(self._z_values) - 1))
            if self.z_slider.value() > self.z_slider.maximum():
                self.z_slider.setValue(self.z_slider.maximum())
        except Exception:
            pass

    def _apply_selection_filter_to_detector(self, detector_id: str) -> None:
        cb = self._detector_show_cbs.get(detector_id)
        if cb is None:
            return

        if self._selected_detectors_filter is None:
            # Show all when nothing is selected.
            if not cb.isChecked():
                cb.setChecked(True)
            return

        should_show = detector_id in self._selected_detectors_filter
        if cb.isChecked() != should_show:
            cb.setChecked(bool(should_show))

    def _is_detector_visible(self, detector_id: str) -> bool:
        if self._selected_detectors_filter is not None and detector_id not in self._selected_detectors_filter:
            return False
        cb = self._detector_show_cbs.get(detector_id)
        if cb is None:
            return True
        try:
            return bool(cb.isChecked())
        except Exception:
            return True

    def _refresh_xaxis_options(self):
        # Schedule UI updates on the event loop to avoid nested selection-change modifications
        def _do():
            keys = set()
            for det_id, det_list in self.multi_coords.items():
                if not det_list:
                    continue
                # Only include axes coming from visible detectors.
                try:
                    if not self._is_detector_visible(det_id):
                        continue
                except Exception:
                    pass

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

                # _preferred_plot_xaxis stays set for the entire run so that every
                # rebuild of the combo (from register_detector, visibility changes,
                # etc.) consistently re-applies the user's chosen axis.
                # It is only cleared by reset_multiaxis() when the run ends.
                desired = self._preferred_plot_xaxis or current
                idx = self.xaxis_combo.findText(desired)
                if idx >= 0:
                    self.xaxis_combo.setCurrentIndex(idx)
                else:
                    # Desired axis not available yet (no data); fall back to
                    # prior selection if possible.
                    idx2 = self.xaxis_combo.findText(current)
                    if idx2 >= 0:
                        self.xaxis_combo.setCurrentIndex(idx2)
            finally:
                try:
                    self.xaxis_combo.blockSignals(False)
                except Exception:
                    pass

        QtCore.QTimer.singleShot(0, _do)

    def set_preferred_plot_xaxis(self, axis_name: str | None) -> None:
        """Request selecting a specific x-axis after the next x-axis refresh.

        Useful when starting a multi-axis run: the available axes are only known
        after the first samples arrive, so we store a preference and apply it
        when `_refresh_xaxis_options()` next runs.
        """
        try:
            name = str(axis_name) if axis_name is not None else ""
        except Exception:
            name = ""
        name = name.strip()
        self._preferred_plot_xaxis = name or None

        # If the axis is already available, apply immediately.
        try:
            if self._preferred_plot_xaxis is not None and hasattr(self, "xaxis_combo"):
                idx = self.xaxis_combo.findText(self._preferred_plot_xaxis)
                if idx >= 0:
                    self.xaxis_combo.setCurrentIndex(idx)
                    self._preferred_plot_xaxis = None
        except Exception:
            pass

    # -----------------------------
    # Load saved data
    # -----------------------------
    def _infer_loaded_detector_id(self, file_path: str, h5_detector: str | None = None) -> str:
        """Infer detector id from file metadata or the StreamSaver filename pattern."""
        try:
            if h5_detector is not None and str(h5_detector).strip():
                return str(h5_detector).strip()
        except Exception:
            pass

        try:
            stem = Path(file_path).stem
            # StreamSaver uses: <timestamp>__<detector_id>__<mode>
            parts = stem.split("__")
            if len(parts) >= 3 and parts[1].strip():
                return parts[1].strip()
        except Exception:
            pass

        return "detector"

    def _on_load_data(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load Data",
            "",
            "Data files (*.h5 *.hdf5 *.txt *.npy *.csv);;All files (*)",
        )
        if not paths:
            return

        # ── Snapshot current display layout before touching anything ──────
        layout_snapshot = self._capture_display_layout()

        try:
            self.status_message.emit(f"Loading {len(paths)} file(s) …", 0)
        except Exception:
            pass

        # ── 1. Parse every selected file ──────────────────────────────────
        parsed: list[tuple] = []
        errors: list[str] = []
        for path in paths:
            try:
                mid, channels, file_layout = self._parse_data_file(path)
                parsed.append((mid, channels, file_layout))
            except Exception as exc:
                errors.append(f"{Path(path).name}: {exc}")

        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Load errors",
                "Some files could not be read:\n" + "\n".join(errors),
            )

        if not parsed:
            self.status_message.emit("Load failed — no valid files.", 6000)
            return

        # ── 2. Group files by measurement UUID ────────────────────────────
        import uuid as _uuid
        groups: dict[str, list[dict]] = {}
        saved_layouts: list[dict] = []        # collect all non-None saved layouts
        for mid, channels, file_layout in parsed:
            key = mid if mid else str(_uuid.uuid4())
            groups.setdefault(key, []).append(channels)
            if file_layout:
                saved_layouts.append(file_layout)

        # ── 3. Warn when files belong to different measurements ───────────
        if len(groups) > 1:
            msg_lines = ["The selected files belong to different measurements:"]
            for k, grp in groups.items():
                total_ch = sum(len(g) for g in grp)
                short_k = (k[:8] + "…") if len(k) >= 8 else k
                msg_lines.append(f"  • {short_k}  ({total_ch} channel(s))")
            msg_lines.append(
                "\nThey will be loaded together.\n"
                "Conflicting channel names are renamed automatically "
                "(e.g. detector1 → detector1_2)."
            )
            QtWidgets.QMessageBox.information(
                self, "Multiple measurements", "\n".join(msg_lines)
            )

        # ── 4. Merge all groups, renaming duplicate channel ids ───────────
        merged: dict[str, np.ndarray] = {}
        for group_list in groups.values():
            for channels in group_list:
                for det_id, arr in channels.items():
                    if det_id not in merged:
                        merged[det_id] = arr
                    else:
                        suffix = 2
                        new_id = f"{det_id}_{suffix}"
                        while new_id in merged:
                            suffix += 1
                            new_id = f"{det_id}_{suffix}"
                        merged[new_id] = arr

        # ── 5. Keep or replace the layout based on the persistent toggle ──
        keep_layout = self._load_keep_layout

        # ── 6. Load data (always data-driven first) ───────────────────────
        self._load_channels_into_view(merged)

        # ── 7. Apply layout ───────────────────────────────────────────────
        if keep_layout:
            # User wants to preserve the current display layout
            self._restore_display_layout(layout_snapshot)
        else:
            # Try to restore the layout saved in the file(s).
            # Use the first file's layout; if none found or restore fails,
            # silently fall back to the data-driven layout already applied.
            if saved_layouts:
                try:
                    self._restore_display_layout(saved_layouts[0])
                except Exception:
                    pass   # data-driven layout already in place — keep it

        # ── 8. Emit view_changed so docks settle into the final state ────────
        # When the user chose to KEEP the current layout, the docks have not
        # moved at all (every internal state change used emit_signal=False).
        # Emitting here would cause _on_live_view_changed to aggressively
        # hide/show docks based on view_mode, overriding the user's layout.
        # So only emit when we actually want the docks to follow the data.
        if not keep_layout:
            try:
                self.view_changed.emit(self.view_mode)
            except Exception:
                pass

        total_pts = sum(len(a) for a in merged.values())
        uuid_count = len(groups)
        label = "same measurement" if uuid_count == 1 else f"{uuid_count} different measurements"
        self.status_message.emit(
            f"Loaded {len(merged)} channel(s), {total_pts} points  ({label})", 8000
        )
        return

        try:
            # Load into numpy arrays
            times = None
            vals = None
            xs = None
            ys = None
            zs = None
            detector_from_file = None

            p = Path(path)
            suffix = p.suffix.lower()

            if suffix in (".h5", ".hdf5"):
                import h5py

                with h5py.File(path, "r") as f:
                    # Two supported shapes:
                    # - stream saver: dataset 'data' columns [timestamp,value,x,y,z]
                    # - camera frame: dataset 'image'
                    if "data" in f:
                        ds = f["data"]
                        try:
                            detector_from_file = ds.attrs.get("detector")
                        except Exception:
                            detector_from_file = None
                        arr = ds[:]
                        if arr.ndim == 1:
                            arr = arr.reshape(1, -1)
                        if arr.shape[1] < 2:
                            raise ValueError("HDF5 'data' dataset must have at least 2 columns")
                        times = np.asarray(arr[:, 0], dtype=float)
                        vals = np.asarray(arr[:, 1], dtype=float)
                        xs = np.asarray(arr[:, 2], dtype=float) if arr.shape[1] > 2 else None
                        ys = np.asarray(arr[:, 3], dtype=float) if arr.shape[1] > 3 else None
                        zs = np.asarray(arr[:, 4], dtype=float) if arr.shape[1] > 4 else None
                    elif "image" in f:
                        img = f["image"][:]
                        self._set_camera_view()
                        # Display like the live camera path (transpose for consistency).
                        try:
                            self.image_view.setImage(np.asarray(img).T, autoLevels=True)
                        except Exception:
                            self.image_view.setImage(np.asarray(img), autoLevels=True)
                        return
                    else:
                        raise KeyError("No 'data' or 'image' dataset found")

            elif suffix == ".npy":
                arr = np.load(path)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[1] < 2:
                    raise ValueError(".npy data must have at least 2 columns: timestamp,value")
                times = np.asarray(arr[:, 0], dtype=float)
                vals = np.asarray(arr[:, 1], dtype=float)
                xs = np.asarray(arr[:, 2], dtype=float) if arr.shape[1] > 2 else None
                ys = np.asarray(arr[:, 3], dtype=float) if arr.shape[1] > 3 else None
                zs = np.asarray(arr[:, 4], dtype=float) if arr.shape[1] > 4 else None

            else:
                # StreamSaver .txt has an unquoted python dict in the final 'meta' column,
                # which may contain commas. Parse by splitting only the first 5 commas.
                t_list: list[float] = []
                v_list: list[float] = []
                x_list: list[float] = []
                y_list: list[float] = []
                z_list: list[float] = []

                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    header = f.readline()
                    # If the file doesn't have a header, treat it as data.
                    if header and ("timestamp" not in header.lower()):
                        # rewind
                        f.seek(0)
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        if ln.startswith("#"):
                            continue
                        parts = ln.split(",", 5)
                        if len(parts) < 2:
                            continue

                        def _to_float(s: str) -> float:
                            s = (s or "").strip()
                            if s == "":
                                return float("nan")
                            try:
                                return float(s)
                            except Exception:
                                return float("nan")

                        t_list.append(_to_float(parts[0]))
                        v_list.append(_to_float(parts[1]))
                        x_list.append(_to_float(parts[2]) if len(parts) > 2 else float("nan"))
                        y_list.append(_to_float(parts[3]) if len(parts) > 3 else float("nan"))
                        z_list.append(_to_float(parts[4]) if len(parts) > 4 else float("nan"))

                if not t_list:
                    raise ValueError("No numeric rows found")

                times = np.asarray(t_list, dtype=float)
                vals = np.asarray(v_list, dtype=float)
                xs = np.asarray(x_list, dtype=float)
                ys = np.asarray(y_list, dtype=float)
                zs = np.asarray(z_list, dtype=float)

            det_id = self._infer_loaded_detector_id(path, detector_from_file)
            try:
                # Show only this detector when loading a single stream.
                self.set_selected_detectors([det_id])
            except Exception:
                pass

            # Decide whether this looks like multi-axis data (has usable X+Y)
            finite_vals = np.isfinite(vals) if vals is not None else None
            finite_xy = None
            try:
                if xs is not None and ys is not None:
                    finite_xy = np.isfinite(xs) & np.isfinite(ys)
            except Exception:
                finite_xy = None

            looks_like_scan = False
            try:
                if finite_xy is not None and finite_vals is not None:
                    n_good = int(np.sum(finite_xy & finite_vals))
                    looks_like_scan = n_good >= 4
            except Exception:
                looks_like_scan = False

            if looks_like_scan:
                # Clear prior scan/strip data so the loaded scan is what renders.
                try:
                    self.reset_1d_detector()
                except Exception:
                    pass
                try:
                    self.reset_multiaxis()
                except Exception:
                    pass

                self.register_detector(det_id)

                mask = np.isfinite(vals)
                if xs is not None:
                    mask = mask & np.isfinite(xs)
                if ys is not None:
                    mask = mask & np.isfinite(ys)
                coords: list[tuple[dict, float]] = []
                zvals: set[float] = set()
                for i in np.where(mask)[0]:
                    st: dict = {}
                    try:
                        st["X"] = float(xs[i])
                        st["Y"] = float(ys[i])
                        if zs is not None and np.isfinite(zs[i]):
                            zf = float(zs[i])
                            st["Z"] = zf
                            zvals.add(zf)
                    except Exception:
                        continue
                    try:
                        coords.append((st, float(vals[i])))
                    except Exception:
                        pass

                self.multi_coords[det_id] = coords
                self._multi_dirty = True
                self._last_multi_render = 0.0
                self._z_values_set = set(zvals)
                self._z_values = sorted(self._z_values_set)
                try:
                    self.z_slider.setMaximum(max(0, len(self._z_values) - 1))
                except Exception:
                    pass

                # Switch to detector view and render once immediately.
                self._set_detector_view()
                try:
                    self._refresh_xaxis_options()
                except Exception:
                    pass
                try:
                    self._update_multiaxis_visualization()
                except Exception:
                    pass
                try:
                    n_pts = len(coords)
                    self.status_message.emit(
                        f"Loaded scan: {Path(path).name}  ({n_pts} points, detector: {det_id})", 8000
                    )
                except Exception:
                    pass

            else:
                # Treat as strip chart (value vs time)
                try:
                    self.prepare_strip_chart_plot()
                except Exception:
                    try:
                        self.reset_multiaxis()
                    except Exception:
                        pass

                self.register_detector(det_id)

                # Use seconds relative to first sample for a clean x-axis.
                try:
                    t0 = float(times[0])
                except Exception:
                    t0 = 0.0
                try:
                    t_rel = np.asarray(times, dtype=float) - t0
                except Exception:
                    t_rel = np.arange(len(vals), dtype=float)

                # Expand window size so loaded traces are visible (cap to UI limit)
                try:
                    desired = int(min(max(len(vals), self._window_size), 10000))
                    if desired != self._window_size:
                        try:
                            self.window_spin.blockSignals(True)
                            self.window_spin.setValue(desired)
                        finally:
                            self.window_spin.blockSignals(False)
                        self.set_window_size(desired)
                except Exception:
                    pass

                mask = np.isfinite(vals)
                try:
                    t_plot = list(t_rel[mask])
                    v_plot = list(vals[mask])
                except Exception:
                    t_plot = list(t_rel)
                    v_plot = list(vals)

                self._detector_times[det_id] = deque(t_plot, maxlen=self._window_size)
                self._detector_buffers[det_id] = deque(v_plot, maxlen=self._window_size)

                # Prefer camera view for strip chart mode (keeps camera panel visible);
                # plot is always visible underneath.
                self._set_camera_view()
                try:
                    n_pts = len(t_plot)
                    self.status_message.emit(
                        f"Loaded strip chart: {Path(path).name}  ({n_pts} samples, detector: {det_id})", 8000
                    )
                except Exception:
                    pass

        except Exception as e:
            try:
                self.status_message.emit(f"Load failed: {Path(path).name} — {e}", 8000)
            except Exception:
                pass
            QtWidgets.QMessageBox.warning(self, "Load Data", f"Failed to load: {path}\n\nError: {e}")

    def _load_multichannel_h5(self, path: str) -> None:
        """Load all detectors from a multi-channel HDF5 file."""
        if multichannel_h5 is None:
            QtWidgets.QMessageBox.warning(self, "Load Data", "multichannel_h5 module not available.")
            return
        try:
            channels = multichannel_h5.load(path)
        except Exception as e:
            self.status_message.emit(f"Load failed: {Path(path).name} — {e}", 8000)
            QtWidgets.QMessageBox.warning(self, "Load Data", f"Failed to load: {path}\n\n{e}")
            return

        if not channels:
            self.status_message.emit(f"No channels found in {Path(path).name}", 6000)
            return

        # Reset prior data
        try:
            self.reset_1d_detector()
        except Exception:
            pass
        try:
            self.reset_multiaxis()
        except Exception:
            pass

        det_ids = list(channels.keys())
        try:
            self.set_selected_detectors(det_ids)
        except Exception:
            pass

        total_pts = 0
        for det_id, arr in channels.items():
            # arr columns: [timestamp, value, x, y, z]
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue
            vals = arr[:, 1]
            xs = arr[:, 2] if arr.shape[1] > 2 else None
            ys = arr[:, 3] if arr.shape[1] > 3 else None
            zs = arr[:, 4] if arr.shape[1] > 4 else None

            finite_vals = np.isfinite(vals)
            has_xy = (xs is not None and ys is not None and
                      np.any(np.isfinite(xs) & np.isfinite(ys) & finite_vals))

            self.register_detector(det_id)

            if has_xy:
                coords: list[tuple[dict, float]] = []
                zvals: set[float] = set()
                for i in range(len(vals)):
                    if not np.isfinite(vals[i]):
                        continue
                    st: dict = {}
                    try:
                        st["X"] = float(xs[i])
                        st["Y"] = float(ys[i])
                        if zs is not None and np.isfinite(zs[i]):
                            zf = float(zs[i])
                            st["Z"] = zf
                            zvals.add(zf)
                    except Exception:
                        continue
                    coords.append((st, float(vals[i])))
                self.multi_coords[det_id] = coords
                total_pts += len(coords)
                for z in zvals:
                    self._z_values_set.add(z)
            else:
                # strip-chart channel
                times = arr[:, 0]
                t0 = float(times[0]) if len(times) > 0 else 0.0
                t_rel = times - t0
                mask = np.isfinite(vals)
                self._detector_times[det_id] = deque(t_rel[mask].tolist(), maxlen=self._window_size)
                self._detector_buffers[det_id] = deque(vals[mask].tolist(), maxlen=self._window_size)
                total_pts += int(mask.sum())

        if self.multi_coords:
            self._z_values = sorted(self._z_values_set)
            try:
                self.z_slider.setMaximum(max(0, len(self._z_values) - 1))
            except Exception:
                pass
            self._multi_dirty = True
            self._last_multi_render = 0.0
            self._set_detector_view()
            try:
                self._refresh_xaxis_options()
            except Exception:
                pass
            try:
                self._update_multiaxis_visualization()
            except Exception:
                pass
        else:
            self._set_camera_view()

        self.status_message.emit(
            f"Loaded {len(det_ids)} channel(s) from {Path(path).name}  ({total_pts} points)", 8000
        )

    def _capture_display_layout(self) -> dict:
        """Return a snapshot of the current display settings.

        Captured state
        --------------
        view_mode      : "camera" | "detector"
        xaxis          : current X-axis combo text
        z_slider       : z-slider integer value
        draw_lines     : bool
        window_size    : int
        """
        snap: dict = {}
        try:
            snap["view_mode"] = str(getattr(self, "view_mode", "camera"))
        except Exception:
            snap["view_mode"] = "camera"
        try:
            snap["xaxis"] = self.xaxis_combo.currentText()
        except Exception:
            snap["xaxis"] = "Index"
        try:
            snap["z_slider"] = self.z_slider.value()
        except Exception:
            snap["z_slider"] = 0
        try:
            snap["draw_lines"] = bool(self.draw_lines_cb.isChecked())
        except Exception:
            snap["draw_lines"] = True
        try:
            snap["window_size"] = int(self._window_size)
        except Exception:
            snap["window_size"] = 200
        return snap

    def get_display_layout_json(self) -> str:
        """Return the current display layout as a JSON string for embedding in HDF5 headers."""
        import json as _json
        try:
            return _json.dumps(self._capture_display_layout())
        except Exception:
            return "{}"

    def _restore_display_layout(self, snap: dict) -> None:
        """Restore display settings from a snapshot produced by _capture_display_layout.

        Does NOT emit view_changed — the caller (_on_load_data) emits it once
        after all layout decisions have been made.
        """
        try:
            mode = snap.get("view_mode", "camera")
            self._apply_view_mode(mode, emit_signal=False)
        except Exception:
            pass

        # Restore X-axis selection — the combo may have been repopulated by the
        # load, so we try to find the same text first; if missing, keep whatever
        # the load chose.
        try:
            target = snap.get("xaxis", "Index")
            idx = self.xaxis_combo.findText(target)
            if idx >= 0:
                self.xaxis_combo.setCurrentIndex(idx)
        except Exception:
            pass

        try:
            self.z_slider.setValue(int(snap.get("z_slider", 0)))
        except Exception:
            pass

        try:
            self.draw_lines_cb.setChecked(bool(snap.get("draw_lines", True)))
        except Exception:
            pass

        try:
            ws = int(snap.get("window_size", 200))
            self.set_window_size(ws)
            self.window_spin.blockSignals(True)
            self.window_spin.setValue(ws)
            self.window_spin.blockSignals(False)
        except Exception:
            pass

        # Re-render with restored settings
        try:
            self._update_plot()
        except Exception:
            pass

    def _on_keep_layout_toggled(self, checked: bool) -> None:
        """Called when the 'Keep Current Layout When Loading' menu item is toggled."""
        self._load_keep_layout = bool(checked)
        # Keep the action check-state in sync if called programmatically
        try:
            self.keep_layout_action.setChecked(self._load_keep_layout)
        except Exception:
            pass

    def _parse_data_file(self, path: str) -> tuple:
        """Parse one data file.

        Returns
        -------
        (measurement_id | None, {det_id: ndarray(N,5)}, layout_dict | None)

        Output arrays always have exactly 5 columns: [timestamp, value, x, y, z].
        Missing columns are filled with NaN.

        Supported formats:
          • multi-channel HDF5  (format attr == "multichannel_stream")
          • single-channel HDF5 (StreamSaver format, dataset 'data')
          • txt / csv           (comma-separated, optional header)
          • npy                 (2-D float array)
        """
        import h5py
        import json as _json

        p = Path(path)
        channels: dict[str, np.ndarray] = {}
        mid: str | None = None
        layout: dict | None = None

        if p.suffix.lower() in (".h5", ".hdf5"):
            with h5py.File(path, "r") as f:
                raw_mid = f.attrs.get("measurement_id", None)
                mid = str(raw_mid) if raw_mid is not None else None

                # Read saved display layout (JSON string)
                raw_layout = f.attrs.get("display_layout", None)
                if raw_layout is not None:
                    try:
                        layout = _json.loads(str(raw_layout))
                    except Exception:
                        layout = None

                if str(f.attrs.get("format", "")) == "multichannel_stream":
                    # multi-channel format (MultiChannelSaver)
                    dets_grp = f.get("detectors")
                    if dets_grp:
                        for det_id in dets_grp:
                            ds = dets_grp[det_id].get("data")
                            if ds is not None:
                                channels[det_id] = np.asarray(ds[:], dtype=float)
                elif "data" in f:
                    # single-channel StreamSaver format
                    arr = np.asarray(f["data"][:], dtype=float)
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    raw_det = f["data"].attrs.get("detector", None)
                    det_id = (
                        str(raw_det)
                        if raw_det is not None
                        else self._infer_loaded_detector_id(path, None)
                    )
                    channels[det_id] = arr
                else:
                    raise KeyError("No 'data' dataset or 'detectors' group found")

        elif p.suffix.lower() in (".txt", ".csv"):
            def _f(s: str) -> float:
                s = (s or "").strip()
                if not s:
                    return float("nan")
                try:
                    return float(s)
                except Exception:
                    return float("nan")

            t_list: list[float] = []
            v_list: list[float] = []
            x_list: list[float] = []
            y_list: list[float] = []
            z_list: list[float] = []
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                header = fh.readline()
                if header and "timestamp" not in header.lower():
                    fh.seek(0)
                for ln in fh:
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    parts = ln.split(",", 5)
                    if len(parts) < 2:
                        continue
                    t_list.append(_f(parts[0]))
                    v_list.append(_f(parts[1]))
                    x_list.append(_f(parts[2]) if len(parts) > 2 else float("nan"))
                    y_list.append(_f(parts[3]) if len(parts) > 3 else float("nan"))
                    z_list.append(_f(parts[4]) if len(parts) > 4 else float("nan"))
            if not t_list:
                raise ValueError("No numeric rows found")
            arr = np.column_stack([
                np.asarray(t_list, float),
                np.asarray(v_list, float),
                np.asarray(x_list, float),
                np.asarray(y_list, float),
                np.asarray(z_list, float),
            ])
            # infer det_id from filename pattern  YYYYMMDD__<id>__stream
            name_parts = p.stem.split("__")
            det_id = name_parts[1] if len(name_parts) >= 2 else p.stem
            channels[det_id] = arr

        elif p.suffix.lower() == ".npy":
            arr = np.load(path)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            channels[p.stem] = arr

        else:
            raise ValueError(f"Unsupported file format: {p.suffix!r}")

        # Normalise every array to exactly (N, 5)
        for k, a in list(channels.items()):
            if a.ndim == 1:
                a = a.reshape(1, -1)
            if a.shape[1] < 5:
                pad = np.full((a.shape[0], 5 - a.shape[1]), float("nan"))
                a = np.hstack([a, pad])
            channels[k] = a

        if not channels:
            raise ValueError("No channel data could be extracted from the file")

        return mid, channels, layout

    def _load_channels_into_view(self, merged: dict) -> None:
        """Display merged {det_id: ndarray(N,5)} in the live tab.

        Channels that have finite X+Y coordinates are treated as scan data
        (multi_coords).  Channels with only timestamp+value become strip-chart
        traces.
        """
        # Reset prior data
        try:
            self.reset_1d_detector()
        except Exception:
            pass
        try:
            self.reset_multiaxis()
        except Exception:
            pass

        det_ids = list(merged.keys())
        try:
            self.set_selected_detectors(det_ids)
        except Exception:
            pass

        has_scan = False

        for det_id, arr in merged.items():
            arr = np.asarray(arr)
            if arr.ndim != 2 or arr.shape[1] < 2:
                continue

            vals = arr[:, 1]
            xs   = arr[:, 2] if arr.shape[1] > 2 else None
            ys   = arr[:, 3] if arr.shape[1] > 3 else None
            zs   = arr[:, 4] if arr.shape[1] > 4 else None

            finite_vals = np.isfinite(vals)
            has_xy = (
                xs is not None and ys is not None
                and np.any(np.isfinite(xs) & np.isfinite(ys) & finite_vals)
            )

            self.register_detector(det_id)

            if has_xy:
                has_scan = True
                coords: list[tuple[dict, float]] = []
                zvals: set[float] = set()
                for i in range(len(vals)):
                    if not np.isfinite(vals[i]):
                        continue
                    st: dict = {}
                    try:
                        st["X"] = float(xs[i])
                        st["Y"] = float(ys[i])
                        if zs is not None and np.isfinite(zs[i]):
                            zf = float(zs[i])
                            st["Z"] = zf
                            zvals.add(zf)
                    except Exception:
                        continue
                    coords.append((st, float(vals[i])))
                self.multi_coords[det_id] = coords
                for z in zvals:
                    self._z_values_set.add(z)
            else:
                # strip-chart trace
                times = arr[:, 0]
                t0 = float(times[0]) if len(times) > 0 else 0.0
                t_rel = times - t0
                mask = np.isfinite(vals)
                self._detector_times[det_id] = deque(
                    t_rel[mask].tolist(), maxlen=self._window_size
                )
                self._detector_buffers[det_id] = deque(
                    vals[mask].tolist(), maxlen=self._window_size
                )

        if has_scan:
            self._z_values = sorted(self._z_values_set)
            try:
                self.z_slider.setMaximum(max(0, len(self._z_values) - 1))
            except Exception:
                pass
            self._multi_dirty = True
            self._last_multi_render = 0.0
            # Update internal state only — dock show/hide is handled by the
            # caller (_on_load_data) after layout preference is applied.
            self._apply_view_mode("detector", emit_signal=False)
            try:
                self._refresh_xaxis_options()
            except Exception:
                pass
            try:
                self._update_multiaxis_visualization()
            except Exception:
                pass
        else:
            self._apply_view_mode("camera", emit_signal=False)

    def export_all_channels(self) -> None:
        """Save all currently loaded/acquired multi-channel data to one HDF5 file."""
        if multichannel_h5 is None:
            QtWidgets.QMessageBox.warning(self, "Export", "multichannel_h5 module not available.")
            return
        if not self.multi_coords:
            QtWidgets.QMessageBox.information(self, "Export", "No multi-channel scan data to export.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"{ts}__multichannel.h5"
        try:
            default_dir = str(Path(__file__).resolve().parents[2] / "data")
        except Exception:
            default_dir = ""

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export All Channels",
            str(Path(default_dir) / default_name),
            "HDF5 files (*.h5 *.hdf5);;All files (*)",
        )
        if not path:
            return

        try:
            self.status_message.emit(f"Exporting channels to {Path(path).name} …", 0)
            out = multichannel_h5.save(path, self.multi_coords)
            self.status_message.emit(
                f"Exported {len(self.multi_coords)} channel(s) → {out.name}", 8000
            )
        except Exception as e:
            self.status_message.emit(f"Export failed: {e}", 8000)
            QtWidgets.QMessageBox.warning(self, "Export", f"Failed to export:\n{e}")

    def _on_draw_lines_toggled(self, checked: bool) -> None:
        """Toggle connecting lines between data points in the multi-axis plot."""
        self._draw_lines = bool(checked)
        # Force immediate re-render with updated line style.
        self._multi_dirty = True
        self._last_multi_render = 0.0

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
    @QtCore.pyqtSlot(str, float, float)
    def add_detector_sample_qt(self, detector_id: str, value: float, timestamp: float):
        """Qt-invokable wrapper around add_detector_sample.

        QMetaObject.invokeMethod requires the target to be registered in the
        Qt meta-object system with a matching signature.
        """
        self.add_detector_sample(detector_id, value, timestamp)

    def add_detector_sample(self, detector_id: str, value: float, timestamp: float | None = None):
        """Add a sample for a named detector. timestamp in seconds. Thread-safe caller should use queued calls."""
        det_id = self._map_and_filter_detector_id(detector_id)
        if det_id is None:
            return
        if timestamp is None:
            timestamp = time.time()
        t_rel = timestamp - self._t0
        # ensure buffers and UI exist
        if det_id not in self._detector_buffers:
            self.register_detector(det_id)

        self._detector_times[det_id].append(t_rel)
        self._detector_buffers[det_id].append(value)

    # -----------------------------
    # multi-axis detector callback
    # -----------------------------
    def queue_multiaxis_sample(self, detector_id: str, state: dict, value: float):
        """Called from worker thread: push sample into thread-safe queue.

        The state dict must already be a copy (not a live reference).
        The GUI timer drains this queue via _update_plot().
        """
        det_id = self._map_and_filter_detector_id(detector_id)
        if det_id is None:
            return
        self._multiaxis_queue.append((det_id, state, value))

    def add_multiaxis_detector(self, detector_id: str, state: dict, value: float):
        """Append a multi-axis sample for a specific detector id."""
        self.multi_coords.setdefault(detector_id, []).append((state.copy(), value))
        self._multi_dirty = True
        # update Z slider range if Z present — O(1) per sample using a tracked set
        if "Z" in state:
            z_val = state["Z"]
            if z_val not in self._z_values_set:
                self._z_values_set.add(z_val)
                self._z_values = sorted(self._z_values_set)
                self.z_slider.setMaximum(max(0, len(self._z_values) - 1))

    # -----------------------------
    # periodic plot update
    # -----------------------------
    def _update_plot(self):
        # Drain multi-axis samples queued from the worker thread.
        # Batching all pending samples before rendering avoids signal-queue
        # overflow and ensures the heatmap reflects the latest data.
        while self._multiaxis_queue:
            try:
                det_id, state, value = self._multiaxis_queue.popleft()
                self.add_multiaxis_detector(det_id, state, value)
            except IndexError:
                break
            except Exception:
                pass

        # 1D detector plot (strip-chart)
        # When a multi-axis run is active, the shared plot is repurposed for
        # numeric scan plotting; do not update or recreate strip-chart curves.
        if self._detector_times and not self.multi_coords:
            any_1d_data = False
            # update each detector curve
            for det_id, times in self._detector_times.items():
                vals = self._detector_buffers.get(det_id, [])
                curve = self._detector_curves.get(det_id)
                try:
                    # If the shared plot was cleared/repurposed, a curve object may
                    # still exist in the dict but no longer be attached to the scene.
                    if curve is None or curve.scene() is None:
                        pen = None
                        try:
                            if curve is not None:
                                pen = curve.opts.get("pen")
                        except Exception:
                            pen = None
                        if pen is None:
                            pen = pg.mkPen(color=(255, 255, 255), width=2)
                        curve = self.plot_widget.plot([], [], pen=pen, name=det_id)
                        self._detector_curves[det_id] = curve

                    curve.setData(list(times), list(vals))
                    try:
                        if len(times) > 0:
                            any_1d_data = True
                    except Exception:
                        pass
                except Exception:
                    pass

            # Show legend only once we actually have data points.
            try:
                self._set_legend_visible(bool(any_1d_data))
            except Exception:
                pass

        # multi-axis visualization (heatmap + numeric plot)
        # Throttle to avoid UI slowdown on large scans; update regardless of view mode
        # so the heatmap is ready when the user switches to detector view.
        try:
            now = time.time()
            if self.multi_coords and self._multi_dirty and (now - self._last_multi_render) >= 0.25:
                self._last_multi_render = now
                self._multi_dirty = False
                self._update_multiaxis_visualization()
                # numeric plot: detector value vs. selected axis
                xaxis = self.xaxis_combo.currentText() if hasattr(self, 'xaxis_combo') else 'Index'

                # On entry to multi-axis numeric plotting, clear the plot once so
                # we don't end up with *both* strip-chart curves and scan curves
                # (which would duplicate legend entries).
                try:
                    if getattr(self, "_plot_mode", "strip") != "multiaxis":
                        self._plot_mode = "multiaxis"
                        self._clear_plot_and_legend()
                        # First data has arrived: refresh x-axis options so the
                        # preferred x-axis (set before data existed) gets applied.
                        self._refresh_xaxis_options()
                except Exception:
                    pass

                any_curve = False
                updated: set[str] = set()
                for det_id, det_list in self.multi_coords.items():
                    if not self._is_detector_visible(det_id):
                        continue
                    if not det_list:
                        continue
                    xs = []
                    ys = []
                    for idx, (s, v) in enumerate(det_list):
                        if not np.isfinite(v):
                            continue
                        if xaxis == 'Index':
                            xs.append(idx)
                        else:
                            val = None
                            try:
                                val = s.get(xaxis)
                            except Exception:
                                val = None
                            xs.append(float(val) if val is not None else idx)
                        ys.append(v)
                    if xs:
                        # Reuse the existing per-detector curve objects. If the
                        # plot was cleared, recreate them once.
                        curve = self._detector_curves.get(det_id)
                        if curve is None or curve.scene() is None:
                            reg_pen = None
                            try:
                                if curve is not None:
                                    reg_pen = curve.opts.get('pen')
                            except Exception:
                                reg_pen = None
                            if reg_pen is None:
                                reg_pen = pg.mkPen(color=(255, 255, 255), width=2)
                            curve = self.plot_widget.plot([], [], pen=pg.mkPen(reg_pen, width=2), name=det_id)
                            self._detector_curves[det_id] = curve

                        # Apply line/dot style based on toggle
                        draw_lines = getattr(self, '_draw_lines', True)
                        try:
                            existing_pen = curve.opts.get('pen')
                            if draw_lines:
                                curve.setPen(existing_pen)
                                curve.setSymbol(None)
                            else:
                                curve.setPen(None)
                                curve.setSymbol('o')
                                curve.setSymbolSize(5)
                                sp = pg.mkPen(existing_pen) if existing_pen else pg.mkPen(color=(255, 255, 255))
                                curve.setSymbolPen(sp)
                                curve.setSymbolBrush(sp.color())
                        except Exception:
                            pass

                        try:
                            curve.setVisible(True)
                        except Exception:
                            pass
                        try:
                            curve.setData(xs, ys)
                        except Exception:
                            pass

                        any_curve = True
                        updated.add(det_id)

                # Hide curves that were not updated (e.g., detector hidden/filtered).
                try:
                    for det_id, curve in list(self._detector_curves.items()):
                        if det_id in updated:
                            continue
                        try:
                            curve.setVisible(False)
                        except Exception:
                            pass
                except Exception:
                    pass
                if any_curve:
                    try:
                        self._set_legend_visible(True)
                    except Exception:
                        pass
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
            if not self._is_detector_visible(det_id):
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
                xi = {v: i for i, v in enumerate(xs)}
                yi = {v: i for i, v in enumerate(ys)}

                arr = np.zeros((len(xs), len(ys)), dtype=np.float32)
                for s, v in det_list:
                    try:
                        arr[xi[s[ax1]], yi[s[ax2]]] = v
                    except KeyError:
                        pass

                try:
                    # Use a contiguous array for display/caching. Non-contiguous
                    # transpose views can sometimes lead to confusing artifacts
                    # (looks like images are "mixed") depending on downstream
                    # consumers and pyqtgraph versions.
                    img2d = np.ascontiguousarray(arr.T)
                    img_view.setImage(img2d, autoLevels=True)
                    try:
                        self._detector_last_images[det_id] = np.asarray(img2d).copy()
                    except Exception:
                        pass
                    if self._levels[0] is not None and self._levels[1] is not None:
                        img_view.getImageItem().setLevels(self._levels[0], self._levels[1])
                except Exception:
                    pass

            elif len(axes) == 3:
                ax1, ax2, ax3 = axes
                xs = sorted(set(s[ax1] for s, _ in det_list))
                ys = sorted(set(s[ax2] for s, _ in det_list))
                zs = sorted(set(s[ax3] for s, _ in det_list))
                xi = {v: i for i, v in enumerate(xs)}
                yi = {v: i for i, v in enumerate(ys)}
                zi = {v: i for i, v in enumerate(zs)}

                arr = np.zeros((len(xs), len(ys), len(zs)), dtype=np.float32)
                for s, v in det_list:
                    try:
                        arr[xi[s[ax1]], yi[s[ax2]], zi[s[ax3]]] = v
                    except KeyError:
                        pass

                idx = min(max(self.z_slider.value(), 0), len(zs) - 1)
                slice_img = arr[:, :, idx]
                try:
                    img2d = np.ascontiguousarray(slice_img.T)
                    img_view.setImage(img2d, autoLevels=True)
                    try:
                        self._detector_last_images[det_id] = np.asarray(img2d).copy()
                    except Exception:
                        pass
                    if self._levels[0] is not None and self._levels[1] is not None:
                        img_view.getImageItem().setLevels(self._levels[0], self._levels[1])
                except Exception:
                    pass

        # Update composite overlay if enabled.
        try:
            if getattr(self, 'detector_image_panel', None) and self.detector_image_panel.overlay_enabled():
                self._update_false_color_overlay()
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

    def _update_false_color_overlay(self, *_args):
        """Build/refresh composite RGB overlay from cached per-detector heatmaps."""
        try:
            if not getattr(self, 'detector_image_panel', None):
                return
            if not self.detector_image_panel.overlay_enabled():
                return
            if not self._detector_last_images:
                return

            # Determine which sources to overlay.
            # Mode A: use all detectors (palette colors)
            # Mode B: explicit R/G/B mapping
            use_all = True
            try:
                use_all = bool(self.detector_image_panel.overlay_use_all_detectors())
            except Exception:
                use_all = True

            overlay_color_mode = "fixed"
            try:
                overlay_color_mode = str(self.detector_image_panel.overlay_color_mode() or "fixed")
            except Exception:
                overlay_color_mode = "fixed"
            if overlay_color_mode not in ("fixed", "cmap"):
                overlay_color_mode = "fixed"

            det_ids: list[str] = []
            channel_map = None
            if use_all:
                # ensure deterministic order
                det_ids = [
                    d
                    for d in self._detector_image_views.keys()
                    if (d in self._detector_last_images) and self._is_detector_visible(d)
                ]
            else:
                try:
                    channel_map = self.detector_image_panel.overlay_channel_map()
                except Exception:
                    channel_map = {"R": None, "G": None, "B": None}
                # in explicit mode, we only need chosen detectors
                det_ids = [
                    d
                    for d in (channel_map.get("R"), channel_map.get("G"), channel_map.get("B"))
                    if d and self._is_detector_visible(d)
                ]

            if not det_ids:
                return

            # all images should share shape; pick first as reference
            ref = self._detector_last_images.get(det_ids[0])
            if ref is None:
                return
            h, w = int(ref.shape[0]), int(ref.shape[1])
            comp = np.zeros((h, w, 3), dtype=np.float32)

            def _norm2d(img2d: np.ndarray) -> np.ndarray:
                arr = np.asarray(img2d, dtype=np.float32)
                finite = np.isfinite(arr)
                if not np.any(finite):
                    return np.zeros((h, w), dtype=np.float32)
                try:
                    vmin = float(np.nanpercentile(arr[finite], 1))
                    vmax = float(np.nanpercentile(arr[finite], 99))
                except Exception:
                    vmin = float(np.nanmin(arr[finite]))
                    vmax = float(np.nanmax(arr[finite]))
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                    return np.zeros((h, w), dtype=np.float32)
                norm = (arr - vmin) / (vmax - vmin)
                return np.clip(np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

            def _lut_for_detector(det_id: str) -> np.ndarray | None:
                """Return uint8 LUT (256,3) for the detector's current ImageView gradient."""
                try:
                    iv = self._detector_image_views.get(det_id)
                    if iv is None:
                        return None
                    hist = getattr(getattr(iv, "ui", None), "histogram", None)
                    grad = getattr(hist, "gradient", None)
                    if grad is None or not hasattr(grad, "getLookupTable"):
                        return None
                    lut = grad.getLookupTable(256, alpha=False)
                    lut = np.asarray(lut)
                    if lut.ndim != 2 or lut.shape[0] < 2:
                        return None
                    # Some versions return Nx4; drop alpha.
                    if lut.shape[1] >= 3:
                        lut = lut[:, :3]
                    # Normalize to uint8 if needed
                    if lut.dtype != np.uint8:
                        lut = np.clip(lut, 0, 255).astype(np.uint8)
                    # Ensure exactly 256 rows by resampling if necessary
                    if lut.shape[0] != 256:
                        xs = np.linspace(0, lut.shape[0] - 1, 256)
                        idx = np.clip(xs.round().astype(int), 0, lut.shape[0] - 1)
                        lut = lut[idx]
                    return lut
                except Exception:
                    return None

            if use_all:
                for det_id in det_ids:
                    img = self._detector_last_images.get(det_id)
                    if img is None or getattr(img, "ndim", 0) != 2:
                        continue
                    if img.shape[0] != h or img.shape[1] != w:
                        continue
                    norm = _norm2d(img)

                    if overlay_color_mode == "cmap":
                        lut = _lut_for_detector(det_id)
                        if lut is None:
                            # fallback to fixed mode for this detector
                            overlay_color_mode_det = "fixed"
                        else:
                            overlay_color_mode_det = "cmap"

                        if overlay_color_mode_det == "cmap":
                            idx = np.clip((norm * 255.0).round().astype(np.int16), 0, 255)
                            layer = lut[idx].astype(np.float32) / 255.0  # (H,W,3)
                            # Screen blend per-channel
                            comp = comp + layer * (1.0 - comp)
                            continue

                    # Fixed false-color tint
                    try:
                        r, g, b = self.detector_image_panel.false_color_for(det_id)
                    except Exception:
                        r, g, b = (1.0, 0.0, 0.0)
                    comp[:, :, 0] = comp[:, :, 0] + (norm * float(r)) * (1.0 - comp[:, :, 0])
                    comp[:, :, 1] = comp[:, :, 1] + (norm * float(g)) * (1.0 - comp[:, :, 1])
                    comp[:, :, 2] = comp[:, :, 2] + (norm * float(b)) * (1.0 - comp[:, :, 2])
            else:
                # Explicit R/G/B sources; each detector feeds exactly one channel.
                if channel_map is None:
                    channel_map = {"R": None, "G": None, "B": None}
                for ch, idx in [("R", 0), ("G", 1), ("B", 2)]:
                    det_id = channel_map.get(ch)
                    if not det_id:
                        continue
                    img = self._detector_last_images.get(det_id)
                    if img is None or getattr(img, "ndim", 0) != 2:
                        continue
                    if img.shape[0] != h or img.shape[1] != w:
                        continue
                    comp[:, :, idx] = _norm2d(img)

            comp = np.clip(comp, 0.0, 1.0)
            rgb8 = (comp * 255.0).astype(np.uint8)
            self.detector_image_panel.set_overlay_image(rgb8)
        except Exception:
            pass