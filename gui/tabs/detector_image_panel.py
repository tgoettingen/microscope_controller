from __future__ import annotations

import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg


class DetectorImagePanel(QtWidgets.QWidget):
    """Container widget that holds multiple per-detector heatmaps.

    Features:
    - Per-detector colormap (gradient) selection.
    - Optional composite "false-color" overlay view.
    """

    overlay_toggled = QtCore.pyqtSignal(bool)
    overlay_settings_changed = QtCore.pyqtSignal()

    _FALSE_COLOR_RGB = [
        (1.0, 0.0, 0.0),  # red
        (0.0, 1.0, 0.0),  # green
        (0.0, 0.0, 1.0),  # blue
        (1.0, 0.0, 1.0),  # magenta
        (0.0, 1.0, 1.0),  # cyan
        (1.0, 1.0, 0.0),  # yellow
    ]

    def __init__(self, parent=None):
        super().__init__(parent)

        self._settings = QtCore.QSettings("MicroscopeController", "DetectorImagePanel")

        self._detectors: list[str] = []
        self._detector_widgets: dict[str, QtWidgets.QWidget] = {}
        self._detector_views: dict[str, pg.ImageView] = {}
        self._detector_cmap_combos: dict[str, QtWidgets.QComboBox] = {}

        try:
            self._default_gradient = str(self._settings.value("default_gradient", "viridis"))
        except Exception:
            self._default_gradient = "viridis"

        # overlay mapping controls: detector -> RGB channels
        try:
            self._overlay_use_all = bool(int(self._settings.value("overlay_use_all", 1)))
        except Exception:
            self._overlay_use_all = True
        self._overlay_map = {"R": "", "G": "", "B": ""}  # empty == None
        try:
            self._overlay_map["R"] = str(self._settings.value("overlay_map/R", "") or "")
            self._overlay_map["G"] = str(self._settings.value("overlay_map/G", "") or "")
            self._overlay_map["B"] = str(self._settings.value("overlay_map/B", "") or "")
        except Exception:
            pass

        # overlay coloring mode: fixed false colors or per-detector colormap
        try:
            self._overlay_color_mode = str(self._settings.value("overlay_color_mode", "fixed") or "fixed")
        except Exception:
            self._overlay_color_mode = "fixed"
        if self._overlay_color_mode not in ("fixed", "cmap"):
            self._overlay_color_mode = "fixed"

        # --- top controls ---
        self.overlay_cb = QtWidgets.QCheckBox("Overlay (false color)")
        try:
            self.overlay_cb.setChecked(bool(int(self._settings.value("overlay_enabled", 0))))
        except Exception:
            self.overlay_cb.setChecked(False)
        self.overlay_cb.toggled.connect(self._on_overlay_toggled)

        self.default_cmap_combo = QtWidgets.QComboBox()
        self._populate_gradients(self.default_cmap_combo)
        self._set_combo_text(self.default_cmap_combo, self._default_gradient)
        self.default_cmap_combo.currentTextChanged.connect(self._on_default_cmap_changed)

        top = QtWidgets.QWidget()
        top_layout = QtWidgets.QHBoxLayout(top)
        top_layout.setContentsMargins(4, 4, 4, 4)
        top_layout.addWidget(self.overlay_cb)
        top_layout.addSpacing(12)
        top_layout.addWidget(QtWidgets.QLabel("Default colormap:"))
        top_layout.addWidget(self.default_cmap_combo)
        top_layout.addStretch(1)

        # --- overlay controls (only meaningful when overlay enabled) ---
        self.overlay_controls = QtWidgets.QWidget()
        o = QtWidgets.QHBoxLayout(self.overlay_controls)
        o.setContentsMargins(4, 0, 4, 4)

        self.overlay_all_cb = QtWidgets.QCheckBox("Use all detectors")
        try:
            self.overlay_all_cb.setChecked(bool(self._overlay_use_all))
        except Exception:
            self.overlay_all_cb.setChecked(True)
        self.overlay_all_cb.toggled.connect(self._on_overlay_settings_changed)
        o.addWidget(self.overlay_all_cb)
        o.addSpacing(12)

        o.addWidget(QtWidgets.QLabel("Overlay colors:"))
        self.overlay_color_combo = QtWidgets.QComboBox()
        self.overlay_color_combo.addItem("Fixed false color", userData="fixed")
        self.overlay_color_combo.addItem("Detector colormap", userData="cmap")
        try:
            # restore selection
            idx = 0 if self._overlay_color_mode == "fixed" else 1
            self.overlay_color_combo.setCurrentIndex(idx)
        except Exception:
            pass
        self.overlay_color_combo.currentIndexChanged.connect(self._on_overlay_color_mode_changed)
        o.addWidget(self.overlay_color_combo)
        o.addSpacing(12)

        self.overlay_r_combo = QtWidgets.QComboBox()
        self.overlay_g_combo = QtWidgets.QComboBox()
        self.overlay_b_combo = QtWidgets.QComboBox()
        for w, name in [(self.overlay_r_combo, "R"), (self.overlay_g_combo, "G"), (self.overlay_b_combo, "B")]:
            w.currentTextChanged.connect(lambda _t, ch=name: self._on_overlay_channel_changed(ch))

        o.addWidget(QtWidgets.QLabel("R:"))
        o.addWidget(self.overlay_r_combo)
        o.addWidget(QtWidgets.QLabel("G:"))
        o.addWidget(self.overlay_g_combo)
        o.addWidget(QtWidgets.QLabel("B:"))
        o.addWidget(self.overlay_b_combo)
        o.addStretch(1)

        self.overlay_controls.setVisible(False)

        # Bundle the controls into a single widget so callers (e.g. LiveTab)
        # can move them into another panel (Detectors dock) without reaching
        # into layouts.
        self.controls_widget = QtWidgets.QWidget()
        cw = QtWidgets.QVBoxLayout(self.controls_widget)
        cw.setContentsMargins(0, 0, 0, 0)
        cw.setSpacing(0)
        cw.addWidget(top)
        cw.addWidget(self.overlay_controls)

        # --- per-detector containers ---
        self.container = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.container)
        self.layout.setSpacing(8)
        self.layout.setContentsMargins(4, 4, 4, 4)

        # --- overlay composite view ---
        self.overlay_view = pg.ImageView()
        try:
            self.overlay_view.ui.roiBtn.hide()
        except Exception:
            pass
        try:
            self.overlay_view.ui.menuBtn.hide()
        except Exception:
            pass
        try:
            # The histogram isn't meaningful for RGB composites; keep it hidden.
            self.overlay_view.ui.histogram.hide()
        except Exception:
            pass
        self.overlay_view.hide()

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.controls_widget)
        main_layout.addWidget(self.overlay_view)
        main_layout.addWidget(self.container)

        self._refresh_overlay_detector_lists()

        # Apply visibility based on persisted overlay state.
        try:
            self._apply_overlay_visibility(bool(self.overlay_cb.isChecked()))
            self.overlay_controls.setVisible(bool(self.overlay_cb.isChecked()))
        except Exception:
            pass

    def _save_setting(self, key: str, value) -> None:
        try:
            self._settings.setValue(key, value)
        except Exception:
            pass

    # -----------------
    # public helpers
    # -----------------
    def add_detector_view(self, detector_id: str, img_view: pg.ImageView) -> None:
        """Add a per-detector ImageView with a colormap selector."""
        if detector_id in self._detector_views:
            return

        self._detectors.append(detector_id)
        self._detector_views[detector_id] = img_view

        # configure the view (keep UI compact)
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
            img_view.ui.histogram.setFixedWidth(36)
        except Exception:
            pass

        # Load per-detector preferred colormap if present; otherwise use default.
        desired = self._default_gradient
        try:
            v = self._settings.value(f"detector_cmap/{detector_id}")
            if v is not None and str(v).strip():
                desired = str(v)
        except Exception:
            pass

        self._apply_gradient_to_imageview(img_view, desired)

        container = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(container)
        vbox.setContentsMargins(2, 2, 2, 2)

        title = QtWidgets.QLabel(detector_id)
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter)

        cmap_row = QtWidgets.QWidget()
        cmap_layout = QtWidgets.QHBoxLayout(cmap_row)
        cmap_layout.setContentsMargins(0, 0, 0, 0)
        cmap_layout.addWidget(QtWidgets.QLabel("Colormap:"))
        cmap_combo = QtWidgets.QComboBox()
        self._populate_gradients(cmap_combo)
        self._set_combo_text(cmap_combo, desired)
        cmap_layout.addWidget(cmap_combo)

        def _on_cmap(name: str, did=detector_id, iv=img_view):
            self._apply_gradient_to_imageview(iv, name)
            self._save_setting(f"detector_cmap/{did}", str(name))

        cmap_combo.currentTextChanged.connect(_on_cmap)

        vbox.addWidget(title)
        vbox.addWidget(cmap_row)
        vbox.addWidget(img_view)

        self._detector_widgets[detector_id] = container
        self._detector_cmap_combos[detector_id] = cmap_combo
        self.layout.addWidget(container)

        self._refresh_overlay_detector_lists()

        # apply current overlay state
        self._apply_overlay_visibility(bool(self.overlay_cb.isChecked()))

    def overlay_enabled(self) -> bool:
        return bool(self.overlay_cb.isChecked())

    def overlay_use_all_detectors(self) -> bool:
        return bool(self._overlay_use_all)

    def overlay_channel_map(self) -> dict[str, str | None]:
        """Return mapping {"R": det_id|None, "G": det_id|None, "B": det_id|None}."""
        out: dict[str, str | None] = {}
        for ch in ("R", "G", "B"):
            val = str(self._overlay_map.get(ch, "") or "").strip()
            out[ch] = val if val else None
        return out

    def overlay_color_mode(self) -> str:
        """Return 'fixed' or 'cmap'.

        Only relevant when overlay is enabled and 'Use all detectors' is on.
        """
        try:
            return str(getattr(self, "_overlay_color_mode", "fixed"))
        except Exception:
            return "fixed"

    def false_color_for(self, detector_id: str) -> tuple[float, float, float]:
        try:
            idx = self._detectors.index(detector_id)
        except ValueError:
            idx = 0
        return self._FALSE_COLOR_RGB[idx % len(self._FALSE_COLOR_RGB)]

    def set_overlay_image(self, rgb_image: np.ndarray) -> None:
        """Set the RGB overlay image (H,W,3) as uint8 or float."""
        try:
            self.overlay_view.setImage(rgb_image, autoLevels=False)
        except Exception:
            try:
                self.overlay_view.setImage(rgb_image)
            except Exception:
                pass

    # -----------------
    # internal
    # -----------------
    def _on_overlay_toggled(self, checked: bool) -> None:
        self._apply_overlay_visibility(bool(checked))
        try:
            self.overlay_controls.setVisible(bool(checked))
        except Exception:
            pass
        self._save_setting("overlay_enabled", 1 if checked else 0)
        try:
            self.overlay_toggled.emit(bool(checked))
        except Exception:
            pass

    def _apply_overlay_visibility(self, overlay: bool) -> None:
        try:
            self.overlay_view.setVisible(bool(overlay))
        except Exception:
            pass
        try:
            self.container.setVisible(not bool(overlay))
        except Exception:
            pass

    def _on_default_cmap_changed(self, name: str) -> None:
        self._default_gradient = str(name)
        self._save_setting("default_gradient", self._default_gradient)
        # Apply to existing views (treat this as a global default)
        for det_id, iv in list(self._detector_views.items()):
            try:
                self._apply_gradient_to_imageview(iv, self._default_gradient)
            except Exception:
                pass
            try:
                combo = self._detector_cmap_combos.get(det_id)
                if combo is not None:
                    self._set_combo_text(combo, self._default_gradient)
            except Exception:
                pass

    def _populate_gradients(self, combo: QtWidgets.QComboBox) -> None:
        names: list[str]
        try:
            from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

            names = sorted(list(Gradients.keys()))
        except Exception:
            names = [
                "viridis",
                "plasma",
                "inferno",
                "magma",
                "cividis",
                "grey",
            ]
        # Add simple RGB ramps (requested) even if pyqtgraph doesn't ship them.
        for extra in ["red", "green", "blue"]:
            if extra not in names:
                names.insert(0, extra)
        combo.clear()
        for n in names:
            combo.addItem(n)

    def _apply_simple_rgb_gradient(self, img_view: pg.ImageView, rgb: tuple[int, int, int]) -> bool:
        """Apply a simple black->color ramp; returns True on success."""
        r, g, b = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

        # Prefer explicit ColorMap API if available.
        try:
            cmap = pg.ColorMap(
                [0.0, 1.0],
                [QtGui.QColor(0, 0, 0), QtGui.QColor(r, g, b)],
            )
            try:
                img_view.setColorMap(cmap)
                return True
            except Exception:
                # older versions might only support imageItem.setLookupTable
                lut = cmap.getLookupTable(0.0, 1.0, 256)
                item = img_view.getImageItem()
                if item is not None and hasattr(item, "setLookupTable"):
                    item.setLookupTable(lut)
                    return True
        except Exception:
            pass

        # Fallback: drive HistogramLUT gradient directly.
        try:
            hist = getattr(getattr(img_view, "ui", None), "histogram", None)
            grad = getattr(hist, "gradient", None)
            if grad is None or not hasattr(grad, "restoreState"):
                return False
            state = {
                "mode": "rgb",
                "ticks": [
                    (0.0, (0, 0, 0, 255)),
                    (1.0, (r, g, b, 255)),
                ],
            }
            grad.restoreState(state)
            return True
        except Exception:
            return False

    def _apply_gradient_to_imageview(self, img_view: pg.ImageView, gradient_name: str) -> None:
        """Apply a predefined colormap/gradient to an ImageView.

        pyqtgraph has had a few API shapes across versions; this tries the
        common paths.
        """
        name = str(gradient_name)

        simple = name.strip().lower()
        if simple in ("red", "green", "blue"):
            rgb = (255, 0, 0) if simple == "red" else (0, 255, 0) if simple == "green" else (0, 0, 255)
            if self._apply_simple_rgb_gradient(img_view, rgb):
                return
        # 1) Newer/explicit helper
        try:
            img_view.setPredefinedGradient(name)
            return
        except Exception:
            pass

        # 2) HistogramLUTWidget gradient preset
        try:
            hist = getattr(getattr(img_view, "ui", None), "histogram", None)
            grad = getattr(hist, "gradient", None)
            if grad is not None and hasattr(grad, "loadPreset"):
                grad.loadPreset(name)
                return
        except Exception:
            pass

        # 3) Restore from Gradients dict
        try:
            from pyqtgraph.graphicsItems.GradientEditorItem import Gradients

            state = Gradients.get(name)
            hist = getattr(getattr(img_view, "ui", None), "histogram", None)
            grad = getattr(hist, "gradient", None)
            if state is not None and grad is not None and hasattr(grad, "restoreState"):
                grad.restoreState(state)
                return
        except Exception:
            pass

    def _refresh_overlay_detector_lists(self) -> None:
        """Refresh overlay detector selectors from current detector list."""
        dets = list(self._detectors)
        items = ["(none)"] + dets
        for combo, ch in [(self.overlay_r_combo, "R"), (self.overlay_g_combo, "G"), (self.overlay_b_combo, "B")]:
            try:
                combo.blockSignals(True)
                combo.clear()
                for it in items:
                    combo.addItem(it)
                # restore selection
                desired = self._overlay_map.get(ch, "") or "(none)"
                if desired and desired != "(none)":
                    idx = combo.findText(desired)
                else:
                    idx = combo.findText("(none)")
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            finally:
                try:
                    combo.blockSignals(False)
                except Exception:
                    pass

        # Enable/disable combos based on "use all"
        try:
            use_all = bool(self.overlay_all_cb.isChecked())
            self.overlay_r_combo.setEnabled(not use_all)
            self.overlay_g_combo.setEnabled(not use_all)
            self.overlay_b_combo.setEnabled(not use_all)
        except Exception:
            pass

    def _on_overlay_settings_changed(self, *_args) -> None:
        try:
            self._overlay_use_all = bool(self.overlay_all_cb.isChecked())
        except Exception:
            self._overlay_use_all = True
        self._save_setting("overlay_use_all", 1 if self._overlay_use_all else 0)
        self._refresh_overlay_detector_lists()
        try:
            self.overlay_settings_changed.emit()
        except Exception:
            pass

    def _on_overlay_color_mode_changed(self, *_args) -> None:
        try:
            mode = self.overlay_color_combo.currentData()
            mode = str(mode) if mode is not None else "fixed"
        except Exception:
            mode = "fixed"
        if mode not in ("fixed", "cmap"):
            mode = "fixed"
        self._overlay_color_mode = mode
        self._save_setting("overlay_color_mode", mode)
        try:
            self.overlay_settings_changed.emit()
        except Exception:
            pass

    def _on_overlay_channel_changed(self, ch: str) -> None:
        if ch not in ("R", "G", "B"):
            return
        combo = {"R": self.overlay_r_combo, "G": self.overlay_g_combo, "B": self.overlay_b_combo}.get(ch)
        if combo is None:
            return
        txt = str(combo.currentText() or "").strip()
        if txt == "(none)":
            txt = ""
        self._overlay_map[ch] = txt
        self._save_setting(f"overlay_map/{ch}", txt)
        try:
            self.overlay_settings_changed.emit()
        except Exception:
            pass

    def _set_combo_text(self, combo: QtWidgets.QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif combo.count() > 0:
            combo.setCurrentIndex(0)
