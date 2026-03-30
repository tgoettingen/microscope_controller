import sys
import time
import threading
import json
import base64
from pathlib import Path
import logging

import importlib.util
import numpy as np

# Prefer importing PyQt6. If it's importable, proceed. If not, give a helpful
# message guiding the user to install PyQt6 in the venv. Avoid failing based on
# distribution metadata (which can be present even after partial uninstalls).
try:
   from PyQt6 import QtWidgets, QtCore
   from PyQt6.QtGui import QAction
except Exception:
   # PyQt6 not importable — check whether PyQt5 is present to give targeted advice
   if importlib.util.find_spec("PyQt5") is not None:
      sys.stderr.write(
         "PyQt6 is not importable but PyQt5 is present in the environment.\n"
         "This code is written for PyQt6. Install PyQt6 in the active venv: `pip install PyQt6`.\n"
      )
   else:
      sys.stderr.write(
         "PyQt6 is not installed in the active Python environment.\n"
         "Install it in the project's venv: `pip install PyQt6`.\n"
      )
   sys.exit(1)

# Make this module runnable both ways:
# - as a script: `python gui/mainwindow.py` (needs repo root on sys.path for `core`)
# - as a module: `python -m gui.mainwindow` (needs `gui/` on sys.path for `tabs`)
try:
   _repo_root = Path(__file__).resolve().parents[1]
   _gui_dir = Path(__file__).resolve().parent
   for _p in (str(_repo_root), str(_gui_dir)):
      if _p not in sys.path:
         sys.path.insert(0, _p)
except Exception:
   pass

from core.factory import build_devices, load_config
from core.orchestrator import Orchestrator
from core.experiment import (
   ExperimentDefinition, Position, ChannelConfig,
   TimeLapseConfig, ZStackConfig,
)
from core.multiaxis import (
   AxisConfig,
   MultiAxisExperiment, MultiAxisRunner,
   XAxis, YAxis, ZAxis,
   ChannelAxis, DetectorAxis, RoundAxis,
)

from tabs.experiment_tab import ExperimentTab
from tabs.live_tab import LiveTab
from tabs.multiaxis_tab import MultiAxisTab
from tabs.camera_control_tab import CameraControlTab
from tabs.multiview_camera_tab import MultiViewCameraTab
from tabs.multiview_control_tab import MultiViewControlTab

# ── Saving toggle ────────────────────────────────────────────────────────────
# Set to False to completely disable HDF5/CSV saving (useful for debugging UI).
_SAVING_ENABLED = True
# ─────────────────────────────────────────────────────────────────────────────

# Robust import for StreamSaver: try local utils, then adjust sys.path
try:
   from utils.stream_saver import StreamSaver
except Exception:
   pkg_root = Path(__file__).resolve().parents[1]
   if str(pkg_root) not in sys.path:
      sys.path.insert(0, str(pkg_root))
   from utils.stream_saver import StreamSaver

try:
   from utils.image_h5_saver import ImageH5Saver
except Exception:
   pkg_root = Path(__file__).resolve().parents[1]
   if str(pkg_root) not in sys.path:
      sys.path.insert(0, str(pkg_root))
   from utils.image_h5_saver import ImageH5Saver


logger = logging.getLogger(__name__)


class MainWindow(QtWidgets.QMainWindow):
   # Thread-safe delivery of multi-axis detector samples into the GUI thread
   multiaxis_sample = QtCore.pyqtSignal(str, object, float)

   def __init__(self, config_path: str = "config/default_devices.json"):
      super().__init__()
      self.setWindowTitle("Microscope Control System")

      try:
         logger.info("MainWindow init (config=%s)", config_path)
      except Exception:
         pass

      # Paths supplied on the command line (or defaults)
      self._config_path = config_path

      # Current detector selection coming from MultiAxisTab.
      # None => no filtering (show all); set[str] => show only these ids.
      self._selected_detectors_for_display: set[str] | None = None

      self.orch_thread: threading.Thread | None = None
      self.orch: Orchestrator | None = None

      self.multi_runner: MultiAxisRunner | None = None
      self.multi_thread: threading.Thread | None = None

      # Multi-view (camera) scan runner
      self.multiview_runner: MultiAxisRunner | None = None
      self.multiview_thread: threading.Thread | None = None

      # Device tracking for multi-axis
      self.devices_built = False
      self.devices_released = True
      self.cam = None
      self.stage = None
      self.focus = None
      self.light = None
      self.fw = None
      self.det = None

      self._t0 = time.time()
      self.stream_savers: dict[str, StreamSaver] = {}
      self.image_saver: ImageH5Saver | None = None
      self._image_saver_out_dir: Path | None = None

      # Webcam preview (UI-only camera control)
      self._webcam = None
      self._webcam_live = False
      self._webcam_timer: QtCore.QTimer | None = None

      # Thread-safe camera settings snapshot (used by worker threads)
      self._camera_exposure_ms: float = 20.0

      # Multi-view camera capture toggle (do not read Qt widgets from worker threads)
      self._multiview_capture_enabled: bool = True

      # Layout persistence
      # - "original" layout: a deterministic "full" dock arrangement shipped in code
      # - "default" layout: auto-saved on every exit (and can be explicitly saved)
      self._original_layout_state: object | None = None
      self._original_layout_geometry: object | None = None
      self._build_ui()
      self._apply_full_layout()
      self._capture_original_layout()
      self._load_layout(kind="default")


   def _close_all_stream_savers(self):
      """Close and remove all active StreamSaver instances.

      This is safe to call multiple times and is used to ensure that stream
      saving stops automatically when a measurement run completes.
      """
      try:
         for saver in list(self.stream_savers.values()):
            try:
               saver.close()
            except Exception:
               pass
      finally:
         try:
            self.stream_savers.clear()
         except Exception:
            pass


   def _close_image_saver(self) -> None:
      """Close and remove the active ImageH5Saver (if any)."""
      saver = getattr(self, "image_saver", None)
      self.image_saver = None
      self._image_saver_out_dir = None
      if saver is None:
         return
      try:
         saver.close()
      except Exception:
         pass


   def _wire_view_menu_dock_sync(self):
      """Keep View menu checkmarks in sync with dock visibility.

      Users can hide/close docks via the dock 'X' button or programmatically.
      This ensures the corresponding View menu actions reflect the actual
      visibility state.
      """
      try:
         actions = getattr(self, "_view_dock_actions", None)
         if not isinstance(actions, dict):
            return

         dock_map = {
            "demo": getattr(self, "demo_dock", None),
            "multiaxis": getattr(self, "multi_dock", None),
            "multiviewctl": getattr(self, "multiviewctl_dock", None),
            "camera": getattr(self, "cam_dock", None),
            "camctl": getattr(self, "camctl_dock", None),
            "multiview": getattr(self, "multiview_dock", None),
            "detimg": getattr(self, "detimg_dock", None),
            "plot": getattr(self, "plot_dock", None),
            "detctl": getattr(self, "detctl_dock", None),
         }

         for key, dock in dock_map.items():
            act = actions.get(key)
            if dock is None or act is None:
               continue
            try:
               dock.visibilityChanged.connect(act.setChecked)
            except Exception:
               pass
            try:
               act.setChecked(bool(dock.isVisible()))
            except Exception:
               pass
      except Exception:
         pass


   def _persist_detector_scaling_to_device_config(self, scale: float, offset: float) -> None:
      """Persist detector scaling into the device config JSON.

      This treats the detector scale/offset as a hardware calibration setting
      so it is stored in `config/default_devices.json` (or the user-supplied
      `--config` file) and re-applied on next startup.
      """
      try:
         cfg = load_config(self._config_path)
      except Exception:
         return

      try:
         det_cfg = cfg.get("detector")
         if isinstance(det_cfg, list):
            for dc in det_cfg:
               if isinstance(dc, dict):
                  dc["scale"] = float(scale)
                  dc["offset"] = float(offset)
         elif isinstance(det_cfg, dict):
            det_cfg["scale"] = float(scale)
            det_cfg["offset"] = float(offset)

         # write back
         import json as _json
         from pathlib import Path as _Path
         p = _Path(self._config_path)
         p.parent.mkdir(parents=True, exist_ok=True)
         with open(p, "w") as f:
            _json.dump(cfg, f, indent=2)
      except Exception:
         return


   def _sync_view_menu_checks(self):
      """One-shot sync of View menu checkmarks from current dock state."""
      try:
         actions = getattr(self, "_view_dock_actions", None)
         if not isinstance(actions, dict):
            return
         pairs = [
            ("demo", getattr(self, "demo_dock", None)),
            ("multiaxis", getattr(self, "multi_dock", None)),
            ("multiviewctl", getattr(self, "multiviewctl_dock", None)),
            ("camera", getattr(self, "cam_dock", None)),
            ("camctl", getattr(self, "camctl_dock", None)),
            ("multiview", getattr(self, "multiview_dock", None)),
            ("detimg", getattr(self, "detimg_dock", None)),
            ("plot", getattr(self, "plot_dock", None)),
            ("detctl", getattr(self, "detctl_dock", None)),
         ]
         for key, dock in pairs:
            act = actions.get(key)
            if dock is None or act is None:
               continue
            try:
               act.setChecked(bool(dock.isVisible()))
            except Exception:
               pass
      except Exception:
         pass

   def closeEvent(self, event):
      """Persist the current layout before closing."""
      # The "default" layout is updated on every clean exit.
      self._save_layout(kind="default")
      # Stop any running experiments
      if self.orch_thread is not None:
         self._stop_experiment()
      if self.multi_thread is not None:
         self._stop_multiaxis()
      event.accept()

   # def _build_ui(self):
   #    self._create_menus()

   #    # --- Create tabs as dock widgets instead of a central tab widget ---
   #    self.demo_tab = ExperimentTab()
   #    self.multi_tab = MultiAxisTab()

   #    # Demo tab dock
   #    self.demo_dock = QtWidgets.QDockWidget("Demo", self)
   #    self.demo_dock.setObjectName("dock_demo")
   #    self.demo_dock.setWidget(self.demo_tab)
   #    self.demo_dock.setAllowedAreas(
   #       QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
   #       QtCore.Qt.DockWidgetArea.RightDockWidgetArea
   #    )
   #    self.demo_dock.setFeatures(
   #       QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
   #       QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
   #       QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
   #    )
   #    self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.demo_dock)

   #    # Multi-axis tab dock
   #    self.multi_dock = QtWidgets.QDockWidget("Multi‑Axis", self)
   #    self.multi_dock.setObjectName("dock_multiaxis")
   #    self.multi_dock.setWidget(self.multi_tab)
   #    self.multi_dock.setAllowedAreas(
   #       QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
   #       QtCore.Qt.DockWidgetArea.RightDockWidgetArea
   #    )
   #    self.multi_dock.setFeatures(
   #       QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
   #       QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
   #       QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
   #    )
   #    self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.multi_dock)

   #    # --- LiveTab instance (we will dock its subwidgets) ---
   #    self.live_tab = LiveTab()
   #    # connect hover info to status bar
   #    self.live_tab.hover_info.connect(lambda s: self.statusBar().showMessage(s))
   #    # connect stream toggle signals from live tab
   #    self.live_tab.stream_toggled.connect(self._on_stream_toggled)

   #    # Create a central placeholder widget (required for QMainWindow)
   #    central = QtWidgets.QWidget()
   #    self.setCentralWidget(central)

   #    # Create docks for live sub-panels so they are resizable, dockable and hideable
   #    try:
   #       # Camera image dock
   #       self.cam_dock = QtWidgets.QDockWidget("Camera", self)
   #       self.cam_dock.setObjectName("dock_camera")
   #       self.cam_dock.setWidget(self.live_tab.camera_panel)
   #       self.cam_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea | QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
   #       self.cam_dock.setFeatures(
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
   #       )
   #       self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.cam_dock)

   #       # Detector images dock (heatmaps)
   #       self.detimg_dock = QtWidgets.QDockWidget("Detector Images", self)
   #       self.detimg_dock.setObjectName("dock_detector_images")
   #       self.detimg_dock.setWidget(self.live_tab.detector_image_panel)
   #       self.detimg_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea | QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
   #       self.detimg_dock.setFeatures(
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
   #       )
   #       self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.detimg_dock)

   #       # Plot dock
   #       self.plot_dock = QtWidgets.QDockWidget("Plot", self)
   #       self.plot_dock.setObjectName("dock_plot")
   #       self.plot_dock.setWidget(self.live_tab.plot_panel)
   #       self.plot_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea | QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
   #       self.plot_dock.setFeatures(
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
   #       )
   #       self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.plot_dock)

   #       # Detector controls dock
   #       self.detctl_dock = QtWidgets.QDockWidget("Detectors", self)
   #       self.detctl_dock.setObjectName("dock_det_controls")
   #       self.detctl_dock.setWidget(self.live_tab.detector_control_panel)
   #       self.detctl_dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.RightDockWidgetArea | QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
   #       self.detctl_dock.setFeatures(
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
   #          QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
   #       )
   #       self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.detctl_dock)

   #       # connect view change signals so docks can be shown/hidden as view changes
   #       try:
   #          self.live_tab.view_changed.connect(self._on_live_view_changed)
   #       except Exception:
   #          pass

   #    except Exception:
   #       pass

   #    # --- Connect signals ---
   #    self.demo_tab.start_requested.connect(self._start_experiment)
   #    self.demo_tab.stop_requested.connect(self._stop_experiment)

   #    self.multi_tab.start_requested.connect(self._start_multiaxis)
   #    self.multi_tab.stop_requested.connect(self._stop_multiaxis)
   def _build_ui(self):
      self._create_menus()

      # --- Create tabs as dock widgets instead of a central tab widget ---
      self.demo_tab = ExperimentTab()
      self.multi_tab = MultiAxisTab()
      self.multiviewctl_tab = MultiViewControlTab()

      # Strip chart dock (historically called "Demo")
      self.demo_dock = QtWidgets.QDockWidget("Strip Chart", self)
      self.demo_dock.setObjectName("dock_demo")
      self.demo_dock.setWidget(self.demo_tab)
      self.demo_dock.setAllowedAreas(
         QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
         QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
         QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
         QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
      )
      self.demo_dock.setFeatures(
         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
      )
      self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.demo_dock)

      # Multi-axis tab dock
      self.multi_dock = QtWidgets.QDockWidget("Multi‑Axis", self)
      self.multi_dock.setObjectName("dock_multiaxis")
      self.multi_dock.setWidget(self.multi_tab)
      self.multi_dock.setAllowedAreas(
         QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
         QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
         QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
         QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
      )
      self.multi_dock.setFeatures(
         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
      )
      self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.multi_dock)

      # Multi-view camera control dock (scan definition for camera capture)
      self.multiviewctl_dock = QtWidgets.QDockWidget("Multi View Control", self)
      self.multiviewctl_dock.setObjectName("dock_multiview_control")
      self.multiviewctl_dock.setWidget(self.multiviewctl_tab)
      self.multiviewctl_dock.setAllowedAreas(
         QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
         QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
         QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
         QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
      )
      self.multiviewctl_dock.setFeatures(
         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
         QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
      )
      self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.multiviewctl_dock)

      # Enable tabbed docking and splitting for all docks
      self.setDockNestingEnabled(True)

      # --- LiveTab instance (we will dock its subwidgets) ---
      self.live_tab = LiveTab()
      # connect hover info to status bar
      self.live_tab.hover_info.connect(lambda s: self.statusBar().showMessage(s))
      # connect stream toggle signals from live tab
      self.live_tab.stream_toggled.connect(self._on_stream_toggled)
      # NOTE: multi-axis samples are delivered via live_tab.queue_multiaxis_sample()
      # (a thread-safe deque) rather than a Qt signal, to prevent queue overflow.

      # Create a central placeholder widget (required for QMainWindow)
      central = QtWidgets.QWidget()
      self.setCentralWidget(central)

      # Create docks for live sub-panels so they are resizable, dockable and hideable
      try:
         # Camera image dock
         self.cam_dock = QtWidgets.QDockWidget("Camera", self)
         self.cam_dock.setObjectName("dock_camera")
         self.cam_dock.setWidget(self.live_tab.camera_panel)
         self.cam_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
            QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
         )
         self.cam_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
         )
         self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.cam_dock)

         # Camera control dock (webcam preview controls)
         self.camctl_tab = CameraControlTab()
         self.camctl_dock = QtWidgets.QDockWidget("Camera Control", self)
         self.camctl_dock.setObjectName("dock_camera_control")
         self.camctl_dock.setWidget(self.camctl_tab)
         self.camctl_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
            QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
         )
         self.camctl_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
         )
         self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.camctl_dock)

         # Wire camera control signals
         try:
            self.camctl_tab.exposure_changed.connect(self._on_camera_exposure_changed)
            self.camctl_tab.frame_rate_changed.connect(self._on_camera_fps_changed)
            self.camctl_tab.snapshot_requested.connect(self._on_camera_snapshot)
            self.camctl_tab.live_toggled.connect(self._on_camera_live_toggled)
         except Exception:
            pass

         # Multi-view camera dock (shows last captured frames)
         self.multiview_tab = MultiViewCameraTab(n_views=4)
         self.multiview_dock = QtWidgets.QDockWidget("Multi View Camera", self)
         self.multiview_dock.setObjectName("dock_multi_view_camera")
         self.multiview_dock.setWidget(self.multiview_tab)
         self.multiview_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
            QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
         )
         self.multiview_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
         )
         self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.multiview_dock)

         # Keep a thread-safe copy of the toggle state (worker threads must not read Qt widgets)
         try:
            self.multiview_tab.enabled_changed.connect(self._on_multiview_enabled_changed)
            self._multiview_capture_enabled = bool(self.multiview_tab.is_enabled())
         except Exception:
            pass

         # Detector images dock (heatmaps)
         self.detimg_dock = QtWidgets.QDockWidget("Detector Images", self)
         self.detimg_dock.setObjectName("dock_detector_images")
         self.detimg_dock.setWidget(self.live_tab.detector_image_panel)
         self.detimg_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
            QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
         )
         self.detimg_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
         )
         self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.detimg_dock)

         # Plot dock
         self.plot_dock = QtWidgets.QDockWidget("Plot", self)
         self.plot_dock.setObjectName("dock_plot")
         self.plot_dock.setWidget(self.live_tab.plot_panel)
         self.plot_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
            QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
         )
         self.plot_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
         )
         self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.plot_dock)

         # Detector controls dock
         self.detctl_dock = QtWidgets.QDockWidget("Detectors", self)
         self.detctl_dock.setObjectName("dock_det_controls")
         self.detctl_dock.setWidget(self.live_tab.detector_control_panel)
         try:
            self.detctl_dock.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
            self.detctl_dock.setMaximumHeight(360)
         except Exception:
            pass
         self.detctl_dock.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
            QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
         )
         self.detctl_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
         )
         self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.detctl_dock)

         # connect view change signals so docks can be shown/hidden as view changes
         try:
            self.live_tab.view_changed.connect(self._on_live_view_changed)
         except Exception:
            pass

      except Exception:
         pass

      # --- Connect signals ---
      self.demo_tab.start_requested.connect(self._start_experiment)
      self.demo_tab.stop_requested.connect(self._stop_experiment)

      self.multi_tab.start_requested.connect(self._start_multiaxis)
      self.multi_tab.stop_requested.connect(self._stop_multiaxis)

      # Multi-view camera scan control
      self.multiviewctl_tab.start_requested.connect(self._start_multiview_scan)
      self.multiviewctl_tab.stop_requested.connect(self._stop_multiview_scan)

      # Detector selection from MultiAxisTab should drive what LiveTab shows.
      try:
         if hasattr(self.multi_tab, 'detectors_changed'):
            self.multi_tab.detectors_changed.connect(self._on_detector_selection_changed)
      except Exception:
         pass

      # Apply current selection once at startup.
      try:
         self._on_detector_selection_changed(self.multi_tab.get_selected_detectors() if hasattr(self.multi_tab, 'get_selected_detectors') else [])
      except Exception:
         pass

      # Keep View menu in sync with dock visibility changes
      self._wire_view_menu_dock_sync()

   # ----------------- webcam preview (Camera Control) -----------------

   def _ensure_webcam(self):
      if self._webcam is not None:
         return self._webcam
      try:
         from devices.webcam_camera import WebcamCamera
      except Exception as exc:
         QtWidgets.QMessageBox.warning(
            self,
            "Webcam",
            f"Webcam support requires opencv-python.\n\nError: {exc}",
         )
         return None

      try:
         cam = WebcamCamera(index=0)
         cam.connect()
      except Exception as exc:
         QtWidgets.QMessageBox.warning(self, "Webcam", f"Could not start webcam.\n\nError: {exc}")
         return None

      self._webcam = cam
      return self._webcam

   def _show_camera_view(self) -> None:
      try:
         if hasattr(self, 'live_tab'):
            self.live_tab._set_camera_view()
      except Exception:
         pass
      try:
         if hasattr(self, 'cam_dock'):
            self.cam_dock.show()
      except Exception:
         pass

   def _push_camera_frame_to_ui(self, rgb: np.ndarray) -> None:
      # rgb is HxWx3 uint8; LiveTab expects an image-like object.
      meta = {"timestamp": time.time(), "source": "webcam"}
      try:
         QtCore.QMetaObject.invokeMethod(
            self.live_tab,
            "update_image",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, rgb),
            QtCore.Q_ARG(dict, meta),
         )
      except Exception:
         # fallback: best-effort direct call (should still work in GUI thread)
         try:
            self.live_tab.update_image(rgb, meta)
         except Exception:
            pass

   def _on_camera_exposure_changed(self, ms: float) -> None:
      # Keep a thread-safe copy for worker threads (never read Qt widgets there)
      try:
         self._camera_exposure_ms = float(ms)
      except Exception:
         pass
      cam = self._ensure_webcam()
      if cam is None:
         return
      try:
         cam.set_exposure(float(ms))
      except Exception:
         pass

   def _on_multiview_enabled_changed(self, enabled: bool) -> None:
      try:
         self._multiview_capture_enabled = bool(enabled)
      except Exception:
         self._multiview_capture_enabled = True

   def _post_multiview_image(self, img: object, meta: dict) -> None:
      """Thread-safe UI update for the multi-view camera dock."""
      try:
         tab = getattr(self, "multiview_tab", None)
         if tab is None:
            return
         QtCore.QMetaObject.invokeMethod(
            tab,
            "add_image",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, img),
            QtCore.Q_ARG(dict, meta),
         )
      except Exception:
         pass

   def _capture_and_post_multiview(self, state: dict, fallback_cam) -> None:
      """Capture an image (blocking) and post it to the multi-view panel.

      Runs on the multi-axis worker thread. Never reads Qt widget state.
      """
      try:
         if not bool(getattr(self, "_multiview_capture_enabled", True)):
            return
      except Exception:
         return

      # Prefer webcam if it is already active; never create it from a worker thread.
      cam = getattr(self, "_webcam", None) or fallback_cam
      if cam is None or not hasattr(cam, "snap"):
         return

      # Apply exposure (best-effort)
      try:
         exp = float(getattr(self, "_camera_exposure_ms", 20.0))
      except Exception:
         exp = 20.0
      try:
         if hasattr(cam, "set_exposure"):
            cam.set_exposure(exp)
      except Exception:
         pass

      try:
         img = cam.snap()
      except Exception:
         return

      meta = {"experiment": "multi", "state": dict(state), "timestamp": time.time(), "source": "multiview"}
      self._post_multiview_image(img, meta)

   def _on_camera_fps_changed(self, fps: float) -> None:
      # If live is running, adjust the timer interval.
      try:
         if self._webcam_live:
            self._start_webcam_timer(float(fps))
      except Exception:
         pass

   def _on_camera_snapshot(self) -> None:
      cam = self._ensure_webcam()
      if cam is None:
         return
      self._show_camera_view()
      try:
         rgb = cam.snap()
      except Exception as exc:
         QtWidgets.QMessageBox.warning(self, "Snapshot", f"Failed to capture frame.\n\nError: {exc}")
         return
      self._push_camera_frame_to_ui(rgb)

   def _start_webcam_timer(self, fps: float | None = None) -> None:
      if self._webcam_timer is None:
         self._webcam_timer = QtCore.QTimer(self)
         self._webcam_timer.timeout.connect(self._on_webcam_timer)

      if fps is None:
         try:
            fps = float(self.camctl_tab.fps_spin.value())
         except Exception:
            fps = 10.0
      if fps <= 0:
         fps = 10.0

      interval_ms = int(max(1, round(1000.0 / float(fps))))
      try:
         self._webcam_timer.stop()
      except Exception:
         pass
      self._webcam_timer.start(interval_ms)

   def _stop_webcam_timer(self) -> None:
      try:
         if self._webcam_timer is not None:
            self._webcam_timer.stop()
      except Exception:
         pass

   def _on_camera_live_toggled(self, enabled: bool) -> None:
      if enabled:
         cam = self._ensure_webcam()
         if cam is None:
            try:
               self.camctl_tab.set_live_checked(False)
            except Exception:
               pass
            return
         self._webcam_live = True
         self._show_camera_view()
         self._start_webcam_timer()
      else:
         self._webcam_live = False
         self._stop_webcam_timer()

   def _on_webcam_timer(self) -> None:
      cam = self._ensure_webcam()
      if cam is None:
         self._stop_webcam_timer()
         return
      try:
         rgb = cam.snap()
      except Exception:
         return
      self._push_camera_frame_to_ui(rgb)
      
   # def _create_menus(self):
   #    menubar = self.menuBar()

   #    file_menu = menubar.addMenu("&File")

   #    save_exp = QAction("Save Experiment", self)
   #    load_exp = QAction("Load Experiment", self)

   #    save_exp.triggered.connect(self.save_full_experiment)
   #    load_exp.triggered.connect(self.load_full_experiment)

   #    file_menu.addAction(save_exp)
   #    file_menu.addAction(load_exp)
   #    file_menu.addSeparator()
      
   #    save_layout_action = QAction("Save Layout as Default", self)
   #    save_layout_action.triggered.connect(self._save_layout)
   #    file_menu.addAction(save_layout_action)

   #    reset_layout_action = QAction("Reset Layout to Default", self)
   #    reset_layout_action.triggered.connect(self._reset_layout)
   #    file_menu.addAction(reset_layout_action)

   #    file_menu.addSeparator()
   #    quit_action = QAction("Quit", self)
   #    quit_action.triggered.connect(self.close)
   #    file_menu.addAction(quit_action)

   #    # --- Action menu ---
   #    action_menu = menubar.addMenu("&Action")

   #    run_multiaxis_action = QAction("Run Multi‑Axis", self)
   #    run_multiaxis_action.setShortcut("Ctrl+R")
   #    run_multiaxis_action.triggered.connect(self._start_multiaxis)
   #    action_menu.addAction(run_multiaxis_action)

   #    stop_measurement_action = QAction("Stop Measurement", self)
   #    stop_measurement_action.setShortcut("Ctrl+S")
   #    stop_measurement_action.triggered.connect(self._stop_multiaxis)
   #    action_menu.addAction(stop_measurement_action)

   #    action_menu.addSeparator()

   #    run_demo_action = QAction("Run Demo Experiment", self)
   #    run_demo_action.setShortcut("Ctrl+D")
   #    run_demo_action.triggered.connect(lambda: self._start_experiment(self.demo_tab.get_config() if hasattr(self.demo_tab, 'get_config') else {}))
   #    action_menu.addAction(run_demo_action)

   #    stop_demo_action = QAction("Stop Demo Experiment", self)
   #    stop_demo_action.setShortcut("Ctrl+E")
   #    stop_demo_action.triggered.connect(self._stop_experiment)
   #    action_menu.addAction(stop_demo_action)

   #    help_menu = menubar.addMenu("&Help")
   #    about_action = QAction("About", self)
   #    about_action.triggered.connect(self._show_about)
   #    help_menu.addAction(about_action)

   #    # View menu for toggling docks
   #    view_menu = menubar.addMenu("&View")
   #    try:
   #       demo_act = QAction("Demo", self, checkable=True)
   #       demo_act.setChecked(True)
   #       demo_act.triggered.connect(lambda checked: self.demo_dock.setVisible(bool(checked)))
   #       view_menu.addAction(demo_act)

   #       multi_act = QAction("Multi‑Axis", self, checkable=True)
   #       multi_act.setChecked(True)
   #       multi_act.triggered.connect(lambda checked: self.multi_dock.setVisible(bool(checked)))
   #       view_menu.addAction(multi_act)

   #       view_menu.addSeparator()

   #       cam_act = QAction("Camera", self, checkable=True)
   #       cam_act.setChecked(True)
   #       cam_act.triggered.connect(lambda checked: self.cam_dock.setVisible(bool(checked)))
   #       view_menu.addAction(cam_act)

   #       detimg_act = QAction("Detector Images", self, checkable=True)
   #       detimg_act.setChecked(True)
   #       detimg_act.triggered.connect(lambda checked: self.detimg_dock.setVisible(bool(checked)))
   #       view_menu.addAction(detimg_act)

   #       plot_act = QAction("Plot", self, checkable=True)
   #       plot_act.setChecked(True)
   #       plot_act.triggered.connect(lambda checked: self.plot_dock.setVisible(bool(checked)))
   #       view_menu.addAction(plot_act)

   #       detctl_act = QAction("Detector Controls", self, checkable=True)
   #       detctl_act.setChecked(True)
   #       detctl_act.triggered.connect(lambda checked: self.detctl_dock.setVisible(bool(checked)))
   #       view_menu.addAction(detctl_act)
   #    except Exception:
   #       pass
   def _create_menus(self):
      menubar = self.menuBar()

      file_menu = menubar.addMenu("&File")

      save_exp = QAction("Save Experiment", self)
      load_exp = QAction("Load Experiment", self)

      save_exp.triggered.connect(self.save_full_experiment)
      load_exp.triggered.connect(self.load_full_experiment)

      file_menu.addAction(save_exp)
      file_menu.addAction(load_exp)
      file_menu.addSeparator()

      save_layout_action = QAction("Save Layout as Default", self)
      save_layout_action.triggered.connect(lambda: self._save_layout(kind="default", notify=True))
      file_menu.addAction(save_layout_action)

      save_layout_file_action = QAction("Save Layout to File…", self)
      save_layout_file_action.triggered.connect(self._save_layout_to_file)
      file_menu.addAction(save_layout_file_action)

      reset_layout_default_action = QAction("Reset Layout to Default", self)
      reset_layout_default_action.triggered.connect(self._reset_layout_to_default)
      file_menu.addAction(reset_layout_default_action)

      reset_layout_original_action = QAction("Reset Layout to Original", self)
      reset_layout_original_action.triggered.connect(self._reset_layout_to_original)
      file_menu.addAction(reset_layout_original_action)

      auto_arrange_action = QAction("Auto‑arrange Visible Panels", self)
      auto_arrange_action.triggered.connect(self._auto_arrange_visible_panels)
      file_menu.addAction(auto_arrange_action)

      file_menu.addSeparator()
      quit_action = QAction("Quit", self)
      quit_action.triggered.connect(self.close)
      file_menu.addAction(quit_action)

      # --- Action menu ---
      action_menu = menubar.addMenu("&Action")

      run_multiaxis_action = QAction("Run Multi‑Axis", self)
      run_multiaxis_action.setShortcut("Ctrl+R")
      run_multiaxis_action.triggered.connect(self._start_multiaxis)
      action_menu.addAction(run_multiaxis_action)

      stop_measurement_action = QAction("Stop Measurement", self)
      stop_measurement_action.setShortcut("Ctrl+S")
      stop_measurement_action.triggered.connect(self._stop_multiaxis)
      action_menu.addAction(stop_measurement_action)

      action_menu.addSeparator()

      run_demo_action = QAction("Run Strip Chart", self)
      run_demo_action.setShortcut("Ctrl+D")
      run_demo_action.triggered.connect(lambda: self._start_experiment(self.demo_tab.get_config() if hasattr(self.demo_tab, 'get_config') else {}))
      action_menu.addAction(run_demo_action)

      stop_demo_action = QAction("Stop Strip Chart", self)
      stop_demo_action.setShortcut("Ctrl+E")
      stop_demo_action.triggered.connect(self._stop_experiment)
      action_menu.addAction(stop_demo_action)

      help_menu = menubar.addMenu("&Help")
      about_action = QAction("About", self)
      about_action.triggered.connect(self._show_about)
      help_menu.addAction(about_action)

      # View menu for toggling docks
      view_menu = menubar.addMenu("&View")
      try:
         self._view_dock_actions = {}

         camctl_act = QAction("Camera Control", self, checkable=True)
         camctl_act.setChecked(True)
         camctl_act.triggered.connect(lambda checked: self.camctl_dock.setVisible(bool(checked)))
         view_menu.addAction(camctl_act)
         self._view_dock_actions["camctl"] = camctl_act

         cam_act = QAction("Camera", self, checkable=True)
         cam_act.setChecked(True)
         cam_act.triggered.connect(lambda checked: self.cam_dock.setVisible(bool(checked)))
         view_menu.addAction(cam_act)
         self._view_dock_actions["camera"] = cam_act

         view_menu.addSeparator()
         multiviewctl_act = QAction("Multi View Control", self, checkable=True)
         multiviewctl_act.setChecked(True)
         multiviewctl_act.triggered.connect(lambda checked: self.multiviewctl_dock.setVisible(bool(checked)))
         view_menu.addAction(multiviewctl_act)
         self._view_dock_actions["multiviewctl"] = multiviewctl_act

         multiview_act = QAction("Multi View Camera", self, checkable=True)
         multiview_act.setChecked(True)
         multiview_act.triggered.connect(lambda checked: self.multiview_dock.setVisible(bool(checked)))
         view_menu.addAction(multiview_act)
         self._view_dock_actions["multiview"] = multiview_act

         view_menu.addSeparator()

         detctl_act = QAction("Detector Controls", self, checkable=True)
         detctl_act.setChecked(True)
         detctl_act.triggered.connect(lambda checked: self.detctl_dock.setVisible(bool(checked)))
         view_menu.addAction(detctl_act)
         self._view_dock_actions["detctl"] = detctl_act
         demo_act = QAction("Strip Chart", self, checkable=True)
         demo_act.setChecked(True)
         demo_act.triggered.connect(lambda checked: self.demo_dock.setVisible(bool(checked)))
         view_menu.addAction(demo_act)
         self._view_dock_actions["demo"] = demo_act

         multi_act = QAction("Multi‑Axis", self, checkable=True)
         multi_act.setChecked(True)
         multi_act.triggered.connect(lambda checked: self.multi_dock.setVisible(bool(checked)))
         view_menu.addAction(multi_act)
         self._view_dock_actions["multiaxis"] = multi_act

         detimg_act = QAction("Detector Images", self, checkable=True)
         detimg_act.setChecked(True)
         detimg_act.triggered.connect(lambda checked: self.detimg_dock.setVisible(bool(checked)))
         view_menu.addAction(detimg_act)
         self._view_dock_actions["detimg"] = detimg_act

         plot_act = QAction("Plot", self, checkable=True)
         plot_act.setChecked(True)
         plot_act.triggered.connect(lambda checked: self.plot_dock.setVisible(bool(checked)))
         view_menu.addAction(plot_act)
         self._view_dock_actions["plot"] = plot_act

      except Exception:
         pass
   
   def _reset_layout_to_default(self):
      """Reset the current layout to the last-saved default layout (if any)."""
      restored = self._load_layout(kind="default")
      if not restored:
         # If no saved default exists, fall back to the shipped full/original layout.
         self._apply_full_layout()
         self._sync_view_menu_checks()

      QtWidgets.QMessageBox.information(
         self,
         "Layout Reset",
         "Layout reset to default.",
      )

   def _reset_layout_to_original(self):
      """Reset the current layout to the shipped (full) original layout."""
      restored = False
      try:
         if self._original_layout_geometry is not None:
            self.restoreGeometry(self._original_layout_geometry)
            restored = True
      except Exception:
         pass
      try:
         if self._original_layout_state is not None:
            self.restoreState(self._original_layout_state)
            restored = True
      except Exception:
         pass

      if not restored:
         self._apply_full_layout()
      self._sync_view_menu_checks()
      QtWidgets.QMessageBox.information(
         self,
         "Layout Reset",
         "Layout reset to original.",
      )

   def _settings(self) -> QtCore.QSettings:
      return QtCore.QSettings("MicroscopeController", "MainWindow")

   def _layout_keys(self, kind: str) -> tuple[str, str]:
      # kind is either "default" or "legacy" (migrated from older versions)
      if kind == "default":
         return ("default/geometry", "default/windowState")
      if kind == "legacy":
         return ("geometry", "windowState")
      raise ValueError(f"Unknown layout kind: {kind}")

   def _apply_full_layout(self) -> None:
      """Apply a deterministic 'full' layout with all panels docked and visible."""
      try:
         docks = [
            getattr(self, "multi_dock", None),
            getattr(self, "demo_dock", None),
            getattr(self, "multiviewctl_dock", None),
            getattr(self, "cam_dock", None),
            getattr(self, "camctl_dock", None),
            getattr(self, "multiview_dock", None),
            getattr(self, "detimg_dock", None),
            getattr(self, "plot_dock", None),
            getattr(self, "detctl_dock", None),
         ]
         docks = [d for d in docks if d is not None]
         for d in docks:
            try:
               d.setFloating(False)
            except Exception:
               pass
            try:
               d.setVisible(True)
            except Exception:
               pass

         # Left column: Multi‑Axis / Strip Chart / Multi View Control
         if getattr(self, "multi_dock", None) is not None:
            self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.multi_dock)
         if getattr(self, "demo_dock", None) is not None and getattr(self, "multi_dock", None) is not None:
            self.splitDockWidget(self.multi_dock, self.demo_dock, QtCore.Qt.Orientation.Vertical)
         if getattr(self, "multiviewctl_dock", None) is not None and getattr(self, "demo_dock", None) is not None:
            self.splitDockWidget(self.demo_dock, self.multiviewctl_dock, QtCore.Qt.Orientation.Vertical)

         # Right column: camera-related + live plots/controls (stacked)
         if getattr(self, "cam_dock", None) is not None:
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.cam_dock)
         if getattr(self, "camctl_dock", None) is not None and getattr(self, "cam_dock", None) is not None:
            self.splitDockWidget(self.cam_dock, self.camctl_dock, QtCore.Qt.Orientation.Vertical)
         if getattr(self, "multiview_dock", None) is not None and getattr(self, "camctl_dock", None) is not None:
            self.splitDockWidget(self.camctl_dock, self.multiview_dock, QtCore.Qt.Orientation.Vertical)
         if getattr(self, "detimg_dock", None) is not None and getattr(self, "multiview_dock", None) is not None:
            self.splitDockWidget(self.multiview_dock, self.detimg_dock, QtCore.Qt.Orientation.Vertical)
         if getattr(self, "plot_dock", None) is not None and getattr(self, "detimg_dock", None) is not None:
            self.splitDockWidget(self.detimg_dock, self.plot_dock, QtCore.Qt.Orientation.Vertical)
         if getattr(self, "detctl_dock", None) is not None and getattr(self, "plot_dock", None) is not None:
            self.splitDockWidget(self.plot_dock, self.detctl_dock, QtCore.Qt.Orientation.Vertical)
      except Exception:
         pass

   def _auto_arrange_visible_panels(self) -> None:
      """Arrange the currently visible docks into a sane layout.

      Heuristics:
      - Plot prefers a full-width horizontal dock at the top.
      - Control panels are stacked so they stay compact.
      - Image/preview panels go to the right.
      """
      try:
         dock_order = [
            ("plot", getattr(self, "plot_dock", None)),
            ("camera", getattr(self, "cam_dock", None)),
            ("detimg", getattr(self, "detimg_dock", None)),
            ("multiview", getattr(self, "multiview_dock", None)),
            ("multiaxis", getattr(self, "multi_dock", None)),
            ("demo", getattr(self, "demo_dock", None)),
            ("multiviewctl", getattr(self, "multiviewctl_dock", None)),
            ("camctl", getattr(self, "camctl_dock", None)),
            ("detctl", getattr(self, "detctl_dock", None)),
         ]
         visible_keys: set[str] = set()
         visible_docks: dict[str, QtWidgets.QDockWidget] = {}
         for key, dock in dock_order:
            if dock is None:
               continue
            try:
               if bool(dock.isVisible()):
                  visible_keys.add(key)
                  visible_docks[key] = dock
            except Exception:
               pass

         if not visible_docks:
            return

         # Start from a known clean base (everything docked) then re-hide the ones
         # that were not visible. This avoids accumulating weird split trees.
         self._apply_full_layout()
         for key, dock in dock_order:
            if dock is None:
               continue
            try:
               dock.setVisible(key in visible_keys)
            except Exception:
               pass
            # Ensure visible docks are docked (not floating)
            if key in visible_keys:
               try:
                  dock.setFloating(False)
               except Exception:
                  pass

         # Prefer Plot full-width at the top.
         plot = visible_docks.get("plot")
         if plot is not None:
            try:
               self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, plot)
            except Exception:
               pass

         # Right side: image/preview docks stacked.
         right_stack = [
            visible_docks.get("camera"),
            visible_docks.get("detimg"),
            visible_docks.get("multiview"),
         ]
         right_stack = [d for d in right_stack if d is not None]
         if right_stack:
            try:
               self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, right_stack[0])
            except Exception:
               pass
            for d in right_stack[1:]:
               try:
                  self.splitDockWidget(right_stack[0], d, QtCore.Qt.Orientation.Vertical)
                  right_stack[0] = d
               except Exception:
                  pass

         # Left side: stack controls & configuration docks to stay compact.
         left_stack = [
            visible_docks.get("multiaxis"),
            visible_docks.get("demo"),
            visible_docks.get("multiviewctl"),
            visible_docks.get("camctl"),
            visible_docks.get("detctl"),
         ]
         left_stack = [d for d in left_stack if d is not None]
         if left_stack:
            try:
               self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, left_stack[0])
            except Exception:
               pass
            for d in left_stack[1:]:
               try:
                  self.splitDockWidget(left_stack[0], d, QtCore.Qt.Orientation.Vertical)
                  left_stack[0] = d
               except Exception:
                  pass

         # Size heuristics: keep controls compact and give images more width.
         try:
            # Plot: prefer not to steal the whole window height.
            if plot is not None:
               self.resizeDocks([plot], [260], QtCore.Qt.Orientation.Vertical)
         except Exception:
            pass

         try:
            # Make the left (controls) column narrow, right (images) column wide.
            left_anchor = None
            for k in ("multiaxis", "demo", "multiviewctl", "camctl", "detctl"):
               if k in visible_docks:
                  left_anchor = visible_docks[k]
                  break

            right_anchor = None
            for k in ("detimg", "camera", "multiview"):
               if k in visible_docks:
                  right_anchor = visible_docks[k]
                  break

            if left_anchor is not None and right_anchor is not None:
               # ~320px control column, rest for images
               self.resizeDocks([left_anchor, right_anchor], [320, 1000], QtCore.Qt.Orientation.Horizontal)
         except Exception:
            pass

         # Keep Multi‑Axis dock compact vertically so its lists scroll.
         try:
            left_vertical_docks = []
            left_vertical_sizes = []
            # Prefer the Multi‑Axis / demo / control docks to be relatively small.
            for key, size in (
               ("multiaxis", 220),
               ("demo", 200),
               ("multiviewctl", 200),
               ("camctl", 180),
               ("detctl", 180),
            ):
               d = visible_docks.get(key)
               if d is not None:
                  left_vertical_docks.append(d)
                  left_vertical_sizes.append(size)
            if left_vertical_docks:
               self.resizeDocks(left_vertical_docks, left_vertical_sizes, QtCore.Qt.Orientation.Vertical)
         except Exception:
            pass

         self._sync_view_menu_checks()
      except Exception:
         pass

   def _capture_original_layout(self) -> None:
      """Capture the shipped original layout state for in-session resets."""
      try:
         self._original_layout_geometry = self.saveGeometry()
      except Exception:
         self._original_layout_geometry = None
      try:
         self._original_layout_state = self.saveState()
      except Exception:
         self._original_layout_state = None

   def _save_layout_to_file(self) -> None:
      """Save current layout to a user-selected JSON file."""
      try:
         default_dir = str(Path.cwd())
      except Exception:
         default_dir = ""

      path, _ = QtWidgets.QFileDialog.getSaveFileName(
         self,
         "Save Layout",
         default_dir,
         "Layout JSON (*.json)",
      )
      if not path:
         return

      try:
         geom = self.saveGeometry()
         state = self.saveState()

         # QByteArray -> bytes -> base64 for JSON serialization
         geom_b64 = base64.b64encode(bytes(geom)).decode("ascii")
         state_b64 = base64.b64encode(bytes(state)).decode("ascii")

         payload = {
            "format": "microscope_controller.layout.v1",
            "saved_at": time.time(),
            "geometry_b64": geom_b64,
            "window_state_b64": state_b64,
         }

         p = Path(path)
         p.parent.mkdir(parents=True, exist_ok=True)
         with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

         QtWidgets.QMessageBox.information(self, "Layout", f"Layout saved to:\n{p}")
      except Exception as exc:
         QtWidgets.QMessageBox.warning(self, "Layout", f"Could not save layout.\n\nError: {exc}")

   def _show_about(self):
      QtWidgets.QMessageBox.information(
            self,
            "About",
            "Microscope Controller\nPyQt6 + pyqtgraph, multi‑axis + detector visualization.",
      )

   # ----------------- classic experiment -----------------

   def _build_experiment(self, cfg: dict) -> ExperimentDefinition:
      positions = [
            Position(x=0.0, y=0.0, z=100.0, label="center"),
            Position(x=1000.0, y=0.0, z=100.0, label="right"),
      ]
      channels = [
            ChannelConfig("BF", 0, 10.0, 20.0),
            ChannelConfig("GFP", 1, 30.0, 50.0),
      ]
      timelapse = TimeLapseConfig(
            n_timepoints=cfg["n_timepoints"],
            interval_s=cfg["interval_s"],
      )
      zstack = ZStackConfig(
            start_z=cfg["z_start"],
            end_z=cfg["z_end"],
            step_z=cfg["z_step"],
      )
      return ExperimentDefinition(
            name="gui_experiment",
            positions=positions,
            channels=channels,
            timelapse=timelapse,
            zstack=zstack,
            metadata={"output_dir": cfg["output_dir"]},
      )

   def _start_experiment(self, cfg: dict):
      if self.orch_thread is not None:
            return

      try:
         logger.info("Starting strip-chart experiment (config=%s)", cfg)
      except Exception:
         pass

      cam, stage, focus, light, fw, det = build_devices(self._config_path)
      # populate available detectors in multi-axis tab
      try:
         det_ids = []
         if isinstance(det, list):
            for d in det:
               det_ids.append(getattr(d, "name", getattr(d, "port", "detector")))
         else:
            det_ids.append(getattr(det, "name", getattr(det, "port", "detector")))
         self.multi_tab.set_available_detectors(det_ids)
      except Exception:
         pass
      # IMPORTANT: detector scaling comes from the device config JSON.
      # Do not override it from the demo/experiment UI.

      self.orch = Orchestrator(
            camera=cam,
            stage=stage,
            focus=focus,
            light=light,
            filter_wheel=fw,
            detector=det,
            on_image=self._on_image,
            on_detector_sample=self._on_detector_sample,
         on_axis_event=self._on_axis_event,
      )
      self.orch.initialize()

      try:
         logger.info("Orchestrator initialized")
      except Exception:
         pass

      # start stream saver(s) for detector(s) selected in UI (if any); otherwise default to all
      try:
         out_dir = Path(cfg.get("output_dir") or Path.cwd() / "data")
         selected = self.multi_tab.get_selected_detectors() if hasattr(self.multi_tab, "get_selected_detectors") else []
         det_list = []
         if selected:
            det_list = selected
         else:
            if isinstance(det, list):
               det_list = [getattr(d, "name", getattr(d, "port", "detector")) for d in det]
            else:
               det_list = [getattr(det, "name", getattr(det, "port", "detector"))]

         # create savers for chosen ids
         for det_id in det_list:
            if _SAVING_ENABLED:
               self.stream_savers[det_id] = StreamSaver(out_dir, det_id)

         try:
            logger.info(
               "Stream saving %s (out_dir=%s detectors=%s)",
               "enabled" if _SAVING_ENABLED else "disabled",
               out_dir,
               det_list,
            )
         except Exception:
            pass

         # brief status so users know where data is written
         try:
            if _SAVING_ENABLED:
               self.statusBar().showMessage(f"Saving detector streams to: {out_dir}", 8000)
            else:
               self.statusBar().showMessage("Saving disabled (debug mode)", 4000)
         except Exception:
            pass
      except Exception:
         pass

      # Strip chart is a continuous acquisition loop (until Stop).
      # Keep the old ExperimentDefinition builder for compatibility, but do not
      # use it here.
      self._t0 = time.time()
      # Ensure the plot is in strip-chart mode (multi-axis runs repurpose/clear it).
      try:
         self.live_tab.prepare_strip_chart_plot()
      except Exception:
         try:
            self.live_tab.reset_multiaxis()
         except Exception:
            pass
      try:
         self.live_tab.reset_1d_detector()
      except Exception:
         pass

      # Apply moving window length (seconds) to the sample-buffer length.
      try:
         interval_s = float(cfg.get("interval_s", 0.05))
         window_s = float(cfg.get("window_time_s", 5.0))
         if interval_s > 0:
            n = int(max(10, min(10000, round(window_s / interval_s))))
            try:
               self.live_tab.set_window_size(n)
            except Exception:
               pass
            try:
               if hasattr(self.live_tab, "window_spin"):
                  self.live_tab.window_spin.setValue(n)
            except Exception:
               pass
      except Exception:
         pass

      # Plot is the primary output for strip chart; keep it visible.
      try:
         if hasattr(self, "plot_dock"):
            self.plot_dock.show()
      except Exception:
         pass

      def worker():
            try:
               # Start/own the run flag so Stop works.
               try:
                  self.orch._running = True
               except Exception:
                  pass

               try:
                  interval_s = float(cfg.get("interval_s", 0.05))
               except Exception:
                  interval_s = 0.05
               if interval_s <= 0:
                  interval_s = 0.05

               # Continuous loop: read detector(s) at fixed interval.
               while True:
                  try:
                     if self.orch is None:
                        break
                     if not getattr(self.orch, "_running", True):
                        break
                  except Exception:
                     # If we can't read the flag, keep running.
                     pass

                  t_start = time.time()

                  # Read detectors (if any) and forward via the existing callback.
                  try:
                     dets = getattr(self.orch, "detectors", []) or []
                  except Exception:
                     dets = []

                  if dets:
                     for d in dets:
                        try:
                           val = d.read_value() if hasattr(d, "read_value") else None
                           if val is None:
                              continue
                           det_id = getattr(d, "name", getattr(d, "port", "detector"))
                           meta = {
                              "experiment": "strip_chart",
                              "timestamp": time.time(),
                              "output_dir": str(out_dir),
                           }
                           self._on_detector_sample(str(det_id), float(val), meta)
                        except Exception:
                           continue
                  else:
                     # No detectors configured; just idle.
                     pass

                  elapsed = time.time() - t_start
                  remaining = interval_s - elapsed
                  if remaining > 0:
                     time.sleep(remaining)
            finally:
               try:
                  self.orch.shutdown()
               except Exception:
                  pass
               # When the measurement finishes, stop stream saving.
               self._close_all_stream_savers()
               self._close_image_saver()
               self.orch = None
               self.orch_thread = None

      self.orch_thread = threading.Thread(target=worker, daemon=True)
      self.orch_thread.start()

   def _on_stream_toggled(self, det_id: str, enabled: bool):
      """Create or close stream saver when user toggles streaming from the LiveTab."""
      try:
         try:
            logger.info("Stream toggle (detector=%s enabled=%s)", det_id, enabled)
         except Exception:
            pass
         if enabled:
            if _SAVING_ENABLED and det_id not in self.stream_savers:
               out_dir = Path(self.demo_tab.output_dir_edit.text() or Path.cwd() / "data")
               self.stream_savers[det_id] = StreamSaver(out_dir, det_id)
         else:
            saver = self.stream_savers.pop(det_id, None)
            if saver:
               # Closing/merging can be slow; never block the UI thread.
               def _close():
                  try:
                     saver.close()
                  except Exception:
                     pass

               threading.Thread(target=_close, daemon=True).start()
      except Exception:
         pass

   def _stop_experiment(self):
      """Stop the demo experiment and wait for shutdown/stream close.

      We avoid blocking the UI thread by polling for worker completion.
      """
      if self.orch is None:
         return

      try:
         logger.info("Stopping strip-chart experiment")
      except Exception:
         pass

      try:
         self.statusBar().showMessage("Stopping experiment… closing files…")
      except Exception:
         pass

      try:
         self.orch.stop()
      except Exception:
         pass

      # Poll for the worker thread to finish. The worker's finally-block
      # shuts down the orchestrator and closes stream savers.
      try:
         timer = QtCore.QTimer(self)
         timer.setInterval(200)

         def _check_done():
            try:
               t = self.orch_thread
               done = (t is None) or (not t.is_alive())
               if not done:
                  return
               timer.stop()
               # Ensure stream savers are closed/cleared even if the worker
               # exited abnormally.
               self._close_all_stream_savers()
               self._close_image_saver()
               try:
                  self.statusBar().showMessage("Experiment finished. Stream saved closed.", 5000)
               except Exception:
                  pass
            except Exception:
               try:
                  timer.stop()
               except Exception:
                  pass

         timer.timeout.connect(_check_done)
         timer.start()
      except Exception:
         # Fallback: do nothing; worker thread will still close savers.
         pass

   # ----------------- multi‑axis experiment -----------------

   def _start_multiaxis(self):
      if self.multi_thread is not None:
            return

      try:
         logger.info(
            "Starting multi-axis run (devices_built=%s devices_released=%s config=%s)",
            getattr(self, "devices_built", None),
            getattr(self, "devices_released", None),
            getattr(self, "_config_path", None),
         )
      except Exception:
         pass

      cfgs: list[AxisConfig] = self.multi_tab.get_axis_configs()
      if not cfgs:
            QtWidgets.QMessageBox.warning(self, "Multi‑Axis", "No axes defined.")
            return

      try:
         axis_types = [getattr(c, "axis_type", "?") for c in cfgs]
         logger.info("Multi-axis axes: n=%s types=%s", len(cfgs), axis_types)
      except Exception:
         pass

      # Build devices only if not already built or if previously released
      if not self.devices_built or self.devices_released:
         self.cam, self.stage, self.focus, self.light, self.fw, self.det = build_devices(self._config_path)
         self.devices_built = True
         self.devices_released = False

         try:
            det_count = len(self.det) if isinstance(self.det, list) else (1 if self.det is not None else 0)
            logger.info(
               "Devices built for multi-axis (camera=%s stage=%s focus=%s light=%s fw=%s detectors=%s)",
               type(self.cam).__name__ if self.cam is not None else None,
               type(self.stage).__name__ if self.stage is not None else None,
               type(self.focus).__name__ if self.focus is not None else None,
               type(self.light).__name__ if self.light is not None else None,
               type(self.fw).__name__ if self.fw is not None else None,
               det_count,
            )
         except Exception:
            pass

         # populate available detectors in multi-axis tab
         try:
            det_ids = []
            if isinstance(self.det, list):
               for d in self.det:
                  det_ids.append(getattr(d, "name", getattr(d, "port", "detector")))
            else:
               det_ids.append(getattr(self.det, "name", getattr(self.det, "port", "detector")))
            self.multi_tab.set_available_detectors(det_ids)
         except Exception:
            pass

      # Use the stored devices
      cam, stage, focus, light, fw, det = self.cam, self.stage, self.focus, self.light, self.fw, self.det
      # Build a device map for pre/post moves and axis dialogs
      device_map = {
         "stage": stage,
         "focus": focus,
         "camera": cam,
         "light": light,
         "fw": fw,
      }
      if isinstance(det, list):
         for d in det:
            device_map[getattr(d, 'name', getattr(d, 'port', 'detector'))] = d
      else:
         device_map[getattr(det, 'name', getattr(det, 'port', 'detector'))] = det

      axes = []
      for cfg in cfgs:
            t = cfg.axis_type
            p = cfg.params
            # device name -> object map for motor lookup
            device_map = {
               "stage": stage,
               "focus": focus,
               "camera": cam,
               "light": light,
               "fw": fw,
            }
            # include detectors by id if available
            if isinstance(det, list):
               for d in det:
                  device_map[getattr(d, "name", getattr(d, "port", "detector"))] = d
            else:
               device_map[getattr(det, "name", getattr(det, "port", "detector"))] = det

            if t == "X":
               motor_devices = [device_map.get(n) for n in p.get("motors", []) if device_map.get(n) is not None]
               axes.append(
                  XAxis(
                     stage,
                     p["start"],
                     p["end"],
                     p["step"],
                     motor_devices=motor_devices or None,
                     motor_mode=p.get("motor_mode", "sequential"),
                     wait_s=p.get("wait", 0.0),
                     sync_timeout=p.get("sync_timeout", 5.0),
                     sync_poll=p.get("sync_poll", 0.01),
                     sync_tol=p.get("sync_tol", 1e-3),
                  )
               )
            elif t == "Y":
               motor_devices = [device_map.get(n) for n in p.get("motors", []) if device_map.get(n) is not None]
               axes.append(
                  YAxis(
                     stage,
                     p["start"],
                     p["end"],
                     p["step"],
                     motor_devices=motor_devices or None,
                     motor_mode=p.get("motor_mode", "sequential"),
                     wait_s=p.get("wait", 0.0),
                     sync_timeout=p.get("sync_timeout", 5.0),
                     sync_poll=p.get("sync_poll", 0.01),
                     sync_tol=p.get("sync_tol", 1e-3),
                  )
               )
            elif t == "Z":
               motor_devices = [device_map.get(n) for n in p.get("motors", []) if device_map.get(n) is not None]
               axes.append(
                  ZAxis(
                     focus,
                     p["start"],
                     p["end"],
                     p["step"],
                     motor_devices=motor_devices or None,
                     motor_mode=p.get("motor_mode", "sequential"),
                     wait_s=p.get("wait", 0.0),
                     sync_timeout=p.get("sync_timeout", 5.0),
                     sync_poll=p.get("sync_poll", 0.01),
                     sync_tol=p.get("sync_tol", 1e-3),
                  )
               )
            elif t == "Channel":
               axes.append(ChannelAxis(cam, light, fw, p["channels"], p.get("wait", 0.0)))
            elif t == "Detector":
               # Detector scaling is read from device config; do not override it via an axis.
               axes.append(DetectorAxis(det, scales=None, wait_s=p.get("wait", 0.0)))
            elif t == "Round":
               axes.append(RoundAxis(p["n_rounds"]))

      self.live_tab.reset_multiaxis()

      # Apply the Multi-Axis tab's default x-axis preference to the Live plot.
      # The Live plot will apply this as soon as the first multi-axis samples
      # arrive and x-axis options are refreshed.
      try:
         if hasattr(self.multi_tab, "get_default_xaxis") and hasattr(self.live_tab, "set_preferred_plot_xaxis"):
            self.live_tab.set_preferred_plot_xaxis(self.multi_tab.get_default_xaxis())
      except Exception:
         pass

      # Also clear strip-chart buffers so stale traces don't remain visible
      # while the multi-axis run is starting (before the first multi-axis samples arrive).
      try:
         self.live_tab.reset_1d_detector()
      except Exception:
         pass
      try:
         if hasattr(self.live_tab, "_clear_plot_and_legend"):
            self.live_tab._clear_plot_and_legend()
      except Exception:
         pass

      # Execute pre-scan positions defined in axis configs
      try:
         for cfg in cfgs:
            p = cfg.params
            # only motor-like axes support pre_pos/post_pos
            if p and 'pre_pos' in p and p['pre_pos'] is not None:
               # determine device to move for this axis
               if cfg.axis_type == 'X' or cfg.axis_type == 'Y':
                  dev = device_map.get('stage')
                  if dev and hasattr(dev, 'move_to'):
                     # get current complementary coord
                     try:
                        cur = dev.get_position()
                        if isinstance(cur, tuple):
                           if cfg.axis_type == 'X':
                              try:
                                 logger.info("Pre-position stage X -> %s (keeping Y=%s)", p['pre_pos'], cur[1])
                              except Exception:
                                 pass
                              dev.move_to(p['pre_pos'], cur[1])
                           else:
                              try:
                                 logger.info("Pre-position stage Y -> %s (keeping X=%s)", p['pre_pos'], cur[0])
                              except Exception:
                                 pass
                              dev.move_to(cur[0], p['pre_pos'])
                        else:
                           try:
                              logger.info("Pre-position stage %s -> %s", cfg.axis_type, p['pre_pos'])
                           except Exception:
                              pass
                           dev.move_to(p['pre_pos'])
                     except Exception:
                        pass
               elif cfg.axis_type == 'Z':
                  dev = device_map.get('focus')
                  if dev and hasattr(dev, 'move_to'):
                     try:
                        try:
                           logger.info("Pre-position focus Z -> %s", p['pre_pos'])
                        except Exception:
                           pass
                        dev.move_to(p['pre_pos'])
                     except Exception:
                        pass
      except Exception:
         pass

      # register detector views in LiveTab and start stream saver(s) for detector(s) selected in UI
      # (if any); otherwise default to all
      try:
         out_dir = Path(self.demo_tab.output_dir_edit.text() or Path.cwd() / "data")
         selected = self.multi_tab.get_selected_detectors() if hasattr(self.multi_tab, "get_selected_detectors") else []
         det_list = []
         if selected:
            det_list = selected
         else:
            if isinstance(det, list):
               det_list = [getattr(d, "name", getattr(d, "port", "detector")) for d in det]
            else:
               det_list = [getattr(det, "name", getattr(det, "port", "detector"))]

         # create savers for chosen ids (do not overwrite existing savers)
         for det_id in det_list:
            # ensure live tab knows about this detector (create image view and controls)
            try:
               self.live_tab.register_detector(det_id)
            except Exception:
               pass
            if _SAVING_ENABLED and det_id not in self.stream_savers:
               self.stream_savers[det_id] = StreamSaver(out_dir, det_id)

         # brief status so users know where data is written
         try:
            if _SAVING_ENABLED:
               self.statusBar().showMessage(f"Saving detector streams to: {out_dir}", 8000)
            else:
               self.statusBar().showMessage("Saving disabled (debug mode)", 4000)
         except Exception:
            pass
         # apply default X-axis selection from MultiAxisTab (if provided)
         try:
            default_x = self.multi_tab.get_default_xaxis() if hasattr(self.multi_tab, 'get_default_xaxis') else None
            if default_x and hasattr(self.live_tab, 'set_xaxis'):
               try:
                  self.live_tab.set_xaxis(default_x)
               except Exception:
                  pass
         except Exception:
            pass
      except Exception:
         pass

      def measure(state: dict):
            # camera image if Channel present
            if "Channel" in state:
               img = cam.snap()
               meta = {"experiment": "multi", "state": state, "timestamp": time.time()}
               self._on_image(img, meta)

               # Also feed the multi-view panel (no additional capture)
               try:
                  self._post_multiview_image(img, dict(meta))
               except Exception:
                  pass
            else:
               # No Channel axis: optionally capture a frame per motor state for the
               # multi-view panel. This blocks the worker thread on snap(), keeping
               # motor stepping and camera exposure in lockstep.
               try:
                  self._capture_and_post_multiview(state, cam)
               except Exception:
                  pass

            # detector value(s)
            if det is not None:
               dets = det if isinstance(det, list) else [det]
               for d in dets:
                  try:
                     val = d.read_value()
                     det_id = getattr(d, "name", getattr(d, "port", "detector"))

                     # Apply current detector selection for display/visualization.
                     try:
                        allowed = getattr(self, '_selected_detectors_for_display', None)
                        if allowed is not None and str(det_id) not in allowed:
                           continue
                     except Exception:
                        pass
                     
                     # Thread-safe GUI update: push directly to live_tab's deque
                     # (avoids Qt signal queue overflow on fast scans)
                     try:
                        self.live_tab.queue_multiaxis_sample(str(det_id), dict(state), float(val))
                     except Exception:
                        pass
                     # stream-save if enabled per detector id
                     try:
                        saver = self.stream_savers.get(det_id)
                        if saver:
                           saver.append_sample(time.time(), float(val), meta=state)
                     except Exception:
                        pass
                  except Exception:
                     continue

      exp = MultiAxisExperiment(axes=axes, measure=measure)
      # pass on_move callback so we can persist axis move events (X/Y/Z/Channel/Detector/...)
      self.multi_runner = MultiAxisRunner(exp, on_move=self._on_axis_move)

      def worker():
            try:
               try:
                  logger.info("Multi-axis worker started")
               except Exception:
                  pass
               self.multi_runner.run()
               try:
                  logger.info("Multi-axis worker finished")
               except Exception:
                  pass
            except Exception:
               try:
                  logger.exception("Multi-axis worker crashed")
               except Exception:
                  pass
            finally:
               # When the measurement finishes, stop stream saving.
               self._close_all_stream_savers()
               self.multi_runner = None
               self.multi_thread = None

      self.multi_thread = threading.Thread(target=worker, daemon=True)
      self.multi_thread.start()

   def _stop_multiaxis(self):
      try:
         logger.info("Stopping multi-axis run")
      except Exception:
         pass

      if self.multi_runner is not None:
            self.multi_runner.stop()

      # Release devices when stopping
      if self.devices_built and not self.devices_released:
         for dev in [self.cam, self.stage, self.focus, self.light, self.fw, self.det]:
               if dev and hasattr(dev, 'disconnect'):
                  try:
                     logger.info("Disconnecting device: %s", type(dev).__name__)
                  except Exception:
                     pass
                  try:
                     dev.disconnect()
                  except Exception:
                     try:
                        logger.exception("Device disconnect failed: %s", type(dev).__name__)
                     except Exception:
                        pass
         self.devices_released = True

      try:
         logger.info("Multi-axis stopped (devices_released=%s)", getattr(self, "devices_released", None))
      except Exception:
         pass
      # close any stream savers
      self._close_all_stream_savers()
      self._close_image_saver()

   # ----------------- multi-view camera scan -----------------

   def _start_multiview_scan(self) -> None:
      if self.multiview_thread is not None:
         return

      try:
         logger.info(
            "Starting multiview scan (devices_built=%s devices_released=%s config=%s)",
            getattr(self, "devices_built", None),
            getattr(self, "devices_released", None),
            getattr(self, "_config_path", None),
         )
      except Exception:
         pass

      try:
         cfgs: list[AxisConfig] = self.multiviewctl_tab.get_axis_configs()
      except Exception:
         cfgs = []

      if not cfgs:
         QtWidgets.QMessageBox.warning(self, "Multi View", "No axes defined.")
         return

      try:
         axis_types = [getattr(c, "axis_type", "?") for c in cfgs]
         logger.info("Multiview axes: n=%s types=%s", len(cfgs), axis_types)
      except Exception:
         pass

      # Build devices only if not already built or if previously released.
      if not self.devices_built or self.devices_released:
         self.cam, self.stage, self.focus, self.light, self.fw, self.det = build_devices(self._config_path)
         self.devices_built = True
         self.devices_released = False

         try:
            logger.info(
               "Devices built for multiview (camera=%s stage=%s focus=%s light=%s fw=%s)",
               type(self.cam).__name__ if self.cam is not None else None,
               type(self.stage).__name__ if self.stage is not None else None,
               type(self.focus).__name__ if self.focus is not None else None,
               type(self.light).__name__ if self.light is not None else None,
               type(self.fw).__name__ if self.fw is not None else None,
            )
         except Exception:
            pass

      cam, stage, focus, light, fw = self.cam, self.stage, self.focus, self.light, self.fw

      axes = []
      for cfg in cfgs:
         t = cfg.axis_type
         p = cfg.params

         # device name -> object map for motor lookup
         device_map = {
            "stage": stage,
            "focus": focus,
            "camera": cam,
            "light": light,
            "fw": fw,
         }

         if t == "X":
            motor_devices = [device_map.get(n) for n in p.get("motors", []) if device_map.get(n) is not None]
            axes.append(
               XAxis(
                  stage,
                  p["start"],
                  p["end"],
                  p["step"],
                  motor_devices=motor_devices or None,
                  motor_mode=p.get("motor_mode", "sequential"),
                  wait_s=p.get("wait", 0.0),
                  sync_timeout=p.get("sync_timeout", 5.0),
                  sync_poll=p.get("sync_poll", 0.01),
                  sync_tol=p.get("sync_tol", 1e-3),
               )
            )
         elif t == "Y":
            motor_devices = [device_map.get(n) for n in p.get("motors", []) if device_map.get(n) is not None]
            axes.append(
               YAxis(
                  stage,
                  p["start"],
                  p["end"],
                  p["step"],
                  motor_devices=motor_devices or None,
                  motor_mode=p.get("motor_mode", "sequential"),
                  wait_s=p.get("wait", 0.0),
                  sync_timeout=p.get("sync_timeout", 5.0),
                  sync_poll=p.get("sync_poll", 0.01),
                  sync_tol=p.get("sync_tol", 1e-3),
               )
            )
         elif t == "Z":
            motor_devices = [device_map.get(n) for n in p.get("motors", []) if device_map.get(n) is not None]
            axes.append(
               ZAxis(
                  focus,
                  p["start"],
                  p["end"],
                  p["step"],
                  motor_devices=motor_devices or None,
                  motor_mode=p.get("motor_mode", "sequential"),
                  wait_s=p.get("wait", 0.0),
                  sync_timeout=p.get("sync_timeout", 5.0),
                  sync_poll=p.get("sync_poll", 0.01),
                  sync_tol=p.get("sync_tol", 1e-3),
               )
            )
         elif t == "Channel":
            axes.append(ChannelAxis(cam, light, fw, p["channels"], p.get("wait", 0.0)))
         elif t == "Round":
            axes.append(RoundAxis(p["n_rounds"]))

      # Clear the multiview grid when starting.
      try:
         self.multiview_tab.clear()
      except Exception:
         pass

      def measure(state: dict):
         # Respect the viewer toggle.
         try:
            if not bool(getattr(self, "_multiview_capture_enabled", True)):
               return
         except Exception:
            return

         # If there is no Channel axis in the state, apply the camera-control exposure.
         if "Channel" not in state:
            try:
               exp = float(getattr(self, "_camera_exposure_ms", 20.0))
               if hasattr(cam, "set_exposure"):
                  cam.set_exposure(exp)
            except Exception:
               pass

         # Best-effort illumination gating (only if methods exist)
         try:
            if hasattr(light, "on"):
               light.on()
         except Exception:
            pass

         try:
            img = cam.snap()
         except Exception:
            return
         finally:
            try:
               if hasattr(light, "off"):
                  light.off()
            except Exception:
               pass

         meta = {"experiment": "multiview", "state": dict(state), "timestamp": time.time()}
         # Push to the multi-view grid
         try:
            self._post_multiview_image(img, dict(meta))
         except Exception:
            pass
         # Also show in the main Camera dock
         try:
            self._push_camera_frame_to_ui(img)
         except Exception:
            pass

      exp = MultiAxisExperiment(axes=axes, measure=measure)
      self.multiview_runner = MultiAxisRunner(exp, on_move=None)

      def worker():
         try:
            try:
               logger.info("Multiview worker started")
            except Exception:
               pass
            self.multiview_runner.run()
            try:
               logger.info("Multiview worker finished")
            except Exception:
               pass
         except Exception:
            try:
               logger.exception("Multiview worker crashed")
            except Exception:
               pass
         finally:
            self.multiview_runner = None
            self.multiview_thread = None

      self.multiview_thread = threading.Thread(target=worker, daemon=True)
      self.multiview_thread.start()

   def _stop_multiview_scan(self) -> None:
      try:
         logger.info("Stopping multiview scan")
      except Exception:
         pass

      if self.multiview_runner is not None:
         try:
            self.multiview_runner.stop()
         except Exception:
            pass

      try:
         logger.info("Multiview stop requested")
      except Exception:
         pass

   def _on_live_view_changed(self, mode: str):
      try:
         if mode == 'camera':
            if hasattr(self, 'cam_dock'):
               self.cam_dock.show()
            if hasattr(self, 'detimg_dock'):
               self.detimg_dock.hide()
            if hasattr(self, 'plot_dock'):
               self.plot_dock.hide()
         elif mode == 'detector':
            if hasattr(self, 'cam_dock'):
               self.cam_dock.hide()
            if hasattr(self, 'detimg_dock'):
               self.detimg_dock.show()
            if hasattr(self, 'plot_dock'):
               self.plot_dock.show()
      except Exception:
         pass

   def _append_event_to_stream_savers(self, event: str, payload: dict):
      """Append an axis/motor event record to all active stream savers.

      Events are recorded in two ways:
      - As a NaN-valued sample in the numeric stream (timeline alignment)
      - As a JSON record in the HDF5 'events' dataset (full fidelity)
      """
      try:
         ts = float(payload.get("timestamp", time.time())) if isinstance(payload, dict) else time.time()
      except Exception:
         ts = time.time()

      meta = {"event": event, "timestamp": ts, "payload": payload}
      for saver in list(self.stream_savers.values()):
         try:
            try:
               if hasattr(saver, "append_event"):
                  saver.append_event(meta)
            except Exception:
               pass
            saver.append_sample(ts, float('nan'), meta=meta)
         except Exception:
            continue

   def _on_axis_event(self, event: str, payload: dict):
      """Receive axis events from the classic Orchestrator."""
      try:
         if not isinstance(payload, dict):
            payload = {"value": payload}

         # Log axis/motion events for traceability.
         try:
            s = repr(payload)
            if len(s) > 2000:
               s = s[:2000] + "…"
            logger.info("Axis event: %s payload=%s", event, s)
         except Exception:
            pass

         self._append_event_to_stream_savers(event, payload)
      except Exception:
         pass

   def _on_axis_move(self, axis_name: str, pos: object, state: dict):
      """Called when an axis apply completes during a multi-axis run.

      NOTE: This runs on the worker thread — never touch Qt widgets directly here.
      """
      try:
         ts = time.time()
         payload = {
            "timestamp": ts,
            "axis": axis_name,
            "pos": pos,
            "state": state,
         }

         # Log every move so hardware actions are traceable from the log file.
         try:
            st = state if isinstance(state, dict) else {"state": state}
            st_s = repr(st)
            if len(st_s) > 2000:
               st_s = st_s[:2000] + "…"
            logger.info("Axis move: axis=%s pos=%r state=%s", axis_name, pos, st_s)
         except Exception:
            pass

         self._append_event_to_stream_savers("axis_move", payload)

         # Post a status bar update to the GUI thread (never call Qt from worker thread).
         try:
            if isinstance(state, dict) and axis_name in ("X", "Y", "Z"):
               x = state.get('X') or state.get('x')
               y = state.get('Y') or state.get('y')
               z = state.get('Z') or state.get('z')
               parts = []
               if x is not None:
                  parts.append(f"x={float(x):.3f}")
               if y is not None:
                  parts.append(f"y={float(y):.3f}")
               if z is not None:
                  parts.append(f"z={float(z):.3f}")
               if parts:
                  msg = "Stage: " + " ".join(parts)
                  QtCore.QMetaObject.invokeMethod(
                     self.statusBar(), "showMessage",
                     QtCore.Qt.ConnectionType.QueuedConnection,
                     QtCore.Q_ARG(str, msg),
                  )
         except Exception:
            pass
      except Exception:
         pass

   # ----------------- callbacks + saving -----------------

   def _on_image(self, img, meta: dict):
      QtCore.QMetaObject.invokeMethod(
            self.live_tab,
            "update_image",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, img),
            QtCore.Q_ARG(dict, meta),
      )
      self._save_image(img, meta)

   def _save_image(self, img, meta: dict):
      # Convert non-JSON objects (e.g., ChannelConfig) to dicts
      def convert(obj):
         if hasattr(obj, "__dict__"):
               return obj.__dict__
         return obj

      safe_meta = {}
      for k, v in meta.items():
         if isinstance(v, dict):
               safe_meta[k] = {kk: convert(vv) for kk, vv in v.items()}
         else:
               safe_meta[k] = convert(v)

      out_dir = None
      # prefer explicit output_dir in meta, otherwise fall back to Demo tab setting
      if isinstance(meta, dict) and meta.get("output_dir"):
         out_dir = Path(meta.get("output_dir"))
      else:
         try:
            out_dir = Path(self.demo_tab.output_dir_edit.text() or Path.cwd() / "data")
         except Exception:
            out_dir = Path.cwd() / "data"
      out_dir.mkdir(parents=True, exist_ok=True)

      if not _SAVING_ENABLED:
         return

      # Lazily create one HDF5 file per run/output_dir.
      try:
         need_new = (self.image_saver is None) or (self._image_saver_out_dir != out_dir)
      except Exception:
         need_new = True

      if need_new:
         try:
            self._close_image_saver()
         except Exception:
            pass
         try:
            exp = None
            if isinstance(meta, dict):
               exp = meta.get("experiment")
            exp = str(exp or "camera")
            ts = time.strftime("%Y%m%d_%H%M%S")
            base_name = f"{ts}__camera__{exp}"
            self.image_saver = ImageH5Saver(out_dir, base_name=base_name, flush_every=1)
            self._image_saver_out_dir = out_dir
         except Exception:
            # If the HDF5 writer cannot be created, silently fall back to no-op.
            self.image_saver = None
            self._image_saver_out_dir = None

      saver = getattr(self, "image_saver", None)
      if saver is None:
         return
      try:
         saver.append_image(img, safe_meta)
      except Exception as e:
         # Avoid taking down acquisition for saving problems.
         # If frame shape changes mid-run, roll over to a new file and retry once.
         try:
            if isinstance(e, ValueError) and "Image shape changed" in str(e):
               try:
                  self._close_image_saver()
               except Exception:
                  pass
               try:
                  exp = None
                  if isinstance(meta, dict):
                     exp = meta.get("experiment")
                  exp = str(exp or "camera")
                  ts = time.strftime("%Y%m%d_%H%M%S")
                  base_name = f"{ts}__camera__{exp}"
                  self.image_saver = ImageH5Saver(out_dir, base_name=base_name, flush_every=1)
                  self._image_saver_out_dir = out_dir
                  self.image_saver.append_image(img, safe_meta)
                  return
               except Exception:
                  pass
         except Exception:
            pass

   def _on_detector_sample(self, *args):
      """Accept either (value, meta) or (detector_id, value, meta)."""
      try:
         if len(args) == 2:
            value, meta = args
            # Some sources do not provide a detector id. If exactly one detector
            # is available in the UI, use that id so selection/filtering works.
            det_id = None
            try:
               if hasattr(self, 'multi_tab') and hasattr(self.multi_tab, 'detector_list'):
                  if self.multi_tab.detector_list.count() == 1:
                     it = self.multi_tab.detector_list.item(0)
                     det_id = it.data(QtCore.Qt.ItemDataRole.UserRole)
            except Exception:
               det_id = None
            if not det_id:
               det_id = "detector"
         elif len(args) == 3:
            det_id, value, meta = args
         else:
            return

         # Apply detector selection filtering for display.
         allowed = getattr(self, '_selected_detectors_for_display', None)
         if allowed is not None and det_id not in allowed:
            # If upstream didn't provide an id and only one detector is selected,
            # map the generic id to that detector.
            if det_id == 'detector' and len(allowed) == 1:
               try:
                  det_id = next(iter(allowed))
               except Exception:
                  return
            else:
               return

         timestamp = meta.get("timestamp", time.time()) if isinstance(meta, dict) else time.time()
         # forward to live tab (queued)
         QtCore.QMetaObject.invokeMethod(
               self.live_tab,
               "add_detector_sample_qt",
               QtCore.Qt.ConnectionType.QueuedConnection,
               QtCore.Q_ARG(str, det_id),
               QtCore.Q_ARG(float, float(value)),
               QtCore.Q_ARG(float, float(timestamp)),
         )
         # stream-save if enabled
         try:
            saver = self.stream_savers.get(det_id)
            if saver:
               saver.append_sample(timestamp, float(value), meta=meta)
         except Exception:
            pass
      except Exception:
         return

   def _on_detector_selection_changed(self, detector_ids: list[str]):
      """Update which detectors are shown in LiveTab based on MultiAxisTab."""
      try:
         wanted = set(detector_ids or [])
         self._selected_detectors_for_display = wanted if wanted else None
      except Exception:
         self._selected_detectors_for_display = None

      try:
         if hasattr(self, 'live_tab') and hasattr(self.live_tab, 'set_selected_detectors'):
            self.live_tab.set_selected_detectors(detector_ids)
      except Exception:
         pass

   def save_full_experiment(self):
      path, _ = QtWidgets.QFileDialog.getSaveFileName(
         self, "Save Experiment", "", "Experiment (*.json)"
      )
      if not path:
         return

      # --- Demo experiment ---
      demo_cfg = {
         "mode": self.demo_tab.mode_combo.currentText(),
         "n_timepoints": self.demo_tab.n_timepoints_spin.value(),
         "interval_s": self.demo_tab.interval_spin.value(),
         "z_start": self.demo_tab.z_start_spin.value(),
         "z_end": self.demo_tab.z_end_spin.value(),
         "z_step": self.demo_tab.z_step_spin.value(),
         "output_dir": self.demo_tab.output_dir_edit.text(),
         # Legacy keys: scaling is defined in config/default_devices.json.
         "det_scale": 1.0,
         "det_offset": 0.0,
      }

      # --- Multi-axis ---
      axes = []
      for cfg in self.multi_tab.get_axis_configs():
         axes.append({
               "axis_type": cfg.axis_type,
               "params": cfg.params
         })

      # Persist selected detectors; semantics: empty list means "all detectors"
      try:
         selected_detectors = self.multi_tab.get_selected_detectors()
      except Exception:
         selected_detectors = []

      data = {
         "demo": demo_cfg,
         "multiaxis": {"axes": axes, "detectors": selected_detectors},
         "output_dir": demo_cfg["output_dir"],
         "devices": {
               "detector_scale": demo_cfg["det_scale"],
               "detector_offset": demo_cfg["det_offset"]
         }
      }

      with open(path, "w") as f:
         json.dump(data, f, indent=2)

   def load_full_experiment(self):
      path, _ = QtWidgets.QFileDialog.getOpenFileName(
         self, "Load Experiment", "", "Experiment (*.json)"
      )
      if not path:
         return

      with open(path) as f:
         data = json.load(f)

      # --- Restore Demo tab ---
      demo = data["demo"]
      self.demo_tab.mode_combo.setCurrentText(demo["mode"])
      self.demo_tab.n_timepoints_spin.setValue(demo["n_timepoints"])
      self.demo_tab.interval_spin.setValue(demo["interval_s"])
      self.demo_tab.z_start_spin.setValue(demo["z_start"])
      self.demo_tab.z_end_spin.setValue(demo["z_end"])
      self.demo_tab.z_step_spin.setValue(demo["z_step"])
      self.demo_tab.output_dir_edit.setText(demo["output_dir"])
      # det_scale/det_offset are ignored (scaling comes from device config)

      # --- Restore Multi-axis tab ---
      self.multi_tab.axis_list.clear()
      for cfg in data["multiaxis"]["axes"]:
         axis_cfg = AxisConfig(cfg["axis_type"], cfg["params"])
         item = QtWidgets.QListWidgetItem(axis_cfg.label())
         item.setData(QtCore.Qt.ItemDataRole.UserRole, axis_cfg)
         self.multi_tab.axis_list.addItem(item)

      # Refresh default x-axis selector based on loaded axes
      try:
         self.multi_tab.refresh_default_xaxis_options()
      except Exception:
         pass

      # Restore detector list + selection from config and experiment
      try:
         cfg = load_config(self._config_path)
         det_cfg = cfg.get("detector", [])
         available: list[str] = []
         if isinstance(det_cfg, list):
            for i, dc in enumerate(det_cfg):
               if isinstance(dc, dict):
                  available.append(dc.get("name") or dc.get("port") or f"detector{i + 1}")
         elif isinstance(det_cfg, dict):
            available.append(det_cfg.get("name") or det_cfg.get("port") or "detector")

         if available:
            self.multi_tab.set_available_detectors(available)

         selected = []
         try:
            selected = (data.get("multiaxis") or {}).get("detectors") or []
         except Exception:
            selected = []

         # Empty list means "all detectors" (matches run behavior)
         if (not selected) and available:
            selected = list(available)

         try:
            if hasattr(self.multi_tab, "set_selected_detectors"):
               self.multi_tab.set_selected_detectors(selected)
         except Exception:
            pass

         # Ensure Live display controls exist for selected detectors
         try:
            for det_id in selected:
               self.live_tab.register_detector(det_id)
         except Exception:
            pass
      except Exception:
         pass
   def _save_layout(self, kind: str = "default", notify: bool = False) -> None:
      """Save window geometry and dock layout to settings."""
      try:
         settings = self._settings()
         g_key, s_key = self._layout_keys(kind)
         settings.setValue(g_key, self.saveGeometry())
         settings.setValue(s_key, self.saveState())

         # Back-compat: keep writing legacy keys too so older versions can read it.
         if kind == "default":
            try:
               lg_key, ls_key = self._layout_keys("legacy")
               settings.setValue(lg_key, settings.value(g_key))
               settings.setValue(ls_key, settings.value(s_key))
            except Exception:
               pass
      except Exception:
         return

      if notify:
         try:
            QtWidgets.QMessageBox.information(self, "Layout", "Default layout saved.")
         except Exception:
            pass

   def load_full_experiment_from_file(self, path: str):
      """Load experiment settings from a JSON file without showing a dialog.

      Used when the path is supplied on the command line.
      """
      try:
         with open(path) as f:
            data = json.load(f)
      except Exception as exc:
         QtWidgets.QMessageBox.warning(self, "Load Experiment", f"Could not read {path}:\n{exc}")
         return

      # Reuse the same restore logic as the interactive dialog.
      # Patch a minimal file-dialog-free path by borrowing the body of
      # load_full_experiment after the file-open step.
      try:
         demo = data["demo"]
         self.demo_tab.mode_combo.setCurrentText(demo["mode"])
         self.demo_tab.n_timepoints_spin.setValue(demo["n_timepoints"])
         self.demo_tab.interval_spin.setValue(demo["interval_s"])
         self.demo_tab.z_start_spin.setValue(demo["z_start"])
         self.demo_tab.z_end_spin.setValue(demo["z_end"])
         self.demo_tab.z_step_spin.setValue(demo["z_step"])
         self.demo_tab.output_dir_edit.setText(demo["output_dir"])
         # det_scale/det_offset are ignored (scaling comes from device config)
      except Exception:
         pass

      try:
         self.multi_tab.axis_list.clear()
         for cfg in data["multiaxis"]["axes"]:
            axis_cfg = AxisConfig(cfg["axis_type"], cfg["params"])
            item = QtWidgets.QListWidgetItem(axis_cfg.label())
            item.setData(QtCore.Qt.ItemDataRole.UserRole, axis_cfg)
            self.multi_tab.axis_list.addItem(item)
         try:
            self.multi_tab.refresh_default_xaxis_options()
         except Exception:
            pass
      except Exception:
         pass

      try:
         cfg = load_config(self._config_path)
         det_cfg = cfg.get("detector", [])
         available: list[str] = []
         if isinstance(det_cfg, list):
            for i, dc in enumerate(det_cfg):
               if isinstance(dc, dict):
                  available.append(dc.get("name") or dc.get("port") or f"detector{i + 1}")
         elif isinstance(det_cfg, dict):
            available.append(det_cfg.get("name") or det_cfg.get("port") or "detector")
         if available:
            self.multi_tab.set_available_detectors(available)
         selected = (data.get("multiaxis") or {}).get("detectors") or []
         if (not selected) and available:
            selected = list(available)
         if hasattr(self.multi_tab, "set_selected_detectors"):
            self.multi_tab.set_selected_detectors(selected)
         for det_id in selected:
            self.live_tab.register_detector(det_id)
      except Exception:
         pass

   def _load_layout(self, kind: str = "default") -> bool:
      """Restore window geometry and dock layout from settings.

      Returns True if a saved layout was found and applied.
      """
      settings = self._settings()

      # Migration: if this is a newer build looking for "default" and only legacy keys exist,
      # promote legacy to default exactly once.
      if kind == "default":
         try:
            dg_key, ds_key = self._layout_keys("default")
            lg_key, ls_key = self._layout_keys("legacy")
            if (settings.value(dg_key) is None and settings.value(ds_key) is None) and (
               settings.value(lg_key) is not None or settings.value(ls_key) is not None
            ):
               if settings.value(lg_key) is not None:
                  settings.setValue(dg_key, settings.value(lg_key))
               if settings.value(ls_key) is not None:
                  settings.setValue(ds_key, settings.value(ls_key))
         except Exception:
            pass

      g_key, s_key = self._layout_keys(kind)
      geometry = settings.value(g_key)
      window_state = settings.value(s_key)

      applied = False
      try:
         if geometry is not None:
            self.restoreGeometry(geometry)
            applied = True
      except Exception:
         pass
      try:
         if window_state is not None:
            self.restoreState(window_state)
            applied = True
      except Exception:
         pass

      # After restoring state, update View menu checkmarks to match.
      self._sync_view_menu_checks()
      return applied

def main():
   import argparse
   # Initialize logging as early as possible (before Qt starts).
   try:
      try:
         from utils.logging_setup import setup_app_logging
      except Exception:
         pkg_root = Path(__file__).resolve().parents[1]
         if str(pkg_root) not in sys.path:
            sys.path.insert(0, str(pkg_root))
         from utils.logging_setup import setup_app_logging

      _, log_file = setup_app_logging(app_name="microscope_controller")
      try:
         logger.info("GUI starting (log=%s)", log_file)
      except Exception:
         pass
   except Exception:
      # Logging must never prevent the GUI from launching.
      pass

   parser = argparse.ArgumentParser(description="Microscope Controller GUI")
   parser.add_argument(
      "--config", "-c",
      default="config/default_devices.json",
      metavar="CONFIG_JSON",
      help="Path to the device config JSON file (default: config/default_devices.json)",
   )
   parser.add_argument(
      "--experiment", "-e",
      default=None,
      metavar="EXPERIMENT_JSON",
      help="Path to an experiment JSON file to load on startup (optional)",
   )
   # parse_known_args so Qt's own argv flags don't cause errors
   args, _ = parser.parse_known_args()

   app = QtWidgets.QApplication(sys.argv)
   win = MainWindow(config_path=args.config)
   win.resize(1400, 900)
   win.show()

   # Load experiment after window is shown so all widgets are ready
   if args.experiment:
      from pathlib import Path as _Path
      exp_path = str(_Path(args.experiment).resolve())
      QtCore.QTimer.singleShot(200, lambda: win.load_full_experiment_from_file(exp_path))

   sys.exit(app.exec())


if __name__ == "__main__":
   main()