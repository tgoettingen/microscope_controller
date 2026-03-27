import sys
import time
import threading
import json
from pathlib import Path
from datetime import datetime

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
# ── Saving toggle ────────────────────────────────────────────────────────────
# Set to False to completely disable HDF5/CSV saving (useful for debugging UI).
_SAVING_ENABLED = True
# ─────────────────────────────────────────────────────────────────────────────

# Robust import for StreamSaver: try local utils, then package import, then adjust sys.path
try:
   from utils.stream_saver import StreamSaver
except Exception:
   try:
      from microscope_controller.utils.stream_saver import StreamSaver
   except Exception:
      import sys
      from pathlib import Path
      pkg_root = Path(__file__).resolve().parents[1]
      if str(pkg_root) not in sys.path:
         sys.path.insert(0, str(pkg_root))
      from utils.stream_saver import StreamSaver

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
from tabs.camera_control_tab import CameraControlTab
from tabs.live_tab import LiveTab
from tabs.multiaxis_tab import MultiAxisTab


class MainWindow(QtWidgets.QMainWindow):
   # Thread-safe delivery of multi-axis detector samples into the GUI thread
   multiaxis_sample = QtCore.pyqtSignal(str, object, float)

   def __init__(self, config_path: str = "config/default_devices.json"):
      super().__init__()
      self.setWindowTitle("Microscope Control System")

      # Paths supplied on the command line (or defaults)
      self._config_path = config_path

      # Current detector selection coming from MultiAxisTab.
      # None => no filtering (show all); set[str] => show only these ids.
      self._selected_detectors_for_display: set[str] | None = None

      self.orch_thread: threading.Thread | None = None
      self.orch: Orchestrator | None = None

      self.multi_runner: MultiAxisRunner | None = None
      self.multi_thread: threading.Thread | None = None

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

      # Strip-chart-only detector streaming state
      self.strip_thread: threading.Thread | None = None
      self.strip_stop_evt: threading.Event | None = None

      # Camera control state
      self._camera_device = None
      self.camera_live_thread: threading.Thread | None = None
      self.camera_live_stop_evt: threading.Event | None = None
      self._build_ui()
      self._load_layout()


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
            "camera": getattr(self, "cam_dock", None),
            "camctl": getattr(self, "camctl_dock", None),
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
            ("camera", getattr(self, "cam_dock", None)),
            ("camctl", getattr(self, "camctl_dock", None)),
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
      """Save layout before closing."""
      self._save_layout()
      # Stop any running experiments
      if self.strip_thread is not None:
         self._stop_strip_chart()
      if self.orch_thread is not None:
         self._stop_experiment()
      if self.multi_thread is not None:
         self._stop_multiaxis()
      self._stop_camera_live()
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

      # Strip chart tab dock
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

         # Camera control dock
         self.camera_control_tab = CameraControlTab()
         self.camctl_dock = QtWidgets.QDockWidget("Camera Control", self)
         self.camctl_dock.setObjectName("dock_camera_control")
         self.camctl_dock.setWidget(self.camera_control_tab)
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
      self.demo_tab.start_requested.connect(self._start_strip_chart)
      self.demo_tab.stop_requested.connect(self._stop_strip_chart)

      # Camera control signals
      try:
         self.camera_control_tab.exposure_changed.connect(self._on_camera_exposure_changed)
         self.camera_control_tab.snapshot_requested.connect(self._on_camera_snapshot_requested)
         self.camera_control_tab.live_toggled.connect(self._on_camera_live_toggled)
      except Exception:
         pass

      self.multi_tab.start_requested.connect(self._start_multiaxis)
      self.multi_tab.stop_requested.connect(self._stop_multiaxis)

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
      save_layout_action.triggered.connect(self._save_layout_interactive)
      file_menu.addAction(save_layout_action)

      reset_layout_action = QAction("Reset Layout to Default", self)
      reset_layout_action.triggered.connect(self._reset_layout)
      file_menu.addAction(reset_layout_action)

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

      run_demo_action = QAction("Start Strip Chart", self)
      run_demo_action.setShortcut("Ctrl+D")
      run_demo_action.triggered.connect(lambda: self._start_strip_chart(self.demo_tab.get_config() if hasattr(self.demo_tab, 'get_config') else {}))
      action_menu.addAction(run_demo_action)

      stop_demo_action = QAction("Stop Strip Chart", self)
      stop_demo_action.setShortcut("Ctrl+E")
      stop_demo_action.triggered.connect(self._stop_strip_chart)
      action_menu.addAction(stop_demo_action)

      help_menu = menubar.addMenu("&Help")
      about_action = QAction("About", self)
      about_action.triggered.connect(self._show_about)
      help_menu.addAction(about_action)

      # View menu for toggling docks
      view_menu = menubar.addMenu("&View")
      try:
         self._view_dock_actions = {}

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

         view_menu.addSeparator()

         cam_act = QAction("Camera", self, checkable=True)
         cam_act.setChecked(True)
         cam_act.triggered.connect(lambda checked: self.cam_dock.setVisible(bool(checked)))
         view_menu.addAction(cam_act)
         self._view_dock_actions["camera"] = cam_act

         camctl_act = QAction("Camera Control", self, checkable=True)
         camctl_act.setChecked(True)
         camctl_act.triggered.connect(lambda checked: self.camctl_dock.setVisible(bool(checked)))
         view_menu.addAction(camctl_act)
         self._view_dock_actions["camctl"] = camctl_act

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

         detctl_act = QAction("Detector Controls", self, checkable=True)
         detctl_act.setChecked(True)
         detctl_act.triggered.connect(lambda checked: self.detctl_dock.setVisible(bool(checked)))
         view_menu.addAction(detctl_act)
         self._view_dock_actions["detctl"] = detctl_act
      except Exception:
         pass
   
   def _reset_layout(self):
      """Reset layout to default (clears saved settings)."""
      settings = QtCore.QSettings("MicroscopeController", "MainWindow")
      settings.remove("geometry")
      settings.remove("windowState")
      QtWidgets.QMessageBox.information(self, "Layout Reset", "Layout reset to default. Restart the application to apply.")

   def _suggest_layout_export_path(self) -> Path:
      """Default path for interactive layout export."""
      try:
         repo_root = Path(__file__).resolve().parents[1]
      except Exception:
         repo_root = Path.cwd()
      out_dir = repo_root / "data"
      try:
         out_dir.mkdir(parents=True, exist_ok=True)
      except Exception:
         pass
      ts = datetime.now().strftime("%Y%m%d_%H%M%S")
      return out_dir / f"layout_{ts}.ini"

   def _show_about(self):
      QtWidgets.QMessageBox.information(
            self,
            "About",
            "Microscope Controller\nPyQt6 + pyqtgraph, multi‑axis + detector visualization.",
      )

   def _coerce_detector_list(self, detector_obj) -> list:
      if detector_obj is None:
         return []
      if isinstance(detector_obj, list):
         return detector_obj
      return [detector_obj]

   def _start_strip_chart(self, cfg: dict):
      """Start detector-only streaming for the strip-chart display."""
      if self.strip_thread is not None and self.strip_thread.is_alive():
         return

      try:
         self.live_tab.reset_1d_detector()
         if hasattr(self.live_tab, "prepare_strip_chart_plot"):
            self.live_tab.prepare_strip_chart_plot()
         if hasattr(self.live_tab, "_set_detector_view"):
            self.live_tab._set_detector_view()
      except Exception:
         pass

      try:
         cam, stage, focus, light, fw, det = build_devices(self._config_path)
      except Exception as exc:
         QtWidgets.QMessageBox.warning(self, "Strip Chart", f"Could not build devices:\n{exc}")
         return

      detectors = self._coerce_detector_list(det)
      if not detectors:
         QtWidgets.QMessageBox.warning(self, "Strip Chart", "No detector configured.")
         return

      # Apply UI scaling to configured detectors (best effort).
      try:
         s = float(getattr(self.demo_tab, "scale_spin").value())
         o = float(getattr(self.demo_tab, "offset_spin").value())
         for d in detectors:
            try:
               if hasattr(d, "set_scale"):
                  d.set_scale(s, o)
            except Exception:
               pass
      except Exception:
         pass

      # Ensure detector traces are registered in the Live panel.
      det_ids: list[str] = []
      for i, d in enumerate(detectors):
         try:
            det_id = getattr(d, "name", None) or getattr(d, "port", None) or f"detector{i + 1}"
            det_id = str(det_id)
            det_ids.append(det_id)
            self.live_tab.register_detector(det_id)
         except Exception:
            continue

      # Make sure LiveTab filtering does not drop strip-chart samples.
      try:
         if hasattr(self, "live_tab") and hasattr(self.live_tab, "set_selected_detectors"):
            self.live_tab.set_selected_detectors(det_ids)
      except Exception:
         pass

      try:
         dt = float(getattr(self.demo_tab, "interval_spin").value())
         sample_period = max(1e-8, dt)
      except Exception:
         sample_period = 0.1

      # Apply moving time window (seconds) from Strip Chart panel by converting
      # it into a sample-count window used by LiveTab buffers.
      try:
         window_time_s = float(cfg.get("window_time_s", 5.0))
         window_samples = max(10, int(round(window_time_s / sample_period)))
         if hasattr(self, "live_tab") and hasattr(self.live_tab, "set_window_size"):
            self.live_tab.set_window_size(window_samples)
      except Exception:
         pass

      self.strip_stop_evt = threading.Event()

      def _worker():
         while self.strip_stop_evt is not None and not self.strip_stop_evt.is_set():
            ts = time.time()
            for i, d in enumerate(detectors):
               try:
                  det_id = getattr(d, "name", None) or getattr(d, "port", None) or f"detector{i + 1}"
                  try:
                     val = float(d.read_value())
                  except TypeError:
                     val = float(d.read_value(wait=0))

                  # add_detector_sample only appends to Python buffers; this is
                  # safe from the worker thread and avoids invokeMethod/slot issues.
                  self.live_tab.add_detector_sample(str(det_id), float(val), float(ts))

                  try:
                     saver = self.stream_savers.get(str(det_id))
                     if saver:
                        saver.append_sample(ts, float(val), meta={"timestamp": ts})
                  except Exception:
                     pass
               except Exception:
                  continue

            # Wait with interrupt support.
            if self.strip_stop_evt.wait(sample_period):
               break

      self.strip_thread = threading.Thread(target=_worker, daemon=True)
      self.strip_thread.start()
      try:
         self.statusBar().showMessage("Strip chart streaming started", 3000)
      except Exception:
         pass

   def _stop_strip_chart(self):
      evt = self.strip_stop_evt
      t = self.strip_thread
      if evt is not None:
         evt.set()
      if t is not None and t.is_alive():
         try:
            t.join(timeout=1.0)
         except Exception:
            pass
      self.strip_stop_evt = None
      self.strip_thread = None
      try:
         self.statusBar().showMessage("Strip chart streaming stopped", 3000)
      except Exception:
         pass

   def _ensure_camera_device(self):
      if self._camera_device is not None:
         return self._camera_device
      try:
         cam, _stage, _focus, _light, _fw, _det = build_devices(self._config_path)
         self._camera_device = cam
      except Exception:
         self._camera_device = None
      return self._camera_device

   def _on_camera_exposure_changed(self, ms: float):
      cam = self._ensure_camera_device()
      if cam is None:
         return
      try:
         cam.set_exposure(float(ms))
      except Exception:
         pass

   def _publish_camera_frame(self, img):
      meta = {"timestamp": time.time(), "source": "camera_control"}
      try:
         QtCore.QMetaObject.invokeMethod(
            self.live_tab,
            "update_image",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(object, img),
            QtCore.Q_ARG(dict, meta),
         )
      except Exception:
         pass

   def _on_camera_snapshot_requested(self):
      cam = self._ensure_camera_device()
      if cam is None:
         QtWidgets.QMessageBox.warning(self, "Camera", "Camera device is not available.")
         return
      try:
         img = cam.snap()
      except Exception as exc:
         QtWidgets.QMessageBox.warning(self, "Camera", f"Snapshot failed:\n{exc}")
         return
      self._publish_camera_frame(img)
      try:
         self.live_tab._set_camera_view()
      except Exception:
         pass

   def _stop_camera_live(self):
      evt = self.camera_live_stop_evt
      t = self.camera_live_thread
      if evt is not None:
         evt.set()
      if t is not None and t.is_alive():
         try:
            t.join(timeout=1.0)
         except Exception:
            pass
      self.camera_live_thread = None
      self.camera_live_stop_evt = None
      try:
         if hasattr(self, "camera_control_tab"):
            self.camera_control_tab.set_live_checked(False)
      except Exception:
         pass

   def _on_camera_live_toggled(self, enabled: bool):
      if not enabled:
         self._stop_camera_live()
         return

      cam = self._ensure_camera_device()
      if cam is None:
         QtWidgets.QMessageBox.warning(self, "Camera", "Camera device is not available.")
         try:
            self.camera_control_tab.set_live_checked(False)
         except Exception:
            pass
         return

      self._stop_camera_live()
      self.camera_live_stop_evt = threading.Event()

      def _worker():
         try:
            if hasattr(cam, "start_live"):
               cam.start_live()
         except Exception:
            pass

         while self.camera_live_stop_evt is not None and not self.camera_live_stop_evt.is_set():
            try:
               img = cam.snap()
               self._publish_camera_frame(img)
            except Exception:
               pass
            if self.camera_live_stop_evt.wait(0.05):
               break

         try:
            if hasattr(cam, "stop_live"):
               cam.stop_live()
         except Exception:
            pass

      self.camera_live_thread = threading.Thread(target=_worker, daemon=True)
      self.camera_live_thread.start()
      try:
         self.live_tab._set_camera_view()
      except Exception:
         pass

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

      # Persist detector calibration into device config (hardware settings)
      try:
         self._persist_detector_scaling_to_device_config(cfg.get("det_scale", 1.0), cfg.get("det_offset", 0.0))
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
      # detector may be a single detector or a list
      try:
         if isinstance(det, list):
            for d in det:
               if hasattr(d, "set_scale"):
                  d.set_scale(cfg["det_scale"], cfg["det_offset"])
         else:
            det.set_scale(cfg["det_scale"], cfg["det_offset"])
      except Exception:
         pass

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

      exp = self._build_experiment(cfg)
      self._t0 = time.time()
      self.live_tab.reset_1d_detector()

      def worker():
            try:
               self.orch.run_experiment(exp)
            finally:
               try:
                  self.orch.shutdown()
               except Exception:
                  pass
               # When the measurement finishes, stop stream saving.
               self._close_all_stream_savers()
               self.orch = None
               self.orch_thread = None

      self.orch_thread = threading.Thread(target=worker, daemon=True)
      self.orch_thread.start()

   def _on_stream_toggled(self, det_id: str, enabled: bool):
      """Create or close stream saver when user toggles streaming from the LiveTab."""
      try:
         if enabled:
            if _SAVING_ENABLED and det_id not in self.stream_savers:
               out_dir = Path(self.demo_tab.output_dir_edit.text() or Path.cwd() / "data")
               try:
                  out_dir.mkdir(parents=True, exist_ok=True)
               except Exception:
                  pass

               ts = datetime.now().strftime("%Y%m%d_%H%M%S")
               suggested = out_dir / f"stream_{det_id}_{ts}.h5"
               path, _ = QtWidgets.QFileDialog.getSaveFileName(
                  self,
                  f"Save Stream ({det_id})",
                  str(suggested),
                  "HDF5 (*.h5 *.hdf5)",
               )
               if not path:
                  # user cancelled: revert checkbox state
                  try:
                     if hasattr(self, "live_tab") and hasattr(self.live_tab, "set_stream_enabled"):
                        self.live_tab.set_stream_enabled(det_id, False)
                  except Exception:
                     pass
                  return

               self.stream_savers[det_id] = StreamSaver(out_dir, det_id, base_path=path)
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

      cfgs: list[AxisConfig] = self.multi_tab.get_axis_configs()
      if not cfgs:
            QtWidgets.QMessageBox.warning(self, "Multi‑Axis", "No axes defined.")
            return

      # Build devices only if not already built or if previously released
      if not self.devices_built or self.devices_released:
         # Persist detector calibration from Demo tab into device config
         try:
            s = self.demo_tab.scale_spin.value() if hasattr(self, 'demo_tab') else 1.0
            o = self.demo_tab.offset_spin.value() if hasattr(self, 'demo_tab') else 0.0
            self._persist_detector_scaling_to_device_config(s, o)
         except Exception:
            pass

         self.cam, self.stage, self.focus, self.light, self.fw, self.det = build_devices(self._config_path)
         self.devices_built = True
         self.devices_released = False

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
               axes.append(DetectorAxis(det, p["scales"], p.get("wait", 0.0)))
            elif t == "Round":
               axes.append(RoundAxis(p["n_rounds"]))

      self.live_tab.reset_multiaxis()

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
                              dev.move_to(p['pre_pos'], cur[1])
                           else:
                              dev.move_to(cur[0], p['pre_pos'])
                        else:
                           dev.move_to(p['pre_pos'])
                     except Exception:
                        pass
               elif cfg.axis_type == 'Z':
                  dev = device_map.get('focus')
                  if dev and hasattr(dev, 'move_to'):
                     try:
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
               self.multi_runner.run()
            finally:
               # When the measurement finishes, stop stream saving.
               self._close_all_stream_savers()
               self.multi_runner = None
               self.multi_thread = None

      self.multi_thread = threading.Thread(target=worker, daemon=True)
      self.multi_thread.start()

   def _stop_multiaxis(self):
      if self.multi_runner is not None:
            self.multi_runner.stop()

      # Release devices when stopping
      if self.devices_built and not self.devices_released:
         for dev in [self.cam, self.stage, self.focus, self.light, self.fw, self.det]:
               if dev and hasattr(dev, 'disconnect'):
                  dev.disconnect()
         self.devices_released = True
      # close any stream savers
      self._close_all_stream_savers()

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
      """Append a metadata-only record to all active stream savers.

      We encode events as samples with NaN value so they appear in the same
      timeline as detector samples in `.meta.jsonl`.
      """
      try:
         ts = float(payload.get("timestamp", time.time())) if isinstance(payload, dict) else time.time()
      except Exception:
         ts = time.time()

      meta = {"event": event, "timestamp": ts, "payload": payload}
      for saver in list(self.stream_savers.values()):
         try:
            saver.append_sample(ts, float('nan'), meta=meta)
         except Exception:
            continue

   def _on_axis_event(self, event: str, payload: dict):
      """Receive axis events from the classic Orchestrator."""
      try:
         if not isinstance(payload, dict):
            payload = {"value": payload}
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

      # Use a timestamped HDF5 file per frame.
      try:
         ts = float(meta.get('timestamp', time.time())) if isinstance(meta, dict) else time.time()
      except Exception:
         ts = time.time()

      # Include microseconds to avoid overwriting frames captured within the same second.
      try:
         from datetime import datetime as _dt
         stamp = _dt.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S_%f')
      except Exception:
         stamp = str(int(ts))

      path = out_dir / f"{stamp}.h5"

      try:
         import h5py

         # Save the image as shown in the UI (LiveTab transposes on display).
         arr = np.asarray(img).T
         with h5py.File(path, 'w') as f:
            f.create_dataset('image', data=arr)
            try:
               f.attrs['timestamp'] = float(ts)
            except Exception:
               pass
            try:
               f.attrs['meta_json'] = json.dumps(safe_meta)
            except Exception:
               pass
      except Exception:
         # If HDF5 writing fails, do not crash acquisition.
         return

      # Optional sidecar for quick inspection/debugging.
      try:
         with open(path.with_suffix(".json"), "w") as f:
            json.dump(safe_meta, f, indent=2)
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
               "add_detector_sample",
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
         "window_time_s": self.demo_tab.window_time_spin.value(),
         "z_start": self.demo_tab.z_start_spin.value(),
         "z_end": self.demo_tab.z_end_spin.value(),
         "z_step": self.demo_tab.z_step_spin.value(),
         "output_dir": self.demo_tab.output_dir_edit.text(),
         "det_scale": self.demo_tab.scale_spin.value(),
         "det_offset": self.demo_tab.offset_spin.value(),
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
      self.demo_tab.interval_spin.setValue(float(demo.get("interval_s", 0.05)))
      self.demo_tab.window_time_spin.setValue(float(demo.get("window_time_s", 5.0)))
      self.demo_tab.z_start_spin.setValue(demo["z_start"])
      self.demo_tab.z_end_spin.setValue(demo["z_end"])
      self.demo_tab.z_step_spin.setValue(demo["z_step"])
      self.demo_tab.output_dir_edit.setText(demo["output_dir"])
      self.demo_tab.scale_spin.setValue(demo["det_scale"])
      self.demo_tab.offset_spin.setValue(demo["det_offset"])

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
   def _save_layout(self):
      """Save window geometry and dock layout to settings."""
      settings = QtCore.QSettings("MicroscopeController", "MainWindow")
      settings.setValue("geometry", self.saveGeometry())
      settings.setValue("windowState", self.saveState())

   def _save_layout_interactive(self):
      """Save layout to a user-chosen file, and also persist as the default."""
      try:
         default_path = self._suggest_layout_export_path()
      except Exception:
         default_path = Path.cwd() / "layout.ini"

      path, _ = QtWidgets.QFileDialog.getSaveFileName(
         self,
         "Save Layout",
         str(default_path),
         "Layout files (*.ini)",
      )
      if not path:
         return

      out_path = Path(path)
      if out_path.suffix.lower() != ".ini":
         out_path = out_path.with_suffix(".ini")
      try:
         out_path.parent.mkdir(parents=True, exist_ok=True)
      except Exception:
         pass

      try:
         file_settings = QtCore.QSettings(str(out_path), QtCore.QSettings.Format.IniFormat)
         file_settings.setValue("geometry", self.saveGeometry())
         file_settings.setValue("windowState", self.saveState())
         file_settings.sync()
      except Exception as exc:
         QtWidgets.QMessageBox.warning(self, "Save Layout", f"Failed to save layout to:\n{out_path}\n\n{exc}")
         return

      # Keep prior behavior: also save as default for next startup.
      try:
         self._save_layout()
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
         self.demo_tab.interval_spin.setValue(float(demo.get("interval_s", 0.05)))
         self.demo_tab.window_time_spin.setValue(float(demo.get("window_time_s", 5.0)))
         self.demo_tab.z_start_spin.setValue(demo["z_start"])
         self.demo_tab.z_end_spin.setValue(demo["z_end"])
         self.demo_tab.z_step_spin.setValue(demo["z_step"])
         self.demo_tab.output_dir_edit.setText(demo["output_dir"])
         self.demo_tab.scale_spin.setValue(demo["det_scale"])
         self.demo_tab.offset_spin.setValue(demo["det_offset"])
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

   def _load_layout(self):
      """Restore window geometry and dock layout from settings."""
      settings = QtCore.QSettings("MicroscopeController", "MainWindow")
      geometry = settings.value("geometry")
      window_state = settings.value("windowState")
      
      if geometry:
         self.restoreGeometry(geometry)
      if window_state:
         self.restoreState(window_state)

      # After restoring state, update View menu checkmarks to match.
      self._sync_view_menu_checks()

def main():
   import argparse
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