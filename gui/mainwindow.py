import sys
import time
import threading
import json
from pathlib import Path

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

from core.factory import build_devices
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


class MainWindow(QtWidgets.QMainWindow):
   def __init__(self):
      super().__init__()
      self.setWindowTitle("Microscope Control System")

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
      self._build_ui()
      self._load_layout()


   def closeEvent(self, event):
      """Save layout before closing."""
      self._save_layout()
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

      # Demo tab dock
      self.demo_dock = QtWidgets.QDockWidget("Demo", self)
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
      self.demo_tab.start_requested.connect(self._start_experiment)
      self.demo_tab.stop_requested.connect(self._stop_experiment)

      self.multi_tab.start_requested.connect(self._start_multiaxis)
      self.multi_tab.stop_requested.connect(self._stop_multiaxis)
      
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
      save_layout_action.triggered.connect(self._save_layout)
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

      run_demo_action = QAction("Run Demo Experiment", self)
      run_demo_action.setShortcut("Ctrl+D")
      run_demo_action.triggered.connect(lambda: self._start_experiment(self.demo_tab.get_config() if hasattr(self.demo_tab, 'get_config') else {}))
      action_menu.addAction(run_demo_action)

      stop_demo_action = QAction("Stop Demo Experiment", self)
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
         demo_act = QAction("Demo", self, checkable=True)
         demo_act.setChecked(True)
         demo_act.triggered.connect(lambda checked: self.demo_dock.setVisible(bool(checked)))
         view_menu.addAction(demo_act)

         multi_act = QAction("Multi‑Axis", self, checkable=True)
         multi_act.setChecked(True)
         multi_act.triggered.connect(lambda checked: self.multi_dock.setVisible(bool(checked)))
         view_menu.addAction(multi_act)

         view_menu.addSeparator()

         cam_act = QAction("Camera", self, checkable=True)
         cam_act.setChecked(True)
         cam_act.triggered.connect(lambda checked: self.cam_dock.setVisible(bool(checked)))
         view_menu.addAction(cam_act)

         detimg_act = QAction("Detector Images", self, checkable=True)
         detimg_act.setChecked(True)
         detimg_act.triggered.connect(lambda checked: self.detimg_dock.setVisible(bool(checked)))
         view_menu.addAction(detimg_act)

         plot_act = QAction("Plot", self, checkable=True)
         plot_act.setChecked(True)
         plot_act.triggered.connect(lambda checked: self.plot_dock.setVisible(bool(checked)))
         view_menu.addAction(plot_act)

         detctl_act = QAction("Detector Controls", self, checkable=True)
         detctl_act.setChecked(True)
         detctl_act.triggered.connect(lambda checked: self.detctl_dock.setVisible(bool(checked)))
         view_menu.addAction(detctl_act)
      except Exception:
         pass
   
   def _reset_layout(self):
      """Reset layout to default (clears saved settings)."""
      settings = QtCore.QSettings("MicroscopeController", "MainWindow")
      settings.remove("geometry")
      settings.remove("windowState")
      QtWidgets.QMessageBox.information(self, "Layout Reset", "Layout reset to default. Restart the application to apply.")

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

      mode = cfg["mode"]
      cam, stage, focus, light, fw, det = build_devices()
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
            self.stream_savers[det_id] = StreamSaver(out_dir, det_id)
      except Exception:
         pass

      exp = self._build_experiment(cfg)
      self._t0 = time.time()
      self.live_tab.reset_1d_detector()

      def worker():
            try:
               self.orch.run_experiment(exp)
            finally:
               self.orch.shutdown()
               self.orch = None
               self.orch_thread = None

      self.orch_thread = threading.Thread(target=worker, daemon=True)
      self.orch_thread.start()

   def _on_stream_toggled(self, det_id: str, enabled: bool):
      """Create or close stream saver when user toggles streaming from the LiveTab."""
      try:
         if enabled:
            if det_id not in self.stream_savers:
               out_dir = Path(self.demo_tab.output_dir_edit.text() or Path.cwd() / "data")
               self.stream_savers[det_id] = StreamSaver(out_dir, det_id)
         else:
            saver = self.stream_savers.pop(det_id, None)
            if saver:
               try:
                  saver.close()
               except Exception:
                  pass
      except Exception:
         pass

   def _stop_experiment(self):
      if self.orch is not None:
            self.orch.stop()

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
         self.cam, self.stage, self.focus, self.light, self.fw, self.det = build_devices()
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
               axes.append(XAxis(stage, p["start"], p["end"], p["step"], motor_devices=motor_devices or None, motor_mode=p.get("motor_mode", "sequential")))
            elif t == "Y":
               motor_devices = [device_map.get(n) for n in p.get("motors", []) if device_map.get(n) is not None]
               axes.append(YAxis(stage, p["start"], p["end"], p["step"], motor_devices=motor_devices or None, motor_mode=p.get("motor_mode", "sequential")))
            elif t == "Z":
               motor_devices = [device_map.get(n) for n in p.get("motors", []) if device_map.get(n) is not None]
               axes.append(ZAxis(focus, p["start"], p["end"], p["step"], motor_devices=motor_devices or None, motor_mode=p.get("motor_mode", "sequential")))
            elif t == "Channel":
               axes.append(ChannelAxis(cam, light, fw, p["channels"], p["wait"]))
            elif t == "Detector":
               axes.append(DetectorAxis(det, p["scales"],p["wait"]))
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
            if det_id not in self.stream_savers:
               self.stream_savers[det_id] = StreamSaver(out_dir, det_id)
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
                      
                      # add to live tab (multiaxis) via queued invokeMethod to avoid
                      # calling Qt widgets from the worker thread
                      try:
                         QtCore.QMetaObject.invokeMethod(
                               self.live_tab,
                               "add_multiaxis_detector",
                               QtCore.Qt.ConnectionType.QueuedConnection,
                               QtCore.Q_ARG(str, det_id),
                               QtCore.Q_ARG(object, state),
                               QtCore.Q_ARG(float, float(val)),
                         )
                      except Exception:
                         # best-effort fallback: call directly (may be unsafe)
                         try:
                            self.live_tab.add_multiaxis_detector(det_id, state, val)
                         except Exception:
                            try:
                               self.live_tab.add_multiaxis_detector(state, val)
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
      # pass on_move callback so we can persist stage-ready events
      self.multi_runner = MultiAxisRunner(exp, on_move=self._on_stage_move)

      def worker():
            try:
               self.multi_runner.run()
            finally:
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
      for s in self.stream_savers.values():
         try:
            s.close()
         except Exception:
            pass
      self.stream_savers.clear()

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

   def _on_stage_move(self, state: dict):
      """Called when a motor/axis move completes during a multi-axis run.

      This will append a metadata-only sample to all active stream savers so
      positions are recorded alongside detector streams.
      """
      try:
         ts = time.time()
         for saver in list(self.stream_savers.values()):
            try:
               saver.append_sample(ts, float('nan'), meta={"event": "stage_ready", "state": state, "timestamp": ts})
            except Exception:
               continue
         # update status bar briefly
         try:
            if isinstance(state, dict):
               # show primary coords if present
               x = state.get('X') or state.get('x')
               y = state.get('Y') or state.get('y')
               if x is not None and y is not None:
                  self.statusBar().showMessage(f"Stage ready x={x:.3f} y={y:.3f}")
               else:
                  self.statusBar().showMessage("Stage ready")
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

      fname = f"{int(meta['timestamp'])}.npy"
      path = out_dir / fname

      np.save(path, img)

      with open(path.with_suffix(".json"), "w") as f:
         json.dump(safe_meta, f, indent=2)

   def _on_detector_sample(self, *args):
      """Accept either (value, meta) or (detector_id, value, meta)."""
      try:
         if len(args) == 2:
            value, meta = args
            det_id = "detector"
         elif len(args) == 3:
            det_id, value, meta = args
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

      data = {
         "demo": demo_cfg,
         "multiaxis": {"axes": axes},
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
      self.demo_tab.scale_spin.setValue(demo["det_scale"])
      self.demo_tab.offset_spin.setValue(demo["det_offset"])

      # --- Restore Multi-axis tab ---
      self.multi_tab.axis_list.clear()
      for cfg in data["multiaxis"]["axes"]:
         axis_cfg = AxisConfig(cfg["axis_type"], cfg["params"])
         item = QtWidgets.QListWidgetItem(axis_cfg.label())
         item.setData(QtCore.Qt.ItemDataRole.UserRole, axis_cfg)
         self.multi_tab.axis_list.addItem(item)
   def _save_layout(self):
      """Save window geometry and dock layout to settings."""
      settings = QtCore.QSettings("MicroscopeController", "MainWindow")
      settings.setValue("geometry", self.saveGeometry())
      settings.setValue("windowState", self.saveState())

   def _load_layout(self):
      """Restore window geometry and dock layout from settings."""
      settings = QtCore.QSettings("MicroscopeController", "MainWindow")
      geometry = settings.value("geometry")
      window_state = settings.value("windowState")
      
      if geometry:
         self.restoreGeometry(geometry)
      if window_state:
         self.restoreState(window_state)

def main():
   app = QtWidgets.QApplication(sys.argv)
   win = MainWindow()
   win.resize(1400, 900)
   win.show()
   sys.exit(app.exec())


if __name__ == "__main__":
   main()