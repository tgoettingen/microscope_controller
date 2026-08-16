import sys
import time
import threading
import json
import base64
from pathlib import Path
import logging
import uuid

import importlib.util
import numpy as np

try:
   from version import APP_VERSION, APP_NAME
except Exception:
   APP_VERSION = "unknown"
   APP_NAME = "Microscope Controller"

# Prefer importing PyQt6. If it's importable, proceed. If not, give a helpful
# message guiding the user to install PyQt6 in the venv. Avoid failing based on
# distribution metadata (which can be present even after partial uninstalls).
try:
   from PyQt6 import QtWidgets, QtCore
   from PyQt6.QtGui import QAction, QActionGroup
   from PyQt6.QtCore import Qt
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
   ChannelAxis, DetectorAxis, RoundAxis, ExcitationAxis,
)

# Plugin system imports
try:
   from plugins.plugin_manager import get_plugin_manager
   from plugins.base_plugin import PluginData
   PLUGINS_AVAILABLE = True
except Exception:
   PLUGINS_AVAILABLE = False
   logger.warning("Plugin system not available")

try:
   from tabs.experiment_tab import ExperimentTab
   from tabs.live_tab import LiveTab
   from tabs.multiaxis_tab import MultiAxisTab
   from tabs.camera_control_tab import CameraControlTab
   from tabs.multiview_camera_tab import MultiViewCameraTab
   from tabs.multiview_control_tab import MultiViewControlTab
except Exception:
   from gui.tabs.experiment_tab import ExperimentTab
   from gui.tabs.live_tab import LiveTab
   from gui.tabs.multiaxis_tab import MultiAxisTab
   from gui.tabs.camera_control_tab import CameraControlTab
   from gui.tabs.multiview_camera_tab import MultiViewCameraTab
   from gui.tabs.multiview_control_tab import MultiViewControlTab

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
   from utils.multichannel_saver import MultiChannelSaver
except Exception:
   MultiChannelSaver = None  # type: ignore

try:
   from utils.image_h5_saver import ImageH5Saver
except Exception:
   pkg_root = Path(__file__).resolve().parents[1]
   if str(pkg_root) not in sys.path:
      sys.path.insert(0, str(pkg_root))
   from utils.image_h5_saver import ImageH5Saver


logger = logging.getLogger(__name__)


def _resolve_motors(device_map: dict, params: dict):
    """Resolve an axis's motor names into parallel device + mode lists.

    Returns ``(motor_devices, motor_modes)`` where ``motor_modes[i]`` is the
    per-device run mode ("synchronized"/"sequential") for ``motor_devices[i]``.
    Names that do not resolve to a device are skipped so both lists stay
    aligned.
    """
    names = params.get("motors", []) or []
    modes_map = params.get("motor_modes", {}) or {}
    default_mode = params.get("motor_mode", "sequential")
    if default_mode not in ("synchronized", "sequential"):
        default_mode = "sequential"
    motor_devices = []
    motor_modes = []
    for n in names:
        dev = device_map.get(n)
        if dev is not None:
            motor_devices.append(dev)
            motor_modes.append(modes_map.get(n, default_mode))
    return motor_devices, motor_modes


class MainWindow(QtWidgets.QMainWindow):
   # Thread-safe delivery of multi-axis detector samples into the GUI thread
   multiaxis_sample = QtCore.pyqtSignal(str, object, float)
   measurement_state_changed = QtCore.pyqtSignal(object)

   def __init__(self, config_path: str = "config/default_devices.json"):
      super().__init__()
      self.setWindowTitle("Microscope Control System")

      try:
         logger.info("MainWindow init (config=%s)", config_path)
      except Exception:
         pass

      # Paths supplied on the command line (or defaults)
      self._config_path = config_path
      try:
         self._config_filename = Path(config_path).name  # Store config filename for title
      except Exception:
         self._config_filename = "config"  # Fallback if path processing fails
      self._experiment_filename: str | None = None  # Store experiment filename for title

      # Current detector selection coming from MultiAxisTab.
      # None => no filtering (show all); set[str] => show only these ids.
      self._selected_detectors_for_display: set[str] | None = None

      self.orch_thread: threading.Thread | None = None
      self.orch: Orchestrator | None = None

      self.multi_runner: MultiAxisRunner | None = None
      self.multi_thread: threading.Thread | None = None

      # Hardware currently reserved by an active run. Each entry is a tuple of
      # (detector-id set, motor-name set). None => that run is not active.
      # Used to let strip-chart and multi-axis run concurrently only when their
      # hardware does not overlap.
      self._strip_reserved: tuple[set[str], set[str]] | None = None
      self._multi_reserved: tuple[set[str], set[str]] | None = None
      # True while the active multi-axis run owns the stream/multichannel savers
      # (i.e. it actually records detector data). A detector-less multi-axis scan
      # that coexists with a running strip chart must not tear down the strip
      # chart's savers.
      self._multi_owns_stream_savers: bool = False

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
      self._mc_saver = None  # MultiChannelSaver instance when active
      self._measurement_id: str | None = None  # Universal ID for current measurement
      self._last_measurement_paths: list[str] = []
      self._last_measurement_label: str | None = None
      self._playback_speed: float = 1.0
      self._measurement_state: str = "Finished"
      self._measurement_kind: str = "Idle"
      self._measurement_status_label: QtWidgets.QLabel | None = None
      self.image_saver: ImageH5Saver | None = None
      self._image_saver_out_dir: Path | None = None

      # Webcam preview (UI-only camera control)
      self._webcam = None
      self._webcam_live = False
      self._webcam_timer: QtCore.QTimer | None = None

      # Stage position update timer for status bar (500ms when multiaxis not running)
      self._stage_position_timer: QtCore.QTimer | None = None
      self._stage_position_label: QtWidgets.QLabel | None = None

      # Thread-safe camera settings snapshot (used by worker threads)
      self._camera_exposure_ms: float = 20.0

      # Multi-view camera capture toggle (do not read Qt widgets from worker threads)
      self._multiview_capture_enabled: bool = True

      # Plugin system
      self._plugin_manager = None
      if PLUGINS_AVAILABLE:
         try:
            self._plugin_manager = get_plugin_manager()
            # Load default plugin directories
            from pathlib import Path
            plugin_dir = Path(__file__).parent.parent / "plugins"
            if plugin_dir.exists():
               self._plugin_manager.add_plugin_directory(plugin_dir)
            # Load custom plugin directory
            custom_plugin_dir = Path(__file__).parent.parent / "plugin"
            if custom_plugin_dir.exists():
               self._plugin_manager.add_plugin_directory(custom_plugin_dir)
            # Discover plugins
            discovered = self._plugin_manager.discover_plugins()
            logger.info(f"Discovered {len(discovered)} plugins")
            # Auto-load all custom plugins
            loaded_count = self._plugin_manager.auto_load_custom_plugins()
            logger.info(f"Auto-loaded {loaded_count} custom plugins")
         except Exception as e:
            logger.warning(f"Failed to initialize plugin manager: {e}")

      # Layout persistence
      # - "original" layout: a deterministic "full" dock arrangement shipped in code
      # - "default" layout: auto-saved on every exit (and can be explicitly saved)
      self._original_layout_state: object | None = None
      self._original_layout_geometry: object | None = None
      self._build_ui()
      self.measurement_state_changed.connect(self._apply_measurement_state)
      self._set_measurement_state("Finished", kind="Idle")
      self._apply_full_layout()
      self._capture_original_layout()
      self._load_layout(kind="default")
      
      # Update window title with loaded filenames
      self._update_window_title()
      
      # Try to build devices eagerly on startup and reload visible panels
      try:
         if self._build_devices_now():
            logger.info("Devices built successfully on startup")
            self._reload_visible_panels_after_init()
      except Exception as e:
         logger.warning("Failed to build devices on startup: %s", e)
      
   def _reload_visible_panels_after_init(self):
      """Reload visible panels after initialization with existing devices (if already built)."""
      try:
         # Only reload if devices are already built
         if not self.devices_built or self.devices_released:
            logger.info("Devices not built yet, skipping panel reload after init")
            return
         
         # Reload Stage Control panel if visible
         if hasattr(self, 'stage_control_dock') and self.stage_control_dock.isVisible():
            if hasattr(self, 'stage_control_tab') and self.stage_control_tab:
               logger.info("Reloading Stage Control panel after initialization")
               self.stage_control_tab.set_stage(self.stage)
               self.stage_control_tab.set_focus(self.focus)
               self.stage_control_tab.set_config_path(self._config_path)
         
         # Reload Stage Calibration panel if visible
         if hasattr(self, 'stage_calibration_dock') and self.stage_calibration_dock.isVisible():
            if hasattr(self, 'stage_calibration_tab') and self.stage_calibration_tab:
               logger.info("Reloading Stage Calibration panel after initialization")
               self.stage_calibration_tab.set_stage(self.stage)
               self.stage_calibration_tab.set_config_path(self._config_path)
         
         # Reload Excitation Control panel if visible
         if hasattr(self, 'excitation_control_dock') and self.excitation_control_dock.isVisible():
            if hasattr(self, 'excitation_control_tab') and self.excitation_control_tab:
               logger.info("Reloading Excitation Control panel after initialization")
               self.excitation_control_tab.set_excitation(self.excitation)
               self.excitation_control_tab.set_config_path(self._config_path)
               
      except Exception as e:
         logger.exception("Failed to reload visible panels after initialization: %s", e)

   def _update_window_title(self):
      """Update window title to show config and experiment filenames."""
      try:
         title_parts = ["Microscope Control System"]
         
         # Add config filename if available
         if hasattr(self, '_config_filename') and self._config_filename:
            title_parts.append(f": {self._config_filename}")
         
         # Add experiment filename if available
         if hasattr(self, '_experiment_filename') and self._experiment_filename:
            title_parts.append(f": {self._experiment_filename}")
         
         self.setWindowTitle("".join(title_parts))
         logger.info("Window title updated: %s", "".join(title_parts))
      except Exception as e:
         logger.warning("Failed to update window title: %s", e)


   def _set_measurement_state(self, state: str, kind: str | None = None) -> None:
      """Request a thread-safe update of the measurement state indicator."""
      try:
         self.measurement_state_changed.emit({"state": str(state), "kind": kind})
      except Exception:
         pass


   @QtCore.pyqtSlot(object)
   def _apply_measurement_state(self, payload: object) -> None:
      """Apply measurement state text on the GUI thread."""
      try:
         if isinstance(payload, dict):
            state = payload.get("state", "Finished")
            kind = payload.get("kind", None)
         else:
            state = payload
            kind = None
         normalized = str(state).strip().title() if state is not None else "Finished"
      except Exception:
         normalized = "Finished"
         kind = None
      if normalized not in {"Running", "Finished"}:
         normalized = "Finished"

      if kind is None:
         kind_text = str(getattr(self, "_measurement_kind", "Idle") or "Idle")
      else:
         try:
            kind_text = str(kind).strip() or "Idle"
         except Exception:
            kind_text = "Idle"

      self._measurement_state = normalized
      self._measurement_kind = kind_text

      # Release hardware reservations when a run finishes so the other mode's
      # start button can be re-enabled. Done before any early return below.
      if normalized == "Finished":
         if kind_text == "Strip Chart":
            self._strip_reserved = None
         elif kind_text == "Multi-Axis":
            self._multi_reserved = None
      self._refresh_run_button_states()

      lbl = getattr(self, "_measurement_status_label", None)
      if lbl is None:
         return
      try:
         if normalized == "Running":
            lbl.setText(f"Measurement: {normalized} ({kind_text})")
         else:
            if kind_text and kind_text != "Idle":
               lbl.setText(f"Measurement: {normalized} ({kind_text})")
            else:
               lbl.setText(f"Measurement: {normalized}")
      except Exception:
         pass
      
      # Control stage position timer based on multiaxis state
      # When multiaxis is running, it updates position itself, so we stop the timer
      # When multiaxis is not running, we use the 500ms timer to update position
      try:
         is_multiaxis_running = (normalized == "Running" and 
                                (kind_text == "Multi-Axis" or kind_text == "Multi View"))
         
         try:
            logger.debug("Measurement state changed: normalized=%s kind_text=%s is_multiaxis_running=%s", 
                        normalized, kind_text, is_multiaxis_running)
         except Exception:
            pass
         
         if is_multiaxis_running:
            try:
               logger.info("Stopping stage position timer - multiaxis is running and will update position itself")
            except Exception:
               pass
            self._stop_stage_position_timer()
         else:
            # When not running (including Finished state), ensure timer is running
            try:
               logger.info("Starting stage position timer - using 500ms timer for position updates")
            except Exception:
               pass
            self._start_stage_position_timer()
      except Exception as e:
         try:
            logger.warning("Error controlling stage position timer: %s", e)
         except Exception:
            pass


   # ----------------- hardware reservation / run gating -----------------

   def _all_detector_ids(self) -> list[str]:
      """Return every detector id known to the application.

      Strip-chart acquisition reads *all* detectors, so this is the set of
      detectors the strip chart will use.
      """
      ids: list[str] = []
      try:
         det = getattr(self, "det", None)
         if isinstance(det, list):
            ids = [getattr(d, "name", getattr(d, "port", "detector")) for d in det]
         elif det is not None:
            ids = [getattr(det, "name", getattr(det, "port", "detector"))]
      except Exception:
         ids = []
      if ids:
         return ids
      # Devices not built yet: fall back to the tab's available-detector list.
      try:
         if hasattr(self.multi_tab, "get_available_detectors"):
            return list(self.multi_tab.get_available_detectors())
      except Exception:
         pass
      return []

   def _strip_chart_hardware(self) -> tuple[set[str], set[str]]:
      """(detectors, motors) the strip chart will use.

      The strip chart polls every detector and never moves motors.
      """
      return set(self._all_detector_ids()), set()

   def _multiaxis_hardware(self) -> tuple[set[str], set[str]]:
      """(detectors, motors) the multi-axis run will use, from the current UI.

      A multi-axis scan only reads detectors when it defines a ``Detector``
      axis; a pure motor/camera scan uses no detectors and can therefore run
      alongside the strip chart. Motors are the motor names referenced by
      X/Y/Z axes (Z maps to the focus device).
      """
      dets: set[str] = set()
      motors: set[str] = set()
      try:
         cfgs = self.multi_tab.get_axis_configs()
      except Exception:
         cfgs = []
      has_detector_axis = any(getattr(c, "axis_type", None) == "Detector" for c in cfgs)
      if has_detector_axis:
         try:
            dets = set(self.multi_tab.get_selected_detectors() or [])
         except Exception:
            dets = set()
      try:
         for cfg in cfgs:
            t = getattr(cfg, "axis_type", None)
            params = getattr(cfg, "params", None) or {}
            if t in ("X", "Y", "Z"):
               for m in params.get("motors", []) or []:
                  motors.add(str(m))
               if t == "Z":
                  motors.add("focus")
      except Exception:
         pass
      return dets, motors

   @staticmethod
   def _hardware_conflict(
      usage: tuple[set[str], set[str]],
      reserved: tuple[set[str], set[str]] | None,
   ) -> str:
      """Return a human-readable description of overlapping hardware, or "".

      ``usage`` is the (detectors, motors) a run wants; ``reserved`` is what an
      already-running run holds. Empty result means there is no conflict.
      """
      if not reserved:
         return ""
      use_dets, use_motors = usage
      res_dets, res_motors = reserved
      det_overlap = sorted(set(use_dets) & set(res_dets))
      motor_overlap = sorted(set(use_motors) & set(res_motors))
      parts: list[str] = []
      if det_overlap:
         parts.append("detector(s): " + ", ".join(det_overlap))
      if motor_overlap:
         parts.append("motor(s): " + ", ".join(motor_overlap))
      return "; ".join(parts)

   def _refresh_run_button_states(self) -> None:
      """Enable/disable the Strip Chart and Multi-Axis start buttons.

      A start button is disabled while its own run is active, and also greyed
      out when starting it would conflict with the hardware reserved by the
      other running mode.
      """
      strip_running = getattr(self, "orch_thread", None) is not None
      multi_running = getattr(self, "multi_thread", None) is not None

      strip_btn = getattr(getattr(self, "demo_tab", None), "start_btn", None)
      if strip_btn is not None:
         try:
            if strip_running:
               strip_btn.setEnabled(False)
               strip_btn.setToolTip("Strip Chart is already running.")
            else:
               conflict = self._hardware_conflict(
                  self._strip_chart_hardware(), self._multi_reserved)
               strip_btn.setEnabled(not conflict)
               strip_btn.setToolTip(
                  "In use by the running Multi‑Axis scan — " + conflict
                  if conflict else "")
         except Exception:
            pass

      multi_btn = getattr(getattr(self, "multi_tab", None), "start_btn", None)
      if multi_btn is not None:
         try:
            if multi_running:
               multi_btn.setEnabled(False)
               multi_btn.setToolTip("Multi‑Axis is already running.")
            else:
               conflict = self._hardware_conflict(
                  self._multiaxis_hardware(), self._strip_reserved)
               multi_btn.setEnabled(not conflict)
               multi_btn.setToolTip(
                  "In use by the running Strip Chart — " + conflict
                  if conflict else "")
         except Exception:
            pass


   def _close_all_stream_savers(self):

      """Close and remove all active StreamSaver instances and any MultiChannelSaver."""
      # Keep last completed measurement output path(s) for replay.
      try:
         last_paths: list[str] = []
         mc = getattr(self, "_mc_saver", None)
         if mc is not None:
            p = getattr(mc, "h5_path", None)
            if p is not None:
               last_paths.append(str(p))
         for saver in list(self.stream_savers.values()):
            p = getattr(saver, "h5_path", None)
            if p is not None:
               last_paths.append(str(p))
         if last_paths:
            self._last_measurement_paths = last_paths
            try:
               if len(last_paths) == 1:
                  self._last_measurement_label = Path(last_paths[0]).name
               else:
                  self._last_measurement_label = f"{len(last_paths)} measurement files ({Path(last_paths[0]).name})"
            except Exception:
               self._last_measurement_label = "Last measurement"
      except Exception:
         pass
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
      self._close_mc_saver()
      self._measurement_id = None

   def _set_playback_speed(self, speed: float) -> None:
      try:
         s = float(speed)
         if s <= 0:
            s = 1.0
      except Exception:
         s = 1.0
      self._playback_speed = s
      try:
         if hasattr(self, "live_tab") and hasattr(self.live_tab, "set_playback_speed"):
            self.live_tab.set_playback_speed(s)
      except Exception:
         pass

   def _build_replay_channels_from_paths(self, paths: list[str]) -> dict[str, np.ndarray]:
      """Parse and merge channels from saved file paths for replay."""
      merged: dict[str, np.ndarray] = {}
      if not paths:
         return merged
      for p in paths:
         try:
            _mid, channels, _layout = self.live_tab._parse_data_file(str(p))
         except Exception:
            continue
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
      return merged

   def _on_play_action(self) -> None:
      """Play loaded source first; fall back to last completed measurement."""
      try:
         loaded = self.live_tab.get_loaded_replay_source() if hasattr(self.live_tab, "get_loaded_replay_source") else None
      except Exception:
         loaded = None

      channels = None
      label = None
      if loaded is not None:
         try:
            channels, label, _paths = loaded
         except Exception:
            channels = None
            label = None
      elif self._last_measurement_paths:
         channels = self._build_replay_channels_from_paths(list(self._last_measurement_paths))
         label = self._last_measurement_label or "Last measurement"

      if not channels:
         try:
            self.statusBar().showMessage("No source to play.", 6000)
         except Exception:
            pass
         return

      try:
         ok = self.live_tab.start_playback(channels, str(label or "source"), speed=self._playback_speed)
      except Exception:
         ok = False

      if not ok:
         try:
            self.statusBar().showMessage("Failed to start playback.", 6000)
         except Exception:
            pass

   def _on_stop_play_action(self) -> None:
      """Stop/pause current playback."""
      try:
         if hasattr(self.live_tab, "stop_playback"):
            self.live_tab.stop_playback()
      except Exception:
         pass

   def _close_mc_saver(self):
      """Close and remove the active MultiChannelSaver (if any)."""
      saver = getattr(self, "_mc_saver", None)
      self._mc_saver = None
      if saver is not None:
         try:
            saver.close()
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

   def _project_root_dir(self) -> Path:
      """Return the repository root directory for this application."""
      try:
         return Path(__file__).resolve().parents[1]
      except Exception:
         return Path.cwd()

   def _set_comport_mode_for_all(self, detector_obj, mode: int | None = None) -> None:
      """Apply ComPort mode to one detector or a detector list.

      If mode is None, each detector keeps its current `mode` attribute.
      """
      try:
         dets = detector_obj if isinstance(detector_obj, list) else [detector_obj]
      except Exception:
         dets = [detector_obj]

      for d in dets:
         if d is None or not hasattr(d, "set_mode"):
            continue
         try:
            target_mode = mode if mode is not None else getattr(d, "mode", None)
            if target_mode is None:
               continue
            d.set_mode(int(target_mode))
         except Exception as e:
            try:
               logger.warning(
                  "Failed to set detector mode for %s: %s",
                  getattr(d, "name", getattr(d, "port", "detector")),
                  e,
               )
            except Exception:
               pass

   def _connect_detector_errors(self, detector_obj) -> None:
      """Show detector errors in the status bar when the detector exposes an error signal."""
      try:
         dets = detector_obj if isinstance(detector_obj, list) else [detector_obj]
      except Exception:
         dets = [detector_obj]

      for d in dets:
         if d is None or not hasattr(d, "error"):
            continue
         try:
            d.error.connect(lambda msg, det=d: self.statusBar().showMessage(
               f"Detector error ({getattr(det, 'name', getattr(det, 'port', 'detector'))}): {msg}",
               8000,
            ))
         except Exception:
            try:
               logger.warning(
                  "Failed to connect detector error signal for %s",
                  getattr(d, "name", getattr(d, "port", "detector")),
               )
            except Exception:
               pass

   def _project_data_dir(self) -> Path:
      """Return the default data directory under the repository root."""
      p = self._project_root_dir() / "data"
      try:
         p.mkdir(parents=True, exist_ok=True)
      except Exception:
         pass
      return p

   def _project_experiments_dir(self) -> Path:
      """Return the default experiments directory under the repository root."""
      p = self._project_root_dir() / "experiments"
      try:
         p.mkdir(parents=True, exist_ok=True)
      except Exception:
         pass
      return p

   def _resolve_output_dir(self, raw_path: object | None = None, *, coerce_legacy_data_path: bool = False) -> Path:
      """Resolve an output directory, defaulting to the project data folder."""
      text = ""
      try:
         text = str(raw_path).strip() if raw_path is not None else ""
      except Exception:
         text = ""

      if text:
         try:
            p = Path(text).expanduser()
            if coerce_legacy_data_path and p.is_absolute() and p.name.lower() == "data":
               try:
                  p.relative_to(self._project_root_dir())
               except Exception:
                  p = self._project_data_dir()
            if not p.is_absolute():
               p = self._project_root_dir() / p
         except Exception:
            p = self._project_data_dir()
      else:
         p = self._project_data_dir()

      try:
         p.mkdir(parents=True, exist_ok=True)
      except Exception:
         pass
      return p


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
            "stage_control": getattr(self, "stage_control_dock", None),
            "excitation_control": getattr(self, "excitation_control_dock", None),
            "stage_calibration": getattr(self, "stage_calibration_dock", None),
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


   def _install_dock_context_menu(self, dock: QtWidgets.QDockWidget, dock_id: str) -> None:
      """Install a context menu on a dock widget with space-filling options."""
      try:
         dock.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
         dock.customContextMenuRequested.connect(
            lambda pos: self._show_dock_context_menu(dock, dock_id, pos)
         )
         
         # Install double-click handler for title bar
         title_bar = dock.titleBarWidget()
         if title_bar is not None:
            title_bar.mouseDoubleClickEvent = lambda event: self._toggle_dock_maximize(dock, dock_id)
      except Exception:
         pass

   def _show_dock_context_menu(self, dock: QtWidgets.QDockWidget, dock_id: str, pos) -> None:
      """Show context menu for dock widget with space-filling options."""
      try:
         menu = QtWidgets.QMenu(self)
         
         fill_action = menu.addAction("Fill Available Space")
         fill_action.triggered.connect(lambda: self._fill_dock_space(dock, dock_id))
         
         maximize_action = menu.addAction("Maximize Panel")
         maximize_action.triggered.connect(lambda: self._toggle_dock_maximize(dock, dock_id))
         
         restore_action = menu.addAction("Restore Normal Size")
         restore_action.triggered.connect(lambda: self._restore_dock_size(dock, dock_id))
         
         menu.addSeparator()
         
         auto_arrange_action = menu.addAction("Auto-Arrange All Panels")
         auto_arrange_action.triggered.connect(self._auto_arrange_visible_panels)
         
         menu.exec(dock.mapToGlobal(pos))
      except Exception:
         pass

   def _fill_dock_space(self, dock: QtWidgets.QDockWidget, dock_id: str) -> None:
      """Make the dock expand to fill available space in its area."""
      try:
         # Get the dock's current geometry
         current_geo = dock.geometry()
         
         # Find sibling docks in the same area
         area = self.dockWidgetArea(dock)
         sibling_docks = self.findChildren(QtWidgets.QDockWidget)
         area_siblings = [d for d in sibling_docks if d != dock and self.dockWidgetArea(d) == area and d.isVisible()]
         
         if not area_siblings:
            # No siblings, already filling space
            return
         
         # Calculate total available space and distribute
         main_window_size = self.size()
         
         # Resize to fill significant portion of available space
         if area in (QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, QtCore.Qt.DockWidgetArea.RightDockWidgetArea):
            # Vertical dock - expand horizontally
             target_width = int(main_window_size.width() * 0.4)  # 40% of window width
             self.resizeDocks([dock] + area_siblings, [target_width] + [200] * len(area_siblings), QtCore.Qt.Orientation.Horizontal)
         else:
            # Horizontal dock - expand vertically
            target_height = int(main_window_size.height() * 0.4)  # 40% of window height
            self.resizeDocks([dock] + area_siblings, [target_height] + [200] * len(area_siblings), QtCore.Qt.Orientation.Vertical)
            
      except Exception:
         pass

   def _toggle_dock_maximize(self, dock: QtWidgets.QDockWidget, dock_id: str) -> None:
      """Toggle dock between normal and maximized size."""
      try:
         # Check if currently maximized (heuristic: if it's very large)
         current_size = dock.size()
         window_size = self.size()
         
         is_maximized = (current_size.width() > window_size.width() * 0.7 or 
                        current_size.height() > window_size.height() * 0.7)
         
         if is_maximized:
            self._restore_dock_size(dock, dock_id)
         else:
            self._maximize_dock(dock, dock_id)
      except Exception:
         pass

   def _maximize_dock(self, dock: QtWidgets.QDockWidget, dock_id: str) -> None:
      """Maximize the dock to fill most of the window."""
      try:
         # Hide other docks temporarily
         all_docks = self.findChildren(QtWidgets.QDockWidget)
         other_docks = [d for d in all_docks if d != dock and d.isVisible()]
         
         # Store visibility state
         if not hasattr(self, '_maximize_state'):
            self._maximize_state = {}
         self._maximize_state[dock_id] = [d.isVisible() for d in other_docks]
         
         for d in other_docks:
            d.setVisible(False)
         
         # Make dock floating and maximize
         dock.setFloating(True)
         dock.showMaximized()
         
      except Exception:
         pass

   def _restore_dock_size(self, dock: QtWidgets.QDockWidget, dock_id: str) -> None:
      """Restore dock to normal size and show other docks."""
      try:
         # Restore from floating state
         if dock.isFloating():
            dock.setFloating(False)
            dock.showNormal()
         
         # Restore other docks visibility
         if hasattr(self, '_maximize_state') and dock_id in self._maximize_state:
            all_docks = self.findChildren(QtWidgets.QDockWidget)
            other_docks = [d for d in all_docks if d != dock]
            visibility_states = self._maximize_state[dock_id]
            
            for i, d in enumerate(other_docks):
               if i < len(visibility_states):
                  d.setVisible(visibility_states[i])
            
            del self._maximize_state[dock_id]
         
         # Auto-arrange to restore sensible layout
         self._auto_arrange_visible_panels()
         
      except Exception:
         pass

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
            ("stage_control", getattr(self, "stage_control_dock", None)),
            ("excitation_control", getattr(self, "excitation_control_dock", None)),
            ("stage_calibration", getattr(self, "stage_calibration_dock", None)),
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
      try:
         self.demo_tab.output_dir_edit.setText(str(self._project_data_dir()))
      except Exception:
         pass
      self.multi_tab = MultiAxisTab(config_path=self._config_path)
      self.multiviewctl_tab = MultiViewControlTab(config_path=self._config_path)

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
      self._install_dock_context_menu(self.demo_dock, "demo")
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
      self._install_dock_context_menu(self.multi_dock, "multiaxis")
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
      self._install_dock_context_menu(self.multiviewctl_dock, "multiviewctl")
      self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.multiviewctl_dock)

      # Enable tabbed docking and splitting for all docks
      self.setDockNestingEnabled(True)

      # --- LiveTab instance (we will dock its subwidgets) ---
      self.live_tab = LiveTab()
      # connect hover info to status bar
      self.live_tab.hover_info.connect(lambda s: self.statusBar().showMessage(s))
      # connect load/save status messages to status bar
      self.live_tab.status_message.connect(lambda msg, ms: self.statusBar().showMessage(msg, ms))
      try:
         self.live_tab.plugin_movement_commands.connect(self._execute_plugin_movement_commands)
      except Exception:
         pass
      try:
         self._measurement_status_label = QtWidgets.QLabel(self)
         self._measurement_status_label.setMinimumWidth(180)
         self.statusBar().addPermanentWidget(self._measurement_status_label)
      except Exception:
         self._measurement_status_label = None
      
      # Add stage position label to status bar
      try:
         self._stage_position_label = QtWidgets.QLabel(self)
         self._stage_position_label.setMinimumWidth(150)
         self._stage_position_label.setText("Stage: --")
         self.statusBar().addPermanentWidget(self._stage_position_label)
      except Exception:
         self._stage_position_label = None
      
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
         self._install_dock_context_menu(self.cam_dock, "camera")
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
         self._install_dock_context_menu(self.camctl_dock, "camctl")
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
         self._install_dock_context_menu(self.multiview_dock, "multiview")
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
         self._install_dock_context_menu(self.detimg_dock, "detimg")
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
         self._install_dock_context_menu(self.plot_dock, "plot")
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
         self._install_dock_context_menu(self.detctl_dock, "detctl")
         self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.detctl_dock)

         # Stage Control dock
         try:
            from gui.tabs.move_motors_tab import StageControlTab
         except ImportError:
            try:
               from tabs.move_motors_tab import StageControlTab
            except Exception:
               logger.warning("Could not import StageControlTab")
               StageControlTab = None
         
         if StageControlTab is not None:
            self.stage_control_tab = StageControlTab(config_path=self._config_path)
            self.stage_control_dock = QtWidgets.QDockWidget("Stage", self)
            self.stage_control_dock.setObjectName("dock_stage_control")
            self.stage_control_dock.setWidget(self.stage_control_tab)
            self.stage_control_dock.setAllowedAreas(
               QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
               QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
               QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
               QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            )
            self.stage_control_dock.setFeatures(
               QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
               QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
               QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
            self._install_dock_context_menu(self.stage_control_dock, "stage_control")
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.stage_control_dock)
            # Initially hide it
            self.stage_control_dock.setVisible(False)
            # Connect dock close event to cleanup
            self.stage_control_dock.closeEvent = lambda e: self._cleanup_stage_control(e)
            logger.info("Stage Control dock created successfully")
         else:
            self.stage_control_tab = None
            self.stage_control_dock = None
            logger.warning("Stage Control dock could not be created - StageControlTab import failed")

         # Excitation Control dock
         try:
            from gui.tabs.excitation_control_tab import ExcitationControlTab
         except ImportError:
            try:
               from tabs.excitation_control_tab import ExcitationControlTab
            except Exception:
               logger.warning("Could not import ExcitationControlTab")
               ExcitationControlTab = None
         
         if ExcitationControlTab is not None:
            # Create with empty device list initially
            self.excitation_control_tab = ExcitationControlTab([])
            self.excitation_control_dock = QtWidgets.QDockWidget("Excitation Control", self)
            self.excitation_control_dock.setObjectName("dock_excitation_control")
            self.excitation_control_dock.setWidget(self.excitation_control_tab)
            try:
               self.excitation_control_dock.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Maximum)
               self.excitation_control_dock.setMaximumHeight(200)  # Keep it compact
            except Exception:
               pass
            self.excitation_control_dock.setAllowedAreas(
               QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
               QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
               QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
               QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            )
            self.excitation_control_dock.setFeatures(
               QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
               QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
               QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
            self._install_dock_context_menu(self.excitation_control_dock, "excitation_control")
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.excitation_control_dock)
            # Initially hide it
            self.excitation_control_dock.setVisible(False)
            logger.info("Excitation Control dock created successfully")
         else:
            self.excitation_control_tab = None
            self.excitation_control_dock = None
            logger.warning("Excitation Control dock could not be created - ExcitationControlTab import failed")

         # Stage Calibration dock
         try:
            from gui.tabs.stage_calibration_tab import StageCalibrationTab
         except ImportError:
            try:
               from tabs.stage_calibration_tab import StageCalibrationTab
            except Exception:
               logger.warning("Could not import StageCalibrationTab")
               StageCalibrationTab = None
         
         if StageCalibrationTab is not None:
            self.stage_calibration_tab = StageCalibrationTab(config_path=self._config_path)
            self.stage_calibration_dock = QtWidgets.QDockWidget("Stage Calibration", self)
            self.stage_calibration_dock.setObjectName("dock_stage_calibration")
            self.stage_calibration_dock.setWidget(self.stage_calibration_tab)
            self.stage_calibration_dock.setAllowedAreas(
               QtCore.Qt.DockWidgetArea.LeftDockWidgetArea |
               QtCore.Qt.DockWidgetArea.RightDockWidgetArea |
               QtCore.Qt.DockWidgetArea.TopDockWidgetArea |
               QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
            )
            self.stage_calibration_dock.setFeatures(
               QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable |
               QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable |
               QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
            self._install_dock_context_menu(self.stage_calibration_dock, "stage_calibration")
            self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.stage_calibration_dock)
            # Initially hide it
            self.stage_calibration_dock.setVisible(False)
            
            # Connect calibration saved signal
            self.stage_calibration_tab.calibration_saved.connect(self._on_calibration_saved)
            logger.info("Stage Calibration dock created successfully")
         else:
            self.stage_calibration_tab = None
            self.stage_calibration_dock = None
            logger.warning("Stage Calibration dock could not be created - StageCalibrationTab import failed")

         # connect view change signals so docks can be shown/hidden as view changes
         try:
            self.live_tab.view_changed.connect(self._on_live_view_changed)
         except Exception:
            pass

         # Add new control docks to View menu (must be after dock creation)
         self._add_new_docks_to_view_menu()

      except Exception as e:
         logger.exception("Critical error during dock creation: %s", e)
         QtWidgets.QMessageBox.critical(self, "Initialization Error", 
            f"Failed to create dock panels: {e}")

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

      try:
         if hasattr(self.multi_tab, 'detector_offset_toggled'):
            self.multi_tab.detector_offset_toggled.connect(self._on_detector_offset_toggled)
         if hasattr(self.multi_tab, 'detector_offset_value_changed'):
            self.multi_tab.detector_offset_value_changed.connect(self._on_detector_offset_value_changed)
      except Exception:
         pass

      # Live-update the plot X axis when the Default X Axis combo changes.
      try:
         if hasattr(self.multi_tab, 'xaxis_changed'):
            def _on_xaxis_changed(name: str):
               try:
                  # Always store as the run-scoped preference so every future
                  # combo rebuild during a run will consistently re-apply it.
                  if hasattr(self.live_tab, 'set_preferred_plot_xaxis'):
                     self.live_tab.set_preferred_plot_xaxis(name)
                  # If multi_coords already has data (e.g. post-run browsing),
                  # immediately apply the axis and force a re-render.
                  if hasattr(self.live_tab, 'multi_coords') and self.live_tab.multi_coords:
                     if hasattr(self.live_tab, 'set_xaxis'):
                        self.live_tab.set_xaxis(name)
                     self.live_tab._multi_dirty = True
                     self.live_tab._last_multi_render = 0.0
               except Exception:
                  pass
            self.multi_tab.xaxis_changed.connect(_on_xaxis_changed)
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

   def _start_stage_position_timer(self) -> None:
      """Start the stage position update timer (500ms interval).
      
      Only updates when multiaxis is not running to avoid resource waste.
      """
      if self._stage_position_timer is None:
         self._stage_position_timer = QtCore.QTimer(self)
         self._stage_position_timer.setInterval(500)  # 500ms as required
         self._stage_position_timer.timeout.connect(self._update_stage_position_display)
         self._stage_position_timer.start()
         try:
            logger.info("Stage position timer started (500ms interval)")
         except Exception:
            pass
      else:
         # Timer already exists, just start it if not running
         if not self._stage_position_timer.isActive():
            self._stage_position_timer.start()
            try:
               logger.info("Stage position timer restarted (500ms interval)")
            except Exception:
               pass

   def _stop_stage_position_timer(self) -> None:
      """Stop the stage position update timer."""
      if self._stage_position_timer is not None:
         try:
            self._stage_position_timer.stop()
            logger.info("Stage position timer stopped")
         except Exception:
            pass

   def _update_stage_position_display(self) -> None:
      """Update the stage position display in the status bar.
      
      Updates position regardless of multiaxis state, but uses different timing:
      - When multiaxis is running: position is updated by the multiaxis system itself
      - When multiaxis is not running: this timer updates position every 500ms
      """
      try:
         stage = getattr(self, 'stage', None)
         if stage is None:
            try:
               logger.debug("Stage object is None - skipping position update")
            except Exception:
               pass
            self._set_stage_position_text("Stage: --")
            return
         
         if not hasattr(stage, 'get_position'):
            try:
               logger.debug("Stage does not have get_position method - skipping position update")
            except Exception:
               pass
            self._set_stage_position_text("Stage: --")
            return
         
         # Get current position with exception handling
         try:
            position = stage.get_position()
            try:
               logger.debug("Stage position retrieved: %s", position)
            except Exception:
               pass
               
            if isinstance(position, (tuple, list)) and len(position) >= 2:
               x, y = float(position[0]), float(position[1])
               self._set_stage_position_text(f"Stage: X={x:.2f}, Y={y:.2f}")
            else:
               # Handle unexpected position format
               try:
                  logger.warning("Unexpected position format: %s (type: %s)", position, type(position))
               except Exception:
                  pass
               self._set_stage_position_text("Stage: --")
         except Exception as pos_error:
            # Position retrieval failed - show error state without crashing
            try:
               logger.warning("Failed to get stage position: %s", pos_error)
            except Exception:
               pass
            self._set_stage_position_text("Stage: error")
            
      except Exception as e:
         # Any other error - don't crash the status bar
         try:
            logger.warning("Error updating stage position display: %s", e)
         except Exception:
            pass
         self._set_stage_position_text("Stage: --")

   def _set_stage_position_text(self, text: str) -> None:
      """Set the stage position label text with thread-safe UI update."""
      try:
         label = getattr(self, '_stage_position_label', None)
         if label is not None:
            label.setText(text)
      except Exception:
         pass

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

      load_hw_cfg = QAction("Load Hardware Config…", self)
      load_hw_cfg.triggered.connect(self.load_hardware_config)
      file_menu.addAction(load_hw_cfg)
      file_menu.addSeparator()

      save_exp = QAction("Save Experiment", self)
      load_exp = QAction("Load Experiment", self)

      save_exp.triggered.connect(self.save_full_experiment)
      load_exp.triggered.connect(self.load_full_experiment)

      file_menu.addAction(save_exp)
      file_menu.addAction(load_exp)
      file_menu.addSeparator()

      load_data_action = QAction("Load Data…", self)
      load_data_action.setShortcut("Ctrl+O")
      load_data_action.triggered.connect(lambda: self.live_tab._on_load_data())
      file_menu.addAction(load_data_action)

      keep_layout_action = QAction("Keep Current Layout When Loading", self)
      keep_layout_action.setCheckable(True)
      keep_layout_action.setChecked(True)
      keep_layout_action.setToolTip(
          "When checked, the current view mode, X axis, Z slice, draw-lines and\n"
          "window size are preserved after loading data.\n"
          "When unchecked, the layout is inferred from the loaded data."
      )
      # Keep in sync with the toolbar button's menu action on live_tab
      def _sync_keep_layout(checked: bool):
          try:
              self.live_tab.keep_layout_action.setChecked(checked)
          except Exception:
              pass
      keep_layout_action.toggled.connect(_sync_keep_layout)
      # Also sync back when live_tab's own action is toggled
      try:
          self.live_tab.keep_layout_action.toggled.connect(keep_layout_action.setChecked)
      except Exception:
          pass
      file_menu.addAction(keep_layout_action)

      save_multichannel_action = QAction("Save as Multi-Channel File", self)
      save_multichannel_action.setCheckable(True)
      save_multichannel_action.setChecked(True)   # mirrors demo_tab default
      save_multichannel_action.setToolTip(
          "When checked, all detector streams are saved into a single\n"
          "multi-channel HDF5 file that can be loaded in one go.\n"
          "When unchecked, each detector gets its own .h5/.txt pair."
      )
      # Bidirectional sync with demo_tab.multichannel_cb
      def _sync_mc_to_cb(checked: bool):
          try:
              self.demo_tab.multichannel_cb.setChecked(checked)
          except Exception:
              pass
      save_multichannel_action.toggled.connect(_sync_mc_to_cb)
      try:
          self.demo_tab.multichannel_cb.toggled.connect(save_multichannel_action.setChecked)
      except Exception:
          pass
      file_menu.addAction(save_multichannel_action)
      file_menu.addSeparator()

      export_channels_action = QAction("Export All Channels…", self)
      export_channels_action.setShortcut("Ctrl+Shift+E")
      export_channels_action.triggered.connect(lambda: self.live_tab.export_all_channels())
      file_menu.addAction(export_channels_action)
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

      # --- Play menu ---
      play_menu = menubar.addMenu("&Play")

      play_action = QAction("Play", self)
      play_action.setShortcut("Ctrl+D")
      play_action.triggered.connect(self._on_play_action)
      play_menu.addAction(play_action)

      stop_play_action = QAction("Stop", self)
      stop_play_action.setShortcut("Ctrl+E")
      stop_play_action.triggered.connect(self._on_stop_play_action)
      play_menu.addAction(stop_play_action)

      speed_menu = play_menu.addMenu("Speed")
      speed_group = QActionGroup(self)
      speed_group.setExclusive(True)
      for txt, val in (("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("5x", 5.0)):
         act = QAction(txt, self)
         act.setCheckable(True)
         if abs(val - 1.0) < 1e-9:
            act.setChecked(True)
         act.triggered.connect(lambda checked, s=val: self._set_playback_speed(s) if checked else None)
         speed_group.addAction(act)
         speed_menu.addAction(act)

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

      except Exception as e:
         logger.exception("Error creating View menu: %s", e)
         pass

      # --- Tools menu ---
      tools_menu = menubar.addMenu("&Tools")
      
      add_plugins_action = QAction("Add Plugins…", self)
      add_plugins_action.triggered.connect(self._add_plugins)
      tools_menu.addAction(add_plugins_action)
      
      manage_plugins_action = QAction("Manage Plugins…", self)
      manage_plugins_action.triggered.connect(self._manage_plugins)
      tools_menu.addAction(manage_plugins_action)
      
      tools_menu.addSeparator()
      
      plugin_info_action = QAction("Plugin Info…", self)
      plugin_info_action.triggered.connect(self._show_plugin_info)
      tools_menu.addAction(plugin_info_action)

      # Help should be the last menu on the menubar.
      help_menu = menubar.addMenu("&Help")
      about_action = QAction("About", self)
      about_action.triggered.connect(self._show_about)
      help_menu.addAction(about_action)
   
   def _add_plugins(self):
      """Add plugins dialog."""
      if not PLUGINS_AVAILABLE or self._plugin_manager is None:
         QtWidgets.QMessageBox.warning(self, "Plugins Not Available", 
            "Plugin system is not available.")
         return
      
      file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
         self,
         "Load Plugin",
         str(Path.home()),
         "Python Files (*.py);;All Files (*)"
      )
      
      if not file_path:
         return
      
      plugin_name = Path(file_path).stem
      success = self._plugin_manager.load_plugin(plugin_name, Path(file_path))
      
      if success:
         QtWidgets.QMessageBox.information(self, "Plugin Loaded", 
            f"Successfully loaded plugin: {plugin_name}")
      else:
         QtWidgets.QMessageBox.warning(self, "Plugin Load Failed", 
            f"Failed to load plugin: {plugin_name}")
   
   def _manage_plugins(self):
      """Manage loaded plugins dialog."""
      if not PLUGINS_AVAILABLE or self._plugin_manager is None:
         QtWidgets.QMessageBox.warning(self, "Plugins Not Available", 
            "Plugin system is not available.")
         return
      
      # Create plugin management dialog
      dialog = QtWidgets.QDialog(self)
      dialog.setWindowTitle("Manage Plugins")
      dialog.setMinimumSize(500, 400)
      
      layout = QtWidgets.QVBoxLayout(dialog)
      
      # Plugin list
      plugin_list = QtWidgets.QListWidget()
      plugins = self._plugin_manager.get_all_plugin_info()
      
      for plugin_name, info in plugins.items():
         item = QtWidgets.QListWidgetItem(f"{plugin_name} ({info.get('version', 'N/A')})")
         item.setData(QtCore.Qt.ItemDataRole.UserRole, plugin_name)
         
         # Add enabled/disabled indicator
         status = "✓" if info.get('enabled', False) else "✗"
         item.setText(f"{status} {item.text()}")
         
         plugin_list.addItem(item)
      
      layout.addWidget(plugin_list)
      
      # Buttons
      button_layout = QtWidgets.QHBoxLayout()
      
      enable_btn = QtWidgets.QPushButton("Enable Selected")
      disable_btn = QtWidgets.QPushButton("Disable Selected")
      configure_btn = QtWidgets.QPushButton("Configure Selected")
      remove_btn = QtWidgets.QPushButton("Remove Selected")
      close_btn = QtWidgets.QPushButton("Close")
      
      button_layout.addWidget(enable_btn)
      button_layout.addWidget(disable_btn)
      button_layout.addWidget(configure_btn)
      button_layout.addWidget(remove_btn)
      button_layout.addStretch()
      button_layout.addWidget(close_btn)
      
      layout.addLayout(button_layout)
      
      # Define refresh function first
      def _refresh_plugin_list(list_widget):
         list_widget.clear()
         plugins = self._plugin_manager.get_all_plugin_info()
         for plugin_name, info in plugins.items():
            item = QtWidgets.QListWidgetItem(f"{plugin_name} ({info.get('version', 'N/A')})")
            item.setData(QtCore.Qt.ItemDataRole.UserRole, plugin_name)
            status = "✓" if info.get('enabled', False) else "✗"
            item.setText(f"{status} {item.text()}")
            list_widget.addItem(item)
      
      # Button connections
      def enable_selected():
         current_item = plugin_list.currentItem()
         if current_item:
            plugin_name = current_item.data(QtCore.Qt.ItemDataRole.UserRole)
            self._plugin_manager.enable_plugin(plugin_name)
            _refresh_plugin_list(plugin_list)
      
      def disable_selected():
         current_item = plugin_list.currentItem()
         if current_item:
            plugin_name = current_item.data(QtCore.Qt.ItemDataRole.UserRole)
            self._plugin_manager.disable_plugin(plugin_name)
            _refresh_plugin_list(plugin_list)
      
      def configure_selected():
         current_item = plugin_list.currentItem()
         if current_item:
            plugin_name = current_item.data(QtCore.Qt.ItemDataRole.UserRole)
            # Show configuration dialog (simplified)
            QtWidgets.QMessageBox.information(self, "Configure Plugin", 
               f"Configuration dialog for {plugin_name} would appear here.")
      
      def remove_selected():
         current_item = plugin_list.currentItem()
         if current_item:
            plugin_name = current_item.data(QtCore.Qt.ItemDataRole.UserRole)
            reply = QtWidgets.QMessageBox.question(self, "Remove Plugin", 
               f"Are you sure you want to remove plugin: {plugin_name}?",
               QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            
            if reply == QtWidgets.QMessageBox.StandardButton.Yes:
               self._plugin_manager.unload_plugin(plugin_name)
               _refresh_plugin_list(plugin_list)
      
      enable_btn.clicked.connect(enable_selected)
      disable_btn.clicked.connect(disable_selected)
      configure_btn.clicked.connect(configure_selected)
      remove_btn.clicked.connect(remove_selected)
      close_btn.clicked.connect(dialog.accept)
      
      dialog.exec()
   
   def _show_plugin_info(self):
      """Show information about loaded plugins."""
      if not PLUGINS_AVAILABLE or self._plugin_manager is None:
         QtWidgets.QMessageBox.warning(self, "Plugins Not Available", 
            "Plugin system is not available.")
         return
      
      plugins = self._plugin_manager.get_all_plugin_info()
      
      if not plugins:
         QtWidgets.QMessageBox.information(self, "Plugin Info", "No plugins loaded.")
         return
      
      info_text = "Loaded Plugins:\n\n"
      for plugin_name, info in plugins.items():
         info_text += f"Name: {info.get('name', plugin_name)}\n"
         info_text += f"Version: {info.get('version', 'N/A')}\n"
         info_text += f"Description: {info.get('description', 'No description')}\n"
         info_text += f"Author: {info.get('author', 'Unknown')}\n"
         info_text += f"Enabled: {info.get('enabled', False)}\n"
         info_text += f"Required Detectors: {', '.join(info.get('required_detectors', []))}\n"
         info_text += f"Required Axes: {', '.join(info.get('required_axes', []))}\n"
         info_text += "-" * 40 + "\n"
      
      QtWidgets.QMessageBox.information(self, "Plugin Information", info_text)
   
   def _execute_plugin_movement_commands(self, commands: list) -> bool:
      """Execute movement commands from plugins.
      
      Args:
          commands: List of movement command dictionaries
          
      Returns:
          True if execution successful, False otherwise
      """
      if not commands:
         logger.warning("No movement commands to execute (empty list)")
         return False
      
      logger.info(f"Executing {len(commands)} plugin movement commands")
      
      try:
         # Collect target positions for all axes
         target_x = None
         target_y = None
         target_z = None
         
         # Get current positions first
         current_x = 0.0
         current_y = 0.0
         current_z = 0.0
         
         try:
            if self.stage and hasattr(self.stage, 'get_position'):
               pos = self.stage.get_position()
               current_x = pos[0] if len(pos) > 0 else 0.0
               current_y = pos[1] if len(pos) > 1 else 0.0
               logger.info(f"Current stage position: X={current_x:.3f}, Y={current_y:.3f}")
         except Exception as e:
            logger.warning(f"Failed to get current stage position: {e}")
         
         try:
            if self.focus and hasattr(self.focus, 'get_position'):
               current_z = self.focus.get_position()
               logger.info(f"Current focus position: Z={current_z:.3f}")
         except Exception as e:
            logger.warning(f"Failed to get current focus position: {e}")
         
         # Process commands to determine target positions
         for cmd in commands:
            axis = cmd.get("axis", "").lower()
            position = cmd.get("position", 0.0)
            relative = cmd.get("relative", False)
            
            logger.info(f"Processing command: axis={axis}, position={position:.3f}, relative={relative}")
            
            if axis == "x":
               if relative:
                  target_x = current_x + position
                  logger.info(f"  Target X (relative): {target_x:.3f}")
               else:
                  target_x = position
                  logger.info(f"  Target X (absolute): {target_x:.3f}")
            
            elif axis == "y":
               if relative:
                  target_y = current_y + position
                  logger.info(f"  Target Y (relative): {target_y:.3f}")
               else:
                  target_y = position
                  logger.info(f"  Target Y (absolute): {target_y:.3f}")
            
            elif axis == "z":
               if relative:
                  target_z = current_z + position
                  logger.info(f"  Target Z (relative): {target_z:.3f}")
               else:
                  target_z = position
                  logger.info(f"  Target Z (absolute): {target_z:.3f}")
         
         # Execute movements
         # For X and Y, move together if both are specified
         if target_x is not None or target_y is not None:
            if self.stage and hasattr(self.stage, 'move_to'):
               # Use current position if not specified
               final_x = target_x if target_x is not None else current_x
               final_y = target_y if target_y is not None else current_y
               
               logger.info(f"Calling stage.move_to(X={final_x:.3f}, Y={final_y:.3f})")
               
               # Convert to steps if needed
               try:
                  from gui.tabs.move_motors_tab import StageControlTab
                  # This conversion logic should match what's in move_motors_tab
                  # For now, just call move_to directly
                  self.stage.move_to(final_x, final_y)
                  logger.info(f"Plugin moved stage to X={final_x:.3f}, Y={final_y:.3f}")
               except Exception as e:
                  logger.warning(f"Failed to move stage: {e}")
                  import traceback
                  traceback.print_exc()
            else:
               logger.warning("Stage not available or does not have move_to method")
         
         # For Z, move separately
         if target_z is not None:
            if self.focus and hasattr(self.focus, 'move_to'):
               try:
                  logger.info(f"Calling focus.move_to(Z={target_z:.3f})")
                  self.focus.move_to(target_z)
                  logger.info(f"Plugin moved Z axis to {target_z:.3f}")
               except Exception as e:
                  logger.warning(f"Failed to move Z axis: {e}")
                  import traceback
                  traceback.print_exc()
            else:
               logger.warning("Focus not available or does not have move_to method")
         
         return True
      except Exception as e:
         logger.exception(f"Error executing plugin movement commands: {e}")
         return False
   
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
      """Apply a deterministic 'full' layout with all panels docked and visible.
      
      Supports 2-4 column layouts based on the number of available panels.
      """
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
            getattr(self, "stage_control_dock", None),
            getattr(self, "excitation_control_dock", None),
            getattr(self, "stage_calibration_dock", None),
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

         # Determine number of columns based on available panel count
         num_docks = len(docks)
         if num_docks <= 6:
            num_columns = 2
         elif num_docks <= 9:
            num_columns = 3
         else:
            num_columns = 4

         if num_columns == 2:
            # Traditional 2-column layout
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

         elif num_columns == 3:
            # 3-column layout: controls | primary images | secondary images
            # Column 1: Multi-Axis / Strip Chart
            if getattr(self, "multi_dock", None) is not None:
               self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.multi_dock)
            if getattr(self, "demo_dock", None) is not None and getattr(self, "multi_dock", None) is not None:
               self.splitDockWidget(self.multi_dock, self.demo_dock, QtCore.Qt.Orientation.Vertical)

            # Column 2: Camera / Detector Images
            if getattr(self, "cam_dock", None) is not None:
               self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.cam_dock)
            if getattr(self, "detimg_dock", None) is not None and getattr(self, "cam_dock", None) is not None:
               self.splitDockWidget(self.cam_dock, self.detimg_dock, QtCore.Qt.Orientation.Vertical)

            # Column 3: Multi View / Controls / Plot
            if getattr(self, "multiview_dock", None) is not None and getattr(self, "cam_dock", None) is not None:
               self.splitDockWidget(self.cam_dock, self.multiview_dock, QtCore.Qt.Orientation.Horizontal)
            if getattr(self, "camctl_dock", None) is not None and getattr(self, "multiview_dock", None) is not None:
               self.splitDockWidget(self.multiview_dock, self.camctl_dock, QtCore.Qt.Orientation.Vertical)
            if getattr(self, "plot_dock", None) is not None and getattr(self, "camctl_dock", None) is not None:
               self.splitDockWidget(self.camctl_dock, self.plot_dock, QtCore.Qt.Orientation.Vertical)
            if getattr(self, "detctl_dock", None) is not None and getattr(self, "plot_dock", None) is not None:
               self.splitDockWidget(self.plot_dock, self.detctl_dock, QtCore.Qt.Orientation.Vertical)
            if getattr(self, "multiviewctl_dock", None) is not None and getattr(self, "detctl_dock", None) is not None:
               self.splitDockWidget(self.detctl_dock, self.multiviewctl_dock, QtCore.Qt.Orientation.Vertical)

         else:  # num_columns == 4
            # 4-column layout: primary controls | secondary controls | primary images | secondary images
            # Column 1: Multi-Axis / Strip Chart
            if getattr(self, "multi_dock", None) is not None:
               self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.multi_dock)
            if getattr(self, "demo_dock", None) is not None and getattr(self, "multi_dock", None) is not None:
               self.splitDockWidget(self.multi_dock, self.demo_dock, QtCore.Qt.Orientation.Vertical)

            # Column 2: Multi View Control / Camera Control / Detector Control
            if getattr(self, "multiviewctl_dock", None) is not None and getattr(self, "multi_dock", None) is not None:
               self.splitDockWidget(self.multi_dock, self.multiviewctl_dock, QtCore.Qt.Orientation.Horizontal)
            if getattr(self, "camctl_dock", None) is not None and getattr(self, "multiviewctl_dock", None) is not None:
               self.splitDockWidget(self.multiviewctl_dock, self.camctl_dock, QtCore.Qt.Orientation.Vertical)
            if getattr(self, "detctl_dock", None) is not None and getattr(self, "camctl_dock", None) is not None:
               self.splitDockWidget(self.camctl_dock, self.detctl_dock, QtCore.Qt.Orientation.Vertical)

            # Column 3: Camera / Detector Images
            if getattr(self, "cam_dock", None) is not None and getattr(self, "multiviewctl_dock", None) is not None:
               self.splitDockWidget(self.multiviewctl_dock, self.cam_dock, QtCore.Qt.Orientation.Horizontal)
            if getattr(self, "detimg_dock", None) is not None and getattr(self, "cam_dock", None) is not None:
               self.splitDockWidget(self.cam_dock, self.detimg_dock, QtCore.Qt.Orientation.Vertical)

            # Column 4: Multi View / Plot
            if getattr(self, "multiview_dock", None) is not None and getattr(self, "cam_dock", None) is not None:
               self.splitDockWidget(self.cam_dock, self.multiview_dock, QtCore.Qt.Orientation.Horizontal)
            if getattr(self, "plot_dock", None) is not None and getattr(self, "multiview_dock", None) is not None:
               self.splitDockWidget(self.multiview_dock, self.plot_dock, QtCore.Qt.Orientation.Vertical)

      except Exception:
         pass

   def _auto_arrange_visible_panels(self) -> None:
      """Arrange the currently visible docks into a sane layout.

      Heuristics:
      - Plot prefers a full-width horizontal dock at the top.
      - Control panels are stacked so they stay compact.
      - Image/preview panels go to the right.
      - Supports 2-4 column layouts based on visible panel count.
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
            ("stage_control", getattr(self, "stage_control_dock", None)),
            ("excitation_control", getattr(self, "excitation_control_dock", None)),
            ("stage_calibration", getattr(self, "stage_calibration_dock", None)),
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

         # Determine number of columns based on visible panel count
         num_visible = len(visible_docks)
         if num_visible <= 4:
            num_columns = 2
         elif num_visible <= 7:
            num_columns = 3
         else:
            num_columns = 4

         # Prefer Plot full-width at the top.
         plot = visible_docks.get("plot")
         if plot is not None:
            try:
               self.addDockWidget(QtCore.Qt.DockWidgetArea.TopDockWidgetArea, plot)
            except Exception:
               pass

         # Define column assignments based on column count
         if num_columns == 2:
            # Traditional 2-column layout
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

            # Size heuristics for 2-column layout
            try:
               if plot is not None:
                  self.resizeDocks([plot], [260], QtCore.Qt.Orientation.Vertical)
            except Exception:
               pass

            try:
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
                  self.resizeDocks([left_anchor, right_anchor], [320, 1000], QtCore.Qt.Orientation.Horizontal)
            except Exception:
               pass

         elif num_columns == 3:
            # 3-column layout: controls | primary images | secondary images
            col1_stack = [
               visible_docks.get("multiaxis"),
               visible_docks.get("demo"),
            ]
            col1_stack = [d for d in col1_stack if d is not None]
            
            col2_stack = [
               visible_docks.get("camera"),
               visible_docks.get("detimg"),
            ]
            col2_stack = [d for d in col2_stack if d is not None]
            
            col3_stack = [
               visible_docks.get("multiview"),
               visible_docks.get("multiviewctl"),
               visible_docks.get("camctl"),
               visible_docks.get("detctl"),
            ]
            col3_stack = [d for d in col3_stack if d is not None]

            # Create column 1 (left)
            if col1_stack:
               try:
                  self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, col1_stack[0])
               except Exception:
                  pass
               for d in col1_stack[1:]:
                  try:
                     self.splitDockWidget(col1_stack[0], d, QtCore.Qt.Orientation.Vertical)
                     col1_stack[0] = d
                  except Exception:
                     pass

            # Create column 2 (center)
            if col2_stack:
               try:
                  self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, col2_stack[0])
               except Exception:
                  pass
               for d in col2_stack[1:]:
                  try:
                     self.splitDockWidget(col2_stack[0], d, QtCore.Qt.Orientation.Vertical)
                     col2_stack[0] = d
                  except Exception:
                     pass

            # Create column 3 (right) by splitting from column 2
            if col3_stack and col2_stack:
               try:
                  self.splitDockWidget(col2_stack[0], col3_stack[0], QtCore.Qt.Orientation.Horizontal)
               except Exception:
                  pass
               for d in col3_stack[1:]:
                  try:
                     self.splitDockWidget(col3_stack[0], d, QtCore.Qt.Orientation.Vertical)
                     col3_stack[0] = d
                  except Exception:
                     pass

            # Size heuristics for 3-column layout
            try:
               if plot is not None:
                  self.resizeDocks([plot], [260], QtCore.Qt.Orientation.Vertical)
            except Exception:
               pass

            try:
               anchors = []
               for k in ("multiaxis", "demo"):
                  if k in visible_docks:
                     anchors.append(visible_docks[k])
                     break
               for k in ("camera", "detimg"):
                  if k in visible_docks:
                     anchors.append(visible_docks[k])
                     break
               for k in ("multiview", "multiviewctl"):
                  if k in visible_docks:
                     anchors.append(visible_docks[k])
                     break
               
               if len(anchors) == 3:
                  self.resizeDocks(anchors, [280, 400, 400], QtCore.Qt.Orientation.Horizontal)
            except Exception:
               pass

         else:  # num_columns == 4
            # 4-column layout: primary controls | secondary controls | primary images | secondary images
            col1_stack = [
               visible_docks.get("multiaxis"),
               visible_docks.get("demo"),
            ]
            col1_stack = [d for d in col1_stack if d is not None]
            
            col2_stack = [
               visible_docks.get("multiviewctl"),
               visible_docks.get("camctl"),
               visible_docks.get("detctl"),
            ]
            col2_stack = [d for d in col2_stack if d is not None]
            
            col3_stack = [
               visible_docks.get("camera"),
               visible_docks.get("detimg"),
            ]
            col3_stack = [d for d in col3_stack if d is not None]
            
            col4_stack = [
               visible_docks.get("multiview"),
            ]
            col4_stack = [d for d in col4_stack if d is not None]

            # Create column 1 (leftmost)
            if col1_stack:
               try:
                  self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, col1_stack[0])
               except Exception:
                  pass
               for d in col1_stack[1:]:
                  try:
                     self.splitDockWidget(col1_stack[0], d, QtCore.Qt.Orientation.Vertical)
                     col1_stack[0] = d
                  except Exception:
                     pass

            # Create column 2
            if col2_stack and col1_stack:
               try:
                  self.splitDockWidget(col1_stack[0], col2_stack[0], QtCore.Qt.Orientation.Horizontal)
               except Exception:
                  pass
               for d in col2_stack[1:]:
                  try:
                     self.splitDockWidget(col2_stack[0], d, QtCore.Qt.Orientation.Vertical)
                     col2_stack[0] = d
                  except Exception:
                     pass

            # Create column 3
            if col3_stack and col2_stack:
               try:
                  self.splitDockWidget(col2_stack[0], col3_stack[0], QtCore.Qt.Orientation.Horizontal)
               except Exception:
                  pass
               for d in col3_stack[1:]:
                  try:
                     self.splitDockWidget(col3_stack[0], d, QtCore.Qt.Orientation.Vertical)
                     col3_stack[0] = d
                  except Exception:
                     pass

            # Create column 4 (rightmost)
            if col4_stack and col3_stack:
               try:
                  self.splitDockWidget(col3_stack[0], col4_stack[0], QtCore.Qt.Orientation.Horizontal)
               except Exception:
                  pass
               for d in col4_stack[1:]:
                  try:
                     self.splitDockWidget(col4_stack[0], d, QtCore.Qt.Orientation.Vertical)
                     col4_stack[0] = d
                  except Exception:
                     pass

            # Size heuristics for 4-column layout
            try:
               if plot is not None:
                  self.resizeDocks([plot], [260], QtCore.Qt.Orientation.Vertical)
            except Exception:
               pass

            try:
               anchors = []
               for k in ("multiaxis", "demo"):
                  if k in visible_docks:
                     anchors.append(visible_docks[k])
                     break
               for k in ("multiviewctl", "camctl"):
                  if k in visible_docks:
                     anchors.append(visible_docks[k])
                     break
               for k in ("camera", "detimg"):
                  if k in visible_docks:
                     anchors.append(visible_docks[k])
                     break
               for k in ("multiview",):
                  if k in visible_docks:
                     anchors.append(visible_docks[k])
                     break
               
               if len(anchors) == 4:
                  self.resizeDocks(anchors, [250, 250, 350, 350], QtCore.Qt.Orientation.Horizontal)
            except Exception:
               pass

         # Vertical sizing for control panels (common to all layouts)
         try:
            left_vertical_docks = []
            left_vertical_sizes = []
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

   def _add_new_docks_to_view_menu(self):
      """Add the new control docks to the View menu after they are created."""
      try:
         menubar = self.menuBar()
         view_menu = None
         
         # Find the View menu
         for action in menubar.actions():
            if action.text() == "&View":
               view_menu = action.menu()
               break
         
         if view_menu is None:
            logger.warning("View menu not found")
            return
         
         # Add separator and new dock items
         view_menu.addSeparator()
         
         # Stage Control
         if hasattr(self, 'stage_control_dock') and self.stage_control_dock is not None:
            try:
               stage_control_act = QAction("Stage", self, checkable=True)
               stage_control_act.setChecked(False)
               stage_control_act.triggered.connect(lambda checked: self._toggle_stage_control_dock(checked))
               view_menu.addAction(stage_control_act)
               if not hasattr(self, '_view_dock_actions'):
                  self._view_dock_actions = {}
               self._view_dock_actions["stage_control"] = stage_control_act
               logger.info("Added Stage to View menu")
            except Exception as e:
               logger.error("Failed to add Stage to View menu: %s", e)
         else:
            logger.warning("Stage Control dock not available for View menu")
         
         # Excitation Control
         if hasattr(self, 'excitation_control_dock') and self.excitation_control_dock is not None:
            try:
               excitation_control_act = QAction("Excitation Control", self, checkable=True)
               excitation_control_act.setChecked(False)
               excitation_control_act.triggered.connect(lambda checked: self._toggle_excitation_control_dock(checked))
               view_menu.addAction(excitation_control_act)
               if not hasattr(self, '_view_dock_actions'):
                  self._view_dock_actions = {}
               self._view_dock_actions["excitation_control"] = excitation_control_act
               logger.info("Added Excitation Control to View menu")
            except Exception as e:
               logger.error("Failed to add Excitation Control to View menu: %s", e)
         else:
            logger.warning("Excitation Control dock not available for View menu")
         
         # Stage Calibration
         if hasattr(self, 'stage_calibration_dock') and self.stage_calibration_dock is not None:
            try:
               stage_calibration_act = QAction("Stage Calibration", self, checkable=True)
               stage_calibration_act.setChecked(False)
               stage_calibration_act.triggered.connect(lambda checked: self._toggle_stage_calibration_dock(checked))
               view_menu.addAction(stage_calibration_act)
               if not hasattr(self, '_view_dock_actions'):
                  self._view_dock_actions = {}
               self._view_dock_actions["stage_calibration"] = stage_calibration_act
               logger.info("Added Stage Calibration to View menu")
            except Exception as e:
               logger.error("Failed to add Stage Calibration to View menu: %s", e)
         else:
            logger.warning("Stage Calibration dock not available for View menu")
         
         view_menu.addSeparator()
         
      except Exception as e:
         logger.exception("Error adding new docks to View menu: %s", e)

   def _toggle_stage_control_dock(self, checked: bool):
      """Toggle Stage Control dock visibility and ensure devices are ready."""
      if checked:
         # Ensure devices are built before showing
         if not self.devices_built or self.devices_released:
            try:
               logger.info("Building devices for stage control panel")
               self.cam, self.stage, self.focus, self.light, self.fw, self.det, self.excitation = build_devices(self._config_path)
               self.devices_built = True
               self.devices_released = False
            except Exception as exc:
               logger.error("Failed to build devices: %s", exc)
               QtWidgets.QMessageBox.critical(self, "Device Error", f"Could not build devices:\n{exc}")
               return False
         
         # Update the tab with devices
         if hasattr(self, 'stage_control_tab') and self.stage_control_tab:
            logger.info(f"Setting stage on stage_control_tab: {self.stage}")
            logger.info(f"Stage type: {type(self.stage)}")
            logger.info(f"Stage has move_to: {hasattr(self.stage, 'move_to')}")
            self.stage_control_tab.set_stage(self.stage)
            self.stage_control_tab.set_focus(self.focus)
            self.stage_control_tab.set_config_path(self._config_path)
         
         self.stage_control_dock.setVisible(True)
         self.stage_control_dock.raise_()
      else:
         self.stage_control_dock.setVisible(False)
      return True

   def _cleanup_stage_control(self, event):
      """Clean up stage control resources when dock is closed."""
      try:
         if hasattr(self, 'stage_control_tab') and self.stage_control_tab:
            self.stage_control_tab.cleanup()
         event.accept()
      except Exception:
         event.accept()

   def _toggle_excitation_control_dock(self, checked: bool):
      """Toggle Excitation Control dock visibility and ensure devices are ready."""
      if checked:
         # Ensure devices are built before showing
         if not self.devices_built or self.devices_released:
            try:
               logger.info("Building devices for excitation control panel")
               self.cam, self.stage, self.focus, self.light, self.fw, self.det, self.excitation = build_devices(self._config_path)
               self.devices_built = True
               self.devices_released = False
            except Exception as exc:
               logger.error("Failed to build devices: %s", exc)
               QtWidgets.QMessageBox.critical(self, "Device Error", f"Could not build devices:\n{exc}")
               return False
         
         if self.excitation is None:
            logger.warning("No excitation devices available for excitation control panel")
            QtWidgets.QMessageBox.warning(self, "No Excitation Devices", "No excitation device available.")
            return False
         
         # Update the tab with devices
         if hasattr(self, 'excitation_control_tab') and self.excitation_control_tab:
            self.excitation_control_tab.set_excitation_devices(self.excitation)
         
         self.excitation_control_dock.setVisible(True)
         self.excitation_control_dock.raise_()
      else:
         self.excitation_control_dock.setVisible(False)
      return True

   def _toggle_stage_calibration_dock(self, checked: bool):
      """Toggle Stage Calibration dock visibility and ensure devices are ready."""
      if checked:
         # Ensure devices are built before showing
         if not self.devices_built or self.devices_released:
            try:
               logger.info("Building devices for stage calibration panel")
               self.cam, self.stage, self.focus, self.light, self.fw, self.det, self.excitation = build_devices(self._config_path)
               self.devices_built = True
               self.devices_released = False
            except Exception as exc:
               logger.error("Failed to build devices: %s", exc)
               QtWidgets.QMessageBox.critical(self, "Device Error", f"Could not build devices:\n{exc}")
               return False
         
         if self.stage is None:
            logger.warning("No stage available for stage calibration panel")
            QtWidgets.QMessageBox.warning(self, "No Stage", "No stage device available.")
            return False
         
         # Update the tab with devices
         if hasattr(self, 'stage_calibration_tab') and self.stage_calibration_tab:
            self.stage_calibration_tab.set_stage(self.stage)
            self.stage_calibration_tab.set_config_path(self._config_path)
         
         self.stage_calibration_dock.setVisible(True)
         self.stage_calibration_dock.raise_()
      else:
         self.stage_calibration_dock.setVisible(False)
      return True

   def _open_move_motors_dialog(self):
      """Toggle Stage Control dock visibility."""
      if hasattr(self, 'stage_control_dock') and self.stage_control_dock is not None:
         # Toggle the dock visibility
         current_visible = self.stage_control_dock.isVisible()
         self._toggle_stage_control_dock(not current_visible)
         # Update menu item state
         if hasattr(self, '_view_dock_actions') and 'stage_control' in self._view_dock_actions:
            act = self._view_dock_actions['stage_control']
            act.blockSignals(True)
            act.setChecked(not current_visible)
            act.blockSignals(False)

   def _open_excitation_control_dialog(self):
      """Toggle Excitation Control dock visibility."""
      if hasattr(self, 'excitation_control_dock') and self.excitation_control_dock is not None:
         # Toggle the dock visibility
         current_visible = self.excitation_control_dock.isVisible()
         self._toggle_excitation_control_dock(not current_visible)
         # Update menu item state
         if hasattr(self, '_view_dock_actions') and 'excitation_control' in self._view_dock_actions:
            act = self._view_dock_actions['excitation_control']
            act.blockSignals(True)
            act.setChecked(not current_visible)
            act.blockSignals(False)

   def _on_calibration_saved(self, x_scale: float, y_scale: float):
      """Called when the calibration wizard has written new scale values."""
      try:
         logger.info("Stage calibration saved: x_scale=%s y_scale=%s", x_scale, y_scale)
      except Exception:
         pass
      try:
         self.statusBar().showMessage(
            f"Stage calibrated: X={x_scale:.4f} steps/mm  Y={y_scale:.4f} steps/mm  "
            "(takes effect on next run)",
            10_000,
         )
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

      # Refuse to start if a running multi-axis scan is already using any of
      # the detectors the strip chart would poll.
      conflict = self._hardware_conflict(self._strip_chart_hardware(), self._multi_reserved)
      if conflict:
         QtWidgets.QMessageBox.warning(
            self,
            "Strip Chart",
            "Cannot start the Strip Chart while the Multi‑Axis scan is using "
            f"the same hardware ({conflict}).\n\n"
            "Stop the Multi‑Axis run, or de-select those detectors from the "
            "Multi‑Axis scan, then try again.",
         )
         return

      try:
         logger.info("Starting strip-chart experiment (config=%s)", cfg)
      except Exception:
         pass

      # Reuse existing devices if they are already built and still connected.
      if not self.devices_built or self.devices_released:
         cam, stage, focus, light, fw, det, excitation = build_devices(self._config_path)
         self.cam, self.stage, self.focus, self.light, self.fw, self.det, self.excitation = cam, stage, focus, light, fw, det, excitation
         self.devices_built = True
         self.devices_released = False
         # Ensure ComPort detectors are in their intended stream mode before acquisition.
         self._set_comport_mode_for_all(det)
         self._connect_detector_errors(det)
      else:
         cam, stage, focus, light, fw, det, excitation = self.cam, self.stage, self.focus, self.light, self.fw, self.det, self.excitation
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

      # Notify plugins that experiment is starting
      if PLUGINS_AVAILABLE and self._plugin_manager:
         try:
            # Get stage range from stage control tab if available
            stage_range = {}
            try:
               if hasattr(self, 'stage_control_tab') and self.stage_control_tab:
                  stage_config = getattr(self.stage_control_tab, 'stage_config', {})
                  stage_range = {
                     'stage_x_min': stage_config.get('x_min'),
                     'stage_x_max': stage_config.get('x_max'),
                     'stage_y_min': stage_config.get('y_min'),
                     'stage_y_max': stage_config.get('y_max'),
                  }
            except Exception as e:
               logger.warning(f"Failed to get stage range: {e}")
            
            for plugin_name, plugin in self._plugin_manager.get_all_plugins().items():
               if plugin.enabled:
                  # Update plugin config with stage range
                  if stage_range:
                     plugin.config.update(stage_range)
                  plugin.on_experiment_start(cfg)
         except Exception as e:
            logger.warning(f"Failed to notify plugins of experiment start: {e}")

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
         out_dir = self._resolve_output_dir(cfg.get("output_dir"))
         # Generate a unique measurement ID for this run
         self._measurement_id = str(uuid.uuid4())
         layout_json = self.live_tab.get_display_layout_json()
         use_multichannel = bool(cfg.get("multichannel", False) and MultiChannelSaver is not None)
         selected = self.multi_tab.get_selected_detectors() if hasattr(self.multi_tab, "get_selected_detectors") else []
         det_list = []
         if selected:
            det_list = selected
         else:
            if isinstance(det, list):
               det_list = [getattr(d, "name", getattr(d, "port", "detector")) for d in det]
            else:
               det_list = [getattr(det, "name", getattr(det, "port", "detector"))]

         if use_multichannel and _SAVING_ENABLED:
            # One shared file for all detectors
            self._close_mc_saver()
            self._mc_saver = MultiChannelSaver(
               out_dir,
               measurement_id=self._measurement_id,
               layout_json=layout_json,
               software_version=APP_VERSION,
               experiment_type="strip_chart",
            )
            try:
               self.statusBar().showMessage(
                  f"Saving multi-channel → {self._mc_saver.h5_path.name}", 8000)
            except Exception:
               pass
         else:
            # create savers for chosen ids
            for det_id in det_list:
               if _SAVING_ENABLED:
                  self.stream_savers[det_id] = StreamSaver(
                     out_dir, det_id,
                     measurement_id=self._measurement_id,
                     layout_json=layout_json,
                     software_version=APP_VERSION,
                     experiment_type="strip_chart",
                  )
            try:
               if _SAVING_ENABLED:
                  self.statusBar().showMessage(f"Saving detector streams to: {out_dir}", 8000)
               else:
                  self.statusBar().showMessage("Saving disabled (debug mode)", 4000)
            except Exception:
               pass

         try:
            logger.info(
               "Stream saving %s (out_dir=%s detectors=%s multichannel=%s)",
               "enabled" if _SAVING_ENABLED else "disabled",
               out_dir,
               det_list,
               use_multichannel,
            )
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
                           sample = d.read_value() if hasattr(d, "read_value") else None
                           if sample is None:
                              continue
                           temp = None
                           if isinstance(sample, tuple):
                              val = float(sample[0])
                              try:
                                 temp = float(sample[1])
                              except Exception:
                                 temp = None
                           else:
                              val = float(sample)
                           det_id = getattr(d, "name", getattr(d, "port", "detector"))
                           meta = {
                              "experiment": "strip_chart",
                              "timestamp": time.time(),
                              "output_dir": str(out_dir),
                           }
                           try:
                              mode = str(getattr(d, "mode", "")).strip().lower()
                              meta["measurement_kind"] = "resistance" if mode == "res" else "voltage"
                           except Exception:
                              pass
                           if temp is not None:
                              meta["temperature"] = temp
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
                  self.orch.shutdown(disconnect_devices=False)
               except Exception:
                  pass
               # When the measurement finishes, stop stream saving. The strip
               # chart owns the stream savers whenever it runs, so always close
               # them here. The image saver, however, may belong to a multi-axis
               # scan running concurrently; only close it when no multi-axis run
               # is active.
               self._close_all_stream_savers()
               if self.multi_thread is None:
                  self._close_image_saver()
               self.orch = None
               self.orch_thread = None
               self._set_measurement_state("Finished", kind="Strip Chart")

      self.orch_thread = threading.Thread(target=worker, daemon=True)
      self.orch_thread.start()
      self._strip_reserved = self._strip_chart_hardware()
      self._set_measurement_state("Running", kind="Strip Chart")
      self._refresh_run_button_states()

   def _on_stream_toggled(self, det_id: str, enabled: bool):
      """Create or close stream saver when user toggles streaming from the LiveTab."""
      try:
         try:
            logger.info("Stream toggle (detector=%s enabled=%s)", det_id, enabled)
         except Exception:
            pass
         if enabled:
            if _SAVING_ENABLED and det_id not in self.stream_savers:
               out_dir = self._resolve_output_dir(self.demo_tab.output_dir_edit.text())
               self.stream_savers[det_id] = StreamSaver(
                  out_dir, det_id,
                  measurement_id=self._measurement_id,
                  layout_json=self.live_tab.get_display_layout_json(),
                  software_version=APP_VERSION,
                  experiment_type="strip_chart",
               )
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
               # exited abnormally. Leave a concurrently running multi-axis
               # scan's image saver untouched.
               self._close_all_stream_savers()
               if self.multi_thread is None:
                  self._close_image_saver()
               try:
                  self.statusBar().showMessage("Experiment finished. Stream saved closed.", 5000)
               except Exception:
                  pass
               self._set_measurement_state("Finished", kind="Strip Chart")
               
               # Notify plugins that experiment has ended
               if PLUGINS_AVAILABLE and self._plugin_manager:
                  try:
                     for plugin_name, plugin in self._plugin_manager.get_all_plugins().items():
                        if plugin.enabled:
                           plugin.on_experiment_end({})
                  except Exception as e:
                     logger.warning(f"Failed to notify plugins of experiment end: {e}")
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

      # Stop stage position timer when multiaxis starts - multiaxis will update position itself
      try:
         logger.info("Stopping stage position timer - multiaxis will handle position updates")
      except Exception:
         pass
      self._stop_stage_position_timer()

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

      # Refuse to start if a running strip chart is already using any of the
      # detectors (or motors) this multi-axis scan needs.
      conflict = self._hardware_conflict(self._multiaxis_hardware(), self._strip_reserved)
      if conflict:
         QtWidgets.QMessageBox.warning(
            self,
            "Multi‑Axis",
            "Cannot start the Multi‑Axis scan while the Strip Chart is using "
            f"the same hardware ({conflict}).\n\n"
            "Stop the Strip Chart, or de-select those detectors from this scan, "
            "then try again.",
         )
         return

      try:
         axis_types = [getattr(c, "axis_type", "?") for c in cfgs]
         logger.info("Multi-axis axes: n=%s types=%s", len(cfgs), axis_types)
      except Exception:
         pass

      # Build devices only if not already built or if previously released
      if not self.devices_built or self.devices_released:
         self.cam, self.stage, self.focus, self.light, self.fw, self.det, self.excitation = build_devices(self._config_path)
         # Ensure ComPort detectors are in their intended stream mode before acquisition.
         self._set_comport_mode_for_all(self.det)
         self._connect_detector_errors(self.det)
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
      cam, stage, focus, light, fw, det, excitation = self.cam, self.stage, self.focus, self.light, self.fw, self.det, self.excitation
      # Build a device map for pre/post moves and axis dialogs
      device_map = {
         "stage": stage,
         "focus": focus,
         "camera": cam,
         "light": light,
         "fw": fw,
         "excitation": excitation,
      }
      # Add individual excitation devices to the map if there are multiple
      if isinstance(excitation, list):
         for exc in excitation:
            if hasattr(exc, 'name'):
               device_map[exc.name] = exc
      if isinstance(det, list):
         for d in det:
            device_map[getattr(d, 'name', getattr(d, 'port', 'detector'))] = d
      else:
         device_map[getattr(det, 'name', getattr(det, 'port', 'detector'))] = det

      # Only stream/measure detector signal when a Detector axis is defined.
      # Without a Detector axis the run is a pure motor/camera scan.
      has_detector_axis = any(getattr(c, "axis_type", None) == "Detector" for c in cfgs)

      # Determine which detectors the Detector axes actually target. When every
      # Detector axis names a specific detector (e.g. "vm2"), the run should use
      # only those detectors. A Detector axis without a specific name means
      # "all detectors" (legacy behaviour).
      axis_detector_names: set[str] | None = set()
      for c in cfgs:
         if getattr(c, "axis_type", None) == "Detector":
            nm = (getattr(c, "params", None) or {}).get("detector")
            if nm:
               axis_detector_names.add(str(nm))
            else:
               # Generic Detector axis → use all detectors.
               axis_detector_names = None
               break
      if axis_detector_names is not None and not axis_detector_names:
         # No specific names collected (and no generic axis) → fall back to all.
         axis_detector_names = None

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
               motor_devices, motor_modes = _resolve_motors(device_map, p)
               axes.append(
                  XAxis(
                     stage,
                     p["start"],
                     p["end"],
                     p["step"],
                     motor_devices=motor_devices or None,
                     motor_mode=p.get("motor_mode", "sequential"),
                     motor_modes=motor_modes or None,
                     wait_s=p.get("wait", 0.0),
                     sync_timeout=p.get("sync_timeout", 5.0),
                     sync_poll=p.get("sync_poll", 0.01),
                     sync_tol=p.get("sync_tol", 1e-3),
                  )
               )
            elif t == "Y":
               motor_devices, motor_modes = _resolve_motors(device_map, p)
               axes.append(
                  YAxis(
                     stage,
                     p["start"],
                     p["end"],
                     p["step"],
                     motor_devices=motor_devices or None,
                     motor_mode=p.get("motor_mode", "sequential"),
                     motor_modes=motor_modes or None,
                     wait_s=p.get("wait", 0.0),
                     sync_timeout=p.get("sync_timeout", 5.0),
                     sync_poll=p.get("sync_poll", 0.01),
                     sync_tol=p.get("sync_tol", 1e-3),
                  )
               )
            elif t == "Z":
               motor_devices, motor_modes = _resolve_motors(device_map, p)
               axes.append(
                  ZAxis(
                     focus,
                     p["start"],
                     p["end"],
                     p["step"],
                     motor_devices=motor_devices or None,
                     motor_mode=p.get("motor_mode", "sequential"),
                     motor_modes=motor_modes or None,
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
               # When a specific detector is named (e.g. "vm2"), use only that one;
               # otherwise fall back to all configured detector(s).
               det_name = p.get("detector")
               det_target = device_map.get(det_name) if det_name else det
               if det_target is None:
                  det_target = det
               axes.append(DetectorAxis(det_target, scales=None, wait_s=p.get("wait", 0.0)))
            elif t == "Excitation":
               # Get the specific excitation device by name if specified
               exc_name = p.get("excitation")
               if exc_name:
                  excitation_device = device_map.get(exc_name)
                  if excitation_device is None:
                     # Try to find it in the list
                     if isinstance(excitation, list):
                        for exc in excitation:
                           if hasattr(exc, 'name') and exc.name == exc_name:
                              excitation_device = exc
                              break
                     else:
                        excitation_device = excitation
               else:
                  # Use the first available excitation device
                  if isinstance(excitation, list):
                     excitation_device = excitation[0] if excitation else None
                  else:
                     excitation_device = excitation
               
               if excitation_device is None:
                  logger.warning("No excitation device available for Excitation axis")
                  continue
               
               axes.append(ExcitationAxis(excitation_device, p.get("states", [True, False]), p.get("wait", 0.0)))
            elif t == "Round":
               axes.append(RoundAxis(p["n_rounds"]))

      # Merge consecutive axes the user grouped together into composite
      # GroupedAxis dimensions (sync/sequential scan with shorter/longer steps).
      try:
         axes = self._apply_axis_grouping(axes, cfgs)
      except Exception:
         logger.exception("Failed to apply axis grouping; running ungrouped")

      # Log the final scan dimensions in execution order (axis 0 = outermost
      # loop, last axis = innermost loop) so the run structure is traceable.
      try:
         logger.info("Multi-axis run: %d scan dimension(s) (outer→inner):", len(axes))
         for i, ax in enumerate(axes):
            scope = "outermost" if i == 0 else ("innermost" if i == len(axes) - 1 else "inner")
            logger.info("  dim %d [%s]: %s = %s", i, scope, type(ax).__name__, ax.name())
      except Exception:
         logger.exception("Failed to log multi-axis scan dimensions")

      self.live_tab.reset_multiaxis()

      # Apply the Multi-Axis tab's default x-axis preference to the Live plot.
      # The Live plot will apply this as soon as the first multi-axis samples
      # arrive and x-axis options are refreshed.
      #
      # Skip this while the Strip Chart is running: the two modes share the Live
      # plot, and changing the x-axis to a scan axis (X/Z/…) would disrupt the
      # running strip chart's time-based x label.
      try:
         strip_running = getattr(self, "orch_thread", None) is not None
         # Tell the Live plot whether the Strip Chart owns it; while it does, the
         # multi-axis scan must not switch plot mode or change the x-axis label.
         try:
            if hasattr(self.live_tab, "set_strip_owns_plot"):
               self.live_tab.set_strip_owns_plot(strip_running)
         except Exception:
            pass
         if strip_running:
            logger.info(
               "Strip Chart is running; not overriding the Live plot x-axis "
               "for the multi-axis scan."
            )
         elif hasattr(self.multi_tab, "get_default_xaxis") and hasattr(self.live_tab, "set_preferred_plot_xaxis"):
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
      # Whether this run records detector data (and therefore owns the stream /
      # multichannel savers). A detector-less scan owns nothing, so it must not
      # tear down a concurrently running strip chart's savers.
      self._multi_owns_stream_savers = False
      try:
         out_dir = self._resolve_output_dir(self.demo_tab.output_dir_edit.text())
         # Generate a new universal measurement ID for this run
         self._measurement_id = str(uuid.uuid4())
         use_multichannel = bool(getattr(self.demo_tab, 'multichannel_cb', None) and
                                 self.demo_tab.multichannel_cb.isChecked() and
                                 MultiChannelSaver is not None)
         selected = self.multi_tab.get_selected_detectors() if hasattr(self.multi_tab, "get_selected_detectors") else []
         det_list = []
         if selected:
            det_list = selected
         else:
            if isinstance(det, list):
               det_list = [getattr(d, "name", getattr(d, "port", "detector")) for d in det]
            else:
               det_list = [getattr(det, "name", getattr(det, "port", "detector"))]

         # Restrict to detectors named by the Detector axes (e.g. only "vm2"
         # when a single vm2 detector axis is defined).
         if axis_detector_names is not None:
            det_list = [d for d in det_list if str(d) in axis_detector_names]

         if not has_detector_axis:
            # No Detector axis defined: this is a pure motor/camera scan, so do
            # not register detectors or create (empty) stream savers.
            det_list = []
            self.statusBar().showMessage(
               "No Detector axis defined → detector signal not streamed", 6000)
         elif use_multichannel and _SAVING_ENABLED:
            # One shared file for all detectors
            self._close_mc_saver()
            self._mc_saver = MultiChannelSaver(
               out_dir,
               measurement_id=self._measurement_id,
               layout_json=self.live_tab.get_display_layout_json(),
               software_version=APP_VERSION,
               experiment_type="multiaxis",
            )
            for det_id in det_list:
               try:
                  self.live_tab.register_detector(det_id)
               except Exception:
                  pass
            try:
               self.statusBar().showMessage(
                  f"Saving multi-channel → {self._mc_saver.h5_path.name}", 8000)
            except Exception:
               pass
         else:
            # create savers for chosen ids (do not overwrite existing savers)
            for det_id in det_list:
               # ensure live tab knows about this detector (create image view and controls)
               try:
                  self.live_tab.register_detector(det_id)
               except Exception:
                  pass
               if _SAVING_ENABLED and det_id not in self.stream_savers:
                  self.stream_savers[det_id] = StreamSaver(
                     out_dir, det_id,
                     measurement_id=self._measurement_id,
                     layout_json=self.live_tab.get_display_layout_json(),
                     software_version=APP_VERSION,
                     experiment_type="multiaxis",
                  )

         # brief status so users know where data is written
         try:
            if _SAVING_ENABLED and not use_multichannel:
               self.statusBar().showMessage(f"Saving detector streams to: {out_dir}", 8000)
            else:
               self.statusBar().showMessage("Saving disabled (debug mode)", 4000)
         except Exception:
            pass
         # Record whether this run owns savers, so a concurrent strip chart's
         # savers are never torn down by this run.
         self._multi_owns_stream_savers = bool(det_list)

         # When the Detector axes name specific detectors, restrict the display
         # (plot curves, detector image panel) to exactly those active detectors
         # so the UI reflects the number of detectors actually in use.
         if has_detector_axis and axis_detector_names is not None:
            try:
               self._selected_detectors_for_display = set(det_list) if det_list else None
            except Exception:
               self._selected_detectors_for_display = None
            try:
               if hasattr(self.live_tab, "set_selected_detectors"):
                  self.live_tab.set_selected_detectors(list(det_list))
            except Exception:
               pass
         # apply default X-axis selection from MultiAxisTab via the preference
         # mechanism (handled by set_preferred_plot_xaxis called earlier above).
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
            if has_detector_axis and det is not None:
               dets = det if isinstance(det, list) else [det]
               # Only read the detectors named by the Detector axes (e.g. just
               # "vm2" when a single vm2 detector axis is defined).
               if axis_detector_names is not None:
                  dets = [
                     d for d in dets
                     if str(getattr(d, "name", getattr(d, "port", "detector"))) in axis_detector_names
                  ]
               for d in dets:
                  try:
                     sample = d.read_value()
                     temp = None
                     if isinstance(sample, tuple):
                        val = float(sample[0])
                        try:
                           temp = float(sample[1])
                        except Exception:
                           temp = None
                     else:
                        val = float(sample)
                     det_id = getattr(d, "name", getattr(d, "port", "detector"))
                     sample_meta = dict(state)
                     try:
                        mode = str(getattr(d, "mode", "")).strip().lower()
                        sample_meta["measurement_kind"] = "resistance" if mode == "res" else "voltage"
                     except Exception:
                        pass
                     if temp is not None:
                        sample_meta["temperature"] = temp
                     else:
                        try:
                           if hasattr(d, "read_temperature"):
                              t = d.read_temperature()
                              if t is not None:
                                 sample_meta["temperature"] = float(t)
                        except Exception:
                           pass

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
                        self.live_tab.queue_multiaxis_sample(str(det_id), sample_meta, float(val))
                     except Exception:
                        pass
                     # stream-save if enabled per detector id
                     try:
                        mc = getattr(self, '_mc_saver', None)
                        if mc is not None:
                           mc.append_sample(str(det_id), time.time(), float(val), meta=sample_meta)
                        else:
                           saver = self.stream_savers.get(det_id)
                           if saver:
                              saver.append_sample(time.time(), float(val), meta=sample_meta)
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
               # When the measurement finishes, stop stream saving — but only
               # if this run actually owns the savers. A detector-less scan that
               # ran alongside the strip chart must leave the strip chart's
               # savers intact.
               if self._multi_owns_stream_savers:
                  self._close_all_stream_savers()
               self._multi_owns_stream_savers = False
               # Release the strip-chart-owns-plot guard set at start (safe to
               # toggle this plain bool from the worker thread).
               try:
                  if hasattr(self.live_tab, "set_strip_owns_plot"):
                     self.live_tab.set_strip_owns_plot(False)
               except Exception:
                  pass
               self.multi_runner = None
               self.multi_thread = None
               self._set_measurement_state("Finished", kind="Multi-Axis")
               # Timer will be restarted by _apply_measurement_state

      self.multi_thread = threading.Thread(target=worker, daemon=True)
      self.multi_thread.start()
      self._multi_reserved = self._multiaxis_hardware()
      self._set_measurement_state("Running", kind="Multi-Axis")
      self._refresh_run_button_states()

   def _stop_multiaxis(self):
      try:
         logger.info("Stopping multi-axis run")
      except Exception:
         pass

      if self.multi_runner is not None:
            self.multi_runner.stop()

      # NOTE: Timer restart is handled by _apply_measurement_state when state changes to "Finished"
      # This is the correct place because it ensures timer is restarted whether multiaxis
      # stops normally (via _set_measurement_state) or is stopped manually (via this method)

      # NOTE: do not automatically disconnect devices when the user presses
      # Stop. Disconnecting here caused the stage (and other devices) to become
      # inactive and prevented subsequent runs from rebuilding cleanly. Keep
      # devices connected so the user can immediately restart a run. If a
      # full release of devices is desired, use the application's shutdown
      # path or an explicit "Release devices" action.

      try:
         logger.info("Multi-axis stopped (devices_released=%s)", getattr(self, "devices_released", None))
      except Exception:
         pass
      # close any stream savers this run owns. If it ran detector-less alongside
      # the strip chart, leave the strip chart's savers and image saver alone.
      if self._multi_owns_stream_savers:
         self._close_all_stream_savers()
         self._multi_owns_stream_savers = False
      if self.orch_thread is None:
         self._close_image_saver()

   # ----------------- multi-view camera scan -----------------

   def _start_multiview_scan(self) -> None:
      if self.multiview_thread is not None:
         return

      # Stop stage position timer when multiview scan starts - multiview will update position itself
      try:
         logger.info("Stopping stage position timer - multiview will handle position updates")
      except Exception:
         pass
      self._stop_stage_position_timer()

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
         self.cam, self.stage, self.focus, self.light, self.fw, self.det, self.excitation = build_devices(self._config_path)
         self._connect_detector_errors(self.det)
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

      cam, stage, focus, light, fw, excitation = self.cam, self.stage, self.focus, self.light, self.fw, self.excitation

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
            "excitation": excitation,
         }
         # Add individual excitation devices to the map if there are multiple
         if isinstance(excitation, list):
            for exc in excitation:
               if hasattr(exc, 'name'):
                  device_map[exc.name] = exc

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
         elif t == "Excitation":
            # Get the specific excitation device by name if specified
            exc_name = p.get("excitation")
            if exc_name:
               excitation_device = device_map.get(exc_name)
               if excitation_device is None:
                  # Try to find it in the list
                  if isinstance(excitation, list):
                     for exc in excitation:
                        if hasattr(exc, 'name') and exc.name == exc_name:
                           excitation_device = exc
                           break
                  else:
                     excitation_device = excitation
            else:
               # Use the first available excitation device
               if isinstance(excitation, list):
                  excitation_device = excitation[0] if excitation else None
               else:
                  excitation_device = excitation
            
            if excitation_device is None:
               logger.warning("No excitation device available for Excitation axis")
               continue
            
            axes.append(ExcitationAxis(excitation_device, p.get("states", [True, False]), p.get("wait", 0.0)))
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
            self._set_measurement_state("Finished", kind="Multi View")
            # Timer will be restarted by _apply_measurement_state

      self.multiview_thread = threading.Thread(target=worker, daemon=True)
      self.multiview_thread.start()
      self._set_measurement_state("Running", kind="Multi View")

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

      # NOTE: Timer restart is handled by _apply_measurement_state when state changes to "Finished"

      try:
         logger.info("Multiview stop requested")
      except Exception:
         pass

   def _on_live_view_changed(self, mode: str):
      try:
         if mode == 'camera':
            if hasattr(self, 'cam_dock'):
               self.cam_dock.show()
               # Ensure the content widget is visible inside the dock
               try:
                  self.live_tab.camera_panel.show()
               except Exception:
                  pass
            if hasattr(self, 'detimg_dock'):
               self.detimg_dock.hide()
            if hasattr(self, 'plot_dock'):
               self.plot_dock.hide()
         elif mode == 'detector':
            if hasattr(self, 'cam_dock'):
               self.cam_dock.hide()
            if hasattr(self, 'detimg_dock'):
               self.detimg_dock.show()
               # Ensure the content widget is visible inside the dock — calling
               # .hide() on a dock's content widget collapses it permanently
               # until it is explicitly re-shown here.
               try:
                  self.live_tab.detector_image_panel.show()
               except Exception:
                  pass
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

   def _apply_axis_grouping(self, axes, cfgs):
      """Apply per-axis grouping / collapse flags to the built axis list.

      ``axes[i]`` corresponds 1:1 to ``cfgs[i]``.
      - An axis flagged ``collapse_one_step`` is wrapped in a ``OneStepAxis`` so
        its whole sweep counts as a single scan step.
      - An axis flagged ``group_with_prev`` joins the group started by the axis
        immediately above it; the group's scan mode / step policy comes from the
        most recent joining axis. Multi-member groups become a ``GroupedAxis``.
      """
      from core.multiaxis import GroupedAxis, OneStepAxis

      if not axes or len(axes) != len(cfgs):
         return axes

      # First, collapse any axis flagged "one step" into a single scan step.
      prepped = []
      for ax, cfg in zip(axes, cfgs):
         params = getattr(cfg, "params", None) or {}
         if params.get("collapse_one_step"):
            prepped.append(OneStepAxis(ax))
         else:
            prepped.append(ax)

      groups: list[list] = []
      group_meta: list[tuple[str, str]] = []
      for ax, cfg in zip(prepped, cfgs):
         params = getattr(cfg, "params", None) or {}
         join = bool(params.get("group_with_prev")) and bool(groups)
         if join:
            groups[-1].append(ax)
            group_meta[-1] = (
               params.get("group_mode", "sync"),
               params.get("group_length", "longer"),
            )
         else:
            groups.append([ax])
            group_meta.append((
               params.get("group_mode", "sync"),
               params.get("group_length", "longer"),
            ))

      result = []
      for members, (mode, length) in zip(groups, group_meta):
         if len(members) == 1:
            result.append(members[0])
         else:
            result.append(GroupedAxis(members, mode=mode, length=length))
      return result

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
         out_dir = self._resolve_output_dir(meta.get("output_dir"))
      else:
         try:
            out_dir = self._resolve_output_dir(self.demo_tab.output_dir_edit.text())
         except Exception:
            out_dir = self._project_data_dir()

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

         temp_value = None
         if isinstance(meta, dict):
            try:
               t = meta.get("temperature", None)
               if t is not None:
                  temp_value = float(t)
            except Exception:
               temp_value = None

         # If upstream did not pass temperature in metadata, read latest cached
         # detector temperature (ComPort mode 1) from the detector instance.
         if temp_value is None:
            try:
               dets = self.det if isinstance(self.det, list) else ([self.det] if self.det is not None else [])
               for d in dets:
                  did = getattr(d, "name", getattr(d, "port", "detector"))
                  if str(did) != str(det_id):
                     continue
                  if hasattr(d, "read_temperature"):
                     t = d.read_temperature()
                     if t is not None:
                        temp_value = float(t)
                  break
            except Exception:
               temp_value = None

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

         if isinstance(meta, dict) and temp_value is not None:
            try:
               meta = dict(meta)
               meta["temperature"] = float(temp_value)
            except Exception:
               pass

         if isinstance(meta, dict):
            try:
               dets = self.det if isinstance(self.det, list) else ([self.det] if self.det is not None else [])
               for d in dets:
                  did = getattr(d, "name", getattr(d, "port", "detector"))
                  if str(did) != str(det_id):
                     continue
                  mode = str(getattr(d, "mode", "")).strip().lower()
                  meta = dict(meta)
                  meta["measurement_kind"] = "resistance" if mode == "res" else "voltage"
                  break
            except Exception:
               pass

         # forward to live tab (queued) with metadata so temperature survives
         try:
            gui_meta = dict(meta) if isinstance(meta, dict) else {}
         except Exception:
            gui_meta = {}
         
         # Process with plugins before forwarding to live tab
         if PLUGINS_AVAILABLE and self._plugin_manager:
            try:
               from plugins.base_plugin import PluginData
               
               # Create plugin data container
               detector_data = {det_id: np.array([float(value)])}
               plugin_data = PluginData(
                  detector_data=detector_data,
                  positions=meta if isinstance(meta, dict) else {},
                  timestamps=np.array([timestamp]),
                  detector_ids=[det_id]
               )
               
               # Process with enabled plugins
               results = self._plugin_manager.process_data_with_plugins(plugin_data)
               
               # Check for movement commands from plugin results
               all_movement_commands = []
               for plugin_name, result in results.items():
                  if result.move_commands:
                     all_movement_commands.extend(result.move_commands)
                     logger.info(f"Plugin {plugin_name} generated {len(result.move_commands)} movement commands from result")
                     for cmd in result.move_commands:
                        logger.info(f"  Command: axis={cmd.get('axis')}, position={cmd.get('position')}, relative={cmd.get('relative')}")
               
               # Special handling: if a plugin is in decoding phase, process it again to ensure we capture movement commands
               # This is needed because plugins transition phases immediately after generating commands
               for plugin_name, plugin in self._plugin_manager._plugins.items():
                  if hasattr(plugin, '_current_phase') and plugin._current_phase == "decoding":
                     logger.info(f"Plugin {plugin_name} is in decoding phase, processing again to capture movement commands")
                     plugin_result = plugin.process_data(plugin_data)
                     if plugin_result.move_commands:
                        all_movement_commands.extend(plugin_result.move_commands)
                        logger.info(f"Plugin {plugin_name} generated {len(plugin_result.move_commands)} movement commands from second call")
                        for cmd in plugin_result.move_commands:
                           logger.info(f"  Command: axis={cmd.get('axis')}, position={cmd.get('position')}, relative={cmd.get('relative')}")
               
               # Also check get_movement_commands for backward compatibility
               movement_commands = self._plugin_manager.get_movement_commands(plugin_data)
               
               if movement_commands:
                  logger.info(f"Plugin generated {len(movement_commands)} movement commands from get_movement_commands")
                  all_movement_commands.extend(movement_commands)
               
               if all_movement_commands:
                  logger.info(f"Executing {len(all_movement_commands)} total movement commands")
                  try:
                     success = self._execute_plugin_movement_commands(all_movement_commands)
                     logger.info(f"Movement commands execution result: {success}")
                  except Exception as e:
                     logger.warning(f"Failed to execute plugin movement commands: {e}")
                     import traceback
                     traceback.print_exc()
               else:
                  logger.debug("No movement commands to execute")
            except Exception as e:
               logger.warning(f"Failed to process data with plugins: {e}")
         
         QtCore.QMetaObject.invokeMethod(
               self.live_tab,
               "add_detector_sample_qt_meta",
               QtCore.Qt.ConnectionType.QueuedConnection,
               QtCore.Q_ARG(str, det_id),
               QtCore.Q_ARG(float, float(value)),
               QtCore.Q_ARG(float, float(timestamp)),
               QtCore.Q_ARG(object, gui_meta),
         )
         # stream-save if enabled
         try:
            mc = getattr(self, '_mc_saver', None)
            if mc is not None:
               mc.append_sample(str(det_id), timestamp, float(value), meta=meta)
            else:
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

      # Detector selection affects which hardware the multi-axis scan claims,
      # so re-evaluate the start-button gating (e.g. while a strip chart runs).
      try:
         self._refresh_run_button_states()
      except Exception:
         pass

   def _on_detector_offset_toggled(self, detector_id: str, enabled: bool):
      """Apply per-detector display-offset toggle from MultiAxisTab UI."""
      try:
         applied = 0.0
         if hasattr(self, 'live_tab') and hasattr(self.live_tab, 'set_detector_display_offset_state'):
            applied = float(self.live_tab.set_detector_display_offset_state(detector_id, enabled=bool(enabled), value=None))
         if hasattr(self, 'multi_tab') and hasattr(self.multi_tab, 'set_detector_offset_state'):
            self.multi_tab.set_detector_offset_state(detector_id, enabled=bool(enabled), value=applied)
      except Exception:
         pass

   def _on_detector_offset_value_changed(self, detector_id: str, value: float):
      """Apply manual per-detector display-offset value from MultiAxisTab UI."""
      try:
         applied = float(value)
         if hasattr(self, 'live_tab') and hasattr(self.live_tab, 'set_detector_display_offset_state'):
            applied = float(self.live_tab.set_detector_display_offset_state(detector_id, enabled=None, value=float(value)))
         if hasattr(self, 'multi_tab') and hasattr(self.multi_tab, 'set_detector_offset_state'):
            self.multi_tab.set_detector_offset_state(detector_id, enabled=None, value=applied)
      except Exception:
         pass

   def save_full_experiment(self):
      try:
         default_path = self._project_experiments_dir() / "experiment.json"
      except Exception:
         default_path = Path("experiment.json")
      path, _ = QtWidgets.QFileDialog.getSaveFileName(
         self, "Save Experiment", str(default_path), "Experiment (*.json)"
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
         "output_dir": str(self._resolve_output_dir(self.demo_tab.output_dir_edit.text())),
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
      
      # Add plugin information if available
      if PLUGINS_AVAILABLE and self._plugin_manager:
         try:
            plugin_configs = self._plugin_manager._plugin_configs
            enabled_plugins = [name for name, plugin in self._plugin_manager.get_all_plugins().items() if plugin.enabled]
            
            data["plugins"] = {
               "configs": plugin_configs,
               "enabled": enabled_plugins
            }
         except Exception as e:
            logger.warning(f"Failed to save plugin information: {e}")

      with open(path, "w") as f:
         json.dump(data, f, indent=2)

   def _release_current_devices(self) -> None:
      """Stop any running acquisitions and disconnect currently-open devices."""
      # Stop running experiments / scans so devices are no longer in use.
      try:
         if self.orch_thread is not None:
            self._stop_experiment()
      except Exception:
         pass
      try:
         if self.multi_thread is not None:
            self._stop_multiaxis()
      except Exception:
         pass
      try:
         if self.multiview_thread is not None:
            self._stop_multiview_scan()
      except Exception:
         pass

      if not self.devices_built or self.devices_released:
         return

      detectors = self.det if isinstance(self.det, list) else [self.det]
      for dev in [self.cam, self.stage, self.focus, self.light, self.fw, self.excitation, *detectors]:
         if dev is None:
            continue
         try:
            dev.disconnect()
         except Exception:
            pass

      self.cam = None
      self.stage = None
      self.focus = None
      self.light = None
      self.fw = None
      self.det = None
      self.excitation = None
      self.devices_built = False
      self.devices_released = True

   def _build_devices_now(self) -> bool:
      """Open/initialize all hardware from the active config up front.

      Building devices eagerly (e.g. right after loading a hardware config)
      means a subsequent measurement can start immediately instead of paying
      the connection/initialization cost at run time. Safe to call when devices
      are already built (it is a no-op in that case).

      Returns True if devices are built and ready, False on failure.
      """
      if self.devices_built and not self.devices_released:
         return True
      try:
         self.cam, self.stage, self.focus, self.light, self.fw, self.det, self.excitation = build_devices(self._config_path)
         # Ensure ComPort detectors are in their intended stream mode and
         # surface connection errors to the user.
         self._set_comport_mode_for_all(self.det)
         self._connect_detector_errors(self.det)
         self.devices_built = True
         self.devices_released = False
         return True
      except Exception:
         logger.exception("Failed to build devices from config %s", self._config_path)
         self.devices_built = False
         self.devices_released = True
         return False

   def load_hardware_config(self):
      """Load a hardware/device config JSON and make it the active config."""
      try:
         default_dir = str(self._project_root_dir() / "config")
      except Exception:
         default_dir = ""

      path, _ = QtWidgets.QFileDialog.getOpenFileName(
         self, "Load Hardware Config", default_dir, "Hardware Config (*.json);;JSON (*.json)"
      )
      if not path:
         return

      try:
         cfg = load_config(path)
      except Exception as exc:
         QtWidgets.QMessageBox.warning(self, "Load Hardware Config", f"Could not read {path}:\n{exc}")
         return

      # Close/disconnect any currently-open hardware before switching configs.
      self._release_current_devices()

      self._config_path = path
      try:
         self._config_filename = Path(path).name  # Update config filename
      except Exception:
         self._config_filename = "config"  # Fallback if path processing fails
      
      # Keep child tabs in sync so new axis dialogs use the selected config.
      try:
         self.multi_tab._config_path = path
      except Exception:
         pass
      try:
         self.multiviewctl_tab._config_path = path
      except Exception:
         pass

      # Update window title with new config filename
      self._update_window_title()

      # Reset the plot legend / registered detectors so stale curves from the
      # previous hardware config do not linger before re-registering below.
      try:
         self.live_tab.clear_detectors()
      except Exception:
         pass

      # Refresh detector availability from the newly loaded hardware config.
      try:
         det_cfg = cfg.get("detector", []) if isinstance(cfg, dict) else []
         available: list[str] = []
         if isinstance(det_cfg, list):
            for i, dc in enumerate(det_cfg):
               if isinstance(dc, dict):
                  available.append(dc.get("name") or dc.get("port") or f"detector{i + 1}")
         elif isinstance(det_cfg, dict):
            available.append(det_cfg.get("name") or det_cfg.get("port") or "detector")

         if available:
            self.multi_tab.set_available_detectors(available)
            if hasattr(self.multi_tab, "set_selected_detectors"):
               self.multi_tab.set_selected_detectors(list(available))
            for det_id in available:
               self.live_tab.register_detector(det_id)
      except Exception:
         pass

      # Open/initialize all hardware now so a subsequent measurement starts
      # immediately instead of building devices at run time.
      try:
         if self._build_devices_now():
            self.statusBar().showMessage(f"Hardware config loaded and devices opened: {path}", 5000)
            
            # Reload Stage Control panel if it's visible
            if hasattr(self, 'stage_control_dock') and self.stage_control_dock.isVisible():
               try:
                  logger.info("Reloading Stage Control panel after hardware config change")
                  if hasattr(self, 'stage_control_tab') and self.stage_control_tab:
                     self.stage_control_tab.set_stage(self.stage)
                     self.stage_control_tab.set_focus(self.focus)
                     self.stage_control_tab.set_config_path(self._config_path)
               except Exception as e:
                  logger.exception("Failed to reload Stage Control panel: %s", e)
            
            # Reload Stage Calibration panel if it's visible
            if hasattr(self, 'stage_calibration_dock') and self.stage_calibration_dock.isVisible():
               try:
                  logger.info("Reloading Stage Calibration panel after hardware config change")
                  if hasattr(self, 'stage_calibration_tab') and self.stage_calibration_tab:
                     self.stage_calibration_tab.set_stage(self.stage)
                     self.stage_calibration_tab.set_config_path(self._config_path)
               except Exception as e:
                  logger.exception("Failed to reload Stage Calibration panel: %s", e)
            
            # Reload Excitation Control panel if it's visible
            if hasattr(self, 'excitation_control_dock') and self.excitation_control_dock.isVisible():
               try:
                  logger.info("Reloading Excitation Control panel after hardware config change")
                  if hasattr(self, 'excitation_control_tab') and self.excitation_control_tab:
                     self.excitation_control_tab.set_excitation(self.excitation)
                     self.excitation_control_tab.set_config_path(self._config_path)
               except Exception as e:
                  logger.exception("Failed to reload Excitation Control panel: %s", e)
            
         else:
            QtWidgets.QMessageBox.warning(
               self, "Load Hardware Config",
               f"Config loaded, but some hardware could not be opened.\nCheck connections, then retry.\n\n{path}",
            )
            self.statusBar().showMessage(f"Hardware config loaded (devices NOT opened): {path}", 6000)
      except Exception:
         logger.exception("Eager device build failed after loading config %s", path)

   def load_full_experiment(self):
      try:
         default_dir = str(self._project_experiments_dir())
      except Exception:
         default_dir = ""
      path, _ = QtWidgets.QFileDialog.getOpenFileName(
         self, "Load Experiment", default_dir, "Experiment (*.json)"
      )
      if not path:
         return

      # Update experiment filename and window title
      try:
         self._experiment_filename = Path(path).name
      except Exception:
         self._experiment_filename = "experiment"  # Fallback if path processing fails
      self._update_window_title()

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
      self.demo_tab.output_dir_edit.setText(
         str(self._resolve_output_dir(demo.get("output_dir"), coerce_legacy_data_path=True))
      )
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
         
         # --- Restore Plugins ---
         if "plugins" in data and PLUGINS_AVAILABLE and self._plugin_manager:
            try:
               plugin_data = data["plugins"]
               plugin_configs = plugin_data.get("configs", {})
               enabled_plugins = plugin_data.get("enabled", [])
               
               # Load plugin configurations
               for plugin_name, config in plugin_configs.items():
                  self._plugin_manager.configure_plugin(plugin_name, config)
               
               # Set enabled state
               for plugin_name in self._plugin_manager.get_all_plugins().keys():
                  if plugin_name in enabled_plugins:
                     self._plugin_manager.enable_plugin(plugin_name)
                  else:
                     self._plugin_manager.disable_plugin(plugin_name)
               
               logger.info(f"Restored {len(enabled_plugins)} plugins from experiment")
            except Exception as e:
               logger.warning(f"Failed to restore plugin information: {e}")
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

      # Update experiment filename and window title
      try:
         self._experiment_filename = Path(path).name
      except Exception:
         self._experiment_filename = "experiment"  # Fallback if path processing fails
      self._update_window_title()

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
         self.demo_tab.output_dir_edit.setText(
            str(self._resolve_output_dir(demo.get("output_dir"), coerce_legacy_data_path=True))
         )
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