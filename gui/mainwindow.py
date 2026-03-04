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

   def _build_ui(self):
      self._create_menus()

      # --- Left panel: Demo + Multi-Axis tabs ---
      self.demo_tab = ExperimentTab()          # renamed ExperimentTab
      self.multi_tab = MultiAxisTab()

      self.left_tabs = QtWidgets.QTabWidget()
      self.left_tabs.addTab(self.demo_tab, "Demo")
      self.left_tabs.addTab(self.multi_tab, "Multi‑Axis")

      # --- Right panel: Live view ---
      self.live_tab = LiveTab()
      # connect hover info to status bar
      self.live_tab.hover_info.connect(lambda s: self.statusBar().showMessage(s))
      # connect stream toggle signals from live tab
      self.live_tab.stream_toggled.connect(self._on_stream_toggled)

      # --- Splitter layout ---
      splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
      splitter.addWidget(self.left_tabs)
      splitter.addWidget(self.live_tab)

      # Optional: set initial sizes
      splitter.setSizes([450, 950])

      self.setCentralWidget(splitter)

      # --- Connect signals ---
      self.demo_tab.start_requested.connect(self._start_experiment)
      self.demo_tab.stop_requested.connect(self._stop_experiment)

      self.multi_tab.start_requested.connect(self._start_multiaxis)
      self.multi_tab.stop_requested.connect(self._stop_multiaxis)

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
      quit_action = QAction("Quit", self)
      quit_action.triggered.connect(self.close)
      file_menu.addAction(quit_action)

      help_menu = menubar.addMenu("&Help")
      about_action = QAction("About", self)
      about_action.triggered.connect(self._show_about)
      help_menu.addAction(about_action)

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
               axes.append(DetectorAxis(det, p["scales"]))
            elif t == "Round":
               axes.append(RoundAxis(p["n_rounds"]))

      self.live_tab.reset_multiaxis()

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
                      # add to live tab (multiaxis) — include detector id
                      try:
                         self.live_tab.add_multiaxis_detector(det_id, state, val)
                      except TypeError:
                         # fallback for older signature
                         self.live_tab.add_multiaxis_detector(state, val)
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
      self.multi_runner = MultiAxisRunner(exp)

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


def main():
   app = QtWidgets.QApplication(sys.argv)
   win = MainWindow()
   win.resize(1400, 900)
   win.show()
   sys.exit(app.exec())


if __name__ == "__main__":
   main()