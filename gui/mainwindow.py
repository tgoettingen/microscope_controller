import sys
import time
import threading
import json
from pathlib import Path

import numpy as np
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtGui import QAction

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
      det.set_scale(cfg["det_scale"], cfg["det_offset"])

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

      # Use the stored devices
      cam, stage, focus, light, fw, det = self.cam, self.stage, self.focus, self.light, self.fw, self.det

      axes = []
      for cfg in cfgs:
            t = cfg.axis_type
            p = cfg.params
            if t == "X":
               axes.append(XAxis(stage, p["start"], p["end"], p["step"]))
            elif t == "Y":
               axes.append(YAxis(stage, p["start"], p["end"], p["step"]))
            elif t == "Z":
               axes.append(ZAxis(focus, p["start"], p["end"], p["step"]))
            elif t == "Channel":
               axes.append(ChannelAxis(cam, light, fw, p["channels"], p["wait"]))
            elif t == "Detector":
               axes.append(DetectorAxis(det, p["scales"]))
            elif t == "Round":
               axes.append(RoundAxis(p["n_rounds"]))

      self.live_tab.reset_multiaxis()

      def measure(state: dict):
            # camera image if Channel present
            if "Channel" in state:
               img = cam.snap()
               meta = {"experiment": "multi", "state": state, "timestamp": time.time()}
               self._on_image(img, meta)

            # detector value
            if det is not None:
               val = det.read_value()
               self.live_tab.add_multiaxis_detector(state, val)

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

      out_dir = Path(meta.get("output_dir") or Path.cwd() / "data")
      out_dir.mkdir(parents=True, exist_ok=True)

      fname = f"{int(meta['timestamp'])}.npy"
      path = out_dir / fname

      np.save(path, img)

      with open(path.with_suffix(".json"), "w") as f:
         json.dump(safe_meta, f, indent=2)

   def _on_detector_sample(self, value: float, meta: dict):
      timestamp = meta["timestamp"]
      self.live_tab.add_detector_sample(value, timestamp)

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