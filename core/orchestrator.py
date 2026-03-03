from __future__ import annotations
import time
from typing import Callable, Optional, Any

from devices.base import Camera, StageXY, FocusZ, LightSource, FilterWheel, Detector
from .experiment import ExperimentDefinition


ImageCallback = Callable[[Any, dict], None]
# DetectorCallback: either (value, meta) or (detector_id, value, meta)
DetectorCallback = Callable[..., None]


class Orchestrator:
    def __init__(
        self,
        camera: Camera,
        stage: StageXY,
        focus: FocusZ,
        light: LightSource,
        filter_wheel: FilterWheel,
        detector: Optional[Detector] | list[Detector] | None = None,
        settle_xy_s: float = 0.05,
        settle_z_s: float = 0.02,
        on_image: Optional[ImageCallback] = None,
        on_detector_sample: Optional[DetectorCallback] = None,
    ):
        self.camera = camera
        self.stage = stage
        self.focus = focus
        self.light = light
        self.filter_wheel = filter_wheel
        # normalize to list of detectors
        if detector is None:
            self.detectors: list[Detector] = []
        elif isinstance(detector, list):
            self.detectors = detector
        else:
            self.detectors = [detector]
        self.detector = detector
        self.settle_xy_s = settle_xy_s
        self.settle_z_s = settle_z_s
        self.on_image = on_image
        self.on_detector_sample = on_detector_sample
        self._running = False

    def initialize(self) -> None:
        for dev in [self.camera, self.stage, self.focus, self.light, self.filter_wheel] + self.detectors:
            if dev is None:
                continue
            dev.connect()
            dev.reset()
        # connect async detector signals if available
        for det in self.detectors:
            try:
                if hasattr(det, "sample_received"):
                    det.sample_received.connect(self._on_detector_signal)
            except Exception:
                pass

    def shutdown(self) -> None:
        for dev in [self.camera, self.stage, self.focus, self.light, self.filter_wheel] + self.detectors:
            if dev is None:
                continue
            try:
                dev.disconnect()
            except Exception:
                pass
        # disconnect signals
        for det in self.detectors:
            try:
                if hasattr(det, "sample_received"):
                    det.sample_received.disconnect(self._on_detector_signal)
            except Exception:
                pass

    def stop(self) -> None:
        self._running = False

    def run_experiment(self, exp: ExperimentDefinition) -> None:
        self._running = True
        start_time = time.time()
        print(f"Starting experiment: {exp.name}")

        for t_idx in exp.iter_timepoints():
            if not self._running:
                break
            t_start = time.time()

            for pos_idx, pos in enumerate(exp.positions):
                if not self._running:
                    break

                self.stage.move_to(pos.x, pos.y)
                time.sleep(self.settle_xy_s)

                base_z = pos.z if pos.z is not None else self.focus.get_position()

                for z in exp.iter_z_positions(base_z):
                    if not self._running:
                        break
                    if z is not None:
                        self.focus.move_to(z)
                        time.sleep(self.settle_z_s)

                    for ch in exp.channels:
                        if not self._running:
                            break
                        self._acquire_channel(exp, t_idx, pos_idx, z, ch)

            if exp.timelapse is not None and self._running:
                elapsed = time.time() - t_start
                remaining = exp.timelapse.interval_s - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        print(f"Experiment finished in {time.time() - start_time:.1f} s")

    def _acquire_channel(self, exp, t_idx, pos_idx, z, ch) -> None:
        self.filter_wheel.set_position(ch.filter_position)
        self.light.set_intensity(ch.light_intensity)
        self.camera.set_exposure(ch.exposure_ms)

        self.light.on()
        img = self.camera.snap()
        self.light.off()

        meta = {
            "experiment": exp.name,
            "t": t_idx,
            "pos": pos_idx,
            "z": z,
            "channel": ch.name,
            "timestamp": time.time(),
        }

        if self.on_image is not None:
            self.on_image(img, meta)
        # synchronous detector reads (if present)
        if self.detectors and self.on_detector_sample is not None:
            for det in self.detectors:
                try:
                    if hasattr(det, "read_value"):
                        val = det.read_value()
                        det_id = getattr(det, "name", getattr(det, "port", "detector"))
                        dmeta = {
                            "experiment": exp.name,
                            "t": t_idx,
                            "pos": pos_idx,
                            "z": z,
                            "channel": ch.name,
                            "timestamp": time.time(),
                        }
                        # prefer (det_id, value, meta)
                        try:
                            self.on_detector_sample(det_id, val, dmeta)
                        except TypeError:
                            # fallback to older (value, meta)
                            self.on_detector_sample(val, dmeta)
                except Exception:
                    continue

    def _on_detector_signal(self, det_id: str, timestamp: float, value: float) -> None:
        # called when detector emits asynchronously
        if self.on_detector_sample is None:
            return
        dmeta = {"timestamp": timestamp, "detector_id": det_id}
        try:
            self.on_detector_sample(det_id, float(value), dmeta)
        except TypeError:
            try:
                self.on_detector_sample(float(value), dmeta)
            except Exception:
                pass