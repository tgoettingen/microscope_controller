from typing import Callable, Any, Dict
import time

from devices.base import Camera, StageXY, FocusZ, LightSource, FilterWheel, Detector
from core.experiment import ChannelConfig
from core.multiaxis import (
    MultiAxisExperiment,
    MultiAxisRunner,
    XAxis,
    YAxis,
    ZAxis,
    ChannelAxis,
    DetectorAxis,
    RoundAxis,
)


def build_camera_multiaxis_runner(
    camera: Camera,
    stage: StageXY,
    focus: FocusZ,
    light: LightSource,
    fw: FilterWheel,
    detector: Detector | list[Detector] | None,
    on_image: Callable[[Any, Dict], None],
    on_detector_sample: Callable[..., None] | None,
) -> MultiAxisRunner:
    channels = [
        ChannelConfig(name="BF", filter_position=0, light_intensity=10.0, exposure_ms=20.0),
        ChannelConfig(name="GFP", filter_position=1, light_intensity=30.0, exposure_ms=50.0),
    ]

    axes = [
        RoundAxis(n_rounds=1),  # or None for infinite
        YAxis(stage, start=0.0, end=1000.0, step=500.0),
        XAxis(stage, start=0.0, end=1000.0, step=500.0),
        ChannelAxis(camera, light, fw, channels, wait_s=0.01),
    ]

    def measure(state: Dict[str, Any]):
        ch: ChannelConfig = state["Channel"]
        x = state["X"]
        y = state["Y"]
        rnd = state["Round"]

        img = camera.snap()
        meta = {
            "experiment": "multiaxis_camera",
            "round": rnd,
            "x": x,
            "y": y,
            "channel": ch.name,
            "timestamp": time.time(),
        }
        if on_image is not None:
            on_image(img, meta)

        if detector is not None and on_detector_sample is not None:
            dets = detector if isinstance(detector, list) else [detector]
            for det in dets:
                try:
                    val = det.read_value()
                    det_id = getattr(det, "name", getattr(det, "port", "detector"))
                    dmeta = {
                        "experiment": "multiaxis_camera",
                        "round": rnd,
                        "x": x,
                        "y": y,
                        "channel": ch.name,
                        "timestamp": time.time(),
                    }
                    try:
                        on_detector_sample(det_id, val, dmeta)
                    except TypeError:
                        on_detector_sample(val, dmeta)
                except Exception:
                    continue

    exp = MultiAxisExperiment(axes=axes, measure=measure)
    return MultiAxisRunner(exp)
 
 
 


def build_detector_multiaxis_runner(
    detector: Detector | list[Detector],
    stage: StageXY,
    focus: FocusZ,
    on_detector_sample: Callable[..., None],
) -> MultiAxisRunner:
    scales = [(1.0, 0.0), (2.0, 0.0)]

    axes = [
        RoundAxis(n_rounds=None),  # infinite rounds until stop()
        ZAxis(focus, start=90.0, end=110.0, step=10.0),
        XAxis(stage, start=0.0, end=1000.0, step=250.0),
        DetectorAxis(detector, scales),
    ]

    def measure(state: Dict[str, Any]):
        dets = detector if isinstance(detector, list) else [detector]
        for det in dets:
            try:
                val = det.read_value()
                det_id = getattr(det, "name", getattr(det, "port", "detector"))
                meta = {
                    "experiment": "multiaxis_detector",
                    "round": state["Round"],
                    "x": state["X"],
                    "z": state["Z"],
                    "scale": state["Detector"],
                    "timestamp": time.time(),
                }
                try:
                    on_detector_sample(det_id, val, meta)
                except TypeError:
                    on_detector_sample(val, meta)
            except Exception:
                continue

    exp = MultiAxisExperiment(axes=axes, measure=measure)
    return MultiAxisRunner(exp)