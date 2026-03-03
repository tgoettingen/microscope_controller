# from __future__ import annotations
# from typing import Literal

# from devices.mock import (
#    MockCamera,
#    MockStageXY,
#    MockFocusZ,
#    MockLightSource,
#    MockFilterWheel,
#    MockDetector,
# )

# Mode = Literal["sim", "real"]




# def build_devices(mode: Mode = "sim"):
#    if mode == "sim":
#       cam = MockCamera()
#       stage = MockStageXY()
#       focus = MockFocusZ()
#       light = MockLightSource()
#       fw = MockFilterWheel()
#       det = MockDetector()
#    else:
#       raise NotImplementedError("Real mode not implemented yet")

#    return cam, stage, focus, light, fw, det


import json
import os
from devices.standa_stage import StandaStageXY
from devices.simulated import SimulatedCamera, SimulatedDetector, SimulatedFilterWheel, SimulatedLight, SimulatedFocus, SimulatedStageXY
from devices.voltage_meter_comport import ComPort

def load_config(path="config/default_devices.json"):
    if not os.path.exists(path):
        # Generate default config
        default_config = {
            "stage": {"type": "simulated"},
            "focus": {"type": "simulated"},
            "camera": {"type": "simulated"},
            "light": {"type": "simulated"},
            "filter_wheel": {"type": "simulated"},
            "detector": {"type": "simulated", "scale": 1.0, "offset": 0.0}
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config
    with open(path) as f:
        return json.load(f)

def build_devices(config_path="config/default_devices.json"):
    cfg = load_config(config_path)

    # Stage
    stage_cfg = cfg.get("stage", {"type": "simulated"})
    if stage_cfg.get("type") == "standa_xy":
        stage = StandaStageXY(
            com_x=stage_cfg["com_x"],
            com_y=stage_cfg["com_y"]
        )
    else:
        stage = SimulatedStageXY()

    # Focus
    focus_cfg = cfg.get("focus", {"type": "simulated"})
    if focus_cfg.get("type") == "simulated":
        focus = SimulatedFocus()
    else:
        focus = SimulatedFocus()  # default

    # Camera
    camera_cfg = cfg.get("camera", {"type": "simulated"})
    if camera_cfg.get("type") == "simulated":
        camera = SimulatedCamera()
    else:
        camera = SimulatedCamera()  # default

    # Light
    light_cfg = cfg.get("light", {"type": "simulated"})
    if light_cfg.get("type") == "simulated":
        light = SimulatedLight()
    else:
        light = SimulatedLight()  # default

    # Filter Wheel
    fw_cfg = cfg.get("filter_wheel", {"type": "simulated"})
    if fw_cfg.get("type") == "simulated":
        fw = SimulatedFilterWheel()
    else:
        fw = SimulatedFilterWheel()  # default

    # Detector
    detector_cfg = cfg.get("detector", {"type": "simulated", "scale": 1.0, "offset": 0.0})
    # Allow detector config to be a list to build multiple detectors
    if isinstance(detector_cfg, list):
        detectors = []
        for dc in detector_cfg:
            if dc.get("type") == "simulated":
                d = SimulatedDetector()
                d.set_scale(dc.get("scale", 1.0), dc.get("offset", 0.0))
            elif dc.get("type") in ("comport", "voltage_comport", "serial_voltage"):
                # build a ComPort detector
                port = dc.get("port")
                baud = int(dc.get("baudrate", 115200))
                fmt = dc.get("format", dc.get("sample_format", "int24"))
                timeout = float(dc.get("read_timeout", 0.1))
                d = ComPort(port=port, baudrate=baud, read_timeout=timeout, sample_format=fmt)
                # set optional scale/offset
                d.set_scale(dc.get("scale", 1.0), dc.get("offset", 0.0))
            else:
                d = SimulatedDetector()
            detectors.append(d)
        detector = detectors
    else:
        if detector_cfg.get("type") == "simulated":
            detector = SimulatedDetector()
            detector.set_scale(detector_cfg.get("scale", 1.0), detector_cfg.get("offset", 0.0))
        elif detector_cfg.get("type") in ("comport", "voltage_comport", "serial_voltage"):
            detector = ComPort(
                port=detector_cfg.get("port"),
                baudrate=int(detector_cfg.get("baudrate", 115200)),
                read_timeout=float(detector_cfg.get("read_timeout", 0.1)),
                sample_format=detector_cfg.get("format", detector_cfg.get("sample_format", "int24")),
            )
            detector.set_scale(detector_cfg.get("scale", 1.0), detector_cfg.get("offset", 0.0))
        else:
            detector = SimulatedDetector()  # default

    return camera, stage, focus, light, fw, detector