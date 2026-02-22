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
    if detector_cfg.get("type") == "simulated":
        detector = SimulatedDetector()
        detector.set_scale(detector_cfg.get("scale", 1.0), detector_cfg.get("offset", 0.0))
    else:
        detector = SimulatedDetector()  # default

    return camera, stage, focus, light, fw, detector