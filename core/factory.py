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
import logging
from devices.multimeter import Multimeter
from devices.standa_stage import StandaStageXY
from devices.simulated import SimulatedCamera, SimulatedDetector, SimulatedFilterWheel, SimulatedLight, SimulatedFocus, SimulatedStageXY
from devices.voltage_meter_comport import ComPort
from devices.scaled import ScaledStageXY, ScaledFocusZ, ScaledLightSource


logger = logging.getLogger(__name__)


def save_config(cfg: dict, path: str = "config/default_devices.json") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _ensure_scaling_blocks(cfg: dict) -> tuple[dict, bool]:
    """Backfill missing scaling keys in an existing config.

    Returns (cfg, changed).
    """
    changed = False

    def _ensure(obj: dict, key: str, default: dict):
        nonlocal changed
        if not isinstance(obj, dict):
            return
        if key not in obj or not isinstance(obj.get(key), dict):
            obj[key] = dict(default)
            changed = True
        else:
            # fill missing keys
            for k, v in default.items():
                if k not in obj[key]:
                    obj[key][k] = v
                    changed = True

    stage = cfg.get("stage")
    if isinstance(stage, dict):
        _ensure(stage, "scaling", {"x_scale": 1.0, "x_offset": 0.0, "y_scale": 1.0, "y_offset": 0.0})

    focus = cfg.get("focus")
    if isinstance(focus, dict):
        _ensure(focus, "scaling", {"scale": 1.0, "offset": 0.0})

    light = cfg.get("light")
    if isinstance(light, dict):
        _ensure(light, "scaling", {"scale": 1.0, "offset": 0.0})

    detector = cfg.get("detector")
    if isinstance(detector, list):
        for dc in detector:
            if isinstance(dc, dict):
                if "scale" not in dc:
                    dc["scale"] = 1.0
                    changed = True
                if "offset" not in dc:
                    dc["offset"] = 0.0
                    changed = True
    elif isinstance(detector, dict):
        if "scale" not in detector:
            detector["scale"] = 1.0
            changed = True
        if "offset" not in detector:
            detector["offset"] = 0.0
            changed = True

    return cfg, changed

def load_config(path="config/default_devices.json"):
    if not os.path.exists(path):
        # Generate default config
        default_config = {
            "stage": {"type": "simulated", "scaling": {"x_scale": 1.0, "x_offset": 0.0, "y_scale": 1.0, "y_offset": 0.0}},
            "focus": {"type": "simulated", "scaling": {"scale": 1.0, "offset": 0.0}},
            "camera": {"type": "simulated"},
            "light": {"type": "simulated", "scaling": {"scale": 1.0, "offset": 0.0}},
            "filter_wheel": {"type": "simulated"},
            "detector": {"type": "simulated", "scale": 1.0, "offset": 0.0}
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(default_config, f, indent=2)
        return default_config
    with open(path) as f:
        cfg = json.load(f)

    # Backfill missing scaling keys and persist the normalized config.
    try:
        cfg, changed = _ensure_scaling_blocks(cfg)
        if changed:
            save_config(cfg, path)
    except Exception:
        pass

    return cfg

def build_devices(config_path="config/default_devices.json"):
    try:
        logger.info("Building devices (config=%s)", os.path.abspath(config_path))
    except Exception:
        pass
    cfg = load_config(config_path)

    # Stage
    stage_cfg = cfg.get("stage", {"type": "simulated"})
    if stage_cfg.get("type") == "StandaStageXY":
        stage = StandaStageXY(
            com_x=stage_cfg["com_x"],
            com_y=stage_cfg["com_y"]
        )
    else:
        stage = SimulatedStageXY()

    # apply stage scaling if configured
    try:
        sc = stage_cfg.get("scaling") if isinstance(stage_cfg, dict) else None
        if isinstance(sc, dict):
            xs = float(sc.get("x_scale", 1.0))
            xo = float(sc.get("x_offset", 0.0))
            ys = float(sc.get("y_scale", 1.0))
            yo = float(sc.get("y_offset", 0.0))
            if xs != 1.0 or xo != 0.0 or ys != 1.0 or yo != 0.0:
                stage = ScaledStageXY(stage, x_scale=xs, x_offset=xo, y_scale=ys, y_offset=yo)
    except Exception:
        pass

    # Focus
    focus_cfg = cfg.get("focus", {"type": "simulated"})
    if focus_cfg.get("type") == "simulated":
        focus = SimulatedFocus()
    else:
        focus = SimulatedFocus()  # default

    # apply focus scaling if configured
    try:
        sc = focus_cfg.get("scaling") if isinstance(focus_cfg, dict) else None
        if isinstance(sc, dict):
            s = float(sc.get("scale", 1.0))
            o = float(sc.get("offset", 0.0))
            if s != 1.0 or o != 0.0:
                focus = ScaledFocusZ(focus, scale=s, offset=o)
    except Exception:
        pass

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

    # apply light scaling if configured
    try:
        sc = light_cfg.get("scaling") if isinstance(light_cfg, dict) else None
        if isinstance(sc, dict):
            s = float(sc.get("scale", 1.0))
            o = float(sc.get("offset", 0.0))
            if s != 1.0 or o != 0.0:
                light = ScaledLightSource(light, scale=s, offset=o)
    except Exception:
        pass

    # Filter Wheel
    fw_cfg = cfg.get("filter_wheel", {"type": "simulated"})
    if fw_cfg.get("type") == "simulated":
        fw = SimulatedFilterWheel()
    else:
        fw = SimulatedFilterWheel()  # default

    # Detector
    detector_cfg = cfg.get("detector", {"type": "simulated"})
    # Allow detector config to be a list to build multiple detectors
    if isinstance(detector_cfg, list):
        detectors = []
        for idx, dc in enumerate(detector_cfg):
            if dc.get("type") == "simulated":
                d = SimulatedDetector()
                d.set_scale(dc.get("scale", 1.0), dc.get("offset", 0.0))
            elif dc.get("type") in ("ComPort", "voltage_comport", "serial_voltage"):
                # build a ComPort detector
                port = dc.get("port")
                baud = int(dc.get("baudrate", 115200))
                fmt = dc.get("format", dc.get("sample_format", "int24"))
                timeout = float(dc.get("read_timeout", 0.1))
                d = ComPort(
                    port=port,
                    baudrate=baud,
                    read_timeout=timeout,
                    sample_format=fmt,
                    name=dc.get("name"),
                )
                # set optional scale/offset
                d.set_scale(dc.get("scale", 1000.0), dc.get("offset", 0.0))
            elif dc.get("type") == "Multimeter":
                d = Multimeter(gpib=dc.get("gpib"), name=dc.get("name"))
                try:
                    d.set_scale(dc.get("scale", 1.0), dc.get("offset", 0.0))
                except Exception:
                    pass
            else:
                raise ValueError(f"Unknown detector type: {dc.get('type')}")

            # Prefer display/ID name from config for UI and saving
            try:
                cfg_name = dc.get("name")
                if cfg_name:
                    d.name = cfg_name
                else:
                    # keep existing d.name if present; otherwise use a stable fallback
                    if not getattr(d, "name", None):
                        d.name = dc.get("port") or f"detector{idx + 1}"
            except Exception:
                pass
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
                name=detector_cfg.get("name"),
            )
            detector.set_scale(detector_cfg.get("scale", 1.0), detector_cfg.get("offset", 0.0))
        elif detector_cfg.get("type") == "Multimeter":
            detector = Multimeter(gpib=detector_cfg.get("gpib"), name=detector_cfg.get("name"))
            try:
                detector.set_scale(detector_cfg.get("scale", 1.0), detector_cfg.get("offset", 0.0))
            except Exception:
                pass
        else:
            detector = SimulatedDetector()  # default

        # Prefer display/ID name from config for UI and saving (single detector config)
        try:
            cfg_name = detector_cfg.get("name") if isinstance(detector_cfg, dict) else None
            if cfg_name:
                detector.name = cfg_name
        except Exception:
            pass

    try:
        det_count = len(detector) if isinstance(detector, list) else (1 if detector is not None else 0)
        logger.info(
            "Built devices (camera=%s stage=%s focus=%s light=%s fw=%s detectors=%s)",
            type(camera).__name__ if camera is not None else None,
            type(stage).__name__ if stage is not None else None,
            type(focus).__name__ if focus is not None else None,
            type(light).__name__ if light is not None else None,
            type(fw).__name__ if fw is not None else None,
            det_count,
        )
    except Exception:
        pass

    return camera, stage, focus, light, fw, detector