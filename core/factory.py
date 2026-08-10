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
from devices.excitation_device import ExcitationDevice, SimulatedExcitationDevice
from devices.motor_specs import (
    CATALOGUE as MOTOR_SPEC_CATALOGUE,
    DEFAULT_DEVICE_TYPE_TO_SPEC,
    get_spec as _get_motor_spec,
    default_spec_for as _default_motor_spec_for,
)


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
        _ensure(stage, "range", {"x_min": None, "x_max": None, "y_min": None, "y_max": None})
        _ensure(stage, "motors", {
            "x": {"motor_spec_id": None, "operating_limits": None},
            "y": {"motor_spec_id": None, "operating_limits": None},
        })

    focus = cfg.get("focus")
    if isinstance(focus, dict):
        _ensure(focus, "scaling", {"scale": 1.0, "offset": 0.0})
        _ensure(focus, "motors", {
            "z": {"motor_spec_id": None, "operating_limits": None},
        })

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
            "stage": {
                "type": "simulated",
                "scaling": {"x_scale": 1.0, "x_offset": 0.0, "y_scale": 1.0, "y_offset": 0.0},
                "range": {"x_min": None, "x_max": None, "y_min": None, "y_max": None},
                "motors": {
                    "x": {"motor_spec_id": "sim_stage_x", "operating_limits": None},
                    "y": {"motor_spec_id": "sim_stage_y", "operating_limits": None},
                },
            },
            "focus": {
                "type": "simulated",
                "scaling": {"scale": 1.0, "offset": 0.0},
                "motors": {
                    "z": {"motor_spec_id": "sim_focus_z", "operating_limits": None},
                },
            },
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

def _merged_motor_limits(
    device_type: str,
    axis_role: str,
    motor_cfg: dict | None,
) -> dict | None:
    """Return a motor-limits dict (suitable for ScaledStageXY/ScaledFocusZ)
    by merging the referenced spec's defaults with any config overrides.

    Returns ``None`` when no limits were configured at all (all defaults +
    no overrides resolve to None).
    """
    if not isinstance(motor_cfg, dict):
        motor_cfg = {}

    spec_id = motor_cfg.get("motor_spec_id")
    spec = None
    if spec_id:
        spec = _get_motor_spec(str(spec_id))
    if spec is None:
        spec = _default_motor_spec_for(str(device_type or ""), str(axis_role or ""))

    # 1. Build base limits from the spec (if any)
    base: dict = {}
    if spec is not None:
        def _r(t):
            return (t[0], t[1]) if t is not None else (None, None)
        base["position"]  = _r(spec.travel_range_steps)
        base["speed_rpm"] = _r(spec.speed_rpm_range)
        base["torque_nm"] = _r(spec.torque_nm_range)
        base["voltage_v"] = _r(spec.voltage_v_range)
        base["power_w"]   = _r(spec.power_w_range)

    # 2. Overlay operating_limits from config (user tightening)
    override = motor_cfg.get("operating_limits") if isinstance(motor_cfg, dict) else None
    merged = dict(base)
    if isinstance(override, dict):
        for k in ("position", "speed_rpm", "torque_nm", "voltage_v", "power_w"):
            v = override.get(k)
            if not isinstance(v, (tuple, list)) or len(v) != 2:
                continue
            def _f(x):
                if x is None:
                    return None
                try:
                    return float(x)
                except Exception:
                    return None
            merged[k] = (_f(v[0]), _f(v[1]))

    # 3. Drop keys that are fully unbounded so callers can easily skip wrapping
    pruned = {
        k: (lo, hi) for k, (lo, hi) in merged.items()
        if (lo is not None or hi is not None)
    }
    return pruned or None


def build_devices(config_path="config/default_devices.json"):
    try:
        logger.info("Building devices (config=%s)", os.path.abspath(config_path))
    except Exception:
        pass
    cfg = load_config(config_path)

    # Stage
    stage_cfg = cfg.get("stage", {"type": "simulated"})
    stage_type = stage_cfg.get("type", "simulated") if isinstance(stage_cfg, dict) else "simulated"
    if stage_cfg.get("type") == "StandaStageXY":
        stage = StandaStageXY(
            com_x=stage_cfg["com_x"],
            com_y=stage_cfg["com_y"]
        )
        stage_type_for_motor = "StandaStageXY"
    else:
        stage = SimulatedStageXY()
        stage_type_for_motor = "simulated"

    # apply stage scaling if configured
    try:
        sc = stage_cfg.get("scaling") if isinstance(stage_cfg, dict) else None
        motors_cfg = stage_cfg.get("motors") if isinstance(stage_cfg, dict) else None
        mx = _merged_motor_limits(stage_type_for_motor, "stage_x", motors_cfg.get("x") if isinstance(motors_cfg, dict) else None) if isinstance(motors_cfg, dict) else _merged_motor_limits(stage_type_for_motor, "stage_x", None)
        my = _merged_motor_limits(stage_type_for_motor, "stage_y", motors_cfg.get("y") if isinstance(motors_cfg, dict) else None) if isinstance(motors_cfg, dict) else _merged_motor_limits(stage_type_for_motor, "stage_y", None)

        if isinstance(sc, dict):
            xs = float(sc.get("x_scale", 1.0))
            xo = float(sc.get("x_offset", 0.0))
            ys = float(sc.get("y_scale", 1.0))
            yo = float(sc.get("y_offset", 0.0))
        else:
            xs, xo, ys, yo = 1.0, 0.0, 1.0, 0.0
        # read optional range block from config (soft user travel limits)
        rc = stage_cfg.get("range") if isinstance(stage_cfg, dict) else None
        x_min = x_max = y_min = y_max = None
        if isinstance(rc, dict):
            def _as_float_or_none(v):
                if v is None:
                    return None
                try:
                    return float(v)
                except Exception:
                    return None
            x_min = _as_float_or_none(rc.get("x_min"))
            x_max = _as_float_or_none(rc.get("x_max"))
            y_min = _as_float_or_none(rc.get("y_min"))
            y_max = _as_float_or_none(rc.get("y_max"))
        if (xs != 1.0 or xo != 0.0 or ys != 1.0 or yo != 0.0
                or any(v is not None for v in (x_min, x_max, y_min, y_max))
                or mx is not None or my is not None):
            stage = ScaledStageXY(
                stage,
                x_scale=xs, x_offset=xo, y_scale=ys, y_offset=yo,
                x_range=(x_min, x_max) if (x_min is not None or x_max is not None) else None,
                y_range=(y_min, y_max) if (y_min is not None or y_max is not None) else None,
                motor_limits_x=mx,
                motor_limits_y=my,
            )
    except Exception:
        logger.exception("Failed to wrap stage with scaling/motor limits")

    # Focus
    focus_cfg = cfg.get("focus", {"type": "simulated"})
    focus_type = focus_cfg.get("type", "simulated") if isinstance(focus_cfg, dict) else "simulated"
    if focus_cfg.get("type") == "simulated":
        focus = SimulatedFocus()
    else:
        focus = SimulatedFocus()  # default

    # apply focus scaling + motor limits if configured
    try:
        sc = focus_cfg.get("scaling") if isinstance(focus_cfg, dict) else None
        motors_cfg = focus_cfg.get("motors") if isinstance(focus_cfg, dict) else None
        mz = _merged_motor_limits(str(focus_type or "simulated"), "focus_z",
                                   motors_cfg.get("z") if isinstance(motors_cfg, dict) else None) \
            if isinstance(motors_cfg, dict) \
            else _merged_motor_limits(str(focus_type or "simulated"), "focus_z", None)
        s = float(sc.get("scale", 1.0)) if isinstance(sc, dict) else 1.0
        o = float(sc.get("offset", 0.0)) if isinstance(sc, dict) else 0.0
        if s != 1.0 or o != 0.0 or mz is not None:
            focus = ScaledFocusZ(focus, scale=s, offset=o, motor_limits=mz)
    except Exception:
        logger.exception("Failed to wrap focus with scaling/motor limits")

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
                    mode=dc.get("mode"),
                    name=dc.get("name"),
                    reader_hz=float(dc.get("reader_hz", 40.0)),
                    ring_buffer_size=int(dc.get("ring_buffer_size", 8192)),
                    frame_length=int(dc.get("frame_length", 9)),
                    frame_header=dc.get("frame_header", "0A01"),
                    frame_trailer=dc.get("frame_trailer", "010A"),
                    overflow_policy=dc.get("overflow_policy", "overwrite"),
                )
                # set optional scale/offset
                d.set_scale(dc.get("scale", 1000.0), dc.get("offset", 0.0))
            elif dc.get("type") == "Multimeter":
                d = Multimeter(
                    gpib=dc.get("gpib"),
                    nplc=float(dc.get("nplc", 1.1)),
                    name=dc.get("name"),
                    mode=dc.get("mode", "volt_dc"),
                )
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
                mode=detector_cfg.get("mode"),
                name=detector_cfg.get("name"),
                reader_hz=float(detector_cfg.get("reader_hz", 40.0)),
                ring_buffer_size=int(detector_cfg.get("ring_buffer_size", 8192)),
                frame_length=int(detector_cfg.get("frame_length", 9)),
                frame_header=detector_cfg.get("frame_header", "0A01"),
                frame_trailer=detector_cfg.get("frame_trailer", "010A"),
                overflow_policy=detector_cfg.get("overflow_policy", "overwrite"),
            )
            detector.set_scale(detector_cfg.get("scale", 1.0), detector_cfg.get("offset", 0.0))
        elif detector_cfg.get("type") == "Multimeter":
            detector = Multimeter(
                gpib=detector_cfg.get("gpib"),
                nplc=float(detector_cfg.get("nplc", 1.1)),
                name=detector_cfg.get("name"),
                mode=detector_cfg.get("mode", "volt_dc"),
            )
            try:
                detector.set_scale(detector_cfg.get("scale", 1.0), detector_cfg.get("offset", 0.0))
            except Exception:
                pass
        else:
            raise ValueError(f"Unknown detector type: {detector_cfg.get('type')}")

        # Prefer display/ID name from config for UI and saving (single detector config)
        try:
            cfg_name = detector_cfg.get("name") if isinstance(detector_cfg, dict) else None
            if cfg_name:
                detector.name = cfg_name
        except Exception:
            pass

    # Excitation source(s) - support single or multiple
    excitation_cfg = cfg.get("excitation")
    if excitation_cfg:
        if isinstance(excitation_cfg, list):
            # Multiple excitation devices
            excitation = []
            for exc_cfg in excitation_cfg:
                if isinstance(exc_cfg, dict):
                    exc_type = exc_cfg.get("type", "simulated")
                    exc_name = exc_cfg.get("name", f"excitation_{len(excitation)}")
                    if exc_type == "ExcitationDevice":
                        simulate = exc_cfg.get("simulate", False)
                        if simulate:
                            exc_device = SimulatedExcitationDevice(name=exc_name)
                        else:
                            exc_device = ExcitationDevice(
                                name=exc_name,
                                port=exc_cfg.get("port"),
                                channel=exc_cfg.get("channel", 0),
                                simulate=False
                            )
                    else:
                        exc_device = SimulatedExcitationDevice(name=exc_name)
                    excitation.append(exc_device)
        elif isinstance(excitation_cfg, dict):
            # Single excitation device
            excitation_type = excitation_cfg.get("type", "simulated")
            if excitation_type == "ExcitationDevice":
                simulate = excitation_cfg.get("simulate", False)
                if simulate:
                    excitation = SimulatedExcitationDevice(name="excitation")
                else:
                    excitation = ExcitationDevice(
                        name="excitation",
                        port=excitation_cfg.get("port"),
                        channel=excitation_cfg.get("channel", 0),
                        simulate=False
                    )
            else:
                # Any other type (including "simulated") uses simulated device
                excitation = SimulatedExcitationDevice(name="excitation")
        else:
            excitation = SimulatedExcitationDevice(name="excitation")
    else:
        excitation = SimulatedExcitationDevice(name="excitation")

    try:
        det_count = len(detector) if isinstance(detector, list) else (1 if detector is not None else 0)
        logger.info(
            "Built devices (camera=%s stage=%s focus=%s light=%s fw=%s detectors=%s excitation=%s)",
            type(camera).__name__ if camera is not None else None,
            type(stage).__name__ if stage is not None else None,
            type(focus).__name__ if focus is not None else None,
            type(light).__name__ if light is not None else None,
            type(fw).__name__ if fw is not None else None,
            det_count,
            type(excitation).__name__ if excitation is not None else None,
        )
    except Exception:
        pass

    return camera, stage, focus, light, fw, detector, excitation