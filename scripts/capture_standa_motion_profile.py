"""Read Standa X/Y move profiles and optionally save them in hardware config.

Default usage is report-only:
    python scripts/capture_standa_motion_profile.py

Explicitly persist current controller speed, acceleration, and deceleration:
    python scripts/capture_standa_motion_profile.py --write-config
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pylablib.devices import Standa

DEFAULT_CONFIG = _ROOT / "config" / "devices_I_lab_mmscale_resistance.json"
PROFILE_FIELDS = {
    "max_velocity": "speed",
    "max_acceleration": "accel",
    "max_deceleration": "decel",
}


def read_profiles(config_path: Path) -> dict[str, dict[str, int]]:
    """Read current motion profiles from both controllers named in config."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    stage = config.get("stage")
    if not isinstance(stage, dict) or stage.get("type") != "StandaStageXY":
        raise ValueError(f"{config_path} does not define a StandaStageXY stage")

    profiles = {}
    for axis, port_key in (("x", "com_x"), ("y", "com_y")):
        port = stage.get(port_key)
        if not port:
            raise ValueError(f"Missing stage.{port_key} in {config_path}")
        controller = None
        try:
            controller = Standa.Standa8SMC(port)
            params = controller.get_move_parameters()
            profiles[axis] = {
                config_key: int(getattr(params, pylablib_key))
                for config_key, pylablib_key in PROFILE_FIELDS.items()
            }
        finally:
            if controller is not None:
                controller.close()
    return profiles


def write_profiles(config_path: Path, profiles: dict[str, dict[str, int]]) -> Path:
    """Back up and atomically update X/Y motion profiles in the config."""
    with config_path.open("r", encoding="utf-8") as config_file:
        config: dict[str, Any] = json.load(config_file)

    stage = config.get("stage")
    if not isinstance(stage, dict) or stage.get("type") != "StandaStageXY":
        raise ValueError(f"{config_path} does not define a StandaStageXY stage")
    motors = stage.setdefault("motors", {})
    if not isinstance(motors, dict):
        raise ValueError("stage.motors must be a JSON object")

    for axis in ("x", "y"):
        if axis not in profiles:
            raise ValueError(f"Missing captured profile for {axis.upper()} axis")
        motor_config = motors.setdefault(axis, {})
        if not isinstance(motor_config, dict):
            raise ValueError(f"stage.motors.{axis} must be a JSON object")
        for field in PROFILE_FIELDS:
            motor_config[field] = int(profiles[axis][field])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup_path = config_path.with_name(f"{config_path.name}.bak.{timestamp}")
    shutil.copy2(config_path, backup_path)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=config_path.parent,
            prefix=f".{config_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            json.dump(config, temp_file, indent=2)
            temp_file.write("\n")
            temp_path = Path(temp_file.name)
        os.replace(temp_path, config_path)
    except Exception:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
        raise
    return backup_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"hardware configuration (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="back up the config and save values read from both controllers",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    try:
        profiles = read_profiles(config_path)
    except Exception as exc:
        print(f"ERROR: could not read Standa motion profiles: {exc}", file=sys.stderr)
        return 1

    print(f"Current Standa move profile ({config_path}):")
    print("Units: stepper speed in microsteps/s; acceleration/deceleration in microsteps/s^2")
    for axis in ("x", "y"):
        profile = profiles[axis]
        print(
            f"  {axis.upper()} ({profile['max_velocity']} microsteps/s, "
            f"accel {profile['max_acceleration']} microsteps/s^2, "
            f"decel {profile['max_deceleration']} microsteps/s^2)"
        )

    if not args.write_config:
        print("Report only; config was not modified. Use --write-config to save these values.")
        return 0

    try:
        backup_path = write_profiles(config_path, profiles)
    except Exception as exc:
        print(f"ERROR: could not update config: {exc}", file=sys.stderr)
        return 1

    print(f"Updated config. Original backed up to: {backup_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
