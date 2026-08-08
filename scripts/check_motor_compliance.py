"""Motor parameter compliance checker.

Usage
-----

    python3 scripts/check_motor_compliance.py \\
            [--config config/default_devices.json] \\
            [--report-dir reports/]

Checks that every motor axis declared in the device config has a complete
set of working-range parameters (speed, torque, voltage, power) and that
the configured values are reasonable when compared against the internal
motor-spec catalogue (:mod:`devices.motor_specs`).

Produces both a human-readable text report and a JSON version suitable for
CI pipelines.  Non-zero exit code if ANY axis fails a check.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running as a standalone script from the project root.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from devices.motor_specs import (  # noqa: E402  (sys.path insertion above)
    CATALOGUE,
    DEFAULT_DEVICE_TYPE_TO_SPEC,
    MotorSpec,
    get_spec,
    default_spec_for,
)
from core.factory import (  # noqa: E402
    load_config,
    _merged_motor_limits,
)


# ---------------------------------------------------------------------------
# Data model for the report
# ---------------------------------------------------------------------------


SEVERITY_INFO = "INFO"
SEVERITY_WARN = "WARN"
SEVERITY_ERROR = "ERROR"


@dataclass
class CheckItem:
    parameter: str           # e.g. "speed_rpm", "torque_nm", "voltage_v", "power_w"
    severity: str            # INFO / WARN / ERROR
    message: str
    configured_min: Optional[float] = None
    configured_max: Optional[float] = None
    spec_min: Optional[float] = None
    spec_max: Optional[float] = None


@dataclass
class AxisReport:
    device_section: str      # "stage" / "focus" / ...
    axis_role: str           # "stage_x" / "stage_y" / "focus_z"
    device_type: str         # "simulated" / "StandaStageXY" / "mock" / ...
    motor_spec_id: Optional[str]
    spec_resolution: str     # how spec_id was resolved: "explicit" / "default" / "none"
    checks: List[CheckItem] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(c.severity == SEVERITY_ERROR for c in self.checks)

    @property
    def warnings(self) -> bool:
        return any(c.severity == SEVERITY_WARN for c in self.checks)


@dataclass
class ComplianceReport:
    config_path: str
    generated_at_utc: str
    axes: List[AxisReport] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(a.failed for a in self.axes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


OPERATING_KEYS: Tuple[Tuple[str, str], ...] = (
    ("speed_rpm", "Speed (rpm)"),
    ("torque_nm", "Torque (N·m)"),
    ("voltage_v", "Voltage (V DC)"),
    ("power_w",   "Power (W)"),
)


def _resolve_spec(
    device_type: str, axis_role: str, motor_cfg: Optional[Dict[str, Any]]
) -> Tuple[Optional[MotorSpec], str, Optional[str]]:
    """Return (spec, resolution-method, spec_id)."""
    if isinstance(motor_cfg, dict):
        explicit = motor_cfg.get("motor_spec_id")
        if explicit:
            spec = get_spec(str(explicit))
            if spec is not None:
                return spec, "explicit", str(explicit)
            # explicit id but unknown -> fall back while warning
            return default_spec_for(device_type, axis_role), "unknown_explicit_default_fallback", str(explicit)
    spec = default_spec_for(device_type, axis_role)
    if spec is None:
        return None, "none", None
    return spec, "default", spec.spec_id


def _as_range_pair(
    value: Any,
) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return (None, None)

    def _f(x):
        if x is None:
            return None
        try:
            return float(x)
        except Exception:
            return None

    return (_f(value[0]), _f(value[1]))


def _check_axis(
    section: str,
    axis_role: str,
    device_type: str,
    motor_cfg: Optional[Dict[str, Any]],
) -> AxisReport:
    spec, resolution, spec_id = _resolve_spec(device_type, axis_role, motor_cfg)
    report = AxisReport(
        device_section=section,
        axis_role=axis_role,
        device_type=device_type,
        motor_spec_id=spec_id,
        spec_resolution=resolution,
    )

    if resolution == "unknown_explicit_default_fallback":
        report.checks.append(CheckItem(
            parameter="spec_id",
            severity=SEVERITY_ERROR,
            message=(
                f"Config motor_spec_id={spec_id!r} is not in the spec catalogue. "
                f"Fell back to default spec for ({device_type}, {axis_role})."
            ),
        ))
    elif resolution == "none":
        report.checks.append(CheckItem(
            parameter="spec_id",
            severity=SEVERITY_WARN,
            message=(
                f"No explicit motor_spec_id and no default mapping for "
                f"({device_type}, {axis_role}). Cannot cross-check against catalogue."
            ),
        ))

    # Get effective operating-limits as would be applied at build time.
    effective = _merged_motor_limits(device_type, axis_role, motor_cfg) or {}
    override_block = (
        motor_cfg.get("operating_limits")
        if isinstance(motor_cfg, dict) else None
    )

    for key, label in OPERATING_KEYS:
        cfg_lo, cfg_hi = _as_range_pair(effective.get(key))
        spec_lo = spec_hi = None
        if spec is not None:
            if key == "speed_rpm":
                spec_lo, spec_hi = spec.speed_rpm_range
            elif key == "torque_nm":
                spec_lo, spec_hi = spec.torque_nm_range
            elif key == "voltage_v":
                spec_lo, spec_hi = spec.voltage_v_range
            elif key == "power_w":
                spec_lo, spec_hi = spec.power_w_range

        item = CheckItem(
            parameter=key,
            severity=SEVERITY_INFO,
            message=f"{label}: OK",
            configured_min=cfg_lo, configured_max=cfg_hi,
            spec_min=spec_lo, spec_max=spec_hi,
        )

        problems: List[str] = []
        severity = SEVERITY_INFO

        # (1) Completeness: at least one bound configured?
        if cfg_lo is None and cfg_hi is None:
            severity = SEVERITY_ERROR
            problems.append("NO limits configured (both min and max are missing)")

        # (2) min > max (invalid range)
        elif (cfg_lo is not None and cfg_hi is not None) and cfg_lo > cfg_hi:
            severity = SEVERITY_ERROR
            problems.append(
                f"Invalid range: min={cfg_lo:.6g} is greater than max={cfg_hi:.6g}"
            )

        # (3) Negative / nonsensical values for physical quantities that must
        #     be non-negative.
        nonneg_ok = True
        for side_name, side_val in (("min", cfg_lo), ("max", cfg_hi)):
            if side_val is None:
                continue
            if side_val < 0 and key != "voltage_v":  # voltage can be bipolar
                severity = SEVERITY_ERROR if severity == SEVERITY_INFO else severity
                problems.append(
                    f"{side_name}={side_val:.6g} is negative — not meaningful for {label}"
                )
                nonneg_ok = False

        # (4) Cross-check against catalogue spec (if available)
        if spec is not None and (cfg_lo is not None or cfg_hi is not None):
            # Tightening is fine.  Warn if configured range extends BEYOND
            # the spec (i.e. asks for something the motor cannot deliver).
            if (spec_lo is not None and cfg_lo is not None and cfg_lo < spec_lo):
                problems.append(
                    f"Configured min ({cfg_lo:.6g}) is below spec min ({spec_lo:.6g})."
                )
                severity = SEVERITY_WARN if severity == SEVERITY_INFO else severity
            if (spec_hi is not None and cfg_hi is not None and cfg_hi > spec_hi):
                problems.append(
                    f"Configured max ({cfg_hi:.6g}) exceeds spec max ({spec_hi:.6g})."
                )
                severity = SEVERITY_WARN if severity == SEVERITY_INFO else severity
            if nonneg_ok and severity == SEVERITY_INFO and not problems:
                # Configured values are both sides and within spec — still
                # check that config doesn't *silently omit* one side that spec
                # explicitly provides (warn).
                missings = []
                if spec_lo is not None and cfg_lo is None:
                    missings.append("min")
                if spec_hi is not None and cfg_hi is None:
                    missings.append("max")
                if missings:
                    severity = SEVERITY_WARN
                    problems.append(
                        f"Spec has explicit {', '.join(missings)} limit but config does not define it."
                    )

        if problems:
            item.severity = severity
            item.message = "; ".join(problems)
        report.checks.append(item)

    # Operating-limit override block sanity: unknown keys
    if isinstance(override_block, dict):
        known = {"position", "speed_rpm", "torque_nm", "voltage_v", "power_w"}
        unknown = [k for k in override_block.keys() if k not in known]
        if unknown:
            report.checks.append(CheckItem(
                parameter="operating_limits",
                severity=SEVERITY_WARN,
                message=(
                    f"Unknown keys in operating_limits: {unknown!r} "
                    f"(expected any of {sorted(known)!r})"
                ),
            ))
    return report


# ---------------------------------------------------------------------------
# Report serialisation
# ---------------------------------------------------------------------------


def _format_range(lo: Optional[float], hi: Optional[float]) -> str:
    def _fmt(v: Optional[float]) -> str:
        return "unset" if v is None else f"{v:.6g}"
    return f"[{_fmt(lo)}, {_fmt(hi)}]"


def _render_text(report: ComplianceReport) -> str:
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append("MOTOR PARAMETER COMPLIANCE REPORT")
    lines.append(f"  Config : {report.config_path}")
    lines.append(f"  Date   : {report.generated_at_utc} UTC")
    lines.append(f"  Axes   : {len(report.axes)}   "
                 f"Failures : {sum(1 for a in report.axes if a.failed)}   "
                 f"Warnings : {sum(1 for a in report.axes if a.warnings)}")
    lines.append("=" * 88)
    lines.append("")

    sev_pad = {"INFO": " ", "WARN": "!", "ERROR": "X"}

    for idx, axis in enumerate(report.axes, 1):
        spec_resolved = (
            f"spec_id={axis.motor_spec_id}  ({axis.spec_resolution})"
            if axis.motor_spec_id else "spec_id=—  (no spec resolved)"
        )
        status = "ERROR" if axis.failed else ("WARN" if axis.warnings else "OK  ")
        lines.append(
            f"[{idx:>2}] {status} {axis.device_section}.{axis.axis_role} "
            f"  device_type={axis.device_type}  {spec_resolved}"
        )
        for c in axis.checks:
            cfg = _format_range(c.configured_min, c.configured_max)
            spc = _format_range(c.spec_min, c.spec_max)
            lines.append(
                f"     {sev_pad.get(c.severity, ' ')} [{c.severity:<5}] "
                f"{c.parameter:<12} limits={cfg:<30} spec={spc:<30} — {c.message}"
            )
        lines.append("")

    # Summary
    lines.append("-" * 88)
    passed = sum(1 for a in report.axes if not a.failed)
    if report.failed:
        lines.append(
            f"RESULT: FAIL  ({passed}/{len(report.axes)} axes passed; "
            f"{len(report.axes) - passed} axes with ERROR-level issues)"
        )
    else:
        warns = sum(1 for a in report.axes if a.warnings)
        lines.append(
            f"RESULT: PASS  ({len(report.axes)}/{len(report.axes)} axes passed; "
            f"{warns} warning(s))"
        )
    return "\n".join(lines)


def _serialise_report(report: ComplianceReport) -> Dict[str, Any]:
    return {
        "config_path": report.config_path,
        "generated_at_utc": report.generated_at_utc,
        "failed": report.failed,
        "axes": [
            {
                **asdict(a),
                "failed": a.failed,
                "warnings": a.warnings,
            }
            for a in report.axes
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _collect_axes_from_config(cfg: Dict[str, Any]) -> List[Tuple[str, str, str, Optional[Dict[str, Any]]]]:
    """Return list of (section, axis_role, device_type, motor_cfg) tuples."""
    axes: List[Tuple[str, str, str, Optional[Dict[str, Any]]]] = []

    stage = cfg.get("stage")
    if isinstance(stage, dict):
        dtype = str(stage.get("type", "simulated"))
        motors = stage.get("motors") if isinstance(stage.get("motors"), dict) else {}
        axes.append(("stage", "stage_x", dtype, motors.get("x") if isinstance(motors, dict) else None))
        axes.append(("stage", "stage_y", dtype, motors.get("y") if isinstance(motors, dict) else None))

    focus = cfg.get("focus")
    if isinstance(focus, dict):
        dtype = str(focus.get("type", "simulated"))
        motors = focus.get("motors") if isinstance(focus.get("motors"), dict) else {}
        axes.append(("focus", "focus_z", dtype, motors.get("z") if isinstance(motors, dict) else None))

    return axes


def build_report(config_path: Path) -> ComplianceReport:
    cfg = load_config(str(config_path))
    report = ComplianceReport(
        config_path=str(config_path),
        generated_at_utc=_dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    )
    for section, role, dtype, motor_cfg in _collect_axes_from_config(cfg):
        report.axes.append(_check_axis(section, role, dtype, motor_cfg))
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Motor parameter compliance checker.",
    )
    parser.add_argument(
        "--config",
        default=str(_ROOT / "config" / "default_devices.json"),
        help="Path to the devices JSON config (default: config/default_devices.json)",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="If provided, write text + JSON reports into this directory.",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.config).resolve()
    if not cfg_path.exists():
        print(f"ERROR: Config not found: {cfg_path}", file=sys.stderr)
        return 2

    report = build_report(cfg_path)
    text = _render_text(report)
    print(text)

    if args.report_dir:
        out_dir = Path(args.report_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        txt_path = out_dir / f"motor_compliance_{stamp}.txt"
        json_path = out_dir / f"motor_compliance_{stamp}.json"
        txt_path.write_text(text, encoding="utf-8")
        json_path.write_text(
            json.dumps(_serialise_report(report), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        print()
        print(f"Wrote text report : {txt_path}")
        print(f"Wrote JSON report : {json_path}")

    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
