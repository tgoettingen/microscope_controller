"""Centralised motor parameter specifications (system parameter library).

Every motorised axis that the system can drive is modelled as a ``MotorSpec``:

- **Speed / velocity** range (rpm)
- **Torque** output range (N·m)
- **Voltage** operating range (V DC)
- **Power** loading range (W)
- Plus mechanical travel range (optional) and notes about the typical motor type.

Specs are looked up by a stable ``spec_id`` string. The device config JSON may
optionally refer to a spec via ``motor_spec_id`` and can further tighten the
limits via an ``operating_limits`` override block.

These values are *recommended safe operating envelope* values for the hardware
we have historically connected to the project (Standa 8MT stages + focus
stepper, plus generic simulated/mock motors).  Treat them as a baseline and
tune them against the actual motor datasheet if a different motor is attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MotorSpec:
    """Declarative specification for a single motor axis.

    All range fields use the convention ``(min, max)``; either side may be
    ``None`` when there is no practical software-enforced bound (we still
    record the fact as "not explicitly known" in the compliance report).
    """

    spec_id: str
    description: str
    # Mechanical
    travel_range_steps: Tuple[Optional[float], Optional[float]] = (None, None)
    # Electrical / performance — (min, max)
    speed_rpm_range: Tuple[Optional[float], Optional[float]] = (None, None)
    torque_nm_range: Tuple[Optional[float], Optional[float]] = (None, None)
    voltage_v_range: Tuple[Optional[float], Optional[float]] = (None, None)
    power_w_range: Tuple[Optional[float], Optional[float]] = (None, None)
    # Misc metadata used by the compliance checker
    notes: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Built-in catalogue
# ---------------------------------------------------------------------------

# --- Standa 8MT190-250 + 8SMC4-USB (real hardware stage axes) ------------
#
#   Datasheet-level representative values for a typical Standa 8MT190 stage:
#     - Supply voltage: 12 ... 24 V DC (controller range)
#     - Motor (2-phase stepper, ~1 A/phase, 0.45 N·m holding)
#     - Travel: 250 mm × 250 mm = ~2 500 000 microsteps @ x40 microstepping
#     - Max linear speed: ~20 mm/s ≈ 400 rpm-equivalent on the leadscrew
#     - Typical power draw: 3 .. 25 W per axis (12-24 V × 0.25-1 A)
# ---------------------------------------------------------------------------

_STANDA_8MT190_X = MotorSpec(
    spec_id="standa_8mt190_x",
    description="Standa 8MT190-250 linear stage, X-axis (8SMC4-USB controller)",
    travel_range_steps=(0.0, 2_500_000.0),
    speed_rpm_range=(0.0, 400.0),
    torque_nm_range=(0.0, 0.45),
    voltage_v_range=(12.0, 24.0),
    power_w_range=(3.0, 25.0),
    notes="Typical 250 mm travel X stage.  Tune travel range vs real stage stroke.",
    tags=("standa", "x", "stage", "stepper"),
)

_STANDA_8MT190_Y = MotorSpec(
    spec_id="standa_8mt190_y",
    description="Standa 8MT190-250 linear stage, Y-axis (8SMC4-USB controller)",
    travel_range_steps=(0.0, 2_500_000.0),
    speed_rpm_range=(0.0, 400.0),
    torque_nm_range=(0.0, 0.45),
    voltage_v_range=(12.0, 24.0),
    power_w_range=(3.0, 25.0),
    notes="Typical 250 mm travel Y stage.  Tune travel range vs real stage stroke.",
    tags=("standa", "y", "stage", "stepper"),
)

# --- Standa focus Z (shorter travel, smaller NEMA 11 equivalent) ---------

_STANDA_FOCUS_Z = MotorSpec(
    spec_id="standa_focus_z",
    description="Standa focus/Z stepper axis (short travel objective positioner)",
    travel_range_steps=(0.0, 500_000.0),
    speed_rpm_range=(0.0, 300.0),
    torque_nm_range=(0.0, 0.15),
    voltage_v_range=(12.0, 24.0),
    power_w_range=(2.0, 15.0),
    notes="Typical 10-50 mm travel Z focus axis.",
    tags=("standa", "z", "focus", "stepper"),
)

# --- Simulated / Mock motors (loose bounds, used for software tests) -----

_SIM_STAGE_X = MotorSpec(
    spec_id="sim_stage_x",
    description="Software-simulated X axis (no real hardware)",
    travel_range_steps=(-1e9, 1e9),
    speed_rpm_range=(0.0, 1_000.0),
    torque_nm_range=(0.0, 1.0),
    voltage_v_range=(5.0, 48.0),
    power_w_range=(0.0, 100.0),
    notes="Permissive simulated bounds. Actual values are model-dependent.",
    tags=("simulated", "x", "stage"),
)

_SIM_STAGE_Y = MotorSpec(
    spec_id="sim_stage_y",
    description="Software-simulated Y axis (no real hardware)",
    travel_range_steps=(-1e9, 1e9),
    speed_rpm_range=(0.0, 1_000.0),
    torque_nm_range=(0.0, 1.0),
    voltage_v_range=(5.0, 48.0),
    power_w_range=(0.0, 100.0),
    notes="Permissive simulated bounds. Actual values are model-dependent.",
    tags=("simulated", "y", "stage"),
)

_SIM_FOCUS_Z = MotorSpec(
    spec_id="sim_focus_z",
    description="Software-simulated Z/focus axis (no real hardware)",
    travel_range_steps=(-1e9, 1e9),
    speed_rpm_range=(0.0, 1_000.0),
    torque_nm_range=(0.0, 1.0),
    voltage_v_range=(5.0, 48.0),
    power_w_range=(0.0, 100.0),
    notes="Permissive simulated bounds. Actual values are model-dependent.",
    tags=("simulated", "z", "focus"),
)

_MOCK_STAGE_X = MotorSpec(
    spec_id="mock_stage_x",
    description="Mock X axis for unit testing (bounds mirror MockStageXY position range)",
    travel_range_steps=(0.0, 100_000.0),
    speed_rpm_range=(0.0, 500.0),
    torque_nm_range=(0.0, 0.5),
    voltage_v_range=(10.0, 30.0),
    power_w_range=(1.0, 50.0),
    notes="Mock/test axis.",
    tags=("mock", "x", "stage"),
)

_MOCK_STAGE_Y = MotorSpec(
    spec_id="mock_stage_y",
    description="Mock Y axis for unit testing (bounds mirror MockStageXY position range)",
    travel_range_steps=(0.0, 100_000.0),
    speed_rpm_range=(0.0, 500.0),
    torque_nm_range=(0.0, 0.5),
    voltage_v_range=(10.0, 30.0),
    power_w_range=(1.0, 50.0),
    notes="Mock/test axis.",
    tags=("mock", "y", "stage"),
)

_MOCK_FOCUS_Z = MotorSpec(
    spec_id="mock_focus_z",
    description="Mock Z/focus axis for unit testing",
    travel_range_steps=(0.0, 5_000.0),
    speed_rpm_range=(0.0, 500.0),
    torque_nm_range=(0.0, 0.3),
    voltage_v_range=(10.0, 30.0),
    power_w_range=(1.0, 30.0),
    notes="Mock/test axis.",
    tags=("mock", "z", "focus"),
)


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------


CATALOGUE: Dict[str, MotorSpec] = {
    s.spec_id: s
    for s in (
        _STANDA_8MT190_X,
        _STANDA_8MT190_Y,
        _STANDA_FOCUS_Z,
        _SIM_STAGE_X,
        _SIM_STAGE_Y,
        _SIM_FOCUS_Z,
        _MOCK_STAGE_X,
        _MOCK_STAGE_Y,
        _MOCK_FOCUS_Z,
    )
}


# Map (device_type, axis_role) -> spec_id used by the compliance checker to
# auto-assign a spec when the JSON config doesn't declare ``motor_spec_id``.
DEFAULT_DEVICE_TYPE_TO_SPEC: Dict[Tuple[str, str], str] = {
    ("simulated", "stage_x"): "sim_stage_x",
    ("simulated", "stage_y"): "sim_stage_y",
    ("simulated", "focus_z"): "sim_focus_z",
    ("StandaStageXY", "stage_x"): "standa_8mt190_x",
    ("StandaStageXY", "stage_y"): "standa_8mt190_y",
    ("mock", "stage_x"): "mock_stage_x",
    ("mock", "stage_y"): "mock_stage_y",
    ("mock", "focus_z"): "mock_focus_z",
}


def get_spec(spec_id: str) -> Optional[MotorSpec]:
    """Return a spec by id, or None if unknown."""
    return CATALOGUE.get(spec_id)


def default_spec_for(device_type: str, axis_role: str) -> Optional[MotorSpec]:
    """Look up the default spec to use for a (device_type, axis_role) pair.

    ``axis_role`` is one of ``stage_x``, ``stage_y``, ``focus_z``.
    """
    sid = DEFAULT_DEVICE_TYPE_TO_SPEC.get((str(device_type or ""), str(axis_role or "")))
    if sid:
        return CATALOGUE.get(sid)
    return None
