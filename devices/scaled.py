from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Dict, Mapping


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers for validating motor operating parameters
# ---------------------------------------------------------------------------


# Keys we recognise in a "motor limits" dict.  Each value is a (min, max)
# tuple where either end can be None.
LIMIT_KEYS = (
    "speed_rpm",
    "torque_nm",
    "voltage_v",
    "power_w",
    "position",  # logical-position travel limit (x_range / y_range / z_range)
)


def _merge_motor_limits(
    base: Mapping | None, override: Mapping | None
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Merge two motor-limits mappings (override wins on a per-key basis).

    Unknown keys are ignored.  Each returned key maps to a (min, max) tuple;
    missing keys default to ``(None, None)``.
    """
    merged: Dict[str, Tuple[Optional[float], Optional[float]]] = {
        k: (None, None) for k in LIMIT_KEYS
    }

    def _apply(m):
        if not isinstance(m, Mapping):
            return
        for k in LIMIT_KEYS:
            v = m.get(k)
            if not isinstance(v, (tuple, list)) or len(v) != 2:
                continue
            lo, hi = v

            def _f(x):
                if x is None:
                    return None
                try:
                    return float(x)
                except Exception:
                    return None

            merged[k] = (_f(lo), _f(hi))

    _apply(base)
    _apply(override)
    return merged


def _check_range(
    value: float,
    rng: Tuple[Optional[float], Optional[float]],
    axis_label: str,
    value_label: str,
) -> None:
    """Raise ``ValueError`` if ``value`` falls outside ``rng``."""
    lo, hi = rng
    if lo is None and hi is None:
        return
    v = float(value)
    if lo is not None and v < lo:
        raise ValueError(
            f"{axis_label} {value_label}={v:.6g} is below the minimum allowed "
            f"limit {lo:.6g}"
        )
    if hi is not None and v > hi:
        raise ValueError(
            f"{axis_label} {value_label}={v:.6g} is above the maximum allowed "
            f"limit {hi:.6g}"
        )


# ---------------------------------------------------------------------------
# ScaledStageXY
# ---------------------------------------------------------------------------


class ScaledStageXY:
    """Wrap a StageXY-like device and apply linear scaling/offset.

    Logical units (used by GUI/scan axes) are converted to underlying hardware
    units via:

        raw = logical * scale + offset

    This is useful when the hardware uses steps but the user wants microns.

    Optional soft limits:

    * ``x_range`` / ``y_range``: ``(min, max)`` on the *logical* coordinate
      (backward-compatible, kept as a convenience).
    * ``motor_limits_x`` / ``motor_limits_y``: merged motor-parameter limits
      dict with keys ``{"speed_rpm","torque_nm","voltage_v","power_w",
      "position"}``, each a ``(min, max)`` tuple (either end may be ``None``).
      ``"position"`` is *merged* with ``x_range``/``y_range`` so both paths
      work.  When ``set_operating_point(axis=..., ...)`` is called the
      electrical parameters are validated against these limits on every
      subsequent ``move_to``.
    """

    def __init__(
        self,
        stage: Any,
        x_scale: float = 1.0,
        x_offset: float = 0.0,
        y_scale: float = 1.0,
        y_offset: float = 0.0,
        x_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        y_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        motor_limits_x: Optional[Mapping] = None,
        motor_limits_y: Optional[Mapping] = None,
    ):
        self._stage = stage
        self.x_scale = float(x_scale)
        self.x_offset = float(x_offset)
        self.y_scale = float(y_scale)
        self.y_offset = float(y_offset)

        def _normalise(r):
            if r is None:
                return (None, None)
            if not isinstance(r, (tuple, list)) or len(r) != 2:
                raise ValueError("range must be a (min, max) tuple or None")
            lo, hi = r

            def _f(v):
                if v is None:
                    return None
                return float(v)

            return (_f(lo), _f(hi))

        # 1. Start with the motor_limits (spec + user operating_limits from
        #    the merged dict) as the *base* configuration.
        # 2. Layer the explicit x_range / y_range on top as *further
        #    tightening* — these always take precedence because they are
        #    either the user's explicit travel-range setting in config or
        #    the calibration-dialog result.
        x_override = {"position": _normalise(x_range)} if x_range is not None else None
        y_override = {"position": _normalise(y_range)} if y_range is not None else None

        self._x_limits = _merge_motor_limits(motor_limits_x, x_override)
        self._y_limits = _merge_motor_limits(motor_limits_y, y_override)

        # Keep the legacy x_range / y_range attributes read by callers.
        self.x_range: Tuple[Optional[float], Optional[float]] = self._x_limits["position"]
        self.y_range: Tuple[Optional[float], Optional[float]] = self._y_limits["position"]

        # Electrical operating point set by the orchestrator (optional).
        self._op_x: Dict[str, Optional[float]] = {
            k: None for k in LIMIT_KEYS if k != "position"
        }
        self._op_y: Dict[str, Optional[float]] = {
            k: None for k in LIMIT_KEYS if k != "position"
        }

        # mirror a few common attributes if present
        try:
            self.name = getattr(stage, "name")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Operating-point helpers (call before move_to to set electrical state)
    # ------------------------------------------------------------------

    def set_operating_point(
        self,
        *,
        axis: str,
        speed_rpm: Optional[float] = None,
        torque_nm: Optional[float] = None,
        voltage_v: Optional[float] = None,
        power_w: Optional[float] = None,
    ) -> None:
        """Record the intended electrical operating point for ``axis``.

        ``axis`` is either ``"x"`` or ``"y"``.

        Subsequent ``move_to`` calls will validate the stored values against
        the motor limits configured for that axis.  Passing ``None`` for a
        parameter clears the previous value.  Raises ``ValueError``
        immediately if the supplied values are already out of range.
        """
        axis = str(axis).lower()
        if axis not in ("x", "y"):
            raise ValueError(f"Unknown axis: {axis!r}, expected 'x' or 'y'")
        limits = self._x_limits if axis == "x" else self._y_limits
        store = self._op_x if axis == "x" else self._op_y
        label = f"Stage {axis.upper()}"
        for key, value in (
            ("speed_rpm", speed_rpm),
            ("torque_nm", torque_nm),
            ("voltage_v", voltage_v),
            ("power_w", power_w),
        ):
            if value is None:
                store[key] = None
                continue
            _check_range(float(value), limits.get(key, (None, None)), label, key)
            store[key] = float(value)

    # ------------------------------------------------------------------
    # Range helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_axis(
        value: float,
        limits: Dict[str, Tuple[Optional[float], Optional[float]]],
        op: Dict[str, Optional[float]],
        axis_name: str,
    ) -> None:
        # Position
        lo, hi = limits.get("position", (None, None))
        v = float(value)
        if lo is not None and v < lo:
            raise ValueError(
                f"Stage {axis_name}={v:.6g} is below minimum allowed limit {lo:.6g}"
            )
        if hi is not None and v > hi:
            raise ValueError(
                f"Stage {axis_name}={v:.6g} is above maximum allowed limit {hi:.6g}"
            )
        # Electrical operating point
        for key in ("speed_rpm", "torque_nm", "voltage_v", "power_w"):
            setting = op.get(key)
            if setting is None:
                continue
            _check_range(
                setting,
                limits.get(key, (None, None)),
                f"Stage {axis_name}",
                key,
            )

    # ------------------------------------------------------------------
    # Motion API
    # ------------------------------------------------------------------

    def move_to(self, x: float, y: float) -> None:
        # Validate logical coordinates against soft limits *before* scaling,
        # and cross-check any electrical operating-point settings.
        # Range limits are in logical coordinates (real units), not steps
        ScaledStageXY._check_axis(x, self._x_limits, self._op_x, "X")
        ScaledStageXY._check_axis(y, self._y_limits, self._op_y, "Y")

        rx = float(x) * self.x_scale + self.x_offset
        ry = float(y) * self.y_scale + self.y_offset
        try:
            logger.info(
                "Stage move_to (scaled) logical=(%s,%s) raw=(%s,%s)",
                x,
                y,
                rx,
                ry,
            )
        except Exception:
            pass
        self._stage.move_to(rx, ry)

    def get_position(self) -> Tuple[float, float]:
        rx, ry = self._stage.get_position()
        # protect against divide-by-zero
        x = (float(rx) - self.x_offset) / (
            self.x_scale if self.x_scale != 0 else 1.0
        )
        y = (float(ry) - self.y_offset) / (
            self.y_scale if self.y_scale != 0 else 1.0
        )
        return x, y

    # ------------------------------------------------------------------
    # Capabilities (merge range info with wrapped stage)
    # ------------------------------------------------------------------

    def get_capabilities(self) -> Dict[str, Any]:
        caps: Dict[str, Any] = {}
        try:
            fn = getattr(self._stage, "get_capabilities")
            if callable(fn):
                got = fn()
                if isinstance(got, dict):
                    caps.update(got)
        except Exception:
            pass
        rng = caps.get("range")
        if not isinstance(rng, dict):
            rng = {}
        if self.x_range != (None, None):
            rng["x"] = (self.x_range[0], self.x_range[1])
        if self.y_range != (None, None):
            rng["y"] = (self.y_range[0], self.y_range[1])
        if rng:
            caps["range"] = rng
        # Motor limits
        def _lim_dict(lims: Dict) -> Dict:
            out = {}
            for k, (lo, hi) in lims.items():
                if (lo, hi) != (None, None):
                    out[k] = (lo, hi)
            return out

        mx = _lim_dict(self._x_limits)
        my = _lim_dict(self._y_limits)
        if mx or my:
            caps["motor_limits"] = {"x": mx, "y": my}
        return caps

    def __getattr__(self, item: str):
        return getattr(self._stage, item)


# ---------------------------------------------------------------------------
# ScaledFocusZ
# ---------------------------------------------------------------------------


class ScaledFocusZ:
    """Wrap a FocusZ-like device and apply linear scaling/offset.

    Supports the same motor-limits pattern as :class:`ScaledStageXY` via a
    ``motor_limits`` mapping and ``set_operating_point()``.
    """

    def __init__(
        self,
        focus: Any,
        scale: float = 1.0,
        offset: float = 0.0,
        z_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
        motor_limits: Optional[Mapping] = None,
    ):
        self._focus = focus
        self.scale = float(scale)
        self.offset = float(offset)

        def _normalise(r):
            if r is None:
                return (None, None)
            if not isinstance(r, (tuple, list)) or len(r) != 2:
                raise ValueError("z_range must be a (min, max) tuple or None")
            lo, hi = r

            def _f(v):
                if v is None:
                    return None
                return float(v)

            return (_f(lo), _f(hi))

        base = {"position": _normalise(z_range)} if z_range is not None else None
        self._limits = _merge_motor_limits(base, motor_limits)
        self.z_range: Tuple[Optional[float], Optional[float]] = self._limits["position"]

        self._op: Dict[str, Optional[float]] = {
            k: None for k in LIMIT_KEYS if k != "position"
        }

        try:
            self.name = getattr(focus, "name")
        except Exception:
            pass

    # ------------------------------------------------------------------
    def set_operating_point(
        self,
        *,
        speed_rpm: Optional[float] = None,
        torque_nm: Optional[float] = None,
        voltage_v: Optional[float] = None,
        power_w: Optional[float] = None,
    ) -> None:
        label = "Focus Z"
        for key, value in (
            ("speed_rpm", speed_rpm),
            ("torque_nm", torque_nm),
            ("voltage_v", voltage_v),
            ("power_w", power_w),
        ):
            if value is None:
                self._op[key] = None
                continue
            _check_range(
                float(value),
                self._limits.get(key, (None, None)),
                label,
                key,
            )
            self._op[key] = float(value)

    # ------------------------------------------------------------------
    @staticmethod
    def _check(
        value: float,
        limits: Dict[str, Tuple[Optional[float], Optional[float]]],
        op: Dict[str, Optional[float]],
    ) -> None:
        lo, hi = limits.get("position", (None, None))
        v = float(value)
        if lo is not None and v < lo:
            raise ValueError(
                f"Focus Z={v:.6g} is below minimum allowed limit {lo:.6g}"
            )
        if hi is not None and v > hi:
            raise ValueError(
                f"Focus Z={v:.6g} is above maximum allowed limit {hi:.6g}"
            )
        for key in ("speed_rpm", "torque_nm", "voltage_v", "power_w"):
            setting = op.get(key)
            if setting is None:
                continue
            _check_range(setting, limits.get(key, (None, None)), "Focus Z", key)

    def move_to(self, z: float) -> None:
        ScaledFocusZ._check(z, self._limits, self._op)
        rz = float(z) * self.scale + self.offset
        try:
            logger.info("Focus move_to (scaled) logical=%s raw=%s", z, rz)
        except Exception:
            pass
        self._focus.move_to(rz)

    def get_position(self) -> float:
        rz = float(self._focus.get_position())
        return (rz - self.offset) / (self.scale if self.scale != 0 else 1.0)

    # ------------------------------------------------------------------
    def get_capabilities(self) -> Dict[str, Any]:
        caps: Dict[str, Any] = {}
        try:
            fn = getattr(self._focus, "get_capabilities")
            if callable(fn):
                got = fn()
                if isinstance(got, dict):
                    caps.update(got)
        except Exception:
            pass
        if self.z_range != (None, None):
            caps.setdefault("range", {})
            caps["range"]["z"] = (self.z_range[0], self.z_range[1])
        out = {}
        for k, (lo, hi) in self._limits.items():
            if (lo, hi) != (None, None):
                out[k] = (lo, hi)
        if out:
            caps["motor_limits"] = {"z": out}
        return caps

    def __getattr__(self, item: str):
        return getattr(self._focus, item)


# ---------------------------------------------------------------------------
# ScaledLightSource
# ---------------------------------------------------------------------------


class ScaledLightSource:
    """Wrap a LightSource-like device and apply linear scaling/offset to intensity."""

    def __init__(
        self,
        light: Any,
        scale: float = 1.0,
        offset: float = 0.0,
    ):
        self._light = light
        self.scale = float(scale)
        self.offset = float(offset)
        try:
            self.name = getattr(light, "name")
        except Exception:
            pass

    def set_intensity(self, percent: float) -> None:
        raw = float(percent) * self.scale + self.offset
        self._light.set_intensity(raw)

    def __getattr__(self, item: str):
        return getattr(self._light, item)
