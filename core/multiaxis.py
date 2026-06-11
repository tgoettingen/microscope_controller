from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Callable
from abc import ABC, abstractmethod
import time

from devices.base import (
    StageXY,
    FocusZ,
    Camera,
    LightSource,
    FilterWheel,
    Detector,
)
from core.experiment import ChannelConfig


# ---------------------------------------------------------
#  Axis configuration model (used by GUI)
# ---------------------------------------------------------

@dataclass
class AxisConfig:
    axis_type: str
    params: dict

    def label(self) -> str:
        """Human-readable label for GUI list."""
        base = self._label_base()
        p = self.params
        prefix = ""
        if p.get("group_with_prev"):
            mode = p.get("group_mode", "sync")
            length = p.get("group_length", "longer")
            prefix += f"↳ [group {mode}/{length}] "
        if p.get("collapse_one_step"):
            prefix += "⊙ [1 step] "
        return f"{prefix}{base}"

    def _label_base(self) -> str:
        t = self.axis_type
        p = self.params

        if t in ("X", "Y", "Z"):
            motors = p.get("motors")
            mode = p.get("motor_mode")
            motor_info = f" motors={motors}" if motors else ""
            mode_info = f" mode={mode}" if mode else ""
            return f"{t}: {p['start']} → {p['end']} (step {p['step']}){motor_info}{mode_info}"
        if t == "Channel":
            return f"Channel axis ({len(p['channels'])} channels)"
        if t == "Detector":
            det_name = p.get("detector")
            prefix = f"Detector axis [{det_name}]" if det_name else "Detector axis"
            scales = None
            try:
                scales = p.get("scales")
            except Exception:
                scales = None
            if scales:
                return f"{prefix} (config scaling; legacy scales ignored)"
            return f"{prefix} (config scaling)"
        if t == "Round":
            return f"Rounds: {p['n_rounds']}"
        return t


# ---------------------------------------------------------
#  Axis base class
# ---------------------------------------------------------

def _stepped_range(start: float, end: float, step: float):
    """Yield evenly spaced positions from ``start`` to ``end`` (inclusive).

    Supports both ascending (``end >= start``) and descending (``end < start``)
    sweeps. ``step`` is treated as a magnitude; its sign is derived from the
    direction of travel, so a descending range like ``100 -> 0`` works even when
    ``step`` is given as a positive number. A non-positive ``step`` yields just
    the start position to avoid an infinite loop.
    """
    mag = abs(float(step))
    if mag <= 0:
        yield float(start)
        return
    if end >= start:
        x = float(start)
        while x <= end + 1e-9:
            yield x
            x += mag
    else:
        x = float(start)
        while x >= end - 1e-9:
            yield x
            x -= mag


class Axis(ABC):
    """Abstract scan axis: defines a sequence of states and how to apply them."""

    @abstractmethod
    def name(self) -> str:
        ...
    @abstractmethod
    def prepare(self) -> None:
        """Called once before the scan starts."""
        ...

    @abstractmethod
    def positions(self) -> Iterable[Any]:
        """Yield all positions for this axis."""
        ...

    @abstractmethod
    def apply(self, pos: Any) -> None:
        """Apply the given position (move stage, set channel, etc.)."""
        ...


# ---------------------------------------------------------
#  Motor axes (X, Y, Z)
# ---------------------------------------------------------

class XAxis(Axis):
    def __init__(
        self,
        stage: StageXY,
        start: float,
        end: float,
        step: float,
        motor_devices: list | None = None,
        motor_mode: str = "sequential",
        motor_modes: list | None = None,
        wait_s: float = 0.0,
        sync_timeout: float = 5.0,
        sync_poll: float = 0.01,
        sync_tol: float = 1e-3,
    ):
        self.stage = stage
        self.start = start
        self.end = end
        self.step = step
        # motor_devices: optional list of motor objects (e.g., StageXY or SingleAxis)
        self.motor_devices = motor_devices or [stage]
        self.motor_mode = motor_mode
        # motor_modes: optional per-device list of "synchronized"/"sequential",
        # aligned with motor_devices. When None, every device uses motor_mode.
        self.motor_modes = motor_modes
        self.wait_s = float(wait_s)
        self.sync_timeout = float(sync_timeout)
        self.sync_poll = float(sync_poll)
        self.sync_tol = float(sync_tol)

    def name(self) -> str:
        return "X"

    def prepare(self) -> None:
        pass

    def positions(self):
        yield from _stepped_range(self.start, self.end, self.step)

    def _move_one(self, dev, pos: float):
        """Issue the X move for a single device and return (dev, target)."""
        if hasattr(dev, "get_position"):
            cur = dev.get_position()
            if isinstance(cur, tuple):
                # StageXY: move X coordinate, keep Y.
                target = (pos, cur[1])
                dev.move_to(target[0], target[1])
            else:
                # SingleAxis-like
                target = pos
                dev.move_to(pos)
        else:
            target = pos
            dev.move_to(pos)
        return dev, target

    def apply(self, pos: float) -> None:
        # if multiple motor devices provided, handle sequential or synchronized moves
        if not self.motor_devices:
            x, y = self.stage.get_position()
            self.stage.move_to(pos, y)
            return

        _apply_motor_moves(
            self.motor_devices, self.motor_modes, self.motor_mode, self._move_one, pos,
            timeout=self.sync_timeout, poll=self.sync_poll, tol=self.sync_tol,
        )

        if self.wait_s > 0:
            time.sleep(self.wait_s)


class YAxis(Axis):
    def __init__(
        self,
        stage: StageXY,
        start: float,
        end: float,
        step: float,
        motor_devices: list | None = None,
        motor_mode: str = "sequential",
        motor_modes: list | None = None,
        wait_s: float = 0.0,
        sync_timeout: float = 5.0,
        sync_poll: float = 0.01,
        sync_tol: float = 1e-3,
    ):
        self.stage = stage
        self.start = start
        self.end = end
        self.step = step
        self.motor_devices = motor_devices or [stage]
        self.motor_mode = motor_mode
        # Optional per-device modes aligned with motor_devices (see XAxis).
        self.motor_modes = motor_modes
        self.wait_s = float(wait_s)
        self.sync_timeout = float(sync_timeout)
        self.sync_poll = float(sync_poll)
        self.sync_tol = float(sync_tol)

    def name(self) -> str:
        return "Y"

    def prepare(self) -> None:
        pass

    def positions(self):
        yield from _stepped_range(self.start, self.end, self.step)

    def _move_one(self, dev, pos: float):
        """Issue the Y move for a single device and return (dev, target)."""
        if hasattr(dev, "get_position"):
            cur = dev.get_position()
            if isinstance(cur, tuple):
                target = (cur[0], pos)
                dev.move_to(target[0], target[1])
            else:
                target = pos
                dev.move_to(pos)
        else:
            target = pos
            dev.move_to(pos)
        return dev, target

    def apply(self, pos: float) -> None:
        if not self.motor_devices:
            x, y = self.stage.get_position()
            self.stage.move_to(x, pos)
            return

        _apply_motor_moves(
            self.motor_devices, self.motor_modes, self.motor_mode, self._move_one, pos,
            timeout=self.sync_timeout, poll=self.sync_poll, tol=self.sync_tol,
        )

        if self.wait_s > 0:
            time.sleep(self.wait_s)


class ZAxis(Axis):
    def __init__(
        self,
        focus: FocusZ,
        start: float,
        end: float,
        step: float,
        motor_devices: list | None = None,
        motor_mode: str = "sequential",
        motor_modes: list | None = None,
        wait_s: float = 0.0,
        sync_timeout: float = 5.0,
        sync_poll: float = 0.01,
        sync_tol: float = 1e-3,
    ):
        self.focus = focus
        self.start = start
        self.end = end
        self.step = step
        self.motor_devices = motor_devices or [focus]
        self.motor_mode = motor_mode
        # Optional per-device modes aligned with motor_devices (see XAxis).
        self.motor_modes = motor_modes
        self.wait_s = float(wait_s)
        self.sync_timeout = float(sync_timeout)
        self.sync_poll = float(sync_poll)
        self.sync_tol = float(sync_tol)

    def name(self) -> str:
        return "Z"

    def prepare(self) -> None:
        pass

    def positions(self):
        yield from _stepped_range(self.start, self.end, self.step)

    def _move_one(self, dev, pos: float):
        """Issue the Z move for a single device and return (dev, target)."""
        dev.move_to(pos)
        return dev, pos

    def apply(self, pos: float) -> None:
        if not self.motor_devices:
            self.focus.move_to(pos)
            return

        _apply_motor_moves(
            self.motor_devices, self.motor_modes, self.motor_mode, self._move_one, pos,
            timeout=self.sync_timeout, poll=self.sync_poll, tol=self.sync_tol,
        )

        if self.wait_s > 0:
            time.sleep(self.wait_s)


# ---------------------------------------------------------
#  Channel axis (filter wheel + illumination + exposure)
# ---------------------------------------------------------

class ChannelAxis(Axis):
    def __init__(
        self,
        camera: Camera,
        light: LightSource,
        fw: FilterWheel,
        channels: List[ChannelConfig],
        wait_s: float = 0.0,
    ):
        self.camera = camera
        self.light = light
        self.fw = fw
        self.channels = channels
        self.wait_s = wait_s

    def name(self) -> str:
        return "Channel"

    def prepare(self) -> None:
        pass

    def positions(self):
        for ch in self.channels:
            yield ch

    def apply(self, pos: ChannelConfig) -> None:
        self.fw.set_position(pos.filter_position)
        self.light.set_intensity(pos.light_intensity)
        self.camera.set_exposure(pos.exposure_ms)
        if self.wait_s > 0:
            time.sleep(self.wait_s)


# ---------------------------------------------------------
#  Detector axis (photodiode, PMT, voltage reader)
# ---------------------------------------------------------

class DetectorAxis(Axis):
    def __init__(self, detector: Detector, scales: List[tuple[float, float]] | None = None, wait_s: float = 0.0):
        self.detector = detector
        self.scales = list(scales) if scales else []
        self.wait_s = float(wait_s)

    def name(self) -> str:
        return "Detector"

    def prepare(self) -> None:
        pass

    def positions(self):
        # Scaling is defined in the device config JSON; do not override it here.
        # Keep this axis as a single no-op step for backward compatibility.
        if not self.scales:
            yield None
            return
        for s in self.scales:
            yield s

    def apply(self, pos: tuple[float, float] | None) -> None:
        # No-op: do not call set_scale(); scaling comes from config.
        if self.wait_s > 0:
            time.sleep(self.wait_s)


# ---------------------------------------------------------
#  Round axis (software axis for repeated scans)
# ---------------------------------------------------------

class RoundAxis(Axis):
    def __init__(self, n_rounds: Optional[int]):
        self.n_rounds = n_rounds

    def name(self) -> str:
        return "Round"

    def prepare(self) -> None:
        pass

    def positions(self):
        if self.n_rounds is None:
            i = 0
            while True:
                yield i
                i += 1
        else:
            for i in range(self.n_rounds):
                yield i

    def apply(self, pos: int) -> None:
        pass


# ---------------------------------------------------------
#  Grouped axis (composite: combine several axes into one dimension)
# ---------------------------------------------------------

class GroupedAxis(Axis):
    """Combine several consecutive axes into a single scan dimension.

    Instead of being nested (the default cartesian-product behaviour), the
    member axes share one loop:

    - ``mode="sync"``: all members advance together; combined position *i*
      applies the *i*-th position of every member.
    - ``mode="sequential"``: members run one after another (concatenated);
      each combined tick moves a single member while the others hold their
      most recent position.

    ``length`` only affects ``"sync"`` runs where members differ in length:

    - ``"shorter"``: stop at the shortest member (``min`` number of steps).
    - ``"longer"``: continue to the longest member; exhausted members hold
      their last position.
    """

    def __init__(self, members: List["Axis"], mode: str = "sync", length: str = "longer"):
        self.members = list(members)
        self.mode = mode if mode in ("sync", "sequential") else "sync"
        self.length = length if length in ("shorter", "longer") else "longer"
        self._member_positions: List[list] = []
        # When every member is itself collapsed into a single scan step (e.g. a
        # back-and-forth made of two collapsed X sweeps), the whole group is one
        # scan step too: all member sweeps run inside a single tick, yielding a
        # single measurement instead of one per member.
        self._all_collapsed = bool(self.members) and all(
            getattr(m, "collapsed_single_step", False) for m in self.members
        )

    def name(self) -> str:
        return " + ".join(m.name() for m in self.members)

    def prepare(self) -> None:
        for m in self.members:
            m.prepare()
        # Materialise each member's positions once so lengths are known.
        self._member_positions = [list(m.positions()) for m in self.members]

    def positions(self):
        plists = self._member_positions
        if not plists:
            return
        if self._all_collapsed:
            # The entire group is a single scan step; apply() drives every
            # member's full sweep in order.
            yield ("collapsed_group",)
            return
        if self.mode == "sync":
            lengths = [len(p) for p in plists if p]
            if not lengths:
                return
            n = min(lengths) if self.length == "shorter" else max(lengths)
            for i in range(n):
                combo = []
                for p in plists:
                    if not p:
                        combo.append(None)
                    elif i < len(p):
                        combo.append(p[i])
                    else:
                        combo.append(p[-1])  # hold last position
                yield ("sync", combo)
        else:  # sequential
            for mi, p in enumerate(plists):
                for val in p:
                    yield ("seq", mi, val)

    def apply(self, pos) -> None:
        if not pos:
            return
        if pos[0] == "collapsed_group":
            # Drive each member's full (already-collapsed) sweep in sequence.
            for m, steps in zip(self.members, self._member_positions):
                for step in steps:
                    m.apply(step)
        elif pos[0] == "sync":
            for m, v in zip(self.members, pos[1]):
                if v is not None:
                    m.apply(v)
        else:  # ("seq", member_index, value)
            _, mi, val = pos
            self.members[mi].apply(val)

    def state_updates(self, pos) -> dict:
        """Map a combined position onto ``{member_name: member_pos}``.

        Lets the runner populate the scan state with each member's own axis
        name so downstream consumers (plotting, saving, Channel/Detector
        detection) keep working as if the axes were not grouped.

        If a member is itself a composite axis (e.g. an ``OneStepAxis`` wrapping
        a collapsed sweep), delegate to its ``state_updates`` so the reported
        value is a clean position rather than an internal sentinel tuple.
        """
        out: dict = {}
        if not pos:
            return out

        def _member_update(m, v):
            su = getattr(m, "state_updates", None)
            if callable(su):
                try:
                    sub = su(v)
                    if sub:
                        out.update(sub)
                        return
                except Exception:
                    pass
            out[m.name()] = v

        if pos[0] == "collapsed_group":
            for m in self.members:
                _member_update(m, None)
        elif pos[0] == "sync":
            for m, v in zip(self.members, pos[1]):
                if v is not None:
                    _member_update(m, v)
        else:
            _, mi, val = pos
            _member_update(self.members[mi], val)
        return out


class OneStepAxis(Axis):
    """Collapse a member axis into a single scan step.

    The wrapped axis is no longer a scan dimension: instead of producing one
    step per position, it yields a single combined step. When that step is
    applied, *all* of the member's positions are driven in sequence (so the
    hardware still performs the full sweep), but the multi-axis scan only sees
    one step for this dimension.
    """

    def __init__(self, member: "Axis"):
        self.member = member
        self._positions: list = []
        self._last = None

    # Marker used by GroupedAxis: when every member of a group carries this,
    # the whole group collapses into a single combined scan step.
    collapsed_single_step = True

    def name(self) -> str:
        return self.member.name()

    def prepare(self) -> None:
        self.member.prepare()
        self._positions = list(self.member.positions())

    def positions(self):
        # A single step represents the whole member sweep.
        yield ("onestep", self._positions)

    def apply(self, pos) -> None:
        if not pos:
            return
        _, plist = pos
        for p in plist:
            self.member.apply(p)
            self._last = p

    def state_updates(self, pos) -> dict:
        # Report the member's last position so downstream consumers see a
        # representative value rather than the internal sentinel tuple.
        return {self.member.name(): self._last}


# ---------------------------------------------------------
#  Multi‑axis experiment + runner
# ---------------------------------------------------------

@dataclass
class MultiAxisExperiment:
    axes: List[Axis]
    measure: Callable[[Dict[str, Any]], Any]


class MultiAxisRunner:
    """Generic N-dimensional scan engine."""

    def __init__(self, experiment: MultiAxisExperiment, on_move: callable | None = None):
        self.exp = experiment
        self._running = False
        # optional callback called when an axis move completes:
        #   on_move(axis_name: str, pos: Any, state: dict)
        self.on_move = on_move

    def stop(self):
        self._running = False

    def run(self):
        self._running = True

        for axis in self.exp.axes:
            axis.prepare()

        state: Dict[str, Any] = {}
        self._recurse_axis(0, state)

    def _recurse_axis(self, axis_idx: int, state: Dict[str, Any]):
        if not self._running:
            return

        if axis_idx >= len(self.exp.axes):
            self.exp.measure(state)
            return

        axis = self.exp.axes[axis_idx]
        for pos in axis.positions():
            if not self._running:
                break
            axis.apply(pos)
            # Composite axes (e.g. GroupedAxis) may map a single combined
            # position onto several member axes; let them populate the state
            # dict with each member's name so downstream consumers (plotting,
            # saving, Channel/Detector detection) keep working unchanged.
            updates = None
            su = getattr(axis, "state_updates", None)
            if callable(su):
                try:
                    updates = su(pos)
                except Exception:
                    updates = None
            if updates:
                state.update(updates)
                try:
                    if callable(self.on_move):
                        for nm, mv in updates.items():
                            self.on_move(nm, mv, state.copy())
                except Exception:
                    pass
            else:
                state[axis.name()] = pos
                # notify interested listeners that a move completed and provide a snapshot of the state
                try:
                    if callable(self.on_move):
                        # provide a shallow copy to avoid accidental mutation by callers
                        self.on_move(axis.name(), pos, state.copy())
                except Exception:
                    pass
            self._recurse_axis(axis_idx + 1, state)


def _apply_motor_moves(
    motor_devices: list,
    motor_modes: list | None,
    default_mode: str,
    move_one: Callable,
    pos: float,
    timeout: float = 5.0,
    poll: float = 0.01,
    tol: float = 1e-3,
):
    """Drive a set of motor devices to ``pos`` honoring per-device modes.

    ``move_one(dev, pos)`` issues the move for a single device and returns a
    ``(dev, target)`` pair used to poll for completion.

    Behavior:
    - When ``motor_modes`` is ``None``, every device uses ``default_mode`` and
      the legacy semantics apply: all devices are moved, then waited on
      together only if ``default_mode == "synchronized"``.
    - When ``motor_modes`` is provided (aligned with ``motor_devices``),
      "sequential" devices are moved and waited on one at a time, while
      "synchronized" devices are moved together and waited on as a group.
    """
    if not motor_devices:
        return

    if motor_modes is None:
        targets = []
        for dev in motor_devices:
            try:
                targets.append(move_one(dev, pos))
            except Exception:
                continue
        if default_mode == "synchronized":
            _wait_for_targets(targets, timeout=timeout, poll=poll, tol=tol)
        return

    sync_targets = []
    for idx, dev in enumerate(motor_devices):
        mode = motor_modes[idx] if idx < len(motor_modes) else default_mode
        try:
            dev_target = move_one(dev, pos)
        except Exception:
            continue
        if mode == "synchronized":
            sync_targets.append(dev_target)
        else:
            # Sequential: wait for this device to finish before the next move.
            _wait_for_targets([dev_target], timeout=timeout, poll=poll, tol=tol)

    if sync_targets:
        _wait_for_targets(sync_targets, timeout=timeout, poll=poll, tol=tol)


def _wait_for_targets(targets: list[tuple], timeout: float = 5.0, poll: float = 0.01, tol: float = 1e-3):
    """Wait until each device reaches its target value.

    targets: list of (device, target) where target is scalar or tuple.
    The function polls device.get_position() if available, otherwise assumes immediate completion.
    """
    if not targets:
        return

    start = time.time()
    remaining = list(targets)

    while remaining and (time.time() - start) < timeout:
        new_remaining = []
        for dev, target in remaining:
            try:
                if not hasattr(dev, "get_position"):
                    # cannot query; assume done
                    continue
                cur = dev.get_position()
                if isinstance(target, tuple) and isinstance(cur, tuple):
                    ok = True
                    for tval, cval in zip(target, cur):
                        if abs(tval - cval) > tol:
                            ok = False
                            break
                    if not ok:
                        new_remaining.append((dev, target))
                else:
                    # scalar compare
                    try:
                        cval = float(cur)
                        tval = float(target)
                        if abs(tval - cval) > tol:
                            new_remaining.append((dev, target))
                    except Exception:
                        # cannot compare, keep waiting
                        new_remaining.append((dev, target))
            except Exception:
                # on error, keep waiting until timeout
                new_remaining.append((dev, target))

        remaining = new_remaining
        if remaining:
            time.sleep(poll)
