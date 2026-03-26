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
            return f"Detector axis (scales={p['scales']})"
        if t == "Round":
            return f"Rounds: {p['n_rounds']}"
        return t


# ---------------------------------------------------------
#  Axis base class
# ---------------------------------------------------------

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
    def __init__(self, stage: StageXY, start: float, end: float, step: float, motor_devices: list | None = None, motor_mode: str = "sequential", sync_timeout: float = 5.0, sync_poll: float = 0.01, sync_tol: float = 1e-3):
        self.stage = stage
        self.start = start
        self.end = end
        self.step = step
        # motor_devices: optional list of motor objects (e.g., StageXY or SingleAxis)
        self.motor_devices = motor_devices or [stage]
        self.motor_mode = motor_mode
        self.sync_timeout = float(sync_timeout)
        self.sync_poll = float(sync_poll)
        self.sync_tol = float(sync_tol)

    def name(self) -> str:
        return "X"

    def prepare(self) -> None:
        pass

    def positions(self):
        x = self.start
        while x <= self.end + 1e-9:
            yield x
            x += self.step

    def apply(self, pos: float) -> None:
        # if multiple motor devices provided, handle sequential or synchronized moves
        if not self.motor_devices:
            x, y = self.stage.get_position()
            self.stage.move_to(pos, y)
            return

        # build move targets per device: list of (dev, target)
        targets = []
        for dev in self.motor_devices:
            try:
                if hasattr(dev, "get_position"):
                    cur = dev.get_position()
                    if isinstance(cur, tuple):
                        # StageXY: move X coordinate
                        target = (pos, cur[1])
                        dev.move_to(target[0], target[1])
                    else:
                        # SingleAxis-like
                        target = pos
                        dev.move_to(pos)
                else:
                    target = pos
                    dev.move_to(pos)
                targets.append((dev, target))
            except Exception:
                # ignore device move failures here
                continue

        if self.motor_mode == "synchronized":
            # wait for all devices to reach their targets
            _wait_for_targets(targets, timeout=self.sync_timeout, poll=self.sync_poll, tol=self.sync_tol)


class YAxis(Axis):
    def __init__(self, stage: StageXY, start: float, end: float, step: float, motor_devices: list | None = None, motor_mode: str = "sequential", sync_timeout: float = 5.0, sync_poll: float = 0.01, sync_tol: float = 1e-3):
        self.stage = stage
        self.start = start
        self.end = end
        self.step = step
        self.motor_devices = motor_devices or [stage]
        self.motor_mode = motor_mode
        self.sync_timeout = float(sync_timeout)
        self.sync_poll = float(sync_poll)
        self.sync_tol = float(sync_tol)

    def name(self) -> str:
        return "Y"

    def prepare(self) -> None:
        pass

    def positions(self):
        y = self.start
        while y <= self.end + 1e-9:
            yield y
            y += self.step

    def apply(self, pos: float) -> None:
        if not self.motor_devices:
            x, y = self.stage.get_position()
            self.stage.move_to(x, pos)
            return

        targets = []
        for dev in self.motor_devices:
            try:
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
                targets.append((dev, target))
            except Exception:
                continue

        if self.motor_mode == "synchronized":
            _wait_for_targets(targets, timeout=self.sync_timeout, poll=self.sync_poll, tol=self.sync_tol)


class ZAxis(Axis):
    def __init__(self, focus: FocusZ, start: float, end: float, step: float, motor_devices: list | None = None, motor_mode: str = "sequential", sync_timeout: float = 5.0, sync_poll: float = 0.01, sync_tol: float = 1e-3):
        self.focus = focus
        self.start = start
        self.end = end
        self.step = step
        self.motor_devices = motor_devices or [focus]
        self.motor_mode = motor_mode
        self.sync_timeout = float(sync_timeout)
        self.sync_poll = float(sync_poll)
        self.sync_tol = float(sync_tol)

    def name(self) -> str:
        return "Z"

    def prepare(self) -> None:
        pass

    def positions(self):
        z = self.start
        while z <= self.end + 1e-9:
            yield z
            z += self.step

    def apply(self, pos: float) -> None:
        if not self.motor_devices:
            self.focus.move_to(pos)
            return

        targets = []
        for dev in self.motor_devices:
            try:
                dev.move_to(pos)
                targets.append((dev, pos))
            except Exception:
                continue

        if self.motor_mode == "synchronized":
            _wait_for_targets(targets, timeout=self.sync_timeout, poll=self.sync_poll, tol=self.sync_tol)


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
    def __init__(self, detector: Detector, scales: List[tuple[float, float]], waits: List[tuple[float,float]]):
        self.detector = detector
        self.scales = scales
        self.waits = waits

    def name(self) -> str:
        return "Detector"

    def prepare(self) -> None:
        pass

    def positions(self):
        for s in self.scales:
            yield s

    def apply(self, pos: tuple[float, float]) -> None:
        scale, offset = pos
        # support single detector or list of detectors
        try:
            if isinstance(self.detector, list):
                for d in self.detector:
                    try:
                        d.set_scale(scale, offset)
                        # if self.waits>0:
                        #     time.sleep(self.waits)
                    except Exception:
                        continue
            else:
                self.detector.set_scale(scale, offset)
        except Exception:
            # best-effort: ignore if detector doesn't support set_scale
            return


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
        # optional callback called when an axis move completes with the current state: on_move(state: dict)
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
            print(f"axis:{axis_idx} is recursing!")
            axis.apply(pos)
            state[axis.name()] = pos
            # notify interested listeners that a move completed and provide a snapshot of the state
            try:
                if callable(self.on_move):
                    # provide a shallow copy to avoid accidental mutation by callers
                    self.on_move(state.copy())
            except Exception:
                pass
            print(state)
            self._recurse_axis(axis_idx + 1, state)


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
