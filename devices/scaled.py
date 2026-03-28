from __future__ import annotations

from typing import Any, Tuple


class ScaledStageXY:
    """Wrap a StageXY-like device and apply linear scaling/offset.

    Logical units (used by GUI/scan axes) are converted to underlying hardware
    units via:

        raw = logical * scale + offset

    This is useful when the hardware uses steps but the user wants microns.
    """

    def __init__(
        self,
        stage: Any,
        x_scale: float = 1.0,
        x_offset: float = 0.0,
        y_scale: float = 1.0,
        y_offset: float = 0.0,
    ):
        self._stage = stage
        self.x_scale = float(x_scale)
        self.x_offset = float(x_offset)
        self.y_scale = float(y_scale)
        self.y_offset = float(y_offset)

        # mirror a few common attributes if present
        try:
            self.name = getattr(stage, "name")
        except Exception:
            pass

    def move_to(self, x: float, y: float) -> None:
        rx = float(x) * self.x_scale + self.x_offset
        ry = float(y) * self.y_scale + self.y_offset
        self._stage.move_to(rx, ry)

    def get_position(self) -> Tuple[float, float]:
        rx, ry = self._stage.get_position()
        # protect against divide-by-zero
        x = (float(rx) - self.x_offset) / (self.x_scale if self.x_scale != 0 else 1.0)
        y = (float(ry) - self.y_offset) / (self.y_scale if self.y_scale != 0 else 1.0)
        return x, y

    def __getattr__(self, item: str):
        return getattr(self._stage, item)


class ScaledFocusZ:
    """Wrap a FocusZ-like device and apply linear scaling/offset."""

    def __init__(self, focus: Any, scale: float = 1.0, offset: float = 0.0):
        self._focus = focus
        self.scale = float(scale)
        self.offset = float(offset)
        try:
            self.name = getattr(focus, "name")
        except Exception:
            pass

    def move_to(self, z: float) -> None:
        rz = float(z) * self.scale + self.offset
        self._focus.move_to(rz)

    def get_position(self) -> float:
        rz = float(self._focus.get_position())
        return (rz - self.offset) / (self.scale if self.scale != 0 else 1.0)

    def __getattr__(self, item: str):
        return getattr(self._focus, item)


class ScaledLightSource:
    """Wrap a LightSource-like device and apply linear scaling/offset to intensity."""

    def __init__(self, light: Any, scale: float = 1.0, offset: float = 0.0):
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
