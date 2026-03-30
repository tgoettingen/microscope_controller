import time
import math
from typing import Dict, Any, Tuple

import logging

import numpy as np

from .base import Camera, StageXY, FocusZ, LightSource, FilterWheel, Detector


logger = logging.getLogger(__name__)


class MockCamera(Camera):
    def __init__(self, name: str = "MockCamera"):
        super().__init__(name)
        self.exposure_ms = 50.0
        self.trigger_mode = "internal"

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {"type": "mock_camera", "trigger_modes": ["internal", "software", "hardware"]}

    def reset(self) -> None:
        self.exposure_ms = 50.0
        self.trigger_mode = "internal"

    def set_exposure(self, ms: float) -> None:
        self.exposure_ms = ms

    def set_trigger_mode(self, mode: str) -> None:
        self.trigger_mode = mode

    def snap(self) -> Any:
        time.sleep(self.exposure_ms / 1000.0)
        x = np.linspace(0, 65535, 512, dtype=np.uint16)
        img = np.tile(x, (512, 1))
        return img

    def start_live(self) -> None:
        pass

    def stop_live(self) -> None:
        pass


class MockStageXY(StageXY):
    def __init__(self, name: str = "MockStageXY"):
        super().__init__(name)
        self._x = 0.0
        self._y = 0.0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {"type": "mock_stage_xy", "range": {"x": (0, 100000), "y": (0, 100000)}}

    def reset(self) -> None:
        self._x, self._y = 0.0, 0.0

    def move_to(self, x: float, y: float) -> None:
        try:
            logger.info("Stage move_to (mock) x=%s y=%s", x, y)
        except Exception:
            pass
        self._x, self._y = x, y
        time.sleep(0.05)

    def get_position(self) -> Tuple[float, float]:
        return self._x, self._y


class MockFocusZ(FocusZ):
    def __init__(self, name: str = "MockFocusZ"):
        super().__init__(name)
        self._z = 100.0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {"type": "mock_focus_z", "range": (0, 5000)}

    def reset(self) -> None:
        self._z = 100.0

    def move_to(self, z: float) -> None:
        try:
            logger.info("Focus move_to (mock) z=%s", z)
        except Exception:
            pass
        self._z = z
        time.sleep(0.02)

    def get_position(self) -> float:
        return self._z


class MockLightSource(LightSource):
    def __init__(self, name: str = "MockLight"):
        super().__init__(name)
        self._intensity = 0.0
        self._on = False

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {"type": "mock_light", "intensity_range": (0, 100)}

    def reset(self) -> None:
        self._intensity = 0.0
        self._on = False

    def set_intensity(self, percent: float) -> None:
        self._intensity = max(0.0, min(100.0, percent))

    def on(self) -> None:
        self._on = True

    def off(self) -> None:
        self._on = False


class MockFilterWheel(FilterWheel):
    def __init__(self, name: str = "MockFilterWheel", positions: int = 6):
        super().__init__(name)
        self._pos = 0
        self._n = positions

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {"type": "mock_filter_wheel", "positions": self._n}

    def reset(self) -> None:
        self._pos = 0

    def set_position(self, index: int) -> None:
        if not (0 <= index < self._n):
            raise ValueError("Invalid filter position")
        self._pos = index
        time.sleep(0.05)

    def get_position(self) -> int:
        return self._pos


class MockDetector(Detector):
    """Simulated photodiode-like detector producing a noisy sine wave."""

    def __init__(self, name: str = "MockDetector"):
        super().__init__(name)
        self._scale = 1.0
        self._offset = 0.0
        self._t0 = time.time()

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_capabilities(self) -> Dict[str, Any]:
        return {"type": "mock_detector", "units": "V"}

    def reset(self) -> None:
        self._scale = 1.0
        self._offset = 0.0
        self._t0 = time.time()

    def set_scale(self, scale: float, offset: float = 0.0) -> None:
        self._scale = scale
        self._offset = offset

    def read_value(self) -> float:
        t = time.time() - self._t0
        raw = math.sin(2 * math.pi * 0.5 * t) + 0.1 * np.random.randn()
        return self._scale * raw + self._offset