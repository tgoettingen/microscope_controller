import time
import numpy as np


def _caps(kind: str, **extra):
    c = {"kind": kind, "simulated": True}
    c.update(extra)
    return c


class SimulatedStageXY:
    """Simple XY stage simulation with instant movement."""

    def __init__(self):
        self.connected = False
        self.x = 0.0
        self.y = 0.0

    def connect(self):
        self.connected = True

    def reset(self):
        # leave position as-is; a real stage would home here
        pass

    def get_capabilities(self):
        return _caps("stage_xy")

    def move_to(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def move_by(self, dx: float, dy: float):
        self.x += float(dx)
        self.y += float(dy)

    def get_position(self):
        return self.x, self.y

    def stop(self):
        pass

    def disconnect(self):
        self.connected = False


class SimulatedFocus:
    """Simulated Z axis."""

    def __init__(self):
        self.connected = False
        self.z = 100.0

    def connect(self):
        self.connected = True

    def reset(self):
        pass

    def get_capabilities(self):
        return _caps("focus_z")

    def move_to(self, z: float):
        self.z = float(z)

    def move_by(self, dz: float):
        self.z += float(dz)

    def get_position(self):
        return self.z

    def stop(self):
        pass

    def disconnect(self):
        self.connected = False


class SimulatedCamera:
    """Simulated camera producing Gaussian blobs."""

    def __init__(self, width=512, height=512):
        self.connected = False
        self.width = width
        self.height = height
        self.exposure_ms = 20.0
        self._trigger_mode = "internal"
        self._live = False

    def connect(self):
        self.connected = True

    def reset(self):
        # default exposure/trigger
        self.exposure_ms = float(self.exposure_ms)
        self._trigger_mode = "internal"
        self._live = False

    def get_capabilities(self):
        return _caps("camera", width=int(self.width), height=int(self.height))

    def snap(self):
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xx, yy = np.meshgrid(x, y)
        img = np.exp(-(xx**2 + yy**2) * 8.0)
        img += 0.1 * np.random.randn(self.height, self.width)
        return img.astype(np.float32)

    def set_exposure(self, ms: float):
        self.exposure_ms = ms

    def set_trigger_mode(self, mode: str) -> None:
        self._trigger_mode = str(mode)

    def start_live(self) -> None:
        self._live = True

    def stop_live(self) -> None:
        self._live = False

    def disconnect(self):
        self.connected = False


class SimulatedLight:
    """Simulated light source with scalar intensity."""

    def __init__(self):
        self.connected = False
        self.intensity = 0.0
        self._on = False

    def connect(self):
        self.connected = True

    def reset(self):
        self.intensity = 0.0
        self._on = False

    def get_capabilities(self):
        return _caps("light_source")

    def set_intensity(self, value: float):
        self.intensity = float(value)

    def on(self) -> None:
        self._on = True

    def off(self) -> None:
        self._on = False

    def disconnect(self):
        self.connected = False


class SimulatedFilterWheel:
    """Simulated filter wheel with discrete positions."""

    def __init__(self):
        self.connected = False
        self.position = 0

    def connect(self):
        self.connected = True

    def reset(self):
        self.position = 0

    def get_capabilities(self):
        return _caps("filter_wheel")

    def set_position(self, pos: int):
        self.position = int(pos)

    def get_position(self) -> int:
        return int(self.position)

    def disconnect(self):
        self.connected = False


class SimulatedDetector:
    """Simulated scalar detector with scale + offset."""

    def __init__(self):
        self.connected = False
        self.scale = 1.0
        self.offset = 0.0

    def connect(self):
        self.connected = True

    def reset(self):
        pass

    def get_capabilities(self):
        return _caps("detector")

    def set_scale(self, scale: float, offset: float = 0.0):
        self.scale = float(scale)
        self.offset = float(offset)

    def read_value(self,wait:float = 0):
        base = np.random.rand()
        if wait>0:
            time.sleep(wait)
        return base * self.scale + self.offset

    def disconnect(self):
        self.connected = False