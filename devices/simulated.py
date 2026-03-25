import time
import numpy as np


class SimulatedStageXY:
    """Simple XY stage simulation with instant movement."""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0

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
        pass


class SimulatedFocus:
    """Simulated Z axis."""

    def __init__(self):
        self.z = 100.0

    def move_to(self, z: float):
        self.z = float(z)

    def move_by(self, dz: float):
        self.z += float(dz)

    def get_position(self):
        return self.z

    def stop(self):
        pass

    def disconnect(self):
        pass


class SimulatedCamera:
    """Simulated camera producing Gaussian blobs."""

    def __init__(self, width=512, height=512):
        self.width = width
        self.height = height
        self.exposure_ms = 20.0

    def snap(self):
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        xx, yy = np.meshgrid(x, y)
        img = np.exp(-(xx**2 + yy**2) * 8.0)
        img += 0.1 * np.random.randn(self.height, self.width)
        return img.astype(np.float32)

    def set_exposure(self, ms: float):
        self.exposure_ms = ms

    def disconnect(self):
        pass


class SimulatedLight:
    """Simulated light source with scalar intensity."""

    def __init__(self):
        self.intensity = 0.0

    def set_intensity(self, value: float):
        self.intensity = float(value)

    def disconnect(self):
        pass


class SimulatedFilterWheel:
    """Simulated filter wheel with discrete positions."""

    def __init__(self):
        self.position = 0

    def set_position(self, pos: int):
        self.position = int(pos)

    def disconnect(self):
        pass


class SimulatedDetector:
    """Simulated scalar detector with scale + offset."""

    def __init__(self):
        self.scale = 1.0
        self.offset = 0.0

    def set_scale(self, scale: float, offset: float):
        self.scale = float(scale)
        self.offset = float(offset)

    def read_value(self,wait:float = 0):
        base = np.random.rand()
        if wait>0:
            time.sleep(wait)
        return base * self.scale + self.offset

    def disconnect(self):
        pass