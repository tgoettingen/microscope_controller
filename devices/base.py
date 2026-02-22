from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple


class Device(ABC):
    def __init__(self, name: str):
        self.name = name
        self.connected = False

    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]: ...
    @abstractmethod
    def reset(self) -> None: ...


class Camera(Device):
    @abstractmethod
    def set_exposure(self, ms: float) -> None: ...
    @abstractmethod
    def set_trigger_mode(self, mode: str) -> None: ...
    @abstractmethod
    def snap(self) -> Any: ...
    @abstractmethod
    def start_live(self) -> None: ...
    @abstractmethod
    def stop_live(self) -> None: ...


class StageXY(Device):
    @abstractmethod
    def move_to(self, x: float, y: float) -> None: ...
    @abstractmethod
    def get_position(self) -> Tuple[float, float]: ...

class SingleAxis(Device):
    @abstractmethod
    def move_to(self, x: float) -> None: ...
    @abstractmethod
    def get_position(self) -> float: ...


class FocusZ(Device):
    @abstractmethod
    def move_to(self, z: float) -> None: ...
    @abstractmethod
    def get_position(self) -> float: ...


class LightSource(Device):
    @abstractmethod
    def set_intensity(self, percent: float) -> None: ...
    @abstractmethod
    def on(self) -> None: ...
    @abstractmethod
    def off(self) -> None: ...


class FilterWheel(Device):
    @abstractmethod
    def set_position(self, index: int) -> None: ...
    @abstractmethod
    def get_position(self) -> int: ...


class Detector(Device):
    """Generic single-point detector (photodiode, PMT, etc.)."""

    @abstractmethod
    def read_value(self) -> float:
        """Return current value (e.g. voltage, counts)."""
        ...

    @abstractmethod
    def set_scale(self, scale: float, offset: float = 0.0) -> None:
        """Configure scaling from raw units to display units."""
        ...