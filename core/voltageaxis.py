from devices.base import Detector
from multiaxis import Axis

class DetectorAxis(Axis):
    """Axis that configures detector scaling; measurement is detector readout."""

    def __init__(self, detector: Detector, scales: list[tuple[float, float]]):
        """
        scales: list of (scale, offset) pairs
        """
        self.detector = detector
        self.scales = scales

    def name(self) -> str:
        return "Detector"

    def prepare(self) -> None:
        pass

    def positions(self):
        for s in self.scales:
            yield s

    def apply(self, pos: tuple[float, float]) -> None:
        scale, offset = pos
        self.detector.set_scale(scale, offset)