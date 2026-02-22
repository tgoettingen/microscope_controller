from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ChannelConfig:
    name: str
    filter_position: int
    light_intensity: float
    exposure_ms: float


@dataclass
class Position:
    x: float
    y: float
    z: float | None = None
    label: str | None = None


@dataclass
class TimeLapseConfig:
    n_timepoints: int
    interval_s: float


@dataclass
class ZStackConfig:
    start_z: float
    end_z: float
    step_z: float


@dataclass
class ExperimentDefinition:
    name: str
    positions: List[Position]
    channels: List[ChannelConfig]
    timelapse: TimeLapseConfig | None = None
    zstack: ZStackConfig | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def iter_timepoints(self):
        if self.timelapse is None:
            yield 0
        else:
            for t in range(self.timelapse.n_timepoints):
                yield t

    def iter_z_positions(self, base_z: float | None = None):
        if self.zstack is None:
            yield base_z
        else:
            zcfg = self.zstack
            z = zcfg.start_z
            while z <= zcfg.end_z + 1e-9:
                yield z
                z += zcfg.step_z