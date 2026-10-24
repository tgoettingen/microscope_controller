"""Multi-channel HDF5 file format helpers.

File structure
--------------
/ (root)
  attrs:
    format   = "multichannel_stream"
    version  = "1"
  detectors/
    <detector_id>/
            data:  float64  (N, 6)   [timestamp, value, x, y, z, temperature]
                attrs: columns = ["timestamp","value","x","y","z","temperature"]
               detector = <detector_id>
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import h5py

FORMAT_ATTR = "multichannel_stream"
COLUMNS = ["timestamp", "value", "x", "y", "z", "temperature"]


def save(
    path: str | Path,
    channels: dict[str, list[tuple[dict, float]]],
    *,
    overwrite: bool = True,
) -> Path:
    """Save multi-axis scan data for multiple detectors into a single HDF5.

    Parameters
    ----------
    path:
        Output file path (will get `.h5` extension if not already).
    channels:
        Mapping of detector_id → list of (state_dict, value) pairs,
        exactly as stored in ``LiveTab.multi_coords``.
    overwrite:
        If True, overwrite an existing file.

    Returns
    -------
    Path of the written file.
    """
    p = Path(path)
    if p.suffix.lower() not in (".h5", ".hdf5"):
        p = p.with_suffix(".h5")
    p.parent.mkdir(parents=True, exist_ok=True)

    mode = "w" if overwrite else "x"
    with h5py.File(p, mode) as f:
        f.attrs["format"] = FORMAT_ATTR
        f.attrs["version"] = "1"
        f.attrs["detector_ids"] = list(channels.keys())

        dets_grp = f.require_group("detectors")
        for det_id, samples in channels.items():
            if not samples:
                continue
            rows: list[list[float]] = []
            for state, value in samples:
                ts = float(state.get("_timestamp", 0.0))
                x = float(state.get("X", state.get("x", float("nan"))))
                y = float(state.get("Y", state.get("y", float("nan"))))
                z = float(state.get("Z", state.get("z", float("nan"))))
                temp = float(state.get("temperature", float("nan")))
                rows.append([ts, float(value), x, y, z, temp])

            arr = np.array(rows, dtype="float64")
            grp = dets_grp.require_group(det_id)
            ds = grp.create_dataset(
                "data",
                data=arr,
                chunks=(min(256, max(1, len(arr))), 6),
            )
            ds.attrs["columns"] = COLUMNS
            ds.attrs["detector"] = det_id

    return p


def is_multichannel(path: str | Path) -> bool:
    """Return True if the HDF5 file uses the multi-channel format."""
    try:
        with h5py.File(path, "r") as f:
            return str(f.attrs.get("format", "")) == FORMAT_ATTR
    except Exception:
        return False


def load(path: str | Path) -> dict[str, np.ndarray]:
    """Load a multi-channel HDF5 file.

    Returns
    -------
    dict mapping detector_id → float64 array of shape (N, 6)
    with columns [timestamp, value, x, y, z, temperature].
    """
    result: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        dets_grp = f.get("detectors")
        if dets_grp is None:
            raise KeyError("No 'detectors' group — not a multi-channel file")
        for det_id in dets_grp:
            ds = dets_grp[det_id].get("data")
            if ds is None:
                continue
            arr = np.asarray(ds[:], dtype="float64")
            result[det_id] = arr
    return result


def load_phased(path: str | Path) -> dict[str, np.ndarray]:
    """Load detector channels, separating samples by saved sequence phase.

    Files without phase metadata retain the original detector-name keys. If a
    detector has samples from multiple phases, each phase is returned as its own
    channel so scans with different axes are not plotted as one incompatible
    coordinate set.
    """
    result: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        dets_grp = f.get("detectors")
        if dets_grp is None:
            raise KeyError("No 'detectors' group — not a multi-channel file")
        for group_name in dets_grp:
            group = dets_grp[group_name]
            ds = group.get("data")
            if ds is None:
                continue
            detector_attr = ds.attrs.get("detector", group.attrs.get("detector"))
            if detector_attr is None:
                detector_attr = group_name
            detector_id = (
                detector_attr.decode("utf-8")
                if isinstance(detector_attr, bytes)
                else str(detector_attr)
            )
            data = np.asarray(ds[:], dtype="float64")
            phase_ds = group.get("experiment_phase")
            if phase_ds is None:
                result[detector_id] = data
                continue

            phases = np.asarray(phase_ds.asstr()[:], dtype=str)
            if len(phases) != len(data):
                raise ValueError(
                    f"Detector {detector_id!r} has {len(data)} samples but "
                    f"{len(phases)} phase labels"
                )
            unique_phases = list(dict.fromkeys(phases.tolist()))
            if len(unique_phases) <= 1:
                result[detector_id] = data
                continue

            for phase in unique_phases:
                mask = phases == phase
                channel_id = (
                    detector_id
                    if not phase
                    else f"{detector_id} [{phase}]"
                )
                if channel_id in result:
                    suffix = 2
                    while f"{channel_id}_{suffix}" in result:
                        suffix += 1
                    channel_id = f"{channel_id}_{suffix}"
                result[channel_id] = data[mask]
    return result
