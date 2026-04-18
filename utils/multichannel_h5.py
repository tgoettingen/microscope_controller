"""Multi-channel HDF5 file format helpers.

File structure
--------------
/ (root)
  attrs:
    format   = "multichannel_stream"
    version  = "1"
  detectors/
    <detector_id>/
      data:  float64  (N, 5)   [timestamp, value, x, y, z]
        attrs: columns = ["timestamp","value","x","y","z"]
               detector = <detector_id>
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import h5py

FORMAT_ATTR = "multichannel_stream"
COLUMNS = ["timestamp", "value", "x", "y", "z"]


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
                rows.append([ts, float(value), x, y, z])

            arr = np.array(rows, dtype="float64")
            grp = dets_grp.require_group(det_id)
            ds = grp.create_dataset(
                "data",
                data=arr,
                chunks=(min(256, max(1, len(arr))), 5),
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
    dict mapping detector_id → float64 array of shape (N, 5)
    with columns [timestamp, value, x, y, z].
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
