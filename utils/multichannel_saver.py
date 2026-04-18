"""MultiChannelSaver — writes all detector streams into a single HDF5 file.

File layout follows utils/multichannel_h5.py:
    /
      attrs: format="multichannel_stream", version="1"
      detectors/
        <detector_id>/
          data  (N, 5) float64  [timestamp, value, x, y, z]
"""

from __future__ import annotations

import threading
import time
import re
from pathlib import Path

import numpy as np
import h5py

from utils.multichannel_h5 import FORMAT_ATTR, COLUMNS


class MultiChannelSaver:
    """Thread-safe, streaming multi-detector HDF5 saver.

    Usage
    -----
        saver = MultiChannelSaver(output_dir)
        # from any thread:
        saver.append_sample("detector1", timestamp, value, meta)
        saver.append_sample("detector2", timestamp, value, meta)
        saver.close()   # flushes and closes the file
    """

    def __init__(
        self,
        output_dir: str | Path,
        flush_every: int = 256,
        base_path: str | Path | None = None,
        measurement_id: str | None = None,
        layout_json: str | None = None,
        software_version: str | None = None,
        experiment_type: str | None = None,
    ):
        if base_path is not None:
            bp = Path(base_path)
            if bp.suffix.lower() in (".h5", ".hdf5"):
                bp = bp.with_suffix("")
            self.output_dir = bp.parent
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.h5_path = bp.with_suffix(".h5")
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.h5_path = self.output_dir / f"{ts}__multichannel.h5"

        self._h5_file = h5py.File(self.h5_path, "w")
        self._h5_file.attrs["format"] = FORMAT_ATTR
        self._h5_file.attrs["version"] = "1"
        if measurement_id is not None:
            self._h5_file.attrs["measurement_id"] = str(measurement_id)
        if layout_json is not None:
            self._h5_file.attrs["display_layout"] = str(layout_json)
        if software_version is not None:
            self._h5_file.attrs["software_version"] = str(software_version)
        if experiment_type is not None:
            self._h5_file.attrs["experiment_type"] = str(experiment_type)
        self._dets_grp = self._h5_file.require_group("detectors")

        # per-detector: dataset handle + buffer
        self._ds: dict[str, h5py.Dataset] = {}
        self._buffers: dict[str, list] = {}

        self._lock = threading.Lock()
        self.flush_every = int(flush_every)
        self._closed = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def append_sample(
        self,
        detector_id: str,
        timestamp: float,
        value: float,
        meta: dict | None = None,
    ) -> None:
        if self._closed:
            return
        meta = meta or {}
        x = y = z = float("nan")
        try:
            state = meta.get("state", meta)
            x = float(state.get("X", state.get("x", float("nan"))))
            y = float(state.get("Y", state.get("y", float("nan"))))
            z = float(state.get("Z", state.get("z", float("nan"))))
        except Exception:
            pass

        with self._lock:
            if self._closed:
                return
            self._ensure_detector(detector_id)
            self._buffers[detector_id].append(
                (float(timestamp), float(value), x, y, z)
            )
            if len(self._buffers[detector_id]) >= self.flush_every:
                self._flush_detector(detector_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for det_id in list(self._buffers):
                self._flush_detector(det_id)
            # write final detector list
            try:
                self._h5_file.attrs["detector_ids"] = list(self._ds.keys())
            except Exception:
                pass
            try:
                self._h5_file.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Internal helpers  (must be called with _lock held)
    # ------------------------------------------------------------------ #

    def _ensure_detector(self, det_id: str) -> None:
        if det_id in self._ds:
            return
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(det_id).strip())
        grp = self._dets_grp.require_group(safe)
        ds = grp.create_dataset(
            "data",
            shape=(0, 5),
            maxshape=(None, 5),
            dtype="float64",
            chunks=(256, 5),
        )
        ds.attrs["columns"] = COLUMNS
        ds.attrs["detector"] = det_id
        self._ds[det_id] = ds
        self._buffers[det_id] = []

    def _flush_detector(self, det_id: str) -> None:
        buf = self._buffers.get(det_id)
        if not buf:
            return
        arr = np.array(buf, dtype="float64")
        ds = self._ds[det_id]
        old = int(ds.shape[0])
        ds.resize(old + len(arr), axis=0)
        ds[old:, :] = arr
        self._h5_file.flush()
        buf.clear()
