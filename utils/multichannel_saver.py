"""MultiChannelSaver — writes all detector streams into a single HDF5 file.

File layout follows utils/multichannel_h5.py:
    /
      attrs: format="multichannel_stream", version="1"
      detectors/
        <detector_id>/
                    data  (N, 6) float64  [timestamp, value, x, y, z, temperature]
                    experiment_phase (N,) UTF-8 phase labels
      images, image_timestamps, image_meta_json (optional camera frames)
"""

from __future__ import annotations

import json
import re
import threading
import time
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
        self._phase_ds: dict[str, h5py.Dataset] = {}
        self._phase_buffers: dict[str, list[str]] = {}
        self._image_ds: h5py.Dataset | None = None
        self._image_timestamps: h5py.Dataset | None = None
        self._image_meta: h5py.Dataset | None = None
        self._image_shape: tuple[int, ...] | None = None
        self._image_dtype: np.dtype | None = None
        self._image_buffer: list[tuple[np.ndarray, float, str]] = []

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
        temp = float("nan")
        try:
            state = meta.get("state", meta)
            x = float(state.get("X", state.get("x", float("nan"))))
            y = float(state.get("Y", state.get("y", float("nan"))))
            z = float(state.get("Z", state.get("z", float("nan"))))
            # Temperature may be attached either to the nested state payload
            # or directly on the sample metadata.
            if isinstance(meta, dict) and "temperature" in meta:
                temp = float(meta.get("temperature", float("nan")))
            else:
                temp = float(state.get("temperature", float("nan")))
        except Exception:
            pass
        phase = str(meta.get("experiment_phase", ""))

        with self._lock:
            if self._closed:
                return
            self._ensure_detector(detector_id)
            self._buffers[detector_id].append(
                (float(timestamp), float(value), x, y, z, temp)
            )
            self._phase_buffers[detector_id].append(phase)
            if len(self._buffers[detector_id]) >= self.flush_every:
                self._flush_detector(detector_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for det_id in list(self._buffers):
                self._flush_detector(det_id)
            self._flush_images()
            # write final detector list
            try:
                self._h5_file.attrs["detector_ids"] = list(self._ds.keys())
            except Exception:
                pass
            try:
                self._h5_file.close()
            except Exception:
                pass

    def append_image(self, image: object, meta: dict | None = None) -> None:
        """Append a camera frame and its JSON metadata to the same measurement file."""
        if self._closed:
            return
        meta = meta or {}
        array = np.asarray(image)
        try:
            timestamp = float(meta.get("timestamp", time.time()))
        except Exception:
            timestamp = time.time()
        try:
            payload = json.dumps(
                meta,
                default=lambda obj: getattr(obj, "__dict__", str(obj)),
            )
        except (TypeError, ValueError):
            payload = json.dumps({"meta": str(meta)})

        with self._lock:
            if self._closed:
                return
            if self._image_shape is None:
                self._image_shape = tuple(int(size) for size in array.shape)
                self._image_dtype = array.dtype
                self._image_ds = self._h5_file.create_dataset(
                    "images",
                    shape=(0,) + self._image_shape,
                    maxshape=(None,) + self._image_shape,
                    dtype=self._image_dtype,
                    chunks=(1,) + self._image_shape,
                )
                self._image_timestamps = self._h5_file.create_dataset(
                    "image_timestamps",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="float64",
                    chunks=(256,),
                )
                self._image_meta = self._h5_file.create_dataset(
                    "image_meta_json",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"),
                    chunks=(256,),
                )
            elif tuple(array.shape) != self._image_shape:
                raise ValueError(
                    f"Image shape changed: expected {self._image_shape}, got {tuple(array.shape)}"
                )
            elif self._image_dtype is not None and array.dtype != self._image_dtype:
                array = array.astype(self._image_dtype, copy=False)

            self._image_buffer.append((array, timestamp, payload))
            if len(self._image_buffer) >= self.flush_every:
                self._flush_images()

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
            shape=(0, 6),
            maxshape=(None, 6),
            dtype="float64",
            chunks=(256, 6),
        )
        ds.attrs["columns"] = COLUMNS
        ds.attrs["detector"] = det_id
        self._ds[det_id] = ds
        self._buffers[det_id] = []
        self._phase_ds[det_id] = grp.create_dataset(
            "experiment_phase",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=(256,),
        )
        self._phase_buffers[det_id] = []

    def _flush_detector(self, det_id: str) -> None:
        buf = self._buffers.get(det_id)
        if not buf:
            return
        arr = np.array(buf, dtype="float64")
        ds = self._ds[det_id]
        old = int(ds.shape[0])
        ds.resize(old + len(arr), axis=0)
        ds[old:, :] = arr
        phase_ds = self._phase_ds[det_id]
        phase_buffer = self._phase_buffers[det_id]
        phase_ds.resize(old + len(phase_buffer), axis=0)
        phase_ds[old:] = phase_buffer
        self._h5_file.flush()
        buf.clear()
        phase_buffer.clear()

    def _flush_images(self) -> None:
        if not self._image_buffer:
            return
        if (
            self._image_ds is None
            or self._image_timestamps is None
            or self._image_meta is None
        ):
            raise RuntimeError("Image datasets were not initialized")
        old = int(self._image_ds.shape[0])
        count = len(self._image_buffer)
        self._image_ds.resize(old + count, axis=0)
        self._image_timestamps.resize(old + count, axis=0)
        self._image_meta.resize(old + count, axis=0)
        for offset, (array, timestamp, payload) in enumerate(self._image_buffer):
            self._image_ds[old + offset, ...] = array
            self._image_timestamps[old + offset] = timestamp
            self._image_meta[old + offset] = payload
        self._h5_file.flush()
        self._image_buffer.clear()
