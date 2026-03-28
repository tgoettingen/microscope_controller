import json
import threading
import time
from pathlib import Path
import re

import numpy as np
import h5py


class ImageH5Saver:
    """Append camera frames + per-frame metadata into a single HDF5 file.

    Layout:
      - /images      : (N, ...) resizable dataset containing the raw frames
      - /timestamps  : (N,) float64 unix timestamps
      - /meta_json   : (N,) utf-8 JSON strings (metadata per frame)

    The first appended frame defines the image shape and dtype.
    """

    def __init__(
        self,
        output_dir: str | Path,
        base_name: str = "camera_frames",
        base_path: str | Path | None = None,
        flush_every: int = 1,
    ):
        self._lock = threading.Lock()
        self._closed = False

        if base_path is not None:
            bp = Path(base_path)
            if bp.suffix.lower() in (".h5", ".hdf5"):
                bp = bp.with_suffix("")
            self.output_dir = bp.parent
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            self.base_name = bp.name
            self.h5_path = bp.with_suffix(".h5")
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            safe = str(base_name).strip().replace(" ", "_")
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
            self.base_name = safe or "camera_frames"
            self.h5_path = self.output_dir / f"{self.base_name}.h5"

        self.flush_every = int(flush_every) if int(flush_every) > 0 else 1

        self._h5_file = h5py.File(self.h5_path, "w")
        self._ds_images = None
        self._img_shape: tuple[int, ...] | None = None
        self._img_dtype: np.dtype | None = None

        self._ds_timestamps = self._h5_file.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype="float64",
            chunks=(max(256, self.flush_every),),
        )
        self._ds_meta = self._h5_file.create_dataset(
            "meta_json",
            shape=(0,),
            maxshape=(None,),
            dtype=h5py.string_dtype(encoding="utf-8"),
            chunks=(max(256, self.flush_every),),
        )

        self._pending = 0

    def _ensure_images_dataset(self, arr: np.ndarray) -> None:
        if self._ds_images is not None:
            return

        self._img_shape = tuple(int(x) for x in arr.shape)
        self._img_dtype = np.asarray(arr).dtype

        # Chunk one frame at a time; simple and robust.
        chunks = (1,) + self._img_shape
        self._ds_images = self._h5_file.create_dataset(
            "images",
            shape=(0,) + self._img_shape,
            maxshape=(None,) + self._img_shape,
            dtype=self._img_dtype,
            chunks=chunks,
        )
        try:
            self._ds_images.attrs["frame_shape"] = list(self._img_shape)
            self._ds_images.attrs["dtype"] = str(self._img_dtype)
        except Exception:
            pass

    def append_image(self, img: object, meta: dict | None = None) -> None:
        if self._closed:
            return

        arr = np.asarray(img)
        meta = meta or {}
        try:
            ts = float(meta.get("timestamp", time.time()))
        except Exception:
            ts = time.time()

        # JSON encode metadata. Caller should pre-sanitize, but be defensive.
        try:
            meta_json = json.dumps(meta, default=lambda o: getattr(o, "__dict__", str(o)))
        except Exception:
            meta_json = json.dumps({"meta": str(meta)})

        with self._lock:
            if self._closed:
                return

            self._ensure_images_dataset(arr)

            if self._img_shape is not None and tuple(arr.shape) != self._img_shape:
                raise ValueError(f"Image shape changed: expected {self._img_shape}, got {tuple(arr.shape)}")

            if self._img_dtype is not None and arr.dtype != self._img_dtype:
                try:
                    arr = arr.astype(self._img_dtype, copy=False)
                except Exception:
                    pass

            idx = int(self._ds_timestamps.shape[0])

            self._ds_timestamps.resize(idx + 1, axis=0)
            self._ds_timestamps[idx] = float(ts)

            self._ds_meta.resize(idx + 1, axis=0)
            self._ds_meta[idx] = meta_json

            assert self._ds_images is not None
            self._ds_images.resize(idx + 1, axis=0)
            self._ds_images[idx, ...] = arr

            self._pending += 1
            if self._pending >= self.flush_every:
                try:
                    self._h5_file.flush()
                except Exception:
                    pass
                self._pending = 0

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._h5_file.flush()
            except Exception:
                pass
            try:
                self._h5_file.close()
            except Exception:
                pass
