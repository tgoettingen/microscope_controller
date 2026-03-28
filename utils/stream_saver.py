import time
import threading
from pathlib import Path
import re
import numpy as np
import h5py
import json


class StreamSaver:
    """Streaming saver that appends samples to an HDF5 file during acquisition.

    The HDF5 dataset 'data' is resizable so each flush simply extends it in
    place — no temp files, no merge step, no data loss on crash (h5py flushes
    after each write).

    Columns: [timestamp, value, x, y, z]

    Usage:
        saver = StreamSaver(output_dir, detector_id, mode='stream', flush_every=256)
        saver.append_sample(timestamp, value, meta_dict)
        saver.close()
    """

    def __init__(
        self,
        output_dir,
        detector_id: str,
        mode: str = "stream",
        flush_every: int = 256,
        base_path: str | Path | None = None,
    ):
        """Create a StreamSaver.

        If base_path is provided, it controls both the folder and filename stem.
        Two files are written: <base_path>.h5 and <base_path>.txt.
        """
        # Resolve output paths
        if base_path is not None:
            bp = Path(base_path)
            # If user selected a concrete file (e.g. .h5), use its stem as base.
            if bp.suffix.lower() in (".h5", ".hdf5", ".txt", ".csv"):
                bp = bp.with_suffix("")
            self.output_dir = bp.parent
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            self.base_name = bp.name
            self.ascii_path = bp.with_suffix(".txt")
            self.h5_path = bp.with_suffix(".h5")
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")

            # Windows-safe detector id for filenames
            safe_id = str(detector_id).strip().replace(" ", "_")
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe_id)
            self.base_name = f"{ts}__{safe_id}__{mode}"
            self.ascii_path = self.output_dir / f"{self.base_name}.txt"
            self.h5_path = self.output_dir / f"{self.base_name}.h5"

        # Open HDF5 file and create a resizable dataset.
        # Columns: timestamp, value, x, y, z
        self._h5_file = h5py.File(self.h5_path, "w")
        self._h5_ds = self._h5_file.create_dataset(
            "data",
            shape=(0, 5),
            maxshape=(None, 5),
            dtype="float64",
            chunks=(256, 5),
        )
        self._h5_ds.attrs["columns"] = ["timestamp", "value", "x", "y", "z"]
        self._h5_ds.attrs["detector"] = str(detector_id)

        # Full-fidelity event log (axis/motor events, etc.)
        try:
            self._h5_events = self._h5_file.create_dataset(
                "events",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                chunks=(256,),
            )
            self._h5_events.attrs["format"] = "json"
        except Exception:
            self._h5_events = None

        self._ascii_file = open(self.ascii_path, "a", buffering=1)
        if self.ascii_path.stat().st_size == 0:
            self._ascii_file.write("timestamp,value,x,y,z,meta\n")

        self._buffer = []
        self._lock = threading.Lock()
        self.flush_every = int(flush_every)
        self._closed = False

    def append_event(self, event: dict) -> None:
        """Append a JSON event record into the HDF5 'events' dataset."""
        if self._closed:
            return
        if getattr(self, "_h5_events", None) is None:
            return
        try:
            payload = json.dumps(event, default=lambda o: getattr(o, "__dict__", str(o)))
        except Exception:
            try:
                payload = json.dumps({"event": str(event)})
            except Exception:
                return

        with self._lock:
            if self._closed:
                return
            ds = getattr(self, "_h5_events", None)
            if ds is None:
                return
            try:
                i = int(ds.shape[0])
                ds.resize(i + 1, axis=0)
                ds[i] = payload
                self._h5_file.flush()
            except Exception:
                pass

    def append_sample(self, timestamp: float, value: float, meta: dict | None = None):
        if self._closed:
            return
        meta = meta or {}
        # Try to extract X/Y/Z positions from meta/state if provided
        x = y = z = ""
        try:
            # meta may contain a 'state' dict with motor positions
            state = None
            if isinstance(meta, dict):
                if "state" in meta and isinstance(meta["state"], dict):
                    state = meta["state"]
                else:
                    # allow top-level keys X/Y/Z or x/y/z
                    state = {k: meta.get(k) for k in ("X", "Y", "Z", "x", "y", "z") if k in meta}
            if state:
                # prefer uppercase keys if present
                if "X" in state:
                    x = state.get("X")
                elif "x" in state:
                    x = state.get("x")
                if "Y" in state:
                    y = state.get("Y")
                elif "y" in state:
                    y = state.get("y")
                if "Z" in state:
                    z = state.get("Z")
                elif "z" in state:
                    z = state.get("z")
        except Exception:
            x = y = z = ""

        # format x/y/z for ascii output (empty if not present)
        def _fmt(v):
            try:
                return f"{float(v):.6f}"
            except Exception:
                return ""

        xs = _fmt(x)
        ys = _fmt(y)
        zs = _fmt(z)

        line = f"{timestamp:.6f},{value:.6g},{xs},{ys},{zs},{meta}\n"
        with self._lock:
            self._ascii_file.write(line)
            # buffer now contains timestamp, value, x, y, z (floats, use nan for missing)
            try:
                xf = float(xs) if xs != "" else float('nan')
            except Exception:
                xf = float('nan')
            try:
                yf = float(ys) if ys != "" else float('nan')
            except Exception:
                yf = float('nan')
            try:
                zf = float(zs) if zs != "" else float('nan')
            except Exception:
                zf = float('nan')

            self._buffer.append((timestamp, float(value), xf, yf, zf))
            if len(self._buffer) >= self.flush_every:
                self._flush_h5()

    def _flush_h5(self):
        """Append buffered samples to the HDF5 dataset."""
        if not self._buffer:
            return
        arr = np.array(self._buffer, dtype=float)
        n = arr.shape[0]
        try:
            old_size = self._h5_ds.shape[0]
            self._h5_ds.resize(old_size + n, axis=0)
            self._h5_ds[old_size:old_size + n, :] = arr
            self._h5_file.flush()
        except Exception:
            pass
        finally:
            self._buffer.clear()

    def close(self):
        with self._lock:
            self._closed = True
            self._flush_h5()
            try:
                self._ascii_file.close()
            except Exception:
                pass
            try:
                self._h5_file.close()
            except Exception:
                pass
