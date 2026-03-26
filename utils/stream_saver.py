import time
import threading
from pathlib import Path
import json
import numpy as np


class StreamSaver:
    """Simple streaming saver that appends ASCII CSV lines and periodically writes a .npy file.

    Usage:
        saver = StreamSaver(output_dir, detector_id, mode='stream', flush_every=100)
        saver.append_sample(timestamp, value, meta_dict)
        saver.close()
    """

    def __init__(self, output_dir, detector_id: str, mode: str = "stream", flush_every: int = 256):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_id = detector_id.replace(" ", "_")
        self.base_name = f"{ts}__{safe_id}__{mode}"
        self.ascii_path = self.output_dir / f"{self.base_name}.txt"
        self.npy_path = self.output_dir / f"{self.base_name}.npy"

        self._ascii_file = open(self.ascii_path, "a", buffering=1)
        # write header if empty — we include optional X/Y/Z columns for stage positions
        if self.ascii_path.stat().st_size == 0:
            # columns: timestamp, value, x, y, z, meta
            self._ascii_file.write("timestamp,value,x,y,z,meta\n")

        self._buffer = []
        self._lock = threading.Lock()
        self.flush_every = int(flush_every)
        # companion metadata json-lines file
        self.meta_path = self.output_dir / f"{self.base_name}.meta.jsonl"
        try:
            self._meta_file = open(self.meta_path, "a", buffering=1)
        except Exception:
            self._meta_file = None

    def append_sample(self, timestamp: float, value: float, meta: dict | None = None):
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
                self._flush_npy()

            # write companion meta json-line for this sample
            if self._meta_file is not None:
                try:
                    record = {
                        "timestamp": float(timestamp),
                        "detector": self.base_name,
                        "value": None if (isinstance(value, float) and np.isnan(value)) else float(value),
                        "x": None if np.isnan(xf) else float(xf),
                        "y": None if np.isnan(yf) else float(yf),
                        "z": None if np.isnan(zf) else float(zf),
                        "meta": meta or {},
                    }
                    # ensure JSON serializable: replace non-serializable objects
                    def _convert(obj):
                        if hasattr(obj, "__dict__"):
                            return obj.__dict__
                        try:
                            return str(obj)
                        except Exception:
                            return None

                    record["meta"] = json.loads(json.dumps(record["meta"], default=_convert))
                    self._meta_file.write(json.dumps(record) + "\n")
                except Exception:
                    pass

    def _flush_npy(self):
        if not self._buffer:
            return
        arr = np.array(self._buffer, dtype=float)
        # if file exists, load and concatenate
        try:
            if self.npy_path.exists():
                existing = np.load(self.npy_path)
                # ensure existing has 5 columns: timestamp,value,x,y,z
                if existing.ndim == 1:
                    existing = existing.reshape(1, -1)
                if existing.shape[1] < 5:
                    # pad with nan columns
                    pad_cols = 5 - existing.shape[1]
                    pad = np.full((existing.shape[0], pad_cols), np.nan, dtype=float)
                    existing = np.hstack((existing, pad))
                combined = np.vstack((existing, arr))
            else:
                combined = arr
            # atomic write via temp file
            tmp = self.npy_path.with_suffix(self.npy_path.suffix + ".tmp")
            np.save(tmp, combined)
            tmp.replace(self.npy_path)
        except Exception:
            # best-effort: try overwrite
            try:
                np.save(self.npy_path, arr)
            except Exception:
                pass
        finally:
            self._buffer.clear()

    def close(self):
        with self._lock:
            self._flush_npy()
            try:
                self._ascii_file.close()
            except Exception:
                pass
            try:
                if self._meta_file is not None:
                    self._meta_file.close()
            except Exception:
                pass
