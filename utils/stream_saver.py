import time
import threading
from pathlib import Path
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
        # write header if empty
        if self.ascii_path.stat().st_size == 0:
            self._ascii_file.write("timestamp,value,meta\n")

        self._buffer = []
        self._lock = threading.Lock()
        self.flush_every = int(flush_every)

    def append_sample(self, timestamp: float, value: float, meta: dict | None = None):
        meta = meta or {}
        line = f"{timestamp:.6f},{value:.6g},{meta}\n"
        with self._lock:
            self._ascii_file.write(line)
            self._buffer.append((timestamp, float(value)))
            if len(self._buffer) >= self.flush_every:
                self._flush_npy()

    def _flush_npy(self):
        if not self._buffer:
            return
        arr = np.array(self._buffer, dtype=float)
        # if file exists, load and concatenate
        try:
            if self.npy_path.exists():
                existing = np.load(self.npy_path)
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
