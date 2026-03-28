from __future__ import annotations

import threading
import time
from typing import Any, Dict

import numpy as np


class WebcamCamera:
    """Simple OpenCV-backed system webcam camera.

    This is intentionally minimal and designed for GUI preview (snapshot + live).
    """

    def __init__(
        self,
        index: int = 0,
    ):
        self.index = int(index)
        self.connected = False
        self._cap = None
        self._lock = threading.Lock()
        self._exposure_ms: float | None = None

    def connect(self) -> None:
        if self.connected:
            return
        import cv2  # lazy import so the rest of the app can import without cv2

        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # retry without backend hint
            cap.release()
            cap = cv2.VideoCapture(self.index)

        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Could not open webcam at index {self.index}")

        self._cap = cap
        self.connected = True

        # Apply cached exposure if set
        try:
            if self._exposure_ms is not None:
                self.set_exposure(self._exposure_ms)
        except Exception:
            pass

    def disconnect(self) -> None:
        with self._lock:
            cap = self._cap
            self._cap = None
            self.connected = False
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

    def reset(self) -> None:
        # no-op for webcam
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        return {"kind": "webcam", "index": self.index}

    def set_exposure(self, ms: float) -> None:
        """Best-effort exposure control.

        Many webcams/drivers ignore this; we store the value regardless.
        """
        self._exposure_ms = float(ms)
        if not self.connected or self._cap is None:
            return

        import cv2

        # Guard property writes; snap() may be running on another thread.
        with self._lock:
            if self._cap is None:
                return

            # OpenCV exposure is backend-specific; try a couple of common forms.
            # Some drivers want negative log2 seconds; others want milliseconds.
            try:
                self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # try manual
            except Exception:
                pass

            # Try milliseconds directly
            try:
                self._cap.set(cv2.CAP_PROP_EXPOSURE, float(ms))
                return
            except Exception:
                pass

            # Try seconds
            try:
                self._cap.set(cv2.CAP_PROP_EXPOSURE, float(ms) / 1000.0)
            except Exception:
                pass

    def snap(self) -> np.ndarray:
        """Return an RGB image as HxWx3 uint8."""
        if not self.connected:
            self.connect()

        import cv2

        with self._lock:
            if self._cap is None:
                raise RuntimeError("Webcam is not connected")

            ok, frame = self._cap.read()

        if not ok or frame is None:
            raise RuntimeError("Failed to read frame from webcam")

        # frame is BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.asarray(rgb, dtype=np.uint8)

    def warmup(self, seconds: float = 0.2) -> None:
        """Best-effort warmup: read frames for a short duration."""
        t_end = time.time() + float(seconds)
        while time.time() < t_end:
            try:
                _ = self.snap()
            except Exception:
                break
