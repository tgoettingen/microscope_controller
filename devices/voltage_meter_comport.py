import threading
import time
from collections import deque
from typing import List
import logging

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

from PyQt6.QtCore import QObject, pyqtSignal

from .base import Detector
from .urt_serial import UartReader


logger = logging.getLogger(__name__)


class ComPort(Detector):
    """COM port voltage reader backed by UartReader.

    Signals:
        sample_received(port: str, timestamp: float, value: float)
        error(message: str)
    """

    class _Emitter(QObject):
        sample_received = pyqtSignal(str, float, float)
        error = pyqtSignal(str)

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        read_timeout: float = 0.1,
        sample_format: str = 'int24',
        mode: int | None = None,
        name: str | None = None,
        reader_hz: float = 40.0,
        ring_buffer_size: int = 8192,
        frame_length: int = 9,
        frame_header: bytes | str | list[int] | None = b'\x0a\x01',
        frame_trailer: bytes | str | list[int] | None = b'\x01\x0a',
        overflow_policy: str = 'overwrite',
    ):
        nm = name if name is not None else port
        Detector.__init__(self, nm)
        self._emitter = ComPort._Emitter()
        self.sample_received = self._emitter.sample_received
        self.error = self._emitter.error
        self.port = port
        self.sample_format = sample_format
        self.mode = mode or 2
        self._scale = 1.0
        self._offset = 0.0
        self._lock = threading.Lock()
        self._last_value: float | None = None
        self._last_scaled_value: float | None = None
        self._last_timestamp: float | None = None
        self._last_temperature: float | None = None
        self._ring_buffer_size = max(1, int(ring_buffer_size))
        self._ring_buffer = deque(maxlen=self._ring_buffer_size)
        self._overflow_count = 0
        self._frames_parsed = 0
        self._frames_rejected = 0

        self._reader = UartReader(port=port, baudrate=baudrate)
        self._reader.add_callback(self._on_sample)
        self.start()

    def _on_sample(self, result: dict) -> None:
        """Callback invoked by UartReader for each successfully parsed frame."""
        voltage = result['voltage']
        ts = result['timestamp'].timestamp()
        scaled = float(self._scale) * float(voltage) + float(self._offset)
        with self._lock:
            self._last_value = float(voltage)
            self._last_scaled_value = float(scaled)
            self._last_timestamp = ts
            self._ring_buffer.append((ts, float(scaled), 0.0))
            self._frames_parsed += 1
        try:
            self.sample_received.emit(self.port, ts, float(scaled))
        except Exception:
            pass

    def set_mode(self, mode: int, wait_for_restart: bool = True) -> None:
        """Record the requested mode. UartReader does not send mode commands."""
        self.mode = int(mode)

    def start(self) -> None:
        try:
            self._reader.connect()
            self._reader.start()
            try:
                self.connected = True
            except Exception:
                pass
        except Exception as e:
            try:
                self.error.emit(f'Failed to open serial port {self.port}: {e}')
            except Exception:
                pass
            logger.error('Failed to open serial port %s: %s', self.port, e)

    def stop(self) -> None:
        self._reader.stop()
        try:
            self.connected = False
        except Exception:
            pass

    def is_open(self) -> bool:
        return self._reader.ser is not None and getattr(self._reader.ser, 'is_open', False)

    def read_value(self) -> float:
        with self._lock:
            if self._last_scaled_value is None:
                return float('nan')
            return float(self._last_scaled_value)


    def connect(self) -> None:
        self.start()

    def disconnect(self) -> None:
        self.stop()

    def set_scale(self, scale: float, offset: float = 0.0) -> None:
        self._scale = float(scale)
        self._offset = float(offset)

    def read_temperature(self) -> float | None:
        """Return last decoded temperature (mode 1), or None when unavailable."""
        with self._lock:
            return self._last_temperature

    def get_recent_samples(self, count: int | None = None) -> list[tuple[float, float, float]]:
        with self._lock:
            data = list(self._ring_buffer)
        if count is None or count >= len(data):
            return data
        return data[-max(0, int(count)):]

    def get_reader_stats(self) -> dict:
        uart_stats = self._reader.get_stats()
        with self._lock:
            return {
                'frames_parsed': int(self._frames_parsed),
                'frames_rejected': int(self._frames_rejected),
                'bytes_discarded': 0,
                'overflow_count': int(self._overflow_count),
                'ring_buffer_size': int(self._ring_buffer_size),
                'ring_buffer_used': int(len(self._ring_buffer)),
                'uart_valid': uart_stats['valid'],
                'uart_error': uart_stats['error'],
            }

    def get_capabilities(self) -> dict:
        return {
            'sample_format': self.sample_format,
            'mode': self.mode,
            'port': self.port,
            'frame_length': 10,
            'frame_header': '0a01',
            'frame_trailer': '010a',
            'ring_buffer_size': self._ring_buffer_size,
        }

    def reset(self) -> None:
        with self._lock:
            self._last_value = None
            self._last_scaled_value = None
            self._last_timestamp = None
            self._last_temperature = None
            self._ring_buffer.clear()
            self._overflow_count = 0
            self._frames_parsed = 0
            self._frames_rejected = 0

    @staticmethod
    def list_ports() -> List[str]:
        if serial is None:
            return []
        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append(p.device)
        return ports
