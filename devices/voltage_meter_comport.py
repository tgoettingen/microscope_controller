from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Callable
import time

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class ComPort(QObject):
    """Framed serial detector for the microscope voltage meter."""

    sample_received = pyqtSignal(str, object, float)
    error = pyqtSignal(str)

    VREF = 2.048
    GAIN = 501
    ADC_MAX = 2 ** 23
    DEFAULT_FRAME_LENGTH = 10
    DEFAULT_FRAME_HEADER = b"\x0a\x01"
    DEFAULT_FRAME_TRAILER = b"\x01\x0a"

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 115200,
        read_timeout: float = 0.1,
        sample_format: str = "int24",
        mode: int | str | None = None,
        name: str | None = None,
        reader_hz: float = 40.0,
        ring_buffer_size: int = 81920,
        frame_length: int = DEFAULT_FRAME_LENGTH,
        frame_header: bytes | str = DEFAULT_FRAME_HEADER,
        frame_trailer: bytes | str = DEFAULT_FRAME_TRAILER,
        overflow_policy: str = "overwrite",
    ) -> None:
        QObject.__init__(self)
        self.name = name or (str(port) if port else "ComPort")
        self.port = port
        self.baudrate = int(baudrate)
        self.read_timeout = float(read_timeout)
        self.sample_format = str(sample_format).lower()
        self.mode = mode
        self.reader_hz = float(reader_hz)
        self._ring_buffer_size = int(ring_buffer_size)
        self._frame_length = int(frame_length)
        self._frame_header = self._marker_bytes(frame_header)
        self._frame_trailer = self._marker_bytes(frame_trailer)
        self._overflow_policy = str(overflow_policy).lower()
        if self.sample_format != "int24":
            raise ValueError(f"Unsupported ComPort sample format: {sample_format}")
        if self._frame_length != self.DEFAULT_FRAME_LENGTH:
            raise ValueError("ComPort supports only the 10-byte production frame")
        if len(self._frame_header) != 2 or len(self._frame_trailer) != 2:
            raise ValueError("ComPort frame markers must each be exactly 2 bytes")
        if self._ring_buffer_size < 1:
            raise ValueError("ring_buffer_size must be positive")
        if self._overflow_policy not in {"overwrite", "reject"}:
            raise ValueError("overflow_policy must be 'overwrite' or 'reject'")

        self.ser = None
        self.connected = False
        self.running = False
        self.last_error: str | None = None
        self.scale = 1.0
        self.offset = 0.0
        self._rx_buffer = bytearray()
        self._ring_buffer: deque[dict[str, Any]] = deque(maxlen=self._ring_buffer_size)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._last_value: float | None = None
        self._last_scaled_value: float | None = None
        self._last_temperature: bytes | None = None
        self._last_timestamp: float | None = None
        self._last_raw = b""
        self._last_adc = 0
        self._frames_parsed = 0
        self._frames_rejected = 0
        self._bytes_discarded = 0
        self._overflow_count = 0
        self.valid_count = 0
        self.error_count = 0

    @staticmethod
    def _marker_bytes(marker: bytes | str) -> bytes:
        if isinstance(marker, bytes):
            return marker
        try:
            return bytes.fromhex(str(marker).replace(" ", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid frame marker: {marker!r}") from exc

    @staticmethod
    def find_ch340_ports():
        if serial is None:
            return [], []
        ports = serial.tools.list_ports.comports()
        ch340, others = [], []
        for item in ports:
            text = " ".join((item.description or "", item.manufacturer or "", item.hwid or "")).upper()
            (ch340 if "CH340" in text else others).append(item)
        return ch340, others

    @staticmethod
    def list_all_ports():
        return list(serial.tools.list_ports.comports()) if serial is not None else []

    @classmethod
    def parse_frame(cls, frame: bytes, header: bytes = DEFAULT_FRAME_HEADER, trailer: bytes = DEFAULT_FRAME_TRAILER):
        if len(frame) != cls.DEFAULT_FRAME_LENGTH or frame[:2] != header or frame[-2:] != trailer:
            return None
        adc_raw = int.from_bytes(frame[2:5], byteorder="big", signed=True)
        voltage = adc_raw * cls.VREF / (cls.GAIN * cls.ADC_MAX)
        return {
            "raw_bytes": bytes(frame),
            "middle": bytes(frame[5:8]),
            "adc_raw": adc_raw,
            "voltage": voltage,
            "temperature_raw": bytes(frame[5:8]),
            "timestamp": time.time(),
        }

    def connect(self, port: str | None = None) -> str:
        if self.connected and self.ser is not None:
            return str(self.port)
        if port:
            self.port = port
        if not self.port:
            ch340, others = self.find_ch340_ports()
            candidates = ch340 or others
            if candidates:
                self.port = candidates[0].device
            else:
                raise RuntimeError("No serial port found")
        if serial is None:
            raise RuntimeError("pyserial is not available")
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.read_timeout,
            )
        except Exception as exc:
            self.connected = False
            self.ser = None
            self._report_error(f"Failed to open serial port {self.port}: {exc}")
            raise
        self.connected = True
        self.last_error = None
        return str(self.port)

    def disconnect(self) -> None:
        self.stop()
        ser, self.ser = self.ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        self.connected = False

    def reset(self) -> None:
        with self._lock:
            self._rx_buffer.clear()
            self._ring_buffer.clear()
            self._last_value = None
            self._last_scaled_value = None
            self._last_temperature = None
            self._last_timestamp = None

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "type": "voltage_comport",
            "sample_format": self.sample_format,
            "frame_length": self._frame_length,
            "supported_modes": [0, 1, 2],
            "temperature": "raw",
        }

    def set_mode(self, mode: int | str | None) -> None:
        if mode is not None:
            try:
                mode = int(mode)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid ComPort mode: {mode!r}") from exc
            if mode not in {0, 1, 2}:
                raise ValueError(f"Unsupported ComPort mode: {mode}")
        self.mode = mode

    def set_scale(self, scale: float, offset: float = 0.0) -> None:
        with self._lock:
            self.scale = float(scale)
            self.offset = float(offset)
            if self._last_value is not None:
                self._last_scaled_value = self.scale * self._last_value + self.offset

    def read_value(self) -> float:
        with self._lock:
            if self._last_scaled_value is None:
                raise RuntimeError("ComPort has not received a sample")
            return float(self._last_scaled_value)

    def read_temperature(self) -> bytes | None:
        with self._lock:
            return self._last_temperature

    def get_voltage(self) -> float:
        return self.read_value()

    def add_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def start(self) -> None:
        if not self.connected:
            self.connect()
        if self.running:
            return
        self.running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, name=f"{self.name}-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.running = False
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def close(self) -> None:
        self.disconnect()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_args) -> None:
        self.disconnect()

    def _report_error(self, message: str) -> None:
        self.last_error = str(message)
        self.error_count += 1
        try:
            self.error.emit(self.last_error)
        except Exception:
            pass

    def _consume_rx_frames(self) -> None:
        while len(self._rx_buffer) >= self._frame_length:
            header_index = self._rx_buffer.find(self._frame_header)
            if header_index < 0:
                keep = len(self._frame_header) - 1
                discarded = max(0, len(self._rx_buffer) - keep)
                del self._rx_buffer[:discarded]
                self._bytes_discarded += discarded
                return
            if header_index:
                del self._rx_buffer[:header_index]
                self._bytes_discarded += header_index
            if len(self._rx_buffer) < self._frame_length:
                return
            frame = bytes(self._rx_buffer[:self._frame_length])
            if frame[-len(self._frame_trailer):] != self._frame_trailer:
                self._frames_rejected += 1
                del self._rx_buffer[:1]
                self._bytes_discarded += 1
                continue
            del self._rx_buffer[:self._frame_length]
            result = self.parse_frame(frame, self._frame_header, self._frame_trailer)
            if result is None:
                self._frames_rejected += 1
                continue
            self._record_sample(result)

    def _record_sample(self, result: dict[str, Any]) -> None:
        with self._lock:
            value = float(result["voltage"])
            scaled = self.scale * value + self.offset
            result["scaled_value"] = scaled
            if len(self._ring_buffer) == self._ring_buffer.maxlen:
                self._overflow_count += 1
                if self._overflow_policy == "reject":
                    return
            self._ring_buffer.append(result)
            self._last_value = value
            self._last_scaled_value = scaled
            self._last_temperature = result["temperature_raw"]
            self._last_timestamp = result["timestamp"]
            self._last_raw = result["raw_bytes"]
            self._last_adc = result["adc_raw"]
            self._frames_parsed += 1
            self.valid_count = self._frames_parsed
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            try:
                callback(result)
            except Exception:
                logger.exception("ComPort callback failed")
        try:
            self.sample_received.emit(self.name, result["timestamp"], scaled)
        except Exception:
            logger.exception("ComPort sample signal failed")

    def _read_loop(self) -> None:
        interval = 1.0 / self.reader_hz if self.reader_hz > 0 else 0.0
        while self.running and not self._stop_event.is_set():
            try:
                if self.ser is None:
                    break
                waiting = getattr(self.ser, "in_waiting", 0)
                chunk = self.ser.read(waiting or 1) if waiting else b""
                if chunk:
                    self._rx_buffer.extend(chunk)
                    self._consume_rx_frames()
                if interval:
                    self._stop_event.wait(interval)
            except Exception as exc:
                self._report_error(f"Serial read failed: {exc}")
                break
        self.running = False

    def get_recent_samples(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._ring_buffer)

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "valid": self._frames_parsed,
                "error": self.error_count,
                "rejected": self._frames_rejected,
                "discarded": self._bytes_discarded,
                "overflow": self._overflow_count,
                "voltage": self._last_scaled_value,
                "raw": self._last_raw,
                "adc": self._last_adc,
                "temperature_raw": self._last_temperature,
            }

    def voltages(self):
        import queue

        values = queue.Queue(maxsize=1)

        def callback(data: dict[str, Any]) -> None:
            try:
                values.put_nowait(data["scaled_value"])
            except queue.Full:
                try:
                    values.get_nowait()
                    values.put_nowait(data["scaled_value"])
                except queue.Empty:
                    pass

        self.add_callback(callback)
        self.start()
        try:
            while self.running:
                try:
                    yield values.get(timeout=0.5)
                except queue.Empty:
                    continue
        finally:
            self.remove_callback(callback)
