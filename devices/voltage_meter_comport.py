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
    DEFAULT_FRAME_LENGTH = 8
    DEFAULT_FRAME_HEADER = b"\x0a\x01"
    DEFAULT_FRAME_TRAILER = b"\x01\x0a"
    COMMAND_HEADER = b"\xaa\x55"
    FIRMWARE_VERSION_COMMAND = 0x08

    def __init__(
        self,
        port: str | None = None,
        baudrate: int = 460800,
        read_timeout: float = 0.1,
        sample_format: str = "int24",
        mode: int | str | None = None,
        name: str | None = None,
        reader_hz: float = 40.0,
        ring_buffer_size: int = 8192,
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
        if self._frame_length not in {8, 10}:
            raise ValueError("ComPort supports only 8-byte API frames or legacy 10-byte frames")
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
        self.GAIN = self.GAIN if hasattr(self, "GAIN") else 501
        self.firmware_version: dict[str, Any] | None = None
        self.product_id: int | None = None
        self._command_response: dict[str, Any] | None = None
        self._command_event = threading.Event()
        self._command_lock = threading.Lock()
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
        self._channel_values: dict[int, float | None] = {1: None, 2: None}
        self._channel_scaled_values: dict[int, float | None] = {1: None, 2: None}
        self._channel_timestamps: dict[int, float | None] = {1: None, 2: None}
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
    def parse_frame(cls, frame: bytes, header: bytes = DEFAULT_FRAME_HEADER, trailer: bytes = DEFAULT_FRAME_TRAILER, gain: int | None = None):
        if frame[:2] not in (header, b"\x0a\x02") or frame[-2:] != trailer:
            return None
        if len(frame) == 8:
            adc_raw = int.from_bytes(frame[2:5], byteorder="big", signed=True)
            if (sum(frame[2:5]) & 0xFF) != frame[5]:
                return None
            middle = frame[5:6]
            channel = frame[1]
        elif len(frame) == 10:
            adc_raw = int.from_bytes(frame[2:5], byteorder="big", signed=True)
            middle = frame[5:8]
            channel = frame[1]
        else:
            return None
        active_gain = cls.GAIN if gain is None else int(gain)
        voltage = adc_raw * cls.VREF / (active_gain * cls.ADC_MAX)
        return {
            "raw_bytes": bytes(frame),
            "middle": middle,
            "channel": channel,
            "adc_raw": adc_raw,
            "voltage": voltage,
            "temperature_raw": middle,
            "timestamp": time.time(),
        }

    def connect(self, port: str | None = None) -> str:
        if self.connected and self.ser is not None:
            return str(self.port)
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(
                f"Previous serial reader for {self.port} is still stopping"
            )
        self._thread = None
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
        try:
            self.read_firmware_version()
        except Exception as exc:
            self._report_error(f"Failed to read ComPort firmware version: {exc}")
        return str(self.port)

    def disconnect(self) -> None:
        self.running = False
        self._stop_event.set()
        ser = self.ser
        if ser is not None:
            for method_name in ("reset_input_buffer", "reset_output_buffer", "cancel_read"):
                method = getattr(ser, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        pass
            try:
                ser.close()
            except Exception:
                pass
        thread = self._thread
        if thread is not None and thread is not threading.current_thread() and thread.is_alive():
            thread.join(timeout=max(1.0, self.read_timeout + 1.0))
        self._thread = thread if thread is not None and thread.is_alive() else None
        self.ser = None
        self.connected = False
        if self._thread is None:
            self.reset()

    def reset(self) -> None:
        with self._lock:
            self._rx_buffer.clear()
            self._ring_buffer.clear()
            self._last_value = None
            self._last_scaled_value = None
            self._last_temperature = None
            self._last_timestamp = None
            self._channel_values = {1: None, 2: None}
            self._channel_scaled_values = {1: None, 2: None}
            self._channel_timestamps = {1: None, 2: None}

    def get_detector_ids(self) -> list[str]:
        if self.product_id == 2:
            return [f"{self.name} CH1", f"{self.name} CH2"]
        return [str(self.name)]

    def get_capabilities(self) -> dict[str, Any]:
        return {
            "type": "voltage_comport",
            "sample_format": self.sample_format,
            "frame_length": self._frame_length,
            "supported_modes": [0, 1, 2],
            "temperature": "raw",
            "gain": self.GAIN,
            "product_id": self.product_id,
            "firmware_version": self.firmware_version,
            "channel_count": 2 if self.product_id == 2 else 1,
            "detector_ids": self.get_detector_ids(),
        }

    @staticmethod
    def build_command_frame(command: int, payload: bytes = b"") -> bytes:
        command = int(command)
        payload = bytes(payload)
        if not 0 <= command <= 0x7F:
            raise ValueError("Command code must be between 0 and 127")
        if len(payload) > 255:
            raise ValueError("Command payload is too long")
        check = (command + len(payload) + sum(payload)) & 0xFF
        return ComPort.COMMAND_HEADER + bytes((command, len(payload))) + payload + bytes((check,))

    @staticmethod
    def parse_command_response(frame: bytes) -> dict[str, Any] | None:
        if len(frame) < 5 or frame[:2] != ComPort.COMMAND_HEADER:
            return None
        length = frame[3]
        if len(frame) != 5 + length:
            return None
        response_command = frame[2]
        status = frame[4]
        data = bytes(frame[5:-1])
        expected_check = ((response_command + length + sum(frame[4:-1])) & 0xFF)
        if frame[-1] != expected_check:
            return None
        return {
            "command": response_command & 0x7F,
            "status": status,
            "data": data,
            "success": status == 0,
            "raw_bytes": bytes(frame),
        }

    def send_command(self, command: int, payload: bytes = b"", timeout: float = 1.0) -> dict[str, Any]:
        if not self.connected or self.ser is None:
            raise RuntimeError("ComPort is not connected")
        with self._command_lock:
            self._command_response = None
            self._command_event.clear()
            frame = self.build_command_frame(command, payload)
            if not self.running:
                reset_input = getattr(self.ser, "reset_input_buffer", None)
                if callable(reset_input):
                    reset_input()
                self.ser.write(frame)
                deadline = time.monotonic() + float(timeout)
                response_buffer = bytearray()
                while time.monotonic() < deadline:
                    chunk = self.ser.read(max(1, getattr(self.ser, "in_waiting", 0) or 1))
                    if chunk:
                        response_buffer.extend(chunk)
                        command_index = response_buffer.find(self.COMMAND_HEADER)
                        if command_index >= 0:
                            if command_index:
                                del response_buffer[:command_index]
                            if len(response_buffer) < 4:
                                continue
                            length = response_buffer[3]
                            total_length = 5 + length
                            if len(response_buffer) < total_length:
                                continue
                            response = self.parse_command_response(bytes(response_buffer[:total_length]))
                            if response is not None:
                                return response
                            del response_buffer[:1]
                    else:
                        time.sleep(0.01)
                raise TimeoutError(f"No response for ComPort command 0x{int(command):02X}")

            self.ser.write(frame)
            if not self._command_event.wait(timeout=float(timeout)):
                raise TimeoutError(f"No response for ComPort command 0x{int(command):02X}")
            response = self._command_response
            if response is None:
                raise RuntimeError("Invalid response for ComPort command")
            return response

    def read_firmware_version(self, timeout: float = 1.0) -> dict[str, Any]:
        response = self.send_command(self.FIRMWARE_VERSION_COMMAND, timeout=timeout)
        if not response["success"]:
            raise RuntimeError(f"Firmware version command failed with status 0x{response['status']:02X}")
        data = response["data"]
        if len(data) != 3:
            raise RuntimeError(f"Invalid firmware version response: {data.hex()}")

        product_id, year, revision = data
        module_name = "single-channel" if product_id == 1 else "dual-channel" if product_id == 2 else "unknown"
        gain = 501 if product_id == 1 else 601 if product_id == 2 else self.GAIN
        self.product_id = product_id
        self.GAIN = int(gain)
        self.firmware_version = {
            "product_id": product_id,
            "year": year,
            "revision": revision,
            "firmware": f"{2000 + year if year < 100 else year}.{revision}",
            "module_name": module_name,
            "gain": gain,
        }
        return dict(self.firmware_version)

    get_firmware_version = read_firmware_version

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
            channel = int(self.mode) if self.mode in (1, 2, "1", "2") else 0
            if channel:
                value = self._channel_scaled_values.get(channel)
            else:
                value = self._last_scaled_value
            if value is None:
                suffix = f" for channel {channel}" if channel else ""
                raise RuntimeError(f"ComPort has not received a sample{suffix}")
            return float(value)

    def read_channel_value(self, channel: int) -> float:
        channel = int(channel)
        if channel not in (1, 2):
            raise ValueError("ComPort channel must be 1 or 2")
        with self._lock:
            value = self._channel_scaled_values.get(channel)
        if value is None:
            raise RuntimeError(f"ComPort has not received channel {channel} sample")
        return float(value)

    def read_all_channels(self) -> dict[int, float | None]:
        with self._lock:
            return dict(self._channel_scaled_values)

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
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"Previous serial reader for {self.port} is still stopping")
        self._thread = None
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
            thread.join(timeout=max(1.0, self.read_timeout + 1.0))
        self._thread = thread if thread is not None and thread.is_alive() else None

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
        while self._rx_buffer:
            live_headers = (self._frame_header, b"\x0a\x02")
            live_indexes = [i for header in live_headers if (i := self._rx_buffer.find(header)) >= 0]
            command_index = self._rx_buffer.find(self.COMMAND_HEADER)
            indexes = live_indexes + ([command_index] if command_index >= 0 else [])
            if not indexes:
                keep = max(len(self.COMMAND_HEADER), 2) - 1
                discarded = max(0, len(self._rx_buffer) - keep)
                del self._rx_buffer[:discarded]
                self._bytes_discarded += discarded
                return

            index = min(indexes)
            if index:
                del self._rx_buffer[:index]
                self._bytes_discarded += index

            if self._rx_buffer.startswith(self.COMMAND_HEADER):
                if len(self._rx_buffer) < 4:
                    return
                length = self._rx_buffer[3]
                total_length = 5 + length
                if len(self._rx_buffer) < total_length:
                    return
                frame = bytes(self._rx_buffer[:total_length])
                del self._rx_buffer[:total_length]
                response = self.parse_command_response(frame)
                if response is None:
                    self._frames_rejected += 1
                    continue
                self._command_response = response
                self._command_event.set()
                continue

            frame_length = 8
            if len(self._rx_buffer) >= 8 and self._rx_buffer[6:8] == self._frame_trailer:
                frame_length = 8
            elif len(self._rx_buffer) >= 10:
                frame_length = self._frame_length if self._frame_length == 10 else 8
            else:
                return
            frame = bytes(self._rx_buffer[:frame_length])
            if frame[-len(self._frame_trailer):] != self._frame_trailer:
                self._frames_rejected += 1
                del self._rx_buffer[:1]
                self._bytes_discarded += 1
                continue
            del self._rx_buffer[:frame_length]
            result = self.parse_frame(frame, self._frame_header, self._frame_trailer, gain=self.GAIN)
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
            channel = int(result.get("channel", 0))
            if channel in (1, 2):
                self._channel_values[channel] = value
                self._channel_scaled_values[channel] = scaled
                self._channel_timestamps[channel] = result["timestamp"]
            self._frames_parsed += 1
            self.valid_count = self._frames_parsed
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            try:
                callback(result)
            except Exception:
                logger.exception("ComPort callback failed")
        try:
            channel = int(result.get("channel", 0))
            if self.product_id == 2 and channel in (1, 2):
                detector_id = f"{self.name} CH{channel}"
            else:
                detector_id = self.name
            self.sample_received.emit(detector_id, result["timestamp"], scaled)
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
                if self.running and not self._stop_event.is_set():
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
                "channels": dict(self._channel_scaled_values),
                "channel_timestamps": dict(self._channel_timestamps),
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
