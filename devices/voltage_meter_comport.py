import threading
import time
from collections import deque
from typing import Optional, List
import struct
import logging

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

from PyQt6.QtCore import QObject, pyqtSignal

from .base import Detector


logger = logging.getLogger(__name__)

def parse_ascii_line(line: bytes | str) -> Optional[float]:
    """Parse an ASCII text line into a float. Returns None on failure."""
    if isinstance(line, bytes):
        try:
            s = line.decode('utf-8', errors='ignore').strip()
        except Exception:
            return None
    else:
        s = str(line).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


class ComPort(Detector):
    """Simple threaded COM port reader that emits timestamped float samples.

    Signals:
        sample_received(timestamp: float, value: float)
        error(message: str)
    """

    # We can't safely mix PyQt QObject metaclass with ABCs across platforms,
    # so embed a small QObject emitter and forward its signals.
    class _Emitter(QObject):
        sample_received = pyqtSignal(str, float, float)
        error = pyqtSignal(str)

    _MODE_COMMANDS = {
        1: b'\x01\x07',  # 5-byte frame: 3-byte voltage + 2-byte temperature
        2: b'\x02\x0e',  # 3-byte frame: voltage only
        3: b'\x03\x09',  # ADC restart sequence on MCU side
    }

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
        # name is used by Device base class; default to port string when not provided
        nm = name if name is not None else port
        Detector.__init__(self, nm)
        # create internal QObject to host signals
        self._emitter = ComPort._Emitter()
        # expose signals on this object for existing consumers
        self.sample_received = self._emitter.sample_received
        self.error = self._emitter.error
        self.port = port
        self.baudrate = baudrate
        self.read_timeout = read_timeout
        # sample_format: 'int16' for binary signed 16-bit little-endian,
        # 'int24' for the 24-bit sensor format (3 bytes, MSB sign, 23-bit magnitude),
        # or 'ascii' for newline-delimited ASCII floats
        self.sample_format = sample_format
        # Device stream mode:
        # 1 => 5-byte frame (voltage + temperature)
        # 2 => 3-byte frame (voltage only)
        # 3 => ADC restart mode command
        self.mode = self._resolve_initial_mode(mode, sample_format)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._serial = None
        self._lock = threading.Lock()
        # last sampled value (raw) and timestamp
        self._last_value: float | None = None
        self._last_timestamp: float | None = None
        self._last_temperature: float | None = None
        self._last_scaled_value: float | None = None
        # scaling/offset for display units
        self._scale = 1.0
        self._offset = 0.0
        self.reader_hz = max(1.0, float(reader_hz))
        self._frame_length = max(5, int(frame_length))
        self._frame_header = self._normalize_marker(frame_header, default=b'\x0a\x01')
        self._frame_trailer = self._normalize_marker(frame_trailer, default=b'\x01\x0a')
        if len(self._frame_header) != 2 or len(self._frame_trailer) != 2:
            raise ValueError('frame_header and frame_trailer must each be exactly 2 bytes')
        self._payload_length = self._frame_length - len(self._frame_header) - len(self._frame_trailer)
        if self._payload_length not in (5, 6):
            raise ValueError(
                f'Expected 5 or 6 payload bytes (3 voltage + 2 temperature [+ optional checksum]), got {self._payload_length}'
            )
        self._rx_buffer = bytearray()
        self._ring_buffer_size = max(1, int(ring_buffer_size))
        self._ring_buffer = deque(maxlen=self._ring_buffer_size)
        pol = str(overflow_policy or 'overwrite').strip().lower()
        self._overflow_policy = pol if pol in ('overwrite', 'reject') else 'overwrite'
        self._overflow_count = 0
        self._frames_parsed = 0
        self._frames_rejected = 0
        self._bytes_discarded = 0
        self._last_overflow_error_ts = 0.0
        self.start()

    @staticmethod
    def _normalize_marker(marker: bytes | str | list[int] | None, default: bytes) -> bytes:
        if marker is None:
            return bytes(default)
        if isinstance(marker, bytes):
            return bytes(marker)
        if isinstance(marker, str):
            text = marker.strip().replace(' ', '').replace('-', '')
            if text.lower().startswith('0x') and len(text) > 2:
                text = text[2:]
            if not text:
                return bytes(default)
            try:
                return bytes.fromhex(text)
            except ValueError:
                return marker.encode('latin1', errors='ignore')
        if isinstance(marker, list):
            try:
                return bytes(int(v) & 0xFF for v in marker)
            except Exception:
                return bytes(default)
        return bytes(default)

    @staticmethod
    def _resolve_initial_mode(mode: int | None, sample_format: str) -> int:
        if mode in (1, 2, 3):
            return int(mode)
        sf = (sample_format or '').lower()
        if sf in ('mode1', 'int24_temp', 'int24_5b', '5bytes', '5byte'):
            return 1
        if sf in ('mode2', 'int24', '3bytes', '3byte'):
            return 2
        return 2

    @staticmethod
    def _decode_24bit_to_voltage(data: bytes) -> float:
        """Decode 3-byte signed-magnitude sample to voltage."""
        if len(data) != 3:
            raise ValueError('data must be exactly 3 bytes')
        val = (data[0] << 16) | (data[1] << 8) | data[2]
        sign = (val >> 23) & 0x1
        magnitude = val & 0x7FFFFF
        fraction = magnitude / float(2 ** 23)
        voltage = fraction * 2.048 / 500.0
        return -voltage if sign == 1 else voltage

    @staticmethod
    def _decode_temperature_16bit(data: bytes) -> float:
        """Decode 2-byte temperature payload as signed little-endian raw value."""
        if len(data) != 2:
            raise ValueError('temperature data must be exactly 2 bytes')
        return float(struct.unpack('<h', data)[0])

    def _send_mode_command(self, mode: int) -> None:
        if mode not in self._MODE_COMMANDS:
            raise ValueError(f'Unsupported mode: {mode}. Expected one of: 1, 2, 3')
        if self._serial is None:
            raise RuntimeError('Serial port is not open')
        cmd = self._MODE_COMMANDS[mode]
        self._serial.write(cmd)

    def _report_error(self, message: str) -> float:
        try:
            self.error.emit(message)
        except Exception:
            pass
        try:
            logger.error("%s", message)
        except Exception:
            pass
        return float('nan')

    def _report_overflow_error(self, message: str) -> None:
        now = time.time()
        if now - self._last_overflow_error_ts < 1.0:
            return
        self._last_overflow_error_ts = now
        self._report_error(message)

    def set_mode(self, mode: int, wait_for_restart: bool = True) -> None:
        """Set device acquisition mode.

        Mode 1: command 0x01 0x07, 5-byte frame (3-byte voltage + 2-byte temp)
        Mode 2: command 0x02 0x0E, 3-byte frame (voltage only)
        Mode 3: command 0x03 0x09, MCU restarts ADC (~2 s)
        """
        if mode not in self._MODE_COMMANDS:
            raise ValueError(f'Unsupported mode: {mode}. Expected one of: 1, 2, 3')
        self.mode = int(mode)
        if self._serial is not None:
            self._send_mode_command(mode)
        if self.mode == 3 and wait_for_restart:
            # MCU side resets ADC and restarts it after about 2 seconds.
            time.sleep(2.0)

    def start(self) -> None:
        if serial is None:
            self.error.emit('pyserial not available; install pyserial')
            return
        if self._running:
            return

        try:
            # Open serial port; for binary int16 sampling we still rely on timeout
            # to avoid blocking shutdown.
            self._serial = serial.Serial(self.port, self.baudrate, timeout=self.read_timeout)
            time.sleep(0.1)  # wait for port to stabilize
            # Reset the voltage meter on connect: mode 3 restarts the ADC on the
            # MCU side. Issue the reset command directly (without overwriting the
            # intended stream mode), then apply the configured mode.
            intended_mode = self.mode
            self._send_mode_command(3)
            time.sleep(2.0)  # MCU resets and restarts the ADC after about 2 seconds
            self.set_mode(intended_mode)

        except Exception as e:
            self.error.emit(f'Failed to open serial port {self.port}: {e}')
            return

        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        # mark connected/started
        try:
            self.connected = True
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._serial is not None and self._serial.is_open:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None
        try:
            self.connected = False
        except Exception:
            pass

    def is_open(self) -> bool:
        return self._serial is not None and getattr(self._serial, 'is_open', False)

    def _reader_loop(self) -> None:
        period = 1.0 / self.reader_hz
        while self._running:
            t0 = time.time()
            ser = self._serial
            if ser is None:
                break
            try:
                available = int(getattr(ser, 'in_waiting', 0) or 0)
                read_count = available if available > 0 else self._frame_length
                chunk = ser.read(read_count)
            except Exception as e:
                self._report_error(f'Read error on {self.port}: {e}')
                chunk = b''

            if chunk:
                self._rx_buffer.extend(chunk)
                self._consume_rx_frames()

            dt = time.time() - t0
            sleep_s = period - dt
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _consume_rx_frames(self) -> None:
        header = self._frame_header
        trailer = self._frame_trailer
        frame_len = self._frame_length

        while len(self._rx_buffer) >= frame_len:
            idx = self._rx_buffer.find(header)
            if idx < 0:
                keep = max(0, len(header) - 1)
                drop_n = max(0, len(self._rx_buffer) - keep)
                if drop_n:
                    del self._rx_buffer[:drop_n]
                    self._bytes_discarded += drop_n
                return
            if idx > 0:
                del self._rx_buffer[:idx]
                self._bytes_discarded += idx
            if len(self._rx_buffer) < frame_len:
                return

            frame = bytes(self._rx_buffer[:frame_len])
            if frame[-len(trailer):] != trailer:
                del self._rx_buffer[0]
                self._bytes_discarded += 1
                self._frames_rejected += 1
                continue

            del self._rx_buffer[:frame_len]
            payload = frame[len(header):frame_len - len(trailer)]
            self._handle_payload(payload)

    def _handle_payload(self, payload: bytes) -> None:
        if len(payload) not in (5, 6):
            self._frames_rejected += 1
            self._report_error(
                f'Unexpected payload length on {self.port}: {len(payload)} (expected 5 or 6)'
            )
            return
        try:
            voltage = self._decode_24bit_to_voltage(payload[:3])
            temperature = self._decode_temperature_16bit(payload[3:5])
        except Exception as e:
            self._frames_rejected += 1
            self._report_error(f'Failed to decode framed payload on {self.port}: {e}')
            return

        ts = time.time()
        try:
            scaled_voltage = float(self._scale) * float(voltage) + float(self._offset)
        except Exception:
            scaled_voltage = float(voltage)

        with self._lock:
            self._last_value = float(voltage)
            self._last_scaled_value = float(scaled_voltage)
            self._last_temperature = float(temperature)
            self._last_timestamp = ts

            full = len(self._ring_buffer) >= self._ring_buffer_size
            if full and self._overflow_policy == 'reject':
                self._overflow_count += 1
                self._report_overflow_error(
                    f'Ring buffer overflow on {self.port}; rejecting sample (size={self._ring_buffer_size})'
                )
            else:
                if full:
                    self._overflow_count += 1
                    self._report_overflow_error(
                        f'Ring buffer overflow on {self.port}; overwriting oldest sample (size={self._ring_buffer_size})'
                    )
                self._ring_buffer.append((ts, float(scaled_voltage), float(temperature)))

            self._frames_parsed += 1

        self.sample_received.emit(self.port, ts, float(scaled_voltage))

    def read_value(self) -> float | tuple[float, float]:
        if self._serial is None:
            return self._report_error('Serial port is not open for ComPort detector')

        with self._lock:
            if self._last_scaled_value is None:
                return float('nan')
            if self.mode == 1:
                if self._last_temperature is None:
                    return float('nan')
                return float(self._last_scaled_value), float(self._last_temperature)
            return float(self._last_scaled_value)


    # ---- Device/Detector compatibility methods ----
    def connect(self) -> None:
        """Open the underlying serial connection and start background reader."""
        self.start()

    def disconnect(self) -> None:
        """Stop reader and close serial port."""
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
        with self._lock:
            return {
                'frames_parsed': int(self._frames_parsed),
                'frames_rejected': int(self._frames_rejected),
                'bytes_discarded': int(self._bytes_discarded),
                'overflow_count': int(self._overflow_count),
                'ring_buffer_size': int(self._ring_buffer_size),
                'ring_buffer_used': int(len(self._ring_buffer)),
                'reader_hz': float(self.reader_hz),
                'frame_length': int(self._frame_length),
                'frame_header': self._frame_header.hex(),
                'frame_trailer': self._frame_trailer.hex(),
                'overflow_policy': self._overflow_policy,
            }

    def get_capabilities(self):
        return {
            "sample_format": self.sample_format,
            "mode": self.mode,
            "supported_modes": [1, 2, 3],
            "port": self.port,
            "frame_length": self._frame_length,
            "frame_header": self._frame_header.hex(),
            "frame_trailer": self._frame_trailer.hex(),
            "reader_hz": self.reader_hz,
            "ring_buffer_size": self._ring_buffer_size,
        }

    def reset(self) -> None:
        with self._lock:
            self._last_value = None
            self._last_scaled_value = None
            self._last_timestamp = None
            self._last_temperature = None
            self._rx_buffer.clear()
            self._ring_buffer.clear()
            self._overflow_count = 0
            self._frames_parsed = 0
            self._frames_rejected = 0
            self._bytes_discarded = 0

    @staticmethod
    def list_ports() -> List[str]:
        if serial is None:
            return []
        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append(p.device)
        return ports
