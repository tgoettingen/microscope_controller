import threading
import time
from typing import Optional, List
import struct

try:
    import serial
    import serial.tools.list_ports
except Exception:
    serial = None

from PyQt6.QtCore import QObject, pyqtSignal

from .base import Detector

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

    # emits: detector_id(str), timestamp(float), value(float)
    sample_received = pyqtSignal(str, float, float)
    error = pyqtSignal(str)

    def __init__(self, port: str, baudrate: int = 115200, read_timeout: float = 0.1, sample_format: str = 'int24', name: str | None = None):
        # name is used by Device base class; default to port string when not provided
        nm = name if name is not None else port
        super().__init__(nm)
        self.port = port
        self.baudrate = baudrate
        self.read_timeout = read_timeout
        # sample_format: 'int16' for binary signed 16-bit little-endian,
        # 'int24' for the 24-bit sensor format (3 bytes, MSB sign, 23-bit magnitude),
        # or 'ascii' for newline-delimited ASCII floats
        self.sample_format = sample_format
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._serial = None
        # last sampled value (raw) and timestamp
        self._last_value: float | None = None
        self._last_timestamp: float | None = None
        # scaling/offset for display units
        self._scale = 1.0
        self._offset = 0.0

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
        # Reader supports two modes:
        # - 'int16': read raw 2-byte signed little-endian integers continuously
        # - 'int24': read raw 3-byte signed-magnitude big-endian samples and convert to voltage
        # - 'ascii': read newline-delimited ASCII floats
        if self.sample_format == 'int16':
            while self._running and self._serial is not None:
                try:
                    data = self._serial.read(2)
                except Exception as e:
                    self.error.emit(f'Read error: {e}')
                    break
                if not data or len(data) < 2:
                    continue
                try:
                    # little-endian signed short
                    val = struct.unpack('<h', data)[0]
                except Exception:
                    continue
                ts = time.time()
                self._last_value = float(val)
                self._last_timestamp = ts
                self.sample_received.emit(self.port, ts, float(val))
        elif self.sample_format == 'int24':
            def _decode_24bit_to_voltage(data: bytes) -> float:
                """Decode 3-byte 24-bit sample to voltage.

                Format: 24 bits total, MSB (first bit) is sign (0=+, 1=-).
                Remaining 23 bits are magnitude; value = (magnitude / 2^23) * 3.3 * 400.
                Data expected in big-endian byte order.
                """
                if len(data) != 3:
                    raise ValueError('data must be exactly 3 bytes')
                val = (data[0] << 16) | (data[1] << 8) | data[2]
                sign = (val >> 23) & 0x1
                magnitude = val & 0x7FFFFF
                fraction = magnitude / float(2 ** 23)
                voltage = fraction * 3.3 / 400.0
                return -voltage if sign == 1 else voltage

            while self._running and self._serial is not None:
                try:
                    data = self._serial.read(3)
                except Exception as e:
                    self.error.emit(f'Read error: {e}')
                    break
                if not data or len(data) < 3:
                    continue
                try:
                    voltage = _decode_24bit_to_voltage(data)
                except Exception:
                    continue
                ts = time.time()
                self._last_value = float(voltage)
                self._last_timestamp = ts
                self.sample_received.emit(self.port, ts, float(voltage))
        else:
            while self._running and self._serial is not None:
                try:
                    line = self._serial.readline()
                except Exception as e:
                    self.error.emit(f'Read error: {e}')
                    break
                if not line:
                    continue
                ts = time.time()
                val = parse_ascii_line(line)
                if val is None:
                    continue
                self._last_value = float(val)
                self._last_timestamp = ts
                self.sample_received.emit(self.port, ts, val)

    def read_value(self) -> float:
        # Reader supports two modes:
        # - 'int16': read raw 2-byte signed little-endian integers continuously
        # - 'int24': read raw 3-byte signed-magnitude big-endian samples and convert to voltage
        # - 'ascii': read newline-delimited ASCII floats
        if self.sample_format == 'int16':
            return self._last_value if self._last_value is not None else 0.0
        elif self.sample_format == 'int24':
            def _decode_24bit_to_voltage(data: bytes) -> float:
                """Decode 3-byte 24-bit sample to voltage.

                Format: 24 bits total, MSB (first bit) is sign (0=+, 1=-).
                Remaining 23 bits are magnitude; value = (magnitude / 2^23) * 3.3 * 400.
                Data expected in big-endian byte order.
                """
                if len(data) != 3:
                    raise ValueError('data must be exactly 3 bytes')
                val = (data[0] << 16) | (data[1] << 8) | data[2]
                sign = (val >> 23) & 0x1
                magnitude = val & 0x7FFFFF
                fraction = magnitude / float(2 ** 23)
                voltage = fraction * 3.3 / 400.0
                return -voltage if sign == 1 else voltage

            while self._running and self._serial is not None:
                try:
                    data = self._serial.read(3)
                except Exception as e:
                    self.error.emit(f'Read error: {e}')
                    break
                if not data or len(data) < 3:
                    continue
                try:
                    voltage = _decode_24bit_to_voltage(data)
                    return voltage
                except Exception:
                    print(f"Failed to decode 24-bit sample: {data}")
                    return
                
        else:
            print("Reading ASCII line...")
            while self._running and self._serial is not None:
                try:
                    line = self._serial.readline()
                except Exception as e:
                    self.error.emit(f'Read error: {e}')
                    break
                if not line:
                    continue
                val = parse_ascii_line(line)
                if val is None:
                    continue
                return val


    # ---- Device/Detector compatibility methods ----
    def connect(self) -> None:
        """Open the underlying serial connection and start background reader."""
        self.start()

    def disconnect(self) -> None:
        """Stop reader and close serial port."""
        self.stop()

    def read_value(self) -> float:
        """Return the most recent sample, scaled. Non-blocking; returns 0.0 if no sample yet."""
        if self._last_value is None:
            return 0.0
        return float(self._last_value) * self._scale + float(self._offset)

    def set_scale(self, scale: float, offset: float = 0.0) -> None:
        self._scale = float(scale)
        self._offset = float(offset)

    def get_capabilities(self):
        return {"sample_format": self.sample_format, "port": self.port}

    def reset(self) -> None:
        self._last_value = None
        self._last_timestamp = None

    @staticmethod
    def list_ports() -> List[str]:
        if serial is None:
            return []
        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append(p.device)
        return ports
