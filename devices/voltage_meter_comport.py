import threading
import time
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

    def __init__(self, port: str, baudrate: int = 115200, read_timeout: float = 0.1, sample_format: str = 'int24', mode: int | None = None, name: str | None = None):
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
        # last sampled value (raw) and timestamp
        self._last_value: float | None = None
        self._last_timestamp: float | None = None
        self._last_temperature: float | None = None
        # scaling/offset for display units
        self._scale = 1.0
        self._offset = 0.0
        self.start()

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
        # self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        # self._thread.start()
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
        while self._running and self._serial is not None:
            sample = self.read_value()
            if self._serial is None:
                break
            ts = time.time()
            if isinstance(sample, tuple):
                val = float(sample[0])
                try:
                    self._last_temperature = float(sample[1])
                except Exception:
                    pass
            else:
                val = float(sample)
            self._last_value = float(val)
            self._last_timestamp = ts
            self.sample_received.emit(self.port, ts, float(val))

    def read_value(self) -> float | tuple[float, float]:
        if self._serial is None:
            return self._report_error('Serial port is not open for ComPort detector')

        # Mode 1: read 5-byte frame (voltage + temperature)
        if self.mode == 1:
            while self._serial is not None:
                try:
                    data = self._serial.read(5)
                except Exception as e:
                    return self._report_error(f'Read error in mode 1: {e}')
                if not data or len(data) < 5:
                    return self._report_error(
                        f'Invalid mode 1 frame on {self.port}: expected 5 bytes, got {len(data) if data else 0}'
                    )
                try:
                    voltage = self._decode_24bit_to_voltage(data[:3])
                    self._last_temperature = self._decode_temperature_16bit(data[3:5])
                    self._last_value = float(voltage)
                    self._last_timestamp = time.time()
                    try:
                        scaled_voltage = float(self._scale) * float(voltage) + float(self._offset)
                    except Exception:
                        scaled_voltage = float(voltage)
                    return scaled_voltage, float(self._last_temperature)
                except Exception as e:
                    return self._report_error(f'Failed to decode mode 1 frame on {self.port}: {e}')

        # Mode 2: read 3-byte frame (voltage)
        if self.mode == 2:
            while self._serial is not None:
                try:
                    data = self._serial.read(3)
                except Exception as e:
                    return self._report_error(f'Read error in mode 2: {e}')
                if not data or len(data) < 3:
                    return self._report_error(
                        f'Invalid mode 2 frame on {self.port}: expected 3 bytes, got {len(data) if data else 0}'
                    )
                try:
                    voltage = self._decode_24bit_to_voltage(data)
                    self._last_value = float(voltage)
                    self._last_timestamp = time.time()
                    try:
                        return float(self._scale) * float(voltage) + float(self._offset)
                    except Exception:
                        return float(voltage)
                except Exception as e:
                    return self._report_error(f'Failed to decode mode 2 frame on {self.port}: {e}')

        # Mode 3 performs ADC reset on MCU; no frame contract defined for read.
        if self.mode == 3:
            raw = self._last_value if self._last_value is not None else 0.0
            try:
                return float(self._scale) * float(raw) + float(self._offset)
            except Exception:
                return float(raw)

        # Legacy ASCII fallback for backward compatibility.
        while self._serial is not None:
            try:
                line = self._serial.readline()
            except Exception as e:
                return self._report_error(f'Read error in ASCII mode: {e}')
            if not line:
                return self._report_error(f'No ASCII sample received on {self.port}')
            val = parse_ascii_line(line)
            if val is None:
                return self._report_error(f'Invalid ASCII sample on {self.port}: {line!r}')
            self._last_value = float(val)
            self._last_timestamp = time.time()
            try:
                return float(self._scale) * float(val) + float(self._offset)
            except Exception:
                return float(val)

        return self._report_error('Serial reader stopped unexpectedly')


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
        return self._last_temperature

    def get_capabilities(self):
        return {
            "sample_format": self.sample_format,
            "mode": self.mode,
            "supported_modes": [1, 2, 3],
            "port": self.port,
        }

    def reset(self) -> None:
        self._last_value = None
        self._last_timestamp = None
        self._last_temperature = None

    @staticmethod
    def list_ports() -> List[str]:
        if serial is None:
            return []
        ports = []
        for p in serial.tools.list_ports.comports():
            ports.append(p.device)
        return ports
