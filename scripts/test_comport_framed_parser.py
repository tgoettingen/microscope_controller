"""
Deterministic validation for ComPort framed parsing and stream realignment.

Covers:
1) clean aligned frames
2) shifted stream (starts in middle of a frame)
3) noise between frames
4) marker mismatch rejection
5) ring buffer overflow signaling

Run:
  python scripts/test_comport_framed_parser.py
"""

import sys
import pathlib
import types

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Allow importing ComPort in headless/non-GUI test environments.
try:
    from PyQt6.QtCore import QObject, pyqtSignal  # noqa: F401
except Exception:
    pyqt6_mod = types.ModuleType("PyQt6")
    qtcore_mod = types.ModuleType("PyQt6.QtCore")

    class _QObject:
        pass

    def _pyqt_signal(*_args, **_kwargs):
        class _Signal:
            def emit(self, *_a, **_k):
                return None

        return _Signal()

    qtcore_mod.QObject = _QObject
    qtcore_mod.pyqtSignal = _pyqt_signal
    pyqt6_mod.QtCore = qtcore_mod
    sys.modules["PyQt6"] = pyqt6_mod
    sys.modules["PyQt6.QtCore"] = qtcore_mod

from devices.voltage_meter_comport import ComPort

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label: str, cond: bool, detail: str = "") -> bool:
    if cond:
        print(f"  {PASS}  {label}")
        return True
    print(f"  {FAIL}  {label}" + (f": {detail}" if detail else ""))
    return False


def encode_voltage_triplet(raw24: int) -> bytes:
    v = int(raw24) & 0xFFFFFF
    return bytes([(v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF])


def encode_temp_le(raw16: int) -> bytes:
    v = int(raw16) & 0xFFFF
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


def make_frame(raw24: int, telemetry: bytes, header: bytes, trailer: bytes) -> bytes:
    if len(telemetry) != 3:
        raise ValueError("production telemetry must contain three bytes")
    return header + encode_voltage_triplet(raw24) + telemetry + trailer


def make_test_instance(ring_buffer_size: int = 8, overflow_policy: str = "overwrite") -> ComPort:
    return ComPort(
        port="TEST",
        name="TEST",
        ring_buffer_size=ring_buffer_size,
        frame_length=10,
        overflow_policy=overflow_policy,
    )


def append_chunks(inst: ComPort, chunks: list[bytes]) -> None:
    for c in chunks:
        inst._rx_buffer.extend(c)
        inst._consume_rx_frames()


def main() -> int:
    print("\n=== TEST 1: aligned frames ===")
    c1 = make_test_instance(ring_buffer_size=16)
    f1 = make_frame(0x001234, b"\x00\x19\x00", c1._frame_header, c1._frame_trailer)
    f2 = make_frame(0x002345, b"\x00\x1a\x00", c1._frame_header, c1._frame_trailer)
    append_chunks(c1, [f1, f2])
    ok = True
    ok &= check("parsed 2 frames", c1._frames_parsed == 2, str(c1._frames_parsed))
    ok &= check("buffer has 2 samples", len(c1.get_recent_samples()) == 2)

    print("\n=== TEST 2: shifted stream realignment ===")
    c2 = make_test_instance(ring_buffer_size=16)
    f = make_frame(0x003456, b"\x00\x1f\x00", c2._frame_header, c2._frame_trailer)
    shifted = f[4:] + f + f
    append_chunks(c2, [shifted])
    ok &= check("realigned and parsed >=2 frames", c2._frames_parsed >= 2, str(c2._frames_parsed))
    ok &= check("discarded bytes > 0", c2._bytes_discarded > 0, str(c2._bytes_discarded))

    print("\n=== TEST 3: noise between frames ===")
    c3 = make_test_instance(ring_buffer_size=16)
    noise = b"\x99\x88\x77\x66\x55"
    append_chunks(
        c3,
        [
            make_frame(0x004567, b"\x00\x29\x00", c3._frame_header, c3._frame_trailer),
            noise,
            make_frame(0x005678, b"\x00\x2a\x00", c3._frame_header, c3._frame_trailer),
        ],
    )
    ok &= check("parsed both valid frames", c3._frames_parsed == 2, str(c3._frames_parsed))
    ok &= check("noise got discarded", c3._bytes_discarded >= len(noise), str(c3._bytes_discarded))

    print("\n=== TEST 4: marker mismatch rejection ===")
    c4 = make_test_instance(ring_buffer_size=16)
    good = make_frame(0x006789, b"\x00\x32\x00", c4._frame_header, c4._frame_trailer)
    bad = good[:-2] + b"\xff\xee"
    append_chunks(c4, [bad + good])
    ok &= check("reject counter incremented", c4._frames_rejected >= 1, str(c4._frames_rejected))
    ok &= check("still parsed the trailing good frame", c4._frames_parsed >= 1, str(c4._frames_parsed))

    print("\n=== TEST 5: overflow signaling ===")
    c5 = make_test_instance(ring_buffer_size=3, overflow_policy="overwrite")
    frames = [make_frame(0x000100 + i, bytes((0, 10 + i, 0)), c5._frame_header, c5._frame_trailer) for i in range(8)]
    append_chunks(c5, frames)
    ok &= check("buffer capped at configured size", len(c5.get_recent_samples()) == 3)
    ok &= check("overflow counter incremented", c5._overflow_count > 0, str(c5._overflow_count))

    print("\n=== RESULT ===")
    if ok:
        print("All framed-parser tests passed.")
        return 0
    print("One or more framed-parser tests failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
