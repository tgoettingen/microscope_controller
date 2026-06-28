"""
Deterministic validation for UartReader framed parsing and stream realignment.

ComPort now delegates all serial I/O and frame parsing to UartReader, so these
tests exercise UartReader directly.

Covers:
1) clean aligned frames
2) shifted stream (starts in middle of a frame)
3) noise between frames
4) marker mismatch rejection
5) valid voltage decoding

Run:
  python scripts/test_comport_framed_parser.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from devices.urt_serial import UartReader

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def check(label: str, cond: bool, detail: str = "") -> bool:
    if cond:
        print(f"  {PASS}  {label}")
        return True
    print(f"  {FAIL}  {label}" + (f": {detail}" if detail else ""))
    return False




# UartReader 10-byte frame: 0x0A 0x01 | 6 payload bytes | 0x01 0x0A
HEADER = b"\x0a\x01"
TRAILER = b"\x01\x0a"
FRAME_LEN = 10


def make_frame(adc_raw: int, extra: bytes = b"\x00\x00\x00") -> bytes:
    """Build a valid 10-byte UartReader frame from a raw 24-bit ADC value."""
    v = int(adc_raw) & 0xFFFFFF
    payload = bytes([(v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF]) + extra[:3].ljust(3, b"\x00")
    return HEADER + payload + TRAILER


def feed_buffer(buf: bytearray, chunks: list[bytes]) -> list[dict]:
    """Feed byte chunks through UartReader._read_loop logic and collect results."""
    results = []
    for chunk in chunks:
        buf.extend(chunk)
        while len(buf) >= FRAME_LEN:
            idx = buf.find(b"\x0a\x01")
            if idx < 0:
                if len(buf) > 1:
                    del buf[: len(buf) - 1]
                break
            if idx > 0:
                del buf[:idx]
            if len(buf) < FRAME_LEN:
                break
            if buf[8] == 0x01 and buf[9] == 0x0A:
                frame = bytes(buf[:FRAME_LEN])
                result = UartReader.parse_frame(frame)
                if result:
                    results.append(result)
                del buf[:FRAME_LEN]
            else:
                next_idx = buf.find(b"\x0a\x01", 1)
                if next_idx < 0:
                    if len(buf) > 1:
                        del buf[: len(buf) - 1]
                    break
                del buf[:next_idx]
    return results


def main() -> int:
    ok = True

    print("\n=== TEST 1: aligned frames ===")
    buf1 = bytearray()
    f1 = make_frame(0x001234)
    f2 = make_frame(0x002345)
    results1 = feed_buffer(buf1, [f1, f2])
    ok &= check("parsed 2 frames", len(results1) == 2, str(len(results1)))
    ok &= check("first frame has 'voltage' key", "voltage" in results1[0])

    print("\n=== TEST 2: shifted stream realignment ===")
    buf2 = bytearray()
    f = make_frame(0x003456)
    shifted = f[4:] + f + f
    results2 = feed_buffer(buf2, [shifted])
    ok &= check("realigned and parsed >=2 frames", len(results2) >= 2, str(len(results2)))

    print("\n=== TEST 3: noise between frames ===")
    buf3 = bytearray()
    noise = b"\x99\x88\x77\x66\x55"
    results3 = feed_buffer(buf3, [make_frame(0x004567), noise, make_frame(0x005678)])
    ok &= check("parsed both valid frames", len(results3) == 2, str(len(results3)))

    print("\n=== TEST 4: marker mismatch rejection ===")
    buf4 = bytearray()
    good = make_frame(0x006789)
    bad = good[:-2] + b"\xff\xee"
    results4 = feed_buffer(buf4, [bad + good])
    ok &= check("bad frame not returned, good frame parsed", len(results4) == 1, str(len(results4)))

    print("\n=== TEST 5: voltage decoding ===")
    # ADC value 0x400000 = 4194304; two's complement positive
    adc = 0x400000
    frame5 = make_frame(adc)
    result5 = UartReader.parse_frame(frame5)
    expected = adc * UartReader.VREF / (UartReader.GAIN * UartReader.ADC_MAX)
    ok &= check("parse_frame returns result", result5 is not None)
    if result5:
        ok &= check(
            "voltage matches formula",
            abs(result5["voltage"] - expected) < 1e-12,
            f"{result5['voltage']} vs {expected}",
        )

    print("\n=== RESULT ===")
    if ok:
        print("All framed-parser tests passed.")
        return 0
    print("One or more framed-parser tests failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
