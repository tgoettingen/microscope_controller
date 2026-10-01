"""Live voltage and serial-frame diagnostic for the ComPort detector.

Example:
  python scripts/test_comport_cli.py
  python scripts/test_comport_cli.py --duration 30 --max-pp-nv 500
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from devices.voltage_meter_comport import ComPort


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read and evaluate live ComPort voltage samples.")
    parser.add_argument("--port", default="/dev/tty.usbserial-210", help="serial device path")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=10.0, help="capture duration in seconds")
    parser.add_argument("--reader-hz", type=float, default=1000.0, help="host serial polling rate")
    parser.add_argument("--read-timeout", type=float, default=0.1)
    parser.add_argument("--max-abs-mean-nv", type=float, default=300.0,
                        help="pass limit for absolute mean voltage in nV")
    parser.add_argument("--max-pp-nv", type=float, default=300.0,
                        help="pass limit for peak-to-peak voltage fluctuation in nV")
    parser.add_argument("--output-prefix", type=Path,
                        help="output path prefix for CSV, PNG, and raw-frame BIN (default: timestamped reports file)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.duration <= 0:
        print("duration must be greater than zero", file=sys.stderr)
        return 2

    samples: list[tuple[float, int, float, str, str]] = []
    sample_lock = threading.Lock()
    detector = ComPort(
        port=args.port,
        baudrate=args.baudrate,
        read_timeout=args.read_timeout,
        reader_hz=args.reader_hz,
        name="comport-cli-test",
    )

    def record_sample(sample: dict) -> None:
        value_nv = float(sample["scaled_value"]) * 1e9
        with sample_lock:
            samples.append((
                float(sample["timestamp"]),
                int(sample["adc_raw"]),
                value_nv,
                bytes(sample["raw_bytes"]).hex(),
                bytes(sample["middle"]).hex(),
            ))

    detector.add_callback(record_sample)
    print(f"Connecting to {args.port} at {args.baudrate} baud; capture {args.duration:g} s")
    try:
        detector.start()
        started = time.monotonic()
        deadline = started + args.duration
        while time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        print("Capture interrupted")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        detector.disconnect()

    elapsed = max(time.monotonic() - started, 1e-9)
    with sample_lock:
        records = list(samples)
    stats = detector.get_stats()
    values = [record[2] for record in records]
    frame_checks = []
    conversion_mismatches = 0
    for _timestamp, adc_raw, voltage_nv, raw_frame_hex, _auxiliary_hex in records:
        frame = bytes.fromhex(raw_frame_hex)
        valid_markers = (
            len(frame) == detector.DEFAULT_FRAME_LENGTH
            and frame[:2] == detector.DEFAULT_FRAME_HEADER
            and frame[-2:] == detector.DEFAULT_FRAME_TRAILER
        )
        valid_checksum = len(frame) == detector.DEFAULT_FRAME_LENGTH and (sum(frame[2:7]) & 0xFF) == frame[7]
        decoded_adc = int.from_bytes(frame[2:5], byteorder="big", signed=True) if valid_markers else None
        decoded_voltage_nv = (
            decoded_adc * detector.VREF / (detector.GAIN * detector.ADC_MAX) * 1e9
            if decoded_adc is not None
            else None
        )
        if decoded_adc != adc_raw or decoded_voltage_nv is None or abs(decoded_voltage_nv - voltage_nv) > 1e-9:
            conversion_mismatches += 1
        frame_checks.append((valid_markers, valid_checksum))

    print(f"\nPort: {args.port}")
    print(f"Elapsed: {elapsed:.3f} s")
    print(f"Valid frames: {len(values)} ({len(values) / elapsed:.1f} samples/s)")
    print(
        "Frame diagnostics: "
        f"rejected={stats['rejected']}, discarded_bytes={stats['discarded']}, "
        f"reader_errors={stats['error']}"
    )

    if not values:
        print("FAIL: no valid frames received")
        return 1

    mean_nv = statistics.fmean(values)
    minimum_nv = min(values)
    maximum_nv = max(values)
    peak_to_peak_nv = maximum_nv - minimum_nv
    stddev_nv = statistics.pstdev(values) if len(values) > 1 else 0.0
    mean_ok = abs(mean_nv) <= args.max_abs_mean_nv
    fluctuation_ok = peak_to_peak_nv <= args.max_pp_nv

    output_prefix = args.output_prefix
    if output_prefix is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = Path("reports") / f"comport_test_{timestamp}"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    png_path = output_prefix.with_suffix(".png")
    raw_path = output_prefix.with_suffix(".bin")
    first_timestamp = records[0][0]
    try:
        raw_path.write_bytes(b"".join(bytes.fromhex(record[3]) for record in records))
        with csv_path.open("w", newline="", encoding="utf-8") as data_file:
            writer = csv.writer(data_file)
            writer.writerow((
                "elapsed_s", "timestamp_unix_s", "adc_raw", "voltage_nv",
                "raw_frame_hex", "auxiliary_bytes_hex",
            ))
            for sample_timestamp, adc_raw, voltage_nv, raw_frame_hex, auxiliary_hex in records:
                writer.writerow((
                    sample_timestamp - first_timestamp,
                    sample_timestamp,
                    adc_raw,
                    voltage_nv,
                    raw_frame_hex,
                    auxiliary_hex,
                ))

        import pyqtgraph as pg
        import pyqtgraph.exporters
        from PyQt6.QtCore import Qt
        from PyQt6.QtWidgets import QApplication

        application = QApplication.instance() or QApplication([])
        elapsed_samples = [record[0] - first_timestamp for record in records]
        plot = pg.PlotWidget()
        plot.resize(1000, 550)
        plot.setBackground("w")
        plot.setTitle(f"ComPort voltage ({len(values)} samples)")
        plot.setLabel("bottom", "Elapsed time", units="s")
        plot.setLabel("left", "Voltage", units="nV")
        plot.showGrid(x=True, y=True, alpha=0.25)
        plot.plot(elapsed_samples, values, pen=pg.mkPen("#176b87", width=1))
        plot.addLine(y=mean_nv, pen=pg.mkPen("#d17c21", width=1, style=Qt.PenStyle.DashLine))
        x_end = max(elapsed_samples[-1], 1e-9)
        plot.setLimits(xMin=0.0, xMax=x_end)
        plot.enableAutoRange(axis="x", enable=False)
        plot.setXRange(0.0, x_end, padding=0.0)
        exporter = pg.exporters.ImageExporter(plot.plotItem)
        exporter.parameters()["width"] = 1600
        exporter.export(str(png_path))
        plot.close()
        del application
    except Exception as exc:
        print(f"ERROR: failed to save data or plot: {exc}", file=sys.stderr)
        return 2

    print(f"CSV data: {csv_path}")
    print(f"Raw frame bytes: {raw_path} ({raw_path.stat().st_size} bytes)")
    print(f"PNG plot: {png_path}")
    print(
        "Raw validation: "
        f"start/end markers={sum(markers for markers, _checksum in frame_checks)}/{len(frame_checks)}, "
        f"checksum={sum(checksum for _markers, checksum in frame_checks)}/{len(frame_checks)}, "
        f"ADC/voltage mismatches={conversion_mismatches}"
    )

    print(f"Voltage mean: {mean_nv:.2f} nV (limit |mean| <= {args.max_abs_mean_nv:g} nV)")
    print(f"Voltage min/max: {minimum_nv:.2f} / {maximum_nv:.2f} nV")
    print(f"Peak-to-peak fluctuation: {peak_to_peak_nv:.2f} nV (limit <= {args.max_pp_nv:g} nV)")
    print(f"Population standard deviation: {stddev_nv:.2f} nV")
    print(f"Result: {'PASS' if mean_ok and fluctuation_ok else 'FAIL'}")
    return 0 if mean_ok and fluctuation_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())