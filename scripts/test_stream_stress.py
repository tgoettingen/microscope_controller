"""
Stress-test StreamSaver under conditions that mirror the real app:
  1. Two detectors written concurrently from a worker thread
  2. Multiple sequential runs (stream_savers dict created/destroyed like mainwindow does)
  3. close() called from a DIFFERENT thread (like _on_stream_toggled background close)
  4. append_sample called AFTER close() (should not crash)
  5. _close_all_stream_savers() called while the worker thread is still writing
  6. Large scan: 10000 samples per detector
"""
import sys, time, threading, tempfile, pathlib, traceback
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import numpy as np
import h5py
from utils.stream_saver import StreamSaver

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"

errors = []

def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}" + (f": {detail}" if detail else ""))
        errors.append(label)

# ── helpers ──────────────────────────────────────────────────────────────────

def verify_h5(path: pathlib.Path, expected_rows: int, label: str):
    try:
        with h5py.File(path, "r") as f:
            arr = f["data"][:]
        check(f"{label}: file exists", True)
        check(f"{label}: shape rows={expected_rows}", arr.shape[0] == expected_rows,
              f"got {arr.shape[0]}")
        check(f"{label}: 5 columns", arr.shape[1] == 5, f"got {arr.shape[1]}")
        check(f"{label}: timestamps increase", bool(np.all(np.diff(arr[:, 0]) >= 0)))
    except Exception as e:
        check(f"{label}: open h5", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
print("\n=== TEST 1: basic single detector, below flush threshold ===")
tmp = pathlib.Path(tempfile.mkdtemp())
s = StreamSaver(tmp, "det_basic", flush_every=256)
for i in range(100):
    s.append_sample(float(i), float(i) * 0.1, {"X": i, "Y": i * 2})
s.close()
h5s = sorted(tmp.glob("*.h5"))
check("one .h5 created", len(h5s) == 1, str(h5s))
if h5s:
    verify_h5(h5s[0], 100, "basic")

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== TEST 2: single detector, multiple flushes ===")
tmp = pathlib.Path(tempfile.mkdtemp())
s = StreamSaver(tmp, "det_flush", flush_every=256)
for i in range(1000):
    s.append_sample(float(i), float(i) * 0.5, {"X": i * 0.1})
s.close()
h5s = sorted(tmp.glob("*.h5"))
check("one .h5 created", len(h5s) == 1)
if h5s:
    verify_h5(h5s[0], 1000, "multi-flush")

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== TEST 3: two detectors written concurrently from worker thread ===")
tmp = pathlib.Path(tempfile.mkdtemp())
N = 5000
stream_savers = {
    "detector1": StreamSaver(tmp, "detector1"),
    "detector2": StreamSaver(tmp, "detector2"),
}

def worker_run(savers: dict, n_samples: int):
    for i in range(n_samples):
        state = {"X": i * 0.5, "Y": i * 0.3}
        for det_id, saver in savers.items():
            saver.append_sample(time.time(), float(i), meta=state)
        time.sleep(0)  # yield

t = threading.Thread(target=worker_run, args=(stream_savers, N), daemon=True)
t.start()
t.join(timeout=30)
check("worker finished in time", not t.is_alive())

for det_id, saver in stream_savers.items():
    saver.close()
stream_savers.clear()

h5s = sorted(tmp.glob("*.h5"))
check("two .h5 files created", len(h5s) == 2, str([p.name for p in h5s]))
for h5 in h5s:
    verify_h5(h5, N, h5.stem)

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== TEST 4: close() called from background thread while worker still writes ===")
tmp = pathlib.Path(tempfile.mkdtemp())
stream_savers = {"detector1": StreamSaver(tmp, "detector1")}
stop_flag = threading.Event()
written = []

def writer():
    i = 0
    while not stop_flag.is_set():
        saver = stream_savers.get("detector1")
        if saver:
            try:
                saver.append_sample(float(i), float(i), {"X": i})
                written.append(i)
            except Exception:
                pass
        i += 1
        time.sleep(0.0001)

wt = threading.Thread(target=writer, daemon=True)
wt.start()
time.sleep(0.3)  # let 3000+ samples accumulate

# Simulate _on_stream_toggled: pop saver and close in background thread
saver = stream_savers.pop("detector1", None)
close_done = threading.Event()

def bg_close():
    try:
        saver.close()
    except Exception as e:
        errors.append(f"TEST4 bg_close exception: {e}")
    close_done.set()

threading.Thread(target=bg_close, daemon=True).start()

stop_flag.set()
close_done.wait(timeout=10)
wt.join(timeout=5)

check("background close completed", close_done.is_set())
h5s = sorted(tmp.glob("*.h5"))
check("TEST4: .h5 file created", len(h5s) == 1)
if h5s:
    with h5py.File(h5s[0], "r") as f:
        rows = f["data"].shape[0]
    check(f"TEST4: rows > 0 (got {rows})", rows > 0)
    print(f"       written={len(written)} saved={rows}")

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== TEST 5: multiple sequential runs (simulate 2 complete multiaxis runs) ===")
tmp = pathlib.Path(tempfile.mkdtemp())

def simulate_run(out_dir, run_id, n_samples=500):
    """Mirrors what _start_multiaxis + worker does."""
    savers = {
        "detector1": StreamSaver(out_dir, f"detector1_run{run_id}"),
        "detector2": StreamSaver(out_dir, f"detector2_run{run_id}"),
    }
    for i in range(n_samples):
        state = {"X": i * 0.5, "Y": i * 0.3}
        for det_id, saver in savers.items():
            saver.append_sample(time.time(), float(i), meta=state)
    # _close_all_stream_savers equivalent
    for saver in savers.values():
        saver.close()
    savers.clear()

simulate_run(tmp, run_id=1)
simulate_run(tmp, run_id=2)

h5s = sorted(tmp.glob("*.h5"))
check("4 .h5 files (2 runs × 2 detectors)", len(h5s) == 4, str([p.name for p in h5s]))
for h5 in h5s:
    verify_h5(h5, 500, h5.stem[-20:])

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== TEST 6: append_sample after close() must not crash ===")
tmp = pathlib.Path(tempfile.mkdtemp())
s = StreamSaver(tmp, "det_postclosed")
s.append_sample(1.0, 2.0, {})
s.close()
try:
    s.append_sample(3.0, 4.0, {})  # should not raise
    check("no crash on post-close append", True)
except Exception as e:
    check("no crash on post-close append", False, str(e))

# ═══════════════════════════════════════════════════════════════════════════
print("\n=== TEST 7: large scan 10000 samples, 2 detectors, perf timing ===")
tmp = pathlib.Path(tempfile.mkdtemp())
N = 10000
savers = {
    "detector1": StreamSaver(tmp, "detector1", flush_every=256),
    "detector2": StreamSaver(tmp, "detector2", flush_every=256),
}
t0 = time.perf_counter()
for i in range(N):
    state = {"X": i * 0.1, "Y": (i % 100) * 0.2}
    for saver in savers.values():
        saver.append_sample(float(i) * 0.001, float(i), meta=state)
for saver in savers.values():
    saver.close()
elapsed = time.perf_counter() - t0
print(f"       {N} samples × 2 detectors in {elapsed:.2f}s ({2*N/elapsed:.0f} samples/s)")
check("completed in < 10s", elapsed < 10, f"{elapsed:.2f}s")
for h5 in sorted(tmp.glob("*.h5")):
    verify_h5(h5, N, h5.stem[-20:])

# ═══════════════════════════════════════════════════════════════════════════
print()
if errors:
    print(f"FAILED: {len(errors)} checks failed:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("All tests passed.")
