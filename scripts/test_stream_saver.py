import time
import tempfile
from pathlib import Path

import sys
from pathlib import Path as _Path

# Ensure repository root is on sys.path when running as a script from ./scripts
_repo_root = _Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np

from utils.stream_saver import StreamSaver


def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="mc_save_test_"))
    print("tmpdir:", tmpdir)

    s1 = StreamSaver(tmpdir, "detector1", flush_every=128)
    s2 = StreamSaver(tmpdir, "detector2", flush_every=128)

    t0 = time.time()
    for i in range(2000):
        ts = t0 + i * 0.001
        meta = {"state": {"X": i % 10, "Y": i % 20, "Z": i % 5}}
        s1.append_sample(ts, float(i), meta=meta)
        s2.append_sample(ts, float(i * 2), meta=meta)

    s1.close()
    s2.close()

    npy_files = sorted(tmpdir.glob("*.npy"))
    print("final npy files:", [p.name for p in npy_files])
    for p in npy_files:
        arr = np.load(p)
        print(p.name, arr.shape)


if __name__ == "__main__":
    main()
