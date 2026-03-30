from __future__ import annotations

import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Final


_MAX_LOG_FILE_BYTES: Final[int] = 100 * 1024 * 1024  # 100 MB
_DEFAULT_RETENTION_DAYS: Final[int] = 365 * 2  # 2 years


def _default_log_dir(app_name: str) -> Path:
    # Windows-friendly default location. Fall back to the user home directory.
    base = os.getenv("MICROSCOPE_CONTROLLER_LOG_DIR")
    if base:
        return Path(base).expanduser().resolve()

    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / app_name / "logs"

    return Path.home() / f".{app_name}" / "logs"


def cleanup_old_logs(log_dir: Path, *, app_name: str, retention_days: int = _DEFAULT_RETENTION_DAYS) -> int:
    """Delete log files older than the retention window.

    Returns the number of files deleted.
    """
    try:
        log_dir = Path(log_dir)
    except Exception:
        return 0

    cutoff_ts = time.time() - (retention_days * 24 * 60 * 60)
    deleted = 0

    # Match both the active file and rotated suffixes like "app.log.1".
    pattern = f"{app_name}.log*"
    try:
        for p in log_dir.glob(pattern):
            try:
                if not p.is_file():
                    continue
                # Don't delete the active log file (even if the clock is wrong)
                if p.name == f"{app_name}.log":
                    continue
                if p.stat().st_mtime < cutoff_ts:
                    p.unlink(missing_ok=True)
                    deleted += 1
            except Exception:
                # Best-effort cleanup; never block app start.
                continue
    except Exception:
        return deleted

    return deleted


def setup_app_logging(
    *,
    app_name: str = "microscope_controller",
    log_dir: str | Path | None = None,
    level: int = logging.DEBUG,
) -> tuple[Path, Path]:
    """Configure application logging.

    - Writes to a rotating file (max 100MB per file)
    - Keeps log files for up to ~2 years by deleting older rotated files on startup

    Returns (log_dir, log_file).
    """
    log_path = Path(log_dir) if log_dir is not None else _default_log_dir(app_name)
    log_path.mkdir(parents=True, exist_ok=True)

    # Housekeeping: delete ancient rotated logs before we start writing.
    cleanup_old_logs(log_path, app_name=app_name, retention_days=_DEFAULT_RETENTION_DAYS)

    log_file = log_path / f"{app_name}.log"

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if setup is called multiple times (e.g., tests).
    if any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "") == str(log_file) for h in root.handlers):
        return log_path, log_file

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s [%(process)d:%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=_MAX_LOG_FILE_BYTES,
        backupCount=1000,  # retention is enforced via time-based cleanup
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # Capture warnings via logging.
    try:
        logging.captureWarnings(True)
    except Exception:
        pass

    # Log unhandled exceptions (main thread, worker threads).
    def _excepthook(exc_type, exc, tb):
        try:
            logging.getLogger(app_name).exception("Unhandled exception", exc_info=(exc_type, exc, tb))
        finally:
            try:
                sys.__excepthook__(exc_type, exc, tb)
            except Exception:
                pass

    try:
        sys.excepthook = _excepthook
    except Exception:
        pass

    try:
        import threading

        def _threading_excepthook(args):
            try:
                logging.getLogger(app_name).exception(
                    "Unhandled thread exception in %s",
                    getattr(args, "thread", None),
                    exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                )
            except Exception:
                pass

        if hasattr(threading, "excepthook"):
            threading.excepthook = _threading_excepthook  # type: ignore[assignment]
    except Exception:
        pass

    logging.getLogger(app_name).info("Logging initialized: %s", log_file)
    return log_path, log_file
