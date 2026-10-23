from __future__ import annotations

import os
import threading
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets

from gui.mainwindow import MainWindow
from gui.tabs.move_motors_tab import StageControlTab


class _SlowStage:
    def __init__(self):
        self.read_started = threading.Event()
        self.read_threads: list[int] = []
        self.read_count = 0

    def get_position(self):
        self.read_count += 1
        self.read_threads.append(threading.get_ident())
        self.read_started.set()
        time.sleep(0.1)
        return 12.5, 4.25


class _QueuedMoveStage(_SlowStage):
    def __init__(self):
        super().__init__()
        self.move_started = threading.Event()
        self.allow_first_move = threading.Event()
        self.moves: list[tuple[float, float]] = []

    def move_to(self, x, y):
        self.moves.append((x, y))
        if len(self.moves) == 1:
            self.move_started.set()
            if not self.allow_first_move.wait(timeout=2):
                raise TimeoutError("test did not release the first move")


class _StatusPollHarness(QtWidgets.QMainWindow):
    stage_position_read_completed = QtCore.pyqtSignal(object)

    def __init__(self, stage):
        super().__init__()
        self.stage = stage
        self.stage_control_tab = None
        self._stage_position_poll_in_flight = False
        self.position_text = ""
        self.stage_position_read_completed.connect(self._apply_result)

    def _set_stage_position_text(self, text):
        self.position_text = text

    @QtCore.pyqtSlot(object)
    def _apply_result(self, result):
        MainWindow._apply_stage_position_read(self, result)


class StageControlResponsivenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_position_refresh_reads_hardware_off_the_ui_thread(self):
        stage = _SlowStage()
        tab = StageControlTab(stage=stage)
        ui_thread_id = threading.get_ident()

        start = time.monotonic()
        tab._auto_refresh_position()
        elapsed = time.monotonic() - start
        tab._auto_refresh_position()

        self.assertLess(elapsed, 0.05)
        self.assertTrue(stage.read_started.wait(timeout=1))
        self.assertEqual(stage.read_count, 1)
        self.assertNotEqual(stage.read_threads[0], ui_thread_id)

        deadline = time.monotonic() + 1
        while tab._position_refresh_in_flight and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()

        self.assertFalse(tab._position_refresh_in_flight)
        self.assertEqual(tab.current_x_label.text(), "12.5")
        self.assertEqual(tab.current_y_label.text(), "4.2")
        tab.close()

    def test_status_bar_refresh_reads_hardware_off_the_ui_thread(self):
        stage = _SlowStage()
        window = _StatusPollHarness(stage)
        ui_thread_id = threading.get_ident()

        start = time.monotonic()
        MainWindow._update_stage_position_display(window)
        elapsed = time.monotonic() - start

        self.assertLess(elapsed, 0.05)
        self.assertTrue(stage.read_started.wait(timeout=1))
        self.assertNotEqual(stage.read_threads[0], ui_thread_id)

        deadline = time.monotonic() + 1
        while window._stage_position_poll_in_flight and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()

        self.assertFalse(window._stage_position_poll_in_flight)
        self.assertEqual(window.position_text, "Stage: X=12.50, Y=4.25")
        window.close()

    def test_repeated_moves_keep_only_the_latest_queued_target(self):
        stage = _QueuedMoveStage()
        tab = StageControlTab(stage=stage)

        tab._move_to_position_xy(1, 1)
        self.assertTrue(stage.move_started.wait(timeout=1))
        tab._move_to_position_xy(1, 1)
        self.assertIsNone(tab._pending_xy_target)
        tab._move_to_position_xy(2, 2)
        tab._move_to_position_xy(3, 3)
        self.assertEqual(tab._pending_xy_target, (3.0, 3.0))

        stage.allow_first_move.set()
        deadline = time.monotonic() + 2
        while (tab._is_moving or len(stage.moves) < 2) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.app.processEvents()

        self.assertEqual(stage.moves, [(1, 1), (3.0, 3.0)])
        self.assertFalse(tab._is_moving)
        tab.close()
