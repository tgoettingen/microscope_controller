from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets

from gui.tabs.multiaxis_tab import MultiAxisTab


class MultiAxisRunButtonStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_running_button_pulses_and_finished_restores_original_style(self):
        tab = MultiAxisTab()
        original_style = tab._run_button_style

        tab.set_scan_running(True)
        self.assertEqual(
            tab._run_button_animation.state(),
            QtCore.QAbstractAnimation.State.Running,
        )
        self.assertEqual(tab.start_btn.styleSheet(), tab._running_button_style)
        self.assertEqual(tab.start_btn.toolTip(), "Multi-axis scan running")

        tab.set_scan_running(False)
        self.assertEqual(
            tab._run_button_animation.state(),
            QtCore.QAbstractAnimation.State.Stopped,
        )
        self.assertEqual(tab.start_btn.styleSheet(), original_style)
        self.assertEqual(tab.start_btn.toolTip(), "")
        self.assertEqual(tab._run_button_opacity.opacity(), 1.0)
        tab.close()


if __name__ == "__main__":
    unittest.main()
