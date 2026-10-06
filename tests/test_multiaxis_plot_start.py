from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from gui.tabs.live_tab import LiveTab


class MultiAxisPlotStartupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_first_sample_initializes_visible_multiaxis_curve(self):
        live_tab = LiveTab()
        live_tab.set_selected_detectors(["vm_test"])
        live_tab.prepare_multiaxis_plot(["vm_test"])

        self.assertEqual(live_tab._plot_mode, "strip")
        live_tab.queue_multiaxis_sample("vm_test", {"X": 1.0}, 2.5)
        live_tab._update_plot()

        self.assertEqual(live_tab._plot_mode, "multiaxis")
        self.assertEqual(len(live_tab.multi_coords["vm_test"]), 1)
        curve = live_tab._detector_curves["vm_test"]
        self.assertIsNotNone(curve.scene())
        self.assertTrue(curve.isVisible())
        self.assertEqual(list(curve.getData()[0]), [0])
        self.assertEqual(list(curve.getData()[1]), [2.5])
        live_tab.close()

    def test_first_resistance_sample_initializes_resistance_curve(self):
        live_tab = LiveTab()
        live_tab.set_selected_detectors(["meter_test"])
        live_tab.prepare_multiaxis_plot(["meter_test"])
        live_tab.queue_multiaxis_sample(
            "meter_test",
            {"X": 1.0, "measurement_kind": "resistance"},
            1250.0,
        )
        live_tab._update_plot()

        curve = live_tab._resistance_curves["meter_test"]
        self.assertEqual(live_tab._plot_mode, "multiaxis")
        self.assertIsNotNone(curve.scene())
        self.assertTrue(curve.isVisible())
        self.assertEqual(list(curve.getData()[1]), [1250.0])
        live_tab.close()


if __name__ == "__main__":
    unittest.main()
