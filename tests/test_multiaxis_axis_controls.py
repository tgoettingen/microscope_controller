from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets

from core.multiaxis import AxisConfig
from gui.dialogs.channel_axis_dialog import ChannelAxisDialog
from gui.dialogs.detector_axis_dialog import DetectorAxisDialog
from gui.dialogs.excitation_axis_dialog import ExcitationAxisDialog
from gui.dialogs.motor_axis_dialog import MotorAxisDialog
from gui.dialogs.round_axis_dialog import RoundAxisDialog
from gui.tabs.multiaxis_tab import MultiAxisTab
from gui.tabs.live_tab import LiveTab


class _ExcitationDevice:
    def __init__(self, name):
        self.name = name


class MultiAxisAxisControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_axis_name_is_included_in_list_label(self):
        config = AxisConfig(
            "X",
            {"start": 0, "end": 10, "step": 1, "display_name": "Sample"},
        )

        self.assertTrue(config.label().startswith("Sample: X:"))

    def test_axis_name_prompt_uses_unique_default(self):
        tab = MultiAxisTab()
        existing = AxisConfig(
            "X",
            {"start": 0, "end": 1, "step": 1, "display_name": "X"},
        )
        item = QtWidgets.QListWidgetItem(existing.label())
        item.setData(QtCore.Qt.ItemDataRole.UserRole, existing)
        tab.axis_list.addItem(item)

        with patch.object(
            QtWidgets.QInputDialog,
            "getText",
            return_value=("Custom X", True),
        ) as get_text:
            name = tab._prompt_axis_name(AxisConfig("X", {}))

        self.assertEqual(name, "Custom X")
        self.assertEqual(get_text.call_args.kwargs["text"], "X 2")
        tab.close()

    def test_axis_list_has_resizable_splitter_panel(self):
        tab = MultiAxisTab()
        tab.resize(300, 400)
        tab.show()
        self.app.processEvents()

        self.assertEqual(
            tab.axis_splitter.orientation(),
            QtCore.Qt.Orientation.Vertical,
        )
        self.assertIs(
            tab.axis_splitter.widget(0).findChild(QtWidgets.QListWidget),
            tab.axis_list,
        )
        self.assertGreater(tab.axis_splitter.handleWidth(), 0)
        self.assertGreater(tab.axis_list.maximumHeight(), 100)

        tab.axis_splitter.setSizes([250, 100])
        self.app.processEvents()
        self.assertGreater(tab.axis_list.height(), 100)
        tab.close()

    def test_control_and_command_shortcuts_reorder_axis(self):
        for modifier in (
            QtCore.Qt.KeyboardModifier.ControlModifier,
            QtCore.Qt.KeyboardModifier.MetaModifier,
        ):
            tab = MultiAxisTab()
            for axis_type in ("X", "Y"):
                config = AxisConfig(
                    axis_type,
                    {"start": 0, "end": 1, "step": 1},
                )
                item = QtWidgets.QListWidgetItem(config.label())
                item.setData(QtCore.Qt.ItemDataRole.UserRole, config)
                tab.axis_list.addItem(item)

            tab.axis_list.setCurrentRow(0)
            event = QtGui.QKeyEvent(
                QtCore.QEvent.Type.KeyPress,
                QtCore.Qt.Key.Key_Down,
                modifier,
            )
            tab._axis_list_key_press(event)

            self.assertEqual(
                tab.axis_list.item(0).data(QtCore.Qt.ItemDataRole.UserRole).axis_type,
                "Y",
            )
            self.assertEqual(
                tab.axis_list.item(1).data(QtCore.Qt.ItemDataRole.UserRole).axis_type,
                "X",
            )
            tab.close()

    def test_round_dialog_restores_round_count(self):
        dialog = RoundAxisDialog(
            config=AxisConfig("Round", {"n_rounds": 17}),
        )

        self.assertEqual(dialog.rounds_spin.value(), 17)
        self.assertEqual(dialog.get_config().params["n_rounds"], 17)

    def test_excitation_dialog_restores_pattern_wait_channel_and_device(self):
        config = AxisConfig(
            "Excitation",
            {
                "states": [True, False, False],
                "wait": 0.35,
                "channel": 3,
                "excitation": "laser_b",
            },
        )
        dialog = ExcitationAxisDialog(
            excitation_devices=[
                _ExcitationDevice("laser_a"),
                _ExcitationDevice("laser_b"),
            ],
            config=config,
        )

        restored = dialog.get_config().params
        self.assertEqual(dialog.on_off_combo.currentData(), "custom")
        self.assertEqual(dialog.custom_sequence_edit.text(), "True,False,False")
        self.assertEqual(dialog.device_combo.currentData(), "laser_b")
        self.assertEqual(restored["states"], [True, False, False])
        self.assertEqual(restored["wait"], 0.35)
        self.assertEqual(restored["channel"], 3)
        self.assertEqual(restored["excitation"], "laser_b")

    def test_channel_dialog_restores_channels_and_wait(self):
        config = AxisConfig(
            "Channel",
            {
                "channels": [
                    {
                        "name": "bright",
                        "filter_position": 2,
                        "light_intensity": 0.7,
                        "exposure_ms": 12.5,
                    },
                ],
                "wait": 0.4,
            },
        )
        dialog = ChannelAxisDialog(config=config)

        restored = dialog.get_config()
        self.assertEqual(dialog.table.rowCount(), 1)
        self.assertEqual(restored.params["channels"][0].name, "bright")
        self.assertEqual(restored.params["channels"][0].filter_position, 2)
        self.assertEqual(restored.params["channels"][0].light_intensity, 0.7)
        self.assertEqual(restored.params["channels"][0].exposure_ms, 12.5)
        self.assertEqual(restored.params["wait"], 0.4)

    def test_new_motor_dialog_uses_one_unit_step_size_and_zero_wait(self):
        with patch("gui.dialogs.motor_axis_dialog._stage_unit", return_value="mm"):
            for axis_type in ("X", "Y"):
                with self.subTest(axis=axis_type):
                    dialog = MotorAxisDialog(axis_type)
                    self.assertEqual(dialog.step_spin.value(), 1.0)
                    self.assertEqual(dialog.step_spin.suffix(), " mm")
                    self.assertEqual(dialog.wait_spin.value(), 0.0)
                    self.assertEqual(
                        [
                            label.text()
                            for label in dialog.findChildren(QtWidgets.QLabel)
                        ].count("Step size"),
                        1,
                    )
                    dialog.close()

    def test_new_motor_z_axis_has_zero_wait(self):
        dialog = MotorAxisDialog("Z")

        self.assertEqual(dialog.wait_spin.value(), 0.0)
        dialog.close()

    def test_new_non_motor_axes_have_zero_wait(self):
        dialogs = (
            ChannelAxisDialog(),
            DetectorAxisDialog(),
            ExcitationAxisDialog(),
        )
        for dialog in dialogs:
            with self.subTest(dialog=type(dialog).__name__):
                self.assertEqual(dialog.wait_spin.value(), 0.0)
            dialog.close()

    def test_detector_offset_applies_to_multiaxis_plot_curve(self):
        live_tab = LiveTab()
        live_tab.register_detector("vm_offset_test")
        live_tab._detector_offset_cbs["vm_offset_test"].setChecked(True)
        live_tab._detector_display_offsets["vm_offset_test"] = 2.5
        live_tab.multi_coords["vm_offset_test"] = [
            ({"X": 0.0}, 10.0),
            ({"X": 1.0}, 12.0),
        ]

        live_tab._update_multiaxis_visualization()

        x_values, y_values = live_tab._detector_curves["vm_offset_test"].getData()
        self.assertEqual(list(x_values), [0.0, 1.0])
        self.assertEqual(list(y_values), [7.5, 9.5])
        live_tab.close()

    def test_detector_offset_applies_to_multiaxis_image(self):
        live_tab = LiveTab()
        detector_id = "vm_image_offset_test"
        live_tab.register_detector(detector_id)
        live_tab._detector_offset_cbs[detector_id].setChecked(True)
        live_tab._detector_display_offsets[detector_id] = 2.5
        live_tab.multi_coords[detector_id] = [
            ({"X": 0.0, "Y": 0.0}, 10.0),
            ({"X": 1.0, "Y": 0.0}, 12.0),
            ({"X": 0.0, "Y": 1.0}, 14.0),
            ({"X": 1.0, "Y": 1.0}, 16.0),
        ]

        live_tab._update_multiaxis_visualization()

        np.testing.assert_array_equal(
            live_tab._detector_last_images[detector_id],
            np.array([[7.5, 9.5], [11.5, 13.5]], dtype=np.float32),
        )
        live_tab.close()


if __name__ == "__main__":
    unittest.main()
