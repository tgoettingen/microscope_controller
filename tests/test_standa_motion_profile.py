from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from devices.standa_stage import StandaAxis, StandaStageXY
from core import factory
from scripts import capture_standa_motion_profile as profile_cli


class StandaMotionProfileTests(unittest.TestCase):
    def test_stage_position_read_waits_until_xy_move_finishes(self):
        move_started = threading.Event()
        allow_move_to_finish = threading.Event()
        position_read_finished = threading.Event()

        x_device = unittest.mock.Mock()
        y_device = unittest.mock.Mock()

        def block_x_move(_delta):
            move_started.set()
            if not allow_move_to_finish.wait(timeout=2):
                raise TimeoutError("test did not release the X move")

        x_device.move_by.side_effect = block_x_move
        x_device.get_position.return_value = 0
        y_device.get_position.return_value = 0

        with patch(
            "devices.standa_stage.Standa.Standa8SMC",
            side_effect=[x_device, y_device],
        ):
            stage = StandaStageXY("COM4", "COM3")

        move_thread = threading.Thread(target=stage.move_to, args=(10, 20))
        move_thread.start()
        self.assertTrue(move_started.wait(timeout=1))

        def read_position():
            stage.get_position()
            position_read_finished.set()

        read_thread = threading.Thread(target=read_position)
        read_thread.start()
        time.sleep(0.05)
        self.assertFalse(position_read_finished.is_set())

        allow_move_to_finish.set()
        move_thread.join(timeout=2)
        read_thread.join(timeout=2)
        self.assertFalse(move_thread.is_alive())
        self.assertFalse(read_thread.is_alive())
        self.assertTrue(position_read_finished.is_set())
        stage.disconnect()

    def test_stage_move_reports_failure_of_either_axis(self):
        x_device = unittest.mock.Mock()
        y_device = unittest.mock.Mock()
        x_device.get_position.return_value = 0
        y_device.get_position.return_value = 0
        x_device.move_by.side_effect = OSError("X controller unavailable")

        with patch(
            "devices.standa_stage.Standa.Standa8SMC",
            side_effect=[x_device, y_device],
        ):
            stage = StandaStageXY("COM4", "COM3")

        with self.assertRaisesRegex(RuntimeError, "X success=False"):
            stage.move_to(10, 20)
        stage.disconnect()

    def test_stage_applies_independent_axis_profiles(self):
        x_device = unittest.mock.Mock()
        y_device = unittest.mock.Mock()
        with patch(
            "devices.standa_stage.Standa.Standa8SMC",
            side_effect=[x_device, y_device],
        ):
            stage = StandaStageXY(
                "COM4",
                "COM3",
                x_max_velocity=12000,
                x_max_acceleration=700,
                x_max_deceleration=800,
                y_max_velocity=9000,
                y_max_acceleration=500,
                y_max_deceleration=600,
            )

        x_device.setup_move.assert_called_once_with(speed=12000, accel=700, decel=800)
        y_device.setup_move.assert_called_once_with(speed=9000, accel=500, decel=600)
        stage.disconnect()
        x_device.close.assert_called_once()
        y_device.close.assert_called_once()

    def test_unconfigured_profile_keeps_controller_defaults(self):
        device = unittest.mock.Mock()
        with patch("devices.standa_stage.Standa.Standa8SMC", return_value=device):
            axis = StandaAxis("COM4")

        device.setup_move.assert_not_called()
        axis.disconnect()

    def test_profile_values_must_be_positive(self):
        with self.assertRaises(ValueError):
            StandaAxis("COM4", max_velocity=0)
        with self.assertRaises(ValueError):
            StandaAxis("COM4", max_velocity=0.5)

    def test_cli_report_only_leaves_config_unchanged_and_closes_controllers(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "devices.json"
            original = {
                "stage": {
                    "type": "StandaStageXY",
                    "com_x": "COM4",
                    "com_y": "COM3",
                    "motors": {"x": {}, "y": {}},
                }
            }
            original_text = json.dumps(original, indent=2)
            config_path.write_text(original_text, encoding="utf-8")
            controllers = [
                unittest.mock.Mock(get_move_parameters=unittest.mock.Mock(
                    return_value=SimpleNamespace(speed=1000, accel=200, decel=300)
                )),
                unittest.mock.Mock(get_move_parameters=unittest.mock.Mock(
                    return_value=SimpleNamespace(speed=4000, accel=500, decel=600)
                )),
            ]
            output = io.StringIO()
            with patch.object(profile_cli.Standa, "Standa8SMC", side_effect=controllers):
                with contextlib.redirect_stdout(output):
                    result = profile_cli.main(["--config", str(config_path)])

            self.assertEqual(result, 0)
            self.assertIn("X (1000 microsteps/s", output.getvalue())
            self.assertIn("Y (4000 microsteps/s", output.getvalue())
            self.assertIn("Report only", output.getvalue())
            self.assertEqual(config_path.read_text(encoding="utf-8"), original_text)
            for controller in controllers:
                controller.close.assert_called_once()

    def test_write_profiles_updates_axes_and_preserves_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "devices.json"
            original = {
                "stage": {
                    "type": "StandaStageXY",
                    "motors": {"x": {"motor_spec_id": None}, "y": {}},
                },
                "other": "preserve",
            }
            config_path.write_text(json.dumps(original, indent=2), encoding="utf-8")
            profiles = {
                "x": {
                    "max_velocity": 1000,
                    "max_acceleration": 200,
                    "max_deceleration": 300,
                },
                "y": {
                    "max_velocity": 4000,
                    "max_acceleration": 500,
                    "max_deceleration": 600,
                },
            }

            backup_path = profile_cli.write_profiles(config_path, profiles)
            updated = json.loads(config_path.read_text(encoding="utf-8"))
            backup = json.loads(backup_path.read_text(encoding="utf-8"))

            self.assertEqual(backup, original)
            self.assertEqual(updated["other"], "preserve")
            self.assertEqual(updated["stage"]["motors"]["x"]["max_velocity"], 1000)
            self.assertEqual(updated["stage"]["motors"]["y"]["max_deceleration"], 600)

    def test_factory_routes_config_values_to_matching_axes(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "devices.json"
            config = {
                "stage": {
                    "type": "StandaStageXY",
                    "com_x": "COM4",
                    "com_y": "COM3",
                    "motors": {
                        "x": {
                            "max_velocity": 1100,
                            "max_acceleration": 210,
                            "max_deceleration": 310,
                        },
                        "y": {
                            "max_velocity": 4200,
                            "max_acceleration": 520,
                            "max_deceleration": 620,
                        },
                    },
                },
                "focus": {"type": "simulated"},
                "camera": {"type": "simulated"},
                "light": {"type": "simulated"},
                "filter_wheel": {"type": "simulated"},
                "detector": {"type": "simulated"},
                "excitation": {"type": "simulated"},
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            with patch("core.factory.StandaStageXY") as stage_factory:
                factory.build_devices(str(config_path))

        kwargs = stage_factory.call_args.kwargs
        self.assertEqual(kwargs["x_max_velocity"], 1100)
        self.assertEqual(kwargs["x_max_acceleration"], 210)
        self.assertEqual(kwargs["x_max_deceleration"], 310)
        self.assertEqual(kwargs["y_max_velocity"], 4200)
        self.assertEqual(kwargs["y_max_acceleration"], 520)
        self.assertEqual(kwargs["y_max_deceleration"], 620)

    def test_cli_write_flag_persists_both_captured_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "devices.json"
            config_path.write_text(json.dumps({
                "stage": {
                    "type": "StandaStageXY",
                    "com_x": "COM4",
                    "com_y": "COM3",
                    "motors": {"x": {}, "y": {}},
                }
            }), encoding="utf-8")
            controllers = [
                unittest.mock.Mock(get_move_parameters=unittest.mock.Mock(
                    return_value=SimpleNamespace(speed=101, accel=202, decel=303)
                )),
                unittest.mock.Mock(get_move_parameters=unittest.mock.Mock(
                    return_value=SimpleNamespace(speed=404, accel=505, decel=606)
                )),
            ]
            output = io.StringIO()
            with patch.object(profile_cli.Standa, "Standa8SMC", side_effect=controllers):
                with contextlib.redirect_stdout(output):
                    result = profile_cli.main([
                        "--config", str(config_path), "--write-config",
                    ])

            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(result, 0)
            self.assertEqual(saved["stage"]["motors"]["x"]["max_acceleration"], 202)
            self.assertEqual(saved["stage"]["motors"]["y"]["max_deceleration"], 606)
            self.assertIn("Original backed up to", output.getvalue())


if __name__ == "__main__":
    unittest.main()
