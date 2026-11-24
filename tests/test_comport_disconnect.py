from __future__ import annotations

import unittest
from unittest.mock import Mock

from devices.voltage_meter_comport import ComPort


class ComPortProtocolTests(unittest.TestCase):
    def test_dual_channel_frame_uses_configured_gain(self):
        frame1 = bytes.fromhex("0A01 001234 46 010A")
        frame2 = bytes.fromhex("0A02 001234 46 010A")

        channel1 = ComPort.parse_frame(frame1, gain=601)
        channel2 = ComPort.parse_frame(frame2, gain=601)

        self.assertIsNotNone(channel1)
        self.assertIsNotNone(channel2)
        self.assertEqual(channel1["channel"], 1)
        self.assertEqual(channel2["channel"], 2)

    def test_channel_values_can_be_read_independently(self):
        detector = ComPort(port="COM_TEST", mode=1)
        detector._channel_scaled_values = {1: 1.25, 2: 2.5}

        self.assertEqual(detector.read_channel_value(1), 1.25)
        self.assertEqual(detector.read_channel_value(2), 2.5)
        self.assertEqual(detector.read_value(), 1.25)

    def test_dual_meter_exposes_two_gui_detector_ids(self):
        detector = ComPort(port="COM_TEST", mode=0)
        detector.product_id = 2
        self.assertEqual(detector.get_detector_ids(), ["COM_TEST CH1", "COM_TEST CH2"])


class ComPortDisconnectTests(unittest.TestCase):
    def test_disconnect_flushes_and_closes_port_then_clears_parser_state(self):
        detector = ComPort(port="COM_TEST")
        serial_port = Mock()
        detector.ser = serial_port
        detector.connected = True
        detector._rx_buffer.extend(b"stale frame bytes")
        detector._ring_buffer.append({"voltage": 1.0})
        detector._last_value = 1.0
        detector._last_scaled_value = 1.0
        detector._last_temperature = b"temp"
        detector._last_timestamp = 123.0

        detector.disconnect()

        serial_port.cancel_read.assert_called_once()
        serial_port.reset_input_buffer.assert_called_once()
        serial_port.reset_output_buffer.assert_called_once()
        serial_port.close.assert_called_once()
        self.assertIsNone(detector.ser)
        self.assertFalse(detector.connected)
        self.assertEqual(detector._rx_buffer, bytearray())
        self.assertEqual(len(detector._ring_buffer), 0)
        self.assertIsNone(detector._last_value)
        self.assertIsNone(detector._last_scaled_value)
        self.assertIsNone(detector._last_temperature)
        self.assertIsNone(detector._last_timestamp)

    def test_stop_retains_reference_if_reader_has_not_exited(self):
        detector = ComPort(port="COM_TEST")

        class StillRunningThread:
            def is_alive(self):
                return True

            def join(self, timeout=None):
                self.timeout = timeout

        reader = StillRunningThread()
        detector._thread = reader
        detector.running = True

        detector.stop()

        self.assertIs(detector._thread, reader)
        self.assertFalse(detector.running)
        self.assertTrue(detector._stop_event.is_set())
        self.assertGreaterEqual(reader.timeout, 1.0)


if __name__ == "__main__":
    unittest.main()
