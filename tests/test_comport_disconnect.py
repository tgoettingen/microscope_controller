from __future__ import annotations

import unittest
from unittest.mock import Mock

from devices.voltage_meter_comport import ComPort


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
