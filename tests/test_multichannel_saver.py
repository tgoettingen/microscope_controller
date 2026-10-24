import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from utils.multichannel_saver import MultiChannelSaver
from utils.multichannel_h5 import load_phased


class MultiChannelSaverTests(unittest.TestCase):
    def test_appends_sequence_phases_to_one_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "20261024_184900__multichannel_my_experiment"
            saver = MultiChannelSaver(temp_dir, base_path=base_path, flush_every=1)
            path = saver.h5_path

            saver.append_sample(
                "detector_a",
                1.0,
                2.0,
                {"state": {"X": 0.5}, "experiment_phase": "Pre experiment"},
            )
            saver.append_sample(
                "detector_a",
                2.0,
                3.0,
                {"state": {"X": 1.0}, "experiment_phase": "Main scan"},
            )
            saver.append_image(
                np.array([[1, 2], [3, 4]], dtype=np.uint8),
                {"timestamp": 2.0, "experiment_phase": "Main scan"},
            )
            saver.close()

            self.assertEqual(path.name, "20261024_184900__multichannel_my_experiment.h5")
            with h5py.File(path, "r") as h5_file:
                samples = h5_file["detectors/detector_a/data"][:]
                phases = h5_file["detectors/detector_a/experiment_phase"].asstr()[:]
                images = h5_file["images"][:]
                image_meta = h5_file["image_meta_json"].asstr()[:]

            self.assertEqual(samples[:, 1].tolist(), [2.0, 3.0])
            self.assertEqual(phases.tolist(), ["Pre experiment", "Main scan"])
            self.assertEqual(images[0].tolist(), [[1, 2], [3, 4]])
            self.assertIn("Main scan", image_meta[0])
            channels = load_phased(path)

            self.assertEqual(
                set(channels),
                {"detector_a [Pre experiment]", "detector_a [Main scan]"},
            )
            self.assertEqual(channels["detector_a [Pre experiment]"][:, 1].tolist(), [2.0])
            self.assertEqual(channels["detector_a [Main scan]"][:, 1].tolist(), [3.0])

    def test_load_phased_keeps_legacy_channel_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            saver = MultiChannelSaver(temp_dir, flush_every=1)
            path = saver.h5_path
            saver.append_sample("detector_a", 1.0, 2.0)
            saver.close()

            channels = load_phased(path)

            self.assertEqual(set(channels), {"detector_a"})
            self.assertEqual(channels["detector_a"][:, 1].tolist(), [2.0])


if __name__ == "__main__":
    unittest.main()
