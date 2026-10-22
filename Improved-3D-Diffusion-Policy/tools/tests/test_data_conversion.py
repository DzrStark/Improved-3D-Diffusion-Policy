"""Conversion regression checks, using small synthetic recordings rather than hardware."""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from zarr.errors import ContainsGroupError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_unitree_zarr import check
from convert_unitree_to_zarr import convert


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "original-host/raw"
        self.output = self.root / "original-host/converted"
        episode = self.raw / "episode_0001"
        episode.mkdir(parents=True)
        frames = []
        for index in range(2):
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            image[:, :16] = [20 + index, 80, 160]
            image[:, 16:] = [180, 40, 10 + index]
            name = "%d_color_1.jpg" % index
            self.assertTrue(cv2.imwrite(str(episode / name), image))
            groups = {name: {"qpos": [group / 10 + index / 100] * 7}
                      for group, name in enumerate(("left_arm", "right_arm", "left_ee", "right_ee"))}
            frames.append({"idx": index, "colors": {"color_1": name},
                           "states": groups, "actions": groups,
                           "record_timestamp_mono_ns": 1_000_000_000 + index * 33_333_333})
        (episode / "data.json").write_text(json.dumps({"data": frames}))
        with contextlib.redirect_stdout(io.StringIO()):
            convert(self.raw, self.output)

    def test_copied_recording_and_zarr_verify_at_new_location(self):
        moved_raw = self.root / "another-host/raw"
        moved_output = self.root / "another-host/converted"
        shutil.move(str(self.raw), str(moved_raw))
        shutil.move(str(self.output), str(moved_output))
        with contextlib.redirect_stdout(io.StringIO()):
            result = check(moved_raw, moved_output, all_images=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["rgb_pixel_check_count"], 2)
        self.assertEqual(result["source_root_at_conversion"], str(self.raw))
        self.assertEqual(result["checked_raw_root"], str(moved_raw))

    def test_changed_source_json_is_rejected_after_copy(self):
        moved_raw = self.root / "another-host/raw"
        shutil.copytree(self.raw, moved_raw)
        path = moved_raw / "episode_0001/data.json"
        data = json.loads(path.read_text())
        data["data"][0]["states"]["left_arm"]["qpos"][0] = 0.5
        path.write_text(json.dumps(data))
        with self.assertRaises(AssertionError):
            check(moved_raw, self.output, all_images=True)

    def test_conversion_refuses_to_overwrite_existing_store(self):
        before = (self.output / "conversion_report.json").read_bytes()
        with self.assertRaises(ContainsGroupError):
            convert(self.raw, self.output)
        self.assertEqual(before, (self.output / "conversion_report.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
