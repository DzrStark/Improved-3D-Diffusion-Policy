#!/usr/bin/env python3
"""Independently check step-1 conversion against all right-hand source JSON."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import zarr


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check(raw_dir, zarr_path):
    raw_dir, zarr_path = Path(raw_dir).resolve(), Path(zarr_path).resolve()
    root = zarr.open_group(str(zarr_path), mode="r")
    report = json.loads((zarr_path / "conversion_report.json").read_text())
    episodes = sorted(raw_dir.glob("episode_*"), key=lambda p: int(p.name.split("_")[-1]))
    ends = root["meta/episode_ends"][:]
    assert root.attrs["conversion_complete"] and report["conversion_complete"]
    assert len(episodes) == len(ends) == len(report["episodes"]) == report["n_episodes"] == 60
    assert root["meta/episode_ends"].dtype == np.dtype("int64")
    assert set(root["data"].array_keys()) == {"img", "state", "action"}
    expected_slices = {"left_arm": [0, 7], "right_arm": [7, 14],
                       "left_ee": [14, 21], "right_ee": [21, 28]}
    assert root.attrs["joint_order"] == list(expected_slices) == report["joint_order"]
    assert root.attrs["joint_slices"] == expected_slices == report["joint_slices"]
    for key, shape, dtype in [("img", (10344, 224, 224, 3), "uint8"),
                              ("state", (10344, 28), "float32"),
                              ("action", (10344, 28), "float32")]:
        array = root[f"data/{key}"]
        assert array.shape == shape and array.dtype == np.dtype(dtype), key
        assert array.compressor.cname == "zstd", key

    start, pixel_checks, small_action_values = 0, [], 0
    time_gaps = []
    sample_episodes = {0, len(episodes) // 2, len(episodes) - 1}
    for episode_index, episode in enumerate(episodes):
        source_bytes = (episode / "data.json").read_bytes()
        frames = json.loads(source_bytes)["data"]
        end = start + len(frames)
        entry = report["episodes"][episode_index]
        assert entry["episode"] == episode.name
        assert entry["source_json"] == f"{episode.name}/data.json"
        assert entry["source_json_sha256"] == hashlib.sha256(source_bytes).hexdigest()
        assert entry["start"] == start and entry["end"] == end == ends[episode_index]
        assert entry["frame_count"] == len(frames)
        assert entry["first_idx"] == 0 and entry["last_idx"] == len(frames) - 1
        assert [f["idx"] for f in frames] == list(range(len(frames)))
        for source_key, array_key in [("states", "state"), ("actions", "action")]:
            expected = np.empty((len(frames), 28), dtype=np.float32)
            # Explicit slice assignment independently verifies the converter's concatenation.
            expected[:, 0:7] = [f[source_key]["left_arm"]["qpos"] for f in frames]
            expected[:, 7:14] = [f[source_key]["right_arm"]["qpos"] for f in frames]
            expected[:, 14:21] = [f[source_key]["left_ee"]["qpos"] for f in frames]
            expected[:, 21:28] = [f[source_key]["right_ee"]["qpos"] for f in frames]
            actual = root[f"data/{array_key}"][start:end]
            assert np.isfinite(actual).all()
            np.testing.assert_array_equal(actual, expected, err_msg=f"{episode.name}/{array_key}")
            if array_key == "action":
                small_action_values += int(np.count_nonzero(np.abs(expected) == np.float32(0.001)))
        for i in range(1, len(frames)):
            dt_ms = (frames[i]["record_timestamp_mono_ns"] - frames[i - 1]["record_timestamp_mono_ns"]) / 1e6
            if dt_ms > 100:
                time_gaps.append({"episode": episode.name, "previous_idx": i - 1,
                                  "idx": i, "dt_ms": dt_ms, "handling": "preserved"})
        if episode_index in sample_episodes:
            for i in sorted({0, len(frames) // 2, len(frames) - 1}):
                path = episode / frames[i]["colors"]["color_1"]
                bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
                assert bgr is not None, path
                expected_rgb = cv2.resize(bgr[:, :, ::-1], (224, 224), interpolation=cv2.INTER_AREA)
                np.testing.assert_array_equal(root["data/img"][start + i], expected_rgb)
                pixel_checks.append({"episode": episode.name, "idx": i, "global_index": start + i})
        start = end
    assert start == int(ends[-1]) == report["n_frames"] == 10344
    assert report["time_gaps"] == time_gaps
    assert np.all(np.diff(np.r_[0, ends]) > 0)
    # Decode every stored image chunk without loading the entire image dataset.
    images = root["data/img"]
    for i in range(0, start, images.chunks[0]):
        block = images[i:i + images.chunks[0]]
        assert block.shape[1:] == (224, 224, 3) and block.dtype == np.uint8
    result = {"passed": True, "n_episodes": len(episodes), "n_frames": start,
              "all_state_action_values_exact_after_float32": True,
              "all_image_chunks_readable": True, "rgb_pixel_checks": pixel_checks,
              "preserved_action_abs_0_001_count": small_action_values, "time_gaps": time_gaps}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path,
                        default=PROJECT_ROOT.parent / "tsingtec_raw_data/dex3_pick/right")
    parser.add_argument("--zarr", type=Path, default=PROJECT_ROOT / "data/dex3_pick_right_rgb.zarr")
    args = parser.parse_args()
    check(args.raw_dir, args.zarr)


if __name__ == "__main__":
    main()
