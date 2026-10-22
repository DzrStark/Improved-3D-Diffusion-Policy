#!/usr/bin/env python3
"""Convert Unitree JSON/JPEG episodes to the image-policy Zarr v2 format."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import zarr
from numcodecs import Blosc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT.parent / "tsingtec_raw_data/dex3_pick/right"
# Never derive this order from JSON dictionary iteration or info.joint_names.
JOINT_GROUPS = ("left_arm", "right_arm", "left_ee", "right_ee")
JOINT_SLICES = {name: [7 * i, 7 * (i + 1)] for i, name in enumerate(JOINT_GROUPS)}


def flatten_qpos(frame, field, context):
    parts = []
    for name in JOINT_GROUPS:
        qpos = np.asarray(frame[field][name]["qpos"], dtype=np.float32)
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise ValueError(f"{context}: {field}.{name}.qpos must contain 7 finite values")
        parts.append(qpos)
    return np.concatenate(parts)


def load_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot decode image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)


def convert(raw_dir, output):
    raw_dir, output = Path(raw_dir).resolve(), Path(output).resolve()
    episodes = sorted(raw_dir.glob("episode_*"), key=lambda p: int(p.name.split("_")[-1]))
    if not episodes:
        raise ValueError(f"No episode_* directories found in {raw_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to overwrite an existing dataset. No implicit resume or skipped frames.
    root = zarr.open_group(str(output), mode="w-")
    data = root.create_group("data")
    meta = root.create_group("meta")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    images = data.create_dataset("img", shape=(0, 224, 224, 3),
                                 chunks=(16, 224, 224, 3), dtype="uint8", compressor=compressor)
    states = data.create_dataset("state", shape=(0, 28), chunks=(256, 28),
                                 dtype="float32", compressor=compressor)
    actions = data.create_dataset("action", shape=(0, 28), chunks=(256, 28),
                                  dtype="float32", compressor=compressor)
    episode_ends = meta.create_dataset("episode_ends", shape=(0,), chunks=(64,),
                                       dtype="int64", compressor=compressor)
    metadata = {
        "schema_version": 1,
        "source_root": str(raw_dir),
        "output": str(output),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "converter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "opencv": cv2.__version__, "zarr": zarr.__version__},
        "joint_order": list(JOINT_GROUPS),
        "joint_slices": JOINT_SLICES,
        "joint_index_labels": [f"{name}[{i}]" for name in JOINT_GROUPS for i in range(7)],
        "joint_order_note": "Zero-based half-open slices; preserve each source qpos order and sign. Joint names are not recorded; no motor mapping is inferred.",
        "joint_units": "rad",
        "state_source": "states.<group>.qpos (actual joint positions)",
        "action_source": "actions.<group>.qpos (absolute target joint positions)",
        "image_processing": {"source": "colors.color_1", "decode": "cv2.IMREAD_COLOR",
                             "color_conversion": "BGR2RGB", "size_wh": [224, 224],
                             "resize": "INTER_AREA, full field of view, no crop",
                             "layout": "NHWC", "dtype": "uint8", "range": [0, 255]},
        "sequence_processing": "Numeric episode order; original JSON frame order; same-frame state/action/image; no shift, resampling, interpolation, sign changes or action zeroing.",
        "time_gap_threshold_ms": 100.0,
        "conversion_complete": False,
    }
    root.attrs.update(metadata)
    manifest, time_gaps = [], []
    total = 0
    for episode in episodes:
        source = episode / "data.json"
        source_bytes = source.read_bytes()
        frames = json.loads(source_bytes)["data"]
        count = len(frames)
        if count == 0:
            raise ValueError(f"Empty episode: {source}")
        episode_images = np.empty((count, 224, 224, 3), dtype=np.uint8)
        episode_states = np.empty((count, 28), dtype=np.float32)
        episode_actions = np.empty((count, 28), dtype=np.float32)
        previous_timestamp = None
        for i, frame in enumerate(frames):
            context = f"{episode.name}/idx={frame['idx']}"
            if frame["idx"] != i:
                raise ValueError(f"{context}: expected contiguous frame idx={i}")
            episode_states[i] = flatten_qpos(frame, "states", context)
            episode_actions[i] = flatten_qpos(frame, "actions", context)
            episode_images[i] = load_rgb(episode / frame["colors"]["color_1"])
            timestamp = frame["record_timestamp_mono_ns"]
            if previous_timestamp is not None:
                dt_ms = (timestamp - previous_timestamp) / 1e6
                if dt_ms > metadata["time_gap_threshold_ms"]:
                    time_gaps.append({"episode": episode.name, "previous_idx": i - 1,
                                      "idx": i, "dt_ms": dt_ms, "handling": "preserved"})
            previous_timestamp = timestamp
        images.append(episode_images, axis=0)
        states.append(episode_states, axis=0)
        actions.append(episode_actions, axis=0)
        start, total = total, total + count
        episode_ends.append(np.asarray([total], dtype=np.int64), axis=0)
        manifest.append({"episode": episode.name, "source_json": str(source.relative_to(raw_dir)),
                         "source_json_sha256": hashlib.sha256(source_bytes).hexdigest(),
                         "frame_count": count, "start": start, "end": total,
                         "first_idx": frames[0]["idx"], "last_idx": frames[-1]["idx"]})
        print(f"[{len(manifest):02d}/{len(episodes):02d}] {episode.name}: {count} frames, [{start}, {total})", flush=True)
        # Release the current episode before allocating the next one.
        del episode_images, episode_states, episode_actions, frames

    ends = episode_ends[:]
    if (not np.all(np.diff(np.r_[0, ends]) > 0)
            or int(ends[-1]) != total
            or any(array.shape[0] != total for array in data.values())):
        raise ValueError("Inconsistent array lengths or episode boundaries")
    report = dict(metadata, conversion_complete=True, n_episodes=len(manifest), n_frames=total,
                  arrays={name: {"shape": list(array.shape), "dtype": str(array.dtype),
                                 "chunks": list(array.chunks)} for name, array in data.items()},
                  episodes=manifest, time_gaps=time_gaps)
    (output / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    root.attrs.update({"conversion_complete": True, "n_episodes": len(manifest), "n_frames": total,
                       "conversion_report": "conversion_report.json"})
    print(f"Done: {len(manifest)} episodes, {total} frames -> {output}")
    print(f"Time gaps > 100 ms (preserved): {json.dumps(time_gaps, ensure_ascii=False)}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="Directory containing episode_*/data.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT.parent / "tsingtec_training_data/right_pick")
    args = parser.parse_args()
    convert(args.raw_dir, args.output)


if __name__ == "__main__":
    main()
