"""Lazy image reads; Zarr remains read-only and recorded feedback stays exogenous."""
import json
from pathlib import Path

import cv2
import numpy as np

from .observations import Observation, assemble_state, processed_rgb, raw_bgr_to_rgb


class RawSource:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.paths = sorted(self.root.glob("episode_*/data.json"),
                            key=lambda p: int(p.parent.name.split("_")[-1]))
        if self.root.is_file():
            self.paths = [self.root]
        if not self.paths:
            raise FileNotFoundError("No recorded episode data.json files")
        self._episode_id, self._frames = None, None

    @property
    def n_episodes(self):
        return len(self.paths)

    def frames(self, episode):
        if self._episode_id != episode:
            self._frames = json.loads(self.paths[episode].read_text())["data"]
            self._episode_id = episode
        return self._frames

    def length(self, episode):
        return len(self.frames(episode))

    def read(self, episode, index):
        if not 0 <= index < self.length(episode):
            raise IndexError(index)
        row = self.frames(episode)[index]
        path = self.paths[episode].parent / row["colors"]["color_1"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Cannot decode: " + str(path))
        return Observation(raw_bgr_to_rgb(bgr), assemble_state(row["states"]),
                           row["record_timestamp_mono_ns"] / 1e9,
                           self.paths[episode].parent.name, index,
                           row.get("record_timestamp_wall_ns")), assemble_state(row["actions"])


class ZarrSource:
    def __init__(self, root, raw_root=None, nominal_hz=30):
        import zarr
        self.root = Path(root).resolve()
        self.store = zarr.open_group(str(self.root), mode="r")
        self.ends = np.asarray(self.store["meta/episode_ends"], dtype=np.int64)
        self.starts = np.r_[0, self.ends[:-1]]
        n = self.store["data/state"].shape[0]
        if (len(self.ends) == 0 or np.any(self.ends <= self.starts) or self.ends[-1] != n
                or self.store["data/state"].shape != (n, 28)
                or self.store["data/action"].shape != (n, 28)
                or self.store["data/img"].shape != (n, 224, 224, 3)
                or self.store.attrs.get("joint_order") != ["left_arm", "right_arm", "left_ee", "right_ee"]):
            raise ValueError("Invalid Zarr dimensions, episode boundaries, or joint order")
        if not np.isfinite(nominal_hz) or nominal_hz <= 0:
            raise ValueError("Invalid nominal_hz")
        self.nominal_hz = nominal_hz
        # Timestamps were not stored in the training arrays. Use an explicit source
        # root and conversion manifest; never follow the machine-specific attrs path.
        self.raw_root = Path(raw_root).resolve() if raw_root else None
        self.manifest = None
        if self.raw_root:
            from .policy import sha256
            report = json.loads((self.root / "conversion_report.json").read_text())
            self.manifest = report["episodes"]
            if len(self.manifest) != len(self.ends):
                raise ValueError("Conversion manifest episode count mismatch")
            for e, row in enumerate(self.manifest):
                path = self.raw_root / row["source_json"]
                if row["start"] != self.starts[e] or row["end"] != self.ends[e] or sha256(path) != row["source_json_sha256"]:
                    raise ValueError("Raw JSON hash/boundaries do not match converted data")
        self._episode_id, self._rows = None, None

    @property
    def n_episodes(self):
        return len(self.ends)

    def length(self, episode):
        return int(self.ends[episode] - self.starts[episode])

    def read(self, episode, index):
        if not 0 <= index < self.length(episode):
            raise IndexError(index)
        g = int(self.starts[episode]) + index
        timestamp, wall, kind = index / self.nominal_hz, None, "nominal_%ghz_not_measured" % self.nominal_hz
        name = "episode_index_%04d" % episode
        if self.manifest:
            item = self.manifest[episode]
            if self._episode_id != episode:
                self._rows = json.loads((self.raw_root / item["source_json"]).read_text())["data"]
                if len(self._rows) != self.length(episode):
                    raise ValueError("Raw episode length mismatch")
                self._episode_id = episode
            row = self._rows[index]
            timestamp, wall = row["record_timestamp_mono_ns"] / 1e9, row.get("record_timestamp_wall_ns")
            kind, name = "recorded_monotonic", item["episode"]
        return Observation(processed_rgb(self.store["data/img"][g]),
                           self.store["data/state"][g], timestamp, name, index, wall, kind), self.store["data/action"][g]


def make_source(config):
    if config["source"] == "raw":
        return RawSource(config["raw_root"])
    if config["source"] == "zarr":
        return ZarrSource(config["zarr"], config.get("raw_root"), config["target_hz"])
    raise ValueError("source must be raw or zarr")
