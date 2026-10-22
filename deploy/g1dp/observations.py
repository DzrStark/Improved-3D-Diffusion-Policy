"""The recorded 28-joint and full-view RGB contract, with no hardware imports."""
from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

GROUPS = ("left_arm", "right_arm", "left_ee", "right_ee")
ARM_PARTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
             "wrist_roll", "wrist_pitch", "wrist_yaw")
# Source: robot_arm.G1_29_JointArmIndex and robot_hand_unitree.*JointIndex.
JOINT_NAMES = tuple(side + "_" + part + "_joint" for side in ("left", "right")
                    for part in ARM_PARTS) + tuple(
    side + "_hand_" + part + "_joint" for side, parts in (
        ("left", ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")),
        ("right", ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")))
    for part in parts)


def joint28(value):
    q = np.asarray(value, dtype=np.float32)
    if q.shape != (28,) or not np.isfinite(q).all():
        raise ValueError("Expected 28 finite actual/target positions in radians")
    return q.copy()


def assemble_state(fields):
    parts = [np.asarray(fields[group]["qpos"], dtype=np.float32) for group in GROUPS]
    if any(part.shape != (7,) for part in parts):
        raise ValueError("Each recorded joint group must have exactly seven entries")
    return joint28(np.concatenate(parts))


def raw_bgr_to_rgb(bgr, binocular=False):
    """JPEG color_1 is already right-eye; a live binocular frame needs splitting."""
    bgr = np.asarray(bgr)
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected uint8 BGR HWC image")
    if binocular:
        if bgr.shape[1] % 2:
            raise ValueError("Binocular width must be even")
        bgr = bgr[:, bgr.shape[1] // 2:]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32)


def processed_rgb(rgb):
    """Zarr is already RGB: do not swap channels, resize, or divide by 255."""
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.shape != (224, 224, 3) or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 255:
        raise ValueError("Expected RGB [224,224,3] in [0,255]")
    return rgb.copy()


@dataclass
class Observation:
    image: np.ndarray
    q: np.ndarray
    timestamp_s: float
    episode: str
    index: int
    wall_timestamp_ns: Optional[int] = None
    timestamp_kind: str = "recorded_monotonic"


class History:
    """Oldest first. Repeat the first measurement once; reset at episode boundary."""
    def __init__(self):
        self.frames = deque(maxlen=2)

    def reset(self):
        self.frames.clear()

    def push(self, frame):
        image, q = processed_rgb(frame.image), joint28(frame.q)
        if not np.isfinite(frame.timestamp_s):
            raise ValueError("Invalid observation timestamp")
        if self.frames and self.frames[-1].episode != frame.episode:
            self.reset()
        if self.frames and frame.timestamp_s <= self.frames[-1].timestamp_s:
            raise ValueError("Measurements must have strictly increasing timestamps")
        # Copies prevent caller mutation and never substitute policy targets for feedback.
        frame = Observation(image, q, frame.timestamp_s, frame.episode, frame.index,
                            frame.wall_timestamp_ns, frame.timestamp_kind)
        if not self.frames:
            self.frames.append(frame)
        self.frames.append(frame)
        return self.arrays()

    def arrays(self):
        if len(self.frames) != 2:
            raise RuntimeError("History has not received a measurement")
        return {"image": np.stack([f.image for f in self.frames]),
                "agent_pos": np.stack([f.q for f in self.frames])}
