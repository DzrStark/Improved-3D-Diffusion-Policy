"""Replaceable camera and joint-only robot boundaries. No SDK imports at import time."""
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .observations import joint28, raw_bgr_to_rgb


@dataclass
class Feedback:
    q: np.ndarray
    received_monotonic_s: float


class Camera(Protocol):
    def read(self):
        """Return image and its reception timestamp in the local monotonic clock."""
        ...


class Robot(Protocol):
    def feedback(self) -> Feedback:
        ...

    def send(self, target, now):
        ...

    def close(self):
        ...


class RecordingRobot:
    """An offline sink. Feedback is supplied from the dataset, independent of targets."""
    def __init__(self):
        self.measurement = None
        self.commands = []
        self.closed = False

    def update_measurement(self, q, timestamp_s):
        self.measurement = Feedback(joint28(q), timestamp_s)

    def feedback(self):
        if self.measurement is None:
            raise RuntimeError("No recorded measurement supplied")
        return Feedback(self.measurement.q.copy(), self.measurement.received_monotonic_s)

    def send(self, target, now):
        if self.closed:
            raise RuntimeError("Recording sink closed")
        if not np.isfinite(now):
            raise ValueError("Invalid command timestamp")
        self.commands.append((float(now), joint28(target)))

    def close(self):
        self.closed = True  # No home pose, grasp, or synthetic feedback update.


class ImageClientCamera:
    """Wrap an explicitly connected ImageClient; construction itself is offline.

    The caller supplies reception timestamps from its subscriber callback. Poll
    time cannot establish freshness; the existing ImageClient caches frames.
    """
    def __init__(self, client, reception_timestamp):
        self.client = client
        self.reception_timestamp = reception_timestamp

    def read(self):
        config = self.client.get_cam_config()["head_camera"]
        if not config["binocular"]:
            raise RuntimeError("Training used the right half of binocular head RGB")
        frame = self.client.get_head_frame()
        if frame is None or frame.bgr is None:
            raise RuntimeError("Missing head frame")
        stamp = float(self.reception_timestamp(frame))
        if not np.isfinite(stamp):
            raise ValueError("Missing frame reception timestamp")
        return raw_bgr_to_rgb(frame.bgr, binocular=True), stamp


class JointOnlyRobot:
    """Inject a verified joint-only transport, never G1_29_ArmController itself.

    transport.read() returns Feedback in *raw hardware order*, and transport.write
    receives arm14 positions, arm14 feedforward torques, hand14 positions. It must
    own its lower-frequency-independent publisher/watchdog and touch no torso,
    legs or base. No controller is instantiated or mode switched here.
    """
    def __init__(self, transport, torque_provider, scale, offset, verified=False):
        if not verified or not getattr(transport, "joint_only", False):
            raise ValueError("Hardware mapping and a joint-only transport must first be verified")
        self.transport, self.torque_provider = transport, torque_provider
        self.scale, self.offset = joint28(scale), joint28(offset)
        if np.any(self.scale == 0) or not callable(torque_provider):
            raise ValueError("Explicit invertible coordinate map and torque provider required")
        self.closed = False

    def feedback(self):
        measured = self.transport.read()
        return Feedback(joint28((joint28(measured.q) - self.offset) / self.scale),
                        measured.received_monotonic_s)

    def send(self, target, now):
        if self.closed:
            raise RuntimeError("Robot adapter closed")
        raw = joint28(joint28(target) * self.scale + self.offset)
        tau = np.asarray(self.torque_provider(raw[:14]), dtype=np.float32)
        if tau.shape != (14,) or not np.isfinite(tau).all():
            raise ValueError("Feedforward torque must contain 14 finite values")
        self.transport.write(raw[:14], tau, raw[14:], now)

    def close(self):
        if not self.closed:
            self.closed = True
            self.transport.stop()  # Must stop publishing; explicitly no automatic home.


def gravity_provider(model, data, arm_q_indices, arm_v_indices, base_q):
    """Build a Pinocchio rnea provider after validating the actual robot model.

    Mirrors the reference IK's zero-velocity/zero-acceleration inverse dynamics;
    it computes feedforward without solving IK. base_q fixes non-arm posture and
    must reflect the measured supported robot posture on the actual hardware.
    """
    import pinocchio as pin
    qi, vi = np.asarray(arm_q_indices), np.asarray(arm_v_indices)
    if qi.shape != (14,) or vi.shape != (14,):
        raise ValueError("Explicit 14-joint q/v indices required")

    def calculate(target):
        q = np.asarray(base_q(), dtype=float).copy()
        if q.shape != (model.nq,) or not np.isfinite(q).all():
            raise ValueError("Invalid measured robot posture for gravity calculation")
        q[qi] = target
        return pin.rnea(model, data, q, np.zeros(model.nv), np.zeros(model.nv))[vi]
    return calculate


def dex3_command(raw_hand7, kp, kd, message_factory=None):
    """Build one HandCmd_ without publishing; order is the side's DDS motor order.

    Supply gains from the hardware review. Reference capture used 1.5 / 0.2,
    while later live code uses 3.0 / 0.2; neither is silently selected here.
    """
    q = np.asarray(raw_hand7, dtype=np.float32)
    if q.shape != (7,) or not np.isfinite(q).all() or not np.isfinite([kp, kd]).all() or kp <= 0 or kd < 0:
        raise ValueError("Expected finite raw hand7 and explicit valid gains")
    if message_factory is None:
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
        message_factory = unitree_hg_msg_dds__HandCmd_
    msg = message_factory()
    if len(msg.motor_cmd) != 7:
        raise ValueError("Expected seven Dex3 motors")
    for i, motor in enumerate(msg.motor_cmd):
        motor.mode = i | 0x10
        motor.q, motor.dq, motor.tau = float(q[i]), 0., 0.
        motor.kp, motor.kd = float(kp), float(kd)
    return msg
