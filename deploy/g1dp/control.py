"""Synchronous receding-horizon scheduling; no threads or device initialization."""
from collections import deque
from enum import Enum

import numpy as np

from .observations import joint28


def positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")
    return float(value)


class Phase(Enum):
    PAUSED = "paused"
    RUNNING = "running"
    STOPPED = "stopped"


class ActionQueue:
    """Returned action[0] aligns with the newest observation, not its next frame.

    Expired slots are dropped, never burst-sent or shifted into the future.
    Offline virtual time may explicitly exclude inference cost; live time may not.
    """
    def __init__(self, execute_steps=4, target_hz=30, max_lateness_s=0.05):
        if isinstance(execute_steps, bool) or not isinstance(execute_steps, int) or not 1 <= execute_steps <= 15:
            raise ValueError("execute_steps must be an integer in [1,15]")
        self.execute_steps = execute_steps
        self.dt = 1 / positive(target_hz, "target_hz")
        self.max_lateness = positive(max_lateness_s, "max_lateness_s")
        self.phase = Phase.PAUSED
        self.queue = deque()
        self.dropped = 0

    def start(self):
        if self.phase == Phase.STOPPED:
            raise RuntimeError("Stopped session cannot restart")
        self.queue.clear()
        self.phase = Phase.RUNNING

    def pause(self):
        if self.phase != Phase.STOPPED:
            self.phase = Phase.PAUSED
        self.queue.clear()

    def stop(self):
        self.queue.clear()
        self.phase = Phase.STOPPED

    def replace(self, action, observation_time_s):
        if self.phase != Phase.RUNNING:
            raise RuntimeError("Queue is not running")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (15, 28) or not np.isfinite(action).all() or not np.isfinite(observation_time_s):
            raise ValueError("Invalid action chunk or timestamp")
        self.queue = deque((observation_time_s + i * self.dt, i, row.copy())
                           for i, row in enumerate(action[:self.execute_steps]))

    def pop(self, now):
        if not np.isfinite(now):
            raise ValueError("Invalid clock")
        if self.phase != Phase.RUNNING:
            return None
        while self.queue and now - self.queue[0][0] > self.max_lateness:
            self.queue.popleft()
            self.dropped += 1
        if self.queue and self.queue[0][0] <= now:
            return self.queue.popleft()
        return None


class SafetyFilter:
    """Bounds must have a stated source; hardware execution needs verified bounds.

    Velocity bounds apply to command increments, using capped elapsed time. This
    limits jumps after a long pause; it does not certify actual motor velocity.
    """
    def __init__(self, lower, upper, velocity, source, max_feedback_age_s=0.15, max_dt_s=1/30):
        self.lower, self.upper, self.velocity = map(joint28, (lower, upper, velocity))
        if np.any(self.lower >= self.upper) or np.any(self.velocity <= 0) or not source:
            raise ValueError("Invalid sourced position/velocity bounds")
        self.source = source
        self.max_age = positive(max_feedback_age_s, "max_feedback_age_s")
        self.max_dt = positive(max_dt_s, "max_dt_s")
        self.previous = None
        self.previous_time = None

    def reset(self):
        self.previous = self.previous_time = None

    def apply(self, prediction, measured, feedback_time_s, now):
        prediction, measured = joint28(prediction), joint28(measured)
        if not np.isfinite([now, feedback_time_s]).all() or not 0 <= now - feedback_time_s <= self.max_age:
            raise RuntimeError("Stale or future-dated actual feedback")
        if np.any(measured < self.lower) or np.any(measured > self.upper):
            raise RuntimeError("Feedback outside configured bounds; inspect coordinates")
        if self.previous_time is not None and now <= self.previous_time:
            raise RuntimeError("Command timestamps must increase")
        dt = self.max_dt if self.previous_time is None else min(now - self.previous_time, self.max_dt)
        anchor = measured if self.previous is None else self.previous
        bounded = np.clip(prediction, self.lower, self.upper)
        # Intersect constraints. Sequential clipping could silently violate the
        # previous-command velocity bound after feedback jumps.
        low = np.maximum(self.lower, np.maximum(anchor - self.velocity * dt,
                                                measured - self.velocity * self.max_dt))
        high = np.minimum(self.upper, np.minimum(anchor + self.velocity * dt,
                                                 measured + self.velocity * self.max_dt))
        if np.any(low > high):
            raise RuntimeError("Feedback jump leaves no feasible rate-limited target")
        sent = np.clip(bounded, low, high)
        self.previous, self.previous_time = sent.copy(), now
        return sent, {"position_clipped": bool(np.any(bounded != prediction)),
                      "rate_clipped": bool(np.any(sent != bounded)), "limit_source": self.source}
