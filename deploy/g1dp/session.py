"""Synchronous hardware integration seam; caller owns the outer periodic loop."""
import time

import numpy as np

from .control import ActionQueue, Phase, positive
from .observations import History, Observation


class Session:
    def __init__(self, policy, camera, robot, safety, execute_steps=4, target_hz=30,
                 max_prediction_age_s=0.5, max_camera_age_s=0.15,
                 max_sensor_skew_s=0.05, clock=time.monotonic):
        self.policy, self.camera, self.robot, self.safety = policy, camera, robot, safety
        self.queue = ActionQueue(execute_steps, target_hz)
        self.history = History()
        self.clock = clock
        self.max_prediction_age = positive(max_prediction_age_s, "max_prediction_age_s")
        self.max_camera_age = positive(max_camera_age_s, "max_camera_age_s")
        self.max_sensor_skew = positive(max_sensor_skew_s, "max_sensor_skew_s")
        self.prediction_time = None
        self.index = 0

    def start(self):
        self.queue.start()
        self.history.reset()
        self.safety.reset()

    def pause(self):
        self.queue.pause()
        self.history.reset()
        self.safety.reset()
        # No writes while paused. A hardware transport must implement its own
        # reviewed hold/release policy and watchdog; this is not a physical e-stop.

    def close(self):
        self.queue.stop()
        self.robot.close()

    def tick(self):
        if self.queue.phase != Phase.RUNNING:
            return {"phase": self.queue.phase.value}
        try:
            image, camera_time = self.camera.read()
            feedback = self.robot.feedback()
            now = self.clock()
            if (not np.isfinite([now, camera_time, feedback.received_monotonic_s]).all()
                    or not 0 <= now - camera_time <= self.max_camera_age
                    or not 0 <= now - feedback.received_monotonic_s <= self.safety.max_age
                    or abs(camera_time - feedback.received_monotonic_s) > self.max_sensor_skew):
                raise RuntimeError("Stale camera/feedback or excessive observation skew")
            arrays = self.history.push(Observation(image, feedback.q, camera_time,
                                                  "session", self.index, timestamp_kind="local_reception"))
            self.index += 1
            event = {"phase": "running", "measured_state": feedback.q.tolist(),
                     "camera_time_s": camera_time, "feedback_time_s": feedback.received_monotonic_s}
            if not self.queue.queue:
                prediction, latency = self.policy.predict(arrays)
                self.prediction_time = camera_time
                self.queue.replace(prediction["action"], camera_time)
                event.update(inference_s=latency, raw_prediction=prediction["action"].tolist())
            now = self.clock()
            age = now - self.prediction_time
            event["prediction_age_s"] = age
            if not 0 <= age <= self.max_prediction_age:
                self.queue.queue.clear()
                return dict(event, outcome="prediction_expired_replan")
            item = self.queue.pop(now)
            if item is None:
                return dict(event, outcome="wait_or_replan", dropped=self.queue.dropped)
            # Inference may block. Re-read feedback immediately before any send.
            feedback = self.robot.feedback()
            now = self.clock()
            due, index, raw = item
            if now - self.prediction_time > self.max_prediction_age:
                self.queue.queue.clear()
                return dict(event, outcome="prediction_expired_replan")
            if now - due > self.queue.max_lateness:
                return dict(event, outcome="target_expired_skip")
            sent, filtering = self.safety.apply(raw, feedback.q, feedback.received_monotonic_s, now)
            self.robot.send(sent, now)
            event.update(outcome="sent", action_index=index, due_time_s=due,
                         sent_time_s=now, raw_target=raw.tolist(), sent_target=sent.tolist(), filtering=filtering,
                         send_feedback_state=feedback.q.tolist(),
                         send_feedback_time_s=feedback.received_monotonic_s,
                         send_feedback_age_s=now-feedback.received_monotonic_s)
            return event
        except BaseException:
            self.close()
            raise
