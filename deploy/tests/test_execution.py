import unittest
from types import SimpleNamespace

import numpy as np

from deploy.g1dp.adapters import Feedback, JointOnlyRobot, RecordingRobot, dex3_command
from deploy.g1dp.control import ActionQueue, Phase, SafetyFilter
from deploy.g1dp.observations import History, Observation, JOINT_NAMES, raw_bgr_to_rgb
from deploy.g1dp.session import Session


def frame(index, episode="a"):
    return Observation(np.full((224, 224, 3), index, np.float32), np.full(28, index),
                       index / 30, episode, index)


class ExecutionTests(unittest.TestCase):
    def test_history_start_order_episode(self):
        h = History()
        self.assertEqual(h.push(frame(0))["agent_pos"][:, 0].tolist(), [0, 0])
        self.assertEqual(h.push(frame(1))["agent_pos"][:, 0].tolist(), [0, 1])
        self.assertEqual(h.push(frame(0, "b"))["agent_pos"][:, 0].tolist(), [0, 0])
        with self.assertRaises(ValueError):
            h.push(frame(0, "b"))
        self.assertEqual(JOINT_NAMES[17], "left_hand_middle_0_joint")
        self.assertEqual(JOINT_NAMES[24], "right_hand_index_0_joint")

    def test_right_eye_bgr(self):
        bgr = np.zeros((10, 20, 3), np.uint8)
        bgr[:, :10] = [10, 20, 30]
        bgr[:, 10:] = [40, 50, 60]
        np.testing.assert_equal(raw_bgr_to_rgb(bgr, True)[0, 0], [60, 50, 40])

    def test_queue_timing_pause_stop_and_expiration(self):
        q = ActionQueue(4, 30, .01)
        a = np.tile(np.arange(15, dtype=np.float32)[:, None], (1, 28))
        with self.assertRaises(RuntimeError):
            q.replace(a, 0)
        q.start(); q.replace(a, 0)
        self.assertEqual(q.pop(0)[1], 0)
        self.assertIsNone(q.pop(.001))
        self.assertEqual(q.pop(1/30)[1], 1)
        self.assertEqual(q.pop(3/30)[1], 3)
        self.assertEqual(q.dropped, 1)
        q.replace(a, 0)
        self.assertIsNone(q.pop(1))
        self.assertEqual(len(q.queue), 0)
        q.pause(); self.assertIsNone(q.pop(0))
        q.stop()
        with self.assertRaises(RuntimeError): q.start()
        with self.assertRaises(ValueError): ActionQueue(16)

    def test_safety_limits_rate_stale_finite_and_reset(self):
        f = SafetyFilter([-1]*28, [1]*28, [.3]*28, "test_only")
        sent, info = f.apply(np.full(28, 2), np.zeros(28), 0, 0)
        np.testing.assert_allclose(sent, .01)
        self.assertTrue(info["position_clipped"])
        with self.assertRaises(RuntimeError): f.apply(sent, np.zeros(28), 0, 1)
        with self.assertRaises(ValueError): f.apply([float("nan")]*28, np.zeros(28), 1, 1)
        with self.assertRaises(RuntimeError): f.apply(sent, np.full(28, .9), 1/30, 1/30)
        f.reset()
        with self.assertRaises(RuntimeError): f.apply(sent, np.zeros(28), 2, 1)

    def test_recording_sink_does_not_fake_feedback(self):
        robot = RecordingRobot()
        robot.update_measurement(np.zeros(28), 1)
        robot.send(np.ones(28), 1)
        np.testing.assert_equal(robot.feedback().q, 0)
        robot.close()
        with self.assertRaises(RuntimeError): robot.send(np.ones(28), 2)

    def test_coordinate_adapter_and_no_implicit_zero_torque(self):
        writes = []
        transport = SimpleNamespace(joint_only=True, read=lambda: Feedback(np.ones(28), 0),
                                    write=lambda *args: writes.append(args), stop=lambda: None)
        with self.assertRaises(ValueError):
            JointOnlyRobot(transport, None, [1]*28, [0]*28)
        robot = JointOnlyRobot(transport, lambda q: np.ones(14)*.2, [2]*28, [.5]*28, verified=True)
        np.testing.assert_allclose(robot.feedback().q, .25)
        robot.send(np.zeros(28), 0)
        np.testing.assert_allclose(writes[0][0], .5)
        np.testing.assert_allclose(writes[0][1], .2)
        robot.close()

    def test_hand_message_no_publish(self):
        factory = lambda: SimpleNamespace(motor_cmd=[SimpleNamespace() for _ in range(7)])
        msg = dex3_command(np.arange(7), 1.5, .2, factory)
        self.assertEqual([m.mode for m in msg.motor_cmd], list(range(16,23)))
        self.assertEqual([m.q for m in msg.motor_cmd], list(range(7)))
        self.assertTrue(all(m.kp == 1.5 and m.kd == .2 for m in msg.motor_cmd))

    def test_session_send_pause_and_exception_cleanup(self):
        now = [0.]
        camera = SimpleNamespace(read=lambda: (frame(0).image, now[0]))
        robot = RecordingRobot()
        robot.update_measurement(np.zeros(28), 0)
        policy = SimpleNamespace(predict=lambda arrays: ({"action": np.ones((15,28),np.float32)}, 0.))
        safety = SafetyFilter([-2]*28, [2]*28, [.3]*28, "test")
        session = Session(policy, camera, robot, safety, clock=lambda: now[0])
        session.start()
        event = session.tick()
        self.assertEqual(event["outcome"], "sent")
        np.testing.assert_allclose(event["sent_target"], .01)
        session.pause()
        self.assertEqual(session.tick()["phase"], "paused")
        self.assertEqual(len(robot.commands), 1)
        session.start()
        now[0] = 1.
        with self.assertRaises(RuntimeError): session.tick()
        self.assertTrue(robot.closed)

    def test_session_rechecks_feedback_and_rejects_latency(self):
        now = [0.]
        camera = SimpleNamespace(read=lambda: (frame(0).image, now[0]))
        robot = RecordingRobot()
        robot.update_measurement(np.zeros(28), 0)
        def slow(arrays):
            now[0] += .2
            return {"action": np.ones((15,28), np.float32)}, .2
        policy = SimpleNamespace(predict=slow)
        safety = SafetyFilter([-2]*28, [2]*28, [.3]*28, "test")
        session = Session(policy, camera, robot, safety, clock=lambda: now[0])
        self.assertEqual(session.tick()["phase"], "paused")
        session.start()
        # Four selected slots are already expired: no compressed burst of targets.
        self.assertEqual(session.tick()["outcome"], "wait_or_replan")
        self.assertEqual(len(robot.commands), 0)
        now[0] = 1.
        with self.assertRaises(RuntimeError): session.tick()
        self.assertEqual(session.queue.phase, Phase.STOPPED)
        self.assertTrue(robot.closed)


if __name__ == "__main__":
    unittest.main()
