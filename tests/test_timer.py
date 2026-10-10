import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from phonewatch.timer import UsageTimer
from phonewatch.storage import Store


class TimerTests(unittest.TestCase):
    def test_debounce_and_usage(self):
        timer = UsageTimer(5)
        self.assertEqual(timer.update(False, 0), (0, False))
        self.assertEqual(timer.update(False, 5), (5, True))
        self.assertEqual(timer.state, "absent")
        self.assertEqual(timer.update(False, 15), (10, False))
        timer.update(True, 16)
        timer.update(False, 17)  # transient phone detection must not stop timer
        self.assertEqual(timer.state, "absent")
        timer.update(True, 18)
        timer.update(True, 23)
        self.assertEqual(timer.state, "present")
        self.assertEqual(timer.update(True, 30), (0, False))

    def test_camera_failure_and_unknown_charge_time(self):
        timer = UsageTimer(0)
        timer.update(False, 0)
        self.assertEqual(timer.update(None, 60), (60, True))
        self.assertEqual(timer.update(False, 120), (60, True))
        self.assertEqual(timer.update(False, 125), (5, False))

    def test_restart_starts_unknown(self):
        self.assertEqual(UsageTimer().state, "unknown")

    def test_unknown_transitions_are_immediate_and_do_not_retroactively_charge(self):
        timer = UsageTimer(0)
        timer.update(True, 0)
        self.assertEqual(timer.update(None, 10), (0, True))
        self.assertEqual(timer.state, "unknown")
        self.assertEqual(timer.update(None, 15), (5, False))
        self.assertEqual(timer.update(True, 20), (5, True))
        self.assertEqual(timer.update(True, 25), (0, False))

    def test_state_colors(self):
        from phonewatch.__main__ import STATE_COLORS
        self.assertEqual(STATE_COLORS["present"], (0,255,0))
        self.assertEqual(STATE_COLORS["absent"], (0,0,255))
        self.assertEqual(STATE_COLORS["unknown"], (0,255,255))

    def test_midnight_and_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp))
            store.record(datetime(2026, 10, 11, 0, 0, 5), 10, "absent", True)
            self.assertEqual(store.total("2026-10-10"), 5)
            self.assertEqual(store.total("2026-10-11"), 5)
            store.close()
            store = Store(Path(tmp))
            self.assertEqual(store.total("2026-10-11"), 5)
            store.close()


if __name__ == "__main__":
    unittest.main()
