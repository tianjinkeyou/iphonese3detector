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
        self.assertEqual(timer.update(False, 5), (0, True))
        self.assertEqual(timer.state, "absent")
        self.assertEqual(timer.update(False, 15), (10, False))
        timer.update(True, 16)
        timer.update(False, 17)  # transient phone detection must not stop timer
        self.assertEqual(timer.state, "absent")
        timer.update(True, 18)
        timer.update(True, 23)
        self.assertEqual(timer.state, "present")
        self.assertEqual(timer.update(True, 30), (0, False))

    def test_camera_failure_does_not_charge_gap(self):
        timer = UsageTimer(0)
        timer.update(False, 0)
        self.assertEqual(timer.update(None, 60), (0, True))
        self.assertEqual(timer.update(False, 120), (0, True))
        self.assertEqual(timer.update(False, 125), (5, False))

    def test_restart_starts_unknown(self):
        self.assertEqual(UsageTimer().state, "unknown")

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
