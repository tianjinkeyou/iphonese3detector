import json
import unittest

from phonewatch.pickup import PickupTracker


class PickupTrackerTests(unittest.TestCase):
    def arm(self, tracker, start=0):
        self.assertEqual(tracker.update(True, start), [])
        events = tracker.update(True, start + tracker.delay)
        self.assertEqual([event["kind"] for event in events], ["baseline"])
        self.assertEqual(tracker.state, "present")
        return events[0]

    def test_startup_absent_is_baseline_and_first_appearance_is_not_return(self):
        tracker = PickupTracker()
        self.assertEqual(tracker.update(False, 0), [])
        baseline = tracker.update(False, 1)
        self.assertEqual(baseline[0]["kind"], "baseline")
        self.assertEqual(baseline[0]["from_state"], "unknown")
        self.assertEqual(baseline[0]["to_state"], "absent")
        self.assertEqual(tracker.update(False, 100), [])
        self.assertEqual(tracker.update(True, 101), [])
        self.assertEqual(tracker.update(True, 102), [])
        self.assertEqual(tracker.state, "present")
        tracker.update(False, 103)
        self.assertEqual(tracker.update(False, 104)[0]["kind"], "pickup")

    def test_complete_cycles_count_once_and_report_confirmed_away_duration(self):
        tracker = PickupTracker()
        self.arm(tracker)
        self.assertEqual(tracker.update(False, 3), [])
        pickup = tracker.update(False, 4)[0]
        self.assertEqual(pickup, {
            "kind": "pickup", "from_state": "present", "to_state": "absent",
            "unknown_seconds": 0.0, "confirmed_after_seconds": 1,
        })
        self.assertEqual(tracker.update(False, 15), [])
        self.assertEqual(tracker.update(True, 16), [])
        returned = tracker.update(True, 17)[0]
        self.assertEqual(returned["kind"], "return")
        self.assertEqual(returned["away_seconds"], 13)
        self.assertEqual(tracker.update(True, 20), [])
        tracker.update(False, 21)
        self.assertEqual(tracker.update(False, 22)[0]["kind"], "pickup")
        tracker.update(True, 23)
        self.assertEqual(tracker.update(True, 24)[0]["away_seconds"], 2)

    def test_short_jitter_does_not_pick_up_or_return(self):
        tracker = PickupTracker()
        self.arm(tracker)
        self.assertEqual(tracker.update(False, 2), [])
        self.assertEqual(tracker.update(True, 2.9), [])
        self.assertEqual(tracker.state, "present")
        tracker.update(False, 3)
        self.assertEqual(tracker.update(False, 4)[0]["kind"], "pickup")
        tracker.update(True, 5)
        self.assertEqual(tracker.update(False, 5.9), [])
        self.assertEqual(tracker.state, "absent")
        tracker.update(True, 7)
        self.assertEqual(tracker.update(True, 8)[0]["kind"], "return")

    def test_unknown_interrupts_delay_but_bridges_short_gap(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(False, 2)
        self.assertEqual(tracker.update(None, 2.5), [])
        self.assertEqual(tracker.state, "unknown")
        self.assertEqual(tracker.update(False, 3), [])
        self.assertEqual(tracker.update(False, 3.5), [])
        pickup = tracker.update(False, 4)[0]
        self.assertEqual(pickup["kind"], "pickup")
        self.assertEqual(pickup["unknown_seconds"], .5)
        self.assertEqual(pickup["confirmed_after_seconds"], 1)
        tracker.update(None, 5)
        tracker.update(True, 6)
        returned = tracker.update(True, 7)[0]
        self.assertEqual(returned["kind"], "return")
        self.assertEqual(returned["unknown_seconds"], 1)
        self.assertEqual(returned["away_seconds"], 3)

    def test_short_unknown_recovery_to_same_baseline_does_not_emit_event(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2)
        self.assertEqual(tracker.update(True, 3), [])
        self.assertEqual(tracker.state, "present")
        tracker.update(False, 4)
        pickup = tracker.update(False, 5)[0]
        self.assertEqual(pickup["unknown_seconds"], 0)

    def test_long_unknown_clears_baseline_and_records_change_without_pickup(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2)
        tracker.update(None, 5)
        self.assertEqual(tracker.update(False, 7), [])
        event = tracker.update(False, 8)[0]
        self.assertEqual(event["kind"], "unobserved_change")
        self.assertEqual(event["from_state"], "present")
        self.assertEqual(event["to_state"], "absent")
        self.assertEqual(event["unknown_seconds"], 5)
        tracker.update(True, 9)
        self.assertEqual(tracker.update(True, 10), [])
        tracker.update(False, 11)
        self.assertEqual(tracker.update(False, 12)[0]["kind"], "pickup")

    def test_long_unknown_detected_on_first_valid_frame_and_same_state_recorded(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2)
        tracker.update(True, 20)
        event = tracker.update(True, 21)[0]
        self.assertEqual(event["kind"], "unobserved_change")
        self.assertEqual(event["from_state"], "present")
        self.assertEqual(event["to_state"], "present")
        self.assertEqual(event["unknown_seconds"], 18)

    def test_long_unknown_drops_previous_cycle_without_fabricating_away_duration(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(False, 2)
        tracker.update(False, 3)
        tracker.update(None, 4)
        tracker.update(True, 10)
        event = tracker.update(True, 11)[0]
        self.assertEqual(event["kind"], "unobserved_change")
        self.assertNotIn("away_seconds", event)
        tracker.update(False, 12)
        self.assertEqual(tracker.update(False, 13)[0]["kind"], "pickup")

    def test_total_uncertainty_over_gap_is_not_a_trusted_transition(self):
        tracker = PickupTracker(delay=1, unknown_gap=2)
        self.arm(tracker)
        tracker.update(None, 2)
        tracker.update(False, 3.5)
        tracker.update(None, 4)
        tracker.update(False, 5)
        event = tracker.update(False, 6)[0]
        self.assertEqual(event["kind"], "unobserved_change")
        self.assertEqual(event["unknown_seconds"], 2.5)

    def test_gap_boundary_is_inclusive(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2)
        tracker.update(False, 4)
        self.assertEqual(tracker.update(False, 5)[0]["kind"], "pickup")

    def test_initial_unknown_is_baseline_only(self):
        tracker = PickupTracker()
        tracker.update(None, 0)
        tracker.update(None, 10)
        tracker.update(False, 20)
        baseline = tracker.update(False, 21)[0]
        self.assertEqual(baseline["kind"], "baseline")
        self.assertEqual(baseline["unknown_seconds"], 20)

    def test_reset_discards_pickup_cycle_and_pending_unknown(self):
        for next_presence in (True, False):
            with self.subTest(next_presence=next_presence):
                tracker = PickupTracker()
                self.arm(tracker)
                tracker.update(False, 2)
                tracker.update(False, 3)
                tracker.update(None, 4)
                tracker.reset(10)
                self.assertEqual(tracker.state, "unknown")
                self.assertEqual(tracker.update(next_presence, 20), [])
                event = tracker.update(next_presence, 21)[0]
                self.assertEqual(event["kind"], "baseline")
                self.assertEqual(event["unknown_seconds"], 0)
                self.assertNotIn("away_seconds", event)

    def test_reset_discards_unconfirmed_candidate(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(False, 2)
        tracker.reset(2.5)
        tracker.update(False, 3)
        self.assertEqual(tracker.update(False, 4)[0]["kind"], "baseline")

    def test_slow_frames_report_actual_confirmation_duration(self):
        tracker = PickupTracker()
        tracker.update(True, 0)
        self.assertEqual(tracker.update(True, 4)[0]["confirmed_after_seconds"], 4)
        tracker.update(False, 5)
        self.assertEqual(tracker.update(False, 10)[0]["confirmed_after_seconds"], 5)

    def test_zero_delay_gap_and_json_primitive_events(self):
        tracker = PickupTracker(delay=0, unknown_gap=0)
        self.assertEqual(tracker.update(True, 0)[0]["kind"], "baseline")
        self.assertEqual(tracker.update(False, 1)[0]["kind"], "pickup")
        self.assertEqual(tracker.update(True, 2)[0]["kind"], "return")
        tracker.update(None, 3)
        event = tracker.update(False, 4)[0]
        self.assertEqual(event["kind"], "unobserved_change")
        self.assertEqual(json.loads(json.dumps(event)), event)

    def test_strong_empty_signal_recovers_one_pickup_after_long_unknown(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2)
        tracker.update(None, 8)
        self.assertEqual(tracker.update(False, 10, resolve_gap=True), [])
        event = tracker.update(False, 11, resolve_gap=True)[0]
        self.assertEqual(event, {
            "kind": "pickup", "from_state": "present", "to_state": "absent",
            "unknown_seconds": 8, "confirmed_after_seconds": 1,
            "gap_recovered": True, "timing_uncertain": True,
        })
        self.assertEqual(tracker.update(False, 12, resolve_gap=True), [])
        self.assertEqual(tracker.update(False, 30, resolve_gap=True), [])
        tracker.update(True, 31)
        returned = tracker.update(True, 32)[0]
        self.assertEqual(returned["kind"], "return")
        self.assertEqual(returned["away_seconds"], 21)
        self.assertEqual(json.loads(json.dumps(event)), event)

    def test_resolve_gap_flag_on_unknown_never_counts_pickup(self):
        tracker = PickupTracker()
        self.arm(tracker)
        for now in (2, 5, 20, 100):
            self.assertEqual(tracker.update(None, now, resolve_gap=True), [])
            self.assertEqual(tracker.state, "unknown")
        tracker.update(False, 101)
        event = tracker.update(False, 102)[0]
        self.assertEqual(event["kind"], "unobserved_change")
        self.assertNotIn("gap_recovered", event)

    def test_recovered_empty_candidate_must_survive_continuous_confirmation(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2)
        tracker.update(False, 10, resolve_gap=True)
        self.assertEqual(tracker.update(None, 10.5, resolve_gap=True), [])
        self.assertEqual(tracker.update(False, 11, resolve_gap=True), [])
        self.assertEqual(tracker.update(False, 11.5, resolve_gap=True), [])
        event = tracker.update(False, 12, resolve_gap=True)[0]
        self.assertEqual(event["kind"], "pickup")
        self.assertEqual(event["unknown_seconds"], 8.5)
        self.assertEqual(event["confirmed_after_seconds"], 1)

    def test_reset_destroys_gap_even_with_strong_empty_signal(self):
        for reset_reason in ("camera", "calibration"):
            with self.subTest(reset_reason=reset_reason):
                tracker = PickupTracker()
                self.arm(tracker)
                tracker.update(None, 2)
                tracker.update(None, 10)
                tracker.reset(11)
                tracker.update(False, 20, resolve_gap=True)
                event = tracker.update(False, 21, resolve_gap=True)[0]
                self.assertEqual(event["kind"], "baseline")
                self.assertNotIn("gap_recovered", event)

    def test_absent_baseline_gap_cannot_create_recovered_pickup(self):
        tracker = PickupTracker()
        tracker.update(False, 0)
        tracker.update(False, 1)
        tracker.update(None, 2)
        tracker.update(False, 10, resolve_gap=True)
        event = tracker.update(False, 11, resolve_gap=True)[0]
        self.assertEqual(event["kind"], "unobserved_change")
        self.assertEqual(event["from_state"], "absent")
        self.assertNotIn("gap_recovered", event)
        tracker.update(True, 12)
        self.assertEqual(tracker.update(True, 13), [])

    def test_strong_present_recovery_does_not_fabricate_pickup(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2)
        tracker.update(True, 10, resolve_gap=True)
        event = tracker.update(True, 11, resolve_gap=True)[0]
        self.assertEqual(event["kind"], "unobserved_change")
        self.assertEqual(event["to_state"], "present")
        self.assertNotIn("gap_recovered", event)

    def test_gap_recovery_requires_flag_at_confirmation(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2)
        tracker.update(False, 10, resolve_gap=True)
        event = tracker.update(False, 11)[0]
        self.assertEqual(event["kind"], "unobserved_change")

    def test_normal_pickup_with_flag_has_no_gap_recovery_fields(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(False, 2, resolve_gap=True)
        event = tracker.update(False, 3, resolve_gap=True)[0]
        self.assertEqual(event, {
            "kind": "pickup", "from_state": "present", "to_state": "absent",
            "unknown_seconds": 0, "confirmed_after_seconds": 1,
        })

    def test_resolve_gap_must_be_boolean_and_validation_does_not_advance_time(self):
        tracker = PickupTracker()
        tracker.update(True, 0)
        for invalid_flag in (None, 0, 1, "true", []):
            with self.subTest(invalid_flag=invalid_flag), self.assertRaises(ValueError):
                tracker.update(False, 10, resolve_gap=invalid_flag)
        self.assertEqual(tracker.update(True, 1)[0]["kind"], "baseline")

    def test_one_strong_empty_frame_counts_quick_removal_and_stable_return_once(self):
        tracker = PickupTracker()
        self.arm(tracker)
        pickup = tracker.update(False, 2, immediate_empty=True)[0]
        self.assertEqual(pickup["kind"], "pickup")
        self.assertTrue(pickup["rapid_empty"])
        self.assertEqual(pickup["confirmed_after_seconds"], 0)
        self.assertEqual(tracker.state, "absent")
        # The replacement appears before the usual absence debounce would pass.
        self.assertEqual(tracker.update(True, 2.1, immediate_empty=True), [])
        self.assertEqual(tracker.update(True, 2.9), [])
        returned = tracker.update(True, 3.1)[0]
        self.assertEqual(returned["kind"], "return")
        self.assertAlmostEqual(returned["away_seconds"], 1.1)
        self.assertNotIn("rapid_empty", returned)
        self.assertEqual(tracker.update(True, 4), [])
        self.assertEqual(json.loads(json.dumps(pickup)), pickup)

    def test_repeated_strong_empty_frames_and_brief_return_do_not_count_again(self):
        tracker = PickupTracker()
        self.arm(tracker)
        self.assertEqual(tracker.update(False, 2, immediate_empty=True)[0]["kind"], "pickup")
        for now in (2.1, 3, 4):
            self.assertEqual(tracker.update(False, now, immediate_empty=True), [])
        tracker.update(True, 5)
        self.assertEqual(tracker.update(False, 5.1, immediate_empty=True), [])
        tracker.update(True, 6)
        self.assertEqual(tracker.update(True, 7)[0]["kind"], "return")
        self.assertEqual(tracker.update(False, 7.1, immediate_empty=True)[0]["kind"], "pickup")

    def test_short_unknown_keeps_present_baseline_for_immediate_empty(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2)
        event = tracker.update(False, 2.1, immediate_empty=True)[0]
        self.assertEqual(event["kind"], "pickup")
        self.assertTrue(event["rapid_empty"])
        self.assertAlmostEqual(event["unknown_seconds"], .1)
        self.assertNotIn("gap_recovered", event)

    def test_long_unknown_requires_gap_resolution_for_immediate_empty(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2)
        tracker.update(None, 8)
        self.assertEqual(tracker.update(False, 10, immediate_empty=True), [])
        event = tracker.update(False, 10.1, resolve_gap=True, immediate_empty=True)[0]
        self.assertEqual(event["kind"], "pickup")
        self.assertTrue(event["rapid_empty"])
        self.assertTrue(event["gap_recovered"])
        self.assertTrue(event["timing_uncertain"])
        self.assertAlmostEqual(event["confirmed_after_seconds"], .1)
        self.assertEqual(event["unknown_seconds"], 8)

    def test_startup_and_reset_cannot_create_immediate_empty_pickup(self):
        for after_reset in (False, True):
            with self.subTest(after_reset=after_reset):
                tracker = PickupTracker()
                if after_reset:
                    self.arm(tracker)
                    tracker.update(None, 2)
                    tracker.update(None, 8)
                    tracker.reset(9)
                self.assertEqual(tracker.update(False, 10, resolve_gap=True, immediate_empty=True), [])
                event = tracker.update(False, 11, resolve_gap=True, immediate_empty=True)[0]
                self.assertEqual(event["kind"], "baseline")
                self.assertNotIn("rapid_empty", event)
                self.assertNotIn("gap_recovered", event)

    def test_absent_gap_does_not_become_immediate_pickup(self):
        tracker = PickupTracker()
        tracker.update(False, 0)
        tracker.update(False, 1)
        tracker.update(None, 2)
        tracker.update(None, 8)
        self.assertEqual(tracker.update(False, 10, resolve_gap=True, immediate_empty=True), [])
        event = tracker.update(False, 11, resolve_gap=True, immediate_empty=True)[0]
        self.assertEqual(event["kind"], "unobserved_change")
        self.assertNotIn("rapid_empty", event)

    def test_unknown_and_present_never_use_immediate_empty_bypass(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(None, 2, resolve_gap=True, immediate_empty=True)
        self.assertEqual(tracker.update(None, 8, resolve_gap=True, immediate_empty=True), [])
        self.assertEqual(tracker.update(True, 10, resolve_gap=True, immediate_empty=True), [])
        event = tracker.update(True, 11, resolve_gap=True, immediate_empty=True)[0]
        self.assertEqual(event["kind"], "unobserved_change")
        self.assertNotIn("rapid_empty", event)

    def test_normal_debounced_empty_does_not_claim_rapid_confirmation(self):
        tracker = PickupTracker()
        self.arm(tracker)
        tracker.update(False, 2)
        event = tracker.update(False, 3, immediate_empty=True)[0]
        self.assertEqual(event["kind"], "pickup")
        self.assertNotIn("rapid_empty", event)
        self.assertEqual(event["confirmed_after_seconds"], 1)

    def test_immediate_empty_must_be_boolean_without_advancing_clock(self):
        tracker = PickupTracker()
        tracker.update(True, 0)
        for invalid_flag in (None, 0, 1, "true", []):
            with self.subTest(invalid_flag=invalid_flag), self.assertRaises(ValueError):
                tracker.update(False, 10, immediate_empty=invalid_flag)
        self.assertEqual(tracker.update(True, 1)[0]["kind"], "baseline")

    def test_invalid_configuration_and_timestamps_are_rejected(self):
        for value in (-1, float("inf"), float("nan")):
            for option in ("delay", "unknown_gap"):
                with self.subTest(value=value, option=option), self.assertRaises(ValueError):
                    PickupTracker(**{option: value})
        tracker = PickupTracker()
        tracker.update(True, 10)
        for invalid_time in (9, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                tracker.update(False, invalid_time)
            with self.assertRaises(ValueError):
                tracker.reset(invalid_time)
        for invalid_presence in (0, 1, "present", []):
            with self.assertRaises(ValueError):
                tracker.update(invalid_presence, 11)
        # Validation failure must not advance the monotonic clock.
        self.assertEqual(tracker.update(True, 10), [])


if __name__ == "__main__":
    unittest.main()
