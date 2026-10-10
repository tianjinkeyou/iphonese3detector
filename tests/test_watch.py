import unittest

from phonewatch.watch import WatchTimer, hand_near_polygon


class WatchTimerTests(unittest.TestCase):
    def test_continuous_hand_presence_activates_without_retroactive_charge(self):
        timer = WatchTimer()
        self.assertEqual(timer.update(True, True, 0), (0, False, False))
        self.assertEqual(timer.update(True, True, 2), (0, False, False))
        self.assertEqual(timer.update(True, True, 3), (0, True, True))
        self.assertTrue(timer.active)
        self.assertEqual(timer.update(True, True, 10), (7, False, False))

    def test_short_hand_presence_does_not_trigger(self):
        timer = WatchTimer()
        timer.update(True, True, 0)
        self.assertEqual(timer.update(False, True, 2), (0, False, False))
        timer.update(True, True, 3)
        self.assertEqual(timer.update(True, True, 5), (0, False, False))
        self.assertEqual(timer.update(True, True, 6), (0, True, True))

    def test_hand_moving_away_charges_previous_active_interval(self):
        timer = WatchTimer(0)
        timer.update(True, True, 0)
        self.assertEqual(timer.update(False, True, 4), (4, False, True))
        self.assertFalse(timer.active)
        self.assertEqual(timer.update(False, True, 9), (0, False, False))

    def test_unknown_hand_or_camera_drops_uncertain_interval(self):
        timer = WatchTimer(0)
        timer.update(True, True, 0)
        self.assertEqual(timer.update(None, True, 60), (0, False, True))
        self.assertEqual(timer.update(None, False, 120), (0, False, False))
        self.assertEqual(timer.update(True, True, 180), (0, True, True))
        self.assertEqual(timer.update(True, True, 182), (2, False, False))

    def test_phone_missing_or_unknown_drops_interval_and_restarts_delay(self):
        timer = WatchTimer()
        timer.update(True, True, 0)
        timer.update(True, True, 3)
        self.assertEqual(timer.update(True, False, 100), (0, False, True))
        self.assertEqual(timer.update(True, True, 200), (0, False, False))
        self.assertEqual(timer.update(True, True, 203), (0, True, True))
        self.assertEqual(timer.update(True, True, 204), (1, False, False))

    def test_uncertain_observation_interrupts_pending_delay(self):
        timer = WatchTimer()
        timer.update(True, True, 0)
        timer.update(None, True, 2)
        timer.update(True, True, 3)
        self.assertEqual(timer.update(True, True, 5), (0, False, False))
        self.assertEqual(timer.update(True, True, 6), (0, True, True))

    def test_calibration_reset_discards_interval(self):
        timer = WatchTimer()
        timer.update(True, True, 0)
        timer.update(True, True, 3)
        timer.reset(20)
        self.assertFalse(timer.active)
        self.assertIsNone(timer.candidate_since)
        self.assertEqual(timer.update(True, True, 100), (0, False, False))
        self.assertEqual(timer.update(True, True, 103), (0, True, True))

    def test_monotonic_timestamps_and_delay_validation(self):
        timer = WatchTimer()
        timer.update(True, True, 10)
        with self.assertRaises(ValueError):
            timer.update(False, True, 9)
        with self.assertRaises(ValueError):
            timer.reset(9)
        with self.assertRaises(ValueError):
            timer.update(True, True, float("nan"))
        for delay in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                WatchTimer(delay)


class HandGeometryTests(unittest.TestCase):
    shape = (100, 200, 3)
    rectangle = [(80, 20), (120, 20), (120, 80), (80, 80)]

    @staticmethod
    def hand(index, x, y):
        points = [(0.0, 0.0)] * 21
        points[index] = (x, y)
        return points

    def test_each_fingertip_inside_or_on_edge_counts(self):
        for index in (4, 8, 12, 16, 20):
            with self.subTest(index=index):
                self.assertTrue(hand_near_polygon([self.hand(index, .5, .5)], self.rectangle, self.shape))
                self.assertTrue(hand_near_polygon([self.hand(index, .4, .5)], self.rectangle, self.shape, 0))

    def test_margin_scales_with_parking_area(self):
        # The target's largest dimension is 60 px, giving a 4.8 px margin.
        near = [self.hand(8, 77 / 200, .5)]
        far = [self.hand(8, 74 / 200, .5)]
        self.assertTrue(hand_near_polygon(near, self.rectangle, self.shape))
        self.assertFalse(hand_near_polygon(near, self.rectangle, self.shape, 0))
        self.assertFalse(hand_near_polygon(far, self.rectangle, self.shape))

    def test_rotated_polygon_uses_actual_shape(self):
        diamond = [(100, 10), (140, 50), (100, 90), (60, 50)]
        self.assertTrue(hand_near_polygon([self.hand(8, .5, .5)], diamond, self.shape, 0))
        # Inside the axis-aligned bounding box, outside the rotated polygon.
        self.assertFalse(hand_near_polygon([self.hand(8, 65 / 200, .15)], diamond, self.shape, 0))

    def test_wrist_or_palm_overlap_alone_does_not_count(self):
        for index in (0, 5, 9, 13, 17):
            self.assertFalse(hand_near_polygon([self.hand(index, .5, .5)], self.rectangle, self.shape))

    def test_empty_malformed_or_out_of_frame_hands_do_not_count(self):
        for hands in (None, [], [None], [[(.5, .5)]], [self.hand(8, float("nan"), .5)], [self.hand(8, -1, .5)]):
            self.assertFalse(hand_near_polygon(hands, self.rectangle, self.shape))

    def test_valid_second_hand_is_checked(self):
        hands = [self.hand(8, .9, .9), self.hand(8, .5, .5)]
        self.assertTrue(hand_near_polygon(hands, self.rectangle, self.shape))

    def test_invalid_target_geometry_raises(self):
        for polygon in ([], [(1, 1), (2, 2)], [(1, 1), (2, 2), (3, 3)], [(0, 0), (10, 0), (0, float("nan"))]):
            with self.assertRaises(ValueError):
                hand_near_polygon([], polygon, self.shape)
        with self.assertRaises(ValueError):
            hand_near_polygon([], self.rectangle, (0, 100))
        with self.assertRaises(ValueError):
            hand_near_polygon([], self.rectangle, self.shape, -0.1)


if __name__ == "__main__":
    unittest.main()
