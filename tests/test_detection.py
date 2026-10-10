import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from phonewatch.detection import PhoneDetector, rotate_view, enhance_dark
from phonewatch.perspective import plane_matrix, rectify, map_polygon, load_plane, save_plane


class ArrayBoxes:
    def __init__(self, boxes):
        self.boxes = np.float32(boxes).reshape(-1, 4)
        self.xyxy = self
    def cpu(self):
        return self
    def numpy(self):
        return self.boxes


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
    def predict(self, image, **kwargs):
        self.calls.append(image.shape)
        return [SimpleNamespace(boxes=ArrayBoxes(next(self.responses)))]


class DetectionTests(unittest.TestCase):
    def test_success_avoids_fallbacks(self):
        model = FakeModel([[[10, 20, 30, 40]]])
        polygons, method = PhoneDetector(model, [67]).detect(np.full((100, 150, 3), 180, np.uint8))
        self.assertEqual(method, "normal")
        self.assertEqual(len(model.calls), 1)
        np.testing.assert_array_equal(polygons[0], [[10,20],[30,20],[30,40],[10,40]])

    def test_rotated_detection_maps_to_original(self):
        image = np.full((100, 150, 3), 180, np.uint8)
        _, matrix = rotate_view(image, 45)
        # A tiny box around the transformed center must map back near original center.
        center = cv2.transform(np.float32([[[75,50]]]), matrix)[0,0]
        x,y = center
        model = FakeModel([[], [[x-2,y-2,x+2,y+2]]])
        polygons, method = PhoneDetector(model, [67]).detect(image)
        self.assertEqual(method, "rotation 45")
        np.testing.assert_allclose(polygons[0].mean(axis=0), [75,50], atol=1)

    def test_dark_fallback_and_fast_mode(self):
        image = np.zeros((100, 100, 3), np.uint8)
        model = FakeModel([[], [[10,10,40,40]]])
        self.assertEqual(PhoneDetector(model, [67]).detect(image)[1], "low-light")
        model = FakeModel([[]])
        self.assertEqual(PhoneDetector(model, [67], mode="fast").detect(image)[0], [])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(enhance_dark(image).shape, image.shape)

    def test_empty_scene_exhausts_bounded_fallbacks(self):
        model = FakeModel([[], [], [], [], []])
        self.assertEqual(PhoneDetector(model, [67]).detect(np.zeros((100,100,3),np.uint8))[0], [])
        self.assertEqual(len(model.calls), 5)

    def test_rotation_does_not_clip_original_corners(self):
        image = np.zeros((100,150,3),np.uint8)
        view, matrix = rotate_view(image,45)
        corners = cv2.transform(np.float32([[[0,0],[150,0],[150,100],[0,100]]]),matrix)[0]
        self.assertTrue(np.all(corners >= -0.01))
        self.assertTrue(np.all(corners[:,0] <= view.shape[1]))
        self.assertTrue(np.all(corners[:,1] <= view.shape[0]))


class PerspectiveTests(unittest.TestCase):
    def test_tilted_plane_round_trip(self):
        points = np.float32([[60,30],[150,50],[170,190],[20,180]])
        frame = np.zeros((240,320,3),np.uint8)
        cv2.fillConvexPoly(frame,points.astype(np.int32),(255,255,255))
        view, inverse = rectify(frame,points)
        matrix,size = plane_matrix(points)
        projected = cv2.perspectiveTransform(points[None],matrix)[0]
        np.testing.assert_allclose(projected,[[0,0],[size[0]-1,0],[size[0]-1,size[1]-1],[0,size[1]-1]],atol=.001)
        np.testing.assert_allclose(map_polygon(projected,inverse),points,atol=1)
        self.assertGreater(view[view.shape[0]//2,view.shape[1]//2].mean(),250)

    def test_invalid_plane_rejected(self):
        for points in ([[0,0],[0,0],[0,0],[0,0]], [[0,0],[100,100],[100,0],[0,100]]):
            with self.assertRaises(ValueError):
                plane_matrix(points)

    def test_calibration_saved_and_resolution_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'plane.json'
            points = [[20,20],[100,30],[120,160],[10,150]]
            save_plane(path,points,(240,320,3))
            np.testing.assert_allclose(load_plane(path,(240,320,3)),points)
            with self.assertRaises(ValueError):
                load_plane(path,(480,640,3))
