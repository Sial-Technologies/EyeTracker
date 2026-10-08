"""Sensor remapping for Orlosky crop and mount flips."""

from __future__ import annotations

import numpy as np

from multcam_gaze.tracking.pixel_remap import TrackerPixelMap
from multcam_gaze.vision.intrinsics import vfov_from_k


def test_tracker_pixel_map_identity_when_orlosky_matches_sensor() -> None:
    w, h = 640, 480
    m = TrackerPixelMap(sensor_width=w, sensor_height=h)
    xs, ys = m.to_sensor(w / 2.0, h / 2.0)
    np.testing.assert_allclose([xs, ys], [w / 2.0, h / 2.0], atol=1e-6)


def test_tracker_pixel_map_undoes_vertical_flip() -> None:
    m = TrackerPixelMap(
        sensor_width=100,
        sensor_height=50,
        orlosky_width=100,
        orlosky_height=50,
        undo_flip_vertical=True,
    )
    xs, ys = m.to_sensor(10.0, 5.0)
    np.testing.assert_allclose([xs, ys], [10.0, 44.0], atol=1e-9)


def test_vfov_from_k_matches_pinhole() -> None:
    # fy such that vfov = 80° at height 480: fy = h / (2 tan(vfov/2))
    h = 480
    vfov = 80.0
    fy = h / (2.0 * np.tan(np.radians(vfov / 2.0)))
    k = np.array([[fy, 0.0, 320.0], [0.0, fy, 240.0], [0.0, 0.0, 1.0]])
    got = vfov_from_k(k, h)
    assert got is not None
    assert abs(got - vfov) < 1e-6
