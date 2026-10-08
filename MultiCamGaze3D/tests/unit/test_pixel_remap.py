"""Sensor remapping for zoom/pan and Orlosky crop."""

from __future__ import annotations

import numpy as np

from multcam_gaze.tracking.pixel_remap import (
    TrackerPixelMap,
    ZoomPan,
    invert_zoom_pan_point,
    zoom_pan_affine,
)
from multcam_gaze.vision.intrinsics import vfov_from_k


def test_invert_zoom_pan_roundtrip_with_pan() -> None:
    w, h = 640, 480
    zoom, pan_x, pan_y = 2.3, 80.0, -30.0
    # Sensor optical center must land at (cx+pan, cy+pan) in the buffer.
    cx, cy = w / 2.0, h / 2.0
    m = zoom_pan_affine(w, h, zoom, pan_x, pan_y)
    buf = m @ np.array([cx, cy, 1.0])
    np.testing.assert_allclose(buf[:2], [cx + pan_x, cy + pan_y], atol=1e-6)
    back = invert_zoom_pan_point(buf[0], buf[1], w, h, zoom, pan_x, pan_y)
    np.testing.assert_allclose(back, [cx, cy], atol=1e-6)


def test_tracker_pixel_map_undoes_zoom_pan_before_unproject() -> None:
    w, h = 640, 480
    zoom, pan_x, pan_y = 2.0, 40.0, 0.0
    # Orlosky == sensor size → crop is identity. Buffer pixel at optical axis.
    buf_x, buf_y = w / 2.0 + pan_x, h / 2.0 + pan_y
    m = TrackerPixelMap(
        sensor_width=w,
        sensor_height=h,
        zoom_pan=ZoomPan(zoom, pan_x, pan_y),
    )
    xs, ys = m.to_sensor(buf_x, buf_y)
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
