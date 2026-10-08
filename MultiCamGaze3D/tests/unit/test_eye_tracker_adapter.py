"""Adapter coordinate convention tests."""

from __future__ import annotations

import numpy as np

from multcam_gaze.tracking import eye_tracker
from multcam_gaze.tracking.eye_tracker_adapter import (
    EyeTrackerAdapter,
    orlosky_to_opencv_direction,
)


def test_orlosky_to_opencv_flips_y() -> None:
    d = orlosky_to_opencv_direction(np.array([0.0, 0.6, 0.8]))
    np.testing.assert_allclose(d, [0.0, -0.6, 0.8], atol=1e-9)


def test_orlosky_to_opencv_skips_y_when_vertical_flip_undone() -> None:
    d = orlosky_to_opencv_direction(
        np.array([0.0, 0.6, 0.8]),
        vertical_flip_undone=True,
    )
    np.testing.assert_allclose(d, [0.0, 0.6, 0.8], atol=1e-9)


def test_orlosky_to_opencv_renormalizes() -> None:
    d = orlosky_to_opencv_direction(np.array([3.0, 4.0, 0.0]))
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-9
    assert d[1] < 0.0


def test_get_gaze_ray_respects_undo_flip_vertical() -> None:
    eye_tracker.reset_eye_tracking_state("left")
    adapter = EyeTrackerAdapter()
    eye_tracker.eye_tracking_states["left"]["last_gaze_dir"] = np.array(
        [0.0, 0.6, 0.8], dtype=np.float64
    )
    eye_tracker.eye_tracking_states["left"]["last_sphere_center"] = np.array(
        [0.0, 0.0, 0.0], dtype=np.float64
    )

    adapter._undo_flip_vertical = False
    ray = adapter.get_gaze_ray("left")
    np.testing.assert_allclose(ray.direction, [0.0, -0.6, 0.8], atol=1e-9)

    adapter._undo_flip_vertical = True
    ray = adapter.get_gaze_ray("left")
    np.testing.assert_allclose(ray.direction, [0.0, 0.6, 0.8], atol=1e-9)
    eye_tracker.reset_eye_tracking_state("left")


def test_process_frame_latches_undo_flip_vertical_from_pixel_map() -> None:
    from unittest.mock import patch

    from multcam_gaze.tracking.pixel_remap import TrackerPixelMap

    adapter = EyeTrackerAdapter()
    pixel_map = TrackerPixelMap(
        sensor_width=640,
        sensor_height=480,
        undo_flip_vertical=True,
    )
    with patch.object(eye_tracker, "process_frame"):
        adapter.process_frame(
            np.zeros((480, 640, 3), dtype=np.uint8),
            eye_id="left",
            pixel_map=pixel_map,
        )
    assert adapter._undo_flip_vertical is True

    pixel_map_off = TrackerPixelMap(
        sensor_width=640,
        sensor_height=480,
        undo_flip_vertical=False,
    )
    with patch.object(eye_tracker, "process_frame"):
        adapter.process_frame(
            np.zeros((480, 640, 3), dtype=np.uint8),
            eye_id="left",
            pixel_map=pixel_map_off,
        )
    assert adapter._undo_flip_vertical is False


def test_get_pupil_confidence() -> None:
    eye_tracker.reset_eye_tracking_state("left")
    adapter = EyeTrackerAdapter()
    assert adapter.get_pupil_confidence("left") == 0.0
    eye_tracker.eye_tracking_states["left"]["last_pupil_confidence"] = 0.91
    assert adapter.get_pupil_confidence("left") == 0.91
    eye_tracker.reset_eye_tracking_state("left")


def test_lock_sphere_center_at_pupil_and_unlock() -> None:
    eye_tracker.reset_eye_tracking_state("left")
    adapter = EyeTrackerAdapter()
    assert adapter.lock_sphere_center_at_pupil("left") is None

    eye_tracker.eye_tracking_states["left"]["last_pupil_center"] = (310, 250)
    eye_tracker.eye_tracking_states["left"]["last_pupil_confidence"] = 0.95
    locked = adapter.lock_sphere_center_at_pupil("left")
    assert locked == (310, 250)
    assert adapter.is_sphere_center_locked("left")
    assert adapter.get_locked_eye_center_ir_px("left") == (310, 250)

    assert adapter.unlock_sphere_center("left")
    assert not adapter.is_sphere_center_locked("left")
    assert adapter.get_locked_eye_center_ir_px("left") is None
    eye_tracker.reset_eye_tracking_state("left")


def test_apply_locked_eye_center_ir_px() -> None:
    eye_tracker.reset_eye_tracking_state("right")
    adapter = EyeTrackerAdapter()
    assert adapter.apply_locked_eye_center_ir_px("right", (100, 200)) == (100, 200)
    assert adapter.get_locked_eye_center_ir_px("right") == (100, 200)
    eye_tracker.reset_eye_tracking_state("right")


def test_readiness_with_locked_center_skips_model_center_count() -> None:
    eye_tracker.reset_eye_tracking_state("left")
    adapter = EyeTrackerAdapter()
    adapter.apply_locked_eye_center_ir_px("left", (191, 205))

    # Locked but radius not adapted yet → not ready.
    r0 = adapter.get_tracking_readiness("left")
    assert not r0.ready
    assert not r0.radius_adapted
    assert r0.n_model_centers >= r0.min_model_centers

    # Simulate radius adaptation after looking around.
    eye_tracker.eye_tracking_states["left"]["max_observed_distance"] = 120.0
    eye_tracker.eye_tracking_states["left"]["last_sphere_radius_ellipse"] = (
        (200, 200),
        (20, 15),
        0.0,
    )
    r1 = adapter.get_tracking_readiness("left")
    assert r1.ready
    assert r1.radius_adapted
    eye_tracker.reset_eye_tracking_state("left")
