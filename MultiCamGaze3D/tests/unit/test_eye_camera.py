"""IR unproject model resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from multcam_gaze.tracking.eye_camera import resolve_eye_unproject_model
from multcam_gaze.tracking.eye_tracker import compute_gaze_vector
from multcam_gaze.tracking.pixel_remap import TrackerPixelMap


def test_resolve_uses_left_eye_npz_when_present() -> None:
    calib = Path(__file__).resolve().parents[2] / "calib"
    if not (calib / "intrinsics" / "left_eye.npz").is_file():
        return  # skip in CI without local calib dumps
    model = resolve_eye_unproject_model(
        sensor_width=640,
        sensor_height=480,
        flip_vertical=True,
        flip_horizontal=False,
        calib_dir=calib,
        role="left_eye",
    )
    assert model.camera_matrix is not None
    assert model.intrinsics_source == "left_eye.npz"
    assert model.pixel_map.undo_flip_vertical is True
    assert model.pixel_map.undo_flip_horizontal is False


def test_compute_gaze_vector_optical_axis_stays_forward() -> None:
    """Pupil on optical axis should not invent yaw after crop remap."""
    w, h = 640, 480
    fy = h / (2.0 * np.tan(np.radians(40.0)))  # vfov 80°
    k = np.array([[fy, 0.0, w / 2.0], [0.0, fy, h / 2.0], [0.0, 0.0, 1.0]])
    pixel_map = TrackerPixelMap(sensor_width=w, sensor_height=h)
    _origin, direction = compute_gaze_vector(
        w / 2.0,
        h / 2.0,
        w / 2.0,
        h / 2.0,
        screen_width=w,
        screen_height=h,
        camera_matrix=k,
        dist_coeffs=np.zeros(5),
        pixel_map=pixel_map,
    )
    assert direction is not None
    assert abs(float(direction[0])) < 0.15
    assert abs(float(direction[1])) < 0.15
