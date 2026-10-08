import numpy as np

from front_pixel_gaze.uv_map import (
    direction_to_yaw_pitch,
    fit_front_uv_map,
    orlosky_to_opencv_direction,
)


def test_orlosky_y_flip():
    d = orlosky_to_opencv_direction(np.array([0.0, 1.0, 0.0]))
    assert d[1] < 0
    d2 = orlosky_to_opencv_direction(
        np.array([0.0, 1.0, 0.0]), vertical_flip_undone=True
    )
    assert d2[1] > 0


def test_affine_roundtrip():
    # Synthetic: u = 320 + 200*yaw, v = 240 - 180*pitch
    dirs = []
    uvs = []
    for yaw, pitch in [(-0.2, 0.1), (0.0, 0.0), (0.25, -0.15), (0.1, 0.2), (-0.15, -0.05)]:
        # Reconstruct a unit direction from yaw/pitch (OpenCV Y-down)
        x = np.sin(yaw) * np.cos(pitch)
        z = np.cos(yaw) * np.cos(pitch)
        y = -np.sin(pitch)
        d = np.array([x, y, z], dtype=np.float64)
        d /= np.linalg.norm(d)
        dirs.append(d)
        uvs.append((320 + 200 * yaw, 240 - 180 * pitch))

    model = fit_front_uv_map(dirs, uvs)
    assert model.rms_px < 1.0
    for d, (u, v) in zip(dirs, uvs):
        pu, pv = model.apply(d)
        assert abs(pu - u) < 2.0
        assert abs(pv - v) < 2.0


def test_yaw_pitch_forward():
    yaw, pitch = direction_to_yaw_pitch(np.array([0.0, 0.0, 1.0]))
    assert abs(yaw) < 1e-9
    assert abs(pitch) < 1e-9
