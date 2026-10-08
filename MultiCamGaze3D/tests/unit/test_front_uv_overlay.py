"""FrontPixelGaze UV map load + pitch-up apply."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from multcam_gaze.runtime.front_uv_overlay import (
    direction_to_yaw_pitch_frontpixel,
    front_uv_to_screen,
    load_front_uv_map,
)


def test_load_and_apply_front_uv_map(tmp_path: Path) -> None:
    path = tmp_path / "front_uv_map.json"
    path.write_text(
        json.dumps(
            {
                "A": [[1000.0, 0.0, 320.0], [0.0, 800.0, 240.0]],
                "n_samples": 4,
                "rms_px": 5.0,
            }
        ),
        encoding="utf-8",
    )
    model = load_front_uv_map(path)
    assert model is not None
    assert model.n_samples == 4
    d = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    yaw, pitch = direction_to_yaw_pitch_frontpixel(d)
    assert abs(yaw) < 1e-9
    assert abs(pitch) < 1e-9
    u, v = model.apply(d, image_size=(640, 480))
    assert abs(u - 320.0) < 1e-6
    assert abs(v - 240.0) < 1e-6


def test_front_uv_to_screen_identity() -> None:
    corners = np.array(
        [[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]], dtype=np.float64
    )
    dst = np.array(
        [[0.0, 0.0], [1920.0, 0.0], [1920.0, 1080.0], [0.0, 1080.0]], dtype=np.float64
    )
    import cv2

    h, _ = cv2.findHomography(corners, dst, method=0)
    assert h is not None
    uv = front_uv_to_screen((50.0, 25.0), h, corners)
    assert uv is not None
    assert abs(uv[0] - 960.0) < 1.0
    assert abs(uv[1] - 540.0) < 1.0
