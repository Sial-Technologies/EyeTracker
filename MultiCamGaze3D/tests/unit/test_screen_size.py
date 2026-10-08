"""Metric screen size and camera↔screen distance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from multcam_gaze.core.transform import Transform
from multcam_gaze.runtime.screen_size import (
    camera_to_screen_distance_mm,
    resolve_screen_physical_size,
    save_screen_config,
    size_from_diagonal_inches,
)


def test_size_from_diagonal_27_inch_16x9() -> None:
    size = size_from_diagonal_inches(27.0, 1920.0, 1080.0)
    # Classic 27" 16:9 ≈ 597.8 × 336.2 mm
    assert abs(size.width_mm - 597.8) < 1.0
    assert abs(size.height_mm - 336.2) < 1.0
    assert abs(size.diagonal_inches - 27.0) < 1e-9
    assert abs(float(np.hypot(size.width_mm, size.height_mm)) / 25.4 - 27.0) < 1e-6


def test_resolve_requires_metric_reference() -> None:
    with pytest.raises(ValueError, match="Metric screen size required"):
        resolve_screen_physical_size(screen_w_px=1920, screen_h_px=1080, config={})


def test_resolve_from_config_diagonal() -> None:
    size = resolve_screen_physical_size(
        screen_w_px=2560,
        screen_h_px=1440,
        config={"diagonal_inches": 32.0},
    )
    assert abs(size.diagonal_inches - 32.0) < 1e-9
    assert size.width_mm > size.height_mm


def test_resolve_from_config_mm_does_not_nest_source(tmp_path: Path) -> None:
    nested = 'config (config (config (diagonal 27" + 1920x1080px aspect)))'
    size = resolve_screen_physical_size(
        screen_w_px=1920,
        screen_h_px=1080,
        config={
            "width_mm": 597.727,
            "height_mm": 336.221,
            "source": nested,
        },
    )
    assert size.source == 'config (diagonal 27" + 1920x1080px aspect)'
    assert size.source.count("config (") == 1

    path = tmp_path / "screen.json"
    save_screen_config(path, size)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["source"] == 'diagonal 27" + 1920x1080px aspect'

    size2 = resolve_screen_physical_size(
        screen_w_px=1920,
        screen_h_px=1080,
        config=saved,
    )
    assert size2.source == size.source
    assert size2.source.count("config (") == 1


def test_camera_to_screen_distance_perpendicular() -> None:
    # Screen facing camera, centre 600 mm along +Z.
    front_from_screen = Transform.from_rotation_translation(
        np.eye(3),
        np.array([0.0, 0.0, 600.0]),
        "front",
        "screen",
    )
    dist = camera_to_screen_distance_mm(front_from_screen)
    assert abs(dist - 600.0) < 1e-6
