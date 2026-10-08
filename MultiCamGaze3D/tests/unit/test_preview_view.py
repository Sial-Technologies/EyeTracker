"""Tests for eye-tracking frame prep (flip before zoom)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from multcam_gaze.hardware.preview_view import (
    compose_eye_preview_panel,
    merge_setup_display,
    prepare_eye_tracking_frame,
)


def test_prepare_eye_tracking_flip_before_zoom() -> None:
    # Distinct corner so flip + zoom order is observable.
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[0, 0] = (0, 0, 255)
    view = {"zoom": 1.0, "pan_x": 0, "pan_y": 0}

    out, fv, fh = prepare_eye_tracking_frame(
        frame, flip_vertical=True, flip_horizontal=False, view=view, zoom_affects_tracking=True
    )
    assert fv is False and fh is False
    # Vertical flip moves top-left pixel to bottom-left.
    np.testing.assert_array_equal(out[39, 0], (0, 0, 255))
    np.testing.assert_array_equal(out[0, 0], (0, 0, 0))


def test_eye_center_front_mm_from_setup() -> None:
    from multcam_gaze.hardware.preview_view import eye_center_front_mm_from_setup

    setup = {"left": {"eye_center_front_mm": [-25.0, 15.0, -30.0]}}
    e = eye_center_front_mm_from_setup(setup, "left")
    np.testing.assert_allclose(e, [-25.0, 15.0, -30.0])
    np.testing.assert_allclose(eye_center_front_mm_from_setup({}, "left"), [0.0, 0.0, 0.0])


def test_eye_center_ir_px_from_setup_roundtrip() -> None:
    from multcam_gaze.hardware.preview_view import (
        eye_center_ir_px_from_setup,
        set_eye_center_ir_px,
    )

    setup: dict = {"left": {"index": 1}}
    assert eye_center_ir_px_from_setup(setup, "left") is None
    set_eye_center_ir_px(setup, "left", (321, 240))
    assert eye_center_ir_px_from_setup(setup, "left") == (321, 240)
    set_eye_center_ir_px(setup, "left", None)
    assert eye_center_ir_px_from_setup(setup, "left") is None
    assert "eye_center_ir_px" not in setup["left"]


def test_effective_tracking_fov_shrinks_with_zoom() -> None:
    from multcam_gaze.hardware.preview_view import effective_tracking_fov_y_deg

    assert effective_tracking_fov_y_deg(False, {"zoom": 2.3}, base_fov_y_deg=80.0) == 80.0
    fov = effective_tracking_fov_y_deg(True, {"zoom": 2.0}, base_fov_y_deg=80.0)
    assert abs(fov - 40.0) < 1e-9
    fov23 = effective_tracking_fov_y_deg(True, {"zoom": 2.3}, base_fov_y_deg=80.0)
    assert abs(fov23 - (80.0 / 2.3)) < 1e-9


def test_prepare_eye_tracking_defers_flip_when_zoom_display_only() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    out, fv, fh = prepare_eye_tracking_frame(
        frame,
        flip_vertical=True,
        flip_horizontal=True,
        view={"zoom": 2.0, "pan_x": 1, "pan_y": -1},
        zoom_affects_tracking=False,
    )
    assert out is frame
    assert fv is True and fh is True


def test_compose_eye_preview_uses_configured_zoom_when_no_overlay() -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[0, 0] = (0, 0, 255)
    view = {"zoom": 1.0, "pan_x": 0, "pan_y": 0}
    panel = compose_eye_preview_panel(
        frame, None, flip_vertical=True, flip_horizontal=False, view=view, zoom_affects_tracking=True
    )
    # Vertical flip of raw with no overlay → bottom-left pixel.
    np.testing.assert_array_equal(panel[39, 0], (0, 0, 255))


def test_compose_eye_preview_prefers_overlay_when_zoom_tracks() -> None:
    raw = np.zeros((20, 20, 3), dtype=np.uint8)
    overlay = np.full((20, 20, 3), 7, dtype=np.uint8)
    panel = compose_eye_preview_panel(
        raw,
        overlay,
        flip_vertical=False,
        flip_horizontal=False,
        view={"zoom": 2.0, "pan_x": 0, "pan_y": 0},
        zoom_affects_tracking=True,
    )
    np.testing.assert_array_equal(panel, overlay)


def test_merge_setup_display_preserves_device_and_updates_view(
    tmp_path: Path,
) -> None:
    setup = {
        "left": {
            "device_id": "USB\\LEFT",
            "index": 1,
            "flip": True,
            "mirror": False,
            "view": {"zoom": 1.4, "pan_x": -50, "pan_y": 50},
            "zoom_affects_tracking": True,
            "eye_center_ir_px": [300, 220],
        },
        "front": {
            "device_id": "USB\\FRONT",
            "index": 3,
            "flip": False,
            "mirror": False,
            "view": {"zoom": 1.0, "pan_x": 0, "pan_y": 0},
        },
    }
    views = {
        "left": {"zoom": 2.0, "pan_x": 10, "pan_y": -20},
        "right": {"zoom": 1.0, "pan_x": 0, "pan_y": 0},
        "front": {"zoom": 1.0, "pan_x": 0, "pan_y": 0},
    }
    flips = {
        "left": {"flip": False, "mirror": True},
        "right": {"flip": False, "mirror": False},
        "front": {"flip": False, "mirror": False},
    }
    merged = merge_setup_display(setup, views, flips)
    assert "right" not in merged  # was never in setup
    assert merged["left"]["device_id"] == "USB\\LEFT"
    assert merged["left"]["index"] == 1
    assert merged["left"]["zoom_affects_tracking"] is True
    assert merged["left"]["flip"] is False
    assert merged["left"]["mirror"] is True
    assert merged["left"]["view"]["zoom"] == 2.0
    assert merged["left"]["eye_center_ir_px"] == [300, 220]
    # Round-trip JSON shape stays valid.
    path = tmp_path / "camera_setup.json"
    path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["left"]["view"]["pan_x"] == 10
