"""Heatmap right-click ground-truth vs predicted gaze compare."""

from __future__ import annotations

import numpy as np

from multcam_gaze.apps.calibrate_left_heatmap import _compare_gaze_click


def test_compare_gaze_click_pixel_and_mm() -> None:
    cmp_ = _compare_gaze_click(
        truth_uv=(100.0, 200.0),
        pred_uv=(130.0, 260.0),
        screen_w=1000,
        screen_h=500,
        width_mm=500.0,
        height_mm=250.0,
        truth_front=None,
        ray_origin_front=None,
        ray_dir_front=None,
    )
    assert cmp_.du_px == 30.0
    assert cmp_.dv_px == 60.0
    assert abs(cmp_.err_px - float(np.hypot(30, 60))) < 1e-9
    # 0.5 mm/px both axes → err_mm = hypot(15, 30)
    assert abs(cmp_.err_mm - float(np.hypot(15, 30))) < 1e-9
    assert cmp_.ang_deg is None


def test_debug_compares_payload_arrays() -> None:
    from multcam_gaze.apps.calibrate_left_heatmap import (
        GazeDebugCompare,
        _debug_compares_payload,
    )

    payload = _debug_compares_payload(
        [
            GazeDebugCompare(
                truth_uv=(10.0, 20.0),
                pred_uv=(12.0, 25.0),
                du_px=2.0,
                dv_px=5.0,
                err_px=5.385,
                err_mm=1.7,
                miss_ray_mm=1.6,
                ang_deg=0.2,
            ),
            GazeDebugCompare(
                truth_uv=(30.0, 40.0),
                pred_uv=None,
                du_px=None,
                dv_px=None,
                err_px=None,
                err_mm=None,
                miss_ray_mm=None,
                ang_deg=None,
            ),
        ]
    )
    assert payload["debug_truth_uv"].shape == (2, 2)
    assert payload["debug_pred_uv"][0, 0] == 12.0
    assert np.isnan(payload["debug_pred_uv"][1, 0])
    assert payload["debug_du_px"][0] == 2.0
    assert np.isnan(payload["debug_dv_px"][1])


def test_compare_gaze_click_ray_miss() -> None:
    origin = np.array([0.0, 0.0, 0.0])
    direction = np.array([0.0, 0.0, 1.0])
    # Point 10 mm off the +Z axis at z=100.
    truth = np.array([10.0, 0.0, 100.0])
    cmp_ = _compare_gaze_click(
        truth_uv=(0.0, 0.0),
        pred_uv=None,
        screen_w=1920,
        screen_h=1080,
        width_mm=600.0,
        height_mm=340.0,
        truth_front=truth,
        ray_origin_front=origin,
        ray_dir_front=direction,
    )
    assert cmp_.pred_uv is None
    assert cmp_.miss_ray_mm is not None
    assert abs(cmp_.miss_ray_mm - 10.0) < 1e-6
    assert cmp_.ang_deg is not None
    assert cmp_.ang_deg > 0.0
