"""Look-at calibration grid layout."""

from __future__ import annotations

from multcam_gaze.apps.calibrate_left_heatmap import nine_or_n_grid


def test_nine_grid_bottom_clear_is_max_not_sum() -> None:
    """PiP clearance replaces margin when larger; do not stack both."""
    w, h = 1920, 1080
    bottom_clear = 272.0  # PREVIEW_H + 2*UI_PAD
    pts = nine_or_n_grid(9, w, h, margin_frac=0.12, bottom_clear_px=bottom_clear)
    vs = sorted({round(v, 1) for _, v in pts})
    assert len(vs) == 3
    assert vs[0] == round(0.12 * h, 1)
    # max(0.12*h, 272) = 272 → bottom row at h - 272 = 808 (not ~678 from sum)
    assert vs[-1] == round(h - bottom_clear, 1)
    assert vs[-1] > 750.0


def test_nine_grid_margin_only_when_clear_small() -> None:
    pts = nine_or_n_grid(9, 1920, 1080, margin_frac=0.12, bottom_clear_px=50.0)
    vs = sorted({round(v, 1) for _, v in pts})
    assert vs[-1] == round(1080 - 0.12 * 1080, 1)


def test_nine_grid_side_clear_keeps_targets_off_markers() -> None:
    """ArUco clearance must win over margin_frac when larger."""
    w, h = 1920, 1080
    side_clear = 280.0  # > 0.12*w so max picks clearance
    pts = nine_or_n_grid(
        9, w, h, margin_frac=0.12, side_clear_px=side_clear, bottom_clear_px=side_clear
    )
    us = sorted({round(u, 1) for u, _ in pts})
    vs = sorted({round(v, 1) for _, v in pts})
    assert us[0] == round(side_clear, 1)
    assert us[-1] == round(w - side_clear, 1)
    assert vs[0] == round(side_clear, 1)
    assert vs[-1] == round(h - side_clear, 1)
