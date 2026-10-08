"""Tests for ray-to-point extrinsic solve and screen pixel mapping."""

from __future__ import annotations

import numpy as np

from multcam_gaze.calibration.ray_extrinsic import (
    RayExtrinsicSample,
    assess_burst_stability,
    point_to_ray_distance,
    robust_average_samples,
    scale_gaze_direction,
    solve_front_from_eye,
)
from multcam_gaze.core.transform import Transform
from multcam_gaze.runtime.screen_pose import pixel_to_front_mm, pixel_to_screen_mm


def test_point_to_ray_distance_zero_on_ray() -> None:
    o = np.array([0.0, 0.0, 0.0])
    d = np.array([0.0, 0.0, 1.0])
    p = np.array([0.0, 0.0, 50.0])
    assert point_to_ray_distance(o, d, p) < 1e-9


def test_pixel_to_screen_mm_center() -> None:
    p = pixel_to_screen_mm(960, 540, 1920, 1080, 600.0, 400.0)
    np.testing.assert_allclose(p, [0.0, 0.0, 0.0], atol=1e-9)


def test_pixel_to_front_identity_screen() -> None:
    front_from_screen = Transform.from_rotation_translation(
        np.eye(3),
        np.array([0.0, 0.0, 500.0]),
        "front",
        "screen",
    )
    p = pixel_to_front_mm(960, 540, front_from_screen, 1920, 1080, 600.0, 400.0)
    np.testing.assert_allclose(p, [0.0, 0.0, 500.0], atol=1e-6)


def test_robust_average_trims_extreme_direction() -> None:
    base_d = np.array([0.0, 0.0, -1.0])
    target = np.array([10.0, 20.0, 300.0])
    origin = np.array([0.1, -0.2, 0.0])
    samples = [
        RayExtrinsicSample(origin + i * 0.01, base_d.copy(), target + i * 0.1)
        for i in range(8)
    ]
    # One bad direction far from the cluster.
    samples.append(
        RayExtrinsicSample(
            origin,
            np.array([1.0, 0.0, 0.0]),
            target,
        )
    )
    avg = robust_average_samples(samples, trim_frac=0.2)
    np.testing.assert_allclose(avg.direction, base_d, atol=0.05)
    np.testing.assert_allclose(avg.target_front, target, atol=1.0)


def test_assess_burst_stability_rejects_high_variance() -> None:
    origin = np.zeros(3)
    target = np.array([0.0, 0.0, 300.0])
    stable = [
        RayExtrinsicSample(origin, np.array([0.0, 0.0, -1.0]), target) for _ in range(50)
    ]
    ok = assess_burst_stability(
        stable, min_samples=45, max_rms_deg=2.5, max_angle_deg=6.0
    )
    assert ok.ok

    noisy = list(stable)
    for _ in range(10):
        noisy.append(
            RayExtrinsicSample(origin, np.array([0.5, 0.0, -1.0]), target)
        )
    # Normalize directions in assess via stack — sample directions need unit-ish.
    noisy[-1] = RayExtrinsicSample(
        origin,
        np.array([1.0, 0.0, 0.0]),
        target,
    )
    bad = assess_burst_stability(
        noisy, min_samples=45, max_rms_deg=2.5, max_angle_deg=6.0
    )
    assert not bad.ok
    assert "gaze" in bad.reason or "too few" in bad.reason


def test_assess_burst_stability_rejects_bad_aruco() -> None:
    origin = np.zeros(3)
    target = np.array([0.0, 0.0, 300.0])
    burst = [
        RayExtrinsicSample(
            origin,
            np.array([0.0, 0.0, -1.0]),
            target + np.array([0.0, 0.0, float(i)]),
            aruco_reproj_px=2.0,
            aruco_markers=3,
        )
        for i in range(50)
    ]
    bad_markers = assess_burst_stability(
        burst,
        min_samples=45,
        max_rms_deg=2.5,
        max_angle_deg=6.0,
        min_aruco_markers=4,
    )
    assert not bad_markers.ok
    assert "ArUco markers" in bad_markers.reason

    deep = [
        RayExtrinsicSample(
            origin,
            np.array([0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 300.0 + 5.0 * i]),
            aruco_reproj_px=1.0,
            aruco_markers=4,
        )
        for i in range(50)
    ]
    bad_z = assess_burst_stability(
        deep,
        min_samples=45,
        max_rms_deg=2.5,
        max_angle_deg=6.0,
        min_aruco_markers=4,
        max_pfront_z_range_mm=60.0,
    )
    assert not bad_z.ok
    assert "P_front Z" in bad_z.reason


def test_plane_distances_near_zero_for_flat_grid() -> None:
    from multcam_gaze.calibration.ray_extrinsic import plane_distances_mm

    samples = [
        RayExtrinsicSample(
            np.zeros(3),
            np.array([0.0, 0.0, -1.0]),
            np.array([float(x), float(y), 500.0]),
        )
        for x in (-100.0, 0.0, 100.0)
        for y in (-50.0, 50.0)
    ]
    d = plane_distances_mm(samples)
    assert float(np.max(np.abs(d))) < 1e-6


def test_solve_front_from_eye_recovers_known_transform() -> None:
    r_true = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    # Coplanar look-ats (Z=400 plane) so consensus-plane projection is a no-op.
    plane_z = 400.0
    grid = [
        (-120.0, -80.0),
        (0.0, -80.0),
        (120.0, -80.0),
        (-120.0, 0.0),
        (0.0, 0.0),
        (120.0, 0.0),
        (-120.0, 80.0),
        (0.0, 80.0),
        (120.0, 80.0),
        (-60.0, -40.0),
        (60.0, 40.0),
        (0.0, 40.0),
    ]
    samples: list[RayExtrinsicSample] = []
    for i, (x, y) in enumerate(grid):
        p_front = np.array([x, y, plane_z], dtype=np.float64)
        d_front = p_front / np.linalg.norm(p_front)
        d_eye = r_true.T @ d_front  # R @ d_eye = d_front
        d_eye = d_eye / np.linalg.norm(d_eye)
        u = 200.0 + 80.0 * (i % 4)
        v = 150.0 + 70.0 * (i // 4)
        samples.append(
            RayExtrinsicSample(np.zeros(3), d_eye, p_front, target_uv=(u, v))
        )

    result = solve_front_from_eye(samples)
    assert result.mean_residual_mm < 5.0
    assert result.n_samples_used >= 3
    eye = result.front_from_eye.translation
    np.testing.assert_allclose(eye, np.zeros(3), atol=1e-9)
    for s in samples:
        d = result.front_from_eye.apply_direction(
            scale_gaze_direction(
                s.direction,
                result.direction_scale,
                scale_yaw=result.scale_yaw,
                scale_pitch=result.scale_pitch,
            )
        )
        assert point_to_ray_distance(eye, d, s.target_front) < 10.0
    assert abs(result.scale_yaw - 1.0) < 0.15
    assert abs(result.scale_pitch - 1.0) < 0.15
    # Diagnostic 2D map still fits; UV grid here is arbitrary so only check it exists.
    assert result.yaw_pitch_to_uv_2x3 is not None
    assert result.yaw_pitch_uv_rmse_px is not None


def test_solve_front_from_eye_uses_cad_origin() -> None:
    r_true = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    eye_origin = np.array([-25.0, 15.0, -30.0], dtype=np.float64)
    plane_z = 400.0
    samples: list[RayExtrinsicSample] = []
    for x, y in (
        (-120.0, -80.0),
        (0.0, -80.0),
        (120.0, -80.0),
        (-120.0, 0.0),
        (0.0, 0.0),
        (120.0, 0.0),
        (-120.0, 80.0),
        (0.0, 80.0),
        (120.0, 80.0),
    ):
        p_front = np.array([x, y, plane_z], dtype=np.float64)
        d_front = (p_front - eye_origin) / np.linalg.norm(p_front - eye_origin)
        d_eye = r_true.T @ d_front
        d_eye = d_eye / np.linalg.norm(d_eye)
        samples.append(RayExtrinsicSample(np.zeros(3), d_eye, p_front, target_uv=(0.0, 0.0)))

    result = solve_front_from_eye(samples, ray_origin_front=eye_origin)
    np.testing.assert_allclose(result.front_from_eye.translation, eye_origin, atol=1e-9)
    assert result.mean_residual_mm < 5.0


def test_refine_eye_origin_front_recovers_cad_from_multi_depth() -> None:
    from multcam_gaze.calibration.ray_extrinsic import refine_eye_origin_front

    r_true = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
    eye_true = np.array([-40.0, 25.0, -37.0], dtype=np.float64)
    groups: list[list[RayExtrinsicSample]] = []
    for plane_z in (500.0, 800.0):
        group: list[RayExtrinsicSample] = []
        for x, y in (
            (-150.0, -90.0),
            (0.0, -90.0),
            (150.0, -90.0),
            (-150.0, 0.0),
            (0.0, 0.0),
            (150.0, 0.0),
            (-150.0, 90.0),
            (0.0, 90.0),
            (150.0, 90.0),
        ):
            p_front = np.array([x, y, plane_z], dtype=np.float64)
            d_front = (p_front - eye_true) / np.linalg.norm(p_front - eye_true)
            d_eye = r_true.T @ d_front
            d_eye = d_eye / np.linalg.norm(d_eye)
            group.append(
                RayExtrinsicSample(np.zeros(3), d_eye, p_front, target_uv=(0.0, 0.0))
            )
        groups.append(group)

    seed = eye_true + np.array([15.0, -10.0, 12.0], dtype=np.float64)
    refined = refine_eye_origin_front(
        groups,
        seed=seed,
        half_extent_mm=25.0,
        step_mm=5.0,
        fine_half_extent_mm=6.0,
        fine_step_mm=2.0,
        project_to_consensus_plane=True,
        coplanar_max_dist_mm=80.0,
        run_full_solve=False,
        seed_prior_per_mm=0.0,
    )
    err_seed = float(np.linalg.norm(seed - eye_true))
    err_best = float(np.linalg.norm(refined.eye_origin_front - eye_true))
    assert err_best < err_seed - 5.0
    assert refined.residual_mm < refined.seed_residual_mm
    np.testing.assert_allclose(refined.eye_origin_front, eye_true, atol=8.0)


def test_scale_gaze_direction_preserves_unit_length() -> None:
    from multcam_gaze.calibration.ray_extrinsic import scale_gaze_direction

    d = scale_gaze_direction(np.array([0.3, -0.2, -1.0]), 1.7)
    assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-9


def test_apply_screen_affine_identity() -> None:
    from multcam_gaze.calibration.ray_extrinsic import apply_screen_affine

    a = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -5.0]], dtype=np.float64)
    u, v = apply_screen_affine((100.0, 200.0), a)
    assert abs(u - 110.0) < 1e-9
    assert abs(v - 195.0) < 1e-9
