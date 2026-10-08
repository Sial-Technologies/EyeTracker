"""Calibration workflows."""

from multcam_gaze.calibration.ray_extrinsic import (
    BurstStability,
    EyeOriginRefineResult,
    RayExtrinsicResult,
    RayExtrinsicSample,
    apply_screen_affine,
    assess_burst_stability,
    kabsch_residual_mm_for_eye_origin,
    p_front_z_stats,
    plane_distances_mm,
    refine_eye_origin_front,
    robust_average_samples,
    scale_gaze_direction,
    solve_front_from_eye,
)

__all__ = [
    "BurstStability",
    "EyeOriginRefineResult",
    "RayExtrinsicResult",
    "RayExtrinsicSample",
    "apply_screen_affine",
    "assess_burst_stability",
    "kabsch_residual_mm_for_eye_origin",
    "p_front_z_stats",
    "plane_distances_mm",
    "refine_eye_origin_front",
    "robust_average_samples",
    "scale_gaze_direction",
    "solve_front_from_eye",
]
