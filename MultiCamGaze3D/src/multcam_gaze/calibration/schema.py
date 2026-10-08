"""Device calibration JSON schema (v1)."""

from __future__ import annotations

CALIBRATION_SCHEMA_VERSION = 1


def empty_device_calibration() -> dict:
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "coordinate_system": "front_screen",
        "camera_intrinsics": {
            "front": {},
            "left_eye": {},
            "right_eye": {},
        },
        "extrinsics": {
            "world_from_front": None,
            "world_from_left_eye": None,
            "world_from_right_eye": None,
            "front_from_left_eye": None,
            "front_from_right_eye": None,
            "left_eye_from_right_eye": None,
        },
        "screen": {
            "width_mm": None,
            "height_mm": None,
            "diagonal_inches": None,
            "size_source": None,
            "world_from_screen": None,
        },
        "gaze_calibration": {
            "left": {},
            "right": {},
            "binocular": {},
        },
        "quality": {
            "front_reprojection_error_px": None,
            "left_reprojection_error_px": None,
            "right_reprojection_error_px": None,
            "gaze_error_mm": None,
            "extrinsic_consistency_translation_mm": None,
            "extrinsic_consistency_rotation_deg": None,
        },
    }
