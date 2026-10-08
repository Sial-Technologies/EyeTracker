"""Shared domain types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

CameraRole = Literal["front", "left_eye", "right_eye"]

PREVIEW_ROLES = ("left", "right", "front")
ROLE_TO_PREVIEW = {"front": "front", "left_eye": "left", "right_eye": "right"}
PREVIEW_TO_ROLE = {v: k for k, v in ROLE_TO_PREVIEW.items()}


@dataclass(frozen=True)
class ReprojectionStats:
    """Reprojection error statistics in pixels."""

    mean_px: float
    rms_px: float
    max_px: float
    num_points: int
    num_markers: int


@dataclass(frozen=True)
class IntrinsicsModel:
    """Pinhole camera intrinsics."""

    camera_matrix: NDArray[np.float64]
    dist_coeffs: NDArray[np.float64]
    image_size: tuple[int, int]
    rms_reprojection_error: float | None = None

    def scaled_for_frame(self, frame_w: int, frame_h: int) -> IntrinsicsModel:
        from multcam_gaze.vision.intrinsics import scale_intrinsics

        k, d = scale_intrinsics(
            self.camera_matrix,
            self.dist_coeffs,
            self.image_size,
            (frame_w, frame_h),
        )
        return IntrinsicsModel(k, d, (frame_w, frame_h), self.rms_reprojection_error)


@dataclass(frozen=True)
class GazeRayInCameraFrame:
    """Unit gaze direction in OpenCV convention (Y-down). Origin may be virtual units."""

    origin: NDArray[np.float64]
    direction: NDArray[np.float64]
    eye_id: str
    valid: bool = True
