"""Resolve IR unprojection model (Phase-0 intrinsics + crop/flip remap)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from multcam_gaze.tracking.eye_tracker import IR_FOV_Y_DEG
from multcam_gaze.tracking.pixel_remap import TrackerPixelMap
from multcam_gaze.types import CameraRole, IntrinsicsModel
from multcam_gaze.vision.intrinsics import load_intrinsics, vfov_from_k, intrinsics_path


@dataclass(frozen=True)
class EyeUnprojectModel:
    """Parameters passed into ``eye_tracker.process_frame`` for correct angles."""

    fov_y_deg: float
    camera_matrix: NDArray[np.float64] | None
    dist_coeffs: NDArray[np.float64] | None
    pixel_map: TrackerPixelMap
    intrinsics_source: str  # "left_eye.npz" | "datasheet"


def resolve_eye_unproject_model(
    *,
    sensor_width: int,
    sensor_height: int,
    flip_vertical: bool,
    flip_horizontal: bool,
    calib_dir: Path | None = None,
    role: CameraRole = "left_eye",
    fallback_fov_y_deg: float = float(IR_FOV_Y_DEG),
) -> EyeUnprojectModel:
    """Build sensor-space unprojection for the raw tracking frame.

    Flips happen inside ``process_frame``; the map undoes those flips and the
    Orlosky 640×480 crop so ``K`` / ``dist`` apply in calibration sensor space.
    """
    pixel_map = TrackerPixelMap(
        sensor_width=int(sensor_width),
        sensor_height=int(sensor_height),
        undo_flip_vertical=bool(flip_vertical),
        undo_flip_horizontal=bool(flip_horizontal),
    )

    model: IntrinsicsModel | None = None
    source = "datasheet"
    if calib_dir is not None:
        path = intrinsics_path(calib_dir, role)
        model = load_intrinsics(path)
        if model is not None:
            source = path.name

    camera_matrix = None
    dist_coeffs = None
    fov_y = float(fallback_fov_y_deg)
    if model is not None:
        scaled = model.scaled_for_frame(int(sensor_width), int(sensor_height))
        camera_matrix = np.asarray(scaled.camera_matrix, dtype=np.float64)
        dist_coeffs = np.asarray(scaled.dist_coeffs, dtype=np.float64)
        vfov = vfov_from_k(camera_matrix, int(sensor_height))
        if vfov is not None:
            fov_y = float(vfov)

    return EyeUnprojectModel(
        fov_y_deg=fov_y,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        pixel_map=pixel_map,
        intrinsics_source=source,
    )
