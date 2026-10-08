"""Typed adapter over legacy eye_tracker module."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from multcam_gaze.tracking import eye_tracker
from multcam_gaze.tracking.eye_camera import EyeUnprojectModel
from multcam_gaze.tracking.pixel_remap import TrackerPixelMap
from multcam_gaze.types import GazeRayInCameraFrame


@dataclass(frozen=True)
class EyeTrackingReadiness:
    """Whether adaptive eyeball size / center have had enough frames."""

    n_model_centers: int
    radius_px: float
    radius_adapted: bool
    min_model_centers: int
    ready: bool


def orlosky_to_opencv_direction(
    direction: NDArray[np.float64],
    *,
    vertical_flip_undone: bool = False,
) -> NDArray[np.float64]:
    """Orlosky/OpenGL gaze is Y-up; front camera / solvePnP is OpenCV Y-down.

    GazeScreen3D applies the same flip in ``gaze_dir_in_cam`` before ray∩plane.
    When ``vertical_flip_undone`` is true (config ``flip: true`` + sensor remap
    undoes that flip for ``K``), the remapped unproject already inverts vertical
    sense vs the upright eye frame — skip the extra Y negate or pitch is crushed.
    """
    d = np.asarray(direction, dtype=np.float64).reshape(3).copy()
    if not vertical_flip_undone:
        d[1] = -d[1]
    n = float(np.linalg.norm(d))
    if n > 1e-12:
        d = d / n
    return d


class EyeTrackerAdapter:
    """Process IR frames and expose gaze ray in OpenCV camera convention (Y-down)."""

    def __init__(self) -> None:
        self._undo_flip_vertical = False

    def process_frame(
        self,
        frame,
        eye_id: str = "left",
        flip_vertical: bool = False,
        flip_horizontal: bool = False,
        *,
        fov_y_deg: float | None = None,
        camera_matrix: NDArray[np.float64] | None = None,
        dist_coeffs: NDArray[np.float64] | None = None,
        pixel_map: TrackerPixelMap | None = None,
        unproject: EyeUnprojectModel | None = None,
    ) -> None:
        if unproject is not None:
            fov_y_deg = unproject.fov_y_deg
            camera_matrix = unproject.camera_matrix
            dist_coeffs = unproject.dist_coeffs
            pixel_map = unproject.pixel_map
        self._undo_flip_vertical = bool(
            pixel_map is not None and pixel_map.undo_flip_vertical
        )
        eye_tracker.process_frame(
            frame,
            eye_id=eye_id,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
            fov_y_deg=fov_y_deg,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            pixel_map=pixel_map,
        )

    def get_preview_frame(self, eye_id: str) -> NDArray[np.uint8] | None:
        """Annotated IR frame (pupil / sphere overlay), or None if not ready."""
        return eye_tracker.get_preview_frame(eye_id)

    def apply_locked_eye_center_ir_px(
        self, eye_id: str, xy: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Freeze Orlosky 2D eyeball center at tracking-buffer pixels."""
        return eye_tracker.apply_locked_eye_center_ir_px(eye_id, xy)

    def lock_sphere_center_at_pupil(
        self, eye_id: str, *, min_confidence: float | None = None
    ) -> tuple[int, int] | None:
        """Lock eyeball center to the latest confident pupil (look-into-IR)."""
        return eye_tracker.lock_sphere_center_at_pupil(
            eye_id, min_confidence=min_confidence
        )

    def freeze_sphere_center(self, eye_id: str) -> tuple[int, int] | None:
        """Freeze current adapted 2D center after warmup (no pupil snap)."""
        return eye_tracker.freeze_sphere_center(eye_id)

    def unlock_sphere_center(self, eye_id: str) -> bool:
        """Resume auto eyeball-center estimation."""
        return bool(eye_tracker.unlock_sphere_center(eye_id))

    def is_sphere_center_locked(self, eye_id: str) -> bool:
        return bool(eye_tracker.is_sphere_center_locked(eye_id))

    def get_locked_eye_center_ir_px(self, eye_id: str) -> tuple[int, int] | None:
        return eye_tracker.get_locked_eye_center_ir_px(eye_id)

    def get_tracking_readiness(self, eye_id: str) -> EyeTrackingReadiness:
        """Sphere radius adapts after enough center samples, or immediately if locked.

        A persisted ``eye_center_ir_px`` lock freezes auto center sampling, so the
        model-center count never reaches ``MIN_MODEL_CENTERS``. Treat a locked
        center as satisfied and only wait for radius adaptation (look around).
        """
        state = eye_tracker.eye_tracking_states.get(eye_id, {})
        centers = state.get("model_centers") or []
        n = len(centers)
        radius = float(state.get("max_observed_distance") or 0.0)
        adapted = radius > 0.0 and state.get("last_sphere_radius_ellipse") is not None
        min_n = int(getattr(eye_tracker, "MIN_MODEL_CENTERS", 30))
        locked = bool(state.get("sphere_center_locked_2d"))
        center_ok = locked or n >= min_n
        return EyeTrackingReadiness(
            n_model_centers=n if not locked else max(n, min_n),
            radius_px=radius if adapted else float(getattr(eye_tracker, "EYE_SPHERE_RADIUS_PX", 202)),
            radius_adapted=adapted,
            min_model_centers=min_n,
            ready=adapted and center_ok,
        )

    def get_pupil_confidence(self, eye_id: str) -> float:
        """Latest pupil ellipse fill ratio (0..1), or 0 if unknown."""
        state = eye_tracker.eye_tracking_states.get(eye_id, {})
        try:
            return float(state.get("last_pupil_confidence") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_gaze_ray(self, eye_id: str) -> GazeRayInCameraFrame:
        eye_tracker.load_eye_tracking_state(eye_id)
        direction = eye_tracker.get_eye_gaze_dir(eye_id)
        state = eye_tracker.eye_tracking_states.get(eye_id, {})
        origin = state.get("last_sphere_center")
        if direction is None or origin is None:
            return GazeRayInCameraFrame(
                np.zeros(3),
                np.array([0.0, 0.0, 1.0]),
                eye_id,
                valid=False,
            )
        # Origin stays in Orlosky virtual units (not used as metric mm by the solver).
        o = np.asarray(origin, dtype=np.float64).reshape(3)
        d = orlosky_to_opencv_direction(
            np.asarray(direction, dtype=np.float64),
            vertical_flip_undone=self._undo_flip_vertical,
        )
        return GazeRayInCameraFrame(o, d, eye_id, valid=True)
