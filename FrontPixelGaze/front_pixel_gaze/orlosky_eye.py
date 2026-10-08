"""Thin wrapper around MultiCamGaze3D's eye_tracker (vendored as gaze_eye_tracker)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from front_pixel_gaze import gaze_eye_tracker as et


@dataclass(frozen=True)
class TrackingReadiness:
    n_model_centers: int
    radius_px: float
    radius_adapted: bool
    min_model_centers: int
    ready: bool


@dataclass(frozen=True)
class GazeSample:
    direction_opengl: NDArray[np.float64] | None
    pupil_confidence: float
    preview_bgr: NDArray[np.uint8] | None
    ready: bool
    # When True, process_frame undid vertical flip for K — skip extra OpenCV Y negate.
    vertical_flip_undone: bool = False


class OrloskyEyeTracker:
    """Same tracking path as MultiCamGaze3D left IR (natural auto eye center)."""

    def __init__(self, eye_id: str = "left") -> None:
        self.eye_id = eye_id
        self._last_flip_vertical = False
        et.reset_eye_tracking_state(eye_id)
        et.set_show_separate_tracking_windows(False)
        et.unlock_sphere_center(eye_id)

    def reset(self) -> None:
        et.reset_eye_tracking_state(self.eye_id)
        et.unlock_sphere_center(self.eye_id)

    def lock_eye_center_ir_px(self, xy: tuple[int, int]) -> tuple[int, int] | None:
        """Freeze 2D eyeball center in the Orlosky 640×480 tracking buffer."""
        return et.apply_locked_eye_center_ir_px(self.eye_id, xy)

    def lock_eye_center_at_pupil(self) -> tuple[int, int] | None:
        """Freeze eyeball center on the latest confident pupil."""
        return et.lock_sphere_center_at_pupil(self.eye_id)

    def unlock_eye_center(self) -> bool:
        return bool(et.unlock_sphere_center(self.eye_id))

    def is_eye_center_locked(self) -> bool:
        return bool(et.is_sphere_center_locked(self.eye_id))

    def process(
        self,
        frame_bgr: NDArray[np.uint8],
        *,
        flip_vertical: bool = False,
        flip_horizontal: bool = False,
        camera_matrix: NDArray[np.float64] | None = None,
        dist_coeffs: NDArray[np.float64] | None = None,
        fov_y_deg: float | None = None,
    ) -> GazeSample:
        # MultiCamGaze / GazeScreen3D contract: RAW frame; flips inside process_frame.
        self._last_flip_vertical = bool(flip_vertical)
        et.process_frame(
            frame_bgr,
            eye_id=self.eye_id,
            flip_vertical=flip_vertical,
            flip_horizontal=flip_horizontal,
            fov_y_deg=fov_y_deg,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        readiness = self.get_readiness()
        direction = et.get_eye_gaze_dir(self.eye_id)
        if direction is not None:
            direction = np.asarray(direction, dtype=np.float64).reshape(3)
        preview = et.get_preview_frame(self.eye_id)
        state = et.eye_tracking_states.get(self.eye_id, {})
        conf = float(state.get("last_pupil_confidence") or 0.0)
        return GazeSample(
            direction_opengl=direction,
            pupil_confidence=conf,
            preview_bgr=None if preview is None else preview.copy(),
            ready=readiness.ready,
            vertical_flip_undone=bool(flip_vertical),
        )

    def get_readiness(self) -> TrackingReadiness:
        state = et.eye_tracking_states.get(self.eye_id, {})
        centers = state.get("model_centers") or []
        n = len(centers)
        radius = float(state.get("max_observed_distance") or 0.0)
        adapted = radius > 0.0 and state.get("last_sphere_radius_ellipse") is not None
        min_n = int(getattr(et, "MIN_MODEL_CENTERS", 30))
        locked = self.is_eye_center_locked()
        # Locked IR center skips waiting for many auto center samples.
        center_ok = locked or n >= min_n
        return TrackingReadiness(
            n_model_centers=n if not locked else max(n, min_n),
            radius_px=radius if adapted else float(getattr(et, "EYE_SPHERE_RADIUS_PX", 202)),
            radius_adapted=adapted,
            min_model_centers=min_n,
            ready=adapted and center_ok,
        )
