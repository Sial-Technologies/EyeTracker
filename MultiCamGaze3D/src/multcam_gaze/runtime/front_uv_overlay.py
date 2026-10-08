"""FrontPixelGaze yaw/pitch→front-UV map overlay (eval A/B vs 3D ray∩plane).

Loads ``FrontPixelGaze/calib/front_uv_map.json`` and applies it with the same
pitch-up convention that package used when fitting. Does not import FrontPixelGaze.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from multcam_gaze.paths import project_root


def default_front_uv_map_path() -> Path:
    """Sibling repo path: ``../FrontPixelGaze/calib/front_uv_map.json``."""
    return project_root().parent / "FrontPixelGaze" / "calib" / "front_uv_map.json"


def direction_to_yaw_pitch_frontpixel(
    direction: NDArray[np.float64],
) -> tuple[float, float]:
    """OpenCV Y-down unit dir → yaw (right+), pitch (up+) — FrontPixelGaze convention."""
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return 0.0, 0.0
    d = d / n
    yaw = float(np.arctan2(d[0], d[2]))
    pitch = float(np.arctan2(-d[1], float(np.hypot(d[0], d[2]))))
    return yaw, pitch


@dataclass(frozen=True)
class FrontUvMap:
    """``[u, v]^T = A @ [yaw, pitch, 1]^T`` — front-camera pixels (FrontPixelGaze)."""

    A: NDArray[np.float64]
    n_samples: int
    rms_px: float
    path: Path | None = None

    def apply(
        self,
        direction_opencv: NDArray[np.float64],
        *,
        image_size: tuple[int, int] | None = None,
    ) -> tuple[float, float]:
        yaw, pitch = direction_to_yaw_pitch_frontpixel(direction_opencv)
        uv = self.A @ np.array([yaw, pitch, 1.0], dtype=np.float64)
        u, v = float(uv[0]), float(uv[1])
        if image_size is not None:
            w, h = image_size
            u = float(np.clip(u, 0.0, w - 1.0))
            v = float(np.clip(v, 0.0, h - 1.0))
        return u, v


def load_front_uv_map(path: Path) -> FrontUvMap | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return FrontUvMap(
        A=np.asarray(data["A"], dtype=np.float64).reshape(2, 3),
        n_samples=int(data.get("n_samples", 0)),
        rms_px=float(data.get("rms_px", 0.0)),
        path=path,
    )


def undistort_image_points(
    pts: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
    dist_coeffs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Raw image points → same space as ``cv2.undistort`` / FrontPixel click UV."""
    reshaped = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(
        reshaped,
        camera_matrix,
        dist_coeffs,
        P=camera_matrix,
    )
    return out.reshape(-1, 2)


def front_uv_to_screen(
    front_uv: tuple[float, float],
    homography: NDArray[np.float64],
    front_corners: NDArray[np.float64],
) -> tuple[float, float] | None:
    """Map front-camera UV → screen UV if inside the detected screen quad."""
    u, v = float(front_uv[0]), float(front_uv[1])
    poly = np.asarray(front_corners, dtype=np.float32).reshape(-1, 1, 2)
    if cv2.pointPolygonTest(poly, (u, v), False) < 0:
        return None
    pt = np.array([[[u, v]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, np.asarray(homography, dtype=np.float64))
    return float(out[0, 0, 0]), float(out[0, 0, 1])
