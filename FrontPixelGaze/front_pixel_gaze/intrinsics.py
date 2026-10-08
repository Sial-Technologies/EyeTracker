"""Load / scale OpenCV camera intrinsics from .npz (copied from MultiCamGaze)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Intrinsics:
    camera_matrix: NDArray[np.float64]
    dist_coeffs: NDArray[np.float64]
    image_size: tuple[int, int]  # (width, height)


def scale_intrinsics(
    camera_k: NDArray[np.float64],
    dist: NDArray[np.float64],
    calib_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    calib_w, calib_h = int(calib_size[0]), int(calib_size[1])
    frame_w, frame_h = int(frame_size[0]), int(frame_size[1])
    if (calib_w, calib_h) == (frame_w, frame_h):
        return camera_k.copy(), dist.copy()
    sx = frame_w / float(calib_w)
    sy = frame_h / float(calib_h)
    k = camera_k.copy()
    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy
    return k, dist.copy()


def load_intrinsics(path: Path) -> Intrinsics:
    if not path.is_file():
        raise FileNotFoundError(f"Intrinsics not found: {path}")
    data = np.load(path, allow_pickle=False)
    k = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    size = np.asarray(data["image_size"], dtype=np.int32).reshape(-1)
    if size.size < 2:
        raise ValueError(f"Bad image_size in {path}")
    return Intrinsics(k, dist, (int(size[0]), int(size[1])))


def for_frame(intrinsics: Intrinsics, frame_size: tuple[int, int]) -> Intrinsics:
    """Return K/dist scaled to the live frame resolution."""
    k, dist = scale_intrinsics(
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        intrinsics.image_size,
        frame_size,
    )
    return Intrinsics(k, dist, frame_size)


def undistort_bgr(
    frame: NDArray[np.uint8],
    intrinsics: Intrinsics,
) -> NDArray[np.uint8]:
    h, w = frame.shape[:2]
    live = for_frame(intrinsics, (w, h))
    return cv2.undistort(frame, live.camera_matrix, live.dist_coeffs)
