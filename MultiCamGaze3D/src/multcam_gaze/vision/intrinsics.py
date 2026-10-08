"""Camera intrinsics load/save/scale."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from multcam_gaze.exceptions import MissingIntrinsicsError
from multcam_gaze.types import CameraRole, IntrinsicsModel


def scale_intrinsics(
    camera_k: np.ndarray,
    dist: np.ndarray,
    calib_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
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


def intrinsics_path(calib_dir: Path, role: CameraRole) -> Path:
    return calib_dir / "intrinsics" / f"{role}.npz"


def save_intrinsics(path: Path, model: IntrinsicsModel, **extra: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "camera_matrix": model.camera_matrix,
        "dist_coeffs": model.dist_coeffs,
        "image_size": np.array(model.image_size, dtype=np.int32),
        "rms_reprojection_error": np.array(
            [model.rms_reprojection_error if model.rms_reprojection_error is not None else -1.0]
        ),
    }
    for k, v in extra.items():
        payload[k] = v
    # Write via file handle so np.savez does not append another ".npz"
    # (path strings/Paths that lack a .npz suffix get one automatically).
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **payload)
    tmp.replace(path)


def load_intrinsics(path: Path) -> IntrinsicsModel | None:
    if not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    k = np.asarray(data["camera_matrix"], dtype=np.float64)
    d = np.asarray(data["dist_coeffs"], dtype=np.float64)
    sz = tuple(int(x) for x in np.asarray(data["image_size"]).ravel()[:2])
    rms_arr = data.get("rms_reprojection_error")
    rms = None
    if rms_arr is not None:
        rv = float(np.asarray(rms_arr).ravel()[0])
        if rv >= 0:
            rms = rv
    return IntrinsicsModel(k, d, sz, rms)


def require_intrinsics(calib_dir: Path, role: CameraRole) -> IntrinsicsModel:
    path = intrinsics_path(calib_dir, role)
    model = load_intrinsics(path)
    if model is None:
        raise MissingIntrinsicsError(
            f"Missing intrinsics for {role}. Run calibrate_intrinsics --role {role} "
            f"(expected {path})"
        )
    return model


def hfov_from_k(camera_k: np.ndarray, image_width: int) -> float | None:
    fx = float(camera_k[0, 0])
    if fx <= 1e-6 or image_width <= 0:
        return None
    return float(np.degrees(2.0 * np.arctan(0.5 * image_width / fx)))


def vfov_from_k(camera_k: np.ndarray, image_height: int) -> float | None:
    """Vertical FOV (degrees) from pinhole fy and image height."""
    fy = float(camera_k[1, 1])
    if fy <= 1e-6 or image_height <= 0:
        return None
    return float(np.degrees(2.0 * np.arctan(0.5 * image_height / fy)))
