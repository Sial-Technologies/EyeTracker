"""Chessboard intrinsics calibration helpers."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from multcam_gaze.types import IntrinsicsModel
from multcam_gaze.vision.intrinsics import hfov_from_k, save_intrinsics


def load_chessboard_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    obj = np.zeros((cols * rows, 3), dtype=np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= float(square_mm)
    return obj


def calibrate_from_poses(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> IntrinsicsModel:
    rms, camera_k, dist, *_ = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
        flags=0,
    )
    return IntrinsicsModel(
        np.asarray(camera_k, dtype=np.float64),
        np.asarray(dist, dtype=np.float64).ravel(),
        image_size,
        float(rms),
    )


def run_chessboard_calibration(
    obj_points: list[np.ndarray],
    img_points: list[np.ndarray],
    image_size: tuple[int, int],
    out_path: Path,
    min_poses: int,
) -> IntrinsicsModel | None:
    if len(obj_points) < min_poses:
        return None
    model = calibrate_from_poses(obj_points, img_points, image_size)
    save_intrinsics(
        out_path,
        model,
        hfov_deg=hfov_from_k(model.camera_matrix, image_size[0]),
    )
    return model
