"""Chessboard intrinsics calibration CLI."""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from multcam_gaze.apps._cli_common import add_standard_paths, resolve_paths
from multcam_gaze.core.logging_config import configure_logging
from multcam_gaze.hardware.camera_reader import CameraReader
from multcam_gaze.hardware.preview_view import load_camera_setup, role_camera_index
from multcam_gaze.vision.chessboard import (
    build_object_points,
    load_chessboard_config,
    run_chessboard_calibration,
)
from multcam_gaze.vision.intrinsics import intrinsics_path, load_intrinsics

WINDOW = "Intrinsics calibration"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics (chessboard).")
    parser.add_argument(
        "--role",
        required=True,
        choices=["front", "left_eye", "right_eye"],
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-index", type=int, default=None)
    add_standard_paths(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    config_dir, calib_dir = resolve_paths(args)
    if args.dry_run:
        print(f"Would calibrate {args.role} -> {intrinsics_path(calib_dir, args.role)}")
        return 0

    cb_cfg = load_chessboard_config(config_dir / "chessboard.json")
    cols = int(cb_cfg["inner_corners_cols"])
    rows = int(cb_cfg["inner_corners_rows"])
    square_mm = float(cb_cfg["square_size_mm"])
    min_poses = int(cb_cfg.get("min_poses", 10))
    board_size = (cols, rows)
    objp = build_object_points(cols, rows, square_mm)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    setup = load_camera_setup(config_dir / "camera_setup.json")
    preview_role = {"front": "front", "left_eye": "left", "right_eye": "right"}[args.role]
    idx, device_id = role_camera_index(setup, preview_role)
    cam_index = args.camera_index if args.camera_index is not None else idx
    if cam_index is None:
        print("No camera index in camera_setup.json; pass --camera-index", file=sys.stderr)
        return 1

    out_path = intrinsics_path(calib_dir, args.role)
    existing = load_intrinsics(out_path)
    result_model = existing

    reader = CameraReader(cam_index, args.width, args.height, device_id=device_id)
    reader.start()
    obj_points: list = []
    img_points: list = []
    last_capture = 0.0

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("SPACE capture | C calibrate | R reset | Q quit")

    try:
        while True:
            ok, frame = reader.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                board_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            corners_refined = None
            if found:
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, board_size, corners_refined, found)
            cv2.putText(
                display,
                f"Poses {len(obj_points)}/{min_poses}+ role={args.role}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                obj_points.clear()
                img_points.clear()
            if key == ord(" ") and found and corners_refined is not None:
                now = time.perf_counter()
                if now - last_capture < 0.4:
                    continue
                last_capture = now
                obj_points.append(objp.copy())
                img_points.append(corners_refined.reshape(-1, 2).astype(np.float32))
                print(f"Captured pose {len(obj_points)}")
            if key == ord("c"):
                h, w = frame.shape[:2]
                model = run_chessboard_calibration(
                    obj_points,
                    img_points,
                    (w, h),
                    out_path,
                    min_poses,
                )
                if model is None:
                    print(f"Need at least {min_poses} poses")
                else:
                    result_model = model
                    print(f"Saved {out_path} RMS={model.rms_reprojection_error:.4f}px")
    finally:
        reader.stop()
        cv2.destroyAllWindows()

    return 0 if result_model is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
