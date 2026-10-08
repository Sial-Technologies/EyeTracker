"""Left-eye + front: look-at calib → live screen heatmap (head may move)."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from multcam_gaze.apps._cli_common import add_standard_paths, resolve_paths
from multcam_gaze.calibration.ray_extrinsic import (
    RayExtrinsicResult,
    RayExtrinsicSample,
    angular_error_deg,
    assess_burst_stability,
    plane_distances_mm,
    point_to_ray_distance,
    robust_average_samples,
    scale_gaze_direction,
    solve_front_from_eye,
)
from multcam_gaze.types import GazeRayInCameraFrame
from multcam_gaze.calibration.storage import CalibrationStore
from multcam_gaze.core.logging_config import configure_logging
from multcam_gaze.core.transform import Transform
from multcam_gaze.exceptions import CalibrationSchemaError
from multcam_gaze.hardware.camera_reader import CameraReader
from multcam_gaze.hardware.preview_view import (
    compose_eye_preview_panel,
    eye_center_front_mm_from_setup,
    eye_center_ir_px_from_setup,
    load_camera_setup,
    prepare_eye_tracking_frame,
    role_camera_index,
    save_camera_setup,
    set_eye_center_ir_px,
)
from multcam_gaze.runtime.front_uv_overlay import (
    FrontUvMap,
    default_front_uv_map_path,
    front_uv_to_screen,
    load_front_uv_map,
)
from multcam_gaze.runtime.gaze_pipeline import gaze_to_world_ray, intersect_gaze_with_screen
from multcam_gaze.runtime.screen_pose import (
    LiveScreenPose,
    estimate_screen_mm,
    primary_screen_size_px,
)
from multcam_gaze.runtime.screen_size import (
    ScreenPhysicalSize,
    camera_to_screen_distance_mm,
    load_screen_config,
    resolve_screen_physical_size,
    save_screen_config,
)
from multcam_gaze.tracking.eye_camera import resolve_eye_unproject_model
from multcam_gaze.tracking.eye_tracker import IR_FOV_Y_DEG, PUPIL_CONFIDENCE_THRESHOLD
from multcam_gaze.tracking.eye_tracker_adapter import EyeTrackerAdapter
from multcam_gaze.vision.intrinsics import require_intrinsics

WINDOW_NAME = "Left Heatmap MVP"
PREVIEW_W = 320
PREVIEW_H = 240
EYE_PIP_W = 280
EYE_PIP_H = 210
UI_PAD = 16
# Soft gate: warn loudly above this; still enter heatmap so residuals are visible.
MAX_ACCEPTABLE_RESIDUAL_MM = 50.0
MAX_DEBUG_COMPARES = 12


@dataclass(frozen=True)
class GazeDebugCompare:
    """One right-click ground-truth vs predicted gaze hit (heatmap debug)."""

    truth_uv: tuple[float, float]
    pred_uv: tuple[float, float] | None
    du_px: float | None
    dv_px: float | None
    err_px: float | None
    err_mm: float | None
    miss_ray_mm: float | None
    ang_deg: float | None


def _in_roi(x: int, y: int, roi: tuple[int, int, int, int]) -> bool:
    x0, y0, w, h = roi
    return x0 <= x < x0 + w and y0 <= y < y0 + h


def _compare_gaze_click(
    *,
    truth_uv: tuple[float, float],
    pred_uv: tuple[float, float] | None,
    screen_w: int,
    screen_h: int,
    width_mm: float,
    height_mm: float,
    truth_front: NDArray[np.float64] | None,
    ray_origin_front: NDArray[np.float64] | None,
    ray_dir_front: NDArray[np.float64] | None,
) -> GazeDebugCompare:
    """Pixel / mm / ray error between click (truth) and predicted gaze hit."""
    du = dv = err_px = err_mm = None
    if pred_uv is not None:
        du = float(pred_uv[0] - truth_uv[0])
        dv = float(pred_uv[1] - truth_uv[1])
        err_px = float(np.hypot(du, dv))
        sx = float(width_mm) / max(float(screen_w), 1.0)
        sy = float(height_mm) / max(float(screen_h), 1.0)
        err_mm = float(np.hypot(du * sx, dv * sy))

    miss_ray_mm = ang_deg = None
    if (
        truth_front is not None
        and ray_origin_front is not None
        and ray_dir_front is not None
    ):
        o = np.asarray(ray_origin_front, dtype=np.float64).reshape(3)
        d = np.asarray(ray_dir_front, dtype=np.float64).reshape(3)
        p = np.asarray(truth_front, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(d))
        if n > 1e-12:
            d = d / n
            miss_ray_mm = float(point_to_ray_distance(o, d, p))
            ang_deg = float(angular_error_deg(o, d, p))

    return GazeDebugCompare(
        truth_uv=truth_uv,
        pred_uv=pred_uv,
        du_px=du,
        dv_px=dv,
        err_px=err_px,
        err_mm=err_mm,
        miss_ray_mm=miss_ray_mm,
        ang_deg=ang_deg,
    )


def _draw_debug_compares(
    canvas: NDArray[np.uint8],
    compares: list[GazeDebugCompare],
) -> None:
    """Overlay click (magenta) vs predicted (cyan) samples; annotate latest."""
    if not compares:
        return
    for i, c in enumerate(compares):
        latest = i == len(compares) - 1
        tu, tv = int(round(c.truth_uv[0])), int(round(c.truth_uv[1]))
        thickness = 2 if latest else 1
        cv2.drawMarker(
            canvas,
            (tu, tv),
            (255, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=22 if latest else 14,
            thickness=thickness,
        )
        if c.pred_uv is None:
            continue
        pu, pv = int(round(c.pred_uv[0])), int(round(c.pred_uv[1]))
        cv2.circle(canvas, (pu, pv), 8 if latest else 5, (255, 255, 0), thickness)
        cv2.line(canvas, (tu, tv), (pu, pv), (200, 200, 255), thickness)

    c = compares[-1]
    if c.pred_uv is None:
        label = "debug: no gaze hit at click"
    else:
        label = (
            f"dbg Δ=({c.du_px:+.0f},{c.dv_px:+.0f})px  "
            f"|err|={c.err_px:.0f}px/{c.err_mm:.1f}mm"
        )
        if c.ang_deg is not None:
            label += f"  ang={c.ang_deg:.2f}°"
        if c.miss_ray_mm is not None:
            label += f"  ray⊥={c.miss_ray_mm:.1f}mm"
    cv2.putText(
        canvas,
        label,
        (UI_PAD, canvas.shape[0] - UI_PAD - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        3,
    )
    cv2.putText(
        canvas,
        label,
        (UI_PAD, canvas.shape[0] - UI_PAD - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 255),
        2,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Left-eye look-at calibration + live screen heatmap (front ArUco)."
    )
    parser.add_argument("--width", type=int, default=640, help="Camera capture width")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height")
    parser.add_argument("--grid", type=int, choices=(9, 16, 25), default=9)
    parser.add_argument(
        "--capture-seconds",
        type=float,
        default=3.0,
        help="Hold duration per look-at point after settle (seconds)",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0.7,
        help="Discard frames this long after SPACE while gaze settles",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=45,
        help="Minimum valid frames required inside the capture window",
    )
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=0.2,
        help="Fraction of most extreme gaze directions to drop before averaging",
    )
    parser.add_argument(
        "--max-rms-deg",
        type=float,
        default=1.5,
        help="Abort point if direction RMS angular deviation exceeds this (deg)",
    )
    parser.add_argument(
        "--max-angle-deg",
        type=float,
        default=4.0,
        help="Abort point if any direction exceeds this angle from the mean (deg)",
    )
    parser.add_argument(
        "--diagonal-inches",
        type=float,
        default=None,
        help='Panel diagonal in inches (metric scale for ArUco). Overrides config/screen.json',
    )
    parser.add_argument(
        "--width-mm",
        type=float,
        default=None,
        help="Screen width mm (with --height-mm); alternative to --diagonal-inches",
    )
    parser.add_argument(
        "--height-mm",
        type=float,
        default=None,
        help="Screen height mm (with --width-mm); alternative to --diagonal-inches",
    )
    parser.add_argument(
        "--dump-samples",
        type=Path,
        default=None,
        help=(
            "Write look-at samples + heatmap right-click debug compares to .npz "
            "(updated on solve, each debug click, and exit)"
        ),
    )
    parser.add_argument(
        "--min-aruco-markers",
        type=int,
        default=4,
        help="Abort point if any capture frame sees fewer than this many corner markers",
    )
    parser.add_argument(
        "--max-aruco-reproj-px",
        type=float,
        default=8.0,
        help="Abort point if mean ArUco reprojection error exceeds this (px)",
    )
    parser.add_argument(
        "--max-pfront-z-range-mm",
        type=float,
        default=60.0,
        help="Abort point if P_front Z swings more than this within the burst (mm)",
    )
    parser.add_argument(
        "--min-pupil-confidence",
        type=float,
        default=0.0,
        help=(
            "Look-at capture: ignore burst frames when pupil fill ratio is below "
            "this (default 0 = off); SPACE still starts. Tracker lock/rays still "
            f"use PUPIL_CONFIDENCE_THRESHOLD={PUPIL_CONFIDENCE_THRESHOLD:.2f}."
        ),
    )
    parser.add_argument(
        "--heatmap-only",
        action="store_true",
        help=(
            "Skip look-at grid; load last calib/device_calibration.json "
            "(or --load-calib) and start in heatmap mode"
        ),
    )
    parser.add_argument(
        "--load-calib",
        type=Path,
        default=None,
        help=(
            "Load this device calibration JSON and skip the look-at grid "
            "(default with --heatmap-only: <calib-dir>/device_calibration.json)"
        ),
    )
    parser.add_argument(
        "--front-uv-map",
        type=Path,
        default=None,
        help=(
            "FrontPixelGaze front_uv_map.json for red 2D overlay "
            f"(default: {default_front_uv_map_path()})"
        ),
    )
    parser.add_argument(
        "--no-front-uv-map",
        action="store_true",
        help="Disable FrontPixelGaze 2D overlay even if the map file exists",
    )
    add_standard_paths(parser)
    return parser.parse_args(argv)


def nine_or_n_grid(
    n: int,
    w: int,
    h: int,
    margin_frac: float = 0.12,
    *,
    bottom_clear_px: float = 0.0,
    side_clear_px: float = 0.0,
) -> list[tuple[float, float]]:
    """Square look-at grid; optional clearance for ArUco / front preview.

    ``side_clear_px`` / ``bottom_clear_px`` are alternate insets (not stacked on
    ``margin_frac``) — use max so targets stay clear of corner markers and PiPs.
    """
    side = int(round(n**0.5))
    if side * side != n:
        raise ValueError(f"grid size must be square (9/16/25), got {n}")
    side_clear = max(0.0, float(side_clear_px))
    mx = max(margin_frac * w, side_clear)
    my_top = max(margin_frac * h, side_clear)
    my_bottom = max(margin_frac * h, max(0.0, float(bottom_clear_px)))
    if my_top + my_bottom >= h:
        my_bottom = margin_frac * h
    if 2.0 * mx >= w:
        mx = margin_frac * w
    xs = np.linspace(mx, w - mx, side)
    ys = np.linspace(my_top, h - my_bottom, side)
    return [(float(x), float(y)) for y in ys for x in xs]


def _marker_clearance(markers) -> int:
    """Inset so UI stays out of corner ArUco rectangles."""
    return int(markers.margin + markers.marker_size + UI_PAD)


def _status_origin(screen_w: int, clear: int) -> tuple[int, int]:
    """Top status band between the upper ArUco markers."""
    return clear, clear


def _bottom_pip_rois(
    screen_w: int,
    screen_h: int,
    clear: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Eye + front PiPs as a bottom-center row, clear of corner markers."""
    gap = UI_PAD
    eye_w, eye_h = EYE_PIP_W, EYE_PIP_H
    front_w, front_h = PREVIEW_W, PREVIEW_H
    row_h = max(eye_h, front_h)
    total_w = eye_w + gap + front_w
    max_w = max(1, screen_w - 2 * clear)
    if total_w > max_w:
        scale = max_w / float(total_w)
        eye_w = max(80, int(eye_w * scale))
        eye_h = max(60, int(eye_h * scale))
        front_w = max(80, int(front_w * scale))
        front_h = max(60, int(front_h * scale))
        row_h = max(eye_h, front_h)
        total_w = eye_w + gap + front_w
    x0 = max(clear, (screen_w - total_w) // 2)
    y0 = screen_h - row_h - UI_PAD
    # Keep the row above the bottom marker band when the strip is wide.
    y0 = min(y0, screen_h - clear - row_h)
    y0 = max(clear, y0)
    eye_roi = (x0, y0 + (row_h - eye_h) // 2, eye_w, eye_h)
    front_roi = (x0 + eye_w + gap, y0 + (row_h - front_h) // 2, front_w, front_h)
    return eye_roi, front_roi


def _draw_target(canvas: NDArray[np.uint8], uv: tuple[float, float], label: str) -> None:
    u, v = int(uv[0]), int(uv[1])
    cv2.circle(canvas, (u, v), 18, (0, 0, 255), 2)
    cv2.circle(canvas, (u, v), 4, (0, 255, 255), -1)
    cv2.putText(
        canvas,
        label,
        (u + 22, v + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def _draw_ab_legend(
    canvas: NDArray[np.uint8],
    *,
    clear: int,
    has_frontpixel: bool,
) -> None:
    """Top-right legend: green = MultiCam 3D, red = FrontPixelGaze 2D."""
    h, w = canvas.shape[:2]
    box_w, box_h = 320, 78 if has_frontpixel else 52
    x1 = w - clear - UI_PAD
    y0 = clear + UI_PAD
    x0 = x1 - box_w
    y1 = y0 + box_h
    if x0 < clear or y1 > h - clear:
        return
    overlay = canvas[y0:y1, x0:x1].copy()
    cv2.rectangle(overlay, (0, 0), (box_w - 1, box_h - 1), (32, 32, 32), -1)
    cv2.addWeighted(overlay, 0.65, canvas[y0:y1, x0:x1], 0.35, 0, canvas[y0:y1, x0:x1])
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (180, 180, 180), 1)

    row1 = y0 + 22
    cx = x0 + 22
    cv2.circle(canvas, (cx, row1), 10, (0, 255, 0), 2)
    cv2.putText(
        canvas,
        "3D  MultiCam ray ∩ plane",
        (x0 + 42, row1 + 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    if has_frontpixel:
        row2 = y0 + 52
        cv2.drawMarker(
            canvas,
            (cx, row2),
            (0, 0, 255),
            markerType=cv2.MARKER_TILTED_CROSS,
            markerSize=16,
            thickness=2,
        )
        cv2.putText(
            canvas,
            "2D  FrontPixel yaw/pitch→UV",
            (x0 + 42, row2 + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )


def _accumulate_heatmap(
    heat: NDArray[np.float32],
    u: float,
    v: float,
    sigma: float = 28.0,
    gain: float = 1.0,
) -> None:
    h, w = heat.shape
    x0 = max(0, int(u - 3 * sigma))
    x1 = min(w, int(u + 3 * sigma) + 1)
    y0 = max(0, int(v - 3 * sigma))
    y1 = min(h, int(v + 3 * sigma) + 1)
    if x1 <= x0 or y1 <= y0:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    blob = np.exp(-((xs - u) ** 2 + (ys - v) ** 2) / (2.0 * sigma * sigma))
    heat[y0:y1, x0:x1] += (gain * blob).astype(np.float32)


def _overlay_heatmap(canvas: NDArray[np.uint8], heat: NDArray[np.float32]) -> NDArray[np.uint8]:
    if float(np.max(heat)) < 1e-6:
        return canvas
    norm = np.clip(heat / (np.max(heat) + 1e-6), 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(canvas, 0.55, colored, 0.45, 0)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    config_dir, calib_dir = resolve_paths(args)

    if args.dry_run:
        print("Would run left-eye heatmap MVP (front + left cameras).")
        return 0

    capture_seconds = max(0.5, float(args.capture_seconds))
    settle_seconds = max(0.0, float(args.settle_seconds))
    min_samples = max(5, int(args.min_samples))
    trim_frac = float(np.clip(args.trim_frac, 0.0, 0.45))
    max_rms_deg = max(0.1, float(args.max_rms_deg))
    max_angle_deg = max(max_rms_deg, float(args.max_angle_deg))
    min_aruco_markers = max(2, int(args.min_aruco_markers))
    max_aruco_reproj_px = max(0.5, float(args.max_aruco_reproj_px))
    max_pfront_z_range_mm = max(5.0, float(args.max_pfront_z_range_mm))
    min_pupil_confidence = float(np.clip(args.min_pupil_confidence, 0.0, 1.0))

    setup = load_camera_setup(config_dir / "camera_setup.json")
    front_idx, front_id = role_camera_index(setup, "front")
    left_idx, left_id = role_camera_index(setup, "left")
    if front_idx is None or left_idx is None:
        print("Need front and left in camera_setup.json", file=sys.stderr)
        return 1

    left_entry = setup.get("left") if isinstance(setup.get("left"), dict) else {}
    flip_v = bool(left_entry.get("flip", False))
    flip_h = bool(left_entry.get("mirror", False))
    eye_origin_front = eye_center_front_mm_from_setup(setup, "left")

    front_intrinsics = require_intrinsics(calib_dir, "front")

    front_reader = CameraReader(front_idx, args.width, args.height, device_id=front_id)
    left_reader = CameraReader(left_idx, args.width, args.height, device_id=left_id)
    front_reader.start()
    left_reader.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Query drawable size after fullscreen.
    _probe = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow(WINDOW_NAME, _probe)
    cv2.waitKey(1)
    # Prefer OS primary metrics - OpenCV window rect is often wrong on Windows and
    # mis-sized canvases place ArUco/targets where the front camera does not see them.
    native = primary_screen_size_px()
    rect = cv2.getWindowImageRect(WINDOW_NAME)
    rect_w, rect_h = int(rect[2] or 0), int(rect[3] or 0)
    if native is not None:
        screen_w, screen_h = native
        if rect_w > 0 and rect_h > 0 and (rect_w != screen_w or rect_h != screen_h):
            print(
                f"  Note: OpenCV window rect {rect_w}x{rect_h} != primary "
                f"{screen_w}x{screen_h}; using primary (ArUco must match monitor)."
            )
    elif rect_w >= 100 and rect_h >= 100:
        screen_w, screen_h = rect_w, rect_h
    else:
        screen_w, screen_h = 1920, 1080

    screen_cfg_path = config_dir / "screen.json"
    try:
        screen_size = resolve_screen_physical_size(
            screen_w_px=screen_w,
            screen_h_px=screen_h,
            diagonal_inches=args.diagonal_inches,
            width_mm=args.width_mm,
            height_mm=args.height_mm,
            config=load_screen_config(screen_cfg_path),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        front_reader.stop()
        left_reader.stop()
        return 1
    width_mm = screen_size.width_mm
    height_mm = screen_size.height_mm
    save_screen_config(screen_cfg_path, screen_size)

    gdi_w, gdi_h = estimate_screen_mm(screen_w, screen_h)
    gdi_diag = float(np.hypot(gdi_w, gdi_h) / 25.4)

    # Scale front intrinsics to live capture size once we have a frame.
    ok_f, front0 = front_reader.read()
    if not ok_f or front0 is None:
        print("Failed to read front camera", file=sys.stderr)
        front_reader.stop()
        left_reader.stop()
        return 1
    fh, fw = front0.shape[:2]
    front_k = front_intrinsics.scaled_for_frame(fw, fh)

    screen_pose = LiveScreenPose(screen_w, screen_h, width_mm, height_mm, front_k)
    clear = _marker_clearance(screen_pose.markers)
    # Keep look-ats clear of corner ArUco and above the bottom-center PiPs.
    pip_clear = float(max(PREVIEW_H, EYE_PIP_H) + 2 * UI_PAD)
    targets = nine_or_n_grid(
        args.grid,
        screen_w,
        screen_h,
        side_clear_px=float(clear),
        bottom_clear_px=max(float(clear), pip_clear),
    )
    samples: list[RayExtrinsicSample] = []
    burst: list[RayExtrinsicSample] = []
    capturing = False
    capture_t0 = 0.0
    front_from_left: Transform | None = None
    direction_scale = 1.0
    scale_yaw = 1.0
    scale_pitch = 1.0
    eye = EyeTrackerAdapter()
    setup_path = config_dir / "camera_setup.json"
    left_ir_lock = eye_center_ir_px_from_setup(setup, "left")
    if left_ir_lock is not None:
        eye.apply_locked_eye_center_ir_px("left", left_ir_lock)
        print(f"[left] Restored eye_center_ir_px lock at {left_ir_lock}")
    heat = np.zeros((screen_h, screen_w), dtype=np.float32)
    heatmap_on = False
    target_idx = 0
    skipped = 0
    mode = "warmup"  # warmup | calib | heatmap
    last_abort_reason = ""
    logged_ir_model = False
    status_x, status_y0 = _status_origin(screen_w, clear)
    eye_roi, front_roi = _bottom_pip_rois(screen_w, screen_h, clear)
    eye_x0, eye_y0, eye_tw, eye_th = eye_roi
    preview_x0, preview_y0, preview_tw, preview_th = front_roi
    debug_compares: list[GazeDebugCompare] = []  # on-screen (last N)
    debug_compares_all: list[GazeDebugCompare] = []  # full session for --dump-samples
    last_solve: RayExtrinsicResult | None = None
    # Live snapshot for right-click compare (updated each frame in heatmap mode).
    live_debug: dict = {
        "pred_uv": None,
        "pred_uv_2d": None,
        "ray_origin": None,
        "ray_dir": None,
        "pose_est": None,
    }

    front_uv_map: FrontUvMap | None = None
    if not args.no_front_uv_map:
        map_path = (
            Path(args.front_uv_map)
            if args.front_uv_map is not None
            else default_front_uv_map_path()
        )
        front_uv_map = load_front_uv_map(map_path)
        if front_uv_map is not None:
            print(
                f"FrontPixelGaze 2D map: {map_path}  "
                f"rms={front_uv_map.rms_px:.1f}px n={front_uv_map.n_samples}"
            )
        else:
            print(
                f"FrontPixelGaze 2D map not found ({map_path}); "
                "green 3D only. Fit in FrontPixelGaze or pass --front-uv-map."
            )

    def _flush_dump(*, quiet: bool = False) -> None:
        if args.dump_samples is None or last_solve is None:
            return
        _dump_samples(
            args.dump_samples,
            samples,
            last_solve,
            debug_compares=debug_compares_all,
            quiet=quiet,
        )

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event != cv2.EVENT_RBUTTONDOWN:
            return
        if mode != "heatmap" or not heatmap_on or front_from_left is None:
            return
        if _in_roi(x, y, eye_roi) or _in_roi(x, y, front_roi):
            print("Debug click ignored (hit IR/front PiP).")
            return
        if not (0 <= x < screen_w and 0 <= y < screen_h):
            return
        truth = (float(x), float(y))
        pred = live_debug.get("pred_uv")
        pred_2d = live_debug.get("pred_uv_2d")
        pose = live_debug.get("pose_est")
        truth_front = None
        if pose is not None:
            truth_front = screen_pose.target_front(truth[0], truth[1], pose)
        cmp_ = _compare_gaze_click(
            truth_uv=truth,
            pred_uv=pred,
            screen_w=screen_w,
            screen_h=screen_h,
            width_mm=width_mm,
            height_mm=height_mm,
            truth_front=truth_front,
            ray_origin_front=live_debug.get("ray_origin"),
            ray_dir_front=live_debug.get("ray_dir"),
        )
        debug_compares_all.append(cmp_)
        debug_compares.append(cmp_)
        if len(debug_compares) > MAX_DEBUG_COMPARES:
            del debug_compares[0 : len(debug_compares) - MAX_DEBUG_COMPARES]
        if cmp_.pred_uv is None and pred_2d is None:
            print(
                f"Debug click truth=({truth[0]:.0f},{truth[1]:.0f}) — "
                "no predicted gaze hit this frame"
            )
            _flush_dump(quiet=True)
            return
        parts = [f"Debug click truth=({truth[0]:.0f},{truth[1]:.0f})"]
        if cmp_.pred_uv is not None:
            parts.append(
                f"3D=({cmp_.pred_uv[0]:.0f},{cmp_.pred_uv[1]:.0f}) "
                f"Δ=({cmp_.du_px:+.0f},{cmp_.dv_px:+.0f})px "
                f"|err|={cmp_.err_px:.0f}px/{cmp_.err_mm:.1f}mm"
            )
        if pred_2d is not None:
            du2 = float(pred_2d[0] - truth[0])
            dv2 = float(pred_2d[1] - truth[1])
            err2 = float(np.hypot(du2, dv2))
            sx = float(width_mm) / max(float(screen_w), 1.0)
            sy = float(height_mm) / max(float(screen_h), 1.0)
            err2_mm = float(np.hypot(du2 * sx, dv2 * sy))
            parts.append(
                f"2D=({pred_2d[0]:.0f},{pred_2d[1]:.0f}) "
                f"Δ=({du2:+.0f},{dv2:+.0f})px "
                f"|err|={err2:.0f}px/{err2_mm:.1f}mm"
            )
        if cmp_.ang_deg is not None and cmp_.miss_ray_mm is not None:
            parts.append(f"ang={cmp_.ang_deg:.2f}° ray⊥={cmp_.miss_ray_mm:.1f}mm")
        print("  ".join(parts))

        # Summarize axis bias across all persisted clicks (helps spot Y-only error).
        with_pred = [c for c in debug_compares_all if c.du_px is not None]
        if len(with_pred) >= 2:
            mean_du = float(np.mean([c.du_px for c in with_pred]))  # type: ignore[arg-type]
            mean_dv = float(np.mean([c.dv_px for c in with_pred]))  # type: ignore[arg-type]
            mean_err = float(np.mean([c.err_mm for c in with_pred]))  # type: ignore[arg-type]
            print(
                f"  debug mean over {len(with_pred)}: "
                f"Δu={mean_du:+.0f}px  Δv={mean_dv:+.0f}px  |err|={mean_err:.1f}mm"
            )
        _flush_dump(quiet=True)
        if args.dump_samples is not None:
            print(
                f"  saved debug click #{len(debug_compares_all)} -> {args.dump_samples}"
            )

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    store = CalibrationStore(calib_dir)
    skip_grid = bool(args.heatmap_only or args.load_calib is not None)
    if skip_grid:
        load_path = args.load_calib if args.load_calib is not None else store.device_path
        try:
            loaded = store.load_left_heatmap(load_path)
        except (OSError, CalibrationSchemaError, KeyError, TypeError, ValueError) as exc:
            print(f"Cannot load calibration for heatmap-only: {exc}", file=sys.stderr)
            front_reader.stop()
            left_reader.stop()
            cv2.destroyAllWindows()
            return 1
        front_from_left = loaded.front_from_left
        direction_scale = loaded.direction_scale
        scale_yaw = loaded.scale_yaw
        scale_pitch = loaded.scale_pitch
        mode = "heatmap"
        heatmap_on = True
        e_loaded = np.asarray(front_from_left.translation, dtype=np.float64).reshape(3)
        e_cfg = np.asarray(eye_origin_front, dtype=np.float64).reshape(3)
        print(
            f"Loaded left heatmap calib from {loaded.path}  "
            f"E={e_loaded.tolist()}  "
            f"yaw={scale_yaw:.3f} pitch={scale_pitch:.3f}"
            + (
                f"  saved_err={loaded.gaze_error_mm:.1f}mm"
                if loaded.gaze_error_mm is not None
                else ""
            )
        )
        if float(np.linalg.norm(e_loaded - e_cfg)) > 1.0:
            print(
                f"WARN: camera_setup eye_center_front_mm={e_cfg.tolist()} differs from "
                f"loaded transform translation={e_loaded.tolist()} — heatmap uses "
                "the loaded file (re-solve after changing E).",
                file=sys.stderr,
            )
        print("Heatmap-only: look around until IR is ready; no look-at grid.")

    def _finish_if_ready(*, auto_heatmap: bool) -> bool:
        """Solve when we have enough look-ats. Returns True if solved."""
        nonlocal front_from_left, direction_scale, scale_yaw, scale_pitch
        nonlocal mode, heatmap_on, last_solve
        if front_from_left is not None:
            return True
        if len(samples) < 3:
            print(
                f"Need at least 3 captured look-ats to solve "
                f"(have {len(samples)}, skipped {skipped}).",
                file=sys.stderr,
            )
            return False
        result = solve_front_from_eye(
            samples,
            eye_frame="left_eye",
            ray_origin_front=eye_origin_front,
        )
        front_from_left = result.front_from_eye
        direction_scale = result.direction_scale
        scale_yaw = result.scale_yaw
        scale_pitch = result.scale_pitch
        last_solve = result
        _report_solve(result, samples)
        _persist(store, result, screen_size)
        if args.dump_samples is not None:
            _dump_samples(
                args.dump_samples,
                samples,
                result,
                debug_compares=debug_compares_all,
            )
        print(
            f"Calibration complete. mean residual "
            f"{result.mean_residual_mm:.2f} mm / "
            f"{result.mean_residual_deg:.2f} deg "
            f"(yaw={result.scale_yaw:.2f}, pitch={result.scale_pitch:.2f}, "
            f"used={result.n_samples_used}/{len(samples)}, skipped={skipped})"
        )
        if result.mean_residual_mm > MAX_ACCEPTABLE_RESIDUAL_MM:
            print(
                f"WARNING: residual > {MAX_ACCEPTABLE_RESIDUAL_MM:.0f} mm - "
                "3D fit is weak; fix ArUco/FOV before trusting heatmap.",
                file=sys.stderr,
            )
        if auto_heatmap:
            print(
                "Press H for heatmap mode "
                "(green=3D MultiCam, red=2D FrontPixelGaze)."
            )
            mode = "heatmap"
            heatmap_on = True
        return True

    print("Left Heatmap MVP")
    print(
        f"  Screen {screen_w}x{screen_h} px | "
        f'{screen_size.diagonal_inches:g}" → {width_mm:.1f}x{height_mm:.1f} mm '
        f"({screen_size.source})"
    )
    print(
        f"  OS/GDI estimate (not used for scale): "
        f"{gdi_w:.1f}x{gdi_h:.1f} mm (~{gdi_diag:.1f}\")"
    )
    print(
        f"  Grid: {args.grid} pts | {capture_seconds:.1f}s capture "
        f"(+{settle_seconds:.1f}s settle) | min {min_samples} frames | "
        f"abort if RMS>{max_rms_deg:.1f}° or max>{max_angle_deg:.1f}°"
    )
    print(
        f"  ArUco gates: markers>={min_aruco_markers}  "
        f"reproj<={max_aruco_reproj_px:.1f}px  "
        f"P_front Z range<={max_pfront_z_range_mm:.0f}mm"
    )
    print(f"  Pupil confidence gate: >={min_pupil_confidence:.2f} (look-at frames only)")
    if front_uv_map is not None:
        print(
            "  Heatmap: green=3D MultiCam ray∩plane; "
            "red=2D FrontPixelGaze yaw/pitch→UV."
        )
    else:
        print("  Heatmap: green=3D MultiCam ray∩plane (no FrontPixelGaze map loaded).")
    if skip_grid:
        print("  Mode: heatmap-only (loaded device calibration; no look-at grid).")
        print("  Look around until IR is ready; right-click to debug; Q to quit.")
    else:
        print("  Warmup first: look around until eyeball radius adapts (see left IR PiP).")
        print("  Then look at each red target, hold still, press SPACE.")
        print(
            "  N = skip this target (glint / bad pupil zone); "
            "need >=3 kept points to solve."
        )
    print(
        "  Heatmap right-click: GT vs green 3D (+ red 2D if loaded); "
        "C clears debug marks."
    )
    if flip_v or flip_h:
        print(
            f"  Left IR flips: vertical={flip_v} horizontal={flip_h} "
            f"(applied inside process_frame; undone for K)"
        )
    # Probe IR unproject once we have a live left frame (printed after first read).
    print(
        f"  Left IR unproject: crop/flip remap + Phase-0 K "
        f"(fallback FOV_y={float(IR_FOV_Y_DEG):.0f}°)"
    )
    print(
        f"  Eye center (front mm): "
        f"[{eye_origin_front[0]:.1f}, {eye_origin_front[1]:.1f}, {eye_origin_front[2]:.1f}] "
        f"(camera_setup left.eye_center_front_mm; OpenCV X right, Y down, Z forward)"
    )

    try:
        while True:
            ok_f, front = front_reader.read()
            ok_l, left = left_reader.read()
            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            canvas[:] = (30, 30, 30)
            # ArUco pasted last so status/PiPs never cover markers for the front camera.

            pose_est = None
            if ok_f and front is not None:
                pose_est = screen_pose.estimate(front)

            gaze = None
            pupil_conf = 0.0
            eye_panel = None
            readiness = eye.get_tracking_readiness("left")
            if ok_l and left is not None:
                track, use_flip_v, use_flip_h = prepare_eye_tracking_frame(
                    left,
                    flip_v,
                    flip_h,
                )
                lh, lw = left.shape[:2]
                unproject = resolve_eye_unproject_model(
                    sensor_width=lw,
                    sensor_height=lh,
                    flip_vertical=flip_v,
                    flip_horizontal=flip_h,
                    calib_dir=calib_dir,
                    role="left_eye",
                )
                if not logged_ir_model:
                    k_note = (
                        f"K/dist from {unproject.intrinsics_source}"
                        if unproject.camera_matrix is not None
                        else f"FOV_y={unproject.fov_y_deg:.1f}° ({unproject.intrinsics_source})"
                    )
                    print(f"  IR model: {k_note}, vfov≈{unproject.fov_y_deg:.1f}°")
                    logged_ir_model = True
                eye.process_frame(
                    track,
                    eye_id="left",
                    flip_vertical=use_flip_v,
                    flip_horizontal=use_flip_h,
                    unproject=unproject,
                )
                gaze = eye.get_gaze_ray("left")
                pupil_conf = eye.get_pupil_confidence("left")
                readiness = eye.get_tracking_readiness("left")
                eye_panel = compose_eye_preview_panel(
                    left,
                    eye.get_preview_frame("left"),
                    flip_v,
                    flip_h,
                )

            if mode == "warmup" and readiness.ready:
                # Freeze validated IR center so it cannot drift during look-ats.
                frozen = eye.freeze_sphere_center("left")
                if frozen is not None:
                    set_eye_center_ir_px(setup, "left", frozen)
                    save_camera_setup(setup_path, setup)
                    print(
                        f"[left] Locked eye_center_ir_px={list(frozen)} "
                        f"for calib (saved {setup_path.name})."
                    )
                mode = "heatmap" if skip_grid else "calib"
                print(
                    f"Eyeball ready: radius≈{readiness.radius_px:.0f}px, "
                    f"centers={readiness.n_model_centers}. Start look-at calibration."
                )

            pupil_ok = (
                gaze is not None
                and gaze.valid
                and pupil_conf >= min_pupil_confidence
            )
            status_lines = [
                f"mode={mode}  aruco={'OK' if pose_est else 'NO'}  "
                f"pupil={'OK' if pupil_ok else 'NO'}"
                f"({pupil_conf:.2f}>={min_pupil_confidence:.2f})  "
                f"kept={len(samples)}  skip={skipped}  "
                f"target={min(target_idx + 1, len(targets))}/{len(targets)}"
            ]
            if eye.is_sphere_center_locked("left"):
                center_status = "center=LOCKED"
            else:
                center_status = (
                    f"centers={readiness.n_model_centers}/{readiness.min_model_centers}"
                )
            status_lines.append(
                f"eye: {center_status}  "
                f"radius={'OK ' + f'{readiness.radius_px:.0f}px' if readiness.radius_adapted else 'warming…'}"
            )
            if pose_est is not None:
                dist_mm = camera_to_screen_distance_mm(pose_est.front_from_screen)
                status_lines.append(
                    f"reproj={pose_est.reprojection_error_px:.2f}px  "
                    f"markers={pose_est.markers_found}  "
                    f"cam↔screen={dist_mm / 1000.0:.3f} m ({dist_mm:.0f} mm)"
                )
            status_lines.append(
                f'screen {screen_size.diagonal_inches:g}"  '
                f"{width_mm:.0f}x{height_mm:.0f} mm  ({screen_size.source})"
            )
            if front_from_left is not None:
                status_lines.append("T_front<-left ready")
            if last_abort_reason:
                status_lines.append(f"last abort: {last_abort_reason}")

            if mode == "warmup":
                status_lines.append(
                    "Warmup: look around slowly until radius adapts (need confident pupil)"
                )
            elif mode == "calib" and target_idx < len(targets):
                uv = targets[target_idx]
                _draw_target(canvas, uv, f"{target_idx + 1}/{len(targets)}")
                if capturing:
                    elapsed = time.perf_counter() - capture_t0
                    if elapsed < settle_seconds:
                        status_lines.append(
                            f"Settling… {elapsed:.1f}/{settle_seconds:.1f}s  hold still"
                        )
                    else:
                        held = elapsed - settle_seconds
                        status_lines.append(
                            f"Capturing… {held:.1f}/{capture_seconds:.1f}s  "
                            f"n={len(burst)} (need >={min_samples})  hold gaze"
                        )
                else:
                    status_lines.append(
                        f"SPACE = capture  |  N = skip this target (glint/pupil)"
                    )
            elif mode == "calib" and target_idx >= len(targets):
                status_lines.append(
                    f"Grid done (kept={len(samples)}, skipped={skipped}). "
                    "Solving… (auto) or H for heatmap"
                )

            live_debug["pred_uv"] = None
            live_debug["pred_uv_2d"] = None
            live_debug["ray_origin"] = None
            live_debug["ray_dir"] = None
            live_debug["pose_est"] = pose_est
            if mode == "heatmap" and heatmap_on and front_from_left is not None:
                canvas = _overlay_heatmap(canvas, heat)
                if (
                    gaze is not None
                    and gaze.valid
                    and pose_est is not None
                ):
                    screen = screen_pose.screen_model(pose_est)
                    if screen is not None:
                        d_eye = scale_gaze_direction(
                            gaze.direction,
                            direction_scale,
                            scale_yaw=scale_yaw,
                            scale_pitch=scale_pitch,
                        )
                        # Origin 0 in eye frame → front_from_left.translation
                        # (= eye_center_front_mm) in world≡front.
                        scaled = GazeRayInCameraFrame(
                            np.zeros(3, dtype=np.float64),
                            d_eye,
                            gaze.eye_id,
                            valid=True,
                        )
                        ray = gaze_to_world_ray(scaled, front_from_left)
                        live_debug["ray_origin"] = ray.origin.copy()
                        live_debug["ray_dir"] = ray.direction.copy()
                        hit = intersect_gaze_with_screen(
                            ray, screen, screen_w, screen_h
                        )
                        if hit.valid and hit.pixels is not None:
                            pu, pv = hit.pixels
                            live_debug["pred_uv"] = (float(pu), float(pv))
                            if 0.0 <= pu < screen_w and 0.0 <= pv < screen_h:
                                _accumulate_heatmap(heat, pu, pv)
                                cv2.circle(
                                    canvas, (int(pu), int(pv)), 10, (0, 255, 0), 2
                                )
                    # FrontPixelGaze 2D: raw eye dir → front UV → ArUco H → screen.
                    if (
                        front_uv_map is not None
                        and ok_f
                        and front is not None
                        and pose_est.front_to_screen_h is not None
                        and pose_est.front_quad_corners is not None
                    ):
                        fh, fw = front.shape[:2]
                        front_uv = front_uv_map.apply(
                            gaze.direction, image_size=(fw, fh)
                        )
                        screen_uv = front_uv_to_screen(
                            front_uv,
                            pose_est.front_to_screen_h,
                            pose_est.front_quad_corners,
                        )
                        if screen_uv is not None:
                            su, sv = screen_uv
                            live_debug["pred_uv_2d"] = (su, sv)
                            if 0.0 <= su < screen_w and 0.0 <= sv < screen_h:
                                cv2.drawMarker(
                                    canvas,
                                    (int(round(su)), int(round(sv))),
                                    (0, 0, 255),
                                    markerType=cv2.MARKER_TILTED_CROSS,
                                    markerSize=18,
                                    thickness=2,
                                )
                _draw_ab_legend(
                    canvas,
                    clear=clear,
                    has_frontpixel=front_uv_map is not None,
                )
                if debug_compares:
                    _draw_debug_compares(canvas, debug_compares)
                    status_lines.append(
                        f"debug clicks={len(debug_compares)}  "
                        "RMB=compare look@click  C=clear marks"
                    )

            y = status_y0 + 24
            for line in status_lines:
                cv2.putText(
                    canvas,
                    line,
                    (status_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (220, 220, 220),
                    2,
                )
                y += 26

            # Left IR PiP - same flip/mirror as multcam-preview.
            if eye_panel is not None:
                thumb = cv2.resize(eye_panel, (eye_tw, eye_th))
                label = "eye READY" if readiness.ready else "eye warmup"
                color = (0, 255, 0) if readiness.ready else (0, 165, 255)
                cv2.putText(
                    thumb, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
                )
                canvas[eye_y0 : eye_y0 + eye_th, eye_x0 : eye_x0 + eye_tw] = thumb

            # Front preview beside eye PiP (bottom-center row).
            if ok_f and front is not None:
                thumb = cv2.resize(front, (preview_tw, preview_th))
                if pose_est is not None:
                    cv2.putText(
                        thumb,
                        "ArUco OK",
                        (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                canvas[
                    preview_y0 : preview_y0 + preview_th,
                    preview_x0 : preview_x0 + preview_tw,
                ] = thumb

            # Markers on top - never occluded by status/PiPs.
            screen_pose.markers.paste_on(canvas)
            if capturing and mode == "calib" and target_idx < len(targets):
                elapsed = time.perf_counter() - capture_t0
                if elapsed >= settle_seconds:
                    if pose_est is not None and pupil_ok:
                        p = screen_pose.target_front(
                            targets[target_idx][0], targets[target_idx][1], pose_est
                        )
                        if p is not None:
                            tu, tv = targets[target_idx]
                            burst.append(
                                RayExtrinsicSample(
                                    origin=gaze.origin.copy(),
                                    direction=gaze.direction.copy(),
                                    target_front=p.copy(),
                                    target_uv=(float(tu), float(tv)),
                                    aruco_reproj_px=float(
                                        pose_est.reprojection_error_px
                                    ),
                                    aruco_markers=int(pose_est.markers_found),
                                )
                            )
                if elapsed >= settle_seconds + capture_seconds:
                    quality = assess_burst_stability(
                        burst,
                        min_samples=min_samples,
                        max_rms_deg=max_rms_deg,
                        max_angle_deg=max_angle_deg,
                        min_aruco_markers=min_aruco_markers,
                        max_aruco_reproj_px=max_aruco_reproj_px,
                        max_pfront_z_range_mm=max_pfront_z_range_mm,
                    )
                    capturing = False
                    if not quality.ok:
                        last_abort_reason = quality.reason
                        print(
                            f"ABORT target {target_idx + 1}/{len(targets)}: "
                            f"{quality.reason} (n={quality.n}, "
                            f"rms={quality.rms_angle_deg:.1f}°, "
                            f"max={quality.max_angle_deg:.1f}°, "
                            f"markers>={quality.min_aruco_markers}, "
                            f"reproj={quality.mean_aruco_reproj_px:.2f}px, "
                            f"ZΔ={quality.p_front_z_range_mm:.1f}mm). Retry SPACE."
                        )
                        burst = []
                    else:
                        avg = robust_average_samples(burst, trim_frac=trim_frac)
                        samples.append(avg)
                        last_abort_reason = ""
                        d = avg.direction
                        uv_txt = (
                            f"uv=({avg.target_uv[0]:.0f},{avg.target_uv[1]:.0f})  "
                            if avg.target_uv is not None
                            else ""
                        )
                        aruco_txt = ""
                        if avg.aruco_reproj_px is not None:
                            aruco_txt = (
                                f"aruco reproj={avg.aruco_reproj_px:.2f}px "
                                f"markers={avg.aruco_markers}  "
                                f"ZΔ={quality.p_front_z_range_mm:.1f}mm  "
                            )
                        p = avg.target_front
                        print(
                            f"Captured {len(samples)}/{len(targets)}  "
                            f"(n={quality.n}, rms={quality.rms_angle_deg:.2f}°, "
                            f"trim={trim_frac:.0%})  {uv_txt}{aruco_txt}"
                            f"dir=({d[0]:+.3f},{d[1]:+.3f},{d[2]:+.3f})  "
                            f"P_front={np.array2string(p, precision=1)}"
                        )
                        burst = []
                        target_idx += 1
                        if target_idx >= len(targets):
                            _finish_if_ready(auto_heatmap=True)

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("r"), ord("R")):
                heat[:] = 0.0

            if key in (ord("c"), ord("C")) and mode == "heatmap":
                debug_compares.clear()
                print(
                    "Cleared on-screen debug marks "
                    f"(persisted dump still has {len(debug_compares_all)} clicks)."
                )

            if key in (ord("n"), ord("N")) and mode == "calib":
                if target_idx >= len(targets):
                    continue
                was_capturing = capturing
                capturing = False
                burst = []
                skipped += 1
                uv = targets[target_idx]
                print(
                    f"SKIP target {target_idx + 1}/{len(targets)} "
                    f"uv=({uv[0]:.0f},{uv[1]:.0f})"
                    + (" (cancelled in-progress capture)" if was_capturing else "")
                    + f"  kept={len(samples)}  skipped={skipped}"
                )
                last_abort_reason = f"skipped target {target_idx + 1}"
                target_idx += 1
                if target_idx >= len(targets):
                    _finish_if_ready(auto_heatmap=True)

            if key in (ord("h"), ord("H")):
                if _finish_if_ready(auto_heatmap=False) or front_from_left is not None:
                    mode = "heatmap"
                    heatmap_on = True
                    print("Heatmap mode ON (ray ∩ live ArUco)")
                else:
                    print("Need calibration samples before heatmap", file=sys.stderr)

            if key == ord(" ") and mode == "calib":
                if target_idx >= len(targets) or capturing:
                    continue
                if not readiness.ready:
                    print("Skip: eyeball size still warming up - look around first")
                    continue
                if pose_est is None:
                    print("Skip: ArUco not valid")
                    continue
                if pose_est.markers_found < min_aruco_markers:
                    print(
                        f"Skip: ArUco markers {pose_est.markers_found} "
                        f"< {min_aruco_markers} - keep all corners visible"
                    )
                    continue
                if pose_est.reprojection_error_px > max_aruco_reproj_px:
                    print(
                        f"Skip: ArUco reproj {pose_est.reprojection_error_px:.2f}px "
                        f"> {max_aruco_reproj_px:.2f}px"
                    )
                    continue
                if gaze is None or not gaze.valid:
                    print("Skip: left pupil/gaze not valid")
                    continue
                burst = []
                capturing = True
                capture_t0 = time.perf_counter()
                last_abort_reason = ""
                print(
                    f"Target {target_idx + 1}/{len(targets)}: "
                    f"settle {settle_seconds:.1f}s + capture {capture_seconds:.1f}s "
                    f"(ArUco markers={pose_est.markers_found}, "
                    f"reproj={pose_est.reprojection_error_px:.2f}px)…"
                )

    finally:
        if args.dump_samples is not None and last_solve is not None:
            _dump_samples(
                args.dump_samples,
                samples,
                last_solve,
                debug_compares=debug_compares_all,
            )
        front_reader.stop()
        left_reader.stop()
        cv2.destroyAllWindows()

    return 0


def _debug_compares_payload(
    compares: list[GazeDebugCompare],
) -> dict[str, NDArray[np.float64]]:
    """Arrays for right-click ground-truth vs predicted gaze (heatmap debug)."""
    n = len(compares)
    truth = np.full((n, 2), np.nan, dtype=np.float64)
    pred = np.full((n, 2), np.nan, dtype=np.float64)
    du = np.full(n, np.nan, dtype=np.float64)
    dv = np.full(n, np.nan, dtype=np.float64)
    err_px = np.full(n, np.nan, dtype=np.float64)
    err_mm = np.full(n, np.nan, dtype=np.float64)
    miss = np.full(n, np.nan, dtype=np.float64)
    ang = np.full(n, np.nan, dtype=np.float64)
    for i, c in enumerate(compares):
        truth[i] = c.truth_uv
        if c.pred_uv is not None:
            pred[i] = c.pred_uv
        if c.du_px is not None:
            du[i] = c.du_px
        if c.dv_px is not None:
            dv[i] = c.dv_px
        if c.err_px is not None:
            err_px[i] = c.err_px
        if c.err_mm is not None:
            err_mm[i] = c.err_mm
        if c.miss_ray_mm is not None:
            miss[i] = c.miss_ray_mm
        if c.ang_deg is not None:
            ang[i] = c.ang_deg
    return {
        "debug_truth_uv": truth,
        "debug_pred_uv": pred,
        "debug_du_px": du,
        "debug_dv_px": dv,
        "debug_err_px": err_px,
        "debug_err_mm": err_mm,
        "debug_miss_ray_mm": miss,
        "debug_ang_deg": ang,
    }


def _dump_samples(
    path: Path,
    samples: list[RayExtrinsicSample],
    result: RayExtrinsicResult,
    *,
    debug_compares: list[GazeDebugCompare] | None = None,
    quiet: bool = False,
) -> None:
    """Save look-at samples (+ optional debug clicks) for offline checks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dirs = np.stack([np.asarray(s.direction, dtype=np.float64) for s in samples])
    origins = np.stack([np.asarray(s.origin, dtype=np.float64) for s in samples])
    targets = np.stack([np.asarray(s.target_front, dtype=np.float64) for s in samples])
    uvs = np.array(
        [
            s.target_uv if s.target_uv is not None else (np.nan, np.nan)
            for s in samples
        ],
        dtype=np.float64,
    )
    reproj = np.array(
        [
            s.aruco_reproj_px if s.aruco_reproj_px is not None else np.nan
            for s in samples
        ],
        dtype=np.float64,
    )
    markers = np.array(
        [
            float(s.aruco_markers) if s.aruco_markers is not None else np.nan
            for s in samples
        ],
        dtype=np.float64,
    )
    plane_dist = plane_distances_mm(samples)
    payload: dict[str, NDArray[np.float64]] = {
        "directions": dirs,
        "origins": origins,
        "targets_front": targets,
        "target_uvs": uvs,
        "aruco_reproj_px": reproj,
        "aruco_markers": markers,
        "plane_dist_mm": plane_dist,
        "direction_scale": np.array([result.direction_scale], dtype=np.float64),
        "scale_yaw": np.array([result.scale_yaw], dtype=np.float64),
        "scale_pitch": np.array([result.scale_pitch], dtype=np.float64),
        "mean_residual_mm": np.array([result.mean_residual_mm], dtype=np.float64),
        "mean_residual_deg": np.array([result.mean_residual_deg], dtype=np.float64),
        "eye_front": result.front_from_eye.translation,
        "R_front_from_eye": result.front_from_eye.rotation,
    }
    if debug_compares:
        payload.update(_debug_compares_payload(debug_compares))
    np.savez(path, **payload)
    if quiet:
        return
    extra = (
        f" + {len(debug_compares)} debug clicks"
        if debug_compares
        else ""
    )
    print(f"Dumped {len(samples)} samples{extra} -> {path}")


def _report_solve(result: RayExtrinsicResult, samples: list[RayExtrinsicSample]) -> None:
    """Print per-point residuals so a bad solve is obvious before heatmap mode."""
    print(
        f"Per-point residuals (miss mm / angular deg)  "
        f"yaw={result.scale_yaw:.3f} pitch={result.scale_pitch:.3f} "
        f"used={result.n_samples_used}:"
    )
    eye = result.front_from_eye.translation
    plane_dist = plane_distances_mm(samples)
    for i, s in enumerate(samples):
        d = result.front_from_eye.apply_direction(
            scale_gaze_direction(
                s.direction,
                result.direction_scale,
                scale_yaw=result.scale_yaw,
                scale_pitch=result.scale_pitch,
            )
        )
        mm = point_to_ray_distance(eye, d, s.target_front)
        deg = angular_error_deg(eye, d, s.target_front)
        p = np.asarray(s.target_front, dtype=np.float64).reshape(3)
        aruco = ""
        if s.aruco_reproj_px is not None:
            aruco = (
                f"  aruco={s.aruco_reproj_px:.2f}px/"
                f"{s.aruco_markers}  plane={plane_dist[i]:+.1f}mm"
            )
        print(
            f"  [{i + 1:02d}]  {mm:7.1f} mm   {deg:5.2f}°  "
            f"Z={p[2]:.0f}mm{aruco}"
        )
    print(
        f"  mean {result.mean_residual_mm:.1f} mm / {result.mean_residual_deg:.2f}°"
    )
    if len(plane_dist):
        abs_d = np.abs(plane_dist)
        print(
            f"  P_front plane |dist| mean={float(abs_d.mean()):.1f} mm  "
            f"max={float(abs_d.max()):.1f} mm  "
            f"Z span={float(np.ptp([s.target_front[2] for s in samples])):.1f} mm"
        )
    if result.mean_residual_px is not None:
        print(
            f"  screen affine polish residual: {result.mean_residual_px:.1f} px "
            "(diagnostic only; heatmap ignores)"
        )
    if result.yaw_pitch_uv_rmse_px is not None:
        print(
            f"  yaw/pitch->UV map RMSE: {result.yaw_pitch_uv_rmse_px:.1f} px "
            "(diagnostic only; heatmap ignores)"
        )


def _persist(
    store: CalibrationStore,
    result: RayExtrinsicResult,
    screen_size: ScreenPhysicalSize,
) -> None:
    front_from_left = result.front_from_eye
    data = store.load()
    data["coordinate_system"] = "front_screen"
    data["extrinsics"]["front_from_left_eye"] = store.transform_to_json(front_from_left)
    # world ≡ front for this MVP
    data["extrinsics"]["world_from_left_eye"] = store.transform_to_json(
        Transform(
            front_from_left.matrix.copy(),
            parent_frame="world",
            child_frame="left_eye",
        )
    )
    data["extrinsics"]["world_from_front"] = store.transform_to_json(
        Transform(np.eye(4, dtype=np.float64), "world", "front")
    )
    data["screen"]["width_mm"] = float(screen_size.width_mm)
    data["screen"]["height_mm"] = float(screen_size.height_mm)
    data["screen"]["diagonal_inches"] = float(screen_size.diagonal_inches)
    data["screen"]["size_source"] = screen_size.source
    data["quality"]["gaze_error_mm"] = float(result.mean_residual_mm)
    left_gaze = {
        "direction_scale": float(result.direction_scale),
        "scale_yaw": float(result.scale_yaw),
        "scale_pitch": float(result.scale_pitch),
        "n_samples_used": int(result.n_samples_used),
        "heatmap_path": "ray_intersect_live_aruco",
    }
    # Keep 2D maps in JSON as diagnostics only (not consumed by heatmap).
    if result.screen_affine_2x3 is not None:
        left_gaze["screen_affine_2x3"] = result.screen_affine_2x3.tolist()
    if result.mean_residual_px is not None:
        left_gaze["affine_residual_px"] = float(result.mean_residual_px)
    if result.yaw_pitch_to_uv_2x3 is not None:
        left_gaze["yaw_pitch_to_uv_2x3"] = result.yaw_pitch_to_uv_2x3.tolist()
    if result.yaw_pitch_uv_rmse_px is not None:
        left_gaze["yaw_pitch_uv_rmse_px"] = float(result.yaw_pitch_uv_rmse_px)
    data["gaze_calibration"]["left"] = left_gaze
    store.save(data)
    print(f"Saved {store.device_path}")


if __name__ == "__main__":
    raise SystemExit(main())
