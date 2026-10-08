"""Display a chessboard pattern for Phase 0 intrinsics (monitor or second screen)."""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from multcam_gaze.apps._cli_common import add_standard_paths, resolve_paths
from multcam_gaze.vision.chessboard import load_chessboard_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a chessboard matching config/chessboard.json (windowed, with margin)."
    )
    parser.add_argument(
        "--square-px",
        type=int,
        default=80,
        help="Pixels per square before display (sharpness; not physical mm).",
    )
    parser.add_argument(
        "--margin-squares",
        type=int,
        default=1,
        help="Quiet white border thickness in squares (helps detection).",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Fill the window/monitor (stretches the pattern).",
    )
    add_standard_paths(parser)
    return parser.parse_args(argv)


def build_board_image(
    inner_cols: int,
    inner_rows: int,
    square_px: int,
    margin_squares: int,
) -> np.ndarray:
    """Build board with (inner_cols+1) x (inner_rows+1) squares + white margin."""
    sq = max(8, square_px)
    margin = max(0, margin_squares)
    n_sq_x = inner_cols + 1
    n_sq_y = inner_rows + 1
    board_w = n_sq_x * sq
    board_h = n_sq_y * sq
    pad = margin * sq
    canvas = np.full((board_h + 2 * pad, board_w + 2 * pad), 255, dtype=np.uint8)
    for y in range(n_sq_y):
        for x in range(n_sq_x):
            if (x + y) % 2 == 0:
                y0 = pad + y * sq
                x0 = pad + x * sq
                canvas[y0 : y0 + sq, x0 : x0 + sq] = 0
    return canvas


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_dir, _ = resolve_paths(args)
    cb_cfg = load_chessboard_config(config_dir / "chessboard.json")
    cols = int(cb_cfg["inner_corners_cols"])
    rows = int(cb_cfg["inner_corners_rows"])
    square_mm = float(cb_cfg["square_size_mm"])

    img = build_board_image(cols, rows, args.square_px, args.margin_squares)
    window = "Chessboard (Phase 0)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    if args.fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        # Comfortable default window; resize/move to the monitor the camera sees.
        h, w = img.shape[:2]
        scale = min(1.0, 1200 / max(w, 1))
        cv2.resizeWindow(window, int(w * scale), int(h * scale))

    print(
        f"Pattern: {cols}x{rows} inner corners "
        f"({cols + 1}x{rows + 1} squares). Config square_size_mm={square_mm}."
    )
    print("Move this window to the monitor the camera will look at. Press any key to close.")
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
