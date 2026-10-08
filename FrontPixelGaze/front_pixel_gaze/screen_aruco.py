"""Corner ArUco on the fullscreen monitor + front→screen homography (2D)."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def _make_detector() -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    # Slightly more permissive — glasses front cams often see small / angled markers.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.01
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(_ARUCO_DICT, params)


_DETECTOR = _make_detector()

MARKER_SIZE_MIN = 120
MARKER_SIZE_MAX = 320
# Larger than MultiCamGaze (÷8): front cam is often ~640px wide; small markers vanish.
MARKER_SIZE_DIVISOR = 4


def primary_screen_size_px() -> tuple[int, int]:
    """Primary monitor size in pixels (DPI-aware when possible)."""
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
        w = int(user32.GetSystemMetrics(0))
        h = int(user32.GetSystemMetrics(1))
        if w >= 100 and h >= 100:
            return w, h
    except Exception:
        pass
    return 1920, 1080


def resolve_heatmap_canvas_size(window_name: str) -> tuple[int, int]:
    """Canvas size matching the OpenCV drawable (avoids fullscreen stretch)."""
    native_w, native_h = primary_screen_size_px()
    try:
        rect = cv2.getWindowImageRect(window_name)
        rect_w, rect_h = int(rect[2] or 0), int(rect[3] or 0)
    except Exception:
        rect_w, rect_h = 0, 0

    if rect_w >= 100 and rect_h >= 100:
        if (rect_w, rect_h) != (native_w, native_h):
            print(
                f"  Heatmap canvas: using OpenCV drawable {rect_w}x{rect_h} "
                f"(OS primary reported {native_w}x{native_h}) — avoids stretch."
            )
        return rect_w, rect_h
    return native_w, native_h


@dataclass(frozen=True)
class ScreenQuad:
    """Four front-image corners of the monitor (TL, TR, BR, BL) + H front→screen."""

    front_corners: NDArray[np.float64]  # (4, 2) in the same space as gaze UV
    homography: NDArray[np.float64]  # 3x3


@dataclass
class ScreenDetectResult:
    """Always reports how many corner markers were seen (even if < 4)."""

    markers_found: int
    found_ids: list[int] = field(default_factory=list)
    quad: ScreenQuad | None = None
    # Per detected id: 4 image corners (for debug overlay), in gaze/undistorted space.
    debug_corners: dict[int, NDArray[np.float64]] = field(default_factory=dict)


class CornerMarkers:
    """ArUco IDs 0–3 in the four corners of the fullscreen canvas."""

    def __init__(self, screen_width: int, screen_height: int, margin: int = 12) -> None:
        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.margin = margin
        short_edge = min(self.screen_width, self.screen_height)
        self.marker_size = min(
            MARKER_SIZE_MAX,
            max(MARKER_SIZE_MIN, short_edge // MARKER_SIZE_DIVISOR),
        )
        self.border = max(10, self.marker_size // 12)
        self._markers_bgr = self._build_markers()
        self._corner_positions = self._build_corner_positions()

    def _build_markers(self) -> dict[int, NDArray[np.uint8]]:
        inner = self.marker_size - 2 * self.border
        markers: dict[int, NDArray[np.uint8]] = {}
        for marker_id in (0, 1, 2, 3):
            raw = cv2.aruco.generateImageMarker(_ARUCO_DICT, marker_id, inner)
            bordered = cv2.copyMakeBorder(
                raw,
                self.border,
                self.border,
                self.border,
                self.border,
                cv2.BORDER_CONSTANT,
                value=255,
            )
            markers[marker_id] = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)
        return markers

    def _build_corner_positions(self) -> dict[int, tuple[int, int]]:
        s = self.marker_size
        m = self.margin
        w, h = self.screen_width, self.screen_height
        return {
            0: (m, m),
            1: (w - m - s, m),
            2: (w - m - s, h - m - s),
            3: (m, h - m - s),
        }

    def paste_on(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        for marker_id, (x, y) in self._corner_positions.items():
            marker = self._markers_bgr[marker_id]
            mh, mw = marker.shape[:2]
            frame[y : y + mh, x : x + mw] = marker
        return frame

    def screen_dst_corners(self) -> NDArray[np.float64]:
        w, h = self.screen_width, self.screen_height
        return np.array(
            [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
            dtype=np.float64,
        )


def _undistort_points(
    pts: NDArray[np.float64],
    camera_matrix: NDArray[np.float64] | None,
    dist_coeffs: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Map raw-image points into the same space as cv2.undistort(frame)."""
    if camera_matrix is None or dist_coeffs is None:
        return pts
    reshaped = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(
        reshaped,
        camera_matrix,
        dist_coeffs,
        P=camera_matrix,
    )
    return out.reshape(-1, 2)


def detect_screen_quad(
    front_bgr_raw: NDArray[np.uint8],
    markers: CornerMarkers,
    *,
    camera_matrix: NDArray[np.float64] | None = None,
    dist_coeffs: NDArray[np.float64] | None = None,
) -> ScreenDetectResult:
    """Detect corner markers in the *raw* front frame.

    Corners are undistorted into the same pixel space as ``cv2.undistort`` /
    click-calib gaze UV when K/dist are provided (MultiCamGaze detects on raw;
    we only remap points so they match the undistorted tracking view).
    """
    gray = cv2.cvtColor(front_bgr_raw, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return ScreenDetectResult(markers_found=0)

    picked: dict[int, NDArray[np.float64]] = {}
    debug: dict[int, NDArray[np.float64]] = {}
    for i, mid in enumerate(ids.ravel()):
        mid_i = int(mid)
        if mid_i not in (0, 1, 2, 3):
            continue
        c_raw = corners[i].reshape(4, 2).astype(np.float64)
        c = _undistort_points(c_raw, camera_matrix, dist_coeffs)
        debug[mid_i] = c
        # Outer corner toward that screen corner (OpenCV order TL,TR,BR,BL).
        picked[mid_i] = c[mid_i]

    found_ids = sorted(picked.keys())
    result = ScreenDetectResult(
        markers_found=len(picked),
        found_ids=found_ids,
        debug_corners=debug,
    )
    if len(picked) < 4:
        return result

    src = np.array([picked[0], picked[1], picked[2], picked[3]], dtype=np.float64)
    dst = markers.screen_dst_corners()
    h_mat, _ = cv2.findHomography(src, dst, method=0)
    if h_mat is None:
        return result

    result.quad = ScreenQuad(
        front_corners=src,
        homography=np.asarray(h_mat, dtype=np.float64),
    )
    return result


def front_uv_to_screen(
    front_uv: tuple[float, float],
    quad: ScreenQuad,
) -> tuple[float, float] | None:
    """Map a front-camera pixel into screen pixels if inside the screen quad."""
    u, v = float(front_uv[0]), float(front_uv[1])
    poly = quad.front_corners.astype(np.float32).reshape(-1, 1, 2)
    if cv2.pointPolygonTest(poly, (u, v), False) < 0:
        return None
    pt = np.array([[[u, v]]], dtype=np.float64)
    out = cv2.perspectiveTransform(pt, quad.homography)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def draw_detections(
    canvas: NDArray[np.uint8],
    result: ScreenDetectResult,
) -> None:
    """Debug overlay: outline any detected corner markers on the tracking view."""
    for mid, corners in result.debug_corners.items():
        pts = np.round(corners).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
        cxy = tuple(np.round(corners.mean(axis=0)).astype(int))
        cv2.putText(
            canvas,
            str(mid),
            cxy,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
