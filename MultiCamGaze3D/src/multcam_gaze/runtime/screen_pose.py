"""Live screen pose from on-screen corner ArUco (world ≡ front camera)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from multcam_gaze.core.transform import Transform
from multcam_gaze.runtime.screen_model import ScreenModel
from multcam_gaze.types import IntrinsicsModel

_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, cv2.aruco.DetectorParameters())

MARKER_SIZE_MIN = 80
MARKER_SIZE_MAX = 220
MARKER_SIZE_DIVISOR = 8
MAX_REPROJ_ERROR_PX = 12.0


def primary_screen_size_px() -> tuple[int, int] | None:
    """Primary monitor pixel size (Windows). OpenCV window rect is often wrong."""
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        w = int(user32.GetSystemMetrics(0))  # SM_CXSCREEN
        h = int(user32.GetSystemMetrics(1))  # SM_CYSCREEN
        if w >= 100 and h >= 100:
            return w, h
    except Exception:
        pass
    return None


def estimate_screen_mm(screen_w_px: int, screen_h_px: int) -> tuple[float, float]:
    """Physical screen size in mm (Windows GDI when available, else 96 DPI fallback)."""
    width_mm = height_mm = None
    try:
        import ctypes

        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        gdi32 = ctypes.windll.gdi32  # type: ignore[attr-defined]
        hdc = user32.GetDC(0)
        if hdc:
            width_mm = float(gdi32.GetDeviceCaps(hdc, 4))
            height_mm = float(gdi32.GetDeviceCaps(hdc, 6))
            horz_res = float(gdi32.GetDeviceCaps(hdc, 8))
            vert_res = float(gdi32.GetDeviceCaps(hdc, 10))
            user32.ReleaseDC(0, hdc)
            if width_mm and height_mm and horz_res > 0 and vert_res > 0:
                scale_x = screen_w_px / horz_res
                scale_y = screen_h_px / vert_res
                return width_mm * scale_x, height_mm * scale_y
    except Exception:
        pass
    return screen_w_px * 25.4 / 96.0, screen_h_px * 25.4 / 96.0


def pixel_to_screen_mm(
    u: float,
    v: float,
    width_px: int,
    height_px: int,
    width_mm: float,
    height_mm: float,
) -> NDArray[np.float64]:
    """Monitor pixel → point on screen plane (X right, Y up, Z=0), origin at center."""
    x_mm = (float(u) / float(width_px) - 0.5) * float(width_mm)
    y_mm = (0.5 - float(v) / float(height_px)) * float(height_mm)
    return np.array([x_mm, y_mm, 0.0], dtype=np.float64)


def screen_point_to_front(
    p_screen: NDArray[np.float64],
    front_from_screen: Transform,
) -> NDArray[np.float64]:
    return front_from_screen.apply_point(p_screen)


def pixel_to_front_mm(
    u: float,
    v: float,
    front_from_screen: Transform,
    width_px: int,
    height_px: int,
    width_mm: float,
    height_mm: float,
) -> NDArray[np.float64]:
    p_screen = pixel_to_screen_mm(u, v, width_px, height_px, width_mm, height_mm)
    return screen_point_to_front(p_screen, front_from_screen)


@dataclass(frozen=True)
class ScreenPoseEstimate:
    front_from_screen: Transform
    reprojection_error_px: float
    markers_found: int


class CornerMarkers:
    """On-screen ArUco IDs 0–3; PnP uses inner black-square corners in mm."""

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
        self.inner_corners_px = self._build_inner_corners()

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

    def _build_inner_corners(self) -> dict[int, NDArray[np.float64]]:
        """OpenCV ArUco corner order for the inner pattern (excluding white border)."""
        b = self.border
        s = self.marker_size
        inner: dict[int, NDArray[np.float64]] = {}
        for marker_id, (x, y) in self._corner_positions.items():
            inner[marker_id] = np.array(
                [
                    [x + b, y + b],
                    [x + s - b - 1, y + b],
                    [x + s - b - 1, y + s - b - 1],
                    [x + b, y + s - b - 1],
                ],
                dtype=np.float64,
            )
        return inner

    def object_points_mm(self, width_mm: float, height_mm: float) -> dict[int, NDArray[np.float64]]:
        points: dict[int, NDArray[np.float64]] = {}
        for marker_id, corners_px in self.inner_corners_px.items():
            points[marker_id] = np.array(
                [
                    pixel_to_screen_mm(
                        px,
                        py,
                        self.screen_width,
                        self.screen_height,
                        width_mm,
                        height_mm,
                    )
                    for px, py in corners_px
                ],
                dtype=np.float64,
            )
        return points

    def paste_on(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        for marker_id, (x, y) in self._corner_positions.items():
            marker = self._markers_bgr[marker_id]
            mh, mw = marker.shape[:2]
            frame[y : y + mh, x : x + mw] = marker
        return frame


class LiveScreenPose:
    """Detect corner markers in the front camera and estimate T_front←screen."""

    def __init__(
        self,
        screen_width_px: int,
        screen_height_px: int,
        width_mm: float,
        height_mm: float,
        intrinsics: IntrinsicsModel,
    ) -> None:
        self.screen_width_px = int(screen_width_px)
        self.screen_height_px = int(screen_height_px)
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.intrinsics = intrinsics
        self.markers = CornerMarkers(screen_width_px, screen_height_px)
        self._object_points = self.markers.object_points_mm(width_mm, height_mm)
        self._prev_rvec: NDArray[np.float64] | None = None
        self._prev_tvec: NDArray[np.float64] | None = None
        self.last: ScreenPoseEstimate | None = None

    def estimate(self, front_bgr: NDArray[np.uint8]) -> ScreenPoseEstimate | None:
        gray = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = _DETECTOR.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            self.last = None
            return None

        obj_list: list[NDArray[np.float64]] = []
        img_list: list[NDArray[np.float64]] = []
        found = 0
        for i, mid in enumerate(ids.ravel()):
            mid_i = int(mid)
            if mid_i not in self._object_points:
                continue
            obj_list.append(self._object_points[mid_i])
            img_list.append(corners[i].reshape(-1, 2).astype(np.float64))
            found += 1
        if found < 2:
            self.last = None
            return None

        object_points = np.vstack(obj_list)
        image_points = np.vstack(img_list)
        k = self.intrinsics.camera_matrix
        dist = self.intrinsics.dist_coeffs
        use_guess = self._prev_rvec is not None and self._prev_tvec is not None
        ok, rvec, tvec = cv2.solvePnP(
            object_points.astype(np.float32),
            image_points.astype(np.float32),
            k,
            dist,
            None if not use_guess else self._prev_rvec,
            None if not use_guess else self._prev_tvec,
            useExtrinsicGuess=use_guess,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            self.last = None
            return None

        projected, _ = cv2.projectPoints(
            object_points.astype(np.float32), rvec, tvec, k, dist
        )
        err = projected.reshape(-1, 2) - image_points
        rms = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
        if rms > MAX_REPROJ_ERROR_PX:
            self.last = None
            return None

        r, _ = cv2.Rodrigues(rvec)
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        # solvePnP: P_front = R @ P_screen + t
        front_from_screen = Transform.from_rotation_translation(
            r, t, "front", "screen"
        )
        self._prev_rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        self._prev_tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        est = ScreenPoseEstimate(front_from_screen, rms, found)
        self.last = est
        return est

    def screen_model(self, estimate: ScreenPoseEstimate | None = None) -> ScreenModel | None:
        est = estimate if estimate is not None else self.last
        if est is None:
            return None
        # ScreenModel uses world_from_screen; world ≡ front for this MVP.
        return ScreenModel(self.width_mm, self.height_mm, est.front_from_screen)

    def target_front(
        self,
        u: float,
        v: float,
        estimate: ScreenPoseEstimate | None = None,
    ) -> NDArray[np.float64] | None:
        est = estimate if estimate is not None else self.last
        if est is None:
            return None
        return pixel_to_front_mm(
            u,
            v,
            est.front_from_screen,
            self.screen_width_px,
            self.screen_height_px,
            self.width_mm,
            self.height_mm,
        )
