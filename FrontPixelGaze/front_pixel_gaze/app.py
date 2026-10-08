"""Front-pixel gaze: click-map calib → optional ArUco screen heatmap.

Warmup → click calibration → track / heatmap.
Heatmap uses calibrated front UV ∩ ArUco screen quad (no 3D ray / eye lock).
"""

from __future__ import annotations

import time
from enum import Enum, auto

import cv2
import numpy as np
from numpy.typing import NDArray

from front_pixel_gaze.cameras import Camera, load_camera_setup
from front_pixel_gaze.heatmap import accumulate_heatmap, overlay_heatmap
from front_pixel_gaze.intrinsics import for_frame, load_intrinsics, undistort_bgr
from front_pixel_gaze.orlosky_eye import OrloskyEyeTracker
from front_pixel_gaze.paths import (
    CAMERA_SETUP_PATH,
    FRONT_UV_MAP_PATH,
    INTRINSICS_DIR,
)
from front_pixel_gaze.screen_aruco import (
    CornerMarkers,
    ScreenDetectResult,
    detect_screen_quad,
    draw_detections,
    front_uv_to_screen,
    primary_screen_size_px,
    resolve_heatmap_canvas_size,
)
from front_pixel_gaze.uv_map import (
    FrontUvMap,
    fit_front_uv_map,
    load_front_uv_map,
    orlosky_to_opencv_direction,
    save_front_uv_map,
)

WINDOW_TRACK = "FrontPixelGaze Tracking"
WINDOW_HEAT = "FrontPixelGaze Heatmap"
MIN_SAMPLES = 5
CAPTURE_SECONDS = 0.5
PIP_WIDTH = 240
TRACK_WINDOW_WIDTH = 640


class Mode(Enum):
    WARMUP = auto()
    CALIB = auto()
    TRACK = auto()
    HEATMAP = auto()


class App:
    def __init__(self) -> None:
        setup = load_camera_setup(CAMERA_SETUP_PATH)
        self.front_cam = Camera(setup.front, name="front")
        self.left_cam = Camera(setup.left, name="left")
        self.front_intr = load_intrinsics(INTRINSICS_DIR / "front.npz")
        self.left_intr = load_intrinsics(INTRINSICS_DIR / "left_eye.npz")
        self.eye = OrloskyEyeTracker()

        self.mode = Mode.WARMUP
        self.uv_map: FrontUvMap | None = load_front_uv_map(FRONT_UV_MAP_PATH)
        self.sample_dirs: list[NDArray[np.float64]] = []
        self.sample_uvs: list[tuple[float, float]] = []

        self._pending_uv: tuple[float, float] | None = None
        self._capture_dirs: list[NDArray[np.float64]] = []
        self._capture_until: float | None = None

        self._front_display_size: tuple[int, int] = (640, 480)
        self._track_show_size: tuple[int, int] = (TRACK_WINDOW_WIDTH, 480)
        # IR PiP in tracking-window (imshow) coordinates: (x0, y0, x1, y1).
        self._pip_rect_show: tuple[int, int, int, int] | None = None
        self._pip_src_size: tuple[int, int] = (640, 480)

        self._screen_w, self._screen_h = primary_screen_size_px()
        self._markers = CornerMarkers(self._screen_w, self._screen_h)
        self._heat = np.zeros((self._screen_h, self._screen_w), dtype=np.float32)
        self._heatmap_window_open = False

        cv2.namedWindow(WINDOW_TRACK, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_TRACK, TRACK_WINDOW_WIDTH, 480)
        # Keep off screen corners so it does not cover ArUco in heatmap mode.
        cv2.moveWindow(WINDOW_TRACK, 80, 80)
        cv2.setMouseCallback(WINDOW_TRACK, self._on_mouse)

    def close(self) -> None:
        self._close_heatmap_window()
        self.front_cam.release()
        self.left_cam.release()
        cv2.destroyAllWindows()

    def _resize_heatmap_buffers(self, width: int, height: int) -> None:
        self._screen_w, self._screen_h = int(width), int(height)
        self._markers = CornerMarkers(self._screen_w, self._screen_h)
        if self._heat.shape != (self._screen_h, self._screen_w):
            self._heat = np.zeros((self._screen_h, self._screen_w), dtype=np.float32)

    def _open_heatmap_window(self) -> None:
        if self._heatmap_window_open:
            return

        # DPI-aware OS size first, then sync to the actual OpenCV drawable so
        # ArUco is not stretched (stretch breaks front-cam detection).
        guess_w, guess_h = primary_screen_size_px()
        cv2.namedWindow(WINDOW_HEAT, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_HEAT, guess_w, guess_h)
        cv2.moveWindow(WINDOW_HEAT, 0, 0)
        cv2.setWindowProperty(WINDOW_HEAT, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        probe = np.zeros((guess_h, guess_w, 3), dtype=np.uint8)
        probe[:] = (24, 24, 24)
        cv2.imshow(WINDOW_HEAT, probe)
        cv2.waitKey(1)

        screen_w, screen_h = resolve_heatmap_canvas_size(WINDOW_HEAT)
        self._resize_heatmap_buffers(screen_w, screen_h)

        # Second probe at the final size so HighGUI locks 1:1 pixel mapping.
        probe2 = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        probe2[:] = (24, 24, 24)
        self._markers.paste_on(probe2)
        cv2.imshow(WINDOW_HEAT, probe2)
        cv2.waitKey(1)

        self._heatmap_window_open = True

    def _close_heatmap_window(self) -> None:
        if not self._heatmap_window_open:
            return
        cv2.destroyWindow(WINDOW_HEAT)
        self._heatmap_window_open = False

    def _enter_heatmap(self) -> bool:
        if self.uv_map is None:
            print("Need a fitted front UV map before heatmap (press F after clicks).")
            return False
        self.mode = Mode.HEATMAP
        self._open_heatmap_window()
        print(
            f"Heatmap mode: fullscreen ArUco ({self._screen_w}x{self._screen_h}). "
            "Point front cam at this monitor. R clears heat, T leaves heatmap."
        )
        return True

    def _show_to_front(self, x: int, y: int) -> tuple[float, float]:
        """Map tracking-window imshow coords → full front-canvas pixels."""
        sw, sh = self._track_show_size
        fw, fh = self._front_display_size
        return (x * fw / max(sw, 1), y * fh / max(sh, 1))

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        # Right-click: lock eyeball center (before / during calib warmup).
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.mode not in (Mode.WARMUP, Mode.CALIB, Mode.TRACK):
                return
            locked = self._lock_eye_from_right_click(x, y)
            if locked is not None:
                print(f"Eye center locked at IR px {locked} (U to unlock)")
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self.mode != Mode.CALIB:
            return
        if self._capture_until is not None:
            return
        fx, fy = self._show_to_front(x, y)
        fw, fh = self._front_display_size
        if not (0 <= fx < fw and 0 <= fy < fh):
            return
        self._pending_uv = (float(fx), float(fy))
        self._capture_dirs = []
        self._capture_until = time.perf_counter() + CAPTURE_SECONDS
        print(f"Capturing gaze for click ({fx:.0f}, {fy:.0f}) for {CAPTURE_SECONDS:.1f}s …")

    def _lock_eye_from_right_click(self, x: int, y: int) -> tuple[int, int] | None:
        """Right-click in IR PiP → lock at that buffer pixel; else lock on pupil."""
        pip = self._pip_rect_show
        if pip is not None:
            x0, y0, x1, y1 = pip
            if x0 <= x < x1 and y0 <= y < y1:
                pw, ph = self._pip_src_size
                u = int(round((x - x0) * pw / max(x1 - x0, 1)))
                v = int(round((y - y0) * ph / max(y1 - y0, 1)))
                u = int(np.clip(u, 0, pw - 1))
                v = int(np.clip(v, 0, ph - 1))
                return self.eye.lock_eye_center_ir_px((u, v))
        # Click outside PiP: lock on current pupil (look into IR).
        return self.eye.lock_eye_center_at_pupil()

    def _finish_capture_if_due(self, direction_opencv: NDArray[np.float64] | None) -> None:
        if self._capture_until is None or self._pending_uv is None:
            return
        if direction_opencv is not None:
            self._capture_dirs.append(direction_opencv.copy())
        if time.perf_counter() < self._capture_until:
            return
        if len(self._capture_dirs) < 3:
            print("Capture failed: not enough valid gaze samples. Try again.")
        else:
            mean_d = np.mean(np.stack(self._capture_dirs, axis=0), axis=0)
            n = float(np.linalg.norm(mean_d))
            if n > 1e-12:
                mean_d = mean_d / n
            self.sample_dirs.append(mean_d)
            self.sample_uvs.append(self._pending_uv)
            print(
                f"Sample {len(self.sample_uvs)}: UV={self._pending_uv} "
                f"from {len(self._capture_dirs)} frames"
            )
        self._pending_uv = None
        self._capture_dirs = []
        self._capture_until = None

    def _try_fit(self) -> bool:
        if len(self.sample_uvs) < MIN_SAMPLES:
            print(f"Need at least {MIN_SAMPLES} samples (have {len(self.sample_uvs)}).")
            return False
        self.uv_map = fit_front_uv_map(self.sample_dirs, self.sample_uvs)
        save_front_uv_map(FRONT_UV_MAP_PATH, self.uv_map)
        print(
            f"Fitted map: n={self.uv_map.n_samples} rms={self.uv_map.rms_px:.1f}px "
            f"→ {FRONT_UV_MAP_PATH}"
        )
        return True

    def _compose_track(
        self,
        front_bgr: NDArray[np.uint8],
        ir_preview: NDArray[np.uint8] | None,
        status_lines: list[str],
        gaze_uv: tuple[float, float] | None,
        detect: ScreenDetectResult | None,
    ) -> NDArray[np.uint8]:
        canvas = front_bgr.copy()
        fh, fw = canvas.shape[:2]
        self._front_display_size = (fw, fh)

        if detect is not None:
            draw_detections(canvas, detect)
            if detect.quad is not None:
                pts = np.round(detect.quad.front_corners).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(canvas, [pts], True, (0, 200, 255), 2)

        if gaze_uv is not None:
            u, v = int(round(gaze_uv[0])), int(round(gaze_uv[1]))
            cv2.drawMarker(
                canvas,
                (u, v),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=28,
                thickness=2,
            )
            cv2.circle(canvas, (u, v), 10, (0, 255, 0), 2)

        for i, (u, v) in enumerate(self.sample_uvs):
            p = (int(round(u)), int(round(v)))
            cv2.circle(canvas, p, 6, (0, 165, 255), -1)
            cv2.putText(
                canvas,
                str(i + 1),
                (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
            )

        if self._pending_uv is not None:
            p = (int(round(self._pending_uv[0])), int(round(self._pending_uv[1])))
            cv2.circle(canvas, p, 12, (0, 255, 255), 2)

        y0 = 24
        for line in status_lines:
            cv2.putText(canvas, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(canvas, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y0 += 22

        self._pip_rect_canvas: tuple[int, int, int, int] | None = None
        if ir_preview is not None and ir_preview.size:
            ph, pw = ir_preview.shape[:2]
            self._pip_src_size = (pw, ph)
            scale = PIP_WIDTH / float(pw)
            pip = cv2.resize(ir_preview, (PIP_WIDTH, max(1, int(round(ph * scale)))))
            ph2, pw2 = pip.shape[:2]
            x0, y0p = 10, fh - ph2 - 10
            if y0p >= 0 and x0 + pw2 <= fw:
                canvas[y0p : y0p + ph2, x0 : x0 + pw2] = pip
                cv2.rectangle(canvas, (x0, y0p), (x0 + pw2, y0p + ph2), (200, 200, 200), 1)
                self._pip_rect_canvas = (x0, y0p, x0 + pw2, y0p + ph2)
                lock_hint = "LOCKED" if self.eye.is_eye_center_locked() else "R-click=lock"
                cv2.putText(
                    canvas,
                    lock_hint,
                    (x0 + 4, y0p + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255) if self.eye.is_eye_center_locked() else (200, 200, 200),
                    1,
                )

        return canvas

    def _compose_heatmap(
        self,
        screen_uv: tuple[float, float] | None,
        aruco_ok: bool,
        markers_found: int,
    ) -> NDArray[np.uint8]:
        canvas = np.zeros((self._screen_h, self._screen_w, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 24)
        canvas = overlay_heatmap(canvas, self._heat)
        self._markers.paste_on(canvas)

        # Keep HUD clear of corner ArUco (same idea as MultiCamGaze clearance).
        clear = self._markers.margin + self._markers.marker_size + 16
        status_x = clear
        status_y = clear + 28

        status = (
            f"HEATMAP  ArUco={'OK' if aruco_ok else 'LOST'} ({markers_found}/4)  "
            f"R=clear  T=track  Q=quit"
        )
        cv2.putText(canvas, status, (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(canvas, status, (status_x, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if screen_uv is not None:
            u, v = int(round(screen_uv[0])), int(round(screen_uv[1]))
            cv2.drawMarker(
                canvas,
                (u, v),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=40,
                thickness=2,
            )
            cv2.circle(canvas, (u, v), 14, (0, 255, 0), 2)
        elif not aruco_ok:
            cv2.putText(
                canvas,
                "Point front camera at this screen (need 4 corner markers)",
                (status_x, status_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 255),
                2,
            )
        return canvas

    def run(self) -> None:
        print("FrontPixelGaze")
        print("  Keep head steady during click calibration.")
        print("  Keys: F=fit  H=heatmap  T=track  C=clear samples  R=clear heat  W=calib  U=unlock  Q=quit")
        print("  Before calib: right-click IR PiP (or pupil) to lock eyeball center.")
        print("  Calib: look at a point, left-click it on the tracking window.")
        if self.uv_map is not None:
            print(f"  Loaded map from {FRONT_UV_MAP_PATH} (rms={self.uv_map.rms_px:.1f}px)")
            self.mode = Mode.TRACK

        while True:
            front_raw = self.front_cam.read()
            left_raw = self.left_cam.read(apply_orientation=False)
            if front_raw is None or left_raw is None:
                print("Camera read failed.")
                break

            front = undistort_bgr(front_raw, self.front_intr)
            front_live = for_frame(
                self.front_intr, (front_raw.shape[1], front_raw.shape[0])
            )
            left_live = for_frame(
                self.left_intr, (left_raw.shape[1], left_raw.shape[0])
            )

            sample = self.eye.process(
                left_raw,
                flip_vertical=self.left_cam.cfg.flip,
                flip_horizontal=self.left_cam.cfg.mirror,
                camera_matrix=left_live.camera_matrix,
                dist_coeffs=left_live.dist_coeffs,
            )
            direction_opencv = None
            if sample.direction_opengl is not None:
                direction_opencv = orlosky_to_opencv_direction(
                    sample.direction_opengl,
                    vertical_flip_undone=sample.vertical_flip_undone,
                )

            readiness = self.eye.get_readiness()
            if self.mode == Mode.WARMUP and readiness.ready:
                self.mode = Mode.CALIB
                print("Warmup complete → calibration mode (click look-at points).")

            if self.mode == Mode.CALIB:
                self._finish_capture_if_due(direction_opencv)

            gaze_uv: tuple[float, float] | None = None
            if (
                self.mode in (Mode.TRACK, Mode.HEATMAP)
                and self.uv_map is not None
                and direction_opencv is not None
            ):
                h, w = front.shape[:2]
                gaze_uv = self.uv_map.apply(direction_opencv, image_size=(w, h))

            detect: ScreenDetectResult | None = None
            screen_uv = None
            if self.mode == Mode.HEATMAP:
                # Detect on RAW front (like MultiCamGaze); remap corners into
                # undistorted space so they match click-calib gaze UV.
                detect = detect_screen_quad(
                    front_raw,
                    self._markers,
                    camera_matrix=front_live.camera_matrix,
                    dist_coeffs=front_live.dist_coeffs,
                )
                if detect.quad is not None and gaze_uv is not None:
                    screen_uv = front_uv_to_screen(gaze_uv, detect.quad)
                    if screen_uv is not None:
                        su = float(np.clip(screen_uv[0], 0, self._screen_w - 1))
                        sv = float(np.clip(screen_uv[1], 0, self._screen_h - 1))
                        screen_uv = (su, sv)
                        accumulate_heatmap(self._heat, su, sv)

            aruco_n = 0 if detect is None else detect.markers_found
            status = [
                f"mode={self.mode.name}  samples={len(self.sample_uvs)}  "
                f"centers={readiness.n_model_centers}/{readiness.min_model_centers}  "
                f"radius={readiness.radius_px:.0f}px  ready={readiness.ready}",
            ]
            if self._capture_until is not None:
                status.append("CAPTURING gaze… hold still")
            if self.uv_map is not None:
                status.append(f"map rms={self.uv_map.rms_px:.1f}px n={self.uv_map.n_samples}")
            if self.mode == Mode.CALIB:
                status.append("Click look-at on this window | F fit | H heatmap")
            elif self.mode == Mode.WARMUP:
                status.append("Look around to adapt eye sphere (warmup)")
            elif self.mode == Mode.TRACK:
                status.append("Tracking | H heatmap | W calib | Q quit")
            else:
                ids = "-" if detect is None or not detect.found_ids else str(detect.found_ids)
                on_screen = "ON screen" if screen_uv is not None else "off screen"
                status.append(
                    f"ArUco {aruco_n}/4 ids={ids} | gaze {on_screen} | T leave heatmap"
                )

            if self.eye.is_eye_center_locked():
                status.append("eye center LOCKED (U unlock)")
            elif self.mode in (Mode.WARMUP, Mode.CALIB):
                status.append("R-click IR PiP (or pupil) to lock eye center")

            track = self._compose_track(
                front,
                sample.preview_bgr,
                status,
                gaze_uv,
                detect,
            )
            # Keep tracking window compact; store show size + PiP for mouse map.
            th, tw = track.shape[:2]
            scale = TRACK_WINDOW_WIDTH / float(tw)
            show_h = max(1, int(round(th * scale)))
            track_show = cv2.resize(track, (TRACK_WINDOW_WIDTH, show_h))
            self._track_show_size = (TRACK_WINDOW_WIDTH, show_h)
            pip_c = getattr(self, "_pip_rect_canvas", None)
            if pip_c is not None:
                x0, y0, x1, y1 = pip_c
                self._pip_rect_show = (
                    int(round(x0 * scale)),
                    int(round(y0 * scale)),
                    int(round(x1 * scale)),
                    int(round(y1 * scale)),
                )
            else:
                self._pip_rect_show = None
            cv2.imshow(WINDOW_TRACK, track_show)

            if self.mode == Mode.HEATMAP and self._heatmap_window_open:
                heat_canvas = self._compose_heatmap(
                    screen_uv,
                    aruco_ok=detect is not None and detect.quad is not None,
                    markers_found=aruco_n,
                )
                cv2.imshow(WINDOW_HEAT, heat_canvas)
            elif self._heatmap_window_open:
                self._close_heatmap_window()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C")) and self.mode != Mode.HEATMAP:
                self.sample_dirs.clear()
                self.sample_uvs.clear()
                print("Cleared samples.")
            if key in (ord("r"), ord("R")):
                self._heat[:] = 0.0
                print("Cleared heatmap.")
            if key in (ord("f"), ord("F")):
                if self._try_fit():
                    self.mode = Mode.TRACK
                    print("Press H for fullscreen ArUco heatmap.")
            if key in (ord("h"), ord("H")):
                self._enter_heatmap()
            if key in (ord("t"), ord("T")):
                if self.uv_map is None:
                    loaded = load_front_uv_map(FRONT_UV_MAP_PATH)
                    if loaded is None:
                        print("No map fitted or saved yet.")
                    else:
                        self.uv_map = loaded
                        self.mode = Mode.TRACK
                        self._close_heatmap_window()
                        print("Loaded saved map → track.")
                else:
                    self.mode = Mode.TRACK
                    self._close_heatmap_window()
            if key in (ord("w"), ord("W")):
                self._close_heatmap_window()
                self.mode = Mode.CALIB if readiness.ready else Mode.WARMUP
            if key in (ord("u"), ord("U")):
                if self.eye.unlock_eye_center():
                    print("Eye center unlocked (auto-estimate resumed).")
                else:
                    print("Eye center was not locked.")

        self.close()


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
