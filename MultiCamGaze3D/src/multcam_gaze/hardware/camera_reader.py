"""Threaded USB camera capture (from GazeScreen3D camera_io)."""

from __future__ import annotations

import sys
import threading
import time

import cv2

from multcam_gaze.hardware import win_cameras

if sys.platform != "win32":
    win_cameras = None  # type: ignore[assignment]

CAPTURE_TARGET_FPS = 30
RECONNECT_FAIL_THRESHOLD = 20
RECONNECT_COOLDOWN_SEC = 1.5

CAMERA_CAPTURE_MODES = (
    (("MSMF", cv2.CAP_MSMF, None),)
    if sys.platform == "win32"
    else (("Auto", cv2.CAP_ANY, None),)
)


def configure_capture(cap, width=None, height=None, fps_request=30):
    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps_request:
        cap.set(cv2.CAP_PROP_FPS, fps_request)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def open_camera_capture(
    index,
    width=None,
    height=None,
    fps_request=CAPTURE_TARGET_FPS,
    preferred_backend=None,
):
    modes = CAMERA_CAPTURE_MODES
    if preferred_backend is not None:
        modes = tuple(m for m in modes if m[0] == preferred_backend) + tuple(
            m for m in modes if m[0] != preferred_backend
        )
    for mode_name, backend, fourcc in modes:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        if fourcc is not None:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        configure_capture(cap, width, height, fps_request)
        ret, frame = cap.read() if cap.isOpened() else (False, None)
        if ret and frame is not None:
            return cap, mode_name
        cap.release()
    return None, None


class CameraReader:
    def __init__(
        self,
        index,
        width=640,
        height=480,
        fps_request=CAPTURE_TARGET_FPS,
        device_id=None,
    ):
        self.index = index
        self.capture_index = index
        self.device_id = device_id
        self._device_absent = False
        self.width = width
        self.height = height
        self.fps_request = fps_request
        self.backend_name = None
        self.cap = None
        self.fps = 0.0
        self.status = "OFFLINE"
        self._lock = threading.Lock()
        self._latest_frame = None
        self._has_frame = False
        self._stop = threading.Event()
        self._thread = None
        self._fail_count = 0
        self._open_capture()

    def _resolve_capture_index(self):
        if not self.device_id or win_cameras is None:
            return self.capture_index
        devices = win_cameras.list_capture_devices()
        if devices is None:
            return self.capture_index
        return win_cameras.find_index_for_device_id(devices, self.device_id)

    def _open_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        resolved = self._resolve_capture_index()
        if resolved is None:
            if not self._device_absent:
                print(f"Camera {self.device_id}: device not present, staying OFFLINE.")
            self._device_absent = True
            self.status = "OFFLINE"
            return
        if self._device_absent:
            print(f"Camera {self.device_id}: device present again at index {resolved}.")
        self._device_absent = False
        if resolved != self.capture_index:
            print(f"Camera {self.device_id}: index moved {self.capture_index} -> {resolved}.")
            self.capture_index = resolved
        self.cap, self.backend_name = open_camera_capture(
            self.capture_index,
            self.width,
            self.height,
            self.fps_request,
            preferred_backend=self.backend_name,
        )
        self.status = "OK" if self.cap is not None else "OFFLINE"
        self._fail_count = 0

    def is_opened(self):
        return self.cap is not None and self.cap.isOpened()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return self.is_opened()
        self._stop.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def _clear_frame(self):
        with self._lock:
            self._latest_frame = None
            self._has_frame = False

    def _reconnect(self):
        self.status = "OFFLINE" if self._device_absent else "RECONNECTING"
        self._clear_frame()
        if not self._device_absent:
            print(f"Camera {self.capture_index}: reconnecting ({self.backend_name or 'auto'})...")
        # Cooldown also rate-limits SetupAPI polling while the device is unplugged.
        time.sleep(RECONNECT_COOLDOWN_SEC)
        if self._stop.is_set():
            return
        self._open_capture()
        if self.is_opened():
            print(f"Camera {self.capture_index}: reconnected with {self.backend_name}.")
        elif not self._device_absent:
            print(f"Camera {self.capture_index}: reconnect failed.")

    def _capture_loop(self):
        frame_count = 0
        window_start = time.perf_counter()
        frame_interval = 1.0 / max(1, int(self.fps_request or CAPTURE_TARGET_FPS))
        while not self._stop.is_set():
            loop_start = time.perf_counter()
            if not self.is_opened():
                self._reconnect()
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._fail_count += 1
                # Stop serving stale frames after a short run of failures.
                if self._fail_count > 5:
                    self._clear_frame()
                if self._fail_count >= RECONNECT_FAIL_THRESHOLD:
                    self._reconnect()
                time.sleep(0.05)
                continue
            self._fail_count = 0
            if self.status != "OK":
                self.status = "OK"
            frame_count += 1
            now = time.perf_counter()
            elapsed = now - window_start
            if elapsed >= 1.0:
                self.fps = frame_count / elapsed
                frame_count = 0
                window_start = now
            owned = frame.copy()
            with self._lock:
                self._latest_frame = owned
                self._has_frame = True
            sleep_time = frame_interval - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self):
        with self._lock:
            if self.status != "OK" or not self._has_frame or self._latest_frame is None:
                return False, None
            return True, self._latest_frame.copy()

    def snapshot_status(self) -> dict[str, object]:
        """fps, status, backend for the HUD capture line."""
        return {"fps": self.fps, "status": self.status, "backend": self.backend_name or "?"}

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status = "OFFLINE"
        self._clear_frame()
