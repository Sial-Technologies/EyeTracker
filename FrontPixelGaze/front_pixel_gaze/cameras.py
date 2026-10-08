"""Open front / left cameras from config/camera_setup.json."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from front_pixel_gaze import win_cameras


@dataclass
class CameraConfig:
    index: int
    flip: bool = False
    mirror: bool = False
    device_id: str | None = None


@dataclass
class CameraSetup:
    left: CameraConfig
    front: CameraConfig


def load_camera_setup(path: Path) -> CameraSetup:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return CameraSetup(
        left=_parse_cam(raw["left"]),
        front=_parse_cam(raw["front"]),
    )


def _parse_cam(block: dict[str, Any]) -> CameraConfig:
    device_id = block.get("device_id")
    return CameraConfig(
        index=int(block["index"]),
        flip=bool(block.get("flip", False)),
        mirror=bool(block.get("mirror", False)),
        device_id=str(device_id) if device_id else None,
    )


def resolve_capture_index(cfg: CameraConfig) -> int | None:
    """Prefer stable device_id → live MSMF index; else config index."""
    if cfg.device_id:
        devices = win_cameras.list_capture_devices()
        if devices is not None:
            found = win_cameras.find_index_for_device_id(devices, cfg.device_id)
            if found is not None:
                return int(found)
            return None
    return int(cfg.index)


def _backend_attempts() -> list[tuple[str, int]]:
    if sys.platform == "win32":
        return [
            ("MSMF", cv2.CAP_MSMF),
            ("DSHOW", cv2.CAP_DSHOW),
            ("ANY", cv2.CAP_ANY),
        ]
    return [("ANY", cv2.CAP_ANY)]


def open_capture(index: int) -> tuple[cv2.VideoCapture | None, str | None]:
    for name, backend in _backend_attempts():
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap, name
        cap.release()
    return None, None


def format_device_list() -> str:
    devices = win_cameras.list_capture_devices()
    if not devices:
        return "(no capture devices enumerated)"
    lines = [f"  [{d['index']}] {d['name']}: {d['device_id']}" for d in devices]
    return "\n".join(lines)


class Camera:
    def __init__(self, cfg: CameraConfig, *, name: str = "cam") -> None:
        self.cfg = cfg
        self.name = name
        resolved = resolve_capture_index(cfg)
        if resolved is None:
            raise RuntimeError(
                f"Could not find {name} camera device_id={cfg.device_id!r}.\n"
                f"Present devices:\n{format_device_list()}\n"
                f"Update config/camera_setup.json index/device_id."
            )
        if cfg.device_id and resolved != cfg.index:
            print(f"{name}: device_id resolved index {cfg.index} -> {resolved}")
        self.capture_index = resolved
        self._cap, backend = open_capture(resolved)
        if self._cap is None:
            raise RuntimeError(
                f"Could not open {name} camera index {resolved}"
                f" (config index={cfg.index}, device_id={cfg.device_id!r}).\n"
                f"Present devices:\n{format_device_list()}\n"
                f"Update config/camera_setup.json. If another app holds the camera, close it."
            )
        self.backend_name = backend
        print(f"{name}: opened index={resolved} via {backend}")

    def read(self, *, apply_orientation: bool = True) -> NDArray[np.uint8] | None:
        """Read one frame.

        For IR eyes, pass ``apply_orientation=False`` and let MultiCamGaze's
        ``process_frame(..., flip_vertical=..., flip_horizontal=...)`` flip —
        same contract as MultiCamGaze / GazeScreen3D (do not pre-flip).
        """
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        if apply_orientation:
            if self.cfg.flip:
                frame = cv2.flip(frame, 0)
            if self.cfg.mirror:
                frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        self._cap.release()
