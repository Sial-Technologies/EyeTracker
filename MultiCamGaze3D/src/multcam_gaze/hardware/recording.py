"""Parallel preview recording to MP4."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2

from multcam_gaze.types import PREVIEW_ROLES


class RecordingManager:
    def __init__(self, output_dir: Path, clip_duration: float = 15.0) -> None:
        self.output_dir = output_dir
        self.clip_duration = clip_duration
        self.is_recording_flag = False
        self.recording_start_time: float | None = None
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.writer_sizes: dict[str, tuple[int, int]] = {}
        self.clip_filenames: dict[str, str] = {}
        self.fps = 30
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def start_recording(self, stream_sizes: dict[str, tuple[int, int]], fps: int = 30) -> bool:
        if self.is_recording_flag or not stream_sizes:
            return False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_writers = {}
        self.writer_sizes = {}
        self.clip_filenames = {}
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.fps = fps
        for role in PREVIEW_ROLES:
            if role not in stream_sizes:
                continue
            width, height = stream_sizes[role]
            if width < 1 or height < 1:
                continue
            filename = str(self.output_dir / f"{role}_{timestamp}.mp4")
            writer = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
            if not writer.isOpened():
                for w in self.video_writers.values():
                    w.release()
                self.video_writers = {}
                return False
            self.video_writers[role] = writer
            self.writer_sizes[role] = (int(width), int(height))
            self.clip_filenames[role] = filename
        if not self.video_writers:
            return False
        self.recording_start_time = time.perf_counter()
        self.is_recording_flag = True
        return True

    def add_frames(self, frames_by_role: dict[str, object]) -> bool:
        if not self.is_recording_flag or not self.video_writers:
            return False
        assert self.recording_start_time is not None
        elapsed = time.perf_counter() - self.recording_start_time
        if elapsed >= self.clip_duration:
            self._finalize_recording()
            return True
        for role, writer in self.video_writers.items():
            frame = frames_by_role.get(role)
            if frame is None:
                continue
            target_w, target_h = self.writer_sizes[role]
            h, w = frame.shape[:2]
            if (w, h) != (target_w, target_h):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
        return False

    def _finalize_recording(self) -> None:
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers = {}
        self.writer_sizes = {}
        self.clip_filenames = {}
        self.is_recording_flag = False
        self.recording_start_time = None

    def stop_recording(self) -> None:
        if self.is_recording_flag:
            self._finalize_recording()

    def is_recording(self) -> bool:
        return self.is_recording_flag

    def get_remaining_time(self) -> float:
        if not self.is_recording_flag or self.recording_start_time is None:
            return 0.0
        elapsed = time.perf_counter() - self.recording_start_time
        return max(0.0, self.clip_duration - elapsed)

    def get_status_line(self) -> str:
        if self.is_recording_flag:
            remaining = self.get_remaining_time()
            n = len(self.video_writers) or len(PREVIEW_ROLES)
            return f"REC {n} previews - {remaining:.1f}s left"
        return f"Record previews {self.clip_duration:.0f}s (Space)"
