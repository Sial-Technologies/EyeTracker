"""Multi-camera preview with zoom/pan and recording."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from multcam_gaze.apps._cli_common import add_standard_paths, resolve_paths
from multcam_gaze.core.logging_config import configure_logging
from multcam_gaze.hardware.camera_reader import CameraReader
from multcam_gaze.hardware.preview_view import (
    apply_camera_display,
    compose_eye_preview_panel,
    eye_center_ir_px_from_setup,
    flip_mirror_from_setup,
    load_camera_setup,
    make_status_panel,
    merge_setup_display,
    prepare_eye_tracking_frame,
    preview_views_from_setup,
    role_camera_index,
    save_camera_setup,
    set_eye_center_ir_px,
    zoom_affects_tracking_from_setup,
)
from multcam_gaze.hardware.recording import RecordingManager
from multcam_gaze.paths import project_root
from multcam_gaze.tracking.eye_camera import resolve_eye_unproject_model
from multcam_gaze.tracking.eye_tracker_adapter import EyeTrackerAdapter
from multcam_gaze.types import PREVIEW_ROLES

EYE_ROLES = ("left", "right")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview and record L/R/Front cameras.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    add_standard_paths(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    config_dir, calib_dir = resolve_paths(args)
    if args.dry_run:
        print("Would open preview for roles:", PREVIEW_ROLES)
        return 0

    setup_path = config_dir / "camera_setup.json"
    setup = load_camera_setup(setup_path)
    views = preview_views_from_setup(setup)
    flips = flip_mirror_from_setup(setup)
    zoom_tracking = zoom_affects_tracking_from_setup(setup)
    readers: dict[str, CameraReader] = {}
    for role in PREVIEW_ROLES:
        idx, device_id = role_camera_index(setup, role)
        if idx is None:
            continue
        r = CameraReader(idx, args.width, args.height, device_id=device_id)
        r.start()
        readers[role] = r

    if not readers:
        print("No cameras configured in camera_setup.json", file=sys.stderr)
        return 1

    eye = EyeTrackerAdapter()
    _apply_ir_center_locks(eye, setup)
    recorder = RecordingManager(project_root() / "recordings")
    focus = next(role for role in PREVIEW_ROLES if role in readers)
    last_sizes = {role: (args.width, args.height) for role in readers}
    panel_rects: dict[str, tuple[int, int, int, int]] = {}
    dirty = False

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        nonlocal dirty
        role = _hit_role(panel_rects, x, y)
        if role not in EYE_ROLES:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            locked = eye.lock_sphere_center_at_pupil(role)
            if locked is None:
                return
            set_eye_center_ir_px(setup, role, locked)
            dirty = True
            print(f"  Saved pending eye_center_ir_px={list(locked)} (Enter to write).")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not eye.unlock_sphere_center(role):
                return
            set_eye_center_ir_px(setup, role, None)
            dirty = True
            print("  Cleared pending eye_center_ir_px (Enter to write).")

    cv2.namedWindow("MultiCam Preview", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("MultiCam Preview", on_mouse)

    print(
        "Keys: 1/2/3 focus | Z/X zoom | WASD pan | R reset view | "
        "F flip | M mirror | Enter save setup | Space record | Q quit"
    )
    print("  Left/right show pupil + eyeball overlay from the eye tracker.")
    print("  Flip/mirror then zoom (same order as tracking when zoom_affects_tracking).")
    print(
        "  Mouse on L/R: left-click locks eye center at current pupil "
        "(look into IR first); right-click unlocks."
    )

    try:
        while True:
            display_panels: dict[str, NDArray[np.uint8]] = {}
            live_panels: dict[str, NDArray[np.uint8]] = {}
            for role, reader in readers.items():
                ok, frame = reader.read()
                if ok and frame is not None:
                    v = views[role]
                    flags = flips[role]
                    if role in EYE_ROLES:
                        panel = _eye_panel_with_overlay(
                            eye,
                            frame,
                            role,
                            flags["flip"],
                            flags["mirror"],
                            v,
                            zoom_tracking.get(role, False),
                            calib_dir,
                        )
                    else:
                        panel = apply_camera_display(
                            frame, flags["flip"], flags["mirror"], v
                        )
                    last_sizes[role] = (panel.shape[1], panel.shape[0])
                    _annotate_panel(panel, role, v, flags, eye)
                    display_panels[role] = panel
                    live_panels[role] = panel
                    continue

                # Keep role slots stable while CameraReader reconnects in the background.
                snap = reader.snapshot_status()
                w, h = last_sizes[role]
                display_panels[role] = make_status_panel(
                    w,
                    h,
                    [
                        role,
                        f"cam{reader.capture_index}",
                        str(snap["status"]),
                        "no frame",
                        "Z/X WASD R F M still edit",
                    ],
                )

            if recorder.is_recording():
                if recorder.add_frames(live_panels):
                    print("Recording finished.")

            stacked, panel_rects = _layout_panels(display_panels, focus)
            save_hint = " | UNSAVED" if dirty else ""
            cv2.putText(
                stacked,
                f"Focus {focus} | {recorder.get_status_line()}{save_hint}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imshow("MultiCam Preview", stacked)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("1") and "left" in readers:
                focus = "left"
            if key == ord("2") and "right" in readers:
                focus = "right"
            if key == ord("3") and "front" in readers:
                focus = "front"
            v = views[focus]
            flags = flips[focus]
            if key == ord("z"):
                v["zoom"] = min(5.0, round(v["zoom"] + 0.1, 1))
                dirty = True
            if key == ord("x"):
                v["zoom"] = max(0.5, round(v["zoom"] - 0.1, 1))
                dirty = True
            if key == ord("w"):
                v["pan_y"] -= 10
                dirty = True
            if key == ord("s"):
                v["pan_y"] += 10
                dirty = True
            if key == ord("a"):
                v["pan_x"] -= 10
                dirty = True
            if key == ord("d"):
                v["pan_x"] += 10
                dirty = True
            if key in (ord("r"), ord("R")):
                v["zoom"] = 1.0
                v["pan_x"] = 0
                v["pan_y"] = 0
                dirty = True
                print(f"{focus}: view reset zoom=1.0 pan=0,0")
            if key in (ord("f"), ord("F")):
                flags["flip"] = not flags["flip"]
                dirty = True
                print(f"{focus}: flip={flags['flip']}")
            if key in (ord("m"), ord("M")):
                flags["mirror"] = not flags["mirror"]
                dirty = True
                print(f"{focus}: mirror={flags['mirror']}")
            if key in (13, 10):  # Enter - persist to camera_setup.json
                _persist(setup_path, setup, views, flips)
                dirty = False
            if key == ord(" "):
                if recorder.is_recording():
                    recorder.stop_recording()
                else:
                    if not live_panels:
                        print("No live preview panels ready.")
                    else:
                        sizes = {r: (p.shape[1], p.shape[0]) for r, p in live_panels.items()}
                        recorder.start_recording(sizes)
    finally:
        if dirty:
            _persist(setup_path, setup, views, flips)
            print("Saved pending preview edits on exit.")
        recorder.stop_recording()
        for r in readers.values():
            r.stop()
        cv2.destroyAllWindows()
    return 0


def _apply_ir_center_locks(eye: EyeTrackerAdapter, setup: dict) -> None:
    for role in EYE_ROLES:
        xy = eye_center_ir_px_from_setup(setup, role)
        if xy is None:
            continue
        eye.apply_locked_eye_center_ir_px(role, xy)
        print(f"[{role}] Restored eye_center_ir_px lock at {xy}")


def _eye_panel_with_overlay(
    eye: EyeTrackerAdapter,
    frame: NDArray[np.uint8],
    eye_id: str,
    flip_v: bool,
    flip_h: bool,
    view: dict,
    zoom_affects_tracking: bool,
    calib_dir: Path,
) -> NDArray[np.uint8]:
    """Run Orlosky tracking and return the preview panel (config flip/zoom + overlay)."""
    tracking_frame, track_flip_v, track_flip_h = prepare_eye_tracking_frame(
        frame, flip_v, flip_h, view, zoom_affects_tracking
    )
    h, w = frame.shape[:2]
    role = "left_eye" if eye_id == "left" else "right_eye"
    unproject = resolve_eye_unproject_model(
        sensor_width=w,
        sensor_height=h,
        flip_vertical=flip_v,
        flip_horizontal=flip_h,
        view=view,
        zoom_affects_tracking=zoom_affects_tracking,
        calib_dir=calib_dir,
        role=role,
    )
    eye.process_frame(
        tracking_frame,
        eye_id=eye_id,
        flip_vertical=track_flip_v,
        flip_horizontal=track_flip_h,
        unproject=unproject,
    )
    return compose_eye_preview_panel(
        frame,
        eye.get_preview_frame(eye_id),
        flip_v,
        flip_h,
        view,
        zoom_affects_tracking,
    )


def _persist(
    setup_path: Path,
    setup: dict,
    views: dict[str, dict],
    flips: dict[str, dict[str, bool]],
) -> None:
    merged = merge_setup_display(setup, views, flips)
    save_camera_setup(setup_path, merged)
    setup.clear()
    setup.update(merged)
    print(f"Saved {setup_path}")


def _annotate_panel(
    panel: NDArray[np.uint8],
    role: str,
    view: dict,
    flags: dict[str, bool],
    eye: EyeTrackerAdapter | None = None,
) -> None:
    cv2.putText(
        panel,
        role,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
    )
    lock_txt = ""
    if eye is not None and role in EYE_ROLES:
        locked = eye.get_locked_eye_center_ir_px(role)
        lock_txt = f" LOCK{locked}" if locked is not None else " auto"
    cv2.putText(
        panel,
        f"z={view['zoom']:.1f} pan={view['pan_x']},{view['pan_y']} "
        f"flip={int(flags['flip'])} mir={int(flags['mirror'])}{lock_txt}",
        (8, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
    )


def _hit_role(
    rects: dict[str, tuple[int, int, int, int]],
    x: int,
    y: int,
) -> str | None:
    for role, (x0, y0, x1, y1) in rects.items():
        if x0 <= x < x1 and y0 <= y < y1:
            return role
    return None


def _layout_panels(
    panels: dict[str, NDArray[np.uint8]],
    focus: str,
) -> tuple[NDArray[np.uint8], dict[str, tuple[int, int, int, int]]]:
    order = [r for r in PREVIEW_ROLES if r in panels]
    if not order:
        return np.zeros((240, 320, 3), dtype=np.uint8), {}
    target_h = max(panels[r].shape[0] for r in order)
    imgs = []
    rects: dict[str, tuple[int, int, int, int]] = {}
    x_cursor = 0
    for role in order:
        panel = panels[role]
        if panel.shape[0] != target_h:
            scale = target_h / float(panel.shape[0])
            panel = cv2.resize(
                panel,
                (max(1, int(round(panel.shape[1] * scale))), target_h),
            )
        else:
            panel = panel.copy()
        border = (255, 255, 0) if role == focus else (180, 180, 180)
        cv2.rectangle(
            panel,
            (0, 0),
            (panel.shape[1] - 1, panel.shape[0] - 1),
            border,
            3 if role == focus else 2,
        )
        w = int(panel.shape[1])
        rects[role] = (x_cursor, 0, x_cursor + w, target_h)
        x_cursor += w
        imgs.append(panel)
    return cv2.hconcat(imgs), rects


if __name__ == "__main__":
    raise SystemExit(main())
