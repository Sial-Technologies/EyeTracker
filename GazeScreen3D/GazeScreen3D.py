"""
GazeScreen3D — eye gaze ∩ ArUco screen plane → monitor pixel + heatmap.

Setup GUI offers two starts:
  - Eye tracking only → L/R pupil tracking (+ optional Front preview)
  - Full GazeScreen3D → ArUco screen pose + gaze heatmap

Usage:
  cd GazeScreen3D
  python GazeScreen3D.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
_ARUCO = os.path.join(REPO, "ArucoScreenPose")
# Prefer GazeScreen3D over ArucoScreenPose for shared names (camera_io).
# Python already puts the script dir on sys.path; a naive insert(0) of Aruco
# would shadow GazeScreen3D's camera_io.
for _p in (_ARUCO, ROOT):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, _ARUCO)
sys.path.insert(0, ROOT)

CAMERA_SETUP_PATH = os.path.join(ROOT, "camera_setup.json")
# Placeholder device_id when no stable OS identifier is available; never resolved.
_SYNTHETIC_ID_PREFIX = "Camera_"

from camera_io import CAMERA_CAPTURE_MODES, CAMERA_OPEN_DELAY_SEC, CameraReader, win_cameras
from gaze_scale_calib import (
    GazeScaleCalib,
    draw_center_calib_target,
    draw_edge_targets,
    draw_front_preview_not_for_c,
    record_edge_from_gaze_samples,
)
from heatmap import GazeHeatmap
from ray_screen import gaze_to_screen, project_gaze_to_front_pixels

import eye_tracker
import screen_pose as sp

WINDOW_NAME = "GazeScreen3D"
DEFAULT_HFOV = sp.DEFAULT_HFOV_DEG
EDGE_CALIB_FRAMES = 12
EDGE_NUDGE = 0.05

# Synthetic arrow codes (outside ASCII), same idea as HeatMap input_poll.
KEY_UP = 0xE001
KEY_DOWN = 0xE002
KEY_LEFT = 0xE003
KEY_RIGHT = 0xE004

_DEBOUNCE_SEC = 0.2
_last_char = None
_last_char_time = 0.0
_WIN32_KEY_VKS = (
    (0x51, ord("q")),
    (0x43, ord("c")),
    (0x52, ord("r")),  # Record button
    (0x5A, ord("z")),  # Zoom in
    (0x58, ord("x")),  # Zoom out
    (0x57, ord("w")),  # Pan up
    (0x41, ord("a")),  # Pan left
    (0x53, ord("s")),  # Pan down
    (0x44, ord("d")),  # Pan right
    (0x50, ord("p")),  # Reset view
    (0x31, ord("1")),  # Focus left preview
    (0x32, ord("2")),  # Focus right preview
    (0x33, ord("3")),  # Focus front preview
    (0x4D, ord("m")),
    (0x4B, ord("k")),
    (0x48, ord("h")),
    (0x56, ord("v")),
    (0x55, ord("u")),
    (0x45, ord("e")),  # reset edge scales
    (0x30, ord("0")),
    (0xBB, ord("=")),
    (0xBD, ord("-")),
    (0xDB, ord("[")),  # shrink vertical range
    (0xDD, ord("]")),  # widen vertical range
    (0xBC, ord(",")),  # shrink horizontal range
    (0xBE, ord(".")),  # widen horizontal range
    (0x26, KEY_UP),
    (0x28, KEY_DOWN),
    (0x25, KEY_LEFT),
    (0x27, KEY_RIGHT),
)

_OPENCV_ARROW_MAP = {
    2490368: KEY_UP,
    2621440: KEY_DOWN,
    2424832: KEY_LEFT,
    2555904: KEY_RIGHT,
}


def _debounced(char):
    global _last_char, _last_char_time
    now = time.perf_counter()
    if char == _last_char and (now - _last_char_time) < _DEBOUNCE_SEC:
        return False
    _last_char = char
    _last_char_time = now
    return True


def poll_key():
    key = cv2.waitKey(1)
    if key != -1:
        char = _OPENCV_ARROW_MAP.get(key, key & 0xFF)
        if sys.platform == "win32":
            user32 = __import__("ctypes").windll.user32
            for vk, mapped in _WIN32_KEY_VKS:
                if mapped == char:
                    user32.GetAsyncKeyState(vk)
        if _debounced(char):
            # Debug: print detected key
            if char == ord('r'):
                print(f"[poll_key] Detected 'r' key (char={char})")
            return char
        return 255

    if sys.platform != "win32":
        return 255

    user32 = __import__("ctypes").windll.user32
    if not hasattr(poll_key, "_prev"):
        poll_key._prev = {}
    prev = poll_key._prev
    for vk, char in _WIN32_KEY_VKS:
        down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
        was_down = prev.get(vk, False)
        prev[vk] = down
        if down and not was_down and _debounced(char):
            # Debug: print detected key
            if char == ord('r'):
                print(f"[poll_key] Detected 'r' key via GetAsyncKeyState (char={char}, vk=0x{vk:02X})")
            return char
    return 255


def _arrow_to_edge(key):
    return {
        KEY_UP: "top",
        KEY_DOWN: "bottom",
        KEY_LEFT: "left",
        KEY_RIGHT: "right",
    }.get(key)


def _bgr_to_photoimage(tk, frame, max_w=420, max_h=315):
    """Convert an OpenCV BGR frame to a tkinter PhotoImage (no Pillow)."""
    fh, fw = frame.shape[:2]
    scale = min(max_w / fw, max_h / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    small = cv2.resize(frame, (nw, nh), interpolation=interp)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    header = f"P6\n{nw} {nh}\n255\n".encode("ascii")
    return tk.PhotoImage(data=header + rgb.tobytes(), format="PPM")


def _placeholder_photo(tk, width, height, rgb=(34, 34, 34)):
    """Solid-color PhotoImage used as a sized placeholder (avoids char-unit Label shrink)."""
    r, g, b = rgb
    row = bytes([r, g, b]) * width
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    return tk.PhotoImage(data=header + row * height, format="PPM")


def _synthetic_device_id(index):
    return f"{_SYNTHETIC_ID_PREFIX}{index}"


def _stable_device_id(device_id):
    """device_id if it identifies a physical device, else None (index-only fallback)."""
    if not device_id or device_id.startswith(_SYNTHETIC_ID_PREFIX):
        return None
    return device_id


def _enumerate_capture_devices(max_cams):
    """[(index, device_id, name), ...] for capture endpoints in CAP_MSMF index order.

    On Windows, SetupAPI's KSCATEGORY_VIDEO_CAMERA interfaces are the MSMF
    endpoints themselves. The PnP Camera class is not: it also lists non-capture
    siblings (e.g. IR MI_02), so pairing it with OpenCV indices by list position
    assigns the wrong IDs. Elsewhere fall back to probing with synthetic IDs.
    """
    devices = win_cameras.list_capture_devices() if win_cameras is not None else None
    if devices is not None:
        return [(d["index"], d["device_id"], d["name"]) for d in devices]

    indices = sp.detect_cameras(
        max_cams=max_cams,
        backends=tuple(backend for _name, backend, _fourcc in CAMERA_CAPTURE_MODES),
    )
    return [(index, _synthetic_device_id(index), f"Camera {index}") for index in indices]


def detect_cameras_with_names(max_cams=10):
    """Enumerate OpenCV indices paired with stable Windows device IDs.

    Returns list of (index, device_id, display_name).
    display_name looks like \"0: USB Camera\" (index suffix on duplicate names).
    """
    devices = _enumerate_capture_devices(max_cams)
    name_counts = {}
    for _index, _did, name in devices:
        name_counts[name] = name_counts.get(name, 0) + 1

    cameras = []
    for index, device_id, name in devices:
        label = f"{name} ({index})" if name_counts.get(name, 0) > 1 else name
        display = f"{index}: {label}"
        cameras.append((index, device_id, display))
    return cameras


def _load_camera_setup():
    """Load camera_setup.json or return empty dict."""
    try:
        with open(CAMERA_SETUP_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


def _save_camera_setup(setup):
    """Persist role → {device_id, index, flip, mirror, view} to camera_setup.json."""
    try:
        with open(CAMERA_SETUP_PATH, "w", encoding="utf-8") as f:
            json.dump(setup, f, indent=2)
            f.write("\n")
    except OSError as exc:
        print(f"Could not save camera setup: {exc}")


def _normalize_preview_view(view):
    """Clamp/coerce a zoom/pan dict to the runtime shape."""
    if not isinstance(view, dict):
        return {"zoom": 1.0, "pan_x": 0, "pan_y": 0}
    try:
        zoom = float(view.get("zoom", 1.0))
    except (TypeError, ValueError):
        zoom = 1.0
    try:
        pan_x = int(round(float(view.get("pan_x", 0))))
    except (TypeError, ValueError):
        pan_x = 0
    try:
        pan_y = int(round(float(view.get("pan_y", 0))))
    except (TypeError, ValueError):
        pan_y = 0
    return {
        "zoom": max(0.5, min(5.0, round(zoom, 1))),
        "pan_x": pan_x,
        "pan_y": pan_y,
    }


def preview_views_from_setup(saved):
    """Build per-role preview zoom/pan from camera_setup.json (or defaults)."""
    views = {role: {"zoom": 1.0, "pan_x": 0, "pan_y": 0} for role in ("left", "right", "front")}
    if not isinstance(saved, dict):
        return views
    for role in views:
        entry = saved.get(role)
        if isinstance(entry, dict) and isinstance(entry.get("view"), dict):
            views[role] = _normalize_preview_view(entry["view"])
    return views


def _view_dict_for_save(preview_views, role):
    v = preview_views.get(role) if isinstance(preview_views, dict) else None
    return _normalize_preview_view(v)


def zoom_affects_tracking_from_setup(saved):
    """Per-role flag: when True, zoom/pan is applied before eye tracking."""
    flags = {role: False for role in ("left", "right", "front")}
    if not isinstance(saved, dict):
        return flags
    for role in flags:
        entry = saved.get(role)
        if isinstance(entry, dict):
            flags[role] = bool(entry.get("zoom_affects_tracking", False))
    return flags


def zoom_affects_tracking_for(choice, role):
    """Resolve zoom_affects_tracking for a role from run() choice dict."""
    if not isinstance(choice, dict):
        return False
    flags = choice.get("zoom_affects_tracking")
    if isinstance(flags, dict):
        return bool(flags.get(role, False))
    return bool(choice.get(f"zoom_affects_tracking_{role}", False))


def _index_for_device_id(cameras_info, device_id):
    """Find current OpenCV index for a saved device_id, or None."""
    if not device_id:
        return None
    for index, did, _display in cameras_info:
        if did == device_id:
            return index
    return None


def _display_for_index(cameras_info, index):
    for idx, _did, display in cameras_info:
        if idx == index:
            return display
    return None


def _device_id_for_index(cameras_info, index):
    for idx, did, _display in cameras_info:
        if idx == index:
            return did
    return _synthetic_device_id(index)


def _parse_camera_selection(value):
    """Parse combobox value → OpenCV index, or None for 'None'/empty."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text == "None":
        return None
    # "2: USB Camera" or bare "2"
    head = text.split(":", 1)[0].strip()
    try:
        return int(head)
    except ValueError:
        return None


def selection_gui():
    """Role-first picker: Left / Right / Front panels, each with dropdown + live preview."""
    import tkinter as tk
    from tkinter import ttk

    state = {
        "cameras_info": detect_cameras_with_names(),
        "role_device": {"left": None, "right": None, "front": None},
    }
    saved = _load_camera_setup()

    def cam_displays(optional=False):
        labels = [display for _idx, _did, display in state["cameras_info"]]
        # Always include None so a missing rematch can show unbound without stealing.
        return ["None"] + labels

    def resolve_saved_role(role_key, optional=False):
        """Match saved device_id to current index; fall back to saved index; else default."""
        entry = saved.get(role_key)
        info = state["cameras_info"]
        if entry and isinstance(entry, dict):
            device_id = entry.get("device_id")
            index = _index_for_device_id(info, device_id)
            if index is None and _stable_device_id(device_id) is None and entry.get("index") is not None:
                # No stable ID was ever known — try last known index if still present.
                # A stable ID that is missing means that device is unplugged; its old
                # index now belongs to another camera, so it must stay unbound.
                want = int(entry["index"])
                if any(idx == want for idx, _d, _n in info):
                    index = want
            if index is not None:
                state["role_device"][role_key] = _device_id_for_index(info, index)
                return _display_for_index(info, index), entry
            # Saved device not present — leave unbound (do not steal another index).
            state["role_device"][role_key] = device_id
            return "None", entry

        # No save: sensible defaults
        displays = [d for d in cam_displays() if d != "None"]
        if role_key == "right":
            return "None", None
        if role_key == "front" and len(displays) > 1:
            return displays[1], None
        return (displays[0] if displays else "None"), None

    left_display, left_saved = resolve_saved_role("left")
    right_display, right_saved = resolve_saved_role("right", optional=True)
    front_display, front_saved = resolve_saved_role("front")

    root = tk.Tk()
    root.title("GazeScreen3D — camera setup")
    root.minsize(1100, 640)
    root.geometry("1280x720")

    tk.Label(
        root,
        text="Pick a camera for each role. Device IDs persist across sessions and index changes.",
        font=("Arial", 12, "bold"),
    ).pack(pady=(10, 2))

    status_row = ttk.Frame(root)
    status_row.pack(pady=(0, 6), fill="x", padx=10)
    status_lbl = tk.Label(
        status_row,
        text="",
        font=("Arial", 9),
        fg="#555",
        anchor="w",
        justify="left",
    )
    status_lbl.pack(side="left", fill="x", expand=True)
    refresh_btn = ttk.Button(status_row, text="Refresh camera list")
    refresh_btn.pack(side="right", padx=(8, 0))

    left_var = tk.StringVar(value=left_display)
    right_var = tk.StringVar(value=right_display)
    front_var = tk.StringVar(value=front_display)
    flip_left = tk.BooleanVar(value=True if left_saved is None else bool(left_saved.get("flip", True)))
    flip_right = tk.BooleanVar(
        value=False if right_saved is None else bool(right_saved.get("flip", False))
    )
    mirror_left = tk.BooleanVar(
        value=False if left_saved is None else bool(left_saved.get("mirror", False))
    )
    mirror_right = tk.BooleanVar(
        value=True if right_saved is None else bool(right_saved.get("mirror", True))
    )
    flip_front = tk.BooleanVar(
        value=False if front_saved is None else bool(front_saved.get("flip", False))
    )
    mirror_front = tk.BooleanVar(
        value=False if front_saved is None else bool(front_saved.get("mirror", False))
    )
    saved_zoom_flags = zoom_affects_tracking_from_setup(saved)
    zoom_track_left = tk.BooleanVar(value=saved_zoom_flags["left"])
    zoom_track_right = tk.BooleanVar(value=saved_zoom_flags["right"])
    zoom_track_vars = {"left": zoom_track_left, "right": zoom_track_right}

    PREVIEW_W, PREVIEW_H = 380, 285
    role_vars = {"left": left_var, "right": right_var, "front": front_var}
    ROLE_META = (
        ("left", "Left IR (required)", left_var, False, "#2e7d32", flip_left, mirror_left, True),
        ("right", "Right IR (optional)", right_var, True, "#1565c0", flip_right, mirror_right, False),
        ("front", "Front camera (required)", front_var, False, "#e65100", flip_front, mirror_front, False),
    )

    # Role owns its reader — panel never shares capture by OpenCV index.
    readers = {"left": None, "right": None, "front": None}
    panels = {}
    comboboxes = {}
    # Camera open/stop is slow on Windows; never block the Tk thread.
    _cam_lock = threading.Lock()
    _open_gen = {"left": 0, "right": 0, "front": 0}

    def stop_previews():
        for role in list(readers):
            _open_gen[role] += 1  # cancel any in-flight open
            reader = readers.get(role)
            readers[role] = None
            if reader is not None:
                reader.stop()
        time.sleep(0.2)

    def role_using_index(index, exclude_role=None):
        for role, var in role_vars.items():
            if role == exclude_role:
                continue
            if _parse_camera_selection(var.get()) == index:
                return role
        return None

    def sync_role_reader(role):
        """Match this role's CameraReader to its dropdown without freezing the UI."""
        index = _parse_camera_selection(role_vars[role].get())
        current = readers.get(role)

        if index is not None:
            other = role_using_index(index, exclude_role=role)
            if other is not None:
                status_lbl.config(
                    text=f"Index {index} already used by {other}. Pick a different camera."
                )
                if current is not None:
                    display = _display_for_index(state["cameras_info"], current.index)
                    if display:
                        role_vars[role].set(display)
                else:
                    role_vars[role].set("None")
                return
            if current is not None and current.index == index:
                state["role_device"][role] = _device_id_for_index(state["cameras_info"], index)
                return

        _open_gen[role] += 1
        gen = _open_gen[role]
        old = readers.get(role)
        readers[role] = None  # preview shows "opening…" immediately
        if index is None:
            state["role_device"][role] = None
        else:
            state["role_device"][role] = _device_id_for_index(state["cameras_info"], index)
            status_lbl.config(text=f"Opening camera {index} for {role}…")
        device_id = state["role_device"][role]

        def worker():
            with _cam_lock:
                if old is not None:
                    old.stop()
                    time.sleep(0.25)
                if _open_gen[role] != gen:
                    return
                if index is None:
                    return
                reader = CameraReader(
                    index,
                    width=640,
                    height=480,
                    device_id=_stable_device_id(device_id),
                )
                reader.start()

            def apply():
                if not root.winfo_exists() or _open_gen[role] != gen:
                    reader.stop()
                    return
                readers[role] = reader
                status_lbl.config(
                    text=f"{role} ready: {_display_for_index(state['cameras_info'], index) or index}"
                )

            try:
                root.after(0, apply)
            except tk.TclError:
                reader.stop()

        threading.Thread(target=worker, daemon=True).start()

    def on_selection_change(role):
        def _handler(*_args):
            sync_role_reader(role)

        return _handler

    def apply_camera_list_to_ui(status_prefix=None):
        """Refresh combobox values; rematch each role by saved device_id."""
        displays = cam_displays()
        for role, _title, var, optional, *_rest in ROLE_META:
            values = cam_displays(optional=optional)
            box = comboboxes.get(role)
            if box is not None:
                box["values"] = values

            device_id = state["role_device"].get(role)
            new_index = _index_for_device_id(state["cameras_info"], device_id) if device_id else None
            if new_index is not None:
                display = _display_for_index(state["cameras_info"], new_index)
                if display and var.get() != display:
                    var.set(display)
            else:
                # Device gone: clear optional roles; keep required showing None until user picks.
                current_idx = _parse_camera_selection(var.get())
                still_there = current_idx is not None and any(
                    idx == current_idx for idx, _d, _n in state["cameras_info"]
                )
                if not still_there:
                    var.set("None")
                    _open_gen[role] += 1
                    old = readers.get(role)
                    readers[role] = None
                    if old is not None:
                        threading.Thread(target=old.stop, daemon=True).start()

        bits = []
        for role in ("left", "right", "front"):
            idx = _parse_camera_selection(role_vars[role].get())
            did = state["role_device"].get(role)
            short = {"left": "L", "right": "R", "front": "F"}[role]
            if idx is None:
                bits.append(f"{short}=—")
            else:
                name = _display_for_index(state["cameras_info"], idx) or str(idx)
                bits.append(f"{short}={name}")
        n = len(state["cameras_info"])
        prefix = (status_prefix + "  |  ") if status_prefix else ""
        status_lbl.config(text=f"{prefix}Detected {n}  |  {', '.join(bits)}")

    roles_row = ttk.Frame(root)
    roles_row.pack(padx=10, pady=4, fill="both", expand=True)

    placeholder = _placeholder_photo(tk, PREVIEW_W, PREVIEW_H)

    for col, (role, title, var, optional, color, flip_var, mirror_var, flip_default_hint) in enumerate(
        ROLE_META
    ):
        cell = ttk.LabelFrame(roles_row, text=title)
        cell.grid(row=0, column=col, padx=8, pady=4, sticky="nsew")
        roles_row.columnconfigure(col, weight=1)

        pick = ttk.Frame(cell)
        pick.pack(pady=(8, 4))
        tk.Label(pick, text="Camera:", font=("Arial", 10)).pack(side="left", padx=(0, 6))
        box = ttk.Combobox(
            pick,
            textvariable=var,
            values=cam_displays(optional=optional),
            width=28,
            state="readonly",
        )
        box.pack(side="left")
        comboboxes[role] = box

        border = tk.Frame(cell, highlightthickness=4, highlightbackground=color)
        border.pack(padx=8, pady=4)
        img_lbl = tk.Label(
            border,
            image=placeholder,
            text="Select a camera",
            compound="center",
            bg="#222",
            fg="#ccc",
            font=("Arial", 13, "bold"),
        )
        img_lbl.pack()

        hint = "Flip = upside-down mount" if flip_default_hint else "Flip / mirror if image looks wrong"
        checks = ttk.Frame(cell)
        checks.pack(pady=6)
        ttk.Checkbutton(checks, text="Flip", variable=flip_var).pack(side="left", padx=4)
        ttk.Checkbutton(checks, text="Mirror", variable=mirror_var).pack(side="left", padx=4)
        if role in zoom_track_vars:
            ttk.Checkbutton(
                checks,
                text="Zoom→track",
                variable=zoom_track_vars[role],
            ).pack(side="left", padx=4)
            hint = f"{hint} | Zoom→track = zoom before pupil tracking"
        tk.Label(cell, text=hint, font=("Arial", 8), fg="#666").pack(pady=(0, 6))

        panels[role] = {
            "var": var,
            "label": img_lbl,
            "photo": placeholder,
            "color": color,
            "flip": flip_var,
            "mirror": mirror_var,
        }
        var.trace_add("write", on_selection_change(role))

    setup_preview_views = preview_views_from_setup(saved)

    def update_previews():
        if not root.winfo_exists():
            return
        live = 0
        bits = []
        for role, _title, var, *_rest in ROLE_META:
            panel = panels[role]
            idx = _parse_camera_selection(var.get())
            short = {"left": "L", "right": "R", "front": "F"}[role]
            reader = readers.get(role)
            if idx is None:
                panel["label"].config(image=placeholder, text="None\n(optional)")
                panel["photo"] = placeholder
                bits.append(f"{short}=—")
                continue
            bits.append(f"{short}={idx}")
            if reader is None:
                panel["label"].config(image=placeholder, text=f"Cam {idx}\nopening…")
                panel["photo"] = placeholder
                continue
            # Guard: role reader must match selected index (strong link).
            if reader.index != idx:
                panel["label"].config(image=placeholder, text=f"Cam {idx}\nsyncing…")
                panel["photo"] = placeholder
                continue
            ret, frame = reader.read()
            if ret and frame is not None:
                live += 1
                # Live flip/mirror preview (same semantics as run()).
                if panel["flip"].get():
                    frame = cv2.flip(frame, 0)
                if panel["mirror"].get():
                    frame = cv2.flip(frame, 1)
                fitted, _, _, _, _ = fit_frame(frame, PREVIEW_W, PREVIEW_H)
                fitted = render_panel_with_view(fitted, setup_preview_views, role)
                try:
                    photo = _bgr_to_photoimage(tk, fitted, PREVIEW_W, PREVIEW_H)
                    panel["label"].config(image=photo, text="")
                    panel["photo"] = photo
                except tk.TclError:
                    return
            else:
                status = reader.status if reader else "OFFLINE"
                panel["label"].config(
                    image=placeholder,
                    text=f"Cam {idx}\n({status})",
                )
                panel["photo"] = placeholder
        n = len(state["cameras_info"])
        opened = sum(1 for r in readers.values() if r is not None)
        # Don't overwrite an in-progress "Opening…" status every tick.
        current = status_lbl.cget("text")
        if not current.startswith("Opening"):
            status_lbl.config(
                text=f"Detected {n}  |  {', '.join(bits)}  |  live {live}/{opened}"
            )
        root.after(66, update_previews)

    def refresh_cameras():
        status_lbl.config(text="Refreshing camera list…")
        root.update_idletasks()
        # Release exclusive handles so re-probe can see every device.
        stop_previews()
        state["cameras_info"] = detect_cameras_with_names()
        # Rematch roles by device_id (indices may have shifted).
        messages = []
        for role in ("left", "right", "front"):
            device_id = state["role_device"].get(role)
            if not device_id:
                continue
            new_idx = _index_for_device_id(state["cameras_info"], device_id)
            if new_idx is None:
                messages.append(f"{role} offline")
            else:
                old_display = role_vars[role].get()
                new_display = _display_for_index(state["cameras_info"], new_idx)
                if new_display and new_display != old_display:
                    messages.append(f"{role}→{new_display}")
        apply_camera_list_to_ui(
            status_prefix=("Remapped: " + ", ".join(messages)) if messages else "Camera list refreshed"
        )
        for role in ("left", "right", "front"):
            sync_role_reader(role)

    refresh_btn.config(command=refresh_cameras)

    tk.Label(
        root,
        text="Tip: cycle each dropdown until the preview matches that role "
        "(eye close-up vs monitor/room). Refresh remaps saved devices if indexes shifted.\n"
        "Eye tracking only: L/R pupil tracking (Front optional). "
        "Full GazeScreen3D: needs Front + ArUco. Click IR = lock | U unlock | Q quit",
        font=("Arial", 9),
        justify="center",
    ).pack(pady=6)

    choice = {}

    def start(mode):
        left_i = _parse_camera_selection(left_var.get())
        right_i = _parse_camera_selection(right_var.get())
        front_i = _parse_camera_selection(front_var.get())
        if left_i is None and right_i is None:
            status_lbl.config(text="Need at least one IR eye camera (Left or Right).")
            return
        if mode == "full" and front_i is None:
            status_lbl.config(text="Full GazeScreen3D needs a Front camera.")
            return
        assigned = [i for i in (left_i, right_i, front_i) if i is not None]
        if len(assigned) != len(set(assigned)):
            status_lbl.config(text="Each role needs a different camera.")
            return
        status_lbl.config(text="Starting… keeping open cameras (no reopen).")
        root.update_idletasks()

        setup = {}
        for role, index in (("left", left_i), ("right", right_i), ("front", front_i)):
            if index is None:
                setup[role] = None
                continue
            flip_map = {"left": flip_left, "right": flip_right, "front": flip_front}
            mirror_map = {"left": mirror_left, "right": mirror_right, "front": mirror_front}
            setup[role] = {
                "device_id": state["role_device"].get(role) or _device_id_for_index(state["cameras_info"], index),
                "index": index,
                "flip": bool(flip_map[role].get()),
                "mirror": bool(mirror_map[role].get()),
                "view": _view_dict_for_save(setup_preview_views, role),
                "zoom_affects_tracking": bool(
                    zoom_track_vars[role].get() if role in zoom_track_vars else False
                ),
            }
        _save_camera_setup(setup)

        choice["mode"] = mode
        choice["left"] = left_i
        choice["right"] = right_i
        choice["front"] = front_i
        choice["flip_left"] = flip_left.get()
        choice["flip_right"] = flip_right.get()
        choice["mirror_left"] = mirror_left.get()
        choice["mirror_right"] = mirror_right.get()
        choice["flip_front"] = flip_front.get()
        choice["mirror_front"] = mirror_front.get()
        choice["preview_views"] = {
            role: _normalize_preview_view(setup_preview_views.get(role))
            for role in ("left", "right", "front")
        }
        choice["zoom_affects_tracking"] = {
            "left": bool(zoom_track_left.get()),
            "right": bool(zoom_track_right.get()),
            "front": False,
        }
        choice["device_ids"] = {
            role: _stable_device_id(entry["device_id"]) if entry else None
            for role, entry in setup.items()
        }
        # Handoff by role (same reader object as the matching preview panel).
        # Also keep index→reader for run() paths that look up by camera index.
        by_role = {}
        by_index = {}
        for role, reader in readers.items():
            if reader is not None:
                by_role[role] = reader
                by_index[reader.index] = reader
            readers[role] = None
        choice["readers_by_role"] = by_role
        choice["readers"] = by_index
        root.destroy()

    def on_close():
        stop_previews()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    
    # Recording control section
    record_frame = tk.Frame(root, bg="#d32f2f", relief="raised", bd=3)
    record_frame.pack(pady=10, padx=10, fill="x")
    
    tk.Label(
        record_frame,
        text="🎥 RECORDING CONTROLS",
        font=("Arial", 14, "bold"),
        bg="#d32f2f",
        fg="white"
    ).pack(pady=5)
    
    recordings_path = os.path.join(ROOT, "recordings")
    if not os.path.exists(recordings_path):
        os.makedirs(recordings_path)
    
    tk.Label(
        record_frame,
        text=f"Save location: {recordings_path}",
        font=("Arial", 9),
        bg="#d32f2f",
        fg="yellow"
    ).pack(pady=2)
    
    # Recording state
    recording_state = {
        "recorder": RecordingManager(output_dir=recordings_path),
        "recording": False,
        "preview_views": setup_preview_views,
        "view_focus": "left",
    }
    
    status_label = tk.Label(
        record_frame,
        text="Ready: record Left / Right / Front previews (15s each, separate MP4s)",
        font=("Arial", 10),
        bg="#d32f2f",
        fg="white"
    )
    status_label.pack(pady=5)
    
    focus_label = tk.Label(
        record_frame,
        text=preview_view_hud_line(recording_state["preview_views"], recording_state["view_focus"]),
        font=("Arial", 9),
        bg="#d32f2f",
        fg="#FFEB3B",
    )
    focus_label.pack(pady=2)

    view_ctrl = tk.Frame(record_frame, bg="#d32f2f")
    view_ctrl.pack(pady=4)

    def gui_view_action(action):
        available = {
            role
            for role in PREVIEW_ROLES
            if readers.get(role) is not None
            and _parse_camera_selection(panels[role]["var"].get()) is not None
        } or set(PREVIEW_ROLES)
        recording_state["view_focus"] = apply_view_action(
            recording_state["preview_views"],
            recording_state["view_focus"],
            action,
            available,
        )
        update_recording_status()

    def _mk(parent, text, action, bg="#555555", width=4):
        return tk.Button(
            parent,
            text=text,
            font=("Arial", 10, "bold"),
            bg=bg,
            fg="white",
            width=width,
            command=lambda a=action: gui_view_action(a),
        )

    focus_row = tk.Frame(view_ctrl, bg="#d32f2f")
    focus_row.pack(side="left", padx=8)
    tk.Label(focus_row, text="Focus", font=("Arial", 9, "bold"), bg="#d32f2f", fg="white").pack()
    fr = tk.Frame(focus_row, bg="#d32f2f")
    fr.pack()
    _mk(fr, "L", "focus_left", "#0288D1").pack(side="left", padx=2)
    _mk(fr, "R", "focus_right", "#0288D1").pack(side="left", padx=2)
    _mk(fr, "F", "focus_front", "#0288D1").pack(side="left", padx=2)

    zoom_row = tk.Frame(view_ctrl, bg="#d32f2f")
    zoom_row.pack(side="left", padx=8)
    tk.Label(zoom_row, text="Zoom", font=("Arial", 9, "bold"), bg="#d32f2f", fg="white").pack()
    zr = tk.Frame(zoom_row, bg="#d32f2f")
    zr.pack()
    _mk(zr, "Z+", "zoom_in", "#2E7D32").pack(side="left", padx=2)
    _mk(zr, "X-", "zoom_out", "#1565C0").pack(side="left", padx=2)
    _mk(zr, "RST", "reset", "#F9A825", width=5).pack(side="left", padx=2)

    pan_row = tk.Frame(view_ctrl, bg="#d32f2f")
    pan_row.pack(side="left", padx=8)
    tk.Label(pan_row, text="Pan", font=("Arial", 9, "bold"), bg="#d32f2f", fg="white").pack()
    pr = tk.Frame(pan_row, bg="#d32f2f")
    pr.pack()
    _mk(pr, "W", "pan_up").grid(row=0, column=1, padx=1, pady=1)
    _mk(pr, "A", "pan_left").grid(row=1, column=0, padx=1, pady=1)
    _mk(pr, "S", "pan_down").grid(row=1, column=1, padx=1, pady=1)
    _mk(pr, "D", "pan_right").grid(row=1, column=2, padx=1, pady=1)
    
    # Button frame
    button_frame = tk.Frame(record_frame, bg="#d32f2f")
    button_frame.pack(pady=10)
    
    def capture_setup_preview_frames():
        """One frame per live preview widget (with per-panel zoom/pan)."""
        frames = {}
        preview_views = recording_state["preview_views"]
        for role in PREVIEW_ROLES:
            panel = panels.get(role)
            reader = readers.get(role)
            if panel is None or reader is None:
                continue
            idx = _parse_camera_selection(panel["var"].get())
            if idx is None or reader.index != idx:
                continue
            ret, frame = reader.read()
            if not ret or frame is None:
                continue
            if panel["flip"].get():
                frame = cv2.flip(frame, 0)
            if panel["mirror"].get():
                frame = cv2.flip(frame, 1)
            fitted, _, _, _, _ = fit_frame(frame, PREVIEW_W, PREVIEW_H)
            frames[role] = render_panel_with_view(fitted, preview_views, role)
        return frames
    
    def update_recording_status():
        recorder = recording_state["recorder"]
        focus_label.config(
            text=preview_view_hud_line(
                recording_state["preview_views"], recording_state["view_focus"]
            )
        )
        if recorder.is_recording():
            remaining = recorder.get_remaining_time()
            status_label.config(
                text=f"🔴 RECORDING L/R/F previews — {remaining:.1f}s remaining"
            )
        else:
            status_label.config(
                text="Ready: record Left / Right / Front previews (15s each, separate MP4s)"
            )
    
    def on_setup_preview_key(event):
        if len(event.char) != 1:
            return
        key = ord(event.char.lower())
        focus = recording_state["view_focus"]
        new_focus, handled = handle_preview_view_key(
            recording_state["preview_views"], focus, key
        )
        if not handled:
            return
        recording_state["view_focus"] = new_focus
        update_recording_status()
        return "break"
    
    root.bind("<Key>", on_setup_preview_key)
    
    def start_recording_gui():
        recorder = recording_state["recorder"]
        if recorder.is_recording():
            print("Already recording!")
            return
        
        frames = capture_setup_preview_frames()
        if not frames:
            print("⚠️ No camera preview available. Select cameras first.")
            return
        sizes = {role: _panel_size_bgr(panel) for role, panel in frames.items()}
        success = recorder.start_recording(sizes, fps=30)
        if success:
            recording_state["recording"] = True
            print(f"🎬 Recording preview streams from setup GUI")
            print(f"   Saving to: {recordings_path}")
            update_recording_status()
            record_preview_frames()
    
    def stop_recording_gui():
        recorder = recording_state["recorder"]
        if recorder.is_recording():
            recorder.stop_recording()
            recording_state["recording"] = False
            print("✋ Recording stopped from GUI")
            update_recording_status()
    
    def reset_recording_gui():
        recording_state["recording"] = False
        for role in PREVIEW_ROLES:
            v = preview_view_for(recording_state["preview_views"], role)
            v["zoom"], v["pan_x"], v["pan_y"] = 1.0, 0, 0
        print("Preview views reset")
        update_recording_status()
    
    def record_preview_frames():
        if not recording_state["recording"]:
            return
        
        recorder = recording_state["recorder"]
        frames = capture_setup_preview_frames()
        if frames:
            clip_done = recorder.add_frames(frames)
            if clip_done:
                recording_state["recording"] = False
                print("✅ Preview recordings finished")
                update_recording_status()
                return
        
        update_recording_status()
        if recording_state["recording"]:
            root.after(33, record_preview_frames)
    
    # Create buttons
    start_btn = tk.Button(
        button_frame,
        text="▶ START RECORDING",
        font=("Arial", 12, "bold"),
        bg="#4CAF50",
        fg="white",
        width=20,
        height=2,
        command=start_recording_gui
    )
    start_btn.pack(side="left", padx=5)
    
    stop_btn = tk.Button(
        button_frame,
        text="⏹ STOP",
        font=("Arial", 12, "bold"),
        bg="#FF5722",
        fg="white",
        width=12,
        height=2,
        command=stop_recording_gui
    )
    stop_btn.pack(side="left", padx=5)
    
    reset_btn = tk.Button(
        button_frame,
        text="🔄 RESET",
        font=("Arial", 12, "bold"),
        bg="#2196F3",
        fg="white",
        width=12,
        height=2,
        command=reset_recording_gui
    )
    reset_btn.pack(side="left", padx=5)
    
    btn_row = tk.Frame(root)
    btn_row.pack(pady=10)
    tk.Button(
        btn_row,
        text="Eye tracking only",
        font=("Arial", 11, "bold"),
        command=lambda: start("eyes"),
    ).pack(side="left", padx=8)
    tk.Button(
        btn_row,
        text="Full GazeScreen3D",
        font=("Arial", 11, "bold"),
        command=lambda: start("full"),
    ).pack(side="left", padx=8)

    def bootstrap():
        apply_camera_list_to_ui(status_prefix="Loaded saved camera setup" if saved else None)
        for role in ("left", "right", "front"):
            sync_role_reader(role)
        update_previews()

    root.after(50, bootstrap)
    root.mainloop()
    stop_previews()
    return choice


def fit_frame(frame, width, height):
    """Letterbox frame into (width, height). Returns canvas, ox, oy, nw, nh."""
    fh, fw = frame.shape[:2]
    scale = min(width / fw, height / fh)
    nw, nh = max(1, int(fw * scale)), max(1, int(fh * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    ox, oy = (width - nw) // 2, (height - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas, ox, oy, nw, nh


def fill_status_panel(canvas, x, y, w, h, lines, border=(80, 80, 80)):
    """Dark placeholder tile with centered status text (avoids a silent black panel)."""
    panel = canvas[y : y + h, x : x + w]
    panel[:] = (24, 24, 24)
    cv2.rectangle(canvas, (x, y), (x + w - 1, y + h - 1), border, 2)
    cy = y + h // 2 - 12 * (len(lines) - 1)
    for i, text in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tx = x + max(8, (w - tw) // 2)
        ty = cy + i * 28
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(canvas, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)


def maximize_cv_window(window_name):
    """Maximize OpenCV window on Windows (call after first imshow)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, window_name)
        if hwnd:
            user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
    except (AttributeError, OSError):
        pass


def lock_eye_sphere_center(eye_id, frame_x, frame_y):
    """Click on IR preview: lock 2D eyeball center (does not replace C gaze→cam)."""
    eye_tracker.load_eye_tracking_state(eye_id)
    eye_tracker.sphere_center_locked_2d = True
    eye_tracker.locked_model_center_avg = (int(frame_x), int(frame_y))
    eye_tracker.prev_model_center_avg = eye_tracker.locked_model_center_avg
    if eye_tracker.last_sphere_center is not None:
        eye_tracker.calibrated_sphere_center = eye_tracker.last_sphere_center.copy()
    eye_tracker.save_eye_tracking_state(eye_id)
    print(f"[{eye_id}] Eye center locked at ({frame_x}, {frame_y}) — click IR preview to set")


def unlock_eye_sphere_centers(active_eyes):
    for eye_id in active_eyes:
        eye_tracker.load_eye_tracking_state(eye_id)
        eye_tracker.sphere_center_locked_2d = False
        eye_tracker.calibrated_sphere_center = None
        eye_tracker.save_eye_tracking_state(eye_id)
    print("Eye centers unlocked (auto-track again).")


def _hit_rect(x, y, rect):
    if rect is None:
        return False
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def make_mouse_handler(layout_state):
    """Map clicks: view controls, record button, then IR preview lock."""

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        view_buttons = layout_state.get("view_control_buttons") or {}
        for action, rect in view_buttons.items():
            if _hit_rect(x, y, rect):
                cb = layout_state.get("view_control_callback")
                if cb is not None:
                    cb(action)
                return
        
        if _hit_rect(x, y, layout_state.get("record_button_rect")):
            cb = layout_state.get("record_button_callback")
            if cb is not None:
                cb()
            return
        
        if not layout_state.get("show_previews", True):
            return
        for eye_id, slot in layout_state.get("eyes", {}).items():
            x0, y0, w, h = slot["rect"]
            if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
                continue
            mapped = map_zoomed_panel_click_to_frame(
                x - x0,
                y - y0,
                w,
                h,
                slot["fit"],
                slot["src_size"],
                slot.get("view"),
                zoom_affects_tracking=bool(slot.get("zoom_affects_tracking", False)),
            )
            if mapped is None:
                return
            lock_eye_sphere_center(eye_id, mapped[0], mapped[1])
            return

    return on_mouse


def map_zoomed_panel_click_to_frame(
    local_x, local_y, panel_w, panel_h, fit, src_size, view=None, zoom_affects_tracking=False
):
    """Map letterboxed panel coords → source frame pixels.

    When zoom_affects_tracking is False (default), invert display zoom/pan first.
    When True, tracking already ran on the zoomed frame so clicks map directly.
    """
    ox, oy, nw, nh = fit
    src_w, src_h = src_size
    x = float(local_x)
    y = float(local_y)
    # Only invert display zoom when it was applied after tracking (display-only mode).
    if view and not zoom_affects_tracking:
        zoom = float(view.get("zoom", 1.0)) or 1.0
        pan_x = float(view.get("pan_x", 0))
        pan_y = float(view.get("pan_y", 0))
        cx, cy = panel_w / 2.0, panel_h / 2.0
        x = (x - cx - pan_x) / zoom + cx
        y = (y - cy - pan_y) / zoom + cy
    if not (ox <= x < ox + nw and oy <= y < oy + nh):
        return None
    frame_x = int((x - ox) * src_w / nw)
    frame_y = int((y - oy) * src_h / nh)
    if not (0 <= frame_x < src_w and 0 <= frame_y < src_h):
        return None
    return frame_x, frame_y


def draw_hud(canvas, lines, x=16, y=28):
    for text in lines:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        y += 24
    return y


def _draw_click_button(canvas, x, y, w, h, text, bg, border, text_color=(255, 255, 255), font_scale=0.55):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    tx = x + max(2, (w - tw) // 2)
    ty = y + (h + th) // 2
    cv2.putText(canvas, text, (tx + 1, ty + 1), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, text, (tx, ty), font, font_scale, text_color, 2, cv2.LINE_AA)
    return (x, y, w, h)


def draw_record_button(canvas, recorder, x, y, w=180, h=50):
    """Draw a clickable record button and return its rect (x, y, w, h)."""
    
    # Determine button state and colors
    if recorder.is_recording():
        # Recording - red background
        bg_color = (0, 0, 200)  # Red in BGR
        text_color = (255, 255, 255)
        text = "STOP"
        border_color = (0, 0, 255)
    else:
        # Ready to record - dark background
        bg_color = (80, 80, 80)  # Gray in BGR
        text_color = (255, 255, 255)
        text = "REC LRF"
        border_color = (200, 200, 200)
    
    rect = _draw_click_button(canvas, x, y, w, h, text, bg_color, border_color, text_color, 0.7)
    
    # Status text below button if recording
    if recorder.is_recording():
        remaining = recorder.get_remaining_time()
        status = f"{remaining:.1f}s"
        cv2.putText(canvas, status, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
    
    return rect


def draw_view_controls(canvas, focus_role, available_roles, x, y):
    """Draw L/R/F focus + WASD pan + Z/X zoom + Reset. Returns {action: rect}."""
    buttons = {}
    bw, bh, gap = 52, 36, 6
    row_h = bh + gap

    # Focus row
    labels = (("focus_left", "L", "left"), ("focus_right", "R", "right"), ("focus_front", "F", "front"))
    cx = x
    for action, label, role in labels:
        active = focus_role == role
        enabled = role in available_roles
        if active:
            bg, border = (0, 180, 255), (0, 255, 255)
        elif enabled:
            bg, border = (60, 60, 60), (180, 180, 180)
        else:
            bg, border = (35, 35, 35), (70, 70, 70)
        buttons[action] = _draw_click_button(canvas, cx, y, bw, bh, label, bg, border)
        cx += bw + gap

    # Zoom row
    y2 = y + row_h
    buttons["zoom_in"] = _draw_click_button(
        canvas, x, y2, bw, bh, "Z+", (50, 90, 50), (100, 200, 100)
    )
    buttons["zoom_out"] = _draw_click_button(
        canvas, x + bw + gap, y2, bw, bh, "X-", (50, 50, 90), (100, 100, 200)
    )
    buttons["reset"] = _draw_click_button(
        canvas, x + 2 * (bw + gap), y2, bw, bh, "RST", (70, 70, 40), (200, 180, 80)
    )

    # WASD pad (right of focus/zoom)
    pad_x = x + 3 * (bw + gap) + 12
    pad_bw, pad_bh = 44, 34
    buttons["pan_up"] = _draw_click_button(
        canvas, pad_x + pad_bw + gap, y, pad_bw, pad_bh, "W", (55, 55, 55), (200, 200, 200)
    )
    buttons["pan_left"] = _draw_click_button(
        canvas, pad_x, y + pad_bh + gap, pad_bw, pad_bh, "A", (55, 55, 55), (200, 200, 200)
    )
    buttons["pan_down"] = _draw_click_button(
        canvas, pad_x + pad_bw + gap, y + pad_bh + gap, pad_bw, pad_bh, "S", (55, 55, 55), (200, 200, 200)
    )
    buttons["pan_right"] = _draw_click_button(
        canvas, pad_x + 2 * (pad_bw + gap), y + pad_bh + gap, pad_bw, pad_bh, "D",
        (55, 55, 55), (200, 200, 200),
    )

    cv2.putText(
        canvas,
        f"focus: {focus_role}",
        (x, y + 2 * row_h + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    return buttons


def apply_zoom_pan(canvas, zoom_level, pan_x, pan_y):
    """Apply zoom and pan transformation to canvas. Returns transformed canvas."""
    if zoom_level == 1.0 and pan_x == 0 and pan_y == 0:
        return canvas
    
    h, w = canvas.shape[:2]
    
    # Calculate zoom matrix
    center_x, center_y = w / 2, h / 2
    
    # Create transformation matrix: translate to origin, scale, translate back, then apply pan
    M = cv2.getRotationMatrix2D((center_x, center_y), 0, zoom_level)
    M[0, 2] += pan_x
    M[1, 2] += pan_y
    
    # Apply transformation
    zoomed = cv2.warpAffine(canvas, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return zoomed


def maybe_apply_tracking_zoom(frame, view, enabled=False):
    """Optional preprocessing layer: apply zoom/pan before eye tracking if enabled.

    When enabled=False (default), returns frame unchanged so zoom stays display-only.
    """
    if not enabled or not isinstance(view, dict):
        return frame
    zoom = float(view.get("zoom", 1.0)) or 1.0
    pan_x = float(view.get("pan_x", 0))
    pan_y = float(view.get("pan_y", 0))
    if zoom == 1.0 and pan_x == 0 and pan_y == 0:
        return frame
    return apply_zoom_pan(frame, zoom, pan_x, pan_y)


def prepare_eye_tracking_frame(frame, flip_vertical, flip_horizontal, view, zoom_affects_tracking):
    """Build the frame + flip flags for eye_tracker.process_frame().

    When zoom_affects_tracking is True, apply flip/mirror then zoom so the tracker
    sees the same region as the zoomed PiP. Flip flags returned as False because
    flips already applied. When False, leave frame raw and let process_frame flip.
    """
    if not zoom_affects_tracking:
        return frame, flip_vertical, flip_horizontal
    out = frame
    if flip_vertical:
        out = cv2.flip(out, 0)
    if flip_horizontal:
        out = cv2.flip(out, 1)
    out = maybe_apply_tracking_zoom(out, view, enabled=True)
    return out, False, False


PREVIEW_ROLES = ("left", "right", "front")


def default_preview_views():
    return {role: {"zoom": 1.0, "pan_x": 0, "pan_y": 0} for role in PREVIEW_ROLES}


def preview_views_from_choice(choice):
    """Prefer live handoff from setup GUI; fall back to camera_setup.json / defaults."""
    if isinstance(choice, dict):
        raw = choice.get("preview_views")
        if isinstance(raw, dict):
            return {
                role: _normalize_preview_view(raw.get(role))
                for role in PREVIEW_ROLES
            }
    return preview_views_from_setup(_load_camera_setup())


def preview_view_for(preview_views, role):
    return preview_views.setdefault(role, {"zoom": 1.0, "pan_x": 0, "pan_y": 0})


def render_panel_with_view(panel, preview_views, role):
    v = preview_view_for(preview_views, role)
    return apply_zoom_pan(panel, v["zoom"], v["pan_x"], v["pan_y"])


def preview_view_hud_line(preview_views, focus_role):
    v = preview_view_for(preview_views, focus_role)
    return (
        f"Preview [{focus_role}]: zoom {v['zoom']:.1f}x pan ({v['pan_x']},{v['pan_y']}) "
        f"| click L/R/F + WASD/Z+/X-/RST"
    )


def apply_view_action(preview_views, focus_role, action, available_roles=None):
    """Apply a view-control action. Returns updated focus_role."""
    available = set(available_roles) if available_roles is not None else set(PREVIEW_ROLES)
    focus_map = {
        "focus_left": "left",
        "focus_right": "right",
        "focus_front": "front",
    }
    if action in focus_map:
        role = focus_map[action]
        if role not in available:
            print(f"Preview '{role}' not available")
            return focus_role
        print(f"Focus → {role}")
        return role

    v = preview_view_for(preview_views, focus_role)
    if action == "zoom_in":
        v["zoom"] = min(5.0, round(v["zoom"] + 0.1, 1))
        print(f"[{focus_role}] zoom {v['zoom']:.1f}x")
    elif action == "zoom_out":
        v["zoom"] = max(0.5, round(v["zoom"] - 0.1, 1))
        print(f"[{focus_role}] zoom {v['zoom']:.1f}x")
    elif action == "pan_up":
        v["pan_y"] -= 10
        print(f"[{focus_role}] pan ({v['pan_x']}, {v['pan_y']})")
    elif action == "pan_down":
        v["pan_y"] += 10
        print(f"[{focus_role}] pan ({v['pan_x']}, {v['pan_y']})")
    elif action == "pan_left":
        v["pan_x"] -= 10
        print(f"[{focus_role}] pan ({v['pan_x']}, {v['pan_y']})")
    elif action == "pan_right":
        v["pan_x"] += 10
        print(f"[{focus_role}] pan ({v['pan_x']}, {v['pan_y']})")
    elif action == "reset":
        v["zoom"], v["pan_x"], v["pan_y"] = 1.0, 0, 0
        print(f"[{focus_role}] view reset")
    return focus_role


def handle_preview_view_key(preview_views, focus_role, key, available_roles=None):
    """Adjust zoom/pan for the focused preview. Returns (focus_role, handled)."""
    key_to_action = {
        ord("1"): "focus_left",
        ord("2"): "focus_right",
        ord("3"): "focus_front",
        ord("z"): "zoom_in",
        ord("x"): "zoom_out",
        ord("w"): "pan_up",
        ord("s"): "pan_down",
        ord("a"): "pan_left",
        ord("d"): "pan_right",
        ord("p"): "reset",
    }
    action = key_to_action.get(key)
    if action is None:
        return focus_role, False
    return apply_view_action(preview_views, focus_role, action, available_roles), True


def _panel_size_bgr(panel):
    h, w = panel.shape[:2]
    return w, h


class RecordingManager:
    """Record left / right / front preview panels to separate MP4 files at once."""
    
    def __init__(self, output_dir="recordings"):
        self.output_dir = output_dir
        self.clip_duration = 15.0  # seconds
        
        self.is_recording_flag = False
        self.recording_start_time = None
        self.video_writers = {}  # role -> cv2.VideoWriter
        self.writer_sizes = {}  # role -> (width, height)
        self.clip_filenames = {}  # role -> path
        self.fps = 30
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def start_recording(self, stream_sizes, fps=30):
        """Start recording. stream_sizes: {role: (width, height)} for active previews."""
        if self.is_recording_flag:
            print("Already recording!")
            return False
        if not stream_sizes:
            print("No preview streams to record.")
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
            filename = os.path.join(self.output_dir, f"{role}_{timestamp}.mp4")
            writer = cv2.VideoWriter(filename, fourcc, fps, (int(width), int(height)))
            if not writer.isOpened():
                print(f"Failed to open video writer for {filename}")
                for w in self.video_writers.values():
                    w.release()
                self.video_writers = {}
                self.writer_sizes = {}
                self.clip_filenames = {}
                return False
            self.video_writers[role] = writer
            self.writer_sizes[role] = (int(width), int(height))
            self.clip_filenames[role] = filename
        
        if not self.video_writers:
            print("No video writers opened.")
            return False
        
        self.recording_start_time = time.perf_counter()
        self.is_recording_flag = True
        print(f"🎬 Started recording {len(self.video_writers)} preview stream(s) ({self.clip_duration:.0f}s):")
        for role, filename in self.clip_filenames.items():
            print(f"   {role}: {filename}")
        return True
    
    def add_frames(self, frames_by_role):
        """Write one frame per active stream. Returns True when clip duration elapsed."""
        if not self.is_recording_flag or not self.video_writers:
            return False
        
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
    
    def _finalize_recording(self):
        if self.video_writers:
            for writer in self.video_writers.values():
                writer.release()
            print(f"✅ Saved {len(self.clip_filenames)} clip(s) ({self.clip_duration:.0f}s each):")
            for role, filename in self.clip_filenames.items():
                print(f"   {role}: {filename}")
            self.video_writers = {}
            self.writer_sizes = {}
            self.clip_filenames = {}
        
        self.is_recording_flag = False
        self.recording_start_time = None
    
    def stop_recording(self):
        if self.is_recording_flag:
            self._finalize_recording()
            print("Recording stopped manually.")
    
    def is_recording(self):
        return self.is_recording_flag
    
    def get_remaining_time(self):
        if not self.is_recording_flag:
            return 0.0
        elapsed = time.perf_counter() - self.recording_start_time
        return max(0.0, self.clip_duration - elapsed)
    
    @property
    def num_clips(self):
        return len(PREVIEW_ROLES)
    
    def get_status_line(self):
        if self.is_recording_flag:
            remaining = self.get_remaining_time()
            n = len(self.video_writers) or len(PREVIEW_ROLES)
            return f"🔴 REC {n} previews - {remaining:.1f}s left"
        return f"Record L/R/F previews {self.clip_duration:.0f}s each (1/2/3 focus, Z/X/WASD)"


def _begin_edge_capture(edge):
    return {"edge": edge, "samples": [], "need": EDGE_CALIB_FRAMES}


def _append_edge_capture(pending, gaze_sample):
    if not pending or not pending.get("edge") or gaze_sample is None:
        return pending, False
    pending["samples"].append(np.asarray(gaze_sample, dtype=np.float64).copy())
    return pending, len(pending["samples"]) >= pending["need"]


def _finish_edge_capture(pending, scale_calib, tracker, width_mm, height_mm, screen_w, screen_h):
    if not pending or not pending.get("edge"):
        return
    ok, message = record_edge_from_gaze_samples(
        scale_calib,
        pending["edge"],
        pending["samples"],
        eye_tracker.R_gaze_to_cam,
        tracker._rotation,
        tracker._translation,
        width_mm,
        height_mm,
        screen_w,
        screen_h,
    )
    print(message if ok else f"Edge calib failed: {message}")


def calibrate_gaze(active_eyes, scale_calib=None):
    """Look at screen center, then align combined gaze to front-camera forward."""
    eye_tracker.calibrate_gaze_to_external(active_eyes)
    ok = bool(eye_tracker.calibrated)
    if ok:
        if scale_calib is not None:
            scale_calib.clear_edges()
        print("C: gaze aligned. Calibrate all 4 edges: look at cross, press matching arrow.")
    return ok


def _take_readers_from_choice(choice, roles):
    """Reuse preview CameraReaders when possible; open missing ones.

    Prefer the exact reader from the matching setup-GUI role panel so a role
    never picks up another panel's stream via index collision.
    """
    pre_by_role = choice.pop("readers_by_role", None) or {}
    pre_by_index = choice.pop("readers", None) or {}
    device_ids = choice.get("device_ids") or {}
    by_index = {}
    fresh_opens = 0

    def take_reader(role, index, label):
        nonlocal fresh_opens
        if index in by_index:
            return by_index[index]

        # pre_by_role and pre_by_index hold the same reader objects; whichever
        # dict we take from, it must leave the other too or cleanup stops it.
        reader = pre_by_role.pop(role, None)
        if reader is not None and reader.index != index:
            # Dropdown changed after open; stale for this role, but another
            # role may still claim it by index, so defer stopping to cleanup.
            reader = None
        if reader is None:
            reader = pre_by_index.pop(index, None)
        else:
            pre_by_index.pop(index, None)
        if reader is None:
            # Back-to-back MSMF opens on a shared USB 2.0 link fail to grab frames.
            if fresh_opens > 0:
                time.sleep(CAMERA_OPEN_DELAY_SEC)
            fresh_opens += 1
            print(f"Opening {label} camera {index}…")
            reader = CameraReader(index, width=640, height=480, device_id=device_ids.get(role))
            reader.start()
            print(f"{label} {index}: {reader.backend_name or 'opening…'}")
        else:
            if not reader.is_opened():
                print(f"Reopening {label} camera {index} (preview handle was offline)…")
                reader._open_capture()
                reader.start()
            print(f"{label} {index}: {reader.backend_name or '?'} (from setup)")
        by_index[index] = reader
        return reader

    out = {}
    for role, index, label in roles:
        if index is None:
            continue
        out[role] = take_reader(role, index, label)

    in_use = {id(r) for r in by_index.values()}
    stopped = set()
    for leftover in list(pre_by_role.values()) + list(pre_by_index.values()):
        if id(leftover) in in_use or id(leftover) in stopped:
            continue
        leftover.stop()
        stopped.add(id(leftover))
    pre_by_role.clear()
    pre_by_index.clear()
    return out


def run_simple_pupil(choice):
    """L/R pupil tracking only — keeps setup GUI flip/mirror; Front is optional raw preview."""
    left_index = choice.get("left")
    right_index = choice.get("right")
    front_index = choice.get("front")
    if left_index is None and right_index is None:
        print("Need at least one IR eye camera.")
        return
    if "zoom_affects_tracking" not in choice:
        choice["zoom_affects_tracking"] = zoom_affects_tracking_from_setup(_load_camera_setup())

    # Keep role identity — do not rename right→left when left is missing.
    active_eyes = tuple(
        eid for eid, idx in (("left", left_index), ("right", right_index)) if idx is not None
    )

    eye_tracker.set_show_separate_tracking_windows(False)
    eye_tracker.calibrated = False
    eye_tracker.reset_gaze_smoothing()

    role_specs = [
        ("left", left_index, "left IR"),
        ("right", right_index, "right IR"),
        ("front", front_index, "Front"),
    ]
    all_readers = _take_readers_from_choice(choice, role_specs)
    readers = {eid: all_readers[eid] for eid in active_eyes if eid in all_readers}
    front_reader = all_readers.get("front")
    
    # Initialize recording manager
    recorder = RecordingManager(output_dir=os.path.join(ROOT, "recordings"))
    print(f"\n{'='*60}")
    print(f"📹 RECORDING SETUP (Eye Tracking Only)")
    print(f"{'='*60}")
    print(f"Output folder: {recorder.output_dir}")
    print(f"Clip duration: {recorder.clip_duration} seconds")
    print(f"Duration: {recorder.clip_duration}s per preview (left/right/front MP4s)")
    print(f"CLICK THE RECORD BUTTON (top-right) to start recording")
    print(f"1/2/3 = focus preview | Z/X/WASD = zoom/pan on focused panel")
    print(f"{'='*60}\n")

    for eye_id in active_eyes:
        eye_tracker.reset_eye_tracking_state(eye_id)

    # Same column order as the setup GUI: Left | Right | Front
    panel_w, panel_h = 640, 480
    slots = []
    for eye_id in ("left", "right"):
        if eye_id in readers:
            slots.append(("eye", eye_id))
    if front_reader is not None:
        slots.append(("front", None))
    n_slots = max(1, len(slots))
    canvas_w = panel_w * n_slots
    canvas_h = panel_h

    preview_views = preview_views_from_choice(choice)
    slot_roles = [
        (eye_id if kind == "eye" else "front") for kind, eye_id in slots
    ]
    view_state = {
        "focus": slot_roles[0] if slot_roles else "left",
        "available": set(slot_roles),
    }

    layout_state = {"show_previews": True, "eyes": {}, "panel_rects": {}}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, canvas_w, canvas_h)
    boot = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    for col, (kind, eye_id) in enumerate(slots):
        ex = col * panel_w
        if kind == "eye":
            r = readers.get(eye_id)
            label = f"{eye_id} cam{r.index if r else '?'}"
            fill_status_panel(boot, ex, 0, panel_w, panel_h, [label, "starting…"])
        else:
            fill_status_panel(
                boot,
                ex,
                0,
                panel_w,
                panel_h,
                [f"Front cam{front_reader.index}", "starting…"],
                border=(0, 140, 255),
            )
    cv2.imshow(WINDOW_NAME, boot)
    cv2.waitKey(1)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_handler(layout_state))

    def on_view_control(action):
        view_state["focus"] = apply_view_action(
            preview_views, view_state["focus"], action, view_state["available"]
        )

    layout_state["view_control_callback"] = on_view_control

    print("Simple pupil tracking ready.")
    print(
        "Panels (L→R): "
        + " | ".join(
            (f"{eid} cam {readers[eid].index}" if kind == "eye" else f"Front cam {front_reader.index}")
            for kind, eid in slots
        )
    )
    print(
        "Click IR preview to lock eye center | U unlock | REC button | "
        "click L/R/F + WASD/Z+/X-/RST | Q quit"
    )

    try:
        while True:
            view_focus = view_state["focus"]
            layout_state["eyes"] = {}
            layout_state["panel_rects"] = {}
            layout_state["view_focus"] = view_focus
            recording_panels = {}
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            for col, (kind, eye_id) in enumerate(slots):
                ex = col * panel_w
                if kind == "eye":
                    reader = readers.get(eye_id)
                    if reader is None:
                        fill_status_panel(canvas, ex, 0, panel_w, panel_h, [f"{eye_id}", "no reader"])
                        continue
                    snap = reader.snapshot_status()
                    ret, frame = reader.read()
                    if not ret:
                        fill_status_panel(
                            canvas,
                            ex,
                            0,
                            panel_w,
                            panel_h,
                            [f"{eye_id} cam{reader.index}", snap["status"], "no frame"],
                        )
                        continue
                    flip_v = choice["flip_left"] if eye_id == "left" else choice["flip_right"]
                    mirror = choice["mirror_left"] if eye_id == "left" else choice["mirror_right"]
                    view = preview_view_for(preview_views, eye_id)
                    zoom_enabled = zoom_affects_tracking_for(choice, eye_id)
                    tracking_frame, track_flip_v, track_mirror = prepare_eye_tracking_frame(
                        frame, flip_v, mirror, view, zoom_enabled
                    )
                    eye_tracker.process_frame(
                        tracking_frame,
                        eye_id=eye_id,
                        flip_vertical=track_flip_v,
                        flip_horizontal=track_mirror,
                    )
                    eye_frame = eye_tracker.get_preview_frame(eye_id)
                    if eye_frame is None:
                        fill_status_panel(
                            canvas,
                            ex,
                            0,
                            panel_w,
                            panel_h,
                            [f"{eye_id} cam{reader.index}", "processing…"],
                        )
                        continue
                    eye_prev, ox, oy, nw, nh = fit_frame(eye_frame, panel_w, panel_h)
                    role = eye_id
                    # When zoom already fed into tracking, skip display zoom to avoid double-apply.
                    if zoom_enabled:
                        display_panel = eye_prev
                    else:
                        display_panel = render_panel_with_view(eye_prev, preview_views, role)
                    canvas[0:panel_h, ex : ex + panel_w] = display_panel
                    recording_panels[role] = display_panel
                    layout_state["panel_rects"][role] = (ex, 0, panel_w, panel_h)

                    locked = eye_tracker.eye_tracking_states[eye_id].get("sphere_center_locked_2d")
                    border = (0, 255, 0) if locked else (180, 180, 180)
                    if role == view_focus:
                        border = (255, 255, 0)
                    cv2.rectangle(canvas, (ex, 0), (ex + panel_w - 1, panel_h - 1), border, 3 if role == view_focus else 2)
                    label = f"{eye_id} cam{reader.index} {'LOCK' if locked else 'click=center'}"
                    cv2.putText(
                        canvas,
                        label,
                        (ex + 12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    fh, fw = eye_frame.shape[:2]
                    layout_state["eyes"][eye_id] = {
                        "rect": (ex, 0, panel_w, panel_h),
                        "fit": (ox, oy, nw, nh),
                        "src_size": (fw, fh),
                        "view": dict(preview_view_for(preview_views, eye_id)),
                        "zoom_affects_tracking": zoom_enabled,
                    }
                else:
                    snap = front_reader.snapshot_status()
                    ret_f, front_frame = front_reader.read()
                    if not ret_f or front_frame is None:
                        fill_status_panel(
                            canvas,
                            ex,
                            0,
                            panel_w,
                            panel_h,
                            [f"Front cam{front_reader.index}", snap["status"], "no frame"],
                            border=(0, 140, 255),
                        )
                        continue
                    if choice.get("flip_front"):
                        front_frame = cv2.flip(front_frame, 0)
                    if choice.get("mirror_front"):
                        front_frame = cv2.flip(front_frame, 1)
                    front_prev, _, _, _, _ = fit_frame(front_frame, panel_w, panel_h)
                    display_panel = render_panel_with_view(front_prev, preview_views, "front")
                    canvas[0:panel_h, ex : ex + panel_w] = display_panel
                    recording_panels["front"] = display_panel
                    layout_state["panel_rects"]["front"] = (ex, 0, panel_w, panel_h)
                    front_border = (255, 255, 0) if view_focus == "front" else (0, 140, 255)
                    cv2.rectangle(
                        canvas, (ex, 0), (ex + panel_w - 1, panel_h - 1), front_border,
                        3 if view_focus == "front" else 2,
                    )
                    cv2.putText(
                        canvas,
                        f"Front cam{front_reader.index} (preview only)",
                        (ex + 12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

            locked_any = any(
                eye_tracker.eye_tracking_states[eid].get("sphere_center_locked_2d")
                for eid in active_eyes
            )
            
            layout_state["recording_panels"] = recording_panels
            recording_status = recorder.get_status_line()
            hud_lines = [
                recording_status,
                preview_view_hud_line(preview_views, view_focus),
                "Eye tracking only — no ArUco / screen gaze",
                "Eye center: LOCKED (U unlock)"
                if locked_any
                else "Eye center: auto (click IR preview to lock)",
                "Q quit | U unlock | REC button | 1/2/3 focus | Z/X/WASD | P reset focus",
            ]
            draw_hud(canvas, hud_lines, x=12, y=panel_h - 120)
            
            # Draw record + view-control buttons
            button_x = canvas_w - 200
            button_y = 20
            button_rect = draw_record_button(canvas, recorder, button_x, button_y)
            layout_state["record_button_rect"] = button_rect
            layout_state["view_control_buttons"] = draw_view_controls(
                canvas, view_focus, view_state["available"], x=12, y=12
            )
            
            # Set up record button callback
            def toggle_recording():
                print(f"\n[BUTTON CLICKED] Record button pressed!")
                if recorder.is_recording():
                    recorder.stop_recording()
                    print("Recording stopped via button.")
                else:
                    panels = layout_state.get("recording_panels") or {}
                    if not panels:
                        print("No preview panels ready to record.")
                        return
                    sizes = {role: _panel_size_bgr(panel) for role, panel in panels.items()}
                    success = recorder.start_recording(sizes, fps=30)
                    if success:
                        print(f"Recording L/R/F previews via button!")
                        print(f"   Saving to: {recorder.output_dir}")
                    else:
                        print(f"Failed to start recording!")
            
            layout_state["record_button_callback"] = toggle_recording

            if recorder.is_recording() and recording_panels:
                recorder.add_frames(recording_panels)

            cv2.imshow(WINDOW_NAME, canvas)
            key = poll_key()
            if key == ord("q"):
                break
            if key == ord("r"):
                if recorder.is_recording():
                    recorder.stop_recording()
                    print("Recording stopped manually.")
                else:
                    if not recording_panels:
                        print("No preview panels ready.")
                    else:
                        sizes = {role: _panel_size_bgr(p) for role, p in recording_panels.items()}
                        if recorder.start_recording(sizes, fps=30):
                            print(f"Recording L/R/F previews → {recorder.output_dir}")
            else:
                new_focus, handled = handle_preview_view_key(
                    preview_views, view_focus, key, view_state["available"]
                )
                if handled:
                    view_state["focus"] = new_focus
                elif key == ord("u"):
                    unlock_eye_sphere_centers(active_eyes)
    finally:
        # Stop recording if active
        recorder.stop_recording()
        
        stopped = set()
        for reader in list(readers.values()) + ([front_reader] if front_reader else []):
            if reader is None or id(reader) in stopped:
                continue
            reader.stop()
            stopped.add(id(reader))
        cv2.destroyAllWindows()


def run(choice):
    left_index = choice["left"]
    right_index = choice.get("right")
    front_index = choice["front"]
    if left_index is None and right_index is None:
        print("Need at least one IR eye camera.")
        return
    if front_index is None:
        print("Need a front camera.")
        return
    if "zoom_affects_tracking" not in choice:
        choice["zoom_affects_tracking"] = zoom_affects_tracking_from_setup(_load_camera_setup())

    if left_index is None:
        left_index, right_index = right_index, None

    active_eyes = tuple(
        eid for eid, idx in (("left", left_index), ("right", right_index)) if idx is not None
    )

    screen_w, screen_h, win_x, win_y = sp.get_window_placement()
    width_mm, height_mm = sp.get_screen_mm(screen_w, screen_h)
    tracker = sp.ScreenPoseTracker(screen_w, screen_h, width_mm, height_mm, hfov_deg=DEFAULT_HFOV)
    heatmap = GazeHeatmap(screen_w, screen_h)
    scale_calib = GazeScaleCalib()

    eye_tracker.set_show_separate_tracking_windows(False)
    eye_tracker.calibrated = False
    eye_tracker.reset_gaze_smoothing()

    all_readers = _take_readers_from_choice(
        choice,
        [
            ("left", left_index, "left IR"),
            ("right", right_index, "right IR"),
            ("front", front_index, "Front"),
        ],
    )
    readers = {eid: all_readers[eid] for eid in active_eyes if eid in all_readers}
    front_reader = all_readers["front"]

    for eye_id in active_eyes:
        eye_tracker.reset_eye_tracking_state(eye_id)

    layout_state = {"show_previews": True, "eyes": {}}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, win_x, win_y)
    cv2.resizeWindow(WINDOW_NAME, screen_w, screen_h)
    # Paint markers/HUD immediately — do not leave a blank buffer while IR processing starts.
    boot = heatmap.render_bgr()
    tracker.markers.paste_on(boot)
    draw_center_calib_target(boot, screen_w, screen_h)
    x1, y1, _, _ = tracker.markers.preview_rect()
    draw_hud(
        boot,
        [
            "GazeScreen3D starting…",
            "Waiting for camera frames",
            "Q quit | C | arrows | E | [ ] , . | U M V K | -/+ FOV",
        ],
        x=x1,
        y=36,
    )
    cv2.imshow(WINDOW_NAME, boot)
    cv2.waitKey(1)
    maximize_cv_window(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_handler(layout_state))

    show_previews = True
    pending_edge = None
    # Zoom/pan is set in the setup GUI and persisted in camera_setup.json — no live HUD here.
    preview_views = preview_views_from_choice(choice)
    
    print("GazeScreen3D ready.")
    print("1) Point front camera at this window until Pose OK (4/4 ArUco corners)")
    print("2) Click IR preview to lock eye center if the yellow circle drifts")
    print("3) Look at the CYAN CROSS at MONITOR CENTER (not the Front preview) and press C")
    print("4) Calibrate edges: look at each cross, press arrow keys (~12 frames each)")
    print("5) Fine-tune: [ ] vertical   , . horizontal")
    print("6) Gaze heatmap follows ray ∩ ArUco plane")
    print("Q quit | C | arrows | E reset | [ ] , . | U M V K | -/+ FOV")
    print("Preview zoom/pan: set in setup GUI (stored in camera_setup.json)")

    try:
        while True:
            layout_state["show_previews"] = show_previews
            layout_state["eyes"] = {}

            # Front + ArUco first so the window stays usable even if IR processing is slow.
            front_display = None
            front_snap = front_reader.snapshot_status()
            ret_f, front_frame = front_reader.read()
            if ret_f and front_frame is not None:
                if choice["flip_front"]:
                    front_frame = cv2.flip(front_frame, 0)
                if choice["mirror_front"]:
                    front_frame = cv2.flip(front_frame, 1)
                fh, fw = front_frame.shape[:2]
                eye_tracker.configure_external_viewport(fw, fh)
                front_display = tracker.process(front_frame)

            # IR eyes (updates preview_frame for PiP / gaze)
            for eye_id, reader in readers.items():
                ret, frame = reader.read()
                if not ret:
                    continue
                flip_v = choice["flip_left"] if eye_id == "left" else choice["flip_right"]
                mirror = choice["mirror_left"] if eye_id == "left" else choice["mirror_right"]
                view = preview_view_for(preview_views, eye_id)
                zoom_enabled = zoom_affects_tracking_for(choice, eye_id)
                tracking_frame, track_flip_v, track_mirror = prepare_eye_tracking_frame(
                    frame, flip_v, mirror, view, zoom_enabled
                )
                eye_tracker.process_frame(
                    tracking_frame,
                    eye_id=eye_id,
                    flip_vertical=track_flip_v,
                    flip_horizontal=track_mirror,
                )

            gaze = eye_tracker.refresh_combined_gaze(active_eyes)
            raw_gaze = eye_tracker.get_raw_combined_gaze_dir()

            if pending_edge is not None and raw_gaze is not None:
                pending_edge, ready = _append_edge_capture(pending_edge, raw_gaze)
                if ready:
                    _finish_edge_capture(
                        pending_edge, scale_calib, tracker, width_mm, height_mm, screen_w, screen_h
                    )
                    pending_edge = None

            hit = None
            if (
                eye_tracker.calibrated
                and gaze is not None
                and tracker.ready
                and tracker._rotation is not None
                and tracker._translation is not None
            ):
                hit = gaze_to_screen(
                    gaze,
                    eye_tracker.R_gaze_to_cam,
                    tracker._rotation,
                    tracker._translation,
                    width_mm,
                    height_mm,
                    screen_w,
                    screen_h,
                    scale_x_left=scale_calib.scale_x_left,
                    scale_x_right=scale_calib.scale_x_right,
                    scale_y_up=scale_calib.scale_y_up,
                    scale_y_down=scale_calib.scale_y_down,
                )
                if hit is not None:
                    heatmap.add_hit(hit["u"], hit["v"], hit["on_screen"])
                    if front_display is not None:
                        uv = project_gaze_to_front_pixels(
                            gaze,
                            eye_tracker.R_gaze_to_cam,
                            front_display.shape[1],
                            front_display.shape[0],
                            eye_tracker.EXT_FX,
                            eye_tracker.EXT_FY,
                            eye_tracker.EXT_CX,
                            eye_tracker.EXT_CY,
                            scale_x_left=scale_calib.scale_x_left,
                            scale_x_right=scale_calib.scale_x_right,
                            scale_y_up=scale_calib.scale_y_up,
                            scale_y_down=scale_calib.scale_y_down,
                        )
                        if uv is not None:
                            cv2.circle(
                                front_display,
                                (int(uv[0]), int(uv[1])),
                                7,
                                (0, 0, 255),
                                -1,
                            )

            # --- Compose (always: markers + HUD even with no camera frames) ---
            canvas = heatmap.render_bgr()
            tracker.markers.paste_on(canvas)

            x1, y1, x2, y2 = tracker.markers.preview_rect()
            
            if show_previews:
                pw, ph = max(160, x2 - x1), max(120, min(360, (y2 - y1) // 2))
                total_preview_h = ph + 8 + min(160, y2 - (y1 + ph + 8))
                preview_canvas = np.zeros((total_preview_h, pw, 3), dtype=np.uint8)
                
                if front_display is not None:
                    front_base, _, _, _, _ = fit_frame(front_display, pw, ph)
                    front_panel = render_panel_with_view(front_base, preview_views, "front")
                    preview_canvas[0:ph, 0:pw] = front_panel
                    cv2.rectangle(preview_canvas, (0, 0), (pw - 1, ph - 1), (180, 180, 180), 1)
                    if not eye_tracker.calibrated:
                        draw_front_preview_not_for_c(preview_canvas, 0, 0, pw, ph)
                    cv2.putText(
                        preview_canvas,
                        "Front",
                        (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    canvas[y1 : y1 + ph, x1 : x1 + pw] = preview_canvas[0:ph, 0:pw]
                else:
                    fill_status_panel(
                        preview_canvas,
                        0,
                        0,
                        pw,
                        ph,
                        [
                            f"Front cam{front_reader.index}",
                            front_snap["status"],
                            "no frame",
                        ],
                        border=(0, 140, 255),
                    )
                    canvas[y1 : y1 + ph, x1 : x1 + pw] = preview_canvas[0:ph, 0:pw]

                eye_y_canvas = y1 + ph + 8
                eye_y_preview = ph + 8
                slot_w = (pw - 8) // max(1, len(active_eyes))
                for i, eye_id in enumerate(active_eyes):
                    ex_canvas = x1 + i * (slot_w + 8)
                    ex_preview = i * (slot_w + 8)
                    eh = min(160, y2 - eye_y_canvas)
                    if eh < 40:
                        break
                    eye_frame = eye_tracker.get_preview_frame(eye_id)
                    reader = readers.get(eye_id)
                    if eye_frame is None:
                        status = reader.snapshot_status()["status"] if reader else "OFFLINE"
                        fill_status_panel(
                            preview_canvas,
                            ex_preview,
                            eye_y_preview,
                            slot_w,
                            eh,
                            [eye_id, status, "no frame"],
                        )
                        canvas[eye_y_canvas : eye_y_canvas + eh, ex_canvas : ex_canvas + slot_w] = \
                            preview_canvas[eye_y_preview : eye_y_preview + eh, ex_preview : ex_preview + slot_w]
                        continue
                    eye_base, ox, oy, nw, nh = fit_frame(eye_frame, slot_w, eh)
                    zoom_enabled = zoom_affects_tracking_for(choice, eye_id)
                    # Tracking already zoomed when flag is set — skip display zoom.
                    if zoom_enabled:
                        eye_panel = eye_base
                    else:
                        eye_panel = render_panel_with_view(eye_base, preview_views, eye_id)
                    preview_canvas[eye_y_preview : eye_y_preview + eh, ex_preview : ex_preview + slot_w] = eye_panel

                    locked = eye_tracker.eye_tracking_states[eye_id].get("sphere_center_locked_2d")
                    border = (0, 255, 0) if locked else (180, 180, 180)
                    cv2.rectangle(
                        preview_canvas,
                        (ex_preview, eye_y_preview),
                        (ex_preview + slot_w, eye_y_preview + eh),
                        border,
                        2,
                    )
                    label = f"{eye_id} {'LOCK' if locked else 'click=center'}"
                    cv2.putText(
                        preview_canvas,
                        label,
                        (ex_preview + 6, eye_y_preview + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    canvas[eye_y_canvas : eye_y_canvas + eh, ex_canvas : ex_canvas + slot_w] = \
                        preview_canvas[eye_y_preview : eye_y_preview + eh, ex_preview : ex_preview + slot_w]
                    
                    fh, fw = eye_frame.shape[:2]
                    layout_state["eyes"][eye_id] = {
                        "rect": (ex_canvas, eye_y_canvas, slot_w, eh),
                        "fit": (ox, oy, nw, nh),
                        "src_size": (fw, fh),
                        "view": dict(preview_view_for(preview_views, eye_id)),
                        "zoom_affects_tracking": zoom_enabled,
                    }

            locked_any = any(
                eye_tracker.eye_tracking_states[eid].get("sphere_center_locked_2d")
                for eid in active_eyes
            )
            lines = [
                f"Front {front_snap['fps']:.0f} fps  "
                f"[{front_snap['status']}]  HFOV {tracker.hfov_deg:.0f}",
                tracker.status_line() if hasattr(tracker, "status_line") else "",
            ]
            if eye_tracker.calibrated:
                lines.append("Gaze to cam: calibrated (C)")
                lines.append(scale_calib.status_line())
            else:
                lines.append("Gaze to cam: guarda croce CIANO al centro MONITOR, premi C")
            lines.append(
                "Eye center: LOCKED (U unlock)" if locked_any else "Eye center: auto (click IR preview to lock)"
            )

            if pending_edge is not None and pending_edge.get("edge"):
                n = len(pending_edge.get("samples", []))
                need = pending_edge.get("need", EDGE_CALIB_FRAMES)
                lines.append(f"Edge calib {pending_edge['edge']}: {n}/{need} frames...")
            elif hit is not None:
                state = "ON screen" if hit["on_screen"] else "OFF screen"
                lines.append(f"Hit {state}  ({hit['u']:.0f}, {hit['v']:.0f}) px")
            elif not tracker.ready:
                if tracker.corner_ids_found and len(tracker.corner_ids_found) < sp.MIN_MARKERS_FOR_POSE:
                    lines.append("Waiting for 4/4 ArUco corners...")
                else:
                    lines.append("Waiting for ArUco screen pose...")
            elif not eye_tracker.calibrated:
                lines.append("Waiting for C calibration...")
            else:
                lines.append("No gaze / no intersection")

            lines.append("Q quit | C | arrows | E | [ ] , . | U M V K | -/+ FOV")
            
            draw_hud(canvas, [ln for ln in lines if ln], x=x1, y=36)

            if eye_tracker.calibrated:
                draw_edge_targets(canvas, screen_w, screen_h, scale_calib.edges_done)
            else:
                draw_center_calib_target(canvas, screen_w, screen_h)

            cv2.imshow(WINDOW_NAME, canvas)

            key = poll_key()
            if key == ord("q"):
                break
            if key == ord("c"):
                calibrate_gaze(active_eyes, scale_calib)
            elif key == ord("e"):
                scale_calib.clear_edges()
                pending_edge = None
                print("Edge scales reset. C kept.")
            elif key == ord("["):
                scale_calib.nudge_vertical(-EDGE_NUDGE)
                print(f"Vertical -  {scale_calib.scales_summary()}")
            elif key == ord("]"):
                scale_calib.nudge_vertical(EDGE_NUDGE)
                print(f"Vertical +  {scale_calib.scales_summary()}")
            elif key == ord(","):
                scale_calib.nudge_horizontal(-EDGE_NUDGE)
                print(f"Horizontal -  {scale_calib.scales_summary()}")
            elif key == ord("."):
                scale_calib.nudge_horizontal(EDGE_NUDGE)
                print(f"Horizontal +  {scale_calib.scales_summary()}")
            elif key == ord("u"):
                unlock_eye_sphere_centers(active_eyes)
            elif key == ord("m"):
                print(f"Markers {'ON' if tracker.markers.toggle() else 'OFF'}")
            elif key == ord("v"):
                show_previews = not show_previews
                print(f"Previews {'ON' if show_previews else 'OFF'}")
            elif key == ord("k"):
                heatmap.clear()
                print("Heatmap cleared")
            elif key in (ord("-"), ord("_")):
                tracker.set_hfov(tracker.hfov_deg - 2.0)
                print(f"HFOV {tracker.hfov_deg:.0f}")
            elif key in (ord("="), ord("+")):
                tracker.set_hfov(tracker.hfov_deg + 2.0)
                print(f"HFOV {tracker.hfov_deg:.0f}")
            elif key == ord("0"):
                tracker.set_hfov(DEFAULT_HFOV)
                print(f"HFOV reset {tracker.hfov_deg:.0f}")
            elif key == ord("h"):
                print(
                    "C = center | arrows = 4 edges | E reset | [ ] vertical , . horizontal | "
                    "click IR = lock eye center | U unlock | zoom set in setup GUI"
                )
            else:
                edge = _arrow_to_edge(key)
                if edge is not None:
                    if not eye_tracker.calibrated:
                        print("Press C at screen center before edge arrows.")
                    elif not tracker.ready or tracker._rotation is None:
                        print("Wait for ArUco pose before edge arrows.")
                    else:
                        pending_edge = _begin_edge_capture(edge)
                        print(
                            f"Capturing {edge} ({EDGE_CALIB_FRAMES} frames) — "
                            f"keep looking at the {edge} cross..."
                        )
    finally:
        stopped = set()
        for reader in list(readers.values()) + [front_reader]:
            if id(reader) in stopped:
                continue
            reader.stop()
            stopped.add(id(reader))
        cv2.destroyAllWindows()


def main():
    choice = selection_gui()
    if not choice:
        return
    if choice.get("left") is None and choice.get("right") is None:
        return
    if choice.get("mode") == "eyes":
        run_simple_pupil(choice)
    else:
        run(choice)


if __name__ == "__main__":
    main()
