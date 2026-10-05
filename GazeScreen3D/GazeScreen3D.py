"""
GazeScreen3D — eye gaze ∩ ArUco screen plane → monitor pixel + heatmap.

Pipeline:
  1. IR eye camera(s) → 3D gaze direction
  2. C at screen center → R_gaze_to_cam (eye space → front camera)
  3. Optional arrow keys at edges → yaw/pitch scales (refine eye→cam only)
  4. Front camera ArUco → screen plane in front-camera coords
  5. Ray from camera origin along rotated gaze ∩ plane → screen pixel

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

from camera_io import CAMERA_CAPTURE_MODES, CameraReader, win_cameras
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
    """Persist role → {device_id, index, flip, mirror} to camera_setup.json."""
    try:
        with open(CAMERA_SETUP_PATH, "w", encoding="utf-8") as f:
            json.dump(setup, f, indent=2)
            f.write("\n")
    except OSError as exc:
        print(f"Could not save camera setup: {exc}")


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
                try:
                    photo = _bgr_to_photoimage(tk, frame, PREVIEW_W, PREVIEW_H)
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
        "After Start: C = screen center calib | click IR = lock eye center | U unlock | Q quit",
        font=("Arial", 9),
        justify="center",
    ).pack(pady=6)

    choice = {}

    def start():
        left_i = _parse_camera_selection(left_var.get())
        right_i = _parse_camera_selection(right_var.get())
        front_i = _parse_camera_selection(front_var.get())
        if left_i is None and right_i is None:
            status_lbl.config(text="Need at least one IR eye camera (Left or Right).")
            return
        if front_i is None:
            status_lbl.config(text="Need a Front camera.")
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
            }
        _save_camera_setup(setup)

        choice["left"] = left_i
        choice["right"] = right_i
        choice["front"] = front_i
        choice["flip_left"] = flip_left.get()
        choice["flip_right"] = flip_right.get()
        choice["mirror_left"] = mirror_left.get()
        choice["mirror_right"] = mirror_right.get()
        choice["flip_front"] = flip_front.get()
        choice["mirror_front"] = mirror_front.get()
        choice["device_ids"] = {
            role: _stable_device_id(entry["device_id"]) if entry else None
            for role, entry in setup.items()
        }
        # Handoff as index → reader (run() expects that shape).
        handed = {}
        for role, reader in readers.items():
            if reader is not None:
                handed[reader.index] = reader
            readers[role] = None
        choice["readers"] = handed
        root.destroy()

    def on_close():
        stop_previews()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    tk.Button(root, text="Start", font=("Arial", 11, "bold"), command=start).pack(pady=10)

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


def make_mouse_handler(layout_state):
    """Map clicks on embedded IR previews to eye-frame coordinates."""

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not layout_state.get("show_previews", True):
            return
        for eye_id, slot in layout_state.get("eyes", {}).items():
            x0, y0, w, h = slot["rect"]
            if not (x0 <= x < x0 + w and y0 <= y < y0 + h):
                continue
            ox, oy, nw, nh = slot["fit"]
            src_w, src_h = slot["src_size"]
            local_x = x - x0 - ox
            local_y = y - y0 - oy
            if not (0 <= local_x < nw and 0 <= local_y < nh):
                return
            frame_x = int(local_x * src_w / nw)
            frame_y = int(local_y * src_h / nh)
            lock_eye_sphere_center(eye_id, frame_x, frame_y)
            return

    return on_mouse


def draw_hud(canvas, lines, x=16, y=28):
    for text in lines:
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        y += 24
    return y


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

    pre_readers = choice.pop("readers", None) or {}
    device_ids = choice.get("device_ids") or {}
    by_index = {}

    def take_reader(role, index, label):
        if index in by_index:
            return by_index[index]
        reader = pre_readers.pop(index, None)
        if reader is None:
            print(f"Opening {label} camera {index}…")
            reader = CameraReader(
                index,
                width=640,
                height=480,
                device_id=device_ids.get(role),
            )
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

    readers = {}
    for eye_id, index in (("left", left_index), ("right", right_index)):
        if index is None:
            continue
        readers[eye_id] = take_reader(eye_id, index, f"{eye_id} IR")
        eye_tracker.reset_eye_tracking_state(eye_id)

    front_reader = take_reader("front", front_index, "Front")

    for leftover in pre_readers.values():
        leftover.stop()
    pre_readers.clear()

    layout_state = {"show_previews": True, "eyes": {}}
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(WINDOW_NAME, win_x, win_y)
    cv2.resizeWindow(WINDOW_NAME, screen_w, screen_h)
    # Create the window handle, then maximize (resizeWindow alone stays small on Windows).
    cv2.imshow(WINDOW_NAME, np.zeros((screen_h, screen_w, 3), dtype=np.uint8))
    cv2.waitKey(1)
    maximize_cv_window(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, make_mouse_handler(layout_state))

    show_previews = True
    pending_edge = None
    print("GazeScreen3D ready.")
    print("1) Point front camera at this window until Pose OK (4/4 ArUco corners)")
    print("2) Click IR preview to lock eye center if the yellow circle drifts")
    print("3) Look at the CYAN CROSS at MONITOR CENTER (not the Front preview) and press C")
    print("4) Calibrate edges: look at each cross, press ↑ ↓ ← → (~12 frames each)")
    print("5) Fine-tune: [ ] vertical   , . horizontal")
    print("6) Gaze heatmap follows ray ∩ ArUco plane")
    print("Q quit | C | arrows | E reset | [ ] , . | U M V K -/+ FOV")

    try:
        while True:
            layout_state["show_previews"] = show_previews
            layout_state["eyes"] = {}

            # --- IR eyes ---
            for eye_id, reader in readers.items():
                ret, frame = reader.read()
                if not ret:
                    continue
                flip_v = choice["flip_left"] if eye_id == "left" else choice["flip_right"]
                mirror = choice["mirror_left"] if eye_id == "left" else choice["mirror_right"]
                eye_tracker.process_frame(
                    frame,
                    eye_id=eye_id,
                    flip_vertical=flip_v,
                    flip_horizontal=mirror,
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

            # --- Front + ArUco ---
            front_display = None
            ret_f, front_frame = front_reader.read()
            if ret_f and front_frame is not None:
                if choice["flip_front"]:
                    front_frame = cv2.flip(front_frame, 0)
                if choice["mirror_front"]:
                    front_frame = cv2.flip(front_frame, 1)
                fh, fw = front_frame.shape[:2]
                eye_tracker.configure_external_viewport(fw, fh)
                front_display = tracker.process(front_frame)

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

            # --- Compose ---
            canvas = heatmap.render_bgr()
            tracker.markers.paste_on(canvas)

            x1, y1, x2, y2 = tracker.markers.preview_rect()
            if show_previews and front_display is not None:
                pw, ph = max(160, x2 - x1), max(120, min(360, (y2 - y1) // 2))
                preview, _, _, _, _ = fit_frame(front_display, pw, ph)
                canvas[y1 : y1 + ph, x1 : x1 + pw] = preview
                cv2.rectangle(canvas, (x1, y1), (x1 + pw, y1 + ph), (180, 180, 180), 1)
                if not eye_tracker.calibrated:
                    draw_front_preview_not_for_c(canvas, x1, y1, pw, ph)
                cv2.putText(
                    canvas,
                    "Front",
                    (x1 + 8, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                eye_y = y1 + ph + 8
                slot_w = (pw - 8) // max(1, len(active_eyes))
                for i, eye_id in enumerate(active_eyes):
                    eye_frame = eye_tracker.get_preview_frame(eye_id)
                    if eye_frame is None:
                        continue
                    ex = x1 + i * (slot_w + 8)
                    eh = min(160, y2 - eye_y)
                    if eh < 40:
                        break
                    eye_prev, ox, oy, nw, nh = fit_frame(eye_frame, slot_w, eh)
                    canvas[eye_y : eye_y + eh, ex : ex + slot_w] = eye_prev

                    locked = eye_tracker.eye_tracking_states[eye_id].get("sphere_center_locked_2d")
                    border = (0, 255, 0) if locked else (180, 180, 180)
                    cv2.rectangle(canvas, (ex, eye_y), (ex + slot_w, eye_y + eh), border, 2)
                    label = f"{eye_id} {'LOCK' if locked else 'click=center'}"
                    cv2.putText(
                        canvas,
                        label,
                        (ex + 6, eye_y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    fh, fw = eye_frame.shape[:2]
                    layout_state["eyes"][eye_id] = {
                        "rect": (ex, eye_y, slot_w, eh),
                        "fit": (ox, oy, nw, nh),
                        "src_size": (fw, fh),
                    }

            locked_any = any(
                eye_tracker.eye_tracking_states[eid].get("sphere_center_locked_2d")
                for eid in active_eyes
            )
            lines = [
                f"Front {front_reader.snapshot_status()['fps']:.0f} fps  "
                f"[{front_reader.snapshot_status()['status']}]  HFOV {tracker.hfov_deg:.0f}",
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
                    "click IR = lock eye center | U unlock"
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
    if not choice or choice.get("front") is None:
        return
    if choice.get("left") is None and choice.get("right") is None:
        return
    run(choice)


if __name__ == "__main__":
    main()
