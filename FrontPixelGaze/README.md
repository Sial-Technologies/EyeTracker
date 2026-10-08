# FrontPixelGaze

Minimal **glasses IR → front-camera pixel** gaze demo. No screen heatmap, no ArUco, no live ray∩plane.

It uses the same idea as Orlosky’s VR remapping: learn an empirical map from eye direction to a shared frame. Here that frame is **front-camera pixels**, learned by click calibration with a **steady head**.

## How click-map works

A calibration point does **not** need MultiCamGaze screen/ArUco geometry. It only needs to be:

1. Something you are actually looking at, and
2. Visible in the front camera so you can click its pixel.

A sticky note, lamp, or a mark on a monitor all work. A flat monitor at fixed distance is convenient because the map is **depth-dependent** (eye offset from the front cam). Keep a similar viewing distance at runtime.

Per sample (head steady, IR rigid on glasses):

```text
(D from Orlosky left IR)  +  (u, v) clicked on the front preview
```

After ≥5 points, fit an affine map:

```text
[u, v]^T = A @ [yaw, pitch, 1]^T
```

Runtime: Orlosky `D` → OpenCV convention → map → crosshair on the undistorted front image.

This is **not** “pixel on the OS monitor as a calibrated plane.” That is MultiCamGaze’s heatmap path. This project only answers: *where in the front camera image does my current gaze correspond (for scenes like calibration).*

## Intrinsics

OpenCV `K` / distortion for `front` and `left_eye` were **copied** from MultiCamGaze3D:

- `calib/intrinsics/front.npz`
- `calib/intrinsics/left_eye.npz`

Front frames are undistorted with `front.npz`. Left `.npz` is loaded for consistency; Orlosky still uses its virtual-sphere unprojection (same as the original tracker).

## Setup

```bash
cd FrontPixelGaze
pip install -e .
# list cameras + stable device_id strings (preferred over raw indices):
python -m front_pixel_gaze.win_cameras
# edit config/camera_setup.json (device_id / index, flip/mirror)
front-pixel-gaze
# or:
python -m front_pixel_gaze.app
```

On Windows, `device_id` is resolved to the live MSMF index each launch (same idea as MultiCamGaze). Indices alone drift when devices are plugged in a different order.

## Usage

1. Wear glasses; sit still relative to the look-at surface.
2. **Warmup** — look around until the eye sphere center/radius adapt (`ready=True`).
   Uses MultiCamGaze’s IR tracker with **natural auto eye-center only** (no
   `eye_center_ir_px` lock).
3. **Calib** — look at a physical point → **left-click** it on the front image → hold ~0.5s. Repeat ≥5 points across the FOV.
4. Press **F** to fit + save `calib/front_uv_map.json` and enter track mode.
5. **Track** — green crosshair = predicted front UV.
6. Press **H** for fullscreen **heatmap**: corner ArUco 0–3 on the monitor; calibrated
   front gaze is tested against the screen quad seen by the front camera (homography).
   Hits accumulate as a JET heatmap. A small tracking window stays open with front + IR.

| Key | Action |
|-----|--------|
| Right-click | Lock eyeball center (IR PiP pixel, or current pupil if outside PiP) |
| U | Unlock eyeball center (resume auto estimate) |
| Left click | Start sample capture at that front pixel (calib mode) |
| F | Fit map from samples, save, track |
| H | Fullscreen ArUco heatmap (needs fitted map) |
| R | Clear heatmap accumulation |
| C | Clear click samples |
| T | Track only (close heatmap window) |
| W | Back to calib / warmup |
| Q / Esc | Quit |

Heatmap geometry is **2D**: `front UV ∩ ArUco screen quad → screen pixels`. No
`eye_center_front_mm` / ray∩plane (that remains MultiCamGaze’s pure-3D path).

## Layout

```text
FrontPixelGaze/
  config/camera_setup.json
  calib/intrinsics/{front,left_eye}.npz
  calib/front_uv_map.json          # written by F (gitignored)
  front_pixel_gaze/
    app.py                         # UI
    gaze_eye_tracker.py            # MultiCamGaze3D tracking/eye_tracker.py (vendored)
    pixel_remap.py                 # MultiCamGaze sensor remap helper
    orlosky_eye.py                 # thin wrapper (raw frame + flip flags + K)
    uv_map.py                      # yaw/pitch → UV affine
    screen_aruco.py                # corner ArUco + front→screen homography
    heatmap.py                     # accumulate / overlay
    cameras.py / intrinsics.py
```




## Out of scope

- Screen heatmap / live monitor UV (see MultiCamGaze3D)
- Persisted `eye_center_ir_px` from config (optional right-click lock only, not auto)
- Right eye / stereo
- Importing `multcam_gaze`
- Re-running chessboard intrinsic calibration
