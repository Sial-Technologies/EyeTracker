# Left-eye heatmap MVP

Minimal path: **front + left only**. Head may move (live ArUco each frame).
Heatmap is **pure 3D**: gaze ray ∩ live ArUco screen plane (no yaw/pitch→UV shim).

## Prerequisites

- Phase 0 intrinsics for `front` (required) and `left_eye` (used for IR unprojection
  when present; otherwise datasheet FOV):

```bash
cd MultiCamGaze3D
pip install -e ".[dev]"
multcam-intrinsics --role front
multcam-intrinsics --role left_eye
```

- `config/camera_setup.json` with working `front` and `left` indices.
- **Metric screen size** in `config/screen.json` (`diagonal_inches`, e.g. `27`)
  or CLI `--diagonal-inches 27` / `--width-mm`+`--height-mm`. This is the ArUco
  PnP scale — OS/GDI size is shown for comparison only and is **not** trusted.
- Optional but recommended: `left.eye_center_front_mm` (eyeball vs front cam, mm).
- Optional but recommended: `left.eye_center_ir_px` (locked Orlosky 2D center).
  Set zoom/pan first in `multcam-preview`, look into the IR camera, left-click the
  left panel to lock at the pupil, Enter to save. Right-click unlocks.
- Left IR mount: set `"flip": true` if the camera is upside-down. When
  `zoom_affects_tracking` is true, flip/mirror is applied **before** zoom/pan so
  pan offsets match the upright frame. Sensor remap undoes that flip for `K`;
  the Orlosky→OpenCV Y conversion then **skips** its extra Y negate so pitch is
  not double-flipped / crushed. Look-at grid bottom sits just above the PiPs
  (`max(margin, pip_clear)`, not margin+pip).

## Run

```bash
multcam-left-heatmap
# required once: set config/screen.json diagonal_inches, or:
#   multcam-left-heatmap --diagonal-inches 27
# optional: --grid 16 --capture-seconds 4 --settle-seconds 1
# optional: --max-rms-deg 2.5 --max-angle-deg 6 --min-samples 45
# optional: --min-aruco-markers 4 --max-aruco-reproj-px 8 --max-pfront-z-range-mm 60
# optional: --width-mm 597 --height-mm 336
# optional: --dump-samples calib/left_samples.npz
# after a successful solve, skip the grid next time:
#   multcam-left-heatmap --heatmap-only
#   multcam-left-heatmap --heatmap-only --load-calib calib/device_calibration.json
multcam-analyze-left-samples calib/left_samples.npz
# optional: refine eye_center_front_mm from near+far dumps (then edit camera_setup):
#   multcam-refine-eye-center calib/left_samples_near.npz calib/left_samples.npz
#   multcam-refine-eye-center ... --write-setup
```

Each successful look-at solve writes **`calib/device_calibration.json`**
(`front_from_left_eye` + yaw/pitch scales). `--heatmap-only` reloads that file.

On-screen status shows **cam↔screen** distance (m / mm) from live ArUco PnP whenever
markers are locked — sanity-check against a tape measure.

Fullscreen window shows corner ArUco (IDs 0–3), a red look-at target, and a
bottom-center row with **left IR** (flip/zoom/pan from `camera_setup.json`) plus
front preview. Status text sits below the top markers. ArUco is drawn last so UI
never covers markers.

| Key / mouse | Action |
|-------------|--------|
| SPACE | Start timed capture of current target (needs warmup + ArUco + pupil) |
| N | Skip current target (glint / bad pupil); cancel in-progress capture too |
| H | Enter / resume heatmap mode (solves if samples ready) |
| R | Clear heatmap accumulation |
| Right-click (heatmap) | Ground-truth look-at vs predicted gaze (Δu/Δv px + mm; console + overlay) |
| C (heatmap) | Clear on-screen debug marks (dump file keeps all clicks) |
| Q / Esc | Quit |

With `--dump-samples path.npz`, look-at samples are written on solve; each heatmap
right-click is appended into the same file (`debug_truth_uv`, `debug_pred_uv`,
`debug_du_px`, …) and flushed again on exit. Inspect with
`multcam-analyze-left-samples path.npz`.

### Warmup (eyeball size)

Without `eye_center_ir_px`, Orlosky adapts eyeball radius only after ~30 eye-center
samples with a confident pupil. With a locked IR center, that count is skipped —
still **look around** so radius can grow from pupil distance. Calibration starts in
**warmup** until the left PiP shows `eye READY` / radius OK. SPACE is blocked until
then.

### Per-point capture

Each SPACE: settle (`--settle-seconds`, default 0.7s) then hold
(`--capture-seconds`, default 3.0s). Needs ≥ `--min-samples` valid frames.
Per-frame append requires pupil fill ratio ≥ `--min-pupil-confidence`
(default `PUPIL_CONFIDENCE_THRESHOLD` = 0.85); SPACE itself is not blocked by
confidence. Abort if:

- gaze direction RMS / max angular spread exceeds limits, or
- ArUco markers drop below `--min-aruco-markers` (default 4), or
- mean ArUco reprojection exceeds `--max-aruco-reproj-px`, or
- `P_front` Z range within the burst exceeds `--max-pfront-z-range-mm`

Accepted bursts trim extremes (`--trim-frac`) then average. Capture logs
reproj / marker count / `P_front` per accepted point; `--dump-samples` stores them
for `multcam-analyze-left-samples` (plane distances + Z span).

**N** abandons the current look-at (and cancels an in-progress capture) without
adding a sample - use this when pupil dies from glint near the IR camera. Solve
still needs ≥ 3 kept points; skipping the whole death-zone row is fine.

After all grid points are captured or skipped, the app solves
\(T_{\mathrm{front}\leftarrow\mathrm{left}}\) and writes `calib/device_calibration.json`.

## World / metric geometry

In this MVP **world ≡ front camera**. With a correct panel size, live ArUco PnP gives
a metric \(T_{\mathrm{front}\leftarrow\mathrm{screen}}\): the front camera’s pose relative
to the monitor (including **cam↔screen** distance shown on the HUD).

| Ingredient | Role |
|------------|------|
| `screen.json` diagonal / W×H mm | Absolute scale of the screen plane |
| Corner ArUco (fullscreen) | Pose of that plane in the front frame |
| IR tracker | Gaze **direction** \(D\) (Orlosky origin is *not* metric) |
| `eye_center_front_mm` | Gaze **origin** \(E\) in front/world mm |
| Look-at calib | Rotation \(R\) (+ soft yaw/pitch scales) mapping \(D\) into front |

**If \(E\) is accurate**, the live ray

\[
E + \lambda\, (R D)
\]

is a correct cast in world≡front coordinates and intersects the live screen plane at the
look-at. Wrong \(E\) (e.g. defaulting to the front-camera origin) leaves a parallax error
that grows when the head moves.

### `eye_center_front_mm` in `camera_setup.json`

OpenCV front frame: **X right, Y down, Z forward** (toward the screen). Example for a
left eye ~2.5 cm left, 1.5 cm below, 3 cm toward the face from the front cam:

```json
"left": {
  ...
  "eye_center_front_mm": [-25.0, 15.0, -30.0],
  "eye_center_ir_px": [320, 240]
}
```

Refine `eye_center_front_mm` with a tape / CAD on the headset. Omit or `[0,0,0]` to
fall back to rays from the front-camera origin (MVP approximation).

`eye_center_ir_px` freezes Orlosky’s **2D** eyeball center in the IR tracking buffer
(not metric \(E\)). Prefer locking while looking into the IR camera so pupil ≈ center;
do this after finalizing flip/zoom/pan that affect tracking.

## Geometry (pipeline detail)

- Panel diagonal (or tape width×height) sets **mm scale**; corner ArUco inherit that
  size from fullscreen pixels. Without it, PnP depth is meaningless.
- Corner ArUco → \(T_{\mathrm{front}\leftarrow\mathrm{screen}}\) every frame (head motion OK);
  perpendicular camera↔screen distance is \(|d|\) of the screen plane in the front frame.
- Look-at pixels → known \(P_i\) in front frame.
- Left eye tracker → direction \(D\); calib learns \(R\) so rays from \(E\) pass near \(P_i\).
  Stored \(T_{\mathrm{front}\leftarrow\mathrm{left}}\) uses translation \(E\) and rotation \(R\).
- Heatmap: live ray ∩ screen plane → soft blob overlay (**not** a 2D UV regression).

When `zoom_affects_tracking` is true, pupil/eye-center pixels are **remapped to raw sensor
space** (undo Orlosky 640×480 crop, digital zoom/pan, then flips) before unprojection.
Prefer Phase 0 `left_eye` intrinsics (`K` + `dist`); otherwise fall back to datasheet
`IR_FOV_Y`. Do **not** use `FOV/z` when pan≠0 — that only matches a centered crop and
shifts the principal point under pan. If config `flip` is true, that undo already
inverts vertical sense vs the upright eye frame — exported gaze must **not** also
apply the usual Orlosky Y-up → OpenCV Y-down negate (see `orlosky_to_opencv_direction`).

Yaw/pitch **direction scales** are a soft residual (prior toward 1) for leftover FOV
error. 2D affine / yaw-pitch→UV fits may still be written as **diagnostics** in the JSON;
they are **not** used for the live heatmap.

**Success gate:** mean ray miss ≲ 30–50 mm on a dump with stable `P_front` (Z span tens of mm),
and looking at calib points puts the green hit near the targets via ray∩plane.
cam↔screen HUD should match a tape measure when the diagonal is correct.

## Out of scope (for now)

Right eye / binocular fusion as the primary heatmap path, 2D UV / affine polish as the
heatmap path. Offline \(E\) refine from dumps: `multcam-refine-eye-center` (still seed
from CAD; not a free absolute eyeball measure).
