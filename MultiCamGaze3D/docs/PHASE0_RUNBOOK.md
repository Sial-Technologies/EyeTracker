# Phase 0 runbook — intrinsics

**Goal:** One lens model per camera → `calib/intrinsics/{role}.npz`

Physical rationale and references: [CALIBRATION_PROCEDURE.md](CALIBRATION_PROCEDURE.md).

---

## Setup (once)

```powershell
cd MultiCamGaze3D
pip install -e ".[dev,viz]"
```

Optional: confirm which feed is which:

```powershell
multcam-preview
```

Camera mapping comes from `config/camera_setup.json` via `--role` (or `--camera-index` to override).

| `--role` | Setup key in `camera_setup.json` |
|----------|----------------------------------|
| `front` | `front` |
| `left_eye` | `left` |
| `right_eye` | `right` |

---

## Pattern (Terminal A)

```powershell
multcam-show-chessboard
```

Move that window to the monitor the camera will see.

- **front (RGB):** monitor or printed board is fine.
- **left_eye / right_eye (IR):** prefer a **printed** board; LCD patterns are often invisible in IR.

Pattern matches `config/chessboard.json` (default **9×6** inner corners). If you use a screen and care about metric scale later, measure one square with a ruler and update `square_size_mm`.

Optional: `multcam-show-chessboard --fullscreen`

---

## Calibrate each camera (Terminal B)

| Order | Command | Output |
|------|---------|--------|
| 1 | `multcam-intrinsics --role front` | `calib/intrinsics/front.npz` |
| 2 | `multcam-intrinsics --role left_eye` | `calib/intrinsics/left_eye.npz` |
| 3 | `multcam-intrinsics --role right_eye` | `calib/intrinsics/right_eye.npz` |

Live preview window: **Intrinsics calibration** (camera feed + corner overlay).

| Key | Action |
|-----|--------|
| `SPACE` | Capture pose (needs green corners) |
| `C` | Calibrate & save (≥10 poses) |
| `R` | Reset captures |
| `Q` | Quit |

Capture **≥10** varied poses (tilt, distance, board position in frame). After `C`, check the printed RMS (typically well under ~1 px for a good set).

Override camera index if needed:

```powershell
multcam-intrinsics --role front --camera-index 3
```

---

## Done when

```text
calib/intrinsics/front.npz
calib/intrinsics/left_eye.npz
calib/intrinsics/right_eye.npz
```

Next: Phase 1 extrinsics — `multcam-extrinsics-poc` (see [CALIBRATION_PROCEDURE.md](CALIBRATION_PROCEDURE.md)).
