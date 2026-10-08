# Architecture

## Layers

| Layer | Packages | May import |
|-------|----------|------------|
| core | `multcam_gaze.core` | numpy only |
| geometry | `multcam_gaze.geometry` | core |
| vision | `multcam_gaze.vision` | core, OpenCV |
| calibration | `multcam_gaze.calibration` | core, vision, geometry |
| hardware | `multcam_gaze.hardware` | core, OpenCV |
| tracking | `multcam_gaze.tracking` | legacy eye_tracker |
| runtime | `multcam_gaze.runtime` | calibration, geometry, core |
| apps | `multcam_gaze.apps` | all of the above |

## Calibration file versioning

`calib/device_calibration.json` includes `schema_version`. Increment when breaking fields change; add migration in `calibration/storage.py`.

## Metric world (left heatmap MVP)

1. `config/screen.json` — panel diagonal / W×H mm (ArUco scale).
2. Live corner ArUco — `T_front_screen` and cam↔screen distance in world≡front.
3. `camera_setup.json` → `eye_center_front_mm` — eyeball origin \(E\) in front mm.
4. IR + look-at solve — rotation \(R\) so \(E + \lambda (R D)\) hits calib points.

See `docs/LEFT_HEATMAP_MVP.md` and `docs/CONVENTIONS.md`.
