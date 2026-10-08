# MultiCamGaze3D

Left-eye + front gaze mapping: Phase 0 intrinsics, then look-at calibration + live screen heatmap.

## Setup

```bash
cd MultiCamGaze3D
pip install -e ".[dev,viz]"
pytest
```

## Commands

| Command | Purpose |
|---------|---------|
| `multcam-show-chessboard` | On-screen chessboard for Phase 0 |
| `multcam-intrinsics --role front` | Chessboard intrinsics (`left_eye` too if needed) |
| `multcam-left-heatmap` | Left-eye look-at calib + live screen heatmap |
| `multcam-preview` | L/R/Front preview + recording |

Common flags: `--config-dir`, `--calib-dir`, `--verbose`, `--dry-run`.

## Docs

- [LEFT_HEATMAP_MVP.md](docs/LEFT_HEATMAP_MVP.md) — Phase 1 operator path
- [PHASE0_RUNBOOK.md](docs/PHASE0_RUNBOOK.md) — Phase 0 checklist
- [CONVENTIONS.md](docs/CONVENTIONS.md) — transforms and frames
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — module layers
