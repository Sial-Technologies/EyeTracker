# Recording Feature

## Overview
The GazeScreen3D application now supports recording video clips of the **preview windows** (camera feeds) with zoom and pan controls.

## Features
- Record **3 separate 15-second clips** in MP4 format
- **Preview window recording** - captures only the camera feeds, not the full screen
- **Zoom controls** (0.5x to 5.0x)
- **Pan controls** to center the view
- Automatic sequential recording
- Visual feedback on recording status
- Easy keyboard control

## What Gets Recorded

The recording captures **only the preview windows**:
- **Full GazeScreen3D mode**: Front camera + Left/Right IR eye cameras (small previews in corner)
- **Eye tracking only mode**: Full-size Left/Right IR cameras + optional Front camera

The recording **does NOT** include:
- Gaze heatmap
- ArUco markers
- Calibration targets
- Main screen overlay

## Usage

### Zoom and Pan Controls
- **Z** - Zoom in (increase 0.1x, max 5.0x)
- **X** - Zoom out (decrease 0.1x, min 0.5x)
- **W** - Pan up
- **A** - Pan left
- **S** - Pan down
- **D** - Pan right
- **P** - Reset view (zoom 1.0x, pan 0,0)

### Recording
Press **R** to start recording the first clip.

### Recording Process
1. **First press of R**: Starts recording clip 1/3 (of the zoomed/panned preview)
2. After 15 seconds, clip 1 is automatically saved and clip 2/3 starts recording
3. After another 15 seconds, clip 2 is saved and clip 3/3 starts recording
4. After the final 15 seconds, all 3 clips are saved

### Manual Control
- Press **R** during recording to stop the current clip early
- Press **R** after all 3 clips are saved to reset and start recording 3 new clips
- Press **R** between clips to start recording the next clip

### Status Display
The recording and view status are shown at the top of the window:
- `Press R to start recording (3x 15s clips)` - Ready to record
- `🔴 REC Clip 1/3 - 14.5s left` - Currently recording (with countdown)
- `View: Zoom 2.0x Pan(50,30)` - Current zoom/pan state
- `⏸ Clips saved: 1/3 (R to record next)` - Clip saved, ready for next
- `✓ Recorded 3/3 clips (R to reset)` - All clips saved, press R to reset

## Workflow Example

1. Start the application and set up cameras
2. **Zoom in** on a specific eye region (press Z multiple times)
3. **Pan** to center the region of interest (WASD keys)
4. **Start recording** (press R)
5. Wait for 3 clips to record automatically (45 seconds total)
6. Find your zoomed preview clips in `GazeScreen3D/recordings/`

## Output Location
Recorded clips are saved to:
```
GazeScreen3D/recordings/
```

## File Naming
Clips are named with the format:
```
clip_<number>_<timestamp>.mp4
```

Example:
- `clip_1_20261002_182530.mp4`
- `clip_2_20261002_182545.mp4`
- `clip_3_20261002_182600.mp4`

## Technical Details
- **Format**: MP4 (mp4v codec)
- **Duration**: 15 seconds per clip
- **Frame Rate**: 30 FPS
- **Resolution**: Matches the preview window size (varies by mode)
- **Content**: Preview windows only with applied zoom/pan transformation
- **Zoom Range**: 0.5x to 5.0x
- **Pan Step**: 10 pixels per key press

## Tips

1. **Zoom before recording**: Set your zoom/pan first, then start recording
2. **Dynamic zoom**: You can adjust zoom/pan while recording - the MP4 captures what you see
3. **Reset view**: Press P at any time to reset to default view (1.0x zoom, no pan)
4. **Preview focus**: Use zoom to focus on specific eye features (pupil, glints, etc.)
5. **Multiple takes**: After recording 3 clips, press R to reset and record 3 more with different zoom/pan

## Notes
- Recording captures the preview window with the current zoom/pan applied
- The same zoom/pan controls work in both "Eye tracking only" and "Full GazeScreen3D" modes
- Zoom/pan transformations do not affect eye tracking accuracy - only the recording and display
- The recordings folder is created automatically if it doesn't exist
- Black borders appear when zooming out below 1.0x or when panning near edges
