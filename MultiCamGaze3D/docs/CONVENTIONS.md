# Coordinate conventions

This document defines the **shared language** used across MultiCamGaze3D:
which axes point where, how transforms are named, and what “world” means in
the left-eye heatmap MVP.

If a number looks wrong by a sign flip or a swap of up/down, start here.

---

## 1. Mental model (read this first)

Imagine three things that matter for the MVP:

1. **Front camera** — the outward-facing camera on the glasses. It sees the
   screen and the ArUco markers.
2. **Left eye (IR)** — the inward-facing IR camera that tracks the pupil and
   produces a **gaze direction** (where the eye is looking).
3. **Screen** — a physical rectangle with known size in millimetres.

In this MVP we treat the **front camera as the world**. Every metric 3D point
(eyeball centre, screen corners, gaze hit) is expressed in millimetres in the
front-camera frame.

Live gaze is then one idea:

> From the eyeball centre \(E\), cast a ray in the calibrated direction until
> it hits the screen plane.

---

## 2. Frames and axis directions

### Front camera (OpenCV)

Used for *all* metric geometry in the MVP (`eye_center_front_mm`, ArUco pose,
gaze hits).

| Axis | Points |
|------|--------|
| **X** | right |
| **Y** | **down** |
| **Z** | forward (toward the screen for the glasses front cam) |

This is the usual OpenCV camera convention. Image \(u\) grows right, \(v\) grows
down — matching X / Y.

### Screen

| Axis | Points |
|------|--------|
| **X** | right |
| **Y** | **up** |
| **Z** | into the display (viewer sits at **negative** Z) |

Screen Y is **up**, front-camera Y is **down**. Converting between them always
needs a careful Y handling (see `runtime/screen_model.py`).

### Screen plane equation

The monitor is a plane:

```text
n · P + d = 0
```

with unit normal \(n\). The perpendicular camera↔screen distance in the front
frame is \(|d|\).

---

## 3. How we name transforms

A rigid transform has a rotation \(R\) and a translation \(t\):

```text
T = [ R  t ]
    [ 0  1 ]
```

### Naming rule

We always name transforms as **destination_from_source**:

| Name in code / JSON | Meaning |
|---------------------|---------|
| `front_from_left_eye` | maps a point expressed in the left-eye frame into the front frame |
| `front_from_screen` | maps a point on the screen into the front frame |
| `world_from_front` | maps front → world (identity in this MVP, because world ≡ front) |

So:

```text
P_front = T_front_from_left @ P_left
```

Read the name left-to-right: **“front from left”** = “give me front coords
*from* left coords”.

In older shorthand you may also see `T_A_B` meaning the same as
`A_from_B` (maps **B → A**).

### Points vs directions

- **Points** (positions in mm): use full \(T\) → rotate *and* translate.
- **Directions** (unit gaze rays): use **only** \(R\):

```text
D_front = R_front_from_left @ D_left
```

A direction has no origin; translating it would be meaningless.

---

## 4. Two different “eye centres” (do not mix them)

`camera_setup.json` has two fields that look similar but answer different
questions.

| Field | Units / space | What it is |
|-------|---------------|------------|
| `eye_center_front_mm` `[x, y, z]` | millimetres in **front OpenCV** | Metric 3D origin \(E\) of the gaze ray in world≡front |
| `eye_center_ir_px` `[u, v]` | pixels in the **IR tracking buffer** | Locked 2D eyeball centre for Orlosky (pupil geometry only) |

**Rule of thumb**

- \(E\) = where the eyeball sits in 3D relative to the front camera →
  `eye_center_front_mm`.
- IR lock = where the eyeball appears in the IR image → `eye_center_ir_px`.

Orlosky’s internal sphere centre is a **virtual** quantity used to estimate
gaze *direction*. It must **never** be used as the metric origin \(E\).

Example (left eye roughly 2.5 cm left, 1.5 cm below, 3 cm toward the face from
the front cam):

```json
"eye_center_front_mm": [-25.0, 15.0, -30.0],
"eye_center_ir_px": [320, 240]
```

Lock `eye_center_ir_px` in preview (left-click while looking into the IR
camera). Refine `eye_center_front_mm` with tape / CAD / `multcam-refine-eye-center`.

---

## 5. World ≡ front (heatmap MVP)

For the left-heatmap path we set:

```text
world = front camera frame (mm)
```

Consequences:

1. `config/screen.json` (diagonal or W×H mm) gives the **absolute scale** of
   the screen.
2. Live corner ArUco + `solvePnP` give `front_from_screen` every frame (head
   motion is OK).
3. Gaze origin is `eye_center_front_mm` (= translation part of
   `front_from_left_eye` after calib).
4. Look-at calibration finds rotation \(R\) (plus soft yaw/pitch scales) so
   rays from \(E\) pass near the known look-at points \(P_i\).

Stored calibration therefore looks like:

```text
front_from_left_eye = [ R | E ]
```

Live cast:

```text
hit = ray starting at E, direction R · D   ∩   live screen plane
```

---

## 6. OpenCV `solvePnP` (sign / inverse trap)

`cv2.solvePnP` with object points in some board/object frame returns \(R, t\)
such that:

```text
P_camera = R @ P_object + t
```

That is already `camera_from_object` (e.g. `front_from_screen`).

If you ever need the opposite pose (`object_from_camera` /
`world_from_camera` in a non-MVP setup), invert:

```text
T_object_from_camera = inverse(T_camera_from_object)
```

Do not invert twice, and do not invert when the JSON field is already named
`front_from_screen`.

---

## 7. Quick checklist when debugging geometry

1. **Which frame is this vector in?** front / screen / left-eye / Orlosky IR.
2. **Point or direction?** points get \(t\); directions do not.
3. **Y up or Y down?** front OpenCV = down; screen = up; Orlosky internal = up.
4. **Metric \(E\) or IR pixel lock?** `eye_center_front_mm` vs `eye_center_ir_px`.
5. **Transform name** should read as `destination_from_source`.

For the operator pipeline and equations in context, see
[`LEFT_HEATMAP_MVP.md`](LEFT_HEATMAP_MVP.md).
