import numpy as np

from multcam_gaze.core.transform import Transform
from multcam_gaze.runtime.gaze_pipeline import gaze_to_world_ray, intersect_gaze_with_screen
from multcam_gaze.runtime.screen_model import ScreenModel
from multcam_gaze.types import GazeRayInCameraFrame


def test_gaze_screen_intersection():
    # Screen in front of the eye (+Z); viewer looks along +Z into the display.
    world_from_screen = Transform.from_rotation_translation(
        np.eye(3),
        np.array([0.0, 0.0, 400.0]),
        "world",
        "screen",
    )
    screen = ScreenModel(600.0, 400.0, world_from_screen)
    world_from_eye = Transform.from_rotation_translation(
        np.eye(3),
        np.array([0.0, 0.0, 0.0]),
        "world",
        "left_eye",
    )
    gaze = GazeRayInCameraFrame(
        np.zeros(3),
        np.array([0.0, 0.0, 1.0]),
        "left",
        valid=True,
    )
    ray = gaze_to_world_ray(gaze, world_from_eye)
    hit = intersect_gaze_with_screen(ray, screen, 1920, 1080)
    assert hit.valid
    assert hit.on_screen
    assert hit.screen_mm is not None
    assert hit.pixels is not None


def test_gaze_behind_plane_rejected():
    world_from_screen = Transform.from_rotation_translation(
        np.eye(3),
        np.array([0.0, 0.0, 400.0]),
        "world",
        "screen",
    )
    screen = ScreenModel(600.0, 400.0, world_from_screen)
    world_from_eye = Transform.from_rotation_translation(
        np.eye(3), np.zeros(3), "world", "left_eye"
    )
    gaze = GazeRayInCameraFrame(
        np.zeros(3),
        np.array([0.0, 0.0, -1.0]),
        "left",
        valid=True,
    )
    hit = intersect_gaze_with_screen(
        gaze_to_world_ray(gaze, world_from_eye), screen, 1920, 1080
    )
    assert not hit.valid


def test_gaze_off_screen_flagged():
    world_from_screen = Transform.from_rotation_translation(
        np.eye(3),
        np.array([0.0, 0.0, 400.0]),
        "world",
        "screen",
    )
    screen = ScreenModel(600.0, 400.0, world_from_screen)
    world_from_eye = Transform.from_rotation_translation(
        np.eye(3), np.zeros(3), "world", "left_eye"
    )
    # Aim far past the right edge of a 600 mm screen at 400 mm depth.
    gaze = GazeRayInCameraFrame(
        np.zeros(3),
        np.array([0.8, 0.0, 0.6]),
        "left",
        valid=True,
    )
    d = gaze.direction / np.linalg.norm(gaze.direction)
    gaze = GazeRayInCameraFrame(gaze.origin, d, "left", valid=True)
    hit = intersect_gaze_with_screen(
        gaze_to_world_ray(gaze, world_from_eye), screen, 1920, 1080
    )
    assert hit.valid
    assert not hit.on_screen
