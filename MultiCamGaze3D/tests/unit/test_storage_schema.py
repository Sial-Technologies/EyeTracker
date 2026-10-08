import numpy as np

from multcam_gaze.calibration.schema import CALIBRATION_SCHEMA_VERSION, empty_device_calibration
from multcam_gaze.calibration.storage import CalibrationStore
from multcam_gaze.core.transform import Transform


def test_empty_calibration_schema():
    data = empty_device_calibration()
    assert data["schema_version"] == CALIBRATION_SCHEMA_VERSION
    assert "extrinsics" in data
    assert data["extrinsics"]["world_from_front"] is None


def test_load_left_heatmap_roundtrip(tmp_path):
    store = CalibrationStore(tmp_path)
    t = Transform.from_rotation_translation(
        np.eye(3, dtype=np.float64),
        np.array([-50.0, 30.0, -77.0], dtype=np.float64),
        "front",
        "left_eye",
    )
    data = store.load()
    data["extrinsics"]["front_from_left_eye"] = store.transform_to_json(t)
    data["gaze_calibration"]["left"] = {
        "direction_scale": 1.2,
        "scale_yaw": 1.3,
        "scale_pitch": 1.1,
        "heatmap_path": "ray_intersect_live_aruco",
    }
    data["quality"]["gaze_error_mm"] = 19.5
    store.save(data)

    loaded = store.load_left_heatmap()
    np.testing.assert_allclose(loaded.front_from_left.translation, [-50.0, 30.0, -77.0])
    assert loaded.scale_yaw == 1.3
    assert loaded.scale_pitch == 1.1
    assert loaded.direction_scale == 1.2
    assert loaded.gaze_error_mm == 19.5
    assert loaded.path == store.device_path
