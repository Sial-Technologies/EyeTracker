import numpy as np
import pytest

from multcam_gaze.types import IntrinsicsModel


@pytest.fixture
def synthetic_intrinsics() -> IntrinsicsModel:
    k = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1.0]], dtype=np.float64)
    d = np.zeros(5, dtype=np.float64)
    return IntrinsicsModel(k, d, (640, 480), 0.5)
