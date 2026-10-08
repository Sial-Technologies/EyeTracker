import numpy as np

from multcam_gaze.core.transform import Transform


def test_compose_and_apply_point():
    t_ab = Transform.from_rotation_translation(
        np.eye(3),
        np.array([1.0, 0.0, 0.0]),
        "A",
        "B",
    )
    t_bc = Transform.from_rotation_translation(
        np.eye(3),
        np.array([0.0, 2.0, 0.0]),
        "B",
        "C",
    )
    t_ac = t_ab.compose(t_bc)
    p_c = np.array([0.0, 0.0, 0.0])
    p_a = t_ac.apply_point(p_c)
    np.testing.assert_allclose(p_a, [1.0, 2.0, 0.0], atol=1e-9)


def test_inverse_roundtrip():
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    t = Transform.from_rotation_translation(r, np.array([10.0, -3.0, 5.0]), "world", "cam")
    p = np.array([1.0, 2.0, 3.0])
    p2 = t.inverse().apply_point(t.apply_point(p))
    np.testing.assert_allclose(p2, p, atol=1e-9)


def test_apply_direction_no_translation():
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    t = Transform.from_rotation_translation(r, np.array([100.0, 0.0, 0.0]), "A", "B")
    d = np.array([1.0, 0.0, 0.0])
    out = t.apply_direction(d)
    np.testing.assert_allclose(out, [0.0, 1.0, 0.0], atol=1e-9)
