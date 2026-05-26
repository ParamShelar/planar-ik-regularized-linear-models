import numpy as np

from planar_ik.geometry import forward_kinematics, inverse_kinematics


def test_ik_fk_round_trip_reachable_points():
    L1, L2 = 100.0, 100.0
    rng = np.random.default_rng(123)
    theta1 = rng.uniform(-np.pi, np.pi, 1000)
    theta2 = rng.uniform(1e-6, np.pi - 1e-6, 1000)
    x, y = forward_kinematics(theta1, theta2, L1, L2)

    theta1_hat, theta2_hat = inverse_kinematics(x, y, L1, L2, elbow="up")
    x_hat, y_hat = forward_kinematics(theta1_hat, theta2_hat, L1, L2)

    err = np.hypot(x - x_hat, y - y_hat)
    assert err.max() < 1e-9


def test_elbow_branches_are_distinct_but_same_endpoint():
    L1, L2 = 100.0, 100.0
    x = np.array([80.0, -90.0, 40.0])
    y = np.array([120.0, 50.0, -130.0])

    up = inverse_kinematics(x, y, L1, L2, elbow="up")
    down = inverse_kinematics(x, y, L1, L2, elbow="down")

    assert not np.allclose(np.column_stack(up), np.column_stack(down))

    x_up, y_up = forward_kinematics(up[0], up[1], L1, L2)
    x_down, y_down = forward_kinematics(down[0], down[1], L1, L2)
    assert np.allclose(x_up, x_down, atol=1e-10)
    assert np.allclose(y_up, y_down, atol=1e-10)
    assert np.allclose(x_up, x, atol=1e-10)
    assert np.allclose(y_up, y, atol=1e-10)
