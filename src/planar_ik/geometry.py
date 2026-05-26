"""Geometry utilities for a two-link planar robot arm."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def forward_kinematics(
    theta1: ArrayLike,
    theta2: ArrayLike,
    L1: float,
    L2: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Map joint angles to planar end-effector coordinates.

    Parameters
    ----------
    theta1, theta2:
        First and second joint angles in radians. Scalars or array-like values
        are accepted and broadcast by NumPy.
    L1, L2:
        Link lengths in millimeters.

    Returns
    -------
    tuple of ndarray
        Arrays ``(x, y)`` with the broadcast shape of the input angles.
    """
    theta1_arr = np.asarray(theta1, dtype=float)
    theta2_arr = np.asarray(theta2, dtype=float)
    x = L1 * np.cos(theta1_arr) + L2 * np.cos(theta1_arr + theta2_arr)
    y = L1 * np.sin(theta1_arr) + L2 * np.sin(theta1_arr + theta2_arr)
    return x, y


def inverse_kinematics(
    x: ArrayLike,
    y: ArrayLike,
    L1: float,
    L2: float,
    elbow: str = "up",
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Solve closed-form inverse kinematics for the planar arm.

    Parameters
    ----------
    x, y:
        Target end-effector coordinates in millimeters.
    L1, L2:
        Link lengths in millimeters.
    elbow:
        Elbow branch to use. Valid values are ``"up"`` and ``"down"``.

    Returns
    -------
    tuple of ndarray
        Arrays ``(theta1, theta2)`` in radians.

    Raises
    ------
    ValueError
        If ``elbow`` is not ``"up"`` or ``"down"``.
    """
    if elbow not in {"up", "down"}:
        raise ValueError("elbow must be 'up' or 'down'")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    r2 = x_arr**2 + y_arr**2
    c2 = (r2 - L1**2 - L2**2) / (2 * L1 * L2)
    c2 = np.clip(c2, -1.0, 1.0)

    s2 = np.sqrt(1 - c2**2)
    if elbow == "down":
        s2 = -s2

    theta2 = np.arctan2(s2, c2)
    theta1 = np.arctan2(y_arr, x_arr) - np.arctan2(
        L2 * np.sin(theta2),
        L1 + L2 * np.cos(theta2),
    )
    return theta1, theta2


def ee_errors(
    X_true: NDArray[np.float64],
    Y_pred: NDArray[np.float64],
    L1: float = 100,
    L2: float = 100,
) -> NDArray[np.float64]:
    """Compute end-effector errors for predicted joint angles.

    Parameters
    ----------
    X_true:
        Target coordinates with shape ``(n_samples, 2)``.
    Y_pred:
        Predicted joint angles with shape ``(n_samples, 2)``.
    L1, L2:
        Link lengths in millimeters.

    Returns
    -------
    ndarray
        Euclidean end-effector errors in millimeters.
    """
    x_true = X_true[:, 0]
    y_true = X_true[:, 1]
    theta1_pred = Y_pred[:, 0]
    theta2_pred = Y_pred[:, 1]
    x_pred, y_pred = forward_kinematics(theta1_pred, theta2_pred, L1, L2)
    return np.hypot(x_true - x_pred, y_true - y_pred)


def reachable_workspace_mask(
    X: NDArray[np.float64],
    L1: float,
    L2: float,
    tol: float = 1e-9,
) -> NDArray[np.bool_]:
    """Return a boolean mask for targets inside the reachable annulus.

    Parameters
    ----------
    X:
        Coordinates with shape ``(n_samples, 2)``.
    L1, L2:
        Link lengths in millimeters.
    tol:
        Numerical tolerance added to both annulus boundaries.

    Returns
    -------
    ndarray
        Boolean mask where ``True`` indicates a reachable point.
    """
    r = np.hypot(X[:, 0], X[:, 1])
    inner = abs(L1 - L2)
    outer = L1 + L2
    return (r >= inner - tol) & (r <= outer + tol)
