"""Dataset generation and preprocessing helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .geometry import forward_kinematics


def generate_dataset(
    N: int,
    L1: float,
    L2: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate planar IK samples from uniformly sampled joint angles.

    Parameters
    ----------
    N:
        Number of samples.
    L1, L2:
        Link lengths in millimeters.
    seed:
        Random seed for deterministic sampling.

    Returns
    -------
    tuple of ndarray
        ``X`` target coordinates and ``Y`` joint angles, each with two columns.
    """
    rng = np.random.default_rng(seed=seed)
    theta1 = rng.uniform(-np.pi, np.pi, N)
    theta2 = rng.uniform(0, np.pi, N)
    x, y = forward_kinematics(theta1, theta2, L1, L2)
    X = np.column_stack([x, y])
    Y = np.column_stack([theta1, theta2])
    return X, Y


def train_val_test_split(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    ratios: Sequence[float],
    seed: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Split arrays into train, validation, and test partitions.

    Parameters
    ----------
    X, Y:
        Feature and target arrays with matching first dimensions.
    ratios:
        Three ratios for train, validation, and test. They must sum to one.
    seed:
        Random seed for the permutation.

    Returns
    -------
    tuple of ndarray
        ``X_train, Y_train, X_val, Y_val, X_test, Y_test``.
    """
    if len(ratios) != 3:
        raise ValueError("ratios must contain train, val, and test values")
    ratio_sum = float(sum(ratios))
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError("ratios must sum to 1.0")
    if len(X) != len(Y):
        raise ValueError("X and Y must contain the same number of rows")

    rng = np.random.default_rng(seed=seed)
    indices = rng.permutation(len(X))
    train_end = int(ratios[0] * len(X))
    val_end = train_end + int(ratios[1] * len(X))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        X[train_idx],
        Y[train_idx],
        X[val_idx],
        Y[val_idx],
        X[test_idx],
        Y[test_idx],
    )


def standardize_train_val_test(
    X_train: NDArray[np.float64],
    X_val: NDArray[np.float64],
    X_test: NDArray[np.float64],
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Standardize splits using statistics fit on the training split only.

    Parameters
    ----------
    X_train, X_val, X_test:
        Train, validation, and test feature arrays.

    Returns
    -------
    tuple of ndarray
        Standardized arrays followed by the training mean and standard
        deviation.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)

    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std
    X_test_s = (X_test - mean) / std
    return X_train_s, X_val_s, X_test_s, mean, std
