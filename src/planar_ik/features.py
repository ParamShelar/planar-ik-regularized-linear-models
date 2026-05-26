"""Feature construction for planar IK regression."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def polynomial_features(
    X: NDArray[np.float64],
    degree: int,
) -> NDArray[np.float64]:
    """Construct total-degree monomials in two variables.

    The feature order matches the original notebook: for each total degree,
    powers are emitted as ``x**power_x * y**power_y`` with ``power_x``
    increasing from zero to the total degree. A bias column is not included.

    Parameters
    ----------
    X:
        Input coordinates with shape ``(n_samples, 2)``.
    degree:
        Maximum total polynomial degree. Must be at least one.

    Returns
    -------
    ndarray
        Polynomial feature matrix without a bias column.
    """
    if degree < 1:
        raise ValueError("degree must be at least 1")
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (n_samples, 2)")

    x = X[:, 0]
    y = X[:, 1]
    features = []

    for total_degree in range(1, degree + 1):
        for power_x in range(total_degree + 1):
            power_y = total_degree - power_x
            features.append((x**power_x) * (y**power_y))

    return np.column_stack(features)
