"""Metrics for evaluating inverse-kinematics regressors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .geometry import ee_errors


def mean_error_calc(
    X_true: NDArray[np.float64],
    Y_pred: NDArray[np.float64],
    L1: float,
    L2: float,
) -> float:
    """Compute mean end-effector error in millimeters.

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
    float
        Mean end-effector error in millimeters.
    """
    return float(ee_errors(X_true, Y_pred, L1, L2).mean())


def summarize_errors(
    name: str,
    X_true: NDArray[np.float64],
    Y_pred: NDArray[np.float64],
    L1: float = 100,
    L2: float = 100,
) -> dict[str, float | str]:
    """Summarize end-effector errors without printing.

    Parameters
    ----------
    name:
        Human-readable label for the model or split.
    X_true:
        Target coordinates with shape ``(n_samples, 2)``.
    Y_pred:
        Predicted joint angles with shape ``(n_samples, 2)``.
    L1, L2:
        Link lengths in millimeters.

    Returns
    -------
    dict
        Summary with ``mean``, ``median``, ``p95``, and ``max`` keys.
    """
    errs = ee_errors(X_true, Y_pred, L1, L2)
    return {
        "name": name,
        "mean": float(errs.mean()),
        "median": float(np.median(errs)),
        "p95": float(np.percentile(errs, 95)),
        "max": float(errs.max()),
    }


def print_summary(summary: dict[str, float | str]) -> None:
    """Print a summary returned by :func:`summarize_errors`.

    Parameters
    ----------
    summary:
        Error summary dictionary.

    Returns
    -------
    None
        This function prints for human-facing scripts and notebooks.
    """
    print(summary["name"])
    print(f"mean: {summary['mean']:.3f} mm")
    print(f"median: {summary['median']:.3f} mm")
    print(f"p95: {summary['p95']:.3f} mm")
    print(f"max: {summary['max']:.3f} mm")


def angle_rmse(
    Y_true: NDArray[np.float64],
    Y_pred: NDArray[np.float64],
) -> float:
    """Compute joint-angle root mean squared error.

    Parameters
    ----------
    Y_true:
        True joint angles.
    Y_pred:
        Predicted joint angles.

    Returns
    -------
    float
        RMSE in radians over both angle dimensions.
    """
    return float(np.sqrt(np.mean((Y_true - Y_pred) ** 2)))
