"""Planar inverse kinematics study package."""

from .datasets import generate_dataset, standardize_train_val_test, train_val_test_split
from .features import polynomial_features
from .geometry import (
    ee_errors,
    forward_kinematics,
    inverse_kinematics,
    reachable_workspace_mask,
)
from .metrics import angle_rmse, mean_error_calc, print_summary, summarize_errors
from .models import (
    TorchRidge,
    TorchRidgeGD,
    assert_torch_numpy_ridge_parity,
    fit_linear,
    fit_ridge,
    predict_linear,
    predict_ridge,
)

__all__ = [
    "TorchRidge",
    "TorchRidgeGD",
    "angle_rmse",
    "assert_torch_numpy_ridge_parity",
    "ee_errors",
    "fit_linear",
    "fit_ridge",
    "forward_kinematics",
    "generate_dataset",
    "inverse_kinematics",
    "mean_error_calc",
    "polynomial_features",
    "predict_linear",
    "predict_ridge",
    "print_summary",
    "reachable_workspace_mask",
    "standardize_train_val_test",
    "summarize_errors",
    "train_val_test_split",
]
