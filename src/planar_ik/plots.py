"""Plotting helpers for experiment artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .experiments import fit_representative_models, normalize_config
from .features import polynomial_features
from .geometry import ee_errors, forward_kinematics, inverse_kinematics, reachable_workspace_mask
from .models import predict_linear, predict_ridge


def plot_val_error_vs_degree(results: dict[str, Any], output_path: str | Path) -> None:
    """Plot unregularized train/validation error versus polynomial degree.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_path:
        PNG path to write.

    Returns
    -------
    None
        Writes a figure to disk.
    """
    degrees = results["degrees"]
    train_mean = results["degree_train_mean"]
    train_std = results["degree_train_std"]
    val_mean = results["degree_val_mean"]
    val_std = results["degree_val_std"]
    best_idx = int(np.nanargmin(val_mean))

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_band(ax, degrees, train_mean, train_std, "train", "#4C78A8")
    _plot_band(ax, degrees, val_mean, val_std, "val", "#F58518")
    ax.axvline(degrees[best_idx], color="black", linestyle="--", linewidth=1)
    ax.scatter([degrees[best_idx]], [val_mean[best_idx]], color="black", zorder=4)
    ax.annotate(
        f"best degree {degrees[best_idx]}\n{val_mean[best_idx]:.2f} mm",
        xy=(degrees[best_idx], val_mean[best_idx]),
        xytext=(8, 16),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_title("Unregularized Polynomial Sweep")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("Mean EE error (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, output_path)


def plot_weight_norm_and_condition_vs_degree(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Plot coefficient norm and condition number versus degree.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_path:
        PNG path to write.

    Returns
    -------
    None
        Writes a figure to disk.
    """
    degrees = results["degrees"]
    norm_mean = results["degree_weight_norm_mean"]
    norm_std = results["degree_weight_norm_std"]
    cond_mean = results["degree_condition_mean"]
    cond_std = results["degree_condition_std"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    _plot_band(axes[0], degrees, norm_mean, norm_std, "mean +/- std", "#4C78A8")
    axes[0].set_yscale("log")
    axes[0].set_title("Weight Norm vs Degree")
    axes[0].set_xlabel("Polynomial degree")
    axes[0].set_ylabel("||W||")
    axes[0].grid(True, alpha=0.25)

    _plot_band(axes[1], degrees, cond_mean, cond_std, "mean +/- std", "#F58518")
    axes[1].set_yscale("log")
    axes[1].set_title("Condition Number vs Degree")
    axes[1].set_xlabel("Polynomial degree")
    axes[1].set_ylabel("cond(Phi.T @ Phi)")
    axes[1].grid(True, alpha=0.25)
    _save(fig, output_path)


def plot_ridge_val_vs_lambda(results: dict[str, Any], output_path: str | Path) -> None:
    """Plot fixed-degree ridge error versus lambda.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_path:
        PNG path to write.

    Returns
    -------
    None
        Writes a figure to disk.
    """
    lambdas = results["lambdas"]
    train_mean = results["ridge_train_mean"]
    train_std = results["ridge_train_std"]
    val_mean = results["ridge_val_mean"]
    val_std = results["ridge_val_std"]
    best_idx = int(np.nanargmin(val_mean))
    fixed_degree = int(results["metadata"]["config"]["ridge_fixed_degree"])

    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_band(ax, lambdas, train_mean, train_std, "train", "#4C78A8")
    _plot_band(ax, lambdas, val_mean, val_std, "val", "#F58518")
    ax.axvline(lambdas[best_idx], color="black", linestyle="--", linewidth=1)
    ax.annotate(
        f"best lambda {lambdas[best_idx]:.2e}\n{val_mean[best_idx]:.2f} mm",
        xy=(lambdas[best_idx], val_mean[best_idx]),
        xytext=(10, 16),
        textcoords="offset points",
        fontsize=9,
    )
    ax.set_xscale("log")
    ax.set_title(f"Ridge Error vs Lambda (degree {fixed_degree})")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Mean EE error (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    _save(fig, output_path)


def plot_ridge_weight_norm_vs_lambda(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Plot ridge coefficient norm versus lambda.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_path:
        PNG path to write.

    Returns
    -------
    None
        Writes a figure to disk.
    """
    lambdas = results["lambdas"]
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_band(
        ax,
        lambdas,
        results["ridge_weight_norm_mean"],
        results["ridge_weight_norm_std"],
        "mean +/- std",
        "#4C78A8",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Ridge Weight Norm vs Lambda")
    ax.set_xlabel("lambda")
    ax.set_ylabel("||W||")
    ax.grid(True, alpha=0.25, which="both")
    _save(fig, output_path)


def plot_ridge_condition_vs_lambda(
    results: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Plot ridge-stabilized condition number versus lambda.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_path:
        PNG path to write.

    Returns
    -------
    None
        Writes a figure to disk.
    """
    lambdas = results["lambdas"]
    fig, ax = plt.subplots(figsize=(7, 5))
    _plot_band(
        ax,
        lambdas,
        results["ridge_condition_mean"],
        results["ridge_condition_std"],
        "mean +/- std",
        "#F58518",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Condition Number vs Lambda")
    ax.set_xlabel("lambda")
    ax.set_ylabel("cond(Phi.T @ Phi + lambda I)")
    ax.grid(True, alpha=0.25, which="both")
    _save(fig, output_path)


def plot_coefficient_paths(
    results: dict[str, Any],
    output_path: str | Path,
    top_k: int = 15,
) -> None:
    """Plot ridge coefficient paths for the largest small-lambda weights.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_path:
        PNG path to write.
    top_k:
        Number of scalar coefficient paths to draw.

    Returns
    -------
    None
        Writes a figure to disk.
    """
    lambdas = results["lambdas"]
    coeffs = results["ridge_coefficients"][0]
    flattened = np.abs(coeffs[0]).reshape(-1)
    top = np.argsort(flattened)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for flat_idx in top:
        coef_idx = int(flat_idx // coeffs.shape[2])
        out_idx = int(flat_idx % coeffs.shape[2])
        label = f"theta{out_idx + 1} c{coef_idx}"
        ax.plot(lambdas, coeffs[:, coef_idx, out_idx], linewidth=1.1, label=label)

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_title("Ridge Coefficient Paths")
    ax.set_xlabel("lambda")
    ax.set_ylabel("coefficient value")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=7, frameon=False)
    _save(fig, output_path)


def plot_grid_val_heatmap(results: dict[str, Any], output_path: str | Path) -> None:
    """Plot aggregated validation error over the degree/lambda grid.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_path:
        PNG path to write.

    Returns
    -------
    None
        Writes a figure to disk.
    """
    grid_val = results["grid_val_mean"]
    degrees = results["grid_degrees"]
    lambdas = results["grid_lambdas"]
    best_flat = int(np.nanargmin(grid_val))
    best_i, best_j = np.unravel_index(best_flat, grid_val.shape)

    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        grid_val,
        aspect="auto",
        origin="lower",
        extent=[
            np.log10(lambdas[0]),
            np.log10(lambdas[-1]),
            degrees[0],
            degrees[-1],
        ],
        cmap="viridis",
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Val EE error (mm)")
    ax.scatter(
        [np.log10(lambdas[best_j])],
        [degrees[best_i]],
        color="red",
        marker="x",
        s=120,
        linewidths=3,
        label=f"best: deg {degrees[best_i]}, lam {lambdas[best_j]:.1e}",
    )
    ax.set_title("Validation Error over Degree/Lambda Grid")
    ax.set_xlabel("log10(lambda)")
    ax.set_ylabel("Polynomial degree")
    ax.legend(loc="upper right")
    _save(fig, output_path)


def plot_workspace_ee_error_heatmap(
    results: dict[str, Any],
    output_path: str | Path,
    grid_size: int = 180,
) -> None:
    """Plot workspace end-effector error for linear and best ridge models.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_path:
        PNG path to write.
    grid_size:
        Number of points per axis in the workspace grid.

    Returns
    -------
    None
        Writes a figure to disk.
    """
    config = normalize_config(results["metadata"]["config"])
    best = results["metadata"]["summary"]["ridge_joint_optimum"]
    models = fit_representative_models(config, int(best["degree"]), float(best["lambda"]))
    L1 = config["L1"]
    L2 = config["L2"]
    outer = L1 + L2

    xs = np.linspace(-outer, outer, grid_size)
    ys = np.linspace(-outer, outer, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    mask = reachable_workspace_mask(points, L1, L2)

    linear_grid = np.full(len(points), np.nan)
    ridge_grid = np.full(len(points), np.nan)
    X_mask = points[mask]
    linear_grid[mask] = ee_errors(
        X_mask,
        predict_linear(models["W_linear"], X_mask),
        L1,
        L2,
    )
    X_mask_s = (X_mask - models["x_mean"]) / models["x_std"]
    Phi_mask = polynomial_features(X_mask_s, int(best["degree"]))
    ridge_grid[mask] = ee_errors(
        X_mask,
        predict_ridge(models["W_ridge"], Phi_mask),
        L1,
        L2,
    )

    linear_grid = linear_grid.reshape(xx.shape)
    ridge_grid = ridge_grid.reshape(xx.shape)
    vmax = float(np.nanpercentile(np.concatenate([linear_grid.ravel(), ridge_grid.ravel()]), 95))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    images = []
    for ax, grid, title in zip(
        axes,
        [linear_grid, ridge_grid],
        ["Linear baseline", "Best polynomial ridge"],
        strict=True,
    ):
        im = ax.imshow(
            np.ma.masked_invalid(grid),
            origin="lower",
            extent=[-outer, outer, -outer, outer],
            cmap="magma",
            vmin=0,
            vmax=vmax,
        )
        images.append(im)
        ax.set_title(title)
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_aspect("equal")
    cbar = fig.colorbar(images[0], ax=axes, shrink=0.86)
    cbar.set_label("EE error (mm)")
    _save(fig, output_path)


def plot_predicted_vs_true_arm_pose(
    results: dict[str, Any],
    output_path: str | Path,
    n_targets: int = 12,
) -> None:
    """Plot analytic and ridge-predicted two-link arm poses.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_path:
        PNG path to write.
    n_targets:
        Number of target poses to draw.

    Returns
    -------
    None
        Writes a figure to disk.
    """
    config = normalize_config(results["metadata"]["config"])
    best = results["metadata"]["summary"]["ridge_joint_optimum"]
    models = fit_representative_models(config, int(best["degree"]), float(best["lambda"]))
    L1 = config["L1"]
    L2 = config["L2"]
    targets = _pose_targets(L1, L2, n_targets)

    X_s = (targets - models["x_mean"]) / models["x_std"]
    Phi = polynomial_features(X_s, int(best["degree"]))
    Y_pred = predict_ridge(models["W_ridge"], Phi)
    theta_true = np.column_stack(inverse_kinematics(targets[:, 0], targets[:, 1], L1, L2))

    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, (target, true_angles, pred_angles) in enumerate(zip(targets, theta_true, Y_pred, strict=True)):
        true_pts = _arm_points(true_angles[0], true_angles[1], L1, L2)
        pred_pts = _arm_points(pred_angles[0], pred_angles[1], L1, L2)
        ax.plot(
            true_pts[:, 0],
            true_pts[:, 1],
            color="#222222",
            linewidth=2.0,
            alpha=0.85,
            label="analytic IK" if idx == 0 else None,
        )
        ax.plot(
            pred_pts[:, 0],
            pred_pts[:, 1],
            color="#D62728",
            linewidth=1.6,
            linestyle="--",
            alpha=0.85,
            label="ridge prediction" if idx == 0 else None,
        )
        ax.scatter([target[0]], [target[1]], color="#4C78A8", s=18, zorder=3)

    outer = L1 + L2
    ax.set_xlim(-outer * 1.05, outer * 1.05)
    ax.set_ylim(-outer * 1.05, outer * 1.05)
    ax.set_aspect("equal")
    ax.set_title("Predicted vs True Arm Poses")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper right")
    _save(fig, output_path)


def make_all_plots(results: dict[str, Any], output_dir: str | Path) -> list[Path]:
    """Generate every required PNG figure.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_dir:
        Directory where PNGs are written.

    Returns
    -------
    list of Path
        Paths to generated figures.
    """
    output_dir = Path(output_dir)
    outputs = [
        output_dir / "val_error_vs_degree.png",
        output_dir / "weight_norm_and_condition_vs_degree.png",
        output_dir / "ridge_val_vs_lambda.png",
        output_dir / "ridge_weight_norm_vs_lambda.png",
        output_dir / "ridge_condition_vs_lambda.png",
        output_dir / "coefficient_paths.png",
        output_dir / "grid_val_heatmap.png",
        output_dir / "workspace_ee_error_heatmap.png",
        output_dir / "predicted_vs_true_arm_pose.png",
    ]
    plot_val_error_vs_degree(results, outputs[0])
    plot_weight_norm_and_condition_vs_degree(results, outputs[1])
    plot_ridge_val_vs_lambda(results, outputs[2])
    plot_ridge_weight_norm_vs_lambda(results, outputs[3])
    plot_ridge_condition_vs_lambda(results, outputs[4])
    plot_coefficient_paths(results, outputs[5])
    plot_grid_val_heatmap(results, outputs[6])
    plot_workspace_ee_error_heatmap(results, outputs[7])
    plot_predicted_vs_true_arm_pose(results, outputs[8])
    return outputs


def _plot_band(
    ax: plt.Axes,
    x: NDArray[np.float64],
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
    label: str,
    color: str,
) -> None:
    ax.plot(x, mean, color=color, marker="o", markersize=3, linewidth=1.5, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0)


def _save(fig: plt.Figure, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pose_targets(L1: float, L2: float, n_targets: int) -> NDArray[np.float64]:
    inner = abs(L1 - L2)
    outer = L1 + L2
    span = outer - inner
    angles = np.linspace(0.0, 2 * np.pi, n_targets, endpoint=False)
    fractions = np.resize(np.array([0.42, 0.62, 0.82]), n_targets)
    radii = inner + fractions * span
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _arm_points(theta1: float, theta2: float, L1: float, L2: float) -> NDArray[np.float64]:
    elbow = np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)])
    hand_x, hand_y = forward_kinematics(theta1, theta2, L1, L2)
    hand = np.array([float(hand_x), float(hand_y)])
    return np.vstack([np.zeros(2), elbow, hand])
