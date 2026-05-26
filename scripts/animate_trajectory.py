from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from planar_ik.experiments import fit_representative_models, load_results, normalize_config
from planar_ik.features import polynomial_features
from planar_ik.geometry import forward_kinematics, inverse_kinematics
from planar_ik.models import predict_linear, predict_ridge


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Render a trajectory comparison animation.")
    parser.add_argument("--run", default="artifacts/latest", help="Path to artifact run directory.")
    parser.add_argument("--frames", type=int, default=150, help="Number of animation frames.")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second.")
    return parser.parse_args()


def main() -> None:
    """Run the trajectory animation CLI.

    Returns
    -------
    None
        Writes a GIF animation.
    """
    args = parse_args()
    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_path = ROOT / run_path
    results = load_results(run_path)
    output = ROOT / "reports" / "figures" / "trajectory.gif"
    output.parent.mkdir(parents=True, exist_ok=True)
    render_animation(results, output, frames=args.frames, fps=args.fps)
    print(f"Wrote {output.relative_to(ROOT)}")


def render_animation(
    results: dict,
    output_path: Path,
    frames: int = 150,
    fps: int = 20,
) -> None:
    """Render the analytic, linear, and ridge trajectory comparison.

    Parameters
    ----------
    results:
        Loaded experiment results.
    output_path:
        GIF path to write.
    frames:
        Number of animation frames.
    fps:
        Frames per second.

    Returns
    -------
    None
        Writes the animation to disk.
    """
    config = normalize_config(results["metadata"]["config"])
    best = results["metadata"]["summary"]["ridge_joint_optimum"]
    models = fit_representative_models(config, int(best["degree"]), float(best["lambda"]))
    L1 = config["L1"]
    L2 = config["L2"]

    X_path = _target_trajectory(L1, L2, frames)
    Y_true = np.column_stack(inverse_kinematics(X_path[:, 0], X_path[:, 1], L1, L2))
    Y_linear = predict_linear(models["W_linear"], X_path)
    X_s = (X_path - models["x_mean"]) / models["x_std"]
    Phi = polynomial_features(X_s, int(best["degree"]))
    Y_ridge = predict_ridge(models["W_ridge"], Phi)

    predictions = [
        ("Analytic IK", Y_true, "#222222"),
        ("Linear baseline", Y_linear, "#4C78A8"),
        ("Best ridge", Y_ridge, "#D62728"),
    ]
    ee_paths = []
    for _, angles, _ in predictions:
        x_ee, y_ee = forward_kinematics(angles[:, 0], angles[:, 1], L1, L2)
        ee_paths.append(np.column_stack([x_ee, y_ee]))

    outer = L1 + L2
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharex=True, sharey=True)
    arm_lines = []
    trail_lines = []
    target_points = []

    for ax, (title, _, color) in zip(axes, predictions, strict=True):
        ax.plot(X_path[:, 0], X_path[:, 1], color="#BBBBBB", linewidth=1.0)
        trail, = ax.plot([], [], color=color, linewidth=1.5, alpha=0.45)
        arm, = ax.plot([], [], color=color, linewidth=2.5, marker="o", markersize=3)
        target, = ax.plot([], [], color="black", marker="x", markersize=5, linestyle="None")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-outer * 1.05, outer * 1.05)
        ax.set_ylim(-outer * 1.05, outer * 1.05)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.18)
        arm_lines.append(arm)
        trail_lines.append(trail)
        target_points.append(target)

    axes[0].set_ylabel("y (mm)")
    for ax in axes:
        ax.set_xlabel("x (mm)")

    def update(frame: int):
        artists = []
        start = max(0, frame - 35)
        for idx, (_, angles, _) in enumerate(predictions):
            pts = _arm_points(angles[frame, 0], angles[frame, 1], L1, L2)
            arm_lines[idx].set_data(pts[:, 0], pts[:, 1])
            trail_lines[idx].set_data(
                ee_paths[idx][start : frame + 1, 0],
                ee_paths[idx][start : frame + 1, 1],
            )
            target_points[idx].set_data([X_path[frame, 0]], [X_path[frame, 1]])
            artists.extend([arm_lines[idx], trail_lines[idx], target_points[idx]])
        return artists

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=True)
    writer = PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=75)
    plt.close(fig)


def _target_trajectory(L1: float, L2: float, frames: int) -> np.ndarray:
    outer = L1 + L2
    inner = abs(L1 - L2)
    span = outer - inner
    t = np.linspace(0.0, 2 * np.pi, frames, endpoint=False)
    if inner < 1e-9:
        x = 0.55 * outer * np.cos(t)
        y = 0.35 * outer * np.sin(2 * t)
    else:
        x = inner + 0.42 * span + 0.22 * span * np.cos(t)
        y = 0.18 * span * np.sin(2 * t)
    return np.column_stack([x, y])


def _arm_points(theta1: float, theta2: float, L1: float, L2: float) -> np.ndarray:
    elbow = np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)])
    hand_x, hand_y = forward_kinematics(theta1, theta2, L1, L2)
    hand = np.array([float(hand_x), float(hand_y)])
    return np.vstack([np.zeros(2), elbow, hand])


if __name__ == "__main__":
    main()
