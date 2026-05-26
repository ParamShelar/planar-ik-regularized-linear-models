"""Experiment orchestration for the planar IK regularization study."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .datasets import generate_dataset, standardize_train_val_test, train_val_test_split
from .features import polynomial_features
from .metrics import angle_rmse, mean_error_calc
from .models import fit_linear, fit_ridge, predict_linear, predict_ridge


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Normalized configuration dictionary.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to load experiment configs") from exc

    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return normalize_config(raw)


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize compact config sequences into explicit Python lists.

    Parameters
    ----------
    config:
        Raw configuration dictionary.

    Returns
    -------
    dict
        Configuration with numeric sweep fields expanded.
    """
    normalized = dict(config)
    normalized["L1"] = float(normalized["L1"])
    normalized["L2"] = float(normalized["L2"])
    normalized["N"] = int(normalized["N"])
    normalized["seeds"] = [int(seed) for seed in normalized["seeds"]]
    normalized["ratios"] = [float(ratio) for ratio in normalized["ratios"]]
    normalized["degrees"] = _expand_sequence(normalized["degrees"], dtype=int)
    normalized["lambdas"] = _expand_sequence(normalized["lambdas"], dtype=float)
    normalized["ridge_fixed_degree"] = int(normalized["ridge_fixed_degree"])
    normalized["grid_degrees"] = _expand_sequence(normalized["grid_degrees"], dtype=int)
    normalized["grid_lambdas"] = _expand_sequence(normalized["grid_lambdas"], dtype=float)
    return normalized


def run_full_study(
    config: dict[str, Any],
    run_id: str | None = None,
    artifacts_dir: str | Path = "artifacts",
) -> dict[str, Any]:
    """Run the complete multi-seed planar IK study and persist artifacts.

    Parameters
    ----------
    config:
        Normalized configuration dictionary.
    run_id:
        Optional run identifier. If omitted, a UTC timestamp is used.
    artifacts_dir:
        Directory where run artifacts are written.

    Returns
    -------
    dict
        In-memory results dictionary containing arrays and metadata.
    """
    config = normalize_config(config)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    artifacts_path = Path(artifacts_dir)
    run_dir = artifacts_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    arrays = _run_all_seeds(config)
    aggregate = _aggregate_arrays(arrays)
    summary = _build_summary(config, arrays, aggregate, run_id)

    metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "summary": summary,
    }
    results = {"metadata": metadata, **arrays, **aggregate}

    _persist_results(results, run_dir)
    _write_config(config, run_dir / "config.yaml")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _update_latest(run_dir, artifacts_path)
    return results


def load_results(run_path: str | Path) -> dict[str, Any]:
    """Load a persisted experiment run.

    Parameters
    ----------
    run_path:
        Path to an artifact run directory or ``artifacts/latest``.

    Returns
    -------
    dict
        Results dictionary containing metadata and NumPy arrays.
    """
    run_dir = Path(run_path)
    npz_path = run_dir / "results.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing results file: {npz_path}")

    loaded = np.load(npz_path, allow_pickle=False)
    results: dict[str, Any] = {}
    for key in loaded.files:
        if key == "metadata_json":
            results["metadata"] = json.loads(str(loaded[key]))
        else:
            results[key] = loaded[key]

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            results["metadata"]["summary"] = json.load(f)
    return results


def representative_seed(config: dict[str, Any]) -> int:
    """Return the preferred seed for representative visualizations.

    Parameters
    ----------
    config:
        Normalized configuration dictionary.

    Returns
    -------
    int
        Seed 42 when present, otherwise the first configured seed.
    """
    seeds = [int(seed) for seed in config["seeds"]]
    return 42 if 42 in seeds else seeds[0]


def fit_representative_models(
    config: dict[str, Any],
    degree: int | None = None,
    lam: float | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit raw-linear and ridge models for a representative seed.

    Parameters
    ----------
    config:
        Normalized configuration dictionary.
    degree:
        Ridge polynomial degree. Defaults to the first grid degree.
    lam:
        Ridge penalty. Defaults to the first grid lambda.
    seed:
        Dataset and split seed. Defaults to :func:`representative_seed`.

    Returns
    -------
    dict
        Fitted weights, standardization statistics, and split arrays.
    """
    config = normalize_config(config)
    seed = representative_seed(config) if seed is None else int(seed)
    degree = int(config["grid_degrees"][0] if degree is None else degree)
    lam = float(config["grid_lambdas"][0] if lam is None else lam)
    split = _prepare_split(config, seed)

    W_linear = fit_linear(split["X_train"], split["Y_train"])
    Phi_train = polynomial_features(split["X_train_s"], degree)
    W_ridge = fit_ridge(Phi_train, split["Y_train"], lam)
    return {
        **split,
        "seed": seed,
        "degree": degree,
        "lambda": lam,
        "W_linear": W_linear,
        "W_ridge": W_ridge,
    }


def _expand_sequence(spec: Any, dtype: type) -> list[Any]:
    if isinstance(spec, (list, tuple)):
        return [dtype(value) for value in spec]
    if isinstance(spec, dict) and "logspace" in spec:
        log_spec = spec["logspace"]
        values = np.logspace(
            float(log_spec["start"]),
            float(log_spec["stop"]),
            int(log_spec["num"]),
        )
        return [dtype(value) for value in values]
    if isinstance(spec, dict):
        start = int(spec["start"])
        stop = int(spec["stop"])
        step = int(spec.get("step", 1))
        return [dtype(value) for value in range(start, stop + 1, step)]
    raise TypeError(f"Unsupported sequence spec: {spec!r}")


def _prepare_split(config: dict[str, Any], seed: int) -> dict[str, NDArray[np.float64]]:
    X, Y = generate_dataset(config["N"], config["L1"], config["L2"], seed)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(
        X,
        Y,
        config["ratios"],
        seed,
    )
    X_train_s, X_val_s, X_test_s, mean, std = standardize_train_val_test(
        X_train,
        X_val,
        X_test,
    )
    return {
        "X": X,
        "Y": Y,
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test": Y_test,
        "X_train_s": X_train_s,
        "X_val_s": X_val_s,
        "X_test_s": X_test_s,
        "x_mean": mean,
        "x_std": std,
    }


def _run_all_seeds(config: dict[str, Any]) -> dict[str, NDArray[np.float64]]:
    seeds = np.asarray(config["seeds"], dtype=int)
    degrees = np.asarray(config["degrees"], dtype=int)
    lambdas = np.asarray(config["lambdas"], dtype=float)
    grid_degrees = np.asarray(config["grid_degrees"], dtype=int)
    grid_lambdas = np.asarray(config["grid_lambdas"], dtype=float)
    fixed_degree = int(config["ridge_fixed_degree"])

    n_seeds = len(seeds)
    n_degrees = len(degrees)
    n_lambdas = len(lambdas)
    n_grid_degrees = len(grid_degrees)
    n_grid_lambdas = len(grid_lambdas)
    ridge_n_coeffs = (fixed_degree + 1) * (fixed_degree + 2) // 2

    baseline_train = np.empty(n_seeds)
    baseline_val = np.empty(n_seeds)
    baseline_test = np.empty(n_seeds)
    baseline_angle_rmse = np.empty((n_seeds, 3))

    degree_train = np.empty((n_seeds, n_degrees))
    degree_val = np.empty((n_seeds, n_degrees))
    degree_test = np.empty((n_seeds, n_degrees))
    degree_weight_norm = np.empty((n_seeds, n_degrees))
    degree_condition = np.empty((n_seeds, n_degrees))

    ridge_train = np.empty((n_seeds, n_lambdas))
    ridge_val = np.empty((n_seeds, n_lambdas))
    ridge_test = np.empty((n_seeds, n_lambdas))
    ridge_weight_norm = np.empty((n_seeds, n_lambdas))
    ridge_condition = np.empty((n_seeds, n_lambdas))
    ridge_coefficients = np.empty((n_seeds, n_lambdas, ridge_n_coeffs, 2))

    grid_val = np.empty((n_seeds, n_grid_degrees, n_grid_lambdas))
    grid_best_degree = np.empty(n_seeds, dtype=int)
    grid_best_lambda = np.empty(n_seeds)
    grid_best_train = np.empty(n_seeds)
    grid_best_val = np.empty(n_seeds)
    grid_best_test = np.empty(n_seeds)

    for s_idx, seed in enumerate(seeds):
        split = _prepare_split(config, int(seed))
        L1 = config["L1"]
        L2 = config["L2"]

        W_linear = fit_linear(split["X_train"], split["Y_train"])
        Y_pred_train = predict_linear(W_linear, split["X_train"])
        Y_pred_val = predict_linear(W_linear, split["X_val"])
        Y_pred_test = predict_linear(W_linear, split["X_test"])
        baseline_train[s_idx] = mean_error_calc(split["X_train"], Y_pred_train, L1, L2)
        baseline_val[s_idx] = mean_error_calc(split["X_val"], Y_pred_val, L1, L2)
        baseline_test[s_idx] = mean_error_calc(split["X_test"], Y_pred_test, L1, L2)
        baseline_angle_rmse[s_idx] = [
            angle_rmse(split["Y_train"], Y_pred_train),
            angle_rmse(split["Y_val"], Y_pred_val),
            angle_rmse(split["Y_test"], Y_pred_test),
        ]

        for d_idx, degree in enumerate(degrees):
            Phi_train = polynomial_features(split["X_train_s"], int(degree))
            Phi_val = polynomial_features(split["X_val_s"], int(degree))
            Phi_test = polynomial_features(split["X_test_s"], int(degree))
            W_poly = fit_linear(Phi_train, split["Y_train"])
            degree_weight_norm[s_idx, d_idx] = np.linalg.norm(W_poly)
            degree_condition[s_idx, d_idx] = np.linalg.cond(Phi_train.T @ Phi_train)
            degree_train[s_idx, d_idx] = mean_error_calc(
                split["X_train"],
                predict_linear(W_poly, Phi_train),
                L1,
                L2,
            )
            degree_val[s_idx, d_idx] = mean_error_calc(
                split["X_val"],
                predict_linear(W_poly, Phi_val),
                L1,
                L2,
            )
            degree_test[s_idx, d_idx] = mean_error_calc(
                split["X_test"],
                predict_linear(W_poly, Phi_test),
                L1,
                L2,
            )

        Phi_train_r = polynomial_features(split["X_train_s"], fixed_degree)
        Phi_val_r = polynomial_features(split["X_val_s"], fixed_degree)
        Phi_test_r = polynomial_features(split["X_test_s"], fixed_degree)
        gram_r = Phi_train_r.T @ Phi_train_r
        ident_r = np.eye(gram_r.shape[0])
        for l_idx, lam in enumerate(lambdas):
            W_r = fit_ridge(Phi_train_r, split["Y_train"], float(lam))
            ridge_coefficients[s_idx, l_idx] = W_r
            ridge_weight_norm[s_idx, l_idx] = np.linalg.norm(W_r)
            ridge_condition[s_idx, l_idx] = np.linalg.cond(gram_r + float(lam) * ident_r)
            ridge_train[s_idx, l_idx] = mean_error_calc(
                split["X_train"],
                predict_ridge(W_r, Phi_train_r),
                L1,
                L2,
            )
            ridge_val[s_idx, l_idx] = mean_error_calc(
                split["X_val"],
                predict_ridge(W_r, Phi_val_r),
                L1,
                L2,
            )
            ridge_test[s_idx, l_idx] = mean_error_calc(
                split["X_test"],
                predict_ridge(W_r, Phi_test_r),
                L1,
                L2,
            )

        for d_idx, degree in enumerate(grid_degrees):
            Phi_train_g = polynomial_features(split["X_train_s"], int(degree))
            Phi_val_g = polynomial_features(split["X_val_s"], int(degree))
            for l_idx, lam in enumerate(grid_lambdas):
                W_g = fit_ridge(Phi_train_g, split["Y_train"], float(lam))
                grid_val[s_idx, d_idx, l_idx] = mean_error_calc(
                    split["X_val"],
                    predict_ridge(W_g, Phi_val_g),
                    L1,
                    L2,
                )

        flat_idx = int(np.nanargmin(grid_val[s_idx]))
        best_i, best_j = np.unravel_index(flat_idx, grid_val[s_idx].shape)
        best_degree = int(grid_degrees[best_i])
        best_lam = float(grid_lambdas[best_j])
        best_errors = _evaluate_ridge_split(config, int(seed), best_degree, best_lam)
        grid_best_degree[s_idx] = best_degree
        grid_best_lambda[s_idx] = best_lam
        grid_best_train[s_idx] = best_errors["train"]
        grid_best_val[s_idx] = best_errors["val"]
        grid_best_test[s_idx] = best_errors["test"]

    return {
        "seeds": seeds,
        "degrees": degrees,
        "lambdas": lambdas,
        "grid_degrees": grid_degrees,
        "grid_lambdas": grid_lambdas,
        "baseline_train": baseline_train,
        "baseline_val": baseline_val,
        "baseline_test": baseline_test,
        "baseline_angle_rmse": baseline_angle_rmse,
        "degree_train": degree_train,
        "degree_val": degree_val,
        "degree_test": degree_test,
        "degree_weight_norm": degree_weight_norm,
        "degree_condition": degree_condition,
        "ridge_train": ridge_train,
        "ridge_val": ridge_val,
        "ridge_test": ridge_test,
        "ridge_weight_norm": ridge_weight_norm,
        "ridge_condition": ridge_condition,
        "ridge_coefficients": ridge_coefficients,
        "grid_val": grid_val,
        "grid_best_degree": grid_best_degree,
        "grid_best_lambda": grid_best_lambda,
        "grid_best_train": grid_best_train,
        "grid_best_val": grid_best_val,
        "grid_best_test": grid_best_test,
    }


def _evaluate_ridge_split(
    config: dict[str, Any],
    seed: int,
    degree: int,
    lam: float,
) -> dict[str, float]:
    split = _prepare_split(config, seed)
    Phi_train = polynomial_features(split["X_train_s"], degree)
    Phi_val = polynomial_features(split["X_val_s"], degree)
    Phi_test = polynomial_features(split["X_test_s"], degree)
    W = fit_ridge(Phi_train, split["Y_train"], lam)
    return {
        "train": mean_error_calc(
            split["X_train"],
            predict_ridge(W, Phi_train),
            config["L1"],
            config["L2"],
        ),
        "val": mean_error_calc(
            split["X_val"],
            predict_ridge(W, Phi_val),
            config["L1"],
            config["L2"],
        ),
        "test": mean_error_calc(
            split["X_test"],
            predict_ridge(W, Phi_test),
            config["L1"],
            config["L2"],
        ),
    }


def _aggregate_arrays(
    arrays: dict[str, NDArray[np.float64]],
) -> dict[str, NDArray[np.float64]]:
    aggregate: dict[str, NDArray[np.float64]] = {}
    for key, value in arrays.items():
        if key in {"seeds", "degrees", "lambdas", "grid_degrees", "grid_lambdas"}:
            continue
        if value.ndim >= 1 and value.shape[0] == len(arrays["seeds"]):
            aggregate[f"{key}_mean"] = np.nanmean(value, axis=0)
            aggregate[f"{key}_std"] = np.nanstd(value, axis=0)
    return aggregate


def _build_summary(
    config: dict[str, Any],
    arrays: dict[str, NDArray[np.float64]],
    aggregate: dict[str, NDArray[np.float64]],
    run_id: str,
) -> dict[str, Any]:
    degrees = arrays["degrees"]
    lambdas = arrays["lambdas"]
    grid_degrees = arrays["grid_degrees"]
    grid_lambdas = arrays["grid_lambdas"]

    unreg_idx = int(np.nanargmin(aggregate["degree_val_mean"]))
    ridge_idx = int(np.nanargmin(aggregate["ridge_val_mean"]))
    grid_flat = int(np.nanargmin(aggregate["grid_val_mean"]))
    grid_i, grid_j = np.unravel_index(grid_flat, aggregate["grid_val_mean"].shape)

    aggregate_grid_degree = int(grid_degrees[grid_i])
    aggregate_grid_lam = float(grid_lambdas[grid_j])
    aggregate_grid = _evaluate_aggregate_ridge(
        config,
        aggregate_grid_degree,
        aggregate_grid_lam,
    )
    per_seed_train = arrays["grid_best_train"]
    per_seed_val = arrays["grid_best_val"]
    per_seed_test = arrays["grid_best_test"]

    def split_block(train: NDArray[np.float64], val: NDArray[np.float64], test: NDArray[np.float64]) -> dict[str, float]:
        return {
            "train_mean": float(np.mean(train)),
            "train_std": float(np.std(train)),
            "val_mean": float(np.mean(val)),
            "val_std": float(np.std(val)),
            "test_mean": float(np.mean(test)),
            "test_std": float(np.std(test)),
        }

    summary = {
        "run_id": run_id,
        "seeds": [int(seed) for seed in arrays["seeds"]],
        "linear_baseline": split_block(
            arrays["baseline_train"],
            arrays["baseline_val"],
            arrays["baseline_test"],
        ),
        "unregularized_polynomial": {
            "degree": int(degrees[unreg_idx]),
            **split_block(
                arrays["degree_train"][:, unreg_idx],
                arrays["degree_val"][:, unreg_idx],
                arrays["degree_test"][:, unreg_idx],
            ),
        },
        "ridge_fixed_degree": {
            "degree": int(config["ridge_fixed_degree"]),
            "lambda": float(lambdas[ridge_idx]),
            **split_block(
                arrays["ridge_train"][:, ridge_idx],
                arrays["ridge_val"][:, ridge_idx],
                arrays["ridge_test"][:, ridge_idx],
            ),
        },
        "ridge_joint_optimum": {
            "degree": aggregate_grid_degree,
            "lambda": aggregate_grid_lam,
            **aggregate_grid,
        },
        "per_seed_joint_optimum_stats": split_block(
            per_seed_train,
            per_seed_val,
            per_seed_test,
        ),
        "per_seed_joint_optimum": {
            "degree": [int(value) for value in arrays["grid_best_degree"]],
            "lambda": [float(value) for value in arrays["grid_best_lambda"]],
            "train": [float(value) for value in arrays["grid_best_train"]],
            "val": [float(value) for value in arrays["grid_best_val"]],
            "test": [float(value) for value in arrays["grid_best_test"]],
        },
    }

    if 42 in arrays["seeds"]:
        idx_42 = int(np.where(arrays["seeds"] == 42)[0][0])
        summary["seed_42"] = {
            "linear_val": float(arrays["baseline_val"][idx_42]),
            "linear_test": float(arrays["baseline_test"][idx_42]),
            "unregularized_best_degree": int(
                degrees[int(np.nanargmin(arrays["degree_val"][idx_42]))]
            ),
            "unregularized_best_val": float(np.nanmin(arrays["degree_val"][idx_42])),
            "ridge_fixed_best_lambda": float(
                lambdas[int(np.nanargmin(arrays["ridge_val"][idx_42]))]
            ),
            "ridge_fixed_best_val": float(np.nanmin(arrays["ridge_val"][idx_42])),
            "grid_best_degree": int(arrays["grid_best_degree"][idx_42]),
            "grid_best_lambda": float(arrays["grid_best_lambda"][idx_42]),
            "grid_best_val": float(arrays["grid_best_val"][idx_42]),
            "grid_best_test": float(arrays["grid_best_test"][idx_42]),
        }
        summary["ridge_notebook_joint_optimum"] = {
            "degree": int(summary["seed_42"]["grid_best_degree"]),
            "lambda": float(summary["seed_42"]["grid_best_lambda"]),
            **_evaluate_aggregate_ridge(
                config,
                int(summary["seed_42"]["grid_best_degree"]),
                float(summary["seed_42"]["grid_best_lambda"]),
            ),
        }
    return summary


def _evaluate_aggregate_ridge(
    config: dict[str, Any],
    degree: int,
    lam: float,
) -> dict[str, float]:
    train = []
    val = []
    test = []
    for seed in config["seeds"]:
        errors = _evaluate_ridge_split(config, int(seed), degree, lam)
        train.append(errors["train"])
        val.append(errors["val"])
        test.append(errors["test"])
    train_arr = np.asarray(train)
    val_arr = np.asarray(val)
    test_arr = np.asarray(test)
    return {
        "train_mean": float(train_arr.mean()),
        "train_std": float(train_arr.std()),
        "val_mean": float(val_arr.mean()),
        "val_std": float(val_arr.std()),
        "test_mean": float(test_arr.mean()),
        "test_std": float(test_arr.std()),
    }


def _persist_results(results: dict[str, Any], run_dir: Path) -> None:
    arrays = {
        key: value
        for key, value in results.items()
        if key != "metadata" and isinstance(value, np.ndarray)
    }
    metadata_json = json.dumps(results["metadata"])
    np.savez_compressed(run_dir / "results.npz", metadata_json=metadata_json, **arrays)


def _write_config(config: dict[str, Any], path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write experiment configs") from exc

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _update_latest(run_dir: Path, artifacts_path: Path) -> None:
    artifacts_path.mkdir(parents=True, exist_ok=True)
    latest = artifacts_path / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)

    try:
        latest.symlink_to(run_dir.name, target_is_directory=True)
    except OSError:
        shutil.copytree(run_dir, latest)
