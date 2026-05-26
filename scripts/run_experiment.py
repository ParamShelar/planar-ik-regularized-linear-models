from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from planar_ik.experiments import load_config, run_full_study
from planar_ik.models import assert_torch_numpy_ridge_parity


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Run the planar IK regularization study.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--parity-check", action="store_true", help="Run NumPy/PyTorch ridge parity check.")
    parser.add_argument("--seeds", nargs="*", type=int, help="Override config seeds.")
    parser.add_argument("--run-id", help="Optional artifact run id.")
    return parser.parse_args()


def main() -> None:
    """Run the experiment CLI.

    Returns
    -------
    None
        Writes artifacts and prints headline results.
    """
    args = parse_args()
    if args.parity_check:
        assert_torch_numpy_ridge_parity()
        print("PyTorch parity check passed.")

    config = load_config(args.config)
    if args.seeds:
        config["seeds"] = args.seeds

    results = run_full_study(config, run_id=args.run_id, artifacts_dir=ROOT / "artifacts")
    summary = results["metadata"]["summary"]
    print(f"Wrote artifacts/{summary['run_id']}")
    print(json.dumps(_headline(summary), indent=2))


def _headline(summary: dict) -> dict:
    return {
        "linear_baseline_val_mm": summary["linear_baseline"]["val_mean"],
        "best_unregularized": {
            "degree": summary["unregularized_polynomial"]["degree"],
            "val_mm": summary["unregularized_polynomial"]["val_mean"],
        },
        "ridge_fixed_degree": {
            "degree": summary["ridge_fixed_degree"]["degree"],
            "lambda": summary["ridge_fixed_degree"]["lambda"],
            "val_mm": summary["ridge_fixed_degree"]["val_mean"],
        },
        "ridge_joint_optimum": {
            "degree": summary["ridge_joint_optimum"]["degree"],
            "lambda": summary["ridge_joint_optimum"]["lambda"],
            "val_mm": summary["ridge_joint_optimum"]["val_mean"],
            "test_mm": summary["ridge_joint_optimum"]["test_mean"],
        },
    }


if __name__ == "__main__":
    main()
