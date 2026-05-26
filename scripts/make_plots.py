from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from planar_ik.experiments import load_results
from planar_ik.plots import make_all_plots


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments.
    """
    parser = argparse.ArgumentParser(description="Generate report figures from an artifact run.")
    parser.add_argument("--run", default="artifacts/latest", help="Path to artifact run directory.")
    return parser.parse_args()


def main() -> None:
    """Run the plot-generation CLI.

    Returns
    -------
    None
        Writes report figures.
    """
    args = parse_args()
    run_path = Path(args.run)
    if not run_path.is_absolute():
        run_path = ROOT / run_path
    results = load_results(run_path)
    outputs = make_all_plots(results, ROOT / "reports" / "figures")
    print("Wrote figures:")
    for path in outputs:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
