"""Rebuild features from included candidates, then retrain all correctness heads."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    # Step 1 - rebuild features
    build_cmd = [
        sys.executable,
        "-m",
        "code.downstream.build_correctness_dataset",
        "--skip_candidates",
        "--overwrite_features",
    ]
    subprocess.run(build_cmd, cwd=PROJECT_ROOT, check=True)

    # Step 2 - retrain heads
    train_cmd = [
        sys.executable,
        "-m",
        "code.downstream.train_all_correctness_heads",
    ]
    train_cmd.extend(sys.argv[1:])
    subprocess.run(train_cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
