"""Recompute frozen transition-model features from the included candidate plans."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "code.downstream.build_validity_dataset",
        "--skip_candidates",
        "--overwrite_features",
    ]
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
