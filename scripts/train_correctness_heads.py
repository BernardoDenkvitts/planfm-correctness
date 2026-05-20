"""Retrain correctness heads from the included feature matrices."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "code.downstream.train_all_correctness_heads",
        "--skip_build",
    ]
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
