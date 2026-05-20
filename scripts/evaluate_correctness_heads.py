"""Evaluate the included trained correctness heads and write summary CSV files."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cmd = [
        sys.executable,
        "-m",
        "code.downstream.evaluate_correctness_story",
    ]
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
