#!/usr/bin/env python
"""Unified command entry point for the main Pick-and-Place workflows."""

from __future__ import annotations

import argparse
import subprocess
import sys


MODULES = {
    "demo": "lessons.lesson_07_pick_place_demo",
    "dls": "benchmark.ik_compare",
    "compare": "benchmark.ik_compare",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rule-based demo, one DLS episode, or the paired benchmark."
    )
    parser.add_argument("command", choices=MODULES)
    args, forwarded = parser.parse_known_args()

    if args.command == "dls":
        forwarded = ["--controllers", "dls", "--episodes", "1", *forwarded]

    command = [sys.executable, "-m", MODULES[args.command], *forwarded]
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
