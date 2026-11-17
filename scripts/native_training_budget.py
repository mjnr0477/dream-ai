#!/usr/bin/env python3
"""Validate and inspect one shared DREAM native-training session deadline."""
from __future__ import annotations

import argparse
import math
import time

MAX_SESSION_SECONDS = 300


def validate_session_seconds(value: str | int) -> int:
    raw = str(value).strip()
    if not raw.isdigit():
        raise ValueError("DREAM_NATIVE_SESSION_MAX_SECONDS must be a whole number from 1 to 300.")
    seconds = int(raw)
    if not 1 <= seconds <= MAX_SESSION_SECONDS:
        raise ValueError("DREAM native training sessions are limited to 1–300 seconds.")
    return seconds


def create_deadline(seconds: str | int, now: float | None = None) -> float:
    budget = validate_session_seconds(seconds)
    started = time.monotonic() if now is None else float(now)
    if not math.isfinite(started) or started < 0:
        raise ValueError("Monotonic start time must be finite and non-negative.")
    return started + budget


def seconds_remaining(deadline: float, now: float | None = None) -> int:
    deadline = float(deadline)
    current = time.monotonic() if now is None else float(now)
    if not math.isfinite(deadline) or not math.isfinite(current) or deadline < 0 or current < 0:
        raise ValueError("Deadline and monotonic time must be finite and non-negative.")
    # Floor rather than round up: the caller must never receive more time than remains.
    return max(0, math.floor(deadline - current))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--seconds", required=True)
    deadline = commands.add_parser("deadline")
    deadline.add_argument("--seconds", required=True)
    remaining = commands.add_parser("remaining")
    remaining.add_argument("--deadline", required=True, type=float)
    args = parser.parse_args()

    try:
        if args.command == "validate":
            print(validate_session_seconds(args.seconds))
        elif args.command == "deadline":
            print(f"{create_deadline(args.seconds):.9f}")
        else:
            print(seconds_remaining(args.deadline))
    except ValueError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
