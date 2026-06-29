from __future__ import annotations

import subprocess


def run_git_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    return result.stdout.strip()


def get_git_commit() -> str | None:
    return run_git_command(["rev-parse", "HEAD"])


def get_git_branch() -> str | None:
    return run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])


def is_git_dirty() -> bool | None:
    status = run_git_command(["status", "--porcelain"])

    if status is None:
        return None

    return bool(status)