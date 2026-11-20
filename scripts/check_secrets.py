"""Scan staged changes or all reachable Git history without displaying credential values."""

import argparse
import shutil
import subprocess
import sys
from pathlib import PurePosixPath


def is_private_config(path):
    file = PurePosixPath(path)
    return (
        file.name == ".env"
        or (file.name.startswith(".env.") and file.name != ".env.example")
        or (file.name == "secrets.toml" and ".streamlit" in file.parts)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="Check the commit index only")
    args = parser.parse_args()
    scanner = shutil.which("gitleaks")
    if scanner is None:
        print("Install Gitleaks before committing; see docs/SECRETS.md.", file=sys.stderr)
        return 1
    listing = (
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"]
        if args.staged
        else ["git", "ls-files", "-z"]
    )
    paths = subprocess.check_output(listing).decode().split("\0")
    private = [p for p in paths if p and is_private_config(p)]
    if private:
        print("Private config must not be tracked: " + ", ".join(private), file=sys.stderr)
        return 1
    command = [
        scanner,
        "git",
        ".",
        "--redact",
        "--no-banner",
        "--no-color",
        "--ignore-gitleaks-allow",
    ]
    command += ["--pre-commit", "--staged"] if args.staged else ["--log-opts=--all"]
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
