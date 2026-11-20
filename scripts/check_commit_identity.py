"""Local guard against accidental AI attribution or overridden human Git identities.

This is an advisory hook, not authentication: Git metadata and local hooks can be changed.
"""

import re
import subprocess
import sys
from pathlib import Path

AI_IDENTITY = re.compile(r"(?:\b(codex|claude|chatgpt|copilot|openai|anthropic)\b|\[bot\])", re.I)


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def validate_identity(name, email, author, committer, message=""):
    if not name.strip() or not email.strip():
        raise ValueError("Configure your own Git user.name and user.email before committing.")
    expected = f"{name} <{email}>"
    if AI_IDENTITY.search(expected):
        raise ValueError("Use your human contributor identity, not an AI or bot identity.")
    for identity in (author, committer):
        if identity != expected:
            raise ValueError("Author and committer must match your configured human Git identity.")
    for line in message.splitlines():
        if re.match(r"\s*(co-authored-by|signed-off-by):", line, re.I) and AI_IDENTITY.search(line):
            raise ValueError(
                "Remove AI authorship/sign-off trailers; retain legitimate human credit."
            )


def main():
    try:
        message = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else ""
        # git var includes the effective identity, including environment overrides, then time/zone.
        author = git("var", "GIT_AUTHOR_IDENT").rsplit(" ", 2)[0]
        committer = git("var", "GIT_COMMITTER_IDENT").rsplit(" ", 2)[0]
        validate_identity(
            git("config", "user.name"), git("config", "user.email"), author, committer, message
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Commit accountability check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
