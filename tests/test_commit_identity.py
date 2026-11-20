import pytest

from scripts.check_commit_identity import validate_identity


def test_human_identity_and_human_coauthor():
    validate_identity(
        "Student",
        "student@example.edu",
        "Student <student@example.edu>",
        "Student <student@example.edu>",
        "Fix\n\nCo-Authored-By: Peer <peer@example.edu>",
    )


@pytest.mark.parametrize(
    "message",
    ["Co-Authored-By: Claude <noreply@anthropic.com>", "Signed-off-by: Codex <bot@openai.com>"],
)
def test_ai_trailer_rejected(message):
    with pytest.raises(ValueError, match="trailers"):
        validate_identity(
            "Student",
            "student@example.edu",
            "Student <student@example.edu>",
            "Student <student@example.edu>",
            message,
        )


def test_override_rejected():
    with pytest.raises(ValueError, match="match"):
        validate_identity(
            "Student",
            "student@example.edu",
            "Other <other@example.edu>",
            "Student <student@example.edu>",
        )


def test_bot_identity_rejected():
    with pytest.raises(ValueError, match="human"):
        validate_identity(
            "Codex", "bot@example.com", "Codex <bot@example.com>", "Codex <bot@example.com>"
        )
