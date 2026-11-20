import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import check_secrets
from soccer_analyzer.detection import SoccerDetector


@pytest.mark.parametrize("path", [".env", "config/.env", ".env.local", ".streamlit/secrets.toml"])
def test_private_config_paths(path):
    assert check_secrets.is_private_config(path)


def test_example_is_shareable():
    assert not check_secrets.is_private_config(".env.example")


def test_hook_blocks_forced_env_add(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "--staged"])
    monkeypatch.setattr(check_secrets.shutil, "which", lambda _: "/bin/gitleaks")
    monkeypatch.setattr(check_secrets.subprocess, "check_output", lambda _: b".env\0")
    scan = Mock()
    monkeypatch.setattr(check_secrets.subprocess, "run", scan)
    assert check_secrets.main() == 1
    scan.assert_not_called()


def test_scan_is_redacted_and_checks_index(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "--staged"])
    monkeypatch.setattr(check_secrets.shutil, "which", lambda _: "/bin/gitleaks")
    monkeypatch.setattr(check_secrets.subprocess, "check_output", lambda _: b"app.py\0")
    scan = Mock(return_value=SimpleNamespace(returncode=1))
    monkeypatch.setattr(check_secrets.subprocess, "run", scan)
    assert check_secrets.main() == 1
    command = scan.call_args.args[0]
    assert "--redact" in command
    assert "--staged" in command
    assert "--ignore-gitleaks-allow" in command


def test_missing_scanner_blocks_commit(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "--staged"])
    monkeypatch.setattr(check_secrets.shutil, "which", lambda _: None)
    assert check_secrets.main() == 1


def test_no_key_never_falls_back_to_sdk_credentials(monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", Mock())
    inference = SimpleNamespace(get_model=Mock())
    monkeypatch.setitem(sys.modules, "inference", inference)
    assert SoccerDetector._load_soccer("test-model") is None
    inference.get_model.assert_not_called()


def test_explicit_personal_key_is_used(monkeypatch):
    key = "test-" + "x" * 24
    monkeypatch.setenv("ROBOFLOW_API_KEY", key)
    monkeypatch.setattr("dotenv.load_dotenv", Mock())
    inference = SimpleNamespace(get_model=Mock())
    monkeypatch.setitem(sys.modules, "inference", inference)
    assert SoccerDetector._load_soccer("test-model") is inference.get_model.return_value
    inference.get_model.assert_called_once_with(model_id="test-model", api_key=key)
