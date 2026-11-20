from pathlib import Path
from soccer_analyzer.release_record import file_sha256, provenance


def test_input_hash_distinguishes_same_named_files_and_source_versions(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PNLCALIB_ROOT", str(tmp_path / "no-models"))
    monkeypatch.setenv("WASB_ROOT", str(tmp_path / "no-ball"))
    clip = tmp_path / "same.mp4"
    clip.write_bytes(b"first")
    first = provenance(clip, root)
    clip.write_bytes(b"other")
    second = provenance(clip, root)
    assert first["input_fingerprint"] != second["input_fingerprint"]
    assert first["code_version"] == second["code_version"]
    assert file_sha256(clip) == second["input_fingerprint"]["sha256"]
    assert all("sha256" in m and "component" in m for m in second["model_identifiers"])
    assert str(tmp_path) not in str(second)
