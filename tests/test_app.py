from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_models_and_requires_upload():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception
    assert app.button[0].disabled
    assert "Soccer" in app.title[0].value
    assert [control.label for control in app.checkbox] == [
        "AI keyframes (OpenAI vision model)",
        "AI field markings (fast pipeline calibration)",
        "AI ball checks (fast pipeline)",
        "Detect players + ball (local YOLO)",
        "Enhance dark/night footage",
        "Re-verify each rendered frame against real lines",
    ]


def test_processing_exports_sampled_video_without_models(tmp_path):
    import cv2
    import numpy as np

    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 60, (160, 100))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(np.full((100, 160, 3), (30, 120, 30), np.uint8))
    writer.release()
    # Exercise the actual app orchestration/render/export; only the expensive model stage is fake.
    script = f"""
from pathlib import Path
exec(compile(Path("app.py").read_text(), "app.py", "exec"))
from types import SimpleNamespace
_setup = lambda path, pitch: (None, None, None, ["test detectors"])
calibrate_clip_auto = lambda *args, **kwargs: (
    {{i: np.diag([105/160, 68/100, 1.0]) for i in range(12)}},
    {{"frames": 12, "covered": 12}},
)
maxf, stride = 12, 3
engine = "Classic three-pass"
upload = SimpleNamespace(name="source.avi", getvalue=lambda: Path({str(source)!r}).read_bytes())
with tempfile.TemporaryDirectory() as workspace, ExitStack() as resources:
    process_upload(upload, workspace, resources)
    with open_capture(str(Path(workspace) / "result.mp4")) as result:
        st.session_state["export_fps"] = result.get(cv2.CAP_PROP_FPS)
        st.session_state["export_frames"] = result.get(cv2.CAP_PROP_FRAME_COUNT)
        st.session_state["export_readable"] = result.read()[0]
st.session_state["workspace_cleaned"] = not Path(workspace).exists()
"""
    app = AppTest.from_string(script).run(timeout=30)
    assert not app.exception
    assert not app.error
    assert app.session_state["export_fps"] == 20
    assert app.session_state["export_frames"] == 4
    assert app.session_state["export_readable"]
    assert app.session_state["workspace_cleaned"]
    assert "4/4 output frames calibrated" in app.success[-1].value


def test_ai_options_are_disabled_and_ignored_without_ai_keyframes(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-key")
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    boxes = {c.label: c for c in app.checkbox}
    assert not boxes["AI field markings (fast pipeline calibration)"].disabled
    boxes["AI keyframes (OpenAI vision model)"].uncheck().run(timeout=20)
    boxes = {c.label: c for c in app.checkbox}
    for label in (
        "AI field markings (fast pipeline calibration)",
        "AI ball checks (fast pipeline)",
    ):
        assert boxes[label].disabled and not boxes[label].value
