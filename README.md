# SoccerMap Beta

Turn a short soccer clip into an annotated video and an estimated field map. The local browser
workspace supports calibration, player tracking, optional ball tracking, synchronized review,
position exports, saved runs, and coaching views. Coverage is availability, not accuracy.

This is a standalone, sanitized release snapshot of `sprint4-recording-ui` at
`fa80b4f382b3e1ad38a03f0170a9b3de4285dd06`, with November 20 beta corrections described in
[release notes](docs/BETA_RELEASE.md). No source repository history, private footage, generated
match data, credentials, weights, or team/course records are included.

## Start the interface and synthetic demo

Use Python 3.11 and Node.js 18 or newer. In the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python workspace.py
```

Open http://127.0.0.1:8765 and choose **Use demo**, then **Run synthetic demo**. This deliberately
labeled fixture needs no models or API keys. It demonstrates the interface, not model accuracy.
On Windows, activate with `.venv\Scripts\Activate.ps1`.

## Process a real clip

Install `requirements.txt` and follow [model setup](docs/SETUP.md). Download weights separately;
none are included here. Configure local paths in your own `.env` using `.env.example`. Use footage
you have permission to process. Start with 300 original frames. Fast mode selects its own cadence;
choose classic mode for the detailed sampling controls. Saved camera setups use classic mode.

Cloud AI keyframes are **off by default** in the browser workspace. Enabling that checkbox sends
sampled frames to OpenAI using your own key, under a $0.30 per-run budget. Model acquisition may
need internet even with cloud analysis off. The experimental command-line pipeline has its own
explicit `--ai` option; review its help and [secret handling](docs/SECRETS.md).

## Verify the code

```sh
make check
```

This runs Ruff, Python tests, JavaScript tests, and compilation. Model-free tests exercise
contracts and failure handling; they do not establish real-world detection accuracy. The Python
checks use injected models and generated clips. CI runs on Python 3.11 and 3.12.

## Source map

- `workspace.py` and `web/`: local HTTP service, clip setup, review, exports and coaching tools.
- `soccer_analyzer/`: geometry, model adapters, fast/classic engines, tracking, export and evaluation.
- `tests/` and `web/*.test.js`: regression tests.
- `app.py`: alternate Streamlit interface; `run.py`, `calibrate.py`, `pythonProject1/`: legacy references.
- `docs/`: setup, architecture, coding standards, testing and provenance.
- `LICENSES/` and `web/vendor/`: upstream notices. See [third-party components](docs/THIRD_PARTY.md).

## Limitations

Mapping can remain unavailable on difficult footage. Ball tracking and team/role estimates are
not reliable ground truth. Off-camera players are absent. Fast runs recompute when restarted.
The prototype supports bounded clips, one operator and one worker; it does not support live
video or promise full-match throughput. Some runtime/model dependencies are not fully locked.
Retain upstream notices and observe each component's license and model/data terms.
