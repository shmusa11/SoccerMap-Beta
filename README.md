# SoccerMap

SoccerMap turns a short soccer clip into an annotated video and an estimated field map that a
coach can review in the browser. It runs on one laptop. It shows only what the camera saw, and
it leaves the map empty whenever it cannot confirm the field.

This repository is the beta release (November 20, 2026). See the
[release notes](docs/BETA_RELEASE.md) for what changed and what is still limited.

## What it does

- **Import** one MP4, MOV, AVI or M4V clip and check it before any model runs.
- **Set up** the field size, the passage to analyze and the engine. Settings are frozen when a
  run starts.
- **Analyze** in a separate worker: player detection and tracking, field calibration, optional
  ball tracking, then rendering. One run at a time; a run can be cancelled and retried.
- **Review** the film and the field map on one timeline, with Team A, Team B, goalkeeper,
  referee and unknown shown by shape as well as colour.
- **Coach tools**: defense coverage, drawing, a 3D view, match stats, average positions,
  back-line height and a player heatmap.
- **Export** the annotated MP4, a positions CSV, the run details and a run log, and reopen
  saved analyses later without processing them again.

## Quick start: the synthetic demo

Needs Python 3.11 and Node.js 18 or newer. No models, footage or API key.

```sh
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python workspace.py
```

Open http://127.0.0.1:8765, choose **Use demo**, then **Run synthetic demo**. The demo is
generated data that shows the interface, including a deliberate gap where the map is
unavailable. The [quick start](docs/QUICKSTART.md) walks from here to an exported result.

## Analyze a real clip

1. Install `requirements.txt` and the models described in [setup](docs/SETUP.md). Weights are
   downloaded separately and are never part of this repository.
2. Put the model paths in your own `.env` (copy `.env.example`).
3. Start the workspace, check **Environment**, import a clip you have permission to use, and
   start with 300 frames.

Cloud AI keyframes are **off by default**. Ticking that box for a run sends a few sampled
frames to OpenAI with your own key, capped at $0.30 per run. With it off, nothing leaves the
computer. Referee and goalkeeper labels in the fast engine come from those keyframes; without
them, people are shown as Team A, Team B or unknown.

On a computer with a GPU (including Apple silicon), untick **Low memory mode** for 1080p
clips. Low memory mode runs every model on the CPU and can reach the 4 GB worker limit.

## Check the code

```sh
make check
```

This runs Ruff (lint and format), the Python tests, the JavaScript tests and a compile pass.
The tests use generated clips and stand-in models, so they need no weights, footage or
network. CI runs the same command on Python 3.11 and 3.12 and scans for secrets.

## Tested with

macOS 27 on an Apple M2 Pro (32 GB RAM), Python 3.11.4, Chrome 158 and Edge 157. Exact
package versions and model checksums are in the [release record](docs/RELEASE_RECORD.md).

## Source map

| Path | Contents |
| --- | --- |
| `workspace.py` | Local HTTP server: clips, runs, the worker process, exports |
| `web/` | Browser interface: setup, review, coach tools, analytics and their tests |
| `soccer_analyzer/` | Calibration, detection, tracking, the fast and classic engines, exports |
| `tests/` | Python regression tests |
| `app.py` | Alternate Streamlit interface |
| `run.py`, `calibrate.py` | Earlier command-line tools, kept for reference |
| `scripts/` | Secret scan and the command-line fast pipeline |
| `docs/` | Setup, quick start, architecture, testing, coding standards, release record |
| `LICENSES/`, `web/vendor/` | Third-party notices; see [third-party components](docs/THIRD_PARTY.md) |

## Known limits

- Mapping can stay unavailable on hard footage, such as fields with football or lacrosse lines.
- Ball tracking is sparse at night, so passes and possession are estimates at best.
- A track ID follows a detected track. It is not a verified player identity.
- Players the camera does not see are absent; nothing is filled in for them.
- Clips are limited to 7200 input frames per run. Live video and full-match processing are
  not supported.
- Models are loaded again for every run, and a retried fast run starts from the beginning.
