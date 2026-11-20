# Setup

## Environment

Run from the repository root. Use an isolated Python 3.11 virtual environment; do not depend
on a teammate's global Conda packages. CI also checks the model-free code on Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

For tests/UI startup alone, install only `requirements-dev.txt`. Runtime model dependencies
are not fully locked yet: record a working environment and external repository revisions
before comparing results. Do not blindly upgrade the research stack together.

## Field calibration: PnLCalib (required for processing)

Follow the [upstream repository](https://github.com/mguti97/PnLCalib) for model downloads.
Clone outside this repository and download the single-view `SV_kp` and `SV_lines` weight
files linked there. The app expects these exact filenames, without a `.pt` suffix, inside
the clone. If your downloads have a suffix, copy/rename them to these names locally.

```bash
# Example sibling location; run while in SoccerMapProject.
git clone https://github.com/mguti97/PnLCalib ../PnLCalib
```

Set `PNLCALIB_ROOT` in `.env` to that absolute directory. Install the upstream dependencies
in the same virtual environment, resolving conflicts rather than silently replacing working
versions. The directory must include `config/hrnetv2_w48.yaml`, `config/hrnetv2_w48_l.yaml`,
`SV_kp`, and `SV_lines`. See [third-party provenance](THIRD_PARTY.md).

## Players and officials

COCO YOLO handles people, on raw frames (no night enhancement). Put the Ultralytics `yolo11s.pt`
in `models/` for the best people recall; otherwise existing local `pythonProject1/yolov8n.pt`
weights are recognized, and failing that the adapter uses `models/yolov8n.pt`, downloading the
standard model on first use through Ultralytics. Internet is required for initial downloads. Keep
weights out of Git.

For referee/goalkeeper labels, follow [personal key setup](SECRETS.md) and put your own
`ROBOFLOW_API_KEY` in your local `.env`. The optional `inference`
runtime loads `football-players-detection-3zvbc/9` locally. Without a successful load, people
are still detected but officials classification is unavailable. Do not assume cold-start
model loading works offline; verify the cache on the intended demo machine.

## Ball detection: WASB (optional)

Clone [WASB-SBDT](https://github.com/nttcom/WASB-SBDT) alongside this repository, follow its
pretrained-model instructions, and place `wasb_soccer_best.pth.tar` at the clone root.
Set `WASB_ROOT` in `.env`. The adapter imports `src/models/hrnet.py` and loads this checkpoint.
Use trusted upstream checkpoints. Without WASB the app reports ball detection unavailable;
the current offline pipeline does not substitute the older online ball tracker.

## Run and check

```bash
python -m streamlit run app.py
make check
```

`make check` runs `ruff` (lint + format check), `pytest` and a compile pass. `ruff` is pinned in
`requirements-dev.txt`; if it is missing, `make check` stops with an install hint. To keep it
out of a shared base environment, use a venv and point the Makefile at it:

```bash
python -m venv .venv-dev && .venv-dev/bin/pip install -r requirements-dev.txt
make check PYTHON=.venv-dev/bin/python
```

Upload a short MP4/MOV/AVI/M4V. Start with standard 105×68 m geometry and default settings.
Long/high-resolution clips take longer than playback; choose a modest raw-frame limit.
Models are cached per browser session and can consume substantial memory across sessions.

The calibration-only visualizer is separate from the current three-pass app:

```bash
python -m soccer_analyzer.calibration.visualize /path/to/clip.mp4 \
  --pnlcalib-root /path/to/PnLCalib --max-frames 60
```

It reads the shell environment rather than the app's `.env` loader; pass the root explicitly
or export it. `run.py` remains the legacy pipeline, not the current app's CLI equivalent.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| App opens but setup fails | Root path, exact weight names, upstream dependencies |
| No ball / no official labels | Optional model availability messages, WASB path, Roboflow key/cache |
| No pitch mapping | Poor calibration is a valid result; inspect a wide daytime baseline first |
| Blank/broken download | Check encoder errors; H.264 is preferred, MPEG-4 browser support varies |
| Import errors around `model`, `models`, `utils`, `inference` | Restart the process after changing external model paths; adapters load external repositories |
| CPU processing is slow | Reduce frame count; hardware acceleration depends on installed PyTorch/platform |
| `make` unavailable | Run the four Python check commands in CONTRIBUTING.md |

Local assets are not supplied by a clean clone. Obtain approved footage separately and keep
its permission/source record with evaluation results. Avoid uploading private footage into issues.
