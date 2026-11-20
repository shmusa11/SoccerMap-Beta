# Release record: beta, November 20, 2026

The versions and files this release was tested with. Every completed run also writes its own
record (input fingerprint, source fingerprint, package versions, model checksums and the
computer it ran on) into `manifest.json` and the downloadable run details.

## Tested environment

macOS 27 on an Apple M2 Pro with 32 GB RAM, Python 3.11.4, Node.js 22.15, Chrome 158 and
Edge 157. CI runs the model-free checks on Ubuntu with Python 3.11 and 3.12.

## Python packages

Direct dependencies from `requirements.txt` and `requirements-dev.txt`, at the versions used
for the real-footage runs and the acceptance tests.

| Package | Tested version |
| --- | --- |
| `numpy` | `2.2.6` |
| `opencv-python` | `4.12.0.88` |
| `scipy` | `1.16.1` |
| `scikit-learn` | `1.8.0` |
| `supervision` | `0.25.1` |
| `torch` | `2.8.0` |
| `ultralytics` | `8.3.186` |
| `inference` | `0.53.0` |
| `onnxruntime` | `1.21.1` |
| `Pillow` | `11.3.0` |
| `matplotlib` | `3.10.5` |
| `tqdm` | `4.65.0` |
| `streamlit` | `1.50.0` |
| `python-dotenv` | `1.0.1` |
| `psutil` | `5.9.0` |
| `pytest` | `8.4.1` |
| `ruff` | `0.11.13` |

## External models

Weights are downloaded separately and never committed. Compare a local file with
`shasum -a 256 <file>`.

| File | Used for | SHA-256 |
| --- | --- | --- |
| `SV_kp` | Field keypoints (PnLCalib), required | `7ea78fa76aaf94976a8eca428d6e3c59697a93430cba1a4603e20284b61f5113` |
| `SV_lines` | Field lines (PnLCalib), required | `d72f4ed71734a2e3df9fa084f666e9b8adaef21bf69bac8952d6d3f970ff7455` |
| `wasb_soccer_best.pth.tar` | Ball detection (WASB), optional | `d0369572807c2baf751880d6cdf3cce9fc6283fa8d153f18af6baf4e64d2646c` |
| `yolo11s.pt` | Player detection (Ultralytics YOLO11s) | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |

## External repositories

| Repository | Revision |
| --- | --- |
| https://github.com/mguti97/PnLCalib | `8c87391d6f4ea40c5e4d65e61529916c7a49ce62` |
| https://github.com/nttcom/WASB-SBDT | `923462cacdeb3353b84ddebdedb3f4b7a8553b0f` |

## Unresolved

- `requirements.txt` lists the model stack without version bounds. Install the versions above
  when reproducing a real-footage run; newer releases are untested.
- Transitive dependencies are not pinned. The versions actually loaded are written into each
  run's `package_versions` record.
- The Roboflow officials model (classic engine) is optional, downloaded on first use and not
  pinned. In the fast engine, referee and goalkeeper roles come from the optional cloud
  keyframes; with cloud AI off those people are shown as Team A, Team B or unknown.
