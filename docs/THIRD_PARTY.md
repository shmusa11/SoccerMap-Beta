# Third-party components and provenance

This inventory supports reproducible setup and the course's attribution requirement. It is
not a complete dependency audit. Preserve upstream notices and record exact versions and
model/data terms before distributing a release. Describing code as a prototype or loading
it through an adapter does not waive upstream license obligations.

| Component | Use | Upstream source / terms to retain |
| --- | --- | --- |
| PnLCalib | Field keypoint/line inference and camera fitting | https://github.com/mguti97/PnLCalib — GPL-2.0; `field_detector.projection_from_cam_params` explicitly identifies code replicated from its inference.py; keypoint table cites upstream coordinates |
| WASB-SBDT | Soccer ball heatmaps and model configuration | https://github.com/nttcom/WASB-SBDT — MIT, copyright NTT Communications Corporation; retain notice for reused model configuration/code; verify checkpoint terms |
| Ultralytics YOLO | Local person detection | https://github.com/ultralytics/ultralytics and https://www.ultralytics.com/license — AGPL-3.0 or applicable commercial terms; verify code/model use |
| Roboflow inference and model | Optional referee/goalkeeper classification | https://github.com/roboflow/inference — package and hosted model terms must be checked separately |
| OpenCV, NumPy, SciPy, scikit-learn | Vision, geometry, optimization and clustering | Preserve installed distributions' license metadata |
| supervision | ByteTrack and detection containers | https://github.com/roboflow/supervision — preserve installed version's license |
| Streamlit | User interface and UI tests | https://github.com/streamlit/streamlit — preserve installed version's license |
| PyTorch and torchvision | External neural model execution | Preserve installed distributions' notices and dependency terms |

External clones, checkpoints, sample datasets, and footage are local assets rather than
Git-tracked dependencies. Record the source URL, revision/checksum, access requirements,
license/permission, and intended use for each. The revisions and checksums tested for this
release are in [RELEASE_RECORD.md](RELEASE_RECORD.md).
