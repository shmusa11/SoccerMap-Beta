# Testing and evaluation

## Automated checks

Install `requirements-dev.txt` and run `make check`. No API key, private video, PyTorch, or
model weights are needed. pytest discovers only `tests/`, so historical experiments in
`scratch/` and `pythonProject1/` do not accidentally run or download models.

| Suite | Coverage |
| --- | --- |
| test_video.py | Unknown frame counts, sampled FPS, invalid limits, capture cleanup, encoder fallback/failure |
| test_calibration.py | Rejected-sample lookup, static-motion gates, matrix validation, injected mapper expiry, pixel-unit RANSAC, decode limits |
| test_field_detector.py | Keypoint fallback retention, input resizing, default model weight paths |
| test_match_processor.py | Ball gap interpolation, sampled track interpolation, batch tails, repeat runs, model failures and cadence validation |
| test_app.py | Model-free Streamlit startup, upload gating, controls, and synthetic-video processing/export/cleanup |
| test_visualize.py | Honest projection legend and text fitting on smaller frames |
| test_commit_identity.py | Human identities, legitimate human coauthors, AI trailers and author overrides |
| test_secrets.py | Private config rejection, redacted staged scanning, missing-scanner failure, and personal-key loading without SDK fallback |

CI executes these checks on Python 3.11 and 3.12. Full inference and visual accuracy are not
established by a model-free pass. Record local Python/package versions and link a remote
workflow run after publishing the changes.

## Visual/model evaluation

Use approved local clips (not committed). Include one wide daytime soccer field, one night
clip, and one multi-sport venue. Keep a fixed raw-frame range and settings for comparisons.

1. Record source permission/identifier, source FPS/resolution, frame range, stride, pitch size,
   code revision, package/model versions/checksums, hardware, and runtime.
2. Inspect overlays at known frame indices. Compare field lines with ground-truth markings;
   track player feet, team consistency, official labels and true ball position.
3. Quantify visible landmark pixel errors and pitch errors where reference measurements exist.
   Report calibration coverage separately from accuracy; an incorrect full-coverage map fails.
4. Count player misses, identity switches, team/official errors, and ball false positives/gaps.
   Include uncalibrated regions and scene cuts rather than excluding difficult frames.
5. Play the exported MP4 in the target browser and download it. Compare duration with the
   sampled source range (allow one sampled-frame rounding difference); source audio is omitted.
6. Attach permitted representative overlays and metrics to the PR/issue. Record failures and
   unavailable assets explicitly. Do not describe old research numbers as new measurements.

## Known test gaps

Full model loading on a fresh machine, multi-session external import isolation, cancellation
through every experimental fallback, variable-frame-rate timing, and variable-pitch geometry
need additional coverage. TEAM_PLAN assigns the next evaluation and implementation tasks.

## Local maintenance verification (2026-09-20)

On macOS with Python 3.11.4: Ruff lint and format checks passed, all **56 tests passed**,
and maintained-source bytecode compilation and `git diff --check` passed. The test run
reported third-party Pyparsing deprecation warnings through the tracking dependencies.
Ruff 0.11.13 was installed in an isolated temporary tool directory for this run; contributors
install it from requirements-dev.txt. Remote CI and the Python 3.12 matrix remain unverified
until the workflow runs after a human publishes the changes.

The synthetic-video app test exercised real video decoding, rendering and MP4 encoding:
12 source frames at 60 FPS, stride 3, produced 4 readable frames at 20 FPS, with temporary
workspace cleanup. Expensive calibration/model outputs were mocked in that test.

Visual checks inspected a synthetic known-geometry overlay and unavailable-calibration views
from the first 12 frames of two local clips (daytime multi-sport and night). Each real sample
had three calibration keyframes and zero accepted anchors; this is an unavailable-state
smoke check, not an accuracy benchmark. A direct CPU PnLCalib check on a night frame loaded
the model and returned eight keypoints and an invertible homography, which still did not
establish acceptance by the independent geometry/line gates. Player/WASB inference and a
complete real-match export were not revalidated in this maintenance pass.

## Camera reuse and tracking regressions

The model-free suite covers four-point geometry, fixed-view checks, rejected views, automatic
inference reuse, source-coordinate UI scaling, saved setup persistence, reviewed automatic
views, memory-budget decisions, height advisories and synthetic pan registration. Ball tests
cover disconnected sequences, cuts, invalid candidates, seam overlap and bounded tile batches.
Generated fixtures do not establish accuracy on real matches.

Recovery tests interrupt tracking and rendering independently, verify that only completed
stages are reused without unnecessary model loads, preserve rejected samples and provenance,
and invalidate checkpoints when inputs change. They exercise atomic-write failure, corrupt
payloads, cancel/resume, completed-run immutability and the single-worker limit.

Passage tests begin at a nonzero frame that is not divisible by the sampling stride. They
check calibration, reference reuse, detector input triples, output frame indices, source and
playback timestamps, CSV fields, truncated end windows, unknown counts and inaccurate seeks.
The multi-sport fallback must propagate memory exhaustion instead of treating it as no field.

Browser verification should include the editor with mouse and keyboard, invalid/collinear
points, an overlay preview, saving and reusing a setup, mismatched views, cancel/retry, demo
provenance, exports and reopening saved analyses. Use one browser at a time on a busy laptop.
Do not run a model benchmark concurrently with browser capture or the full test suite.

## Measuring ball quality

New real-run JSON includes `ball_image` in original image pixels and `ball_measured` provenance.
Keep human annotations outside Git. Each annotation names the original source frame:

```json
[
  {"index": 0, "visible": true, "point": [640, 340]},
  {"index": 3, "visible": false}
]
```

These are schema examples, not observations from a match. Label visible and invisible samples
before comparing runs. Choose a pixel tolerance appropriate to the source resolution and keep
it fixed across the comparison. Unlabelled frames are excluded; unsampled labelled frames are
reported and count as missing predictions. A wrong location counts as both a false positive
and a miss. Empty metric denominators remain unavailable.

```bash
python -m soccer_analyzer.evaluation .soccermap/runs/RUN_ID/result.json scratch/ball-labels.json --tolerance-px 12
python -m soccer_analyzer.evaluation .soccermap/runs/RUN_ID/result.json scratch/ball-labels.json --tolerance-px 12 --measured-only
```

Report precision, recall, localization error and sampling settings together. Neither mapping
coverage nor the number of interpolated ball positions is ball accuracy.
