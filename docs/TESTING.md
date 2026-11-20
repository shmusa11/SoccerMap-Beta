# Testing and evaluation

## Automated checks

Install `requirements-dev.txt` and run `make check`. No API key, private video, PyTorch or
model weights are needed. pytest discovers only `tests/`, and the browser logic in `web/` is
tested with Node's built-in test runner.

| Suite | Coverage |
| --- | --- |
| test_workspace.py | Settings validation, upload checks, run start/cancel/retry/delete, single worker, interrupted restart, run log, named missing models, provenance, CSV contract, download names |
| test_fast_workspace.py, test_fast_pipeline.py | Fast-engine result conversion, cloud consent, interpolation labels, camera-segment IDs, source frame indices |
| test_calibration.py, test_homography.py, test_joint_calib.py, test_field_markings.py, test_field_detector.py, test_multisport.py, test_registration.py, test_height_check.py | Field geometry, rejected samples, injected detectors, camera fits |
| test_match_processor.py, test_ball_track.py, test_ball_wasb.py, test_player_tracker.py, test_onfield.py, test_detection.py | Tracking, ball path gaps and cuts, on-field filtering |
| test_checkpoints.py | Stage checkpoints, stale or corrupt files, retry without reloading models |
| test_video.py, test_visualize.py | Frame limits, sampled frame rate, encoder fallback, overlay legend and class tags |
| test_camera_setup.py | Four-point setup, preview, reuse and rejected views |
| test_release_record.py, test_resources.py, test_evaluation.py | Run provenance, memory limits, ball-quality metrics |
| test_app.py | Streamlit start, upload gating and a synthetic end-to-end run |
| test_secrets.py | Private config rejection and staged secret scanning |
| web/*.test.js | Coverage counts, the five classes, frame lookup at boundaries, averages, back-line height, heatmap bins, passage windows, coach insights |

CI runs these checks on Python 3.11 and 3.12. A passing model-free suite does not establish
detection or calibration accuracy on real footage.

## Acceptance testing

Release acceptance is a separate pass against the running application: each requirement has a
positive and a negative case with a stated pass criterion. Cases drive the HTTP interface and
Chrome, use generated clips and injected result fixtures where a known answer is needed, and
use a real clip for the run, privacy and memory cases. Record the build, tester, date,
environment and the observed response of every step, and keep failures in the record.

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

Accuracy against independently labelled frames, cold-start timing, a second machine with the
full model stack, variable-frame-rate clips and browsers other than Chrome and Edge are not
covered yet.

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
