# Beta release notes: November 20, 2026

The beta is built on the Sprint 4 browser workspace and the fast keyframe engine from the
interim release. This file lists what changed for the beta and what is still limited.

## Privacy and consent

- Cloud AI keyframes need an explicit tick for each run and default to off, even when an
  OpenAI key is configured. A missing key falls back to local processing with a warning.
- The setup and progress screens say when frames would leave the computer.

## Honest positions

- The fast engine keeps the "filled-in" label on interpolated player positions, so analytics
  that use measured positions can leave them out.
- Gap filling stops where camera motion could not be verified. It no longer bridges a cut.
- Track IDs are scoped to a camera segment, and the CSV carries the segment ID.
- A seek now lands the film on the same sampled frame as the map. Before, a seek to a frame
  boundary could show the film one sample behind.

## Review and labelling

- Team A, Team B, goalkeeper, referee and unknown each have their own marker shape and legend
  entry, and video boxes carry a text tag (A, B, GK, REF, ?). Nothing depends on colour alone.
- Coverage is shown as counts and a percentage, for example "75 of 150 sampled frames mapped
  (50.0%)". A run with no frames shows no result.
- The map heading names the pitch outline as estimated geometry.
- Every time, distance and count on screen names its unit.
- Heatmaps use 5 m bins and state the bin size, sample count and time range. Average positions
  list their sample counts. Back-line height needs an explicitly chosen attacking direction.

## Runs, recovery and exports

- Runs report detection and tracking, calibration and rendering as separate stages.
- Each run writes `run.log` with the run ID, stage, elapsed time and error type. It contains
  no error messages, frames, paths or environment values.
- A missing required model is named before a run starts, and an unreadable weight file fails
  with a next step.
- Retry creates a new run and keeps the stopped attempt. An interrupted run says so after a
  restart. Deleting an analysis asks first and keeps the original clip.
- Completed runs record the input SHA-256, a source fingerprint, package versions, model
  checksums, the encoded video's codec and size, and the computer they ran on.
- The CSV adds `run_id`, `segment_id`, `frame_index`, `entity_class` and `observation_status`.
  Downloads are named after their run.
- Environment reports this computer's OS, CPU, RAM, free disk and Python version, and says
  when it is below the tested minimum of 16 GB RAM and 5 GB free disk.

## Known issues

- Low memory mode runs every model on the CPU. On a 1920 x 886 clip it reached the 4 GB
  worker limit and stopped with a message. Untick it on a computer with a GPU.
- Models are loaded again for every run.
- With cloud AI off, the fast engine does not assign referee or goalkeeper roles.
- Calibration can stay unavailable on shared fields, and night ball detections are sparse.
- Accuracy against hand-labelled frames has not been measured. Coverage is not accuracy.
- Only Chrome and Edge on macOS were tested.

## Checking this release

`make check` runs the lint, format, Python, JavaScript and compile checks without models,
footage or network access. CI runs it on Python 3.11 and 3.12. Tested package versions and
model checksums are in [RELEASE_RECORD.md](RELEASE_RECORD.md).
