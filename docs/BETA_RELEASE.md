# November 20 beta release

Base: sprint4-recording-ui, fa80b4f382b3e1ad38a03f0170a9b3de4285dd06.

The release preserves the browser workspace and fast engine with these focused corrections:

1. Cloud frame analysis now requires an explicit per-run checkbox, defaulting off even if an
   OpenAI key is already configured. Missing keys fall back to local processing with a warning.
2. FastPipeline preserves interpolated player labels when converting tracking results. Downstream
   measured-only analytics can distinguish these estimates from detected observations.
3. Camera-compensated gap filling stops when a camera-motion chain is unavailable, including
   window edges. It no longer bridges a failed camera-motion step with an unverified box.
4. The interface offers fast/classic engine selection and explains which controls fast mode
   chooses itself. Privacy wording discloses the optional external frame transmission.

Regression tests cover opt-in/off behavior, missing credentials, conversion provenance and
failed-motion gap handling. Existing synthetic pan and occlusion tests remain part of the suite.
Full real-model accuracy and clean model setup on a second machine remain separate acceptance
activities. Historical match statistics are not republished as new measurements.

The original team's work is retained without rewriting its history. This public snapshot omits
raw footage/screenshots, generated analytics, historical camera calibrations, private project
planning records and local developer paths. The frozen prototype remains in pythonProject1;
its example input path is now local_clip.mp4. No new blanket license is asserted for dependencies.

## Acceptance fixes completed in the final beta package

New completed runs include input and source-tree SHA-256 fingerprints, a local weight inventory
with checksums, package versions, and the encoded video's codec, dimensions, frame count and FPS.
Synthetic output explicitly identifies its generated input and records no model execution.
CSV exports retain existing columns and add the required run, segment, original-frame, entity
and observation-status fields. Fast-path track IDs are scoped to verified camera-motion segments.

The default stopped-run action now retries as a NEW run, preserving the original attempt.
The checkpoint-resume API remains a separate advanced operation. History offers confirmed
analysis deletion that preserves original inputs and other runs, and refuses active-run deletion.
Observed-position summaries carry their run, interval and inclusion rules; back-line height
requires an explicit direction. Tests exercise exact average and 25/75-metre defender fixtures.

The accompanying course demo reuses the existing captioned interim recording at the user's
request. It shows the established analysis/coaching workflow; it predates these beta corrections.
Full-model accuracy and deployment limitations remain documented rather than inferred from
successful contract tests.
