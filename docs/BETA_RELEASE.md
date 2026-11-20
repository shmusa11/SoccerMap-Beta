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
