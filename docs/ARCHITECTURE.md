# Architecture

For a possible next direction, see [reviewable mapping by camera segment](ARCHITECTURE_PROPOSAL.md).
That proposal is unimplemented and should be evaluated against the baseline described here.

The app has two engines. The **fast keyframe pipeline** (`fast_pipeline.py`, default) is
described in section 0. The **classic three-pass pipeline** (sections 1-3) is kept as the
comparison baseline. Both assume a soccer pitch is a known plane and estimate an image-pixel to
pitch-metre homography for each usable viewpoint.

## 0. Fast keyframe pipeline

Profiling the classic pipeline showed ~80% of wall time in calibration (the field CNN on every
4th raw frame, then a whole-clip re-run at stride 2). The fast pipeline decodes once and runs
expensive models only on sparse keyframes:

| Stage | Module | What runs |
| --- | --- | --- |
| Camera motion | `fast_pipeline.frame_motion` | LK flow homography between rendered frames (640 px) |
| AI field markings (with an oracle) | `calibration/joint_calib.py` (default), `calibration/field_markings.py`, `calibration/refcore.py` | Default (`markings_method="joint"`, round 4): the camera only turns and zooms, so the whole window is fitted with ONE shared camera centre. After the sweep, keyframes are chosen by view coverage (widest view first, then the views farthest in pan / tilt / zoom, ≥ 1 s apart, skipping candidates whose paint an overlay hides; 2 per motion segment, doubled up to 8 when nothing is confirmed, +1 when views stay uncovered) and gpt-6-sol names their soccer markings; when any answer gives a non-white soccer-line colour it also reads the American-football yard numbers at two keyframes and, if one is read, aligns the football grid too (assumes the soccer pitch is centred on the football field). A video left unconfirmed in 2 windows is not asked again (per-video note in the oracle cache; a window counts only when the full keyframe budget of answers naming markings came back with no AI call skipped for budget or failed, and a confirmed window resets the note). One robust bundle fits every render frame's rotation and focal, the shared centre and the pitch dims to the landmarks, the camera motion, direct SIFT registrations between keyframes and the paint of many frames (forward and backward from every keyframe). A frame is shown only if two fits from disjoint halves of the keyframe answers both agree with it within 1.5 m (median) and 3 m (worst tenth of the view), at least half of its view was seen by a keyframe, no fitted pitch dimension of its segment is pinned on its allowed bound, a repeat of the whole fit with perturbed inputs (camera motion jittered AND keyframe images resampled to 0.67x and back) also confirms it within 1.5 m (median) and 3 m (worst tenth), its registration is consistent and its own paint mask supports it; everything else stays unavailable. When more keyframes are asked after a confirmed attempt, the LAST attempt decides (frames both attempts show must agree within the same tolerances); an attempt with less evidence is never preferred. Legacy (`markings_method="window"`): Every 2 s gpt-6-sol names the visible soccer markings with confidences; a constrained pinhole camera (fitted pitch dims) is fitted per keyframe from its own answer (answers within ±3 s only when its own is too thin); a keyframe is kept only when a fit from different answers agrees within 2.5 m, the two keyframes related by direct SIFT registration (not the motion chain); confirmed fits are then pooled, carried to every render frame and verified against that frame's 960 px paint mask. When any keyframe is kept it replaces the anchor / propagation / multi-sport path below. When answers exist but none is confirmed, the multi-sport fallback is skipped and detector-path frames must agree with the nearest keyframe's markings (median ≤ 2.5 m) or stay unavailable; the field CNN anchors only every 6 s while markings are asked |
| Anchors | `FastPipeline._anchor` | Field CNN only at the start, after camera motion, or every 3 s; raw and CLAHE input tried until the first verified anchor |
| Gates | `clip_calibrator.select_calibration`, `calibration/homography.py` | Sign-normalised H; real pinhole camera above the pitch (no shear/fold/mirror); camera beside the field preferred; spread; line support >= 0.42 |
| Pooled refit | `FastPipeline._pooled_refine`, `ls_calibrator.refine_homography` | Median camera position of the anchors held fixed; rotation + focal re-fit to the painted lines |
| Propagation | `FastPipeline._propagate` | Anchors carried along the chained motion both ways; every detection frame re-verified against its own line mask; failures stay unavailable |
| Players | `FastPipeline.run`, `player_tracker.py` | YOLO11s (no CLAHE, boxes down to conf 0.08) at ~7.5 Hz, then an offline camera-compensated tracker: two-stage association, 3 s memory, duplicate suppression, stitching, gap fill that follows the pan; every player has a box on every rendered frame |
| AI keyframes (optional) | `vision_oracle.py` | Every 2 s a vision model returns JSON boxes with team/role; on failed anchors (only when the markings path is off) it names fixed-vocabulary landmarks that seed a calibration |
| Labels | `vote_labels`, `FastPipeline._labels` | Oracle votes give each track its role, team and on-field status; jersey colours learnt from those tracks label the rest |
| Ball | `ball_wasb.peaks`, `ball_track.py` | Every render frame: top-20 WASB heatmap peaks (CLAHE input at night) + camera-compensated motion blobs against the neighbouring distinct raw frames; logistic appearance score; one second-order DP over the window with a physical speed limit and a "no ball" state; with an oracle, ~1 Hz gpt-6-luna crop checks anchor it (skipped at night). Frames without a trusted ball stay empty |
| Render | `FastPipeline.render` | Draws the tracker's boxes (oracle boxes the detector missed seed tracks) minus off-field tracks; frames are counted sequentially, never seeked |

Without an `OPENAI_API_KEY` the pipeline runs offline: colour-clustered teams, no landmark
seeds. Run it from the app ("Engine") or `scripts/run_fast_pipeline.py` (prints a per-stage
timing table). The oracle sends sampled frames to OpenAI under a per-run USD cap; see
[SECRETS.md](SECRETS.md).
Coordinates use a corner origin: x follows pitch length, y follows pitch width. Frames use
original input indices throughout. The ground-plane approximation is inaccurate for an
airborne ball; a 2D ball marker is not a 3D position estimate.

Runs can select a bounded passage using `start_frame` and `max_frames`. Sampling starts at
that original frame; model observations, camera mappings and annotation indices remain in
the source coordinate system. JSON `time` is relative to the exported passage for playback;
`source_time` and CSV `timestamp_s` retain the source timestamp, with `clip_time_s` recorded
separately. A seek whose reported position is inaccurate fails instead of mislabelling frames.
WASB receives consecutive raw frames within the selected passage, so its boundary samples
may lack the preceding/following context needed for a prediction.

## 1. Calibration

The browser workspace and Streamlit app share `calibrate_clip_auto`. It compares CNN and
labelled-keypoint homographies, rejects weak geometry with spread/plausibility/line-support
checks, and measures RANSAC residuals in image pixels. Rejected samples remain unavailable.
The workspace retries at a denser cadence when coverage is low and can attempt the existing
multi-sport fallback. Coverage measures mapping availability, not positional accuracy.

`CameraReference` compares each sampled image directly with a verified reference. Distributed
optical-flow inliers must support a stationary view, with patch agreement and bounded corner
displacement. It never chains small movements, which would hide cumulative pan or zoom.
Verified stationary views skip field inference. Low texture and uncertain matches fail closed.
The automatic path keeps at most four recent reference images. `CalibrationLookup` retains
explicit rejected samples and uses original frame indices for nearby lookups.

The workspace's four-point editor uses named field markings in source pixels. `manual.py`
rejects duplicate, collinear, off-image and folded correspondences. Four points fit exactly;
only the user's overlay review establishes whether the intended markings were selected.
Saved setups include pitch dimensions, image resolution, the mapping and a reference frame.
They can also be saved from a reviewed valid frame of a completed analysis. Each run snapshots
its selected setup. Saved-view runs require camera agreement on each output sample and bypass
PnLCalib entirely; a changed or uncertain view stays unmapped.

`RegisteredAnchor` is an optional experiment for overlapping pans. ORB matches propose a
transform to an independently calibrated view; distributed inliers, reprojection error,
geometry and current line support must all pass. It retains at most three anchors and never
chains transferred mappings. This is panorama-style registration, not a stitched panorama.
It is off by default and lacks real-venue validation.

The player-height advisory infers a pinhole camera only when focal geometry is identifiable.
It compares upright, measured, on-pitch player boxes against a broad height range. It requires
at least 20 usable estimates, excludes clipped boxes and interpolation, and can request review.
It never upgrades or invalidates a calibration. Cropping, distortion, crouching and false boxes
can affect it; it cannot detect all mirrored or translated mappings.

## 2. Detection and global solves

`MatchProcessor.sweep` receives warmed detector instances. It advances the codec sequentially,
retrieving image pixels only for player samples or scheduled ball triples, then batches
`SoccerDetector`'s local COCO YOLO player detections, and advances ByteTrack on sampled frames.
Roboflow's optional local model supplies referee/goalkeeper votes. WASB uses three consecutive
raw frames and returns ball candidates for the middle frame on the selected cadence.
Skipped frames never become adjacent WASB inputs. Run statistics distinguish original frames
advanced, images retrieved, player samples and ball scans. Grabbing may still decode codec
dependencies; fewer retrieved images does not imply the same reduction in total runtime.

Jersey LAB samples from on-pitch tracks feed a chroma-weighted K=2 fit. Whole-track voting
produces one team per track. Official classification combines model votes and color evidence.
When no calibration exists, the app can still display detections and estimate team colors
without pitch filtering; those team assignments are less reliable. Short track gaps are
interpolated at processed frame indices for at most half a second, with jump and detected-cut
guards. `MatchData.interpolated` records their provenance; observed-position analytics exclude
them. Detected scene cuts reset tracker identities with an offset to avoid ID collisions.

`BallPathSolver` scores detection candidates minus speed/gap penalties in bounded temporal
neighborhoods. Long occlusions and detected cuts split the graph so later sequences survive.
The processor uses source FPS for its 0.8 s association and 0.3 s interpolation limits.
WASB uses overlapping tiles in bounded batches and merges seam duplicates. Body-overlapping
candidates are downweighted, since a real airborne ball can overlap a player. Persistent
static false positives, acceleration consistency and night recall remain evaluation tasks.

## 3. Rendering and resource lifetime

The app maps player feet and ball candidates using valid calibrations, then combines the
broadcast overlay with `draw_topdown`. Fixed colors: team A red, B blue, referees yellow,
goalkeepers magenta, unknown gray. `CoverageMap` aggregates geometric view coverage.
Workspace exports append source-frame and map-availability captions outside the footage,
without presenting a fabricated confidence score. Interpolated boxes use separated corners;
interpolated ball/player map positions use hollow markers. The source pixel coordinates in
JSON remain unchanged by the caption band. Legacy calibration tools retain their diagnostic HUD.

`video.py` handles unknown frame-count metadata, source-FPS/stride timing, capture cleanup,
and checked MP4 encoder fallback. Temporary uploads/results live in a per-run directory;
`ExitStack` releases render resources on failure/cancellation. Streamlit receives result bytes
before temporary files disappear. Output is sampled constant-FPS video without source audio;
variable-frame-rate timestamps are not preserved exactly. MPEG-4 fallback playback depends on
the browser. The download remains available even if embedded playback is unsupported.

The browser worker loads field models first, releases them and unused accelerator cache, then
loads player/ball models. Low memory mode defaults to CPU, two CPU threads, 1280-pixel player
inputs and a single-image batch. Accelerated mode uses a two-image batch and limits MPS/CUDA
allocator usage. A worker monitor checks RSS and system memory every second; it can stop a run
without publishing a partial result. This is a guard, not an OS-level memory reservation.

Completed calibration and tracking stages are stored as atomic JSON checkpoints inside the
run folder. Resuming a stopped run retains its settings and camera snapshot, validates the
source/settings/code/package/weight signature and reuses only completed stages. Tracking also
names the digest of its calibration samples. Corrupt, partial or mismatched checkpoints are
ignored and recomputed. Reused stages report zero current-attempt time and inference calls;
historical inference counts are named separately. Rendering restarts from its first frame.
Filesystem fingerprints detect ordinary source/weight edits, not forged metadata. External
model repository source revisions are not pinned; restart a fresh run after changing them.
Only one worker can run, including resumed jobs, and completed analyses remain immutable.

The clip catalogue accepts uploaded copies or CLI-linked local originals. `--import-video`
stores metadata and a filesystem fingerprint without copying the source. Changed/missing
links become unavailable until reimported. The HTTP API exposes opaque clip IDs and public
metadata; registering arbitrary filesystem paths is CLI-only. Analysis history lists completed,
running and stopped runs, and the stopped screen shows retained settings before resuming.

Heavy detectors are reused within one Streamlit session. This prevents per-clip settings from
leaking across users, at the cost of additional memory for simultaneous sessions. External
model adapters still manipulate process-wide imports; production concurrency requires further
isolation. Processing is synchronous, with progress and previews between stages.

## Supporting and historical paths

| Module/path | Role |
| --- | --- |
| `pitch_model.py` | Named geometry, line/circle samples |
| `field_detector.py` | `FieldDetector` protocol, observations, cached and PnLCalib adapters |
| `mapper.py` | Stateful calibration interface, confidence/expiry, optional optical flow |
| `keypoint_calib.py`, `line_support.py` | Correspondences, RANSAC, geometry and line evidence |
| `ls_calibrator.py`, `multisport.py` | Line-based optimization and multi-sport research |
| `fusion.py` | Coverage; experimental camera fusion excluded from current app |
| `distortion.py`, `pose_graph.py`, `intrinsics.py` | Experimental correction/smoothing/intrinsics |
| `visualize.py`, `validate.py` | Overlay CLI and report utilities; distinct from current app |
| `detection.py` | Current shared detector/color primitives plus historical online trackers |
| `run.py` → `pipeline.py` | Legacy CLI; imports old player/team/data-model helpers |
| `calibrate.py` | Original manual calibration tool, kept for reference |

PnLCalib keypoints start in model-input space (960×540), then convert to source pixels.
PnLCalib and WASB have generic module names that can shadow other packages. Maintain their
adapter isolation. Variable field sizes currently have inconsistent geometry assumptions
between adapters; use 105×68 for the baseline and treat other sizes as experimental.

## Object-oriented design evidence

- **Abstraction:** `FieldDetector` is a protocol describing `detect(frame, index)` independently
  of a concrete inference backend; `FieldObservation` defines its output contract.
- **Polymorphism:** `PitchMapper` accepts either a cached or PnLCalib detector and invokes the
  same `detect` method. Tests substitute a mock through the constructor.
- **Encapsulation:** `PitchMapper` manages `_current`, `_frames_since_detection`, and optional
  `_flow`, exposing mapping/confidence operations rather than requiring caller state edits.
- **Inheritance:** `ball_wasb._attrdict` subclasses `dict`, delegates initialization through
  `super()`, and adds recursive attribute access for the external model configuration.

Composition and constructor injection connect the pipeline. These examples document actual
code; they are not claims that every module has been fully refactored into an ideal OO design.
