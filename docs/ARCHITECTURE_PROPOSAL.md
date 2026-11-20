# Proposal: reviewable mapping by camera segment

Status: design proposal, not implemented or accuracy-validated. Keep the current pipeline
as the comparison baseline. This is a potential next experiment, not a prerequisite for
shipping the class MVP.

## Change the problem we ask the system to solve

The current goal is fully automatic mapping from arbitrary footage. A useful alternative
is: produce a trustworthy map with a small, measured amount of human assistance. When the
field is ambiguous or barely visible, repeatedly adding automatic fallbacks may still leave
the geometry underdetermined. Let a reviewer confirm field landmarks on a few selected
frames, then measure how far those anchors can safely support nearby frames.

I recommend testing this assisted workflow before replacing detectors or adding more model
ensembles. It targets the calibration bottleneck directly and gives the class a demonstrable
product even when fully automatic calibration fails. It costs reviewer time, and it will
not fix missing ball detections or an airborne ball's inaccurate ground-plane projection.

## Proposed flow

```mermaid
flowchart LR
    A[Video and timestamps] --> B[Camera segments]
    B --> C[Cached observations]
    C --> D[Automatic calibration and tracks]
    D --> E[Review uncertain anchors]
    E --> F[Segment solves and validation]
    F --> G[Map and video export]
```

A camera segment is a continuous view between cuts or tracking failures; it can include
panning and zooming. It is not assumed to have one constant homography.

1. **Ingest once.** Record source hash, frame index, presentation timestamp, resolution and
   pitch dimensions. Detect candidate cuts, with a review override for false cut detections.
2. **Store observations independently of decisions.** Persist field landmarks/lines, player
   boxes and ball candidates with model versions and settings. Keep automatic observations
   immutable so correcting a landmark does not require repeating expensive model inference.
3. **Request targeted assistance.** Show representative uncertain frames. A reviewer selects
   named field landmarks and confirms orientation/dimensions; ambiguous geometry remains
   unresolved. A direct planar fit needs at least four well-spread, non-collinear point
   correspondences; prefer extra points and check landmarks excluded from the fit. Fewer
   or poorly distributed visible points are a reason to request another frame, not to guess.
4. **Solve within segments.** Start with independent verified anchors plus short-range motion
   tracking. Re-estimate a mapping for each supported frame and terminate propagation on
   motion/line residual failures. Reset at cuts. Do not smooth homography coefficients
   blindly, assume a fixed camera, or force an entire clip into one camera pose.
5. **Separate association from projection.** Track players and ball candidates in image space
   first, then project only through supported geometry. Break interpolation at segment
   boundaries; cross-segment player identity stays unknown unless independently established.
6. **Render evidence and uncertainty.** Distinguish automatic, human-assisted, propagated and
   unavailable estimates. Save accepted/rejected evidence and residuals, not a confidence
   percentage presented as a calibrated probability. Never fill unsupported positions merely
   to make the map look complete.

This does not propose repeating the previously rejected pooled fixed-camera fusion or the
unconstrained global camera solve described in STATUS.md. Any later temporal optimization
must be compared against independent anchors and support camera motion explicitly.

## Small application architecture

Keep one Python application, with Streamlit as the UI and one isolated worker process per
local job. Use explicit stage functions behind a job runner; no distributed services are
needed for the class project. The same runner should be callable from a CLI and tests.

| Boundary | Contract |
| --- | --- |
| Clip manifest | Source hash, timestamps, segment IDs, pitch geometry, schema version |
| Observation store | Frame-indexed model outputs and provenance; no credentials |
| Review records | Landmark corrections, frame/segment, reviewer and revision; original observations retained |
| Solve outputs | Mapping, validity, residuals, provenance and invalidation reason per supported frame; tracks/candidates separate |
| Job runner | Stage status, cancellation, resumable artifacts and isolated model imports |
| Renderer | Consumes saved solve outputs; changing display settings does not rerun inference |

Start with a local job directory, JSON manifests and array files. Write artifacts atomically;
mark a stage complete only after validation. Cache keys must include source content, model
checkpoint identity, settings and schema/code version. Edits invalidate dependent solves
and renders, not unrelated inference. Store no API key in cache keys, artifacts or logs.
Keep footage and outputs ignored. Storage limits and explicit cleanup belong in the design.
Worker isolation addresses the current process-wide external imports; it does not implement
multi-user credential isolation or a hosted service.

## One bounded experiment before committing to a redesign

Use the same permitted day, night and multi-sport clips and hardware as the baseline.
Reserve independent labelled frames for evaluation; do not assess accuracy only on clicked
landmarks. Measure landmark/pitch error, valid-map coverage, catastrophic wrong placements,
review time, total runtime and peak storage. Report automated and assisted results separately.

Suggested initial budget: at most two minutes of review per one-minute clip, with up to five
anchor frames. Agree on task-appropriate error limits before the experiment. Adopt this path
only if it increases coverage at the agreed error limit without increasing catastrophic
placements, within the review budget. If it fails, record the result and retain the current
pipeline. Coverage alone is not success.

Split the spike across the existing five-person roles:

- Member 1: manifest, stage boundaries and replayable baseline runner.
- Member 2: anchor review schema, geometric validation and assisted calibration prototype.
- Member 3: segment boundaries and player-track interpolation guards.
- Member 4: cached ball observations and stage runtime/storage measurements.
- Member 5: minimal review UI, independent evaluation and fresh-run/replay checks.

Begin with saved observations for one short clip and a basic landmark editor; evaluate the
assistance hypothesis before undertaking the full worker/cache refactor. Keep the existing
app operational throughout. If the experiment succeeds, add persisted observations first,
then the review workflow, then worker isolation and resumability in separate reviewed PRs.
