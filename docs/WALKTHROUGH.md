# Recording the Sprint 4 interface walkthrough

Run the new local workspace from the repository root:

```bash
source .venv/bin/activate
python workspace.py
```

Open **http://127.0.0.1:8765**. This workspace is the product: it runs every automatic
analysis through the fast keyframe pipeline (`fast_pipeline.py`: sparse verified field
calibration, camera-motion-compensated player tracking, ball path tracker, on-field filter),
adapted to the workspace result format by `soccer_analyzer/fast_workspace.py`. When
`OPENAI_API_KEY` is set in `.env`, AI keyframes (field markings, ball checks, on-field votes)
are used with a hard $0.30 cap per run, and the spend is shown under processing details; without
a key the pipeline runs fully offline. Saved four-point camera setups use the classic
three-pass path (`engine="classic"`). The older Streamlit app (`app.py`) is legacy. Each
analysis runs in a separate worker process.
The workspace keeps import, setup and review in one local workflow. A light workspace and
compact navigation put the video, field geometry and controls first. On narrow screens the
navigation moves above the workspace and paired views stack. Synthetic labels and mapping
warnings stay visible during recording.

## Camera setup for real footage

1. Import a short clip and open **Place four points** under **Camera setup**.
2. Load a source frame with several clear field markings. Choose each point's named marking
   from the list, check its number on the reference diagram, then click the matching video point.
3. Drag to adjust, use the pixel inputs, or focus the image and use arrow keys (Shift moves
   ten pixels). Zoom the image for a distant intersection. Avoid three points on the same line.
4. Select **Preview field lines**. Check several lines away from the four points. A perfectly
   fitted four-point set can still name the wrong markings.
5. Confirm the overlay, name the setup and save it. Choose it for later clips with the same
   resolution and view. The run checks the reference on each sampled frame; panning, zooming
   or uncertainty leaves the map unavailable.
6. After an automatic run, pause on a valid mapped frame and choose **Save this camera view**.
   Review the still overlay before saving it for future clips.

Keep **Low memory mode** enabled on a busy laptop. It uses CPU inference and smaller player
inputs, with a tradeoff in speed and distant-player detail. Models load in separate phases.
The worker stops if its RSS exceeds 4 GB or available system memory drops below 1 GB.
Leave **Try panorama registration** off for the baseline; it is an unvalidated experiment.

## A reliable recording path

1. Show **Environment**. Explain that file checks are not a full model health test and that
   optional ball models are separate from the required field models.
2. Return to **Analyze a clip**. Choose **Use demo**. Explicitly identify it as synthetic,
   locally generated data that demonstrates the interface, not computer-vision accuracy.
3. Show the field dimensions, starting source frame, raw-frame limit, sampling stride, optional detection and colors.
   The default demo covers 60 seconds. Real footage defaults to a shorter 300-frame run.
4. Start a run, click **Cancel analysis**, then **Resume analysis** to keep its settings.
   Completed stages are recovered only when inputs match. Use **Review settings & retry** to
   start a separate run with different settings. **Analysis history** retains stopped runs
   across browser and server restarts.
   Cancellation terminates the worker and does not expose a partial export as a complete result.
5. In **Watch** (formerly Synchronized review), play and seek. Toggle map IDs, ball and team filters.
   The video and interactive map use one time control. Video annotations remain fixed;
   layer switches change the interactive map only.
6. Click **Show mapping gap**. The demo's 18–24 second gap hides field positions while the
   synthetic video continues. No stale points are carried across the gap.
7. In **Team Shape**, choose an interval and team. Select defender tracks to calculate their
   mean distance from the defended goal. Only valid frames where all selected defenders are
   observed together contribute. Reverse direction to explain the defended-goal reference.
8. Switch to **Player heatmap**, select a track, and explain that brighter cells mean more
   observed samples. Missing views do not contribute; this is not full-match occupancy.
9. In **Export**, download MP4, CSV and run JSON. Synthetic provenance is retained in all three.
   Save the analysis, open **Analysis history**, and reopen it. Refresh to show persistence.
10. If demonstrating actual inference, import a permitted short daytime clip and run it with
    the configured models. Describe the actual result and coverage rather than using the demo
    as evidence of algorithmic performance. Never include private footage in Git.

## Coach tools in Watch and Match stats

These run in the browser from the completed result, with no model calls or extra server work.

- **Layouts.** Choose Film + map, Film + 3D, Film, Map or 3D above the panes. **Expand** (or F)
  fills the screen. Space plays or pauses; the arrow keys step one sampled frame.
- **Defense coverage.** Shades the space each defending player reaches first (team colour) and
  highlights open space between the ball and the defended goal that attackers reach first
  (amber, labelled with its area on the map). Distances between the deepest visible
  defenders appear when the back line is stretched. By default it follows the team without
  the ball; choose a team when possession is unclear. Areas outside the camera view are
  dimmed and never reported as gaps, because unseen defenders cannot be judged.
- **Draw** (or D). Arrow, pen and zone tools in four colours. A drawing made on a mapped view is
  stored in field metres, so it appears on the map, on the 3D grass and pinned to the field
  in the film as the camera pans. On an unmapped film frame it stays on the film only.
- **3D.** *Free* orbits (double-click re-frames the play), *Top* is a tactical view, *Film
  angle* matches the camera recovered from each frame's field mapping, and *Player view*
  shows the game from a selected track (*Eyes* or *Behind*). Click any player to follow
  them. Players are stylised figures driven by smoothed tracked positions; translucent
  figures are gap-filled. The 3D engine (Three.js, vendored under `web/vendor/`) loads only
  when a 3D layout is opened, uses the low-power GPU and renders only while something moves.
- **Match stats.** Possession, completed passes, pass accuracy, balls lost, tracked distance,
  sprints, top speed and average team width/length; key players (most involved, playmaker,
  most ground covered, fastest); a pass network; clickable moments that also appear as
  ticks under the timeline; and a per-track table with *Watch* buttons that open Player view.
  **Swap attacking directions** corrects the automatic goal-side guess.

Limits: passes and possession come from the nearest player to the ball, so missed ball
detections hide passes and contested balls are skipped. Distances use smoothed positions only
while a track is on camera. Track IDs are not verified people, and fragmented tracks split
one player's numbers. Coverage uses a time-to-reach model with typical speeds, not measured
player ability, and the ball is placed on the ground (no measured height).

## Implemented and still limited

Implemented: local upload and decode validation, configuration validation, fixed per-run
settings, asynchronous progress, process cancellation, review, invalid-calibration handling,
observed-sample average positions and heatmaps, selected-defender height, CSV/JSON/MP4 export,
saved completed analyses and restart recovery. Real inference runs through the existing
three-pass algorithm. Routine tests use generated fixtures and injected detectors.

Still experimental: real calibration/ball accuracy, nonstandard field dimensions,
multi-sport footage, team assignments and full-match performance. A saved setup applies to a fixed view. A tripod camera that pans or zooms still needs a
new mapping; the optional registration experiment has not been validated across venues. A track
ID is not a verified person. Exports are sampled, silent, constant-frame-rate video.
The legacy Streamlit entry point retains its original workflow and session model cache;
workspace workers load models afresh to isolate imports and support cancellation.

## Local files and access

The server listens on loopback only and rejects cross-origin writes and unexpected Host
headers. Do not expose this development server to a network. There is no account system.
Uploaded copies and generated results live in `.soccermap/`, which is excluded from Git.
Browser uploads are limited to 500 MB. For a larger local match, link the original file at
startup without copying it:

```bash
python workspace.py --import-video "/path/to/match.mp4"
```

Repeat `--import-video` for more files. Linked clips persist in the local catalogue; leave
them at their original locations. If a file changes, link it again. The browser accepts only
clip IDs and cannot register arbitrary local paths. Each run still processes a bounded passage
of at most 7,200 original frames; use a later starting frame to examine another passage.
Choose a linked or uploaded file under **Imported clips**. In **Processing detail**, set
**Starting source frame** and **Maximum input frames**; the hint shows the source time and
actual range, trimmed at the end of the video. The point editor starts at that same frame.
Review playback starts at zero for each passage, while source indices and CSV timestamps
retain their location in the original match.
Completed runs remain there even before **Save analysis** is clicked; saving adds a bookmark.
Closing the tab does not erase local files. Stop the server before manually removing the data
directory if you no longer need its footage/results. Keep sufficient free disk space for
the uploaded source plus MP4 exports. Only one analysis can run at once.

## Checks

Use Node.js 18 or newer for the dependency-free browser-logic tests. `make check` includes
them alongside the Python checks; the standalone command below is useful during UI work.

```bash
make check PYTHON=.venv/bin/python
node --test web/*.test.js
```

Browser review should include importing a corrupt clip, cancel/retry, the deliberate demo
gap, all three exports, saved-result reopen, and narrow-screen navigation. Tests do not
establish accuracy on real matches. Inspect real output overlays separately before making
claims about calibration quality.
