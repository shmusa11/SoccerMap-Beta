# Quick start

One page from a prepared computer to an exported result. It uses the built-in synthetic demo,
so it needs no models, footage or API key. Setup itself is in [SETUP.md](SETUP.md).

## 1. Start the workspace

```sh
source .venv/bin/activate
python workspace.py
```

Open http://127.0.0.1:8765 in Chrome or Edge. Everything runs on this computer.

## 2. Check the environment

Open **Environment**. Each model file is listed as *File found* or *Missing file*, and the last
line reports this computer's OS, CPU, RAM, free disk and Python version. The demo works even
when model files are missing; real clips need the two required field models.

## 3. Choose a clip and run

1. Open **Analyze a clip** and press **Use demo**. (For your own footage press **Choose a
   local clip** and pick an MP4, MOV, AVI or M4V file.)
2. On **Set up**, keep the field at 105 x 68 m and press **Run synthetic demo**.
3. Wait for the progress bar. The review opens by itself when the run is complete.

## 4. Review

- Press **Play**, or drag the timeline. The film and the field map always show the same
  source frame; its number is under the timeline.
- The legend gives each class its own shape: Team A circle, Team B square, goalkeeper diamond,
  referee triangle, unknown ring with "?". A dashed outline is a filled-in (interpolated)
  position, not a detection.
- The status line reports coverage as counts, for example *540 of 600 sampled frames mapped
  (90.0%)*. Coverage says how often a map was available. It is not an accuracy score.

## 5. What an unavailable frame looks like

Press **Show mapping gap**. The demo has a deliberate gap from 00:18 to 00:24. Inside it:

- the status line reads *Map unavailable*,
- the field map is covered and no player marker is drawn,
- the film keeps playing, and
- the exported CSV leaves `x_m` and `y_m` blank for those rows.

SoccerMap does this whenever it cannot confirm the field in a frame. It never reuses an older
map or invents a position to fill the gap.

## 6. Export and save

Open **Export**:

| Button | File | Contents |
| --- | --- | --- |
| Download MP4 | `soccermap-<run>-result.mp4` | Annotated film beside the field map (H.264) |
| Download CSV | `soccermap-<run>-positions.csv` | One row per observation, in metres |
| Download run details | `soccermap-run.json` | Settings, coverage, timings and provenance |
| Download run log | `soccermap-<run>-run.log` | Each stage with its elapsed time |

Press **Save analysis**. The run now appears under **Analysis history** with its run ID, and
reopens without being processed again.

## If something stops

| Message | What to do |
| --- | --- |
| Required field models are missing: ... | Open Environment, install the named file, press Check again |
| This file is not a readable video | Export the clip as an H.264 MP4 and import it again |
| An analysis is already running | Wait for it or press Cancel analysis; one run at a time |
| Analysis exceeded its 4 GB memory budget | Use a shorter passage, or untick Low memory mode on a computer with a GPU |
| Interrupted: the server was stopped | Open the run in Analysis history and press Retry as new analysis |
