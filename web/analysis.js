/** Observed-sample analytics. Coordinates are corner-origin metres. */
export class PlaybackClock {
  start(time, timestamp, end) {
    this.time = time;
    this.timestamp = timestamp;
    this.end = end;
  }
  at(timestamp) {
    return Math.min(
      this.end,
      this.time + Math.max(0, timestamp - this.timestamp) / 1000,
    );
  }
}

// Sampled times are rounded to microseconds, so a time on a frame boundary still
// belongs to that frame rather than the one before it.
const TIME_TOLERANCE_S = 1e-4;

/** The sampled frame shown at a passage time (the last frame starting at or before it). */
export function frameAt(frames, time) {
  if (!frames.length) return null;
  let lo = 0,
    hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].time <= time + TIME_TOLERANCE_S) lo = mid;
    else hi = mid - 1;
  }
  return frames[lo];
}

/** Calibration coverage as counts and a percentage; null when no frames were rendered. */
export function coverageSummary(frames) {
  if (!frames?.length) return null;
  const valid = frames.filter((f) => f.valid).length,
    percent = (100 * valid) / frames.length;
  return {
    valid,
    total: frames.length,
    percent,
    text: `${valid} of ${frames.length} sampled frames mapped (${percent.toFixed(1)}%)`,
  };
}

/** The five displayed classes. Each has a text tag and a marker shape besides its colour. */
export const CLASSES = {
  teamA: { label: "Team A", tag: "A", shape: "circle" },
  teamB: { label: "Team B", tag: "B", shape: "square" },
  goalkeeper: { label: "Goalkeeper", tag: "GK", shape: "diamond" },
  referee: { label: "Referee", tag: "REF", shape: "triangle" },
  unknown: { label: "Unknown", tag: "?", shape: "ring" },
};

/** Class of one observation; an unresolved team or role stays "unknown". */
export function entityClass(player) {
  if (player.kind === "referee" || player.team === -2) return "referee";
  if (player.kind === "goalkeeper" || player.team === -3) return "goalkeeper";
  if (player.team === 0) return "teamA";
  if (player.team === 1) return "teamB";
  return "unknown";
}

export const HEATMAP_BIN_M = 5;

/** Count finite samples per square bin: Map of "column,row" -> count (corner origin). */
export function heatmapBins(samples, size = HEATMAP_BIN_M) {
  const bins = new Map();
  for (const p of samples) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    const key = `${Math.floor(p.x / size)},${Math.floor(p.y / size)}`;
    bins.set(key, (bins.get(key) || 0) + 1);
  }
  return bins;
}

export function summarize(result, { start, end, team, selected, direction }) {
  if (
    !Number.isFinite(start) ||
    !Number.isFinite(end) ||
    start < 0 ||
    end <= start ||
    end > result.duration + 0.01
  ) {
    return {
      error: "Choose a window with an end after its start, within this clip.",
    };
  }
  const frames = result.frames.filter((f) => f.time >= start && f.time < end);
  const valid = frames.filter((f) => f.valid),
    tracks = new Map(),
    samples = [];
  for (const frame of valid)
    for (const player of frame.players) {
      if (
        player.team !== team ||
        player.kind !== "player" ||
        player.measured === false ||
        !Number.isFinite(player.x) ||
        !Number.isFinite(player.y)
      )
        continue;
      const track = tracks.get(player.id) || {
        id: player.id,
        x: 0,
        y: 0,
        count: 0,
        team,
      };
      track.x += player.x;
      track.y += player.y;
      track.count++;
      tracks.set(player.id, track);
      if (selected.includes(player.id)) samples.push(player);
    }
  const averages = [...tracks.values()].map((t) => ({
    ...t,
    x: t.x / t.count,
    y: t.y / t.count,
  }));
  // Back-line height uses the chosen players present together in each valid frame.
  const heights = valid
    .map((f) =>
      f.players.filter(
        (p) =>
          p.team === team &&
          p.kind === "player" &&
          p.measured !== false &&
          selected.includes(p.id) &&
          Number.isFinite(p.x) &&
          Number.isFinite(p.y),
      ),
    )
    .filter((ps) => selected.length > 0 && ps.length === selected.length)
    .map(
      (ps) =>
        ps.reduce(
          (sum, p) =>
            sum + (direction === "right" ? p.x : result.settings.length - p.x),
          0,
        ) / ps.length,
    );
  return {
    frames: frames.length,
    valid: valid.length,
    coverage: frames.length ? valid.length / frames.length : 0,
    averages,
    samples,
    run_id: result.run_id || null,
    interval: { start, end },
    validity_policy:
      "valid calibration and finite measured player observations only",
    height:
      heights.length && ["left", "right"].includes(direction)
        ? heights.reduce((a, b) => a + b, 0) / heights.length
        : null,
    heightFrames: heights.length,
  };
}
