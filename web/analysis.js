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

export function frameAt(frames, time) {
  if (!frames.length) return null;
  let lo = 0,
    hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].time <= time) lo = mid;
    else hi = mid - 1;
  }
  return frames[lo];
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
    height: heights.length
      ? heights.reduce((a, b) => a + b, 0) / heights.length
      : null,
    heightFrames: heights.length,
  };
}
