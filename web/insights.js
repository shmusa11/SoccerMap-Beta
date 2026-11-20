/**
 * Coach insights derived from one SoccerMap result: smoothed tracks, possession, passes,
 * player and team numbers, defensive coverage and the film camera. Pure functions with no
 * DOM, so Node tests can run them. Pitch coordinates are corner-origin metres.
 *
 * Everything here is estimated from tracked samples. Players outside the camera view are
 * missing, a track ID is not a verified person, and possession comes from ball proximity.
 * Treat the numbers as coaching prompts, not official match statistics.
 */

export const TEAM_A = 0;
export const TEAM_B = 1;

const SPLIT_GAP = 0.6; // s: a longer absence starts a new track segment
const SMOOTH_SIGMA = 0.22; // s: Gaussian smoothing; mapped positions jitter by metres
const MAX_SPEED = 10.5; // m/s: faster apparent motion is mapping noise, not running
const SPRINT_SPEED = 5.5; // m/s (~20 km/h) high-intensity running for amateur play
const CONTROL_RADIUS = 2.3; // m: ball this close to a player counts as being on the ball
const CONTESTED = 0.7; // m: two players this similar in distance make a sample ambiguous
const PASS_MIN = 5; // m: shorter ball travel between teammates is not counted as a pass
const EVENT_MAX_GAP = 5; // s: longer gaps mean the ball was lost from view
const REACTION = 0.5; // s: reaction time before a player starts moving to a spot
const RUN_SPEED = 5.5; // m/s: typical closing speed used for time-to-reach

const finite = (v) => Number.isFinite(v);
const hypot = Math.hypot;

/** Median of three, used to remove single-sample mapping spikes. */
function median3(a, b, c) {
  return Math.max(Math.min(a, b), Math.min(Math.max(a, b), c));
}

function lowerBound(values, target) {
  let lo = 0,
    hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (values[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

function majority(counts, fallback) {
  let best = fallback,
    most = -1;
  for (const [key, count] of Object.entries(counts))
    if (count > most) [best, most] = [key, count];
  return best;
}

/** Split, de-spike and smooth one track; segments never bridge an unmapped frame. */
function buildSegments(samples, invalidTimes) {
  const segments = [];
  let current = [];
  const flush = () => {
    if (current.length) segments.push(current);
    current = [];
  };
  for (const s of samples) {
    const prev = current[current.length - 1];
    if (prev) {
      const blocked =
        s.t - prev.t > SPLIT_GAP ||
        lowerBound(invalidTimes, prev.t) !== lowerBound(invalidTimes, s.t);
      if (blocked) flush();
    }
    current.push(s);
  }
  flush();
  return segments.map((raw) => {
    const n = raw.length;
    const t = Float64Array.from(raw, (s) => s.t);
    const px = raw.map((s, i) =>
      i > 0 && i < n - 1 ? median3(raw[i - 1].x, s.x, raw[i + 1].x) : s.x,
    );
    const py = raw.map((s, i) =>
      i > 0 && i < n - 1 ? median3(raw[i - 1].y, s.y, raw[i + 1].y) : s.y,
    );
    const x = new Float64Array(n),
      y = new Float64Array(n),
      vx = new Float64Array(n),
      vy = new Float64Array(n);
    const measured = raw.map((s) => s.measured);
    // Gaussian-weighted local linear fit: unbiased at segment ends (a plain moving average
    // pulls endpoints inward and undercounts distance), and its slope is the velocity.
    let lo = 0,
      hi = 0;
    for (let i = 0; i < n; i++) {
      while (t[i] - t[lo] > 3 * SMOOTH_SIGMA) lo++;
      while (hi + 1 < n && t[hi + 1] - t[i] <= 3 * SMOOTH_SIGMA) hi++;
      let s0 = 0,
        s1 = 0,
        s2 = 0,
        sx = 0,
        sy = 0,
        sxt = 0,
        syt = 0;
      for (let j = lo; j <= hi; j++) {
        const d = t[j] - t[i];
        const w =
          Math.exp(-0.5 * (d / SMOOTH_SIGMA) ** 2) * (measured[j] ? 1 : 0.5);
        s0 += w;
        s1 += w * d;
        s2 += w * d * d;
        sx += w * px[j];
        sy += w * py[j];
        sxt += w * px[j] * d;
        syt += w * py[j] * d;
      }
      const det = s0 * s2 - s1 * s1;
      if (det > 1e-9 * s0 * s0) {
        x[i] = (s2 * sx - s1 * sxt) / det;
        y[i] = (s2 * sy - s1 * syt) / det;
        vx[i] = (s0 * sxt - s1 * sx) / det;
        vy[i] = (s0 * syt - s1 * sy) / det;
      } else {
        x[i] = sx / s0;
        y[i] = sy / s0;
      }
      const speed = hypot(vx[i], vy[i]);
      if (speed > MAX_SPEED) {
        vx[i] *= MAX_SPEED / speed;
        vy[i] *= MAX_SPEED / speed;
      }
    }
    return { t, x, y, vx, vy, measured, start: t[0], end: t[n - 1] };
  });
}

/** Interpolated smoothed state of a track at time t, or null outside its segments. */
export function sampleTrack(track, time, tolerance = 0.09) {
  for (const seg of track.segments) {
    if (time < seg.start - tolerance || time > seg.end + tolerance) continue;
    const n = seg.t.length;
    if (n === 1 || time <= seg.start) return state(track, seg, 0, 0, 0);
    if (time >= seg.end) return state(track, seg, n - 1, n - 1, 0);
    const i = Math.max(1, lowerBound(seg.t, time));
    const f = (time - seg.t[i - 1]) / (seg.t[i] - seg.t[i - 1] || 1);
    return state(track, seg, i - 1, i, f);
  }
  return null;
}

function state(track, seg, a, b, f) {
  const lerp = (arr) => arr[a] + (arr[b] - arr[a]) * f;
  const vx = lerp(seg.vx),
    vy = lerp(seg.vy);
  return {
    id: track.id,
    team: track.team,
    kind: track.kind,
    x: lerp(seg.x),
    y: lerp(seg.y),
    vx,
    vy,
    speed: hypot(vx, vy),
    measured: f < 0.5 ? seg.measured[a] : seg.measured[b],
  };
}

/** Collect mapped samples per track from valid frames only. */
function collectTracks(frames, invalidTimes) {
  const byId = new Map();
  for (const frame of frames) {
    if (!frame.valid) continue;
    for (const p of frame.players) {
      if (!finite(p.x) || !finite(p.y)) continue;
      let track = byId.get(p.id);
      if (!track) {
        track = { id: p.id, kinds: {}, teams: {}, samples: [] };
        byId.set(p.id, track);
      }
      track.kinds[p.kind] = (track.kinds[p.kind] || 0) + 1;
      track.teams[p.team] = (track.teams[p.team] || 0) + 1;
      track.samples.push({
        t: frame.time,
        x: p.x,
        y: p.y,
        measured: p.measured !== false,
      });
    }
  }
  return [...byId.values()].map((track) => {
    const segments = buildSegments(track.samples, invalidTimes);
    const kind = majority(track.kinds, "player");
    const rawTeam = Number(majority(track.teams, "-1"));
    let sumX = 0,
      sumY = 0;
    for (const s of track.samples) {
      sumX += s.x;
      sumY += s.y;
    }
    return {
      id: track.id,
      kind,
      rawTeam,
      team: kind === "referee" ? -2 : rawTeam,
      segments,
      samples: track.samples.length,
      seen: segments.reduce((sum, s) => sum + (s.end - s.start), 0),
      meanX: sumX / track.samples.length,
      meanY: sumY / track.samples.length,
    };
  });
}

/**
 * Which goal each team defends. When play is in one half, the defending team usually has
 * the deepest outfield player toward that goal. Returns {0: "left"|"right", 1: ...}.
 */
export function inferSides(frames, length, flip = false) {
  let vote = 0,
    total = 0;
  for (const frame of frames) {
    if (!frame.valid) continue;
    const xs = { 0: [], 1: [] };
    for (const p of frame.players)
      if (p.kind === "player" && (p.team === 0 || p.team === 1) && finite(p.x))
        xs[p.team].push(p.x);
    if (xs[0].length < 2 || xs[1].length < 2) continue;
    const all = [...xs[0], ...xs[1]];
    const play =
      frame.ball && finite(frame.ball[0])
        ? frame.ball[0]
        : all.reduce((a, b) => a + b, 0) / all.length;
    if (play < length / 2) {
      vote += Math.min(...xs[0]) < Math.min(...xs[1]) ? 1 : -1;
    } else {
      vote += Math.max(...xs[0]) > Math.max(...xs[1]) ? -1 : 1;
    }
    total++;
  }
  const zeroLeft = vote >= 0 !== flip;
  return {
    0: zeroLeft ? "left" : "right",
    1: zeroLeft ? "right" : "left",
    confidence: total ? Math.abs(vote) / total : 0,
  };
}

function nearestOnBall(players, bx, by) {
  let best = null,
    second = Infinity;
  for (const p of players) {
    if (p.team !== 0 && p.team !== 1) continue;
    const d = hypot(p.x - bx, p.y - by);
    if (!best || d < best.d) {
      if (best) second = best.d;
      best = { id: p.id, team: p.team, d };
    } else if (d < second) second = d;
  }
  if (!best || best.d > CONTROL_RADIUS || second - best.d < CONTESTED)
    return null;
  return best;
}

/** Spells where one player is clearly closest to the ball for at least two samples. */
function touchSpells(ball, playersAt) {
  const spells = [];
  let current = null;
  const close = () => {
    if (current && current.n >= 2) {
      const last = spells[spells.length - 1];
      if (last && last.id === current.id && current.t0 - last.t1 <= 1.5) {
        last.t1 = current.t1;
        last.end = current.end;
        last.n += current.n;
      } else spells.push(current);
    }
    current = null;
  };
  for (const b of ball) {
    const owner = nearestOnBall(playersAt(b.t), b.x, b.y);
    if (!owner) {
      if (current && b.t - current.t1 > 0.8) close();
      continue;
    }
    if (current && current.id === owner.id && b.t - current.t1 <= 0.8) {
      current.t1 = b.t;
      current.end = [b.x, b.y];
      current.n++;
      continue;
    }
    close();
    current = {
      id: owner.id,
      team: owner.team,
      t0: b.t,
      t1: b.t,
      n: 1,
      start: [b.x, b.y],
      end: [b.x, b.y],
    };
  }
  close();
  return spells;
}

/** Passes between teammates and changes of possession between consecutive spells. */
function ballEvents(spells) {
  const events = [];
  for (let i = 1; i < spells.length; i++) {
    const a = spells[i - 1],
      b = spells[i];
    if (b.t0 - a.t1 > EVENT_MAX_GAP || a.id === b.id) continue;
    const length = hypot(b.start[0] - a.end[0], b.start[1] - a.end[1]);
    const base = {
      t: a.t1,
      tEnd: b.t0,
      from: a.id,
      to: b.id,
      x0: a.end[0],
      y0: a.end[1],
      x1: b.start[0],
      y1: b.start[1],
      length,
    };
    if (a.team === b.team) {
      if (length >= PASS_MIN)
        events.push({ ...base, type: "pass", team: a.team });
    } else
      events.push({
        ...base,
        type: length >= PASS_MIN ? "intercepted" : "tackle",
        team: a.team,
        wonBy: b.team,
      });
  }
  return events;
}

function speedSeries(seg) {
  const n = seg.t.length,
    out = new Float64Array(n);
  let a = 0,
    b = 0;
  for (let i = 0; i < n; i++) {
    while (seg.t[i] - seg.t[a] > 0.25) a++;
    while (b + 1 < n && seg.t[b + 1] - seg.t[i] <= 0.25) b++;
    const dt = seg.t[b] - seg.t[a];
    out[i] =
      dt >= 0.3 ? hypot(seg.x[b] - seg.x[a], seg.y[b] - seg.y[a]) / dt : 0;
  }
  return out;
}

function playerStats(track, events, spells) {
  let distance = 0,
    top = 0,
    sprints = 0;
  for (const seg of track.segments) {
    for (let i = 1; i < seg.t.length; i++) {
      const step = hypot(seg.x[i] - seg.x[i - 1], seg.y[i] - seg.y[i - 1]);
      if (step <= MAX_SPEED * (seg.t[i] - seg.t[i - 1])) distance += step;
    }
    const speeds = speedSeries(seg);
    let runStart = null;
    for (let i = 0; i < speeds.length; i++) {
      const v = speeds[i] <= MAX_SPEED ? speeds[i] : 0;
      top = Math.max(top, v);
      if (v >= SPRINT_SPEED) {
        if (runStart === null) runStart = seg.t[i];
      } else {
        if (runStart !== null && seg.t[i - 1] - runStart >= 0.5) sprints++;
        runStart = null;
      }
    }
    if (runStart !== null && seg.end - runStart >= 0.5) sprints++;
  }
  const touches = spells.filter((s) => s.id === track.id).length;
  const passes = events.filter(
    (e) => e.type === "pass" && e.from === track.id,
  ).length;
  const received = events.filter(
    (e) => e.type === "pass" && e.to === track.id,
  ).length;
  const lost = events.filter(
    (e) => e.type !== "pass" && e.from === track.id,
  ).length;
  const won = events.filter(
    (e) => e.type !== "pass" && e.to === track.id,
  ).length;
  return {
    id: track.id,
    team: track.team,
    kind: track.kind,
    seen: track.seen,
    distance,
    workRate: track.seen > 0 ? (distance / track.seen) * 60 : 0,
    topSpeed: top,
    sprints,
    touches,
    passes,
    received,
    lost,
    won,
    involvement: touches + passes + received + won,
    x: track.meanX,
    y: track.meanY,
  };
}

function shapeStats(frames, team) {
  let n = 0,
    width = 0,
    depth = 0,
    spread = 0;
  for (const frame of frames) {
    if (!frame.valid) continue;
    const ps = frame.players.filter(
      (p) =>
        p.team === team && p.kind === "player" && finite(p.x) && finite(p.y),
    );
    if (ps.length < 6) continue;
    const xs = ps.map((p) => p.x),
      ys = ps.map((p) => p.y);
    const cx = xs.reduce((a, b) => a + b, 0) / ps.length,
      cy = ys.reduce((a, b) => a + b, 0) / ps.length;
    width += Math.max(...ys) - Math.min(...ys);
    depth += Math.max(...xs) - Math.min(...xs);
    spread += ps.reduce((s, p) => s + hypot(p.x - cx, p.y - cy), 0) / ps.length;
    n++;
  }
  return n
    ? { frames: n, width: width / n, depth: depth / n, spread: spread / n }
    : { frames: 0, width: null, depth: null, spread: null };
}

/**
 * Build the insight model for a result. `flip` swaps which goal each team defends when the
 * automatic guess is wrong. Returns stats plus `playersAt(t)` for animation and overlays.
 */
export function buildModel(result, { flip = false } = {}) {
  const L = result.settings.length,
    W = result.settings.width;
  const frames = result.frames;
  const invalidTimes = frames.filter((f) => !f.valid).map((f) => f.time);
  const tracks = collectTracks(frames, invalidTimes);
  const defends = inferSides(frames, L, flip);
  // A goalkeeper belongs to the team defending the goal on its side of halfway.
  for (const track of tracks)
    if (track.kind === "goalkeeper")
      track.team =
        (track.meanX < L / 2 ? "left" : "right") === defends[0] ? 0 : 1;
  const byId = new Map(tracks.map((t) => [t.id, t]));

  const playersAt = (time) => {
    const out = [];
    for (const track of tracks) {
      const s = sampleTrack(track, time);
      if (s) out.push(s);
    }
    return out;
  };

  const ball = [];
  for (const f of frames)
    if (
      f.valid &&
      Array.isArray(f.ball) &&
      finite(f.ball[0]) &&
      finite(f.ball[1])
    )
      ball.push({
        t: f.time,
        x: f.ball[0],
        y: f.ball[1],
        measured: f.ball_measured !== false,
      });
  const spells = touchSpells(ball, playersAt);
  const events = ballEvents(spells);

  // Team in possession: the latest spell's team, held while the ball is in transit to a
  // teammate or for a short time afterwards. Unknown stretches count for nobody.
  const ownerAt = (time) => {
    const i = lowerBound(
      spells.map((s) => s.t0),
      time + 1e-9,
    );
    const last = spells[i - 1];
    if (!last) return null;
    if (time <= last.t1 + 2) return last.team;
    const next = spells[i];
    if (next && next.team === last.team && next.t0 - last.t1 <= EVENT_MAX_GAP)
      return last.team;
    return null;
  };
  const owned = { 0: 0, 1: 0 };
  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    const dt = (frames[i + 1]?.time ?? f.time) - f.time;
    if (!f.valid || dt <= 0) continue;
    const team = ownerAt(f.time);
    if (team === 0 || team === 1) owned[team] += dt;
  }
  const ownedTotal = owned[0] + owned[1];

  const players = tracks
    .filter((t) => t.kind !== "referee" && (t.team === 0 || t.team === 1))
    .map((t) => playerStats(t, events, spells));
  const duration = result.duration || frames[frames.length - 1]?.time || 0;
  const teams = [0, 1].map((team) => {
    const mine = players.filter((p) => p.team === team);
    const passes = events.filter(
      (e) => e.type === "pass" && e.team === team,
    ).length;
    const failed = events.filter(
      (e) => e.type === "intercepted" && e.team === team,
    ).length;
    const lost = events.filter(
      (e) => e.type !== "pass" && e.team === team,
    ).length;
    return {
      team,
      defends: defends[team],
      possession: ownedTotal ? owned[team] / ownedTotal : null,
      passes,
      attempts: passes + failed,
      accuracy: passes + failed ? passes / (passes + failed) : null,
      lost,
      distance: mine.reduce((s, p) => s + p.distance, 0),
      sprints: mine.reduce((s, p) => s + p.sprints, 0),
      topSpeed: Math.max(0, ...mine.map((p) => p.topSpeed)),
      shape: shapeStats(frames, team),
    };
  });

  const regular = players.filter(
    (p) => p.kind === "player" && p.seen >= Math.max(3, duration * 0.2),
  );
  const pick = (key, min = 0) => {
    const best = regular.reduce(
      (a, p) => (!a || p[key] > a[key] ? p : a),
      null,
    );
    return best && best[key] > min ? best : null;
  };
  const keyPlayers = [
    { role: "Most involved", stat: "involvement", player: pick("involvement") },
    { role: "Playmaker", stat: "passes", player: pick("passes") },
    { role: "Most ground covered", stat: "workRate", player: pick("workRate") },
    { role: "Fastest", stat: "topSpeed", player: pick("topSpeed", 1) },
  ].filter((k) => k.player);

  const mapped = frames.filter((f) => f.valid).length;
  return {
    length: L,
    width: W,
    duration,
    tracks,
    trackById: byId,
    defends,
    ball,
    spells,
    events,
    players,
    teams,
    keyPlayers,
    ballCoverage: mapped ? ball.length / mapped : 0,
    playersAt,
    ownerAt,
  };
}

/** Ball position at time t, interpolated over short gaps only. */
export function ballAt(model, time) {
  const b = model.ball;
  if (!b.length) return null;
  const i = lowerBound(
    b.map((s) => s.t),
    time,
  );
  const next = b[i],
    prev = b[i - 1];
  if (prev && next && next.t - prev.t <= 0.5) {
    const f = (time - prev.t) / (next.t - prev.t || 1);
    return {
      x: prev.x + (next.x - prev.x) * f,
      y: prev.y + (next.y - prev.y) * f,
      measured: prev.measured && next.measured,
    };
  }
  const near = [prev, next]
    .filter(Boolean)
    .find((s) => Math.abs(s.t - time) <= 0.1);
  return near ? { x: near.x, y: near.y, measured: near.measured } : null;
}

/** Pass network for one team: average positions as nodes and pass counts as links. */
export function passNetwork(model, team) {
  const nodes = model.players
    .filter((p) => p.team === team && (p.touches || p.passes || p.received))
    .map((p) => ({ id: p.id, x: p.x, y: p.y, weight: p.involvement }));
  const links = new Map();
  for (const e of model.events) {
    if (e.type !== "pass" || e.team !== team) continue;
    const key = e.from < e.to ? `${e.from}-${e.to}` : `${e.to}-${e.from}`;
    const link = links.get(key) || {
      a: Math.min(e.from, e.to),
      b: Math.max(e.from, e.to),
      count: 0,
    };
    link.count++;
    links.set(key, link);
  }
  return { nodes, links: [...links.values()] };
}

/* ---------- Linear algebra for the film camera ---------- */

export function invert3(m) {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h,
    B = -(d * i - f * g),
    C = d * h - e * g;
  const det = a * A + b * B + c * C;
  if (!det || !finite(det)) return null;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const scale3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const norm3 = (a) => Math.sqrt(dot3(a, a));

/**
 * Recover a pinhole camera from an image-to-pitch homography, assuming square pixels and a
 * centred principal point (the same assumptions as the Python height check). Returns the
 * camera in viewer coordinates (X along length, Y up, Z along width, origin at the centre
 * spot): position, a camera-to-world rotation (columns are right, up, back) and the vertical
 * field of view in degrees. Returns null when the focal length is ambiguous.
 */
export function filmCamera(homography, imageWidth, imageHeight, length, width) {
  const G = invert3(homography);
  if (!G) return null;
  const cx = imageWidth / 2,
    cy = imageHeight / 2;
  const col = (k) => [G[0][k] - cx * G[2][k], G[1][k] - cy * G[2][k], G[2][k]];
  const a = col(0),
    b = col(1),
    c = col(2);
  // K^-1 columns must be orthogonal and equal length: solve for 1/f^2 in least squares.
  const k1 = a[0] * b[0] + a[1] * b[1],
    k2 = a[0] ** 2 + a[1] ** 2 - b[0] ** 2 - b[1] ** 2;
  const r1 = -a[2] * b[2],
    r2 = b[2] ** 2 - a[2] ** 2;
  const den = k1 * k1 + k2 * k2;
  if (den < 1e-16) return null;
  const invF2 = (k1 * r1 + k2 * r2) / den;
  if (!(invF2 > 0) || !finite(invF2)) return null;
  const focal = 1 / Math.sqrt(invF2);
  if (focal < imageWidth * 0.3 || focal > imageWidth * 5) return null;
  const unproject = (v) => [v[0] / focal, v[1] / focal, v[2]];
  let p1 = unproject(a),
    p2 = unproject(b),
    t = unproject(c);
  const s = (norm3(p1) + norm3(p2)) / 2;
  if (!(s > 1e-12)) return null;
  p1 = scale3(p1, 1 / s);
  p2 = scale3(p2, 1 / s);
  t = scale3(t, 1 / s);
  if (Math.abs(dot3(p1, p2)) > 0.12 || Math.abs(norm3(p1) - norm3(p2)) > 0.12)
    return null;
  // The ground under the image centre must lie in front of the camera; the homography's
  // overall sign is arbitrary.
  const H = homography;
  const hw = H[2][0] * cx + H[2][1] * cy + H[2][2];
  const qx = (H[0][0] * cx + H[0][1] * cy + H[0][2]) / hw,
    qy = (H[1][0] * cx + H[1][1] * cy + H[1][2]) / hw;
  if (p1[2] * qx + p2[2] * qy + t[2] < 0) {
    p1 = scale3(p1, -1);
    p2 = scale3(p2, -1);
    t = scale3(t, -1);
  }
  // Orthonormalise the plane axes; "up" points from the field toward the camera.
  const ex = scale3(p1, 1 / norm3(p1));
  const proj = dot3(p2, ex);
  let ez = [p2[0] - proj * ex[0], p2[1] - proj * ex[1], p2[2] - proj * ex[2]];
  ez = scale3(ez, 1 / norm3(ez));
  const up = cross3(ez, ex);
  // Camera position in pitch axes: C = -R^T t with R = [ex ez up] (columns, camera frame).
  const pitchC = [-dot3(ex, t), -dot3(ez, t), -dot3(up, t)];
  if (pitchC[2] < 0) return null; // mirrored mapping: the camera would sit below the field
  // Rows of R_cv (camera <- viewer) for viewer axes X=ex, Y=up, Z=ez.
  // Viewer camera axes in world: right = row0 of R_cv, up = -row1, back = -row2.
  const R = [
    [ex[0], up[0], ez[0]],
    [ex[1], up[1], ez[1]],
    [ex[2], up[2], ez[2]],
  ];
  const right = R[0],
    camUp = scale3(R[1], -1),
    back = scale3(R[2], -1);
  return {
    position: [pitchC[0] - length / 2, pitchC[2], pitchC[1] - width / 2],
    // Column-major basis vectors: camera right, up and back in viewer coordinates.
    basis: [right, camUp, back],
    fov: (2 * Math.atan(imageHeight / 2 / focal) * 180) / Math.PI,
    focal,
    aspect: imageWidth / imageHeight,
  };
}

/** Which pitch cells the camera saw in this frame (cell centres projected into the image). */
export function visibleMask(homography, imageWidth, imageHeight, nx, ny, cell) {
  const mask = new Uint8Array(nx * ny);
  const G = homography && invert3(homography);
  if (!G) return mask;
  const H = homography;
  const w = (H[2][0] * imageWidth) / 2 + (H[2][1] * imageHeight) / 2 + H[2][2];
  const centre = [
    ((H[0][0] * imageWidth) / 2 + (H[0][1] * imageHeight) / 2 + H[0][2]) / w,
    ((H[1][0] * imageWidth) / 2 + (H[1][1] * imageHeight) / 2 + H[1][2]) / w,
  ];
  const sign =
    Math.sign(G[2][0] * centre[0] + G[2][1] * centre[1] + G[2][2]) || 1;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) * cell,
        y = (j + 0.5) * cell;
      const z = G[2][0] * x + G[2][1] * y + G[2][2];
      if (z * sign <= 0) continue;
      const u = (G[0][0] * x + G[0][1] * y + G[0][2]) / z,
        v = (G[1][0] * x + G[1][1] * y + G[1][2]) / z;
      if (u >= 0 && u <= imageWidth && v >= 0 && v <= imageHeight)
        mask[j * nx + i] = 1;
    }
  return mask;
}

function reachTime(p, x, y) {
  const qx = p.x + p.vx * REACTION,
    qy = p.y + p.vy * REACTION;
  return REACTION + hypot(x - qx, y - qy) / RUN_SPEED;
}

/**
 * Defensive coverage at one moment. For each cell, `control` is the defending team's chance
 * to reach it first (time-to-reach logistic, after Spearman's pitch-control model). Gaps are
 * visible cells between the ball and the defended goal that the attackers would reach first.
 * `visible` limits gaps to what the camera saw: unseen defenders cannot be judged.
 */
export function coverageAt(
  model,
  players,
  ball,
  { defending, visible, cell = 1.5 },
) {
  const L = model.length,
    W = model.width;
  const nx = Math.ceil(L / cell),
    ny = Math.ceil(W / cell);
  const control = new Float32Array(nx * ny);
  const gap = new Uint8Array(nx * ny);
  const zone = new Uint8Array(nx * ny);
  const def = players.filter((p) => p.team === defending);
  const att = players.filter((p) => p.team === 1 - defending);
  const empty = {
    nx,
    ny,
    cell,
    control,
    gap,
    zone,
    gaps: [],
    backLine: [],
    defending,
  };
  if (!def.length) return empty;
  const ownGoalLeft = model.defends[defending] === "left";
  const goalX = ownGoalLeft ? 0 : L;
  // Danger zone: from the ball (or the attackers' most advanced player) back to goal.
  let lineX = ball ? ball.x : null;
  if (lineX === null && att.length)
    lineX = ownGoalLeft
      ? Math.min(...att.map((p) => p.x))
      : Math.max(...att.map((p) => p.x));
  if (lineX === null) lineX = ownGoalLeft ? L / 2 : L / 2;
  for (let j = 0; j < ny; j++) {
    const y = (j + 0.5) * cell;
    for (let i = 0; i < nx; i++) {
      const x = (i + 0.5) * cell;
      let tDef = Infinity,
        tAtt = Infinity;
      for (const p of def) tDef = Math.min(tDef, reachTime(p, x, y));
      for (const p of att) tAtt = Math.min(tAtt, reachTime(p, x, y));
      const k = j * nx + i;
      // P(defence first) = logistic of the arrival-time difference (sigma 0.45 s).
      control[k] = att.length
        ? 1 / (1 + Math.exp((-1.8138 * (tAtt - tDef)) / 0.45))
        : Math.max(0, Math.min(1, (2.2 - tDef) / 1.4));
      const behindBall = ownGoalLeft ? x <= lineX + 3 : x >= lineX - 3;
      const nearGoal = Math.abs(x - goalX) <= 45;
      const seen = !visible || visible[k];
      if (seen && behindBall && nearGoal) {
        zone[k] = 1;
        if (control[k] < 0.35 && tDef > 1.0) gap[k] = 1;
      }
    }
  }
  const gaps = components(gap, nx, ny, cell).filter((g) => g.area >= 25);
  // Deepest visible outfield defenders, ordered across the pitch, with the distances
  // between neighbours. Partial views can hide a defender, so this is a prompt to check.
  const line = def
    .filter((p) => p.kind === "player")
    .sort((a, b) => Math.abs(a.x - goalX) - Math.abs(b.x - goalX))
    .slice(0, 4)
    .filter((p) => Math.abs(p.x - goalX) <= 40)
    .sort((a, b) => a.y - b.y);
  const backLine =
    line.length >= 3
      ? line.slice(1).map((p, i) => {
          const q = line[i];
          const distance = hypot(p.x - q.x, p.y - q.y);
          return {
            a: q,
            b: p,
            distance,
            // A flat back four across a full-width pitch sits ~13-15 m apart.
            level: distance > 20 ? "wide" : distance > 16 ? "stretched" : "ok",
          };
        })
      : [];
  return { nx, ny, cell, control, gap, zone, gaps, backLine, defending };
}

function components(mask, nx, ny, cell) {
  const seen = new Uint8Array(mask.length),
    out = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || seen[start]) continue;
    const stack = [start];
    seen[start] = 1;
    let count = 0,
      sx = 0,
      sy = 0;
    while (stack.length) {
      const k = stack.pop(),
        i = k % nx,
        j = (k - i) / nx;
      count++;
      sx += i;
      sy += j;
      for (const [di, dj] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]) {
        const a = i + di,
          b = j + dj;
        if (a < 0 || b < 0 || a >= nx || b >= ny) continue;
        const n = b * nx + a;
        if (mask[n] && !seen[n]) {
          seen[n] = 1;
          stack.push(n);
        }
      }
    }
    out.push({
      area: count * cell * cell,
      x: (sx / count + 0.5) * cell,
      y: (sy / count + 0.5) * cell,
    });
  }
  return out.sort((a, b) => b.area - a.area);
}

/**
 * Defending team at a moment: chosen explicitly, or the team without the ball. When nobody
 * clearly has the ball, the team whose goal is in the ball's half is treated as defending.
 */
export function defendingTeam(model, time, choice = "auto", ball = null) {
  if (choice === "0" || choice === 0) return 0;
  if (choice === "1" || choice === 1) return 1;
  const owner = model.ownerAt(time);
  if (owner === 0 || owner === 1) return 1 - owner;
  if (!ball || !model.defends) return null;
  const side = ball.x < model.length / 2 ? "left" : "right";
  return model.defends[0] === side ? 0 : 1;
}
