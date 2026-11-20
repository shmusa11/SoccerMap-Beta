import test from "node:test";
import assert from "node:assert/strict";
import {
  CLASSES,
  HEATMAP_BIN_M,
  coverageSummary,
  entityClass,
  frameAt,
  heatmapBins,
  summarize,
  PlaybackClock,
} from "./analysis.js";
const result = {
  duration: 4,
  settings: { length: 100 },
  frames: [
    {
      time: 0,
      valid: true,
      players: [
        { id: 1, team: 0, kind: "player", x: 10, y: 20 },
        { id: 2, team: 0, kind: "player", x: 30, y: 40 },
      ],
    },
    {
      time: 1,
      valid: false,
      players: [{ id: 1, team: 0, kind: "player", x: 999, y: 999 }],
    },
    {
      time: 2,
      valid: true,
      players: [
        { id: 1, team: 0, kind: "player", x: 20, y: 30 },
        { id: 2, team: 0, kind: "player", x: null, y: null },
      ],
    },
    { time: 3, valid: true, players: [] },
  ],
};
test("fallback playback advances independently of range input step rounding", () => {
  const clock = new PlaybackClock();
  clock.start(0, 1000, 59.9);
  let rangeValue = 0;
  for (let frame = 1; frame <= 60; frame++) {
    rangeValue = Math.round(clock.at(1000 + (frame * 1000) / 60) / 0.1) * 0.1;
  }
  assert.equal(rangeValue, 1);
  clock.start(18, 3000, 59.9);
  assert.equal(clock.at(3500), 18.5);
  assert.equal(clock.at(100000), 59.9);
});
test("timeline never borrows a future calibrated frame across a gap", () => {
  assert.equal(frameAt(result.frames, 1.9).valid, false);
  assert.equal(frameAt(result.frames, 2).time, 2);
});
test("analytics exclude invalid and missing positions and require selected defenders together", () => {
  const options = {
    start: 0,
    end: 4,
    team: 0,
    selected: [1, 2],
    direction: "right",
  };
  const s = summarize(result, options);
  assert.equal(s.coverage, 0.75);
  assert.equal(s.averages[0].x, 15);
  assert.equal(s.height, 20);
  assert.equal(s.heightFrames, 1);
  assert.equal(s.samples.length, 3);
  assert.equal(summarize(result, { ...options, direction: "left" }).height, 80);
  assert.equal(summarize(result, { ...options, selected: [] }).height, null);
});
test("empty and invalid windows remain unavailable", () => {
  assert.ok(summarize(result, { start: 2, end: 1 }).error);
  const s = summarize(result, {
    start: 1,
    end: 2,
    team: 0,
    selected: [1],
    direction: "right",
  });
  assert.equal(s.height, null);
  assert.equal(s.averages.length, 0);
  assert.equal(s.coverage, 0);
});

test("interpolated players never contribute to observed-position analytics", () => {
  const result = {
    duration: 1,
    settings: { length: 105 },
    frames: [
      {
        time: 0,
        valid: true,
        players: [
          { id: 1, team: 0, kind: "player", x: 10, y: 10, measured: true },
          { id: 2, team: 0, kind: "player", x: 80, y: 10, measured: false },
        ],
      },
    ],
  };
  const summary = summarize(result, {
    start: 0,
    end: 1,
    team: 0,
    selected: [1, 2],
    direction: "right",
  });
  assert.equal(summary.averages.length, 1);
  assert.equal(summary.samples.length, 1);
  assert.equal(summary.height, null);
});

test("exact acceptance averages direction and summary provenance", () => {
  const r = {
    run_id: "fixture-run",
    duration: 3,
    settings: { length: 100 },
    frames: [
      {
        time: 0,
        valid: true,
        players: [{ id: 1, team: 0, kind: "player", x: 10, y: 20 }],
      },
      {
        time: 1,
        valid: true,
        players: [{ id: 1, team: 0, kind: "player", x: 20, y: 40 }],
      },
      {
        time: 2,
        valid: false,
        players: [{ id: 1, team: 0, kind: "player", x: 999, y: 999 }],
      },
    ],
  };
  const options = { start: 0, end: 3, team: 0, selected: [1] };
  const first = summarize(r, options);
  assert.deepEqual(
    [first.averages[0].x, first.averages[0].y, first.averages[0].count],
    [15, 30, 2],
  );
  assert.equal(first.height, null);
  assert.equal(first.run_id, "fixture-run");
  assert.deepEqual(first.interval, { start: 0, end: 3 });
  for (let i = 0; i < 3; i++) assert.deepEqual(summarize(r, options), first);
  r.frames = [
    {
      time: 0,
      valid: true,
      players: [
        { id: 1, team: 0, kind: "player", x: 20, y: 20 },
        { id: 2, team: 0, kind: "player", x: 30, y: 40 },
      ],
    },
  ];
  assert.equal(
    summarize(r, { ...options, selected: [1, 2], direction: "right" }).height,
    25,
  );
  assert.equal(
    summarize(r, { ...options, selected: [1, 2], direction: "left" }).height,
    75,
  );
});

test("coverage reports both counts and a one-decimal percentage", () => {
  const frames = Array.from({ length: 100 }, (_, i) => ({ valid: i < 75 }));
  const coverage = coverageSummary(frames);
  assert.deepEqual(
    [coverage.valid, coverage.total, coverage.percent],
    [75, 100, 75],
  );
  assert.equal(coverage.text, "75 of 100 sampled frames mapped (75.0%)");
  assert.equal(
    coverageSummary([{ valid: false }]).text.startsWith("0 of 1"),
    true,
  );
  // No rendered frames means no result, never a 0/0 division.
  assert.equal(coverageSummary([]), null);
  assert.equal(coverageSummary(undefined), null);
});

test("five classes keep a tag and shape, and unresolved observations stay unknown", () => {
  const samples = [
    [{ team: 0, kind: "player" }, "teamA"],
    [{ team: 1, kind: "player" }, "teamB"],
    [{ team: -2, kind: "referee" }, "referee"],
    [{ team: -3, kind: "goalkeeper" }, "goalkeeper"],
    [{ team: 1, kind: "goalkeeper" }, "goalkeeper"],
    [{ team: -1, kind: "player" }, "unknown"],
    [{ kind: "player" }, "unknown"],
  ];
  for (const [player, expected] of samples)
    assert.equal(entityClass(player), expected);
  assert.equal(Object.keys(CLASSES).length, 5);
  assert.equal(new Set(Object.values(CLASSES).map((c) => c.shape)).size, 5);
  assert.equal(new Set(Object.values(CLASSES).map((c) => c.tag)).size, 5);
});

test("heatmap bins are 5 m squares built from finite samples only", () => {
  const bins = heatmapBins([
    { x: 1, y: 1 },
    { x: 4.9, y: 4.9 },
    { x: 5, y: 1 },
    { x: 52, y: 33 },
    { x: null, y: 3 },
    { x: NaN, y: NaN },
  ]);
  assert.equal(HEATMAP_BIN_M, 5);
  assert.deepEqual([...bins].sort(), [
    ["0,0", 2],
    ["1,0", 1],
    ["10,6", 1],
  ]);
  assert.equal(heatmapBins([]).size, 0);
});

test("a time on a rounded frame boundary selects that frame, not the one before", () => {
  // 31.59 fps sampled every 2 frames: times are stored rounded to microseconds.
  const step = 2 / 31.59080949101609;
  const frames = Array.from({ length: 150 }, (_, k) => ({
    index: 2 * k,
    time: Math.round(k * step * 1e6) / 1e6,
  }));
  for (const k of [0, 1, 74, 75, 149]) {
    assert.equal(frameAt(frames, k * step).index, 2 * k);
    assert.equal(frameAt(frames, k * step + step / 2).index, 2 * k);
  }
  assert.equal(frameAt(frames, 75 * step - step / 2).index, 148);
});
