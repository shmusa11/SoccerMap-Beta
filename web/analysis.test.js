import test from "node:test";
import assert from "node:assert/strict";
import { frameAt, summarize, PlaybackClock } from "./analysis.js";
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
  const r = {run_id:"fixture-run", duration:3, settings:{length:100},frames:[
    {time:0,valid:true,players:[{id:1,team:0,kind:"player",x:10,y:20}]},
    {time:1,valid:true,players:[{id:1,team:0,kind:"player",x:20,y:40}]},
    {time:2,valid:false,players:[{id:1,team:0,kind:"player",x:999,y:999}]}
  ]};
  const options={start:0,end:3,team:0,selected:[1]};
  const first=summarize(r,options);
  assert.deepEqual([first.averages[0].x,first.averages[0].y,first.averages[0].count],[15,30,2]);
  assert.equal(first.height,null);
  assert.equal(first.run_id,"fixture-run");
  assert.deepEqual(first.interval,{start:0,end:3});
  for(let i=0;i<3;i++) assert.deepEqual(summarize(r,options),first);
  r.frames=[{time:0,valid:true,players:[{id:1,team:0,kind:"player",x:20,y:20},{id:2,team:0,kind:"player",x:30,y:40}]}];
  assert.equal(summarize(r,{...options,selected:[1,2],direction:"right"}).height,25);
  assert.equal(summarize(r,{...options,selected:[1,2],direction:"left"}).height,75);
});
