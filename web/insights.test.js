import assert from "node:assert/strict";
import test from "node:test";
import {
  ballAt,
  buildModel,
  coverageAt,
  defendingTeam,
  filmCamera,
  inferSides,
  invert3,
  passNetwork,
  visibleMask,
} from "./insights.js";

const settings = { length: 105, width: 68 };

/** Frames at 10 Hz; `layout(t)` returns players, `ballPath(t)` the ball or null. */
function result(seconds, layout, ballPath, valid = () => true) {
  const frames = [];
  for (let i = 0; i <= seconds * 10; i++) {
    const time = i / 10;
    frames.push({
      index: i * 3,
      time,
      valid: valid(time),
      players: layout(time),
      ball: ballPath(time),
      ball_measured: true,
    });
  }
  return { settings, duration: seconds, frames, source: { fps: 30 } };
}

const player = (id, team, x, y, kind = "player") => ({
  id,
  team,
  kind,
  x,
  y,
  measured: true,
});

test("a ball moving between teammates is one completed pass", () => {
  const layout = () => [
    player(1, 0, 40, 30),
    player(2, 0, 55, 30),
    player(3, 1, 70, 12),
    player(4, 1, 75, 55),
  ];
  const ballPath = (t) =>
    t <= 1
      ? [40.5, 30]
      : t < 1.6
        ? [40.5 + ((t - 1) / 0.6) * 14, 30]
        : [54.5, 30];
  const model = buildModel(result(3, layout, ballPath));
  const passes = model.events.filter((e) => e.type === "pass");
  assert.equal(passes.length, 1);
  assert.equal(passes[0].from, 1);
  assert.equal(passes[0].to, 2);
  assert.ok(passes[0].length > 12 && passes[0].length < 16);
  assert.equal(model.teams[0].passes, 1);
  assert.equal(model.teams[0].possession, 1);
  assert.equal(model.players.find((p) => p.id === 1).passes, 1);
  assert.equal(model.players.find((p) => p.id === 2).received, 1);
  const network = passNetwork(model, 0);
  assert.deepEqual(network.links, [{ a: 1, b: 2, count: 1 }]);
});

test("losing the ball to an opponent is a turnover, not a pass", () => {
  const layout = () => [
    player(1, 0, 40, 30),
    player(5, 0, 30, 10),
    player(3, 1, 52, 30),
    player(4, 1, 75, 55),
  ];
  const ballPath = (t) =>
    t <= 1 ? [40.5, 30] : t < 1.5 ? [46, 30] : [51.5, 30];
  const model = buildModel(result(3, layout, ballPath));
  assert.equal(model.events.filter((e) => e.type === "pass").length, 0);
  const lost = model.events.filter((e) => e.type !== "pass");
  assert.equal(lost.length, 1);
  assert.equal(lost[0].team, 0);
  assert.equal(lost[0].wonBy, 1);
  assert.equal(model.teams[0].accuracy, 0);
});

test("contested balls between two close players are not counted as touches", () => {
  const layout = () => [
    player(1, 0, 40, 30),
    player(2, 1, 40.6, 30),
    player(3, 0, 20, 10),
    player(4, 1, 80, 50),
  ];
  const model = buildModel(result(2, layout, () => [40.3, 30]));
  assert.equal(model.spells.length, 0);
  assert.equal(model.teams[0].possession, null);
});

test("goal sides follow the deepest defender and can be flipped", () => {
  const frames = [
    {
      time: 0,
      valid: true,
      ball: [30, 34],
      players: [
        player(1, 0, 10, 30),
        player(2, 0, 35, 30),
        player(3, 1, 25, 20),
        player(4, 1, 40, 40),
      ],
    },
  ];
  assert.equal(inferSides(frames, 105)[0], "left");
  assert.equal(inferSides(frames, 105, true)[0], "right");
});

test("smoothed tracks never bridge an unmapped frame", () => {
  const layout = (t) => [player(1, 0, 20 + t * 5, 30), player(2, 0, 30, 30)];
  const model = buildModel(
    result(
      3,
      layout,
      () => null,
      (t) => Math.abs(t - 1.5) > 0.01,
    ),
  );
  const track = model.trackById.get(1);
  assert.equal(track.segments.length, 2);
  assert.equal(
    model.playersAt(1.5).find((p) => p.id === 1),
    undefined,
  );
  const mid = model.playersAt(0.75).find((p) => p.id === 1);
  assert.ok(Math.abs(mid.x - 23.75) < 0.3);
  assert.ok(Math.abs(mid.vx - 5) < 0.5);
});

test("distance and speed come from smoothed motion", () => {
  const layout = (t) => [player(1, 0, 10 + t * 6, 30), player(2, 0, 30, 20)];
  const model = buildModel(result(4, layout, () => null));
  const runner = model.players.find((p) => p.id === 1);
  assert.ok(Math.abs(runner.distance - 24) < 1.5, runner.distance);
  assert.ok(Math.abs(runner.topSpeed - 6) < 0.5, runner.topSpeed);
  assert.equal(runner.sprints, 1);
  const still = model.players.find((p) => p.id === 2);
  assert.ok(still.distance < 0.5);
  assert.equal(still.sprints, 0);
});

test("ball positions interpolate only over short gaps", () => {
  const model = buildModel(
    result(
      2,
      () => [player(1, 0, 10, 10)],
      (t) => (t < 0.55 || t > 1.5 ? [t * 10, 20] : null),
    ),
  );
  assert.ok(Math.abs(ballAt(model, 0.25).x - 2.5) < 1e-6);
  assert.equal(ballAt(model, 1), null);
});

test("an unmarked attacker in front of goal is reported as a coverage gap", () => {
  const model = buildModel(
    result(
      0,
      () => [player(1, 0, 50, 30)],
      () => null,
    ),
  );
  model.defends = { 0: "left", 1: "right" };
  const still = (id, team, x, y) => ({
    id,
    team,
    kind: "player",
    x,
    y,
    vx: 0,
    vy: 0,
  });
  const defenders = [
    still(1, 0, 25, 10),
    still(2, 0, 25, 20),
    still(3, 0, 30, 15),
  ];
  const open = coverageAt(
    model,
    [...defenders, still(9, 1, 14, 50)],
    { x: 35, y: 40 },
    {
      defending: 0,
    },
  );
  assert.ok(open.gaps.length >= 1);
  const biggest = open.gaps[0];
  assert.ok(
    biggest.y > 34,
    `gap should be on the open side, got y=${biggest.y}`,
  );
  const tight = coverageAt(
    model,
    [
      still(1, 0, 15, 20),
      still(2, 0, 15, 34),
      still(3, 0, 15, 48),
      still(4, 0, 25, 27),
      still(5, 0, 25, 41),
      still(6, 0, 6, 34),
      still(9, 1, 30, 34),
    ],
    { x: 30, y: 34 },
    { defending: 0 },
  );
  assert.ok(
    tight.gaps.every(
      (g) => Math.abs(g.y - 34) > 18 || g.area < open.gaps[0].area,
    ),
  );
  assert.equal(tight.backLine.length, 3);
  assert.ok(tight.backLine.every((s) => s.distance > 0));
});

test("auto coverage defends against the team in possession", () => {
  const model = { ownerAt: (t) => (t < 1 ? 0 : null) };
  assert.equal(defendingTeam(model, 0.5), 1);
  assert.equal(defendingTeam(model, 2), null);
  // Without possession, the team whose goal is in the ball's half defends.
  const sided = { ...model, length: 105, defends: { 0: "left", 1: "right" } };
  assert.equal(defendingTeam(sided, 2, "auto", { x: 80, y: 30 }), 1);
  assert.equal(defendingTeam(sided, 2, "auto", { x: 20, y: 30 }), 0);
  assert.equal(defendingTeam(model, 2, "0"), 0);
});

function lookAtCamera(eye, target, focal, w, h) {
  // Pitch frame (x, y, z) with z = -height, the convention of a top-down map.
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const norm = (a) => a.map((v) => v / Math.hypot(...a));
  const cross = (a, b) => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  const forward = norm(sub(target, eye));
  const right = norm(cross(forward, [0, 0, -1]));
  const down = cross(forward, right);
  const R = [right, down, forward];
  const t = R.map(
    (row) => -(row[0] * eye[0] + row[1] * eye[1] + row[2] * eye[2]),
  );
  const K = [
    [focal, 0, w / 2],
    [0, focal, h / 2],
    [0, 0, 1],
  ];
  const plane = [0, 1, 2].map((r) => [R[r][0], R[r][1], t[r]]);
  const G = K.map((row) =>
    [0, 1, 2].map(
      (c) => row[0] * plane[0][c] + row[1] * plane[1][c] + row[2] * plane[2][c],
    ),
  );
  return invert3(G);
}

test("the film camera is recovered from an image-to-pitch homography", () => {
  const H = lookAtCamera([52.5, 88, -15], [52.5, 34, 0], 1500, 1920, 1080);
  const camera = filmCamera(H, 1920, 1080, 105, 68);
  assert.ok(camera);
  assert.ok(Math.abs(camera.position[0]) < 0.05);
  assert.ok(Math.abs(camera.position[1] - 15) < 0.05);
  assert.ok(Math.abs(camera.position[2] - 54) < 0.05);
  assert.ok(Math.abs(camera.focal - 1500) < 1);
  const [right, up, back] = camera.basis;
  assert.ok(right[0] > 0.99);
  assert.ok(up[1] > 0.9);
  assert.ok(
    back[2] > 0.9,
    "camera looks from the near side toward the far side",
  );
  // A scaled, sign-flipped homography describes the same camera.
  const flipped = filmCamera(
    H.map((row) => row.map((v) => -3 * v)),
    1920,
    1080,
    105,
    68,
  );
  assert.ok(Math.abs(flipped.position[1] - 15) < 0.05);
});

test("visible cells are the ones projecting inside the image", () => {
  const H = lookAtCamera([52.5, 88, -15], [52.5, 34, 0], 1500, 1920, 1080);
  const mask = visibleMask(H, 1920, 1080, 21, 14, 5);
  const at = (x, y) => mask[Math.floor(y / 5) * 21 + Math.floor(x / 5)];
  assert.equal(at(52.5, 34), 1);
  assert.equal(at(2, 66), 0, "the near corner is outside a 65 degree view");
});
