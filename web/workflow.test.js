import test from "node:test";
import assert from "node:assert/strict";
import { passageWindow, runAction } from "./workflow.js";

test("later passages keep source indices and stop at the end of a clip", () => {
  assert.deepEqual(passageWindow({ frames: 300, fps: 30 }, 240, 300), {
    start: 240,
    end: 300,
    seconds: 2,
    sourceSeconds: 8,
  });
  assert.equal(passageWindow({ frames: 300 }, 300, 30), null);
  assert.equal(passageWindow({}, -1, 30), null);
  assert.equal(passageWindow({}, 1.5, 30), null);
  assert.equal(passageWindow({}, 10, 30).end, 40);
});

test("history keeps stopped runs actionable without treating them as completed", () => {
  assert.equal(runAction({ state: "complete" }), "Open analysis");
  assert.equal(runAction({ state: "running" }), "View progress");
  assert.equal(
    runAction({ state: "interrupted", resumable: true }),
    "Review & resume",
  );
  assert.equal(
    runAction({ state: "failed", resumable: false }),
    "View details",
  );
});
