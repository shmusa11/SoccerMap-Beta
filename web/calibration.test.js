import test from "node:test";
import assert from "node:assert/strict";
import { CalibrationEditor, imagePoint } from "./calibration.js";

test("point placement preserves source pixels through responsive sizing and zoom", () => {
  assert.deepEqual(
    imagePoint(
      260,
      140,
      { left: 20, top: 5, width: 480, height: 270 },
      1920,
      1080,
    ),
    { x: 960, y: 540 },
  );
  assert.deepEqual(
    imagePoint(
      500,
      275,
      { left: 20, top: 5, width: 960, height: 540 },
      1920,
      1080,
    ),
    { x: 960, y: 540 },
  );
});

test("dragged points are clamped inside the original image", () => {
  assert.deepEqual(
    imagePoint(
      -100,
      2000,
      { left: 20, top: 5, width: 480, height: 270 },
      1920,
      1080,
    ),
    { x: 0, y: 1079 },
  );
});

test("point handles retain their displayed size when the image is resized", () => {
  let displayedWidth = 960;
  const circles = [];
  const context = {
    drawImage() {},
    beginPath() {},
    fill() {},
    stroke() {},
    fillText() {},
    arc(x, y, radius) {
      circles.push({ x, y, radius });
    },
  };
  const editor = {
    baseImage: {},
    active: 0,
    points: [{ x: 960, y: 540 }],
    canvas: {
      width: 1920,
      getBoundingClientRect: () => ({ width: displayedWidth }),
      getContext: () => context,
    },
  };
  for (const width of [960, 318, 1440]) {
    displayedWidth = width;
    CalibrationEditor.prototype.drawImage.call(editor);
    const circle = circles.at(-1);
    assert.ok(Math.abs((circle.radius * width) / 1920 - 12) < 1e-9);
    assert.equal(circle.x, 960);
    assert.equal(circle.y, 540);
  }
});
