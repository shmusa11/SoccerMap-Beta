/**
 * Shared drawing for the field overlays: defensive coverage and coach drawings. All input is
 * in pitch metres; each view supplies `project([x, y]) -> [px, py] | null`, so the same marks
 * appear on the 2D map, painted on the 3D grass, and pinned to the field in the film.
 */

const GAP = [255, 193, 59];
const UNSEEN = "rgba(12, 24, 18, 0.42)";

function rgb(hex) {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

let heatCanvas = null;

/** Low-resolution coverage image (one pixel per cell); scaled up smoothly when drawn. */
function heatImage(cov, color) {
  heatCanvas ||= document.createElement("canvas");
  if (heatCanvas.width !== cov.nx || heatCanvas.height !== cov.ny) {
    heatCanvas.width = cov.nx;
    heatCanvas.height = cov.ny;
  }
  const ctx = heatCanvas.getContext("2d");
  const image = ctx.createImageData(cov.nx, cov.ny);
  const [r, g, b] = rgb(color);
  for (let k = 0; k < cov.control.length; k++) {
    const o = k * 4;
    if (cov.visible && !cov.visible[k]) {
      image.data.set([12, 24, 18, 105], o);
    } else if (cov.zone?.[k] && cov.control[k] < 0.47) {
      // Feathered: strongest where attackers clearly arrive first.
      const open = Math.min(1, (0.47 - cov.control[k]) / 0.3);
      image.data.set([...GAP, Math.round(175 * open)], o);
    } else if (cov.gap[k]) {
      image.data.set([...GAP, 165], o);
    } else if (cov.control[k] >= 0.5) {
      image.data.set(
        [r, g, b, Math.round(40 + 190 * (cov.control[k] - 0.5))],
        o,
      );
    }
  }
  ctx.putImageData(image, 0, 0);
  return heatCanvas;
}

function pill(
  ctx,
  text,
  x,
  y,
  { fill = "rgba(18,30,24,0.82)", ink = "#fff", size = 12 } = {},
) {
  ctx.font = `600 ${size}px -apple-system, "Segoe UI", sans-serif`;
  const w = ctx.measureText(text).width + size * 1.1,
    h = size * 1.75;
  ctx.fillStyle = fill;
  ctx.beginPath();
  ctx.roundRect(x - w / 2, y - h / 2, w, h, h / 2);
  ctx.fill();
  ctx.fillStyle = ink;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y + 0.5);
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
}

/**
 * Coverage on an axis-aligned pitch view. `rect` is the pixel rectangle of the full pitch.
 * `scale` sizes labels and line widths for the target (1 = map pixels).
 */
export function drawCoverage(
  ctx,
  cov,
  project,
  rect,
  { color, scale = 1, labels = true },
) {
  if (!cov) return;
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(heatImage(cov, color), rect.x, rect.y, rect.w, rect.h);
  // Back line: distances between the deepest visible defenders.
  for (const seg of cov.backLine) {
    const a = project([seg.a.x, seg.a.y]),
      b = project([seg.b.x, seg.b.y]);
    if (!a || !b) continue;
    ctx.strokeStyle =
      seg.level === "wide"
        ? "#ff6b4a"
        : seg.level === "stretched"
          ? "#ffc13b"
          : "rgba(255,255,255,0.85)";
    ctx.lineWidth = 2.5 * scale;
    ctx.setLineDash(seg.level === "ok" ? [6 * scale, 5 * scale] : []);
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
    ctx.setLineDash([]);
    if (labels && seg.level !== "ok")
      pill(
        ctx,
        `${Math.round(seg.distance)} m`,
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2,
        {
          fill:
            seg.level === "wide"
              ? "rgba(140,36,20,0.9)"
              : "rgba(110,76,10,0.9)",
          size: 11 * scale,
        },
      );
  }
  for (const g of labels ? cov.gaps.slice(0, 2) : []) {
    const p = project([g.x, g.y]);
    if (p)
      pill(ctx, `Open space · ${Math.round(g.area)} m²`, p[0], p[1], {
        fill: "rgba(92,62,0,0.88)",
        ink: "#ffe7a6",
        size: 11.5 * scale,
      });
  }
  ctx.restore();
}

export { UNSEEN };

const HEAD = 1.9; // m
const HEAD_ANGLE = 0.5; // rad

function strokeOutlined(ctx, color, width, path) {
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(10,18,14,0.55)";
  ctx.lineWidth = width + 3;
  path();
  ctx.stroke();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  path();
  ctx.stroke();
}

function polyline(ctx, points) {
  ctx.beginPath();
  points.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
}

/**
 * Coach drawings. Pitch marks are shaped in metres then projected, so arrowheads and circles
 * lie on the grass in perspective. Film-only marks (drawn where no field map exists) use
 * `projectFilm` with normalised [0, 1] coordinates.
 */
export function drawMarks(
  ctx,
  marks,
  project,
  { width = 3, projectFilm = null } = {},
) {
  ctx.save();
  for (const mark of marks) {
    const film = mark.space === "film";
    if (film && !projectFilm) continue;
    const toScreen = film ? projectFilm : project;
    const pts = mark.points;
    if (!pts.length) continue;
    if (mark.tool === "pen") {
      const screen = pts.map(toScreen).filter(Boolean);
      if (screen.length > 1)
        strokeOutlined(ctx, mark.color, width, () => polyline(ctx, screen));
    } else if (mark.tool === "circle") {
      const [cx, cy] = pts[0],
        [ex, ey] = pts[pts.length - 1];
      const r = Math.hypot(ex - cx, ey - cy);
      if (r <= 0) continue;
      // Film-space circles use the frame aspect so they stay round on screen.
      const ring = [];
      for (let i = 0; i <= 56; i++) {
        const a = (i / 56) * Math.PI * 2;
        ring.push([
          cx + r * Math.cos(a),
          cy + r * Math.sin(a) * (film ? mark.aspect || 1 : 1),
        ]);
      }
      const screen = ring.map(toScreen).filter(Boolean);
      if (screen.length > 2) {
        polyline(ctx, screen);
        ctx.fillStyle = mark.color + "26";
        ctx.fill();
        strokeOutlined(ctx, mark.color, width, () => polyline(ctx, screen));
      }
    } else {
      const a = pts[0],
        b = pts[pts.length - 1];
      const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (len <= 0) continue;
      const head = film
        ? Math.min(0.035, len * 0.4)
        : Math.min(HEAD, len * 0.45);
      const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
      const wing = (s) => [
        b[0] - head * Math.cos(ang + s * HEAD_ANGLE),
        b[1] -
          head * Math.sin(ang + s * HEAD_ANGLE) * (film ? mark.aspect || 1 : 1),
      ];
      // Sample the shaft so a projected arrow follows the ground in perspective.
      const shaft = [];
      for (let i = 0; i <= 12; i++)
        shaft.push([
          a[0] + ((b[0] - a[0]) * i) / 12,
          a[1] + ((b[1] - a[1]) * i) / 12,
        ]);
      const s = shaft.map(toScreen).filter(Boolean);
      const tip = toScreen(b),
        left = toScreen(wing(1)),
        right = toScreen(wing(-1));
      if (s.length < 2 || !tip || !left || !right) continue;
      strokeOutlined(ctx, mark.color, width, () => polyline(ctx, s));
      ctx.beginPath();
      ctx.moveTo(...tip);
      ctx.lineTo(...left);
      ctx.lineTo(...right);
      ctx.closePath();
      ctx.fillStyle = "rgba(10,18,14,0.55)";
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(10,18,14,0.55)";
      ctx.stroke();
      ctx.fillStyle = mark.color;
      ctx.fill();
    }
  }
  ctx.restore();
}

/**
 * Project pitch metres into display pixels with a pitch-to-image matrix, scaled by (sx, sy)
 * and offset by (ox, oy). `sign` is the homogeneous sign of points in front of the camera;
 * points behind it return null instead of wrapping onto the screen.
 */
export function projector(pitchToImage, sx, sy, ox = 0, oy = 0, sign = 1) {
  const G = pitchToImage;
  return ([x, y]) => {
    const w = G[2][0] * x + G[2][1] * y + G[2][2];
    if (!(w * sign > 1e-9)) return null;
    const u = (G[0][0] * x + G[0][1] * y + G[0][2]) / w,
      v = (G[1][0] * x + G[1][1] * y + G[1][2]) / w;
    return Number.isFinite(u) && Number.isFinite(v)
      ? [ox + u * sx, oy + v * sy]
      : null;
  };
}
