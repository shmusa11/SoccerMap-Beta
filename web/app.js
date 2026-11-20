import { frameAt, summarize, PlaybackClock } from "./analysis.js";
import { CalibrationEditor } from "./calibration.js";
import { passageWindow, runAction } from "./workflow.js";
import {
  ballAt,
  buildModel,
  coverageAt,
  defendingTeam,
  filmCamera,
  invert3,
  passNetwork,
  visibleMask,
} from "./insights.js";
import { drawCoverage, drawMarks, projector } from "./overlay.js";
import { Telestrator } from "./telestrator.js";

const $ = (id) => document.getElementById(id);
const escape = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const count = (n, one, many) => `${n} ${n === 1 ? one : many}`;
// Fitted field dimensions are fractional; show one decimal unless whole.
const metres = (v) => (Number.isInteger(v) ? String(v) : v.toFixed(1));
const stamp = (seconds) =>
  `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
let upload = null,
  job = null,
  result = null,
  timer = null,
  uploading = false;
let playbackFallback = false,
  manualPlaying = false;
const manualClock = new PlaybackClock();
const settingsForm = $("settings-form");
const video = $("result-video");
const titles = {
  import: [
    "Analyze a clip",
    "Turn your footage into a field map you can review.",
  ],
  configure: [
    "Set up your clip",
    "Choose a camera setup and how much footage to process.",
  ],
  processing: [
    "Analysis in progress",
    "Analysis uses the local worker and your selected cloud preference.",
  ],
  review: [
    "Review your analysis",
    "Compare the movement in the film and on the field.",
  ],
  saved: [
    "Analysis history",
    "Reopen a result or continue a stopped analysis on this computer.",
  ],
  environment: ["Local models", "Local components for real-video analysis."],
  about: [
    "About SoccerMap",
    "What this prototype can show, and what still needs work.",
  ],
};
const cameraEditor = new CalibrationEditor({
  api,
  currentUpload: () => upload,
  defaultFrame: () => Number(settingsForm.elements.start_frame.value) || 0,
  dimensions: () => ({
    length: Number(settingsForm.elements.namedItem("length").value),
    width: Number(settingsForm.elements.namedItem("width").value),
  }),
  report: notice,
  changed: () => {
    const setup = cameraEditor.setup();
    for (const key of ["length", "width"]) {
      settingsForm.elements.namedItem(key).disabled = Boolean(setup);
      if (setup) settingsForm.elements.namedItem(key).value = setup[key];
    }
    $("camera-note").textContent = setup
      ? `Using ${setup.name}. Every sampled view is checked against the reference; unmatched views stay off the map.`
      : upload?.id === "demo"
        ? "Camera setup applies to real footage. The demo uses generated geometry."
        : "Automatic detection reuses a verified field while the view stays still.";
  },
});

async function api(path, body, options = {}) {
  const response = await fetch(
    path,
    body === undefined
      ? options
      : {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-SoccerMap": "local",
          },
          body: JSON.stringify(body),
          ...options,
        },
  );
  const data = await response.json();
  if (!response.ok)
    throw Error(data.error || "The local request failed. Try again.");
  return data;
}
function notice(message) {
  $("notice").textContent = message;
  $("notice").hidden = !message;
}
function handle(fn) {
  return async (event) => {
    try {
      await fn(event);
    } catch (error) {
      notice(error.message);
    }
  };
}
function on(id, event, fn) {
  $(id).addEventListener(event, handle(fn));
}
function route(view) {
  if (location.hash === `#${view}`) show(view);
  else location.hash = view;
}
async function show(view) {
  if (!(view in titles)) view = "import";
  if (view === "configure" && !upload) view = "import";
  if (view === "review" && !result) view = job ? "processing" : "import";
  if (view === "processing" && !job) view = "import";
  document.querySelectorAll(".view").forEach((el) => {
    el.hidden = el.id !== `view-${view}`;
  });
  $("page-title").textContent = titles[view][0];
  $("page-subtitle").textContent = titles[view][1];
  document.querySelectorAll("[data-nav]").forEach((el) => {
    el.removeAttribute("aria-current");
    if (
      el.dataset.nav ===
      (["saved", "environment"].includes(view) ? view : "analyze")
    )
      el.setAttribute("aria-current", "page");
  });
  document.querySelector(".steps").hidden = [
    "saved",
    "environment",
    "about",
  ].includes(view);
  ["import", "configure", "review"].forEach((step) => {
    $(`step-${step}`).classList.toggle(
      "current",
      step === (view === "processing" ? "configure" : view),
    );
  });
  if (view !== "review") {
    video.pause();
    manualPlaying = false;
  }
  notice("");
  if (view === "import") await loadClips();
  if (view === "saved") await loadSaved();
  if (view === "environment") await loadEnvironment();
  if (view === "review") renderPlayback();
  if (view === "processing") renderProgress();
}
window.addEventListener(
  "hashchange",
  handle(() => show(location.hash.slice(1))),
);

function config() {
  const form = new FormData(settingsForm),
    settings = {};
  [
    "length",
    "width",
    "start_frame",
    "max_frames",
    "stride",
    "ball_every",
  ].forEach((key) => {
    settings[key] = Number(settingsForm.elements.namedItem(key).value);
  });
  [
    "players",
    "ball",
    "enhance",
    "multisport",
    "verify_lines",
    "low_memory",
    "register_views",
    "ai_keyframes",
  ].forEach((key) => {
    settings[key] = form.has(key);
  });
  settings.engine = settingsForm.elements.engine.value;
  settings.camera_setup = $("camera-setup").value || null;
  return settings;
}
async function configureClip() {
  $("clip-summary").innerHTML =
    `<div><strong>${escape(upload.name)}</strong><p>${upload.id === "demo" ? "60-second synthetic fixture · 30 fps · no model downloads" : `${upload.width} × ${upload.height} · ${upload.fps.toFixed(2)} fps · ${upload.duration ? stamp(upload.duration) : "unknown duration"}`}</p></div><span class="tag">${upload.id === "demo" ? "Synthetic demo" : "Local clip"}</span>`;
  settingsForm.elements.max_frames.value = upload.id === "demo" ? 1800 : 300;
  settingsForm.elements.start_frame.value = 0;
  settingsForm.elements.start_frame.disabled = upload.id === "demo";
  settingsForm.elements.start_frame.max =
    upload.frames > 0 ? upload.frames - 1 : 100000000;
  $("start-run").textContent =
    upload.id === "demo" ? "Run synthetic demo" : "Run analysis";
  $("run-note").textContent =
    upload.id === "demo"
      ? "Generated positions. Demo settings control sampling and visible layers; field-detection options apply to real clips."
      : "Settings are fixed when the run starts. First-time model setup may download weights.";
  updateDuration();
  await cameraEditor.refresh();
  route("configure");
}
function updateDuration() {
  const passage = passageWindow(
    upload,
    Number(settingsForm.elements.start_frame.value),
    Number(settingsForm.elements.max_frames.value),
  );
  $("duration-hint").textContent = passage
    ? `${passage.seconds.toFixed(1)} seconds from ${stamp(passage.sourceSeconds)} in the source. Original frames ${passage.start}–${passage.end - 1}.`
    : "Choose a starting frame inside the video and a positive passage length.";
  $("stride-value").textContent = settingsForm.elements.stride.value;
}
async function loadClips() {
  const clips = await api("/api/uploads");
  $("imported-clips").hidden = !clips.length;
  const list = $("clip-list");
  list.innerHTML = clips
    .map(
      (clip) =>
        `<article class="saved-row"><div><h3>${escape(clip.name)}</h3><p>${clip.storage === "linked" ? "Linked original" : "Local copy"} · ${clip.duration ? stamp(clip.duration) : "Unknown duration"}${clip.available ? "" : " · Missing or changed; import again"}</p></div><button data-clip="${clip.id}" ${clip.available ? "" : "disabled"}>Use clip</button></article>`,
    )
    .join("");
  list.querySelectorAll("[data-clip]").forEach((button) =>
    button.addEventListener(
      "click",
      handle(async () => {
        if (uploading || job?.state === "running")
          throw Error(
            "Finish the import or active analysis before changing clips.",
          );
        upload = clips.find((clip) => clip.id === button.dataset.clip);
        await configureClip();
      }),
    ),
  );
}
async function uploadFile(file) {
  if (!file || uploading) return;
  if (job?.state === "running")
    throw Error("Cancel or finish the active analysis before changing clips.");
  if (
    !/\.(mp4|mov|avi|m4v)$/i.test(file.name) ||
    file.size === 0 ||
    file.size > 500 * 1024 * 1024
  )
    throw Error("Choose a non-empty MP4, MOV, AVI or M4V under 500 MB.");
  uploading = true;
  $("clip-file").disabled = true;
  $("use-demo").disabled = true;
  notice("Importing and checking the video on this computer…");
  try {
    const response = await fetch("/api/uploads", {
      method: "POST",
      headers: {
        "X-SoccerMap": "local",
        "Content-Type": "application/octet-stream",
        "X-File-Name": encodeURIComponent(file.name),
      },
      body: file,
    });
    const data = await response.json();
    if (!response.ok) throw Error(data.error);
    upload = data;
    await configureClip();
  } finally {
    uploading = false;
    $("clip-file").disabled = false;
    $("use-demo").disabled = false;
    $("clip-file").value = "";
  }
}
on("clip-file", "change", (e) => uploadFile(e.target.files[0]));
on("use-demo", "click", async () => {
  if (job?.state === "running")
    throw Error("Cancel or finish the active analysis before changing clips.");
  upload = { id: "demo", name: "Riverside training · synthetic demo", fps: 30 };
  await configureClip();
});
on("dropzone", "dragover", (e) => {
  e.preventDefault();
  $("dropzone").classList.add("dragover");
});
on("dropzone", "dragleave", () => $("dropzone").classList.remove("dragover"));
on("dropzone", "drop", (e) => {
  e.preventDefault();
  $("dropzone").classList.remove("dragover");
  return uploadFile(e.dataTransfer.files[0]);
});
on("change-clip", "click", () => route("import"));
settingsForm.addEventListener("input", updateDuration);
settingsForm.addEventListener(
  "submit",
  handle(async (e) => {
    e.preventDefault();
    if (!upload || !settingsForm.reportValidity()) return;
    if (!$("camera-editor").hidden) {
      cameraEditor.report(
        "Save the camera setup, or close the editor before starting this run.",
      );
      return;
    }
    $("start-run").disabled = true;
    notice("");
    try {
      job = await api("/api/runs", { upload: upload.id, settings: config() });
      result = null;
      sessionStorage.setItem("soccermap-run", job.id);
      route("processing");
      renderProgress();
      schedulePoll();
    } finally {
      $("start-run").disabled = false;
    }
  }),
);
function renderProgress() {
  const active = job.state === "running";
  $("process-run-name").textContent = job.name;
  const retained = job.settings;
  $("process-settings").innerHTML = [
    [
      "Passage",
      `Source frame ${retained.start_frame || 0} · up to ${retained.max_frames} original frames`,
    ],
    [
      "Camera",
      job.synthetic
        ? "Synthetic demo geometry"
        : job.camera_name || "Automatic field detection",
    ],
    [
      "Sampling",
      `Every ${retained.stride} input frame${retained.stride === 1 ? "" : "s"}${retained.ball && retained.players ? ` · ball every ${retained.ball_every} samples` : " · ball off"}`,
    ],
    [
      "Processing",
      `${retained.low_memory ? "Low memory mode" : "Accelerated mode"} · players ${retained.players ? "on" : "off"}`,
    ],
  ]
    .map(([key, value]) => `<dt>${key}</dt><dd>${escape(value)}</dd>`)
    .join("");
  $("process-description").textContent = active
    ? "With cloud AI off, footage stays local. If enabled in fast mode, sampled frames go to OpenAI. You can leave this page open while processing finishes."
    : "Your source video is unchanged. Resume with these settings, or start a new analysis.";
  if (location.hash === "#processing")
    $("page-title").textContent = active
      ? "Analysis in progress"
      : "Analysis stopped";
  $("process-title").textContent = active
    ? job.synthetic
      ? "Building the synthetic walkthrough"
      : "Analyzing your clip"
    : job.state === "complete"
      ? "Ready to review"
      : "Your analysis has stopped";
  $("process-stage").textContent = job.error || job.stage;
  $("process-percent").textContent = `${Math.round(job.progress * 100)}%`;
  $("process-progress").value = job.progress;
  const elapsed = Math.max(
    0,
    (active ? Date.now() / 1000 : job.updated || job.created) -
      (job.attempt_started || job.created),
  );
  $("process-time").textContent =
    `${stamp(elapsed)} elapsed${job.camera_name ? ` · Camera: ${job.camera_name}` : ""}. Processing time depends on the clip and this computer.`;
  $("cancel-run").hidden = !active;
  $("resume-run").hidden = !job.resumable;
  $("resume-note").hidden = !job.resumable;
  $("retry-run").hidden = active || job.state === "complete";
  $("retry-run").textContent = upload
    ? "Review settings & retry"
    : "Choose a clip to retry";
}
function schedulePoll() {
  clearTimeout(timer);
  timer = setTimeout(handle(poll), 600);
}
async function poll() {
  if (!job) return;
  try {
    job = await api(`/api/runs/${job.id}`);
    renderProgress();
    if (job.state === "complete") {
      await openResult(job.id);
    } else if (job.state === "running") schedulePoll();
  } catch (error) {
    notice(
      "Connection to the local workspace was lost. Keep this page open and restart the server.",
    );
    schedulePoll();
  }
}
on("cancel-run", "click", async () => {
  $("cancel-run").disabled = true;
  clearTimeout(timer);
  try {
    job = await api(`/api/runs/${job.id}/cancel`, {});
    renderProgress();
    if (job.state === "complete") await openResult(job.id);
  } finally {
    $("cancel-run").disabled = false;
  }
});
on("resume-run", "click", async () => {
  $("resume-run").disabled = true;
  try {
    job = await api(`/api/runs/${job.id}/resume`, {});
    result = null;
    renderProgress();
    schedulePoll();
  } finally {
    $("resume-run").disabled = false;
  }
});
on("retry-run", "click", () => route(upload ? "configure" : "import"));

async function openResult(id) {
  clearTimeout(timer);
  video.pause();
  manualPlaying = false;
  job = await api(`/api/runs/${id}`);
  upload = job.upload
    ? (await api("/api/uploads")).find(
        (clip) => clip.id === job.upload && clip.available,
      ) || null
    : null;
  result = await api(`/api/runs/${id}/result`);
  sessionStorage.setItem("soccermap-run", id);
  $("review-name").textContent = job.name;
  $("review-detail").textContent =
    `${metres(result.settings.length)} × ${metres(result.settings.width)} m field · ${result.frames.length} sampled frames · ${stamp(result.duration)}`;
  $("provenance").textContent = result.synthetic
    ? "Synthetic demo · Invented players and positions. This demonstrates the interface; no footage was analyzed."
    : "Local analysis · Estimated positions from your clip. Missing mappings and ball observations remain unavailable.";
  $("provenance").classList.toggle("real", !result.synthetic);
  $("run-warnings").textContent = result.synthetic
    ? ""
    : result.warnings.join(" ");
  $("run-warnings").hidden = !result.warnings.length || result.synthetic;
  $("repair-mapping").hidden =
    result.synthetic || !upload || result.coverage >= 1;
  $("save-current-view").hidden =
    result.synthetic || !job.upload || !result.frames.some((f) => f.homography);
  $("review-camera").hidden = true;
  const diagnostics = $("run-diagnostics");
  diagnostics.replaceChildren();
  if (result.timings) {
    const title = document.createElement("h2");
    title.textContent = "Processing time";
    diagnostics.append(title);
    const list = document.createElement("dl");
    const labels = {
      models: "Model loading",
      calibration: "Field mapping",
      detection_and_solve: "Detection and paths",
      analysis: "Field, players and ball",
      render: "Video rendering",
    };
    for (const [key, seconds] of Object.entries(result.timings)) {
      const term = document.createElement("dt"),
        value = document.createElement("dd");
      term.textContent = labels[key] || key;
      value.textContent = `${seconds.toFixed(1)} s`;
      list.append(term, value);
    }
    diagnostics.append(list);
    const detail = document.createElement("p");
    detail.textContent = `${result.calibration?.inference_calls ?? "Unknown"} field-model calls · ${result.calibration?.reused ?? 0} views reused after a camera check. Coverage measures availability, not accuracy.`;
    const ai = result.ai_keyframes;
    if (ai?.enabled)
      detail.textContent += ` AI keyframes: ${"$"}${ai.usd.toFixed(2)} of the ${"$"}${ai.cap_usd.toFixed(2)} per-run cap.`;
    if (result.reused_stages?.length)
      detail.textContent += ` Recovered completed stages: ${result.reused_stages.join(", ")}. Times above cover this attempt.`;
    diagnostics.append(detail);
  }
  $("review-facts").innerHTML =
    `<span><strong>${new Set(result.frames.flatMap((f) => f.players.map((p) => p.id))).size}</strong>Observed track IDs</span><span><strong>${Math.round(result.coverage * 100)}%</strong>Sampled frames with a map</span><span><strong>${result.ball_available ? "Enabled" : "Unavailable"}</strong>Ball layer</span><span><strong>${result.synthetic ? "Synthetic" : "Local"}</strong>${result.synthetic ? "Illustrative data" : "Processing on this computer"}</span>`;
  $("timeline").max = Math.max(
    0,
    result.duration - result.settings.stride / result.source.fps,
  );
  $("timeline").step = result.settings.stride / result.source.fps;
  $("timeline").value = 0;
  $("interval-start").value = 0;
  $("interval-end").value = result.duration.toFixed(2);
  $("interval-start").max = result.duration;
  $("interval-end").max = result.duration;
  $("jump-gap").hidden = !result.frames.some((f) => !f.valid);
  playbackFallback = false;
  $("video-error").hidden = true;
  video.src = `/api/runs/${id}/review`;
  video.load();
  $("download-video").href = `/api/runs/${id}/video?download=1`;
  $("download-csv").href = `/api/runs/${id}/csv?download=1`;
  $("export-note").textContent = result.synthetic
    ? "All downloads are labeled as synthetic demo data, including the video itself."
    : "Exported files describe this completed run. Review warnings and missing data before sharing.";
  $("save-run").textContent = job.saved
    ? "Saved on this computer"
    : "Save analysis";
  $("save-run").disabled = job.saved;
  makeTrackChoices();
  prepareInsights();
  chooseTab("playback");
  route("review");
}
let reviewedFrame = null;
on("save-current-view", "click", async () => {
  video.pause();
  manualPlaying = false;
  const frame = frameAt(result.frames, Number($("timeline").value));
  const data = await api(`/api/runs/${job.id}/camera/preview`, {
    frame: frame.index,
  });
  reviewedFrame = data.frame;
  $("review-camera-image").src = data.image;
  $("review-camera-frame").textContent =
    `Source frame ${data.frame}. Check several field lines before reusing this view.`;
  $("review-camera-checked").checked = false;
  $("confirm-current-view").disabled = true;
  $("review-camera-saved").textContent = "";
  $("review-camera").hidden = false;
  $("review-camera-name").focus();
});
on("review-camera-checked", "change", () => {
  $("confirm-current-view").disabled = !$("review-camera-checked").checked;
});
on("close-current-view", "click", () => {
  $("review-camera").hidden = true;
});
on("confirm-current-view", "click", async () => {
  if (!$("review-camera-checked").checked || reviewedFrame === null) return;
  $("confirm-current-view").disabled = true;
  try {
    const saved = await api(`/api/runs/${job.id}/camera/save`, {
      frame: reviewedFrame,
      name: $("review-camera-name").value,
      reviewed: true,
    });
    $("review-camera-saved").textContent =
      `Saved “${saved.name}”. Choose it under Camera setup for another clip with this view.`;
  } catch (error) {
    $("confirm-current-view").disabled = false;
    throw error;
  }
});
on("repair-mapping", "click", async () => {
  await configureClip();
  for (const [key, value] of Object.entries(result.settings)) {
    const control = settingsForm.elements.namedItem(key);
    if (!control) continue;
    if (control.type === "checkbox") control.checked = value;
    else control.value = value;
  }
  $("camera-setup").value = "";
  cameraEditor.changed();
  updateDuration();
  await cameraEditor.open();
});
for (const name of ["length", "width"])
  settingsForm.elements.namedItem(name).addEventListener("input", () => {
    $("camera-editor").hidden = true;
    cameraEditor.invalidate();
  });
function chooseTab(name) {
  ["playback", "stats", "shape", "export"].forEach((tab) => {
    $(`tab-${tab}`).hidden = tab !== name;
    document
      .querySelector(`[data-tab="${tab}"]`)
      .setAttribute("aria-pressed", String(tab === name));
  });
  if (name !== "playback") {
    video.pause();
    manualPlaying = false;
  }
  if (name === "shape") renderShape();
  if (name === "stats") renderStats();
  if (name === "playback") requestAnimationFrame(renderPlayback);
}
document
  .querySelectorAll("[data-tab]")
  .forEach((button) =>
    button.addEventListener("click", () => chooseTab(button.dataset.tab)),
  );

function field(canvas) {
  // Draw in the canvas's original logical size, with a backing store matched to the
  // displayed size so the map stays sharp when a layout enlarges it.
  canvas.dataset.logicalWidth ||= canvas.width;
  canvas.dataset.logicalHeight ||= canvas.height;
  const width = Number(canvas.dataset.logicalWidth),
    height = Number(canvas.dataset.logicalHeight),
    margin = 34;
  if (canvas.clientWidth) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const backing = Math.round(canvas.clientWidth * dpr);
    if (canvas.width !== backing) {
      canvas.width = backing;
      canvas.height = Math.round((backing * height) / width);
    }
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);
  const sx = (width - 2 * margin) / result.settings.length,
    sy = (height - 2 * margin) / result.settings.width;
  const point = (x, y) => [margin + x * sx, margin + y * sy];
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#385b3e";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#c7d8b8";
  ctx.lineWidth = 1.5;
  const rect = (x, y, w, h) => ctx.strokeRect(...point(x, y), w * sx, h * sy);
  rect(0, 0, result.settings.length, result.settings.width);
  ctx.beginPath();
  ctx.moveTo(...point(result.settings.length / 2, 0));
  ctx.lineTo(...point(result.settings.length / 2, result.settings.width));
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(
    ...point(result.settings.length / 2, result.settings.width / 2),
    9.15 * sx,
    9.15 * sy,
    0,
    0,
    Math.PI * 2,
  );
  ctx.stroke();
  for (const side of [0, 1]) {
    rect(
      side ? result.settings.length - 16.5 : 0,
      (result.settings.width - 40.32) / 2,
      16.5,
      40.32,
    );
    rect(
      side ? result.settings.length - 5.5 : 0,
      (result.settings.width - 18.32) / 2,
      5.5,
      18.32,
    );
  }
  ctx.fillStyle = "#dae5d0";
  ctx.font = "12px Arial";
  ctx.fillText(`0, 0`, 10, 20);
  ctx.fillText(`${metres(result.settings.length)} m`, width - 65, 20);
  return {
    ctx,
    point,
    sx,
    sy,
    rect: {
      x: margin,
      y: margin,
      w: result.settings.length * sx,
      h: result.settings.width * sy,
    },
    // Pitch metres under a client-space pointer position.
    toPitch(clientX, clientY) {
      const box = canvas.getBoundingClientRect();
      const lx = ((clientX - box.left) / box.width) * width,
        ly = ((clientY - box.top) / box.height) * height;
      return [(lx - margin) / sx, (ly - margin) / sy];
    },
  };
}
function dot(ctx, point, player, ids = true, selected = false) {
  if (!Number.isFinite(player.x) || !Number.isFinite(player.y)) return;
  const [x, y] = point(player.x, player.y);
  ctx.beginPath();
  ctx.arc(x, y, selected ? 10 : 8, 0, Math.PI * 2);
  ctx.fillStyle =
    player.team === 0
      ? $("home-color").value
      : player.team === 1
        ? $("away-color").value
        : "#ced0b9";
  if (player.measured !== false) ctx.fill();
  ctx.strokeStyle = player.measured === false ? ctx.fillStyle : "#fff9df";
  ctx.setLineDash(player.measured === false ? [3, 2] : []);
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);
  if (selected) {
    ctx.beginPath();
    ctx.arc(x, y, 15, 0, Math.PI * 2);
    ctx.strokeStyle = "#ffcf4a";
    ctx.lineWidth = 3;
    ctx.stroke();
  }
  if (ids) {
    ctx.fillStyle = "white";
    ctx.font = "bold 12px Arial";
    ctx.textAlign = "center";
    ctx.fillText(player.id, x, y - 14);
    ctx.textAlign = "left";
  }
}
function renderPlayback() {
  if (!result) return;
  const time = Number($("timeline").value),
    frame = frameAt(result.frames, time);
  $("clock").textContent = `Passage ${stamp(time)} / ${stamp(result.duration)}`;
  $("source-position").textContent = frame
    ? `Source ${stamp(frame.source_time ?? frame.index / result.source.fps)} · original frame ${frame.index}`
    : "Source position unavailable";
  $("play").textContent = !video.paused || manualPlaying ? "Pause" : "Play";
  $("play").setAttribute(
    "aria-label",
    `${!video.paused || manualPlaying ? "Pause" : "Play"} review`,
  );
  $("coverage-status").classList.toggle("unavailable", !frame.valid);
  $("coverage-status").textContent = frame.valid
    ? `Map available at ${stamp(time)} · ${frame.players.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)).length} mapped tracks (${frame.players.filter((p) => p.measured === false && Number.isFinite(p.x)).length} interpolated). ${Math.round(result.coverage * 100)}% of sampled frames have a valid map.`
    : `Map unavailable at ${stamp(time)}. Source video continues; projected positions and this frame’s analytics are excluded.`;
  if (!model) return;
  // Smoothed, interpolated positions keep the map and 3D fluid between sampled frames.
  const players = frame.valid ? model.playersAt(time) : [];
  const ball =
    frame.valid && $("show-ball").checked ? ballAt(model, time) : null;
  const cov = review.coverage && frame.valid ? coverageFor(frame) : null;
  const shown = players.filter(
    (p) =>
      $("map-team").value === "all" || p.team === Number($("map-team").value),
  );
  if (has("map")) {
    const view = field($("pitch"));
    mapView = view;
    $("map-unavailable").hidden = frame.valid;
    if (frame.valid && cov)
      drawCoverage(view.ctx, cov, (p) => view.point(...p), view.rect, {
        color: teamColor(cov.defending),
      });
    drawMarks(view.ctx, telestrator.all(), (p) => view.point(...p), {
      width: 3,
    });
    for (const player of shown)
      dot(
        view.ctx,
        view.point,
        player,
        $("show-ids").checked,
        player.id === review.selected,
      );
    if (ball) {
      const { ctx } = view;
      ctx.beginPath();
      ctx.arc(...view.point(ball.x, ball.y), 6, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      if (ball.measured !== false) ctx.fill();
      ctx.setLineDash(ball.measured === false ? [3, 2] : []);
      ctx.strokeStyle = "#172a20";
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  if (has("film")) drawFilmMarks(frame);
  if (has("3d")) update3D(frame, time, shown, ball, cov);
  updateCoverageNote(frame, cov);
}

/* ---------------- Coach tools: layouts, coverage, drawing, 3D, stats ---------------- */

const review = {
  layout: "film-map",
  coverage: false,
  selected: null,
  camera: "orbit",
  flip: false,
};
let model = null,
  scene = null,
  sceneLoading = null,
  mapView = null,
  playing = false;
const frameCache = new Map();
const telestrator = new Telestrator(() => {
  if (result) renderPlayback();
});
const LAYOUTS = {
  "film-map": ["film", "map"],
  "film-3d": ["film", "3d"],
  film: ["film"],
  map: ["map"],
  "3d": ["3d"],
};
const has = (pane) => LAYOUTS[review.layout].includes(pane);
const teamColor = (team) =>
  team === 0
    ? $("home-color").value
    : team === 1
      ? $("away-color").value
      : "#ced0b9";
const teamName = (team) =>
  team === 0 ? "Team A" : team === 1 ? "Team B" : "Unknown";
const playerName = (p) => `#${p.id}${p.kind === "goalkeeper" ? " GK" : ""}`;
const remember = (key, value) => {
  try {
    localStorage.setItem(`soccermap-${key}`, value);
  } catch {
    /* Storage is a convenience only. */
  }
};
const recall = (key) => {
  try {
    return localStorage.getItem(`soccermap-${key}`);
  } catch {
    return null;
  }
};

function prepareInsights() {
  model = buildModel(result, { flip: review.flip });
  frameCache.clear();
  review.selected = null;
  telestrator.clear();
  if (
    scene &&
    (scene.L !== result.settings.length || scene.W !== result.settings.width)
  ) {
    scene.dispose();
    scene = null;
    sceneLoading = null;
  }
  const options = model.players
    .filter((p) => p.kind !== "referee")
    .sort((a, b) => a.team - b.team || b.seen - a.seen)
    .map(
      (p) =>
        `<option value="${p.id}">${teamName(p.team)} · ${escape(playerName(p))}</option>`,
    );
  $("pov-player").innerHTML = options.join("");
  renderMoments();
  const { width: w, height: h } = result.source;
  $("scene3d").style.setProperty(
    "--film-aspect",
    w && h ? `${w} / ${h}` : "16 / 9",
  );
  setLayout(recall("layout") || "film-map", false);
  if (!$("tab-stats").hidden) renderStats();
}

/** Per-frame geometry, computed once: inverse homography, film camera, visible cells. */
function frameGeometry(frame) {
  if (!frame?.homography) return null;
  let geo = frameCache.get(frame.index);
  if (geo) return geo;
  const H = frame.homography,
    { width: w, height: h } = result.source;
  const G = invert3(H);
  let sign = 1;
  if (G) {
    const cw = H[2][0] * w * 0.5 + H[2][1] * h * 0.5 + H[2][2];
    const qx = (H[0][0] * w * 0.5 + H[0][1] * h * 0.5 + H[0][2]) / cw,
      qy = (H[1][0] * w * 0.5 + H[1][1] * h * 0.5 + H[1][2]) / cw;
    sign = Math.sign(G[2][0] * qx + G[2][1] * qy + G[2][2]) || 1;
  }
  geo = {
    G,
    sign,
    camera:
      w && h
        ? filmCamera(H, w, h, result.settings.length, result.settings.width)
        : null,
    visible: null,
  };
  frameCache.set(frame.index, geo);
  return geo;
}

function coverageFor(frame) {
  const defending = defendingTeam(
    model,
    frame.time,
    $("coverage-team").value,
    ballAt(model, frame.time),
  );
  if (defending === null) return null;
  const geo = frameGeometry(frame);
  const cell = 1,
    nx = Math.ceil(result.settings.length / cell),
    ny = Math.ceil(result.settings.width / cell);
  if (geo && !geo.visible && result.source.width)
    geo.visible = visibleMask(
      frame.homography,
      result.source.width,
      result.source.height,
      nx,
      ny,
      cell,
    );
  const key = `${frame.index}|${defending}|${review.flip}`;
  if (coverageFor.key === key) return coverageFor.value;
  const value = coverageAt(
    model,
    model.playersAt(frame.time),
    ballAt(model, frame.time),
    {
      defending,
      visible: geo?.visible || null,
      cell,
    },
  );
  value.visible = geo?.visible || null;
  coverageFor.key = key;
  coverageFor.value = value;
  return value;
}

function updateCoverageNote(frame, cov) {
  if (!review.coverage) return;
  const note = $("coverage-note");
  if (!frame.valid) note.textContent = "Coverage needs a mapped frame.";
  else if (!cov)
    note.textContent =
      "No ball or possession right now. Choose a team to see its coverage.";
  else {
    const wide = cov.backLine.filter((s) => s.level !== "ok").length;
    note.textContent = `${teamName(cov.defending)} defending · ${cov.gaps.length ? `${cov.gaps.length} open area${cov.gaps.length > 1 ? "s" : ""}` : "no large open areas"}${wide ? ` · ${wide} stretched gap${wide > 1 ? "s" : ""} in the back line` : ""}.`;
  }
}

/** Content rectangle of the video inside its element (object-fit: contain). */
function videoBox() {
  const vw = video.videoWidth || result.source.width || 16,
    vh = video.videoHeight || result.source.height || 9;
  const bw = video.clientWidth,
    bh = video.clientHeight;
  const k = Math.min(bw / vw, bh / vh);
  return { x: (bw - vw * k) / 2, y: (bh - vh * k) / 2, w: vw * k, h: vh * k };
}

function drawFilmMarks(frame) {
  const canvas = $("film-marks");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.round(video.clientWidth * dpr),
    h = Math.round(video.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const marks = telestrator.all();
  if (!marks.length) return;
  const box = videoBox();
  const geo = frame.valid ? frameGeometry(frame) : null;
  const k = box.w / (result.source.width || box.w);
  const project = geo?.G
    ? projector(geo.G, k, k, box.x, box.y, geo.sign)
    : () => null;
  drawMarks(ctx, marks, project, {
    width: 4,
    projectFilm: ([u, v]) => [box.x + u * box.w, box.y + v * box.h],
  });
}

/** Where a film pointer lands: on the field when this frame is mapped, else on the image. */
function locateOnFilm(event) {
  const rect = video.getBoundingClientRect(),
    box = videoBox();
  const px = event.clientX - rect.left - box.x,
    py = event.clientY - rect.top - box.y;
  if (px < 0 || py < 0 || px > box.w || py > box.h) return null;
  const frame = frameAt(result.frames, Number($("timeline").value));
  if (frame?.valid && frame.homography && result.source.width) {
    const H = frame.homography,
      k = result.source.width / box.w;
    const u = px * k,
      v = py * k;
    const w = H[2][0] * u + H[2][1] * v + H[2][2];
    const x = (H[0][0] * u + H[0][1] * v + H[0][2]) / w,
      y = (H[1][0] * u + H[1][1] * v + H[1][2]) / w;
    const L = result.settings.length,
      W = result.settings.width;
    if (Number.isFinite(x) && x > -15 && x < L + 15 && y > -15 && y < W + 15)
      return { space: "pitch", point: [x, y] };
  }
  return {
    space: "film",
    point: [px / box.w, py / box.h],
    aspect: box.w / box.h,
  };
}

async function ensureScene() {
  if (scene) return scene;
  sceneLoading ||= import("./scene3d.js").then(({ Pitch3D }) => {
    scene = new Pitch3D($("scene3d"), {
      length: result.settings.length,
      width: result.settings.width,
      onSelect: (id) => {
        selectPlayer(id);
        if (review.camera !== "pov") setCamera("pov");
      },
    });
    scene.setColors($("home-color").value, $("away-color").value);
    scene.setMode(review.camera);
    const time = Number($("timeline").value);
    scene.frameAction(model.playersAt(time), ballAt(model, time));
    scene.setDrawing(telestrator.active);
    telestrator.attach(scene.renderer.domElement, (event) => {
      const point = scene.pitchAt(event.clientX, event.clientY);
      return point ? { space: "pitch", point } : null;
    });
    return scene;
  });
  return sceneLoading;
}

let overlayCanvas = null;
function update3D(frame, time, players, ball, cov) {
  if (!scene) {
    ensureScene().then(
      () => renderPlayback(),
      (error) => notice(`3D view unavailable: ${error.message}`),
    );
    return;
  }
  const geo = frame.valid ? frameGeometry(frame) : null;
  scene.setShowIds($("show-ids").checked);
  scene.update({
    time,
    players,
    ball,
    film: geo?.camera || null,
    selected: review.selected,
  });
  const marks = telestrator.all();
  if (cov || marks.length) {
    const ppm = 12,
      L = result.settings.length,
      W = result.settings.width;
    overlayCanvas ||= document.createElement("canvas");
    if (overlayCanvas.width !== L * ppm) {
      overlayCanvas.width = L * ppm;
      overlayCanvas.height = W * ppm;
    }
    const ctx = overlayCanvas.getContext("2d");
    ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    const project = ([x, y]) => [x * ppm, y * ppm];
    if (cov)
      drawCoverage(
        ctx,
        cov,
        project,
        { x: 0, y: 0, w: L * ppm, h: W * ppm },
        {
          color: teamColor(cov.defending),
          scale: 2.2,
          // Text painted on the grass reads badly from low 3D angles; the map keeps labels.
          labels: false,
        },
      );
    drawMarks(ctx, marks, project, { width: 5 });
    scene.setOverlay(overlayCanvas);
  } else scene.setOverlay(null);
  const missing =
    review.camera === "pov" && review.selected !== null && frame.valid
      ? !players.some((p) => p.id === review.selected)
      : false;
  $("scene-unavailable").hidden = frame.valid && !missing;
  $("scene-unavailable-title").textContent = frame.valid
    ? `#${review.selected} is off camera`
    : "Map unavailable";
  $("scene-unavailable-text").textContent = frame.valid
    ? "This player is not tracked at this moment. Pick another player or keep playing."
    : "Field geometry could not be confirmed at this moment.";
  $("scene-hint").textContent = {
    orbit: "Drag to orbit · scroll to zoom · click a player to see their view",
    film:
      frame.valid && !geo?.camera
        ? "Film angle unavailable for this frame. Showing the last matched angle."
        : "Matches the film camera. Players are rebuilt from tracked positions.",
    top: "Tactical view from above",
    pov:
      review.selected === null
        ? "Click a player to see the game from their view"
        : `Seeing the game as #${review.selected} · click another player to switch`,
  }[review.camera];
}

function selectPlayer(id) {
  review.selected = id;
  if (id !== null) $("pov-player").value = String(id);
  renderPlayback();
}

function setCamera(mode) {
  review.camera = mode;
  document
    .querySelectorAll("[data-camera]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.camera === mode)),
    );
  $("pov-controls").hidden = mode !== "pov";
  if (mode === "pov" && review.selected === null) {
    // Start with whoever is nearest the ball, else the most involved player.
    const time = Number($("timeline").value);
    const ball = ballAt(model, time);
    const here = model
      .playersAt(time)
      .filter((p) => p.team === 0 || p.team === 1);
    const near = ball
      ? here.sort(
          (a, b) =>
            Math.hypot(a.x - ball.x, a.y - ball.y) -
            Math.hypot(b.x - ball.x, b.y - ball.y),
        )[0]
      : null;
    const fallback = model.keyPlayers[0]?.player;
    review.selected = near?.id ?? fallback?.id ?? here[0]?.id ?? null;
    if (review.selected !== null)
      $("pov-player").value = String(review.selected);
  }
  scene?.setMode(mode);
  renderPlayback();
}

function setLayout(name, save = true) {
  if (!LAYOUTS[name]) name = "film-map";
  review.layout = name;
  if (save) remember("layout", name);
  $("playback-grid").dataset.layout = name;
  $("pane-film").hidden = !has("film");
  $("pane-map").hidden = !has("map");
  $("pane-3d").hidden = !has("3d");
  document
    .querySelectorAll("[data-layout]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.layout === name)),
    );
  if (!has("film") && !video.paused) {
    // The film drives the clock; keep it playing even when hidden.
  }
  requestAnimationFrame(() => {
    scene?.resize();
    renderPlayback();
  });
}

function setDrawing(on) {
  telestrator.active = on;
  $("toggle-draw").setAttribute("aria-pressed", String(on));
  $("draw-bar").hidden = !on;
  $("playback-grid").classList.toggle("drawing", on);
  scene?.setDrawing(on);
  if (on) {
    video.pause();
    manualPlaying = false;
  }
  renderPlayback();
}

function seek(time) {
  const t = Math.max(0, Math.min(Number($("timeline").max), time));
  $("timeline").value = t;
  if (!playbackFallback) video.currentTime = t;
  renderPlayback();
}

function renderMoments() {
  const track = $("moments");
  track.replaceChildren();
  if (!model || !result.duration) return;
  for (const e of model.events) {
    const b = document.createElement("button");
    b.className = `moment ${e.type === "pass" ? "pass" : "turnover"}`;
    b.style.left = `${(e.t / result.duration) * 100}%`;
    b.style.setProperty(
      "--team",
      teamColor(e.type === "pass" ? e.team : e.wonBy),
    );
    b.title = `${stamp(e.t)} · ${describe(e)}`;
    b.setAttribute("aria-label", b.title);
    b.addEventListener("click", () => seek(e.t - 1));
    track.append(b);
  }
}

function describe(e) {
  return e.type === "pass"
    ? `${teamName(e.team)} pass #${e.from} → #${e.to} (${Math.round(e.length)} m)`
    : `${teamName(e.wonBy)} won the ball: #${e.to} from #${e.from}`;
}

const fmt = {
  pct: (v) => (v === null || v === undefined ? "–" : `${Math.round(v * 100)}%`),
  m: (v) => (v === null || v === undefined ? "–" : `${Math.round(v)} m`),
  kmh: (v) => `${(v * 3.6).toFixed(1)} km/h`,
  s: (v) => `${Math.round(v)} s`,
};

function watchPlayer(id, time = null) {
  chooseTab("playback");
  if (!has("3d")) setLayout("film-3d");
  review.selected = id;
  $("pov-player").value = String(id);
  setCamera("pov");
  const track = model.trackById.get(id);
  const visibleNow = model
    .playersAt(Number($("timeline").value))
    .some((p) => p.id === id);
  if (time !== null) seek(time);
  else if (!visibleNow && track?.segments.length) seek(track.segments[0].start);
}

function renderStats() {
  if (!result || !model) return;
  // Without any mapped frame there is nothing to measure: say so instead of showing zeros.
  const mapped = model.players.length > 0;
  $("stats-body").hidden = !mapped;
  if (!mapped) {
    $("stats-quality").textContent =
      "The field could not be mapped in this passage, so player positions and stats are unavailable. On the Watch tab, choose Adjust field setup to place four points, then run the analysis again.";
    $("stats-quality").classList.add("unavailable");
    return;
  }
  const [A, B] = model.teams;
  const arrow = (team) =>
    model.defends[team] === "left" ? "attacks →" : "attacks ←";
  const quality = [];
  quality.push(
    `Ball seen in ${Math.round(model.ballCoverage * 100)}% of mapped frames`,
  );
  quality.push(`Team A ${arrow(0)}, Team B ${arrow(1)}`);
  if (model.ballCoverage < 0.3)
    quality.push("Few ball detections, so passes are likely undercounted");
  $("stats-quality").textContent = `${quality.join(" · ")}.`;
  $("stats-quality").classList.toggle("unavailable", model.ballCoverage < 0.3);
  const rows = [
    ["Possession", A.possession, B.possession, fmt.pct],
    ["Passes completed", A.passes, B.passes, (v) => String(v)],
    ["Pass accuracy", A.accuracy, B.accuracy, fmt.pct],
    ["Ball lost", A.lost, B.lost, (v) => String(v)],
    [
      "Distance covered (tracked)",
      A.distance,
      B.distance,
      (v) => (v >= 1000 ? `${(v / 1000).toFixed(2)} km` : fmt.m(v)),
    ],
    ["Sprints", A.sprints, B.sprints, (v) => String(v)],
    ["Top speed", A.topSpeed, B.topSpeed, fmt.kmh],
    ["Average width", A.shape.width, B.shape.width, fmt.m],
    ["Average length (back to front)", A.shape.depth, B.shape.depth, fmt.m],
  ];
  $("team-compare").innerHTML =
    `<div class="compare-head"><span><i class="home-dot"></i>Team A</span><span><i class="away-dot"></i>Team B</span></div>` +
    rows
      .map(([label, a, b, f]) => {
        const av = Number(a) || 0,
          bv = Number(b) || 0,
          total = av + bv;
        const share = total > 0 ? (av / total) * 100 : 50;
        const empty =
          (a === null || a === undefined) && (b === null || b === undefined);
        return `<div class="compare-row${empty ? " empty" : ""}"><strong>${escape(f(a))}</strong><div><span>${escape(label)}</span><div class="bar"><i data-share="${share}"></i></div></div><strong>${escape(f(b))}</strong></div>`;
      })
      .join("");
  const statText = {
    involvement: (p) =>
      `${count(p.touches, "touch", "touches")} · ${count(p.passes, "pass", "passes")} · ${p.received} received`,
    passes: (p) => `${count(p.passes, "completed pass", "completed passes")}`,
    workRate: (p) => `${Math.round(p.workRate)} m per minute on camera`,
    topSpeed: (p) => `${fmt.kmh(p.topSpeed)} top speed`,
  };
  $("key-players").innerHTML = model.keyPlayers.length
    ? model.keyPlayers
        .map(
          (k) =>
            `<article class="key-card" data-team-color="${k.player.team}"><span class="key-role">${escape(k.role)}</span><strong>${escape(playerName(k.player))}</strong><span class="key-team">${teamName(k.player.team)}</span><p>${escape(statText[k.stat](k.player))}</p><button data-watch="${k.player.id}">Watch their view</button></article>`,
        )
        .join("")
    : '<p class="help">Not enough tracked play in this passage to pick key players.</p>';
  renderNetwork();
  const list = $("moment-list");
  list.innerHTML = model.events.length
    ? model.events
        .map(
          (e, i) =>
            `<li><button data-moment="${i}"><time>${stamp(e.t)}</time><i data-team-fill="${e.type === "pass" ? e.team : e.wonBy}"></i>${escape(describe(e))}</button></li>`,
        )
        .join("")
    : '<li class="help">No passes or turnovers were detected. The ball may be missing from too many frames.</li>';
  const players = [...model.players]
    .filter((p) => p.seen >= 1)
    .sort((a, b) => b.involvement - a.involvement || b.seen - a.seen);
  $("player-table").innerHTML =
    "<thead><tr><th>Player</th><th>Team</th><th>On camera</th><th>Distance</th><th>Top speed</th><th>Sprints</th><th>Touches</th><th>Passes</th><th>Received</th><th></th></tr></thead><tbody>" +
    players
      .map(
        (p) =>
          `<tr><td><strong>${escape(playerName(p))}</strong></td><td><i class="team-dot" data-team-fill="${p.team}"></i>${teamName(p.team)}</td><td>${fmt.s(p.seen)}</td><td>${fmt.m(p.distance)}</td><td>${fmt.kmh(p.topSpeed)}</td><td>${p.sprints}</td><td>${p.touches}</td><td>${p.passes}</td><td>${p.received}</td><td><button data-watch="${p.id}">Watch</button></td></tr>`,
      )
      .join("") +
    "</tbody>";
  // The CSP blocks inline style attributes, so colours and widths are applied via CSSOM.
  const stats = $("tab-stats");
  stats.querySelectorAll("[data-share]").forEach((el) => {
    el.style.width = `${el.dataset.share}%`;
  });
  stats.querySelectorAll("[data-team-color]").forEach((el) => {
    el.style.setProperty("--team", teamColor(Number(el.dataset.teamColor)));
  });
  stats.querySelectorAll("[data-team-fill]").forEach((el) => {
    el.style.background = teamColor(Number(el.dataset.teamFill));
  });
  document
    .querySelectorAll("#tab-stats [data-watch]")
    .forEach((b) =>
      b.addEventListener("click", () => watchPlayer(Number(b.dataset.watch))),
    );
  list.querySelectorAll("[data-moment]").forEach((b) =>
    b.addEventListener("click", () => {
      const e = model.events[Number(b.dataset.moment)];
      watchPlayer(e.from, Math.max(0, e.t - 1.5));
      setCamera("orbit");
    }),
  );
}

function renderNetwork() {
  const team = Number(
    document.querySelector("#network-team [aria-pressed='true']").dataset.team,
  );
  const { ctx, point } = field($("network"));
  const net = passNetwork(model, team);
  const color = teamColor(team);
  const byId = new Map(net.nodes.map((n) => [n.id, n]));
  const most = Math.max(1, ...net.links.map((l) => l.count));
  for (const link of net.links) {
    const a = byId.get(link.a),
      b = byId.get(link.b);
    if (!a || !b) continue;
    ctx.beginPath();
    ctx.moveTo(...point(a.x, a.y));
    ctx.lineTo(...point(b.x, b.y));
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 1.5 + (5 * link.count) / most;
    ctx.stroke();
  }
  const top = Math.max(1, ...net.nodes.map((n) => n.weight));
  for (const n of net.nodes) {
    const [x, y] = point(n.x, n.y);
    ctx.beginPath();
    ctx.arc(x, y, 9 + (11 * n.weight) / top, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = "#fff9df";
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = "white";
    ctx.font = "bold 12px Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(n.id, x, y);
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
  }
  const passes = net.links.reduce((s, l) => s + l.count, 0);
  $("network-caption").textContent = net.nodes.length
    ? `${teamName(team)} ${model.defends[team] === "left" ? "attacks to the right" : "attacks to the left"}. Circles sit at each player's average tracked position and grow with involvement; lines thicken with more passes (${passes} completed).`
    : `${teamName(team)} has no detected passes in this passage.`;
}
// While the film plays, redraw once per display frame from its clock; nothing runs when paused.
function playbackLoop() {
  if (!playing || !result) return;
  $("timeline").value = Math.min(video.currentTime, Number($("timeline").max));
  renderPlayback();
  requestAnimationFrame(playbackLoop);
}
video.addEventListener("timeupdate", () => {
  if (!result || playing) return;
  $("timeline").value = Math.min(video.currentTime, Number($("timeline").max));
  renderPlayback();
});
video.addEventListener("play", () => {
  playing = true;
  requestAnimationFrame(playbackLoop);
});
video.addEventListener("pause", () => {
  playing = false;
  renderPlayback();
});
video.addEventListener("seeked", () => {
  if (!playing) renderPlayback();
});
video.addEventListener("error", () => {
  playbackFallback = true;
  $("video-error").hidden = false;
});
on("play", "click", async () => {
  if (playbackFallback) {
    manualPlaying = !manualPlaying;
    if (manualPlaying) {
      const end = Number($("timeline").max);
      const start = Number($("timeline").value);
      manualClock.start(start >= end ? 0 : start, performance.now(), end);
      requestAnimationFrame(tick);
    }
    renderPlayback();
    return;
  }
  if (video.paused) {
    if (video.ended) video.currentTime = 0;
    await video.play();
  } else video.pause();
});
function tick(now) {
  if (!manualPlaying || !result) return;
  const time = manualClock.at(now);
  $("timeline").value = time;
  if (time >= Number($("timeline").max)) manualPlaying = false;
  renderPlayback();
  if (manualPlaying) requestAnimationFrame(tick);
}
on("timeline", "input", () => {
  if (!playbackFallback) video.currentTime = Number($("timeline").value);
  else
    manualClock.start(
      Number($("timeline").value),
      performance.now(),
      Number($("timeline").max),
    );
  renderPlayback();
});
on("jump-gap", "click", () => {
  const frame = result.frames.find((f) => !f.valid);
  if (frame) {
    video.pause();
    manualPlaying = false;
    $("timeline").value = frame.time;
    if (!playbackFallback) video.currentTime = frame.time;
    renderPlayback();
  }
});
["show-ids", "show-ball", "map-team"].forEach((id) =>
  on(id, "change", renderPlayback),
);
["home-color", "away-color"].forEach((id) =>
  on(id, "input", () => {
    document.documentElement.style.setProperty(
      id === "home-color" ? "--home" : "--away",
      $(id).value,
    );
    if (result) {
      scene?.setColors($("home-color").value, $("away-color").value);
      renderMoments();
      renderPlayback();
      renderShape();
      if (!$("tab-stats").hidden) renderStats();
    }
  }),
);

document
  .querySelectorAll("[data-layout]")
  .forEach((b) =>
    b.addEventListener("click", () => setLayout(b.dataset.layout)),
  );
document
  .querySelectorAll("[data-camera]")
  .forEach((b) =>
    b.addEventListener("click", () => setCamera(b.dataset.camera)),
  );
document.querySelectorAll("[data-pov]").forEach((b) =>
  b.addEventListener("click", () => {
    document
      .querySelectorAll("[data-pov]")
      .forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    scene?.setPovStyle(b.dataset.pov);
  }),
);
on("pov-player", "change", () => selectPlayer(Number($("pov-player").value)));
on("toggle-coverage", "click", () => {
  review.coverage = !review.coverage;
  $("toggle-coverage").setAttribute("aria-pressed", String(review.coverage));
  $("coverage-legend").hidden = !review.coverage;
  if (review.coverage && review.layout === "film") setLayout("film-map");
  renderPlayback();
});
on("coverage-team", "change", renderPlayback);
on("toggle-draw", "click", () => setDrawing(!telestrator.active));
on("draw-done", "click", () => setDrawing(false));
on("draw-undo", "click", () => telestrator.undo());
on("draw-clear", "click", () => telestrator.clear());
document.querySelectorAll("[data-tool]").forEach((b) =>
  b.addEventListener("click", () => {
    telestrator.tool = b.dataset.tool;
    document
      .querySelectorAll("[data-tool]")
      .forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
  }),
);
document.querySelectorAll("[data-color]").forEach((b) => {
  b.style.setProperty("--swatch", b.dataset.color);
  b.addEventListener("click", () => {
    telestrator.color = b.dataset.color;
    document
      .querySelectorAll("[data-color]")
      .forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
  });
});
telestrator.attach($("film-marks"), (event) =>
  result ? locateOnFilm(event) : null,
);
telestrator.attach($("pitch"), (event) =>
  mapView
    ? { space: "pitch", point: mapView.toPitch(event.clientX, event.clientY) }
    : null,
);
$("pitch").addEventListener("click", (event) => {
  if (telestrator.active || !mapView || !model) return;
  const [x, y] = mapView.toPitch(event.clientX, event.clientY);
  const time = Number($("timeline").value);
  const nearest = model
    .playersAt(time)
    .map((p) => ({ p, d: Math.hypot(p.x - x, p.y - y) }))
    .sort((a, b) => a.d - b.d)[0];
  selectPlayer(nearest && nearest.d < 2.5 ? nearest.p.id : null);
});
on("expand-review", "click", async () => {
  const panel = document.querySelector(".review-panel");
  if (document.fullscreenElement) await document.exitFullscreen();
  else await panel.requestFullscreen();
});
document.addEventListener("fullscreenchange", () => {
  const full = Boolean(document.fullscreenElement);
  $("expand-review").setAttribute("aria-pressed", String(full));
  $("expand-review").querySelector("span").textContent = full
    ? "Exit full screen"
    : "Expand";
  requestAnimationFrame(() => {
    scene?.resize();
    renderPlayback();
  });
});
window.addEventListener("resize", () => {
  if (result && location.hash === "#review")
    requestAnimationFrame(renderPlayback);
});
document.querySelectorAll("#network-team [data-team]").forEach((b) =>
  b.addEventListener("click", () => {
    document
      .querySelectorAll("#network-team [data-team]")
      .forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    renderNetwork();
  }),
);
on("flip-sides", "click", () => {
  review.flip = !review.flip;
  const selected = review.selected;
  model = buildModel(result, { flip: review.flip });
  coverageFor.key = null;
  review.selected = selected;
  renderMoments();
  renderStats();
  renderPlayback();
});
// Keyboard: Space plays, arrows step one sample, D draws, F fills the screen, Esc stops drawing.
document.addEventListener("keydown", (event) => {
  if (location.hash !== "#review" || $("tab-playback").hidden || !result)
    return;
  if (
    event.target.closest("input, select, textarea") ||
    event.metaKey ||
    event.ctrlKey
  )
    return;
  // Focused buttons keep their own Space/arrow behaviour.
  if (
    event.target.closest("button") &&
    [" ", "ArrowLeft", "ArrowRight"].includes(event.key)
  )
    return;
  const step = result.settings.stride / result.source.fps;
  if (event.key === " ") {
    event.preventDefault();
    $("play").click();
  } else if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
    event.preventDefault();
    video.pause();
    seek(
      Number($("timeline").value) + (event.key === "ArrowRight" ? step : -step),
    );
  } else if (event.key === "d" || event.key === "D")
    setDrawing(!telestrator.active);
  else if (event.key === "f" || event.key === "F") $("expand-review").click();
  else if (event.key === "Escape" && telestrator.active) setDrawing(false);
});

function makeTrackChoices() {
  const team = Number($("shape-team").value);
  const ids = [
    ...new Set(
      result.frames.flatMap((f) =>
        f.players
          .filter((p) => p.team === team && p.kind === "player")
          .map((p) => p.id),
      ),
    ),
  ].sort((a, b) => a - b);
  $("track-choices").innerHTML =
    '<p class="help">Choose the defenders for back-line height, or select one or more tracks for the heatmap. IDs are tracks, not player identities.</p>' +
    ids
      .map(
        (id) =>
          `<label><input type="checkbox" value="${id}">Track ${id}</label>`,
      )
      .join("");
  $("track-choices")
    .querySelectorAll("input")
    .forEach((el) => el.addEventListener("change", renderShape));
}
function renderShape() {
  if (!result) return;
  const team = Number($("shape-team").value),
    selected = [...$("track-choices").querySelectorAll("input:checked")].map(
      (el) => Number(el.value),
    );
  const summary = summarize(result, {
    start: Number($("interval-start").value),
    end: Number($("interval-end").value),
    team,
    selected,
    direction: $("direction").value,
  });
  const { ctx, point, sx, sy } = field($("shape-pitch"));
  if (summary.error) {
    $("shape-summary").textContent = summary.error;
    $("shape-caption").textContent =
      "Choose a valid time window to calculate observations.";
    return;
  }
  const heat = $("shape-mode").value === "heatmap";
  if (heat && selected.length) {
    const bins = new Map();
    for (const p of summary.samples) {
      const key = `${Math.min(19, Math.floor((p.x / result.settings.length) * 20))},${Math.min(13, Math.floor((p.y / result.settings.width) * 14))}`;
      bins.set(key, (bins.get(key) || 0) + 1);
    }
    const max = Math.max(1, ...bins.values());
    for (const [key, count] of bins) {
      const [x, y] = key.split(",").map(Number);
      ctx.fillStyle = `rgba(242,202,88,${0.15 + (0.7 * count) / max})`;
      ctx.fillRect(
        ...point(
          (x * result.settings.length) / 20,
          (y * result.settings.width) / 14,
        ),
        (result.settings.length / 20) * sx,
        (result.settings.width / 14) * sy,
      );
    }
  } else if (!heat) {
    for (const player of summary.averages)
      dot(ctx, point, player, true, selected.includes(player.id));
  }
  $("shape-caption").textContent = heat
    ? selected.length
      ? `${summary.samples.length} valid observed samples · brighter cells contain more samples · not occupancy outside the camera view.`
      : "Select a track above to show its observed-sample density."
    : `Average positions over valid observations · attacking ${$("direction").value}.`;
  $("shape-summary").innerHTML =
    `<p><strong>${summary.averages.length}</strong>Visible tracks in this window</p><p><strong>${Math.round(summary.coverage * 100)}%</strong>Valid mapping coverage · ${summary.valid}/${summary.frames} sampled frames</p><p><strong>${summary.height === null ? "Unavailable" : summary.height.toFixed(1) + " m"}</strong>Selected defenders’ mean distance from the defended goal line<br><small>${selected.length ? `${summary.heightFrames} frames with all ${selected.length} selected tracks visible together` : "Select the defenders above to calculate back-line height."}</small></p>`;
}
on("shape-team", "change", () => {
  makeTrackChoices();
  renderShape();
});
["shape-mode", "interval-start", "interval-end", "direction"].forEach((id) =>
  on(id, "input", renderShape),
);
on("download-settings", "click", async () => {
  const data = await api(`/api/runs/${job.id}/settings`);
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = result.synthetic
    ? "soccermap-synthetic-run.json"
    : "soccermap-run.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
on("save-run", "click", async () => {
  job = await api(`/api/runs/${job.id}/save`, {});
  $("save-run").textContent = "Saved on this computer";
  $("save-run").disabled = true;
});

async function loadSaved() {
  const runs = await api("/api/runs");
  $("saved-list").innerHTML = runs.length
    ? runs
        .map(
          (run) =>
            `<article class="saved-row"><div><h2>${escape(run.name)}</h2><p>${escape(new Date(run.created * 1000).toLocaleString())} · ${escape(run.state)}${run.saved ? " · Saved" : ""} · ${run.synthetic ? "Synthetic demo" : "Local clip"}</p></div><button data-open="${run.id}">${runAction(run)}</button></article>`,
        )
        .join("")
    : '<div class="empty panel"><h2>No analyses yet</h2><p>Completed and stopped runs will appear here, so you can return to them later.</p><a href="#import" class="button primary">Analyze a clip</a></div>';
  $("saved-list")
    .querySelectorAll("[data-open]")
    .forEach((button) =>
      button.addEventListener(
        "click",
        handle(async () => {
          const selected = runs.find((run) => run.id === button.dataset.open);
          if (selected.state === "complete") return openResult(selected.id);
          job = await api(`/api/runs/${selected.id}`);
          sessionStorage.setItem("soccermap-run", job.id);
          upload = null;
          route("processing");
          renderProgress();
          if (job.state === "running") schedulePoll();
        }),
      ),
    );
}
async function loadEnvironment() {
  const env = await api("/api/environment");
  $("environment-list").innerHTML =
    env.checks
      .map(
        (check) =>
          `<div class="environment-row"><strong>${escape(check.name)}<br><small>${check.required ? "Required" : "Optional"}</small></strong><span class="${check.ready ? "ready" : "missing"}">${check.ready ? "File found" : "Missing file"}</span><code>${escape(check.path)}</code></div>`,
      )
      .join("") + `<p class="help">${escape(env.note)}</p>`;
}
on("refresh-environment", "click", loadEnvironment);

async function boot() {
  const requested = location.hash.slice(1);
  await show(location.hash.slice(1) || "import");
  const id = sessionStorage.getItem("soccermap-run");
  if (id) {
    try {
      job = await api(`/api/runs/${id}`);
      if (job.state === "running") {
        route("processing");
        renderProgress();
        schedulePoll();
      } else if (job.resumable && requested === "processing") {
        route("processing");
        renderProgress();
      } else if (
        job.state === "complete" &&
        ["review", "processing"].includes(location.hash.slice(1))
      )
        await openResult(id);
    } catch {
      sessionStorage.removeItem("soccermap-run");
    }
  }
}
boot().catch((error) => notice(error.message));
