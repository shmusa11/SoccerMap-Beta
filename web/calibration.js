/** Four-point setup editor. All image coordinates refer to the original decoded frame. */
const $ = (id) => document.getElementById(id);

export function imagePoint(clientX, clientY, rect, width, height) {
  return {
    x: Math.max(
      0,
      Math.min(width - 1, ((clientX - rect.left) * width) / rect.width),
    ),
    y: Math.max(
      0,
      Math.min(height - 1, ((clientY - rect.top) * height) / rect.height),
    ),
  };
}

export class CalibrationEditor {
  constructor({
    api,
    currentUpload,
    dimensions,
    defaultFrame = () => 0,
    report,
    changed,
  }) {
    Object.assign(this, {
      api,
      currentUpload,
      dimensions,
      defaultFrame,
      report,
      changed,
    });
    this.report = (message) => {
      if ($("camera-editor").hidden) return report(message);
      $("setup-error").textContent = message;
      $("setup-error").hidden = !message;
      if (message) $("setup-error").scrollIntoView({ block: "nearest" });
    };
    this.points = [null, null, null, null];
    this.active = 0;
    this.frame = 0;
    this.revision = 0;
    this.cameras = [];
    this.canvas = $("setup-image");
    const listen = (id, type, fn) =>
      $(id).addEventListener(type, async (event) => {
        try {
          this.report("");
          await fn(event);
        } catch (error) {
          this.report(error.message);
        }
      });
    listen("open-camera-editor", "click", () => this.open());
    listen("close-camera-editor", "click", () => {
      $("camera-editor").hidden = true;
    });
    listen("load-camera-frame", "click", () => this.loadFrame());
    listen("setup-marking", "change", () => {
      this.selected[this.active] = $("setup-marking").value;
      this.invalidate();
      this.render();
    });
    for (const axis of ["x", "y"])
      listen(`point-${axis}`, "input", () => {
        const value = $(`point-${axis}`).valueAsNumber;
        if (!Number.isFinite(value) || !this.points[this.active]) return;
        this.points[this.active][axis] = Math.max(
          0,
          Math.min(
            (axis === "x" ? this.canvas.width : this.canvas.height) - 1,
            value,
          ),
        );
        this.invalidate();
        this.render(false);
      });
    listen("reset-points", "click", () => {
      this.points = [null, null, null, null];
      this.active = 0;
      this.invalidate();
      this.render();
    });
    listen("preview-camera", "click", () => this.preview());
    listen("save-camera", "click", () => this.save());
    listen("setup-reviewed", "change", () => {
      $("save-camera").disabled =
        !this.previewImage || !$("setup-reviewed").checked;
    });
    listen("setup-zoom", "input", () => {
      this.canvas.style.width = `${Number($("setup-zoom").value) * 100}%`;
    });
    this.canvas.addEventListener("pointerdown", (event) => {
      if (!this.baseImage) return;
      const rect = this.canvas.getBoundingClientRect();
      const point = imagePoint(
        event.clientX,
        event.clientY,
        rect,
        this.canvas.width,
        this.canvas.height,
      );
      const near = this.points.findIndex(
        (p) =>
          p &&
          Math.hypot(p.x - point.x, p.y - point.y) <
            (20 * this.canvas.width) / rect.width,
      );
      if (near >= 0) this.active = near;
      this.dragging = true;
      this.canvas.setPointerCapture(event.pointerId);
      this.place(point);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (this.dragging)
        this.place(
          imagePoint(
            event.clientX,
            event.clientY,
            this.canvas.getBoundingClientRect(),
            this.canvas.width,
            this.canvas.height,
          ),
        );
    });
    const release = () => {
      this.dragging = false;
    };
    this.canvas.addEventListener("pointerup", release);
    this.canvas.addEventListener("pointercancel", release);
    this.canvas.addEventListener("keydown", (event) => {
      const delta = {
        ArrowLeft: [-1, 0],
        ArrowRight: [1, 0],
        ArrowUp: [0, -1],
        ArrowDown: [0, 1],
      }[event.key];
      if (!delta && event.key !== "Enter") return;
      event.preventDefault();
      if (!this.baseImage) return;
      const p = this.points[this.active] || {
        x: this.canvas.width / 2,
        y: this.canvas.height / 2,
      };
      const step = event.shiftKey ? 10 : 1;
      this.place({
        x: Math.max(
          0,
          Math.min(this.canvas.width - 1, p.x + (delta?.[0] || 0) * step),
        ),
        y: Math.max(
          0,
          Math.min(this.canvas.height - 1, p.y + (delta?.[1] || 0) * step),
        ),
      });
    });
    $("camera-setup").addEventListener("change", () => this.changed());
    // Canvas pixels do not respond to CSS layout. Redraw only the image layer when its
    // displayed size changes, preserving readable handles through zoom and rotation.
    this.resizeObserver = new ResizeObserver(() => this.drawImage());
    this.resizeObserver.observe(this.canvas);
  }

  async refresh() {
    this.cameras = await this.api("/api/cameras");
    const select = $("camera-setup"),
      previous = select.value;
    select.replaceChildren(new Option("Automatic field detection", ""));
    for (const camera of this.cameras)
      select.add(
        new Option(
          `${camera.name} · ${camera.length} × ${camera.width} m`,
          camera.id,
        ),
      );
    if (this.cameras.some((c) => c.id === previous)) select.value = previous;
    const demo = this.currentUpload()?.id === "demo";
    select.disabled = demo;
    $("open-camera-editor").disabled = demo;
    if (demo) select.value = "";
    $("camera-editor").hidden = true;
    this.changed();
  }

  setup() {
    return this.cameras.find((c) => c.id === $("camera-setup").value);
  }

  async open() {
    const upload = this.currentUpload();
    if (!upload || upload.id === "demo") return;
    this.landmarks = await this.api(
      `/api/landmarks?length=${this.dimensions().length}&width=${this.dimensions().width}`,
    );
    this.selected = [
      "left_penalty_top_goal_line",
      "left_penalty_top_inner",
      "left_penalty_bottom_inner",
      "left_penalty_bottom_goal_line",
    ];
    $("setup-marking").replaceChildren(
      ...this.landmarks.map((p) => new Option(p.label, p.id)),
    );
    this.points = [null, null, null, null];
    this.active = 0;
    $("setup-frame").max = Math.max(0, upload.frames - 1);
    $("setup-frame").value = this.defaultFrame();
    $("setup-name").value =
      upload.name.replace(/\.[^.]+$/, "").slice(0, 65) + " camera";
    $("camera-editor").hidden = false;
    await this.loadFrame();
    $("camera-editor").scrollIntoView({ behavior: "smooth", block: "start" });
    this.canvas.focus({ preventScroll: true });
  }

  async loadFrame() {
    const index = Number($("setup-frame").value);
    const uploadId = this.currentUpload().id;
    const request = ++this.revision;
    $("load-camera-frame").disabled = true;
    try {
      const data = await this.api(
        `/api/uploads/${uploadId}/frame?index=${index}`,
      );
      const image = await this.image(data.image);
      if (request !== this.revision || uploadId !== this.currentUpload().id)
        return;
      this.baseImage = image;
      this.frame = data.index;
      this.canvas.width = data.width;
      this.canvas.height = data.height;
      this.points = [null, null, null, null];
      this.active = 0;
      this.invalidate();
      this.render();
    } finally {
      $("load-camera-frame").disabled = false;
    }
  }

  image(src) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () =>
        reject(
          Error(
            "The frame preview could not load. Try loading the frame again.",
          ),
        );
      image.src = src;
    });
  }

  place(point) {
    this.points[this.active] = point;
    this.invalidate();
    this.render();
  }
  invalidate() {
    this.revision++;
    this.previewImage = null;
    $("setup-reviewed").checked = false;
    $("setup-review").hidden = true;
    $("save-camera").disabled = true;
  }

  payload() {
    return {
      upload: this.currentUpload().id,
      frame: this.frame,
      ...this.dimensions(),
      name: $("setup-name").value,
      points: this.points.map((p, i) =>
        p ? { landmark: this.selected[i], x: p.x, y: p.y } : null,
      ),
    };
  }

  async preview() {
    const revision = this.revision;
    $("preview-camera").disabled = true;
    try {
      const data = await this.api("/api/cameras/preview", this.payload());
      const image = await this.image(data.image);
      if (revision !== this.revision) return;
      this.previewImage = image;
      $("setup-review").hidden = false;
      this.render();
      $("setup-reviewed").focus({ preventScroll: true });
    } finally {
      $("preview-camera").disabled = this.points.some((p) => !p);
    }
  }

  async save() {
    if (!this.previewImage || !$("setup-reviewed").checked) return;
    $("save-camera").disabled = true;
    try {
      const setup = await this.api("/api/cameras", {
        ...this.payload(),
        reviewed: true,
      });
      await this.refresh();
      $("camera-setup").value = setup.id;
      this.changed();
      $("camera-setup").focus();
    } finally {
      $("save-camera").disabled = false;
    }
  }

  drawImage() {
    if (!this.baseImage) return;
    const displayedWidth = this.canvas.getBoundingClientRect().width;
    if (!displayedWidth) return;
    const ctx = this.canvas.getContext("2d");
    ctx.drawImage(this.previewImage || this.baseImage, 0, 0);
    const scale = this.canvas.width / displayedWidth;
    this.points.forEach((point, index) => {
      if (!point) return;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 12 * scale, 0, Math.PI * 2);
      ctx.fillStyle = index === this.active ? "#fff" : "#173e2d";
      ctx.fill();
      ctx.lineWidth = 2 * scale;
      ctx.strokeStyle = index === this.active ? "#173e2d" : "#fff";
      ctx.stroke();
      ctx.fillStyle = index === this.active ? "#173e2d" : "#fff";
      ctx.font = `600 ${13 * scale}px system-ui`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(index + 1), point.x, point.y);
    });
  }

  render(updateInputs = true) {
    if (!this.baseImage) return;
    this.drawImage();
    const tabs = $("setup-points"),
      restoreFocus = tabs.contains(document.activeElement);
    tabs.replaceChildren();
    for (let i = 0; i < 4; i++) {
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `Point ${i + 1}${this.points[i] ? " · placed" : ""}`;
      button.setAttribute("aria-pressed", String(i === this.active));
      button.addEventListener("click", () => {
        this.active = i;
        this.render();
      });
      tabs.append(button);
    }
    if (restoreFocus) tabs.children[this.active].focus({ preventScroll: true });
    $("setup-marking").value = this.selected[this.active];
    if (updateInputs)
      for (const axis of ["x", "y"])
        $(`point-${axis}`).value = this.points[this.active]
          ? Math.round(this.points[this.active][axis])
          : "";
    for (const axis of ["x", "y"])
      $(`point-${axis}`).disabled = !this.points[this.active];
    $("point-x").max = this.canvas.width - 1;
    $("point-y").max = this.canvas.height - 1;
    $("preview-camera").disabled = this.points.some((p) => !p);
    $("setup-count").textContent =
      `${this.points.filter(Boolean).length} of 4 points placed · source frame ${this.frame}`;
    this.drawField();
  }

  drawField() {
    const canvas = $("setup-field"),
      ctx = canvas.getContext("2d");
    const { length, width } = this.dimensions();
    const sx = (canvas.width - 40) / length,
      sy = (canvas.height - 40) / width;
    ctx.fillStyle = "#e7eee9";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#54735f";
    ctx.lineWidth = 1.5;
    const rect = (x, y, w, h) =>
      ctx.strokeRect(20 + x * sx, 20 + y * sy, w * sx, h * sy);
    rect(0, 0, length, width);
    ctx.beginPath();
    ctx.moveTo(20 + (length / 2) * sx, 20);
    ctx.lineTo(20 + (length / 2) * sx, canvas.height - 20);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(
      20 + (length / 2) * sx,
      20 + (width / 2) * sy,
      9.15 * sx,
      9.15 * sy,
      0,
      0,
      2 * Math.PI,
    );
    ctx.stroke();
    for (const depth of [16.5, 5.5]) {
      const boxWidth = depth === 16.5 ? 40.32 : 18.32;
      rect(0, (width - boxWidth) / 2, depth, boxWidth);
      rect(length - depth, (width - boxWidth) / 2, depth, boxWidth);
    }
    this.selected.forEach((id, i) => {
      const point = this.landmarks.find((p) => p.id === id);
      if (!point) return;
      ctx.beginPath();
      ctx.arc(
        20 + point.x * sx,
        20 + point.y * sy,
        i === this.active ? 11 : 9,
        0,
        Math.PI * 2,
      );
      ctx.fillStyle = i === this.active ? "#173e2d" : "#65766c";
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "600 12px system-ui";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), 20 + point.x * sx, 20 + point.y * sy);
    });
  }
}
