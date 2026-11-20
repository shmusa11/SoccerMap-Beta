/**
 * Lightweight 3D replay of tracked positions. Everything is procedural (no model files):
 * a striped pitch texture, goals, and simple jointed players whose run cycle follows their
 * tracked speed. It renders on demand, so a paused review costs no GPU time.
 *
 * Viewer coordinates: X along the length, Y up, Z along the width, origin at the centre spot.
 * Pitch (x, y) metres map to (x - length/2, 0, y - width/2), matching the 2D map orientation.
 */
import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const SKIN = "#d9a77f";
const KIT = {
  referee: { jersey: "#262a2c", shorts: "#151718", socks: "#151718" },
  unknown: { jersey: "#b8bdb0", shorts: "#6f7469", socks: "#b8bdb0" },
};
const KEEPER = { 0: "#f2c14e", 1: "#3cc49a" };
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));

function shade(hex, factor) {
  const c = new THREE.Color(hex);
  c.multiplyScalar(factor);
  return c;
}

/** Striped turf with painted markings, drawn once into a mip-mapped canvas texture. */
function pitchTexture(L, W, margin, anisotropy) {
  const width = 2048,
    ppm = width / (L + 2 * margin),
    height = Math.round((W + 2 * margin) * ppm);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const stripes = 18,
    stripe = L / stripes;
  for (let i = -2; i < stripes + 2; i++) {
    ctx.fillStyle = i % 2 ? "#3b7c37" : "#438a3d";
    ctx.fillRect((margin + i * stripe) * ppm, 0, stripe * ppm + 1, height);
  }
  // Fine grain so the turf does not read as flat plastic up close.
  let seed = 7;
  const random = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  for (let i = 0; i < 26000; i++) {
    ctx.fillStyle =
      random() < 0.5 ? "rgba(0,0,0,0.05)" : "rgba(255,255,255,0.035)";
    ctx.fillRect(random() * width, random() * height, 2, 2);
  }
  const p = (x, y) => [(x + margin) * ppm, (y + margin) * ppm];
  ctx.strokeStyle = "rgba(250,252,246,0.93)";
  ctx.fillStyle = "rgba(250,252,246,0.93)";
  ctx.lineWidth = 0.13 * ppm;
  const rect = (x, y, w, h) => ctx.strokeRect(...p(x, y), w * ppm, h * ppm);
  const arc = (x, y, r, a0, a1) => {
    ctx.beginPath();
    ctx.arc(...p(x, y), r * ppm, a0, a1);
    ctx.stroke();
  };
  const spot = (x, y) => {
    ctx.beginPath();
    ctx.arc(...p(x, y), 0.16 * ppm, 0, Math.PI * 2);
    ctx.fill();
  };
  rect(0, 0, L, W);
  ctx.beginPath();
  ctx.moveTo(...p(L / 2, 0));
  ctx.lineTo(...p(L / 2, W));
  ctx.stroke();
  arc(L / 2, W / 2, 9.15, 0, Math.PI * 2);
  spot(L / 2, W / 2);
  for (const side of [0, 1]) {
    const gx = side ? L : 0,
      dir = side ? -1 : 1;
    rect(side ? L - 16.5 : 0, (W - 40.32) / 2, 16.5, 40.32);
    rect(side ? L - 5.5 : 0, (W - 18.32) / 2, 5.5, 18.32);
    spot(gx + dir * 11, W / 2);
    // Penalty arc: the part of the 9.15 m circle outside the box.
    const half = Math.acos(5.5 / 9.15);
    arc(
      gx + dir * 11,
      W / 2,
      9.15,
      side ? Math.PI - half : -half,
      side ? Math.PI + half : half,
    );
    for (const y of [0, W])
      arc(
        gx,
        y,
        1,
        side ? (y ? Math.PI : Math.PI / 2) : y ? -Math.PI / 2 : 0,
        side ? (y ? 1.5 * Math.PI : Math.PI) : y ? 0 : Math.PI / 2,
      );
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = anisotropy;
  return texture;
}

function netTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, 64, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

function ballTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 128;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#f7f7f2";
  ctx.fillRect(0, 0, 256, 128);
  ctx.fillStyle = "#23262a";
  for (const [x, y] of [
    [32, 64],
    [96, 30],
    [96, 98],
    [160, 64],
    [224, 30],
    [224, 98],
    [128, 4],
    [128, 124],
  ]) {
    ctx.beginPath();
    for (let k = 0; k < 5; k++) {
      const a = (k / 5) * Math.PI * 2 - Math.PI / 2;
      ctx.lineTo(x + Math.cos(a) * 13, y + Math.sin(a) * 13);
    }
    ctx.fill();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function skyTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 2;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, 0, 256);
  g.addColorStop(0, "#6f9fcf");
  g.addColorStop(0.55, "#b9d3e6");
  g.addColorStop(1, "#dfe9e4");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 2, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function labelTexture(text) {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(18,30,24,0.78)";
  ctx.beginPath();
  ctx.roundRect(14, 8, 100, 48, 22);
  ctx.fill();
  ctx.fillStyle = "#ffffff";
  ctx.font = "bold 32px -apple-system, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, 64, 33);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/** Shared geometry for every player figure (built once). */
function playerGeometry() {
  return {
    torso: new THREE.CapsuleGeometry(0.17, 0.36, 4, 12),
    shorts: new THREE.CylinderGeometry(0.17, 0.2, 0.24, 12),
    head: new THREE.SphereGeometry(0.115, 16, 12),
    hair: new THREE.SphereGeometry(0.12, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2),
    neck: new THREE.CylinderGeometry(0.05, 0.06, 0.1, 8),
    thigh: new THREE.CapsuleGeometry(0.075, 0.3, 4, 8),
    shin: new THREE.CapsuleGeometry(0.062, 0.3, 4, 8),
    foot: new THREE.BoxGeometry(0.1, 0.07, 0.25),
    upper: new THREE.CapsuleGeometry(0.058, 0.2, 4, 8),
    fore: new THREE.CapsuleGeometry(0.05, 0.2, 4, 8),
    ring: new THREE.RingGeometry(0.55, 0.78, 40),
    shadowDisc: new THREE.CircleGeometry(0.45, 20),
  };
}

class Figure {
  constructor(geo, id) {
    this.id = id;
    this.phase = (id * 1.37) % (Math.PI * 2);
    this.amp = 0;
    this.heading = 0;
    this.mats = {
      jersey: new THREE.MeshStandardMaterial({ roughness: 0.75 }),
      shorts: new THREE.MeshStandardMaterial({ roughness: 0.8 }),
      socks: new THREE.MeshStandardMaterial({ roughness: 0.8 }),
      skin: new THREE.MeshStandardMaterial({ color: SKIN, roughness: 0.65 }),
      boot: new THREE.MeshStandardMaterial({
        color: "#1b1d1f",
        roughness: 0.5,
      }),
      hair: new THREE.MeshStandardMaterial({
        color: "#2b2118",
        roughness: 0.9,
      }),
    };
    const mesh = (g, m, x = 0, y = 0, z = 0) => {
      const o = new THREE.Mesh(g, m);
      o.position.set(x, y, z);
      o.castShadow = true;
      o.userData.id = id;
      return o;
    };
    this.root = new THREE.Group();
    this.body = new THREE.Group();
    this.root.add(this.body);
    this.upper = new THREE.Group();
    this.upper.position.y = 0.95;
    this.body.add(this.upper);
    const torso = mesh(geo.torso, this.mats.jersey, 0, 0.3, 0);
    torso.scale.set(1, 1, 0.72);
    this.upper.add(torso, mesh(geo.neck, this.mats.skin, 0, 0.6, 0));
    this.head = new THREE.Group();
    this.head.position.y = 0.72;
    this.head.add(
      mesh(geo.head, this.mats.skin),
      mesh(geo.hair, this.mats.hair, 0, 0.02, -0.01),
    );
    this.upper.add(this.head);
    this.body.add(mesh(geo.shorts, this.mats.shorts, 0, 0.9, 0));
    this.legs = [-1, 1].map((side) => {
      const hip = new THREE.Group();
      hip.position.set(0.095 * side, 0.9, 0);
      hip.add(mesh(geo.thigh, this.mats.skin, 0, -0.2, 0));
      const knee = new THREE.Group();
      knee.position.y = -0.42;
      knee.add(
        mesh(geo.shin, this.mats.socks, 0, -0.2, 0),
        mesh(geo.foot, this.mats.boot, 0, -0.42, 0.05),
      );
      hip.add(knee);
      this.body.add(hip);
      return { hip, knee };
    });
    this.arms = [-1, 1].map((side) => {
      const shoulder = new THREE.Group();
      shoulder.position.set(0.235 * side, 0.5, 0);
      shoulder.rotation.z = 0.12 * side;
      shoulder.add(mesh(geo.upper, this.mats.jersey, 0, -0.14, 0));
      const elbow = new THREE.Group();
      elbow.position.y = -0.29;
      elbow.add(mesh(geo.fore, this.mats.skin, 0, -0.13, 0));
      shoulder.add(elbow);
      this.upper.add(shoulder);
      return { shoulder, elbow };
    });
    this.ring = new THREE.Mesh(
      geo.ring,
      new THREE.MeshBasicMaterial({
        color: "#ffcf4a",
        transparent: true,
        opacity: 0.95,
        depthWrite: false,
      }),
    );
    this.ring.rotation.x = -Math.PI / 2;
    this.ring.position.y = 0.04;
    this.ring.visible = false;
    this.root.add(this.ring);
    this.label = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: labelTexture(String(id)),
        depthWrite: false,
        transparent: true,
      }),
    );
    this.label.scale.set(0.95, 0.475, 1);
    this.label.position.y = 2.2;
    this.root.add(this.label);
    this.kitKey = null;
  }

  setKit(team, kind, colors) {
    const key = `${team}|${kind}|${colors[0]}|${colors[1]}`;
    if (key === this.kitKey) return;
    this.kitKey = key;
    let kit;
    if (kind === "referee" || team === -2) kit = KIT.referee;
    else if (team !== 0 && team !== 1) kit = KIT.unknown;
    else if (kind === "goalkeeper")
      kit = {
        jersey: KEEPER[team],
        shorts: shade(KEEPER[team], 0.45),
        socks: KEEPER[team],
      };
    else
      kit = {
        jersey: colors[team],
        shorts: shade(colors[team], 0.42),
        socks: colors[team],
      };
    this.mats.jersey.color.set(kit.jersey);
    this.mats.shorts.color.set(kit.shorts);
    this.mats.socks.color.set(kit.socks);
  }

  setGhost(ghost) {
    // Interpolated (gap-filled) positions render translucent, like hollow map markers.
    for (const m of Object.values(this.mats)) {
      m.transparent = ghost;
      m.opacity = ghost ? 0.45 : 1;
      m.depthWrite = !ghost;
    }
  }

  /** Advance the run cycle. `dt` is playback time, so a paused clip holds the pose. */
  animate(dt, speed) {
    const target = clamp((speed - 0.3) / 6, 0, 1);
    this.amp += (target - this.amp) * clamp(dt * 6, 0, 1);
    const a = this.amp;
    const cadence = speed > 0.35 ? 0.8 + 0.13 * Math.min(speed, 9) : 0.6;
    this.phase += Math.PI * 2 * cadence * dt;
    const s = Math.sin(this.phase),
      c = Math.cos(this.phase);
    const swing = 0.08 + 0.62 * a;
    this.legs[0].hip.rotation.x = -s * swing;
    this.legs[1].hip.rotation.x = s * swing;
    this.legs[0].knee.rotation.x =
      0.06 + (0.25 + 1.1 * a) * Math.max(0, c) * Math.min(1, a * 3);
    this.legs[1].knee.rotation.x =
      0.06 + (0.25 + 1.1 * a) * Math.max(0, -c) * Math.min(1, a * 3);
    const arm = 0.06 + 0.6 * a;
    this.arms[0].shoulder.rotation.x = s * arm;
    this.arms[1].shoulder.rotation.x = -s * arm;
    for (const { elbow } of this.arms) elbow.rotation.x = -(0.25 + 1.0 * a);
    this.upper.rotation.x = 0.03 + 0.16 * a;
    this.body.position.y = Math.abs(s) * 0.055 * a;
  }
}

const MODES = ["film", "orbit", "top", "pov"];

export class Pitch3D {
  constructor(host, { length, width, onSelect, onPick }) {
    this.host = host;
    this.L = length;
    this.W = width;
    this.onSelect = onSelect;
    this.onPick = onPick;
    this.mode = "orbit";
    this.povStyle = "eyes";
    this.colors = ["#d54b3a", "#4172cd"];
    this.showIds = true;
    this.figures = new Map();
    this.state = {
      time: 0,
      players: [],
      ball: null,
      film: null,
      selected: null,
    };
    this.lastTime = null;
    this.frameRequested = false;
    this.drawing = false;
    this.loop = this.loop.bind(this);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "low-power",
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.0;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.setAttribute(
      "aria-label",
      "3D replay of tracked positions",
    );
    renderer.domElement.style.background = "#b9d3e6";
    host.append(renderer.domElement);
    this.renderer = renderer;

    const scene = new THREE.Scene();
    scene.background = skyTexture();
    scene.fog = new THREE.Fog("#cfdde2", 140, 360);
    this.scene = scene;
    this.camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 1200);
    this.camera.position.set(-28, 42, 78);
    this.camera.lookAt(0, 0, 0);
    this.controls = new OrbitControls(this.camera, renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.maxPolarAngle = Math.PI * 0.49;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 220;
    this.controls.addEventListener("change", () => this.request());

    scene.add(new THREE.HemisphereLight("#e4f0ff", "#3d5a2c", 1.25));
    const sun = new THREE.DirectionalLight("#fff1dc", 2.4);
    sun.position.set(-45, 85, 35);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    Object.assign(sun.shadow.camera, {
      left: -72,
      right: 72,
      top: 50,
      bottom: -50,
      near: 10,
      far: 220,
    });
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.03;
    scene.add(sun);

    const margin = 6;
    const turf = new THREE.Mesh(
      new THREE.PlaneGeometry(length + 2 * margin, width + 2 * margin),
      new THREE.MeshStandardMaterial({
        map: pitchTexture(
          length,
          width,
          margin,
          renderer.capabilities.getMaxAnisotropy(),
        ),
        roughness: 0.95,
      }),
    );
    turf.rotation.x = -Math.PI / 2;
    turf.receiveShadow = true;
    scene.add(turf);
    this.ground = turf;
    const apron = new THREE.Mesh(
      new THREE.PlaneGeometry(600, 420),
      new THREE.MeshStandardMaterial({ color: "#2f5f2e", roughness: 1 }),
    );
    apron.rotation.x = -Math.PI / 2;
    apron.position.y = -0.02;
    apron.receiveShadow = true;
    scene.add(apron);
    this.addSurroundings(margin);
    for (const side of [-1, 1]) this.addGoal(side);

    this.geo = playerGeometry();
    this.ball = new THREE.Mesh(
      new THREE.SphereGeometry(0.2, 24, 16),
      new THREE.MeshStandardMaterial({ map: ballTexture(), roughness: 0.45 }),
    );
    this.ball.castShadow = true;
    this.ball.visible = false;
    scene.add(this.ball);

    this.overlayTexture = null;
    this.overlay = new THREE.Mesh(
      new THREE.PlaneGeometry(length, width),
      new THREE.MeshBasicMaterial({
        transparent: true,
        depthWrite: false,
        toneMapped: false,
      }),
    );
    this.overlay.rotation.x = -Math.PI / 2;
    this.overlay.position.y = 0.03;
    this.overlay.renderOrder = 1;
    this.overlay.visible = false;
    scene.add(this.overlay);

    this.raycaster = new THREE.Raycaster();
    this.groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
    this.bindPointer();
    renderer.domElement.addEventListener("dblclick", () => {
      if (this.mode === "orbit")
        this.frameAction(this.state.players, this.state.ball);
    });
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();
  }

  addSurroundings(margin) {
    const boardMat = new THREE.MeshStandardMaterial({
      color: "#1f3a2c",
      roughness: 0.7,
    });
    const trimMat = new THREE.MeshStandardMaterial({
      color: "#d8e3d1",
      roughness: 0.6,
    });
    const L = this.L,
      W = this.W,
      off = margin + 1.5;
    const board = (w, x, z, rotY) => {
      const g = new THREE.Group();
      const panel = new THREE.Mesh(
        new THREE.BoxGeometry(w, 0.9, 0.12),
        boardMat,
      );
      panel.position.y = 0.45;
      panel.castShadow = panel.receiveShadow = true;
      const trim = new THREE.Mesh(
        new THREE.BoxGeometry(w, 0.06, 0.13),
        trimMat,
      );
      trim.position.y = 0.88;
      g.add(panel, trim);
      g.position.set(x, 0, z);
      g.rotation.y = rotY;
      this.scene.add(g);
    };
    board(L + 2 * off, 0, -(W / 2 + off), 0);
    board(L + 2 * off, 0, W / 2 + off, 0);
    board(W + 2 * off, -(L / 2 + off), 0, Math.PI / 2);
    board(W + 2 * off, L / 2 + off, 0, Math.PI / 2);
    // A low stepped stand on the far side gives the scene depth without many polygons.
    const standMat = new THREE.MeshStandardMaterial({
      color: "#8e9a92",
      roughness: 0.9,
    });
    for (let row = 0; row < 6; row++) {
      const step = new THREE.Mesh(
        new THREE.BoxGeometry(L * 0.8, 0.5, 0.9),
        standMat,
      );
      step.position.set(0, 0.25 + row * 0.5, -(W / 2 + off + 2.5 + row * 0.9));
      step.scale.y = 1 + row;
      step.position.y = (0.5 * (1 + row)) / 2;
      step.receiveShadow = true;
      this.scene.add(step);
    }
  }

  addGoal(side) {
    const postMat = new THREE.MeshStandardMaterial({
      color: "#ffffff",
      roughness: 0.35,
    });
    const netMap = netTexture();
    const g = new THREE.Group();
    const x = (side * this.L) / 2;
    const post = new THREE.CylinderGeometry(0.06, 0.06, 2.44, 12);
    for (const z of [-3.66, 3.66]) {
      const p = new THREE.Mesh(post, postMat);
      p.position.set(0, 1.22, z);
      p.castShadow = true;
      g.add(p);
    }
    const bar = new THREE.Mesh(
      new THREE.CylinderGeometry(0.06, 0.06, 7.44, 12),
      postMat,
    );
    bar.rotation.x = Math.PI / 2;
    bar.position.set(0, 2.44, 0);
    bar.castShadow = true;
    g.add(bar);
    const depth = 2.0;
    const net = (w, h, repeatW, repeatH) => {
      const map = netMap.clone();
      map.needsUpdate = true;
      map.repeat.set(repeatW, repeatH);
      return new THREE.Mesh(
        new THREE.PlaneGeometry(w, h),
        new THREE.MeshBasicMaterial({
          map,
          transparent: true,
          opacity: 0.6,
          side: THREE.DoubleSide,
          depthWrite: false,
        }),
      );
    };
    const back = net(7.32, 2.44, 40, 13);
    back.rotation.y = Math.PI / 2;
    back.position.set(side * depth, 1.22, 0);
    const top = net(depth, 7.32, 11, 40);
    top.rotation.x = -Math.PI / 2;
    top.position.set((side * depth) / 2, 2.44, 0);
    g.add(back, top);
    for (const z of [-3.66, 3.66]) {
      const s = net(depth, 2.44, 11, 13);
      s.position.set((side * depth) / 2, 1.22, z);
      g.add(s);
    }
    g.position.x = x;
    this.scene.add(g);
  }

  bindPointer() {
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener("pointerdown", (e) => {
      down = { x: e.clientX, y: e.clientY };
    });
    el.addEventListener("pointerup", (e) => {
      if (!down || this.drawing) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      down = null;
      if (moved > 6) return;
      const hit = this.pick(e.clientX, e.clientY);
      if (hit !== null) this.onSelect?.(hit);
    });
  }

  ray(clientX, clientY) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    return this.raycaster;
  }

  /** Track ID under the pointer, or null. */
  pick(clientX, clientY) {
    const meshes = [];
    for (const f of this.figures.values())
      if (f.root.visible)
        f.root.traverse(
          (o) => o.isMesh && o.userData.id !== undefined && meshes.push(o),
        );
    const hit = this.ray(clientX, clientY).intersectObjects(meshes, false)[0];
    return hit ? hit.object.userData.id : null;
  }

  /** Pitch coordinates (metres) of the ground under the pointer, or null. */
  pitchAt(clientX, clientY) {
    const point = new THREE.Vector3();
    if (!this.ray(clientX, clientY).ray.intersectPlane(this.groundPlane, point))
      return null;
    return [point.x + this.L / 2, point.z + this.W / 2];
  }

  /** Point the free camera at the play: the ball, else the players' centre. */
  frameAction(players, ball) {
    const pts = ball ? [ball] : players;
    if (!pts.length) return;
    const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length - this.L / 2,
      cz = pts.reduce((s, p) => s + p.y, 0) / pts.length - this.W / 2;
    this.controls.target.set(cx, 0, cz);
    this.camera.position.set(cx - 12, 22, cz + 32);
    this.camera.up.set(0, 1, 0);
    this.controls.update();
    this.request();
  }

  setDrawing(on) {
    this.drawing = on;
    this.controls.enabled = !on && this.mode === "orbit";
    this.renderer.domElement.style.cursor = on ? "crosshair" : "";
  }

  setMode(mode) {
    if (!MODES.includes(mode) || mode === this.mode) return;
    const previous = this.mode;
    this.mode = mode;
    this.controls.enabled = mode === "orbit" && !this.drawing;
    if (mode === "orbit") {
      // Orbit around the ground point the camera was looking at, so the view does not jump.
      const dir = new THREE.Vector3();
      this.camera.getWorldDirection(dir);
      const target = new THREE.Vector3();
      const ray = new THREE.Ray(this.camera.position.clone(), dir);
      if (!ray.intersectPlane(this.groundPlane, target) || previous === "pov")
        target.set(0, 0, 0);
      target.x = clamp(target.x, -this.L / 2, this.L / 2);
      target.z = clamp(target.z, -this.W / 2, this.W / 2);
      this.controls.target.copy(target);
      if (this.camera.position.y < 8)
        this.camera.position.set(target.x - 20, 30, target.z + 45);
      this.camera.up.set(0, 1, 0);
    }
    this.request();
  }

  setPovStyle(style) {
    this.povStyle = style;
    this.request();
  }

  setColors(home, away) {
    this.colors = [home, away];
    for (const f of this.figures.values()) f.kitKey = null;
    this.request();
  }

  setShowIds(on) {
    this.showIds = on;
    this.request();
  }

  /** Paint an externally drawn canvas (coverage, drawings) onto the field. */
  setOverlay(canvas) {
    if (!canvas) {
      this.overlay.visible = false;
      this.request();
      return;
    }
    if (!this.overlayTexture || this.overlayTexture.image !== canvas) {
      this.overlayTexture?.dispose();
      this.overlayTexture = new THREE.CanvasTexture(canvas);
      this.overlayTexture.colorSpace = THREE.SRGBColorSpace;
      this.overlayTexture.anisotropy =
        this.renderer.capabilities.getMaxAnisotropy();
      this.overlay.material.map = this.overlayTexture;
      this.overlay.material.needsUpdate = true;
    } else this.overlayTexture.needsUpdate = true;
    this.overlay.visible = true;
    this.request();
  }

  /** New playback state: smoothed players, ball, film camera pose and selection. */
  update(state) {
    this.state = state;
    this.request();
  }

  request() {
    if (this.frameRequested) return;
    this.frameRequested = true;
    requestAnimationFrame(this.loop);
  }

  visible() {
    return this.host.offsetParent !== null && this.host.clientWidth > 0;
  }

  loop() {
    this.frameRequested = false;
    if (!this.visible()) return;
    const keepGoing = this.step();
    this.renderer.render(this.scene, this.camera);
    if (keepGoing) this.request();
  }

  /** Apply state to the scene and move the camera. True while the camera is still settling. */
  step() {
    const { time, players, ball, selected } = this.state;
    const now = performance.now();
    const wall = this.lastWall
      ? Math.min(0.1, (now - this.lastWall) / 1000)
      : 0;
    this.lastWall = now;
    let dt = this.lastTime === null ? 0 : time - this.lastTime;
    const jumped = Math.abs(dt) > 0.4;
    dt = jumped ? 0 : Math.max(0, dt);
    this.lastTime = time;

    const seen = new Set();
    for (const p of players) {
      let f = this.figures.get(p.id);
      if (!f) {
        f = new Figure(this.geo, p.id);
        this.figures.set(p.id, f);
        this.scene.add(f.root);
      }
      seen.add(p.id);
      f.setKit(p.team, p.kind, this.colors);
      f.setGhost(p.measured === false);
      f.root.visible = !(
        this.mode === "pov" &&
        this.povStyle === "eyes" &&
        p.id === selected
      );
      f.root.position.set(p.x - this.L / 2, 0, p.y - this.W / 2);
      let target = f.heading;
      if (p.speed > 0.7) target = Math.atan2(p.vx, p.vy);
      else if (ball) target = Math.atan2(ball.x - p.x, ball.y - p.y);
      const turn = wrapAngle(target - f.heading);
      f.heading =
        jumped || !f.placed ? target : f.heading + turn * clamp(dt * 8, 0, 1);
      f.placed = true;
      f.root.rotation.y = f.heading;
      f.animate(dt, p.speed);
      f.ring.visible = p.id === selected;
      f.label.visible = this.showIds && p.kind !== "referee";
    }
    for (const [id, f] of this.figures)
      if (!seen.has(id)) f.root.visible = false;

    if (ball) {
      const x = ball.x - this.L / 2,
        z = ball.y - this.W / 2;
      if (this.ball.visible && !jumped) {
        const dx = x - this.ball.position.x,
          dz = z - this.ball.position.z;
        this.ball.rotation.x += dz / 0.2;
        this.ball.rotation.z -= dx / 0.2;
      }
      this.ball.position.set(x, 0.2, z);
      this.ball.visible = true;
    } else this.ball.visible = false;

    return this.moveCamera(wall);
  }

  /** Camera target for the current mode; returns {position, look|quaternion, fov} or null. */
  cameraTarget() {
    const { players, ball, film, selected } = this.state;
    const aspect = this.camera.aspect;
    if (this.mode === "film" && film) {
      const [right, up, back] = film.basis;
      const m = new THREE.Matrix4().makeBasis(
        new THREE.Vector3(...right),
        new THREE.Vector3(...up),
        new THREE.Vector3(...back),
      );
      const quaternion = new THREE.Quaternion().setFromRotationMatrix(m);
      // Keep the film's horizontal coverage when the pane is narrower than the footage.
      const half = Math.tan((film.fov * Math.PI) / 360);
      const fov =
        aspect < film.aspect
          ? (2 * Math.atan((half * film.aspect) / aspect) * 180) / Math.PI
          : film.fov;
      return {
        position: new THREE.Vector3(...film.position),
        quaternion,
        fov,
        rate: 10,
      };
    }
    if (this.mode === "top")
      return {
        position: new THREE.Vector3(0, 108, 0.01),
        look: new THREE.Vector3(0, 0, 0),
        up: new THREE.Vector3(0, 0, -1),
        fov:
          aspect >= this.L / this.W
            ? 36
            : (2 *
                Math.atan(
                  (Math.tan((36 * Math.PI) / 360) * (this.L / this.W)) / aspect,
                ) *
                180) /
              Math.PI,
        rate: 5,
      };
    if (this.mode === "pov") {
      const p = players.find((q) => q.id === selected);
      if (!p) return null;
      const f = this.figures.get(p.id);
      const at = new THREE.Vector3(p.x - this.L / 2, 0, p.y - this.W / 2);
      const heading = f ? f.heading : 0;
      const facing = new THREE.Vector3(Math.sin(heading), 0, Math.cos(heading));
      const focus = ball
        ? new THREE.Vector3(ball.x - this.L / 2, 0.4, ball.y - this.W / 2)
        : at.clone().addScaledVector(facing, 12).setY(1.2);
      if (this.povStyle === "eyes") {
        const eye = at.clone().setY(1.68).addScaledVector(facing, 0.12);
        if (focus.distanceTo(eye) < 1.5)
          focus.copy(eye).addScaledVector(facing, 6).setY(1.0);
        return { position: eye, look: focus, fov: 70, rate: 9 };
      }
      const toward = focus.clone().sub(at).setY(0);
      if (toward.lengthSq() < 4) toward.copy(facing);
      toward.normalize();
      const position = at.clone().addScaledVector(toward, -6.5).setY(3.1);
      const look = at.clone().addScaledVector(toward, 5).setY(1.0);
      return { position, look, fov: 55, rate: 7 };
    }
    return null;
  }

  moveCamera(wall) {
    if (this.mode === "orbit") {
      if (this.camera.fov !== 42) {
        this.camera.fov = 42;
        this.camera.updateProjectionMatrix();
      }
      return this.controls.update();
    }
    const target = this.cameraTarget();
    if (!target) return false;
    let quaternion = target.quaternion;
    if (!quaternion) {
      const m = new THREE.Matrix4().lookAt(
        target.position,
        target.look,
        target.up || new THREE.Vector3(0, 1, 0),
      );
      quaternion = new THREE.Quaternion().setFromRotationMatrix(m);
    }
    const k = wall > 0 ? 1 - Math.exp(-target.rate * wall) : 1;
    const settle =
      this.camera.position.distanceTo(target.position) > 0.02 ||
      this.camera.quaternion.angleTo(quaternion) > 0.002 ||
      Math.abs(this.camera.fov - target.fov) > 0.05;
    this.camera.position.lerp(target.position, k);
    this.camera.quaternion.slerp(quaternion, k);
    this.camera.fov += (target.fov - this.camera.fov) * k;
    this.camera.updateProjectionMatrix();
    return settle;
  }

  resize() {
    const w = this.host.clientWidth,
      h = this.host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.request();
  }

  dispose() {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.scene.traverse((o) => {
      o.geometry?.dispose();
      for (const m of [o.material].flat())
        if (m) {
          m.map?.dispose();
          m.dispose();
        }
    });
    this.renderer.dispose();
    this.renderer.domElement.remove();
  }
}
