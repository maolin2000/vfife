"""The page `vfife view` writes (post/view.py): one HTML file, the result's
data embedded as base64, three.js from a CDN. Three placeholders:
__TITLE__, __META__ (the JSON the exporter builds) and __BLOB__ (the
quantised displacement and field bytes)."""

PAGE = r'''<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · vfife view</title>
<style>
  :root { --bg: #0d1418; --panel: rgba(16, 24, 29, 0.86); --edge: #2a3a44; --ink: #e6ecef; --dim: #93a4ad;
          --accent: #ffb347; --hot: #ff6a4d; --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
          --body: system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 var(--body); overflow: hidden; }
  canvas { display: block; }
  .panel { position: absolute; background: var(--panel); border: 1px solid var(--edge); border-radius: 6px; backdrop-filter: blur(6px); }
  #head { top: 16px; left: 16px; padding: 12px 16px; max-width: min(460px, calc(100vw - 32px)); }
  #head h1 { font: 600 18px/1.25 var(--body); margin: 0 0 4px; letter-spacing: 0.01em; }
  #head .sub { color: var(--dim); font: 12px var(--mono); }
  .acts { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
  .acts .btn[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); }
  #read { top: 16px; right: 16px; padding: 10px 14px; min-width: 220px; font: 13px var(--mono); }
  #read .row { display: flex; justify-content: space-between; gap: 16px; padding: 2px 0; }
  #read .row span:first-child { color: var(--dim); }
  #read .hot { color: var(--hot); }
  #read select { font: 12px var(--mono); background: #101a20; color: var(--ink); border: 1px solid var(--edge); border-radius: 4px; padding: 2px 4px; max-width: 150px; }
  #ramp { height: 8px; border-radius: 4px; margin: 8px 0 2px; background: linear-gradient(90deg,#30123b,#4662d7,#36aaf9,#1ae4b6,#72fe5e,#c8ef34,#faba39,#f66b19,#ca2a04); }
  #ticks { display: flex; justify-content: space-between; font: 11px var(--mono); color: var(--dim); }
  #foot { left: 16px; right: 16px; bottom: 16px; padding: 10px 14px 12px; }
  #chart { width: 100%; height: 70px; display: block; }
  .ctlrow { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 8px; }
  .btn { font: 500 13px var(--body); color: var(--ink); background: transparent; border: 1px solid var(--edge); border-radius: 4px; padding: 5px 10px; cursor: pointer; }
  .btn.primary { background: var(--accent); color: #1a1408; border-color: var(--accent); }
  #scrub { flex: 1; min-width: 140px; accent-color: var(--accent); }
  .ctlrow label { color: var(--dim); font-size: 12px; }
  .ctlrow select { font: 12px var(--mono); background: #101a20; color: var(--ink); border: 1px solid var(--edge); border-radius: 4px; padding: 3px 4px; }
  #time { font: 13px var(--mono); min-width: 74px; }
  #note { color: var(--dim); font-size: 12px; margin-top: 6px; }
  @media (max-width: 720px) { #read { display: none; } #head { max-width: calc(100vw - 32px); } }
</style>
</head>
<body>
<div class="panel" id="head">
  <h1>__TITLE__</h1>
  <div class="sub" id="sub"></div>
</div>
<div class="panel" id="read">
  <div class="row"><span id="r-t-label">时间 time</span><span id="r-t">–</span></div>
  <div class="row"><span>帧 frame</span><span id="r-f">–</span></div>
  <div class="row"><span>已断 broken</span><span id="r-n">–</span></div>
  <div class="row"><span>云图 field</span><select id="field"></select></div>
  <div id="ramp"></div>
  <div id="ticks"><span id="t-lo"></span><span id="t-mid"></span><span id="t-hi"></span></div>
</div>
<div class="panel" id="foot">
  <canvas id="chart"></canvas>
  <div class="ctlrow">
    <button class="btn primary" id="play">暂停</button>
    <span id="time">0.00 s</span>
    <input id="scrub" type="range" min="0" max="1000" value="0" aria-label="time">
    <label>位移放大 <select id="amp"></select></label>
    <label>速度 <select id="speed"><option>0.25</option><option>0.5</option><option selected>1</option><option>2</option><option>4</option></select></label>
    <label>监测 <select id="mon"></select></label>
    <label><input type="checkbox" id="rel" checked> 相对首帧放大</label>
    <label><input type="checkbox" id="hide"> 隐藏已断构件</label>
    <label title="节点场本来连续；单元场在节点上按同一截面的构件取平均"><input type="checkbox" id="smooth" checked> 连续着色</label>
    <label><input type="checkbox" id="axes" checked> 坐标轴</label>
    <button class="btn" id="view">换视角</button>
  </div>
  <div id="note"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const META = __META__;
const BLOB = "__BLOB__";
const $ = id => document.getElementById(id);
(function main() {
  if (typeof THREE === "undefined") {
    $("note").textContent = "three.js 没有加载：这页要联网从 cdnjs.cloudflare.com 和 cdn.jsdelivr.net 取 three.js；"
      + "连不上时用 vfife view --scripts 目录 把 three.js 写进页面（three.js not loaded: the page fetches it from "
      + "cdnjs.cloudflare.com and cdn.jsdelivr.net; where those are not reachable, vfife view --scripts DIR writes it into the page）。";
    return;
  }
  // ---------------------------------------------------------------- data
  const raw = Uint8Array.from(atob(BLOB), c => c.charCodeAt(0));
  const nodes = META.nodes, nn = nodes.length;
  // one ACT per result file: its frames, fields, monitors and breaks;
  // decoded on first use, the geometry is shared
  const acts = META.acts.map(a => Object.assign({ ready: false }, a));
  function decode(A) {
    if (A.ready) return A;
    const nf = A.frames;
    // only the particles the run used are stored (the spares a split may
    // need are parked where they are until it does)
    const ids = A.nodes ? new Int32Array(raw.buffer.slice(A.nodes.offset, A.nodes.offset + A.nodes.n * 4)) : null;
    const ns = ids ? ids.length : nn;
    const q16 = new Int16Array(raw.buffer.slice(A.offset, A.offset + nf * ns * 3 * 2));
    A.pos = new Float32Array(nf * nn * 3);
    for (let f = 0; f < nf; f++) for (let n = 0; n < nn; n++) {
      const o = (f * nn + n) * 3;
      A.pos[o] = nodes[n][0]; A.pos[o + 1] = nodes[n][1]; A.pos[o + 2] = nodes[n][2];
    }
    for (let f = 0; f < nf; f++) for (let k = 0; k < ns; k++) {
      const n = ids ? ids[k] : k, o = (f * nn + n) * 3, oq = (f * ns + k) * 3;
      A.pos[o] += q16[oq] * A.scale; A.pos[o + 1] += q16[oq + 1] * A.scale; A.pos[o + 2] += q16[oq + 2] * A.scale;
    }
    A.fields = A.fields.map(f => {
      if (f.derived !== undefined) return Object.assign({}, f);
      if (f.constant !== undefined) {
        // a field that never changed: one value for every frame and item
        const v = f.constant === null ? 255 : 0;
        return Object.assign({}, f, { data: new Uint8Array(nf * f.n).fill(v) });
      }
      return Object.assign({}, f, { data: new Uint8Array(raw.buffer, f.offset, nf * f.n) });
    });
    A.byName = {};
    for (const f of A.fields) A.byName[f.on + ":" + f.name] = f;
    // cause 2: a massless connector taken out because it hung from nothing
    // massive -- not debris; it is not drawn from then on
    A.dropped = {};
    for (const ev of A.failures) if (ev[3] >= 2) A.dropped[ev[1] | 0] = ev[0];
    if (A.failed_from_events) {
      // member_failed rebuilt from the event list: failed from its event on
      const ne = META.n_elements, data = new Uint8Array(nf * ne);
      for (const ev of A.failures) {
        const e = ev[1] | 0;
        for (let f = 0; f < nf; f++) if (A.times[f] >= ev[0] - 1e-9) data[f * ne + e] = 254;
      }
      A.byName["elements:member_failed"] = { name: "member_failed", on: "elements", lo: 0, hi: 1, n: ne, data };
    }
    // the largest displacement anywhere, for the auto amplification: from
    // the FIRST frame, which is the state the run started in (a dead-load
    // state on a bridge), so amplifying does not blow up the dead sag
    let umax = 0, umax0 = 0;
    for (let i = 0; i < q16.length; i++) {
      umax = Math.max(umax, Math.abs(q16[i]));
      umax0 = Math.max(umax0, Math.abs(q16[i] - q16[i % (ns * 3)]));
    }
    A.umax = umax * A.scale; A.umax0 = umax0 * A.scale; A.nf = nf;
    // the connectivity of every frame, once members have moved onto other
    // particles (a split-off segment); null while nothing moved
    A.connAt = null;
    if (A.conn) {
      // per element, its connectivity changes as [frame, a, b], in order
      const byEl = {};
      for (const [f, e, a, b] of A.conn.changes) (byEl[e] = byEl[e] || []).push([f, a, b]);
      const base = A.conn.base, mesh0 = {};
      for (const p of META.parts) if (p.kind === "line" && p.group === "elements") p.owner.forEach((o, i) => { mesh0[o] = p.conn[i]; });
      A.connAt = (f, e) => {
        let c = base ? base[e] : mesh0[e];
        const ch = byEl[e];
        if (ch) for (const x of ch) { if (x[0] <= f) c = [x[1], x[2]]; else break; }
        return c;
      };
    }
    A.ready = true;
    return A;
  }
  let A = decode(acts[0]);
  const failedOf = group => A.byName[group + ":member_failed"] || null;
  // the model's frame: z up in the model is y up on screen, centred on the box
  const [lo, hi] = META.bbox;
  const centre = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  const diag = Math.max(1e-9, Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]));
  const T = (x, y, z, out) => out.set(x - centre[0], z - centre[2], -(y - centre[1]));
  const thickness = META.thickness || diag / 300;
  const sizes = META.sizes;               // [depth, width] per element from the case, or null
  // a relaxation or static act runs on a load factor, not in seconds, and an
  // arc-length one goes back and below zero: labelled λ and drawn against the
  // record's index
  // a creep / ageing relaxation runs on the concrete's age in days: the same
  // index axis, labelled by the age
  const isAge = () => A.axis === "age" && A.ages;
  const isLoad = () => A.axis === "load" || A.axis === "age";
  const fText = (f, d) => isAge() ? `${A.ages[f].toFixed(Math.max(d - 2, 1))} d` : isLoad() ? `λ ${A.times[f].toFixed(d)}` : `${A.times[f].toFixed(d)} s`;
  function subtitle() {
    const tEnd = isAge() ? `${A.ages[0].toFixed(1)} .. ${A.ages[A.nf - 1].toFixed(1)} d`
      : isLoad() ? `λ ${Math.min(...A.times).toFixed(3)} .. ${Math.max(...A.times).toFixed(3)}` : `${A.times[A.nf - 1].toFixed(3)} s`;
    $("sub").textContent = `${META.family} · ${nn} 节点 · ${META.n_elements} 单元 · ${A.nf} 帧 · ${tEnd} · vfife view`;
    $("r-t-label").textContent = isAge() ? "龄期 age" : isLoad() ? "荷载因子 load factor" : "时间 time";
  }
  subtitle();
  if (acts.length > 1) {
    const row = document.createElement("div"); row.className = "acts"; row.id = "acts";
    acts.forEach((a, k) => {
      const b = document.createElement("button"); b.className = "btn"; b.textContent = a.name;
      b.setAttribute("aria-pressed", k === 0 ? "true" : "false");
      b.addEventListener("click", () => useAct(k));
      row.appendChild(b);
    });
    $("head").appendChild(row);
  }

  // --------------------------------------------------------------- scene
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.autoClear = false;                  // the axis triad is drawn over the scene
  document.body.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1418);
  const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, diag / 1000, diag * 50);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xcfe3ee, 0x1a2228, 0.95));
  const sun = new THREE.DirectionalLight(0xffffff, 0.8); sun.position.set(diag, 2 * diag, diag); scene.add(sun);
  const grid = new THREE.GridHelper(diag * 2, 20, 0x2a3a44, 0x1c272e);
  grid.position.y = lo[2] - centre[2] - thickness; scene.add(grid);
  const VIEWS = [[0.65, 0.4, 0.85], [0, 0.1, 1.2], [0, 1.3, 0.01], [1.2, 0.15, 0]];
  let viewIndex = 0;
  function setView(k) {
    const v = VIEWS[k % VIEWS.length];
    camera.position.set(v[0] * diag, v[1] * diag, v[2] * diag);
    // the target a little below the centre: the model sits in the upper part
    // of the screen, clear of the time axis
    controls.target.set(0, -0.12 * diag, 0); controls.update();
  }
  setView(0);
  $("view").addEventListener("click", () => setView(++viewIndex));

  // ------------------------------------------------------------- axes
  // THE MODEL'S GLOBAL AXES, twice: an axis box on three edges of the
  // model's bounding box with ticks in model units, and an orientation
  // triad in the corner that turns with the view. Screen y is model z.
  const AX = [["X", 0xff7a6b], ["Y", 0x7fd98c], ["Z", 0x6fa8ff]];
  function label(text, colour, px) {
    const cv = document.createElement("canvas"), g = cv.getContext("2d"), fs = 48;
    g.font = `500 ${fs}px ui-monospace, Menlo, Consolas, monospace`;
    cv.width = Math.ceil(g.measureText(text).width) + 16; cv.height = fs + 16;
    g.font = `500 ${fs}px ui-monospace, Menlo, Consolas, monospace`;
    g.fillStyle = colour; g.textBaseline = "middle"; g.fillText(text, 8, cv.height / 2);
    const tex = new THREE.CanvasTexture(cv); tex.minFilter = THREE.LinearFilter;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, sizeAttenuation: false, depthTest: false, transparent: true }));
    const h = px / 800 * 0.73;                          // px tall on an 800 px view at fov 40
    sp.scale.set(h * cv.width / cv.height, h, 1); sp.renderOrder = 10;
    return sp;
  }
  function niceStep(range, n) {
    const raw = range / n, p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }
  const axes = new THREE.Group(); scene.add(axes);
  (function buildAxes() {
    const segs = [], v = new THREE.Vector3(), w = new THREE.Vector3();
    const o = lo.slice(), tick = diag / 150;
    const unit = META.units && META.units.length ? META.units.length : "m";
    for (let k = 0; k < 3; k++) {
      const a0 = lo[k], a1 = hi[k];
      if (a1 - a0 < 1e-9) continue;
      const p = o.slice(), q = o.slice(); q[k] = a1;
      T(p[0], p[1], p[2], v); T(q[0], q[1], q[2], w); segs.push(v.x, v.y, v.z, w.x, w.y, w.z);
      // an axis short next to the model (a deck's width beside its length)
      // gets only its two outermost ticks: five would print on each other
      const short = a1 - a0 < 0.1 * diag;
      const st = niceStep(a1 - a0, short ? 2 : 5), first = Math.ceil(a0 / st - 1e-9) * st;
      const lastT = Math.floor(a1 / st + 1e-9) * st;
      // ticks point away from the box: down for x and y, forward for z
      for (let t = first; t <= a1 + 1e-9 * st; t += st) {
        if (short && t > first + 1e-9 * st && t < lastT - 1e-9 * st) continue;
        const r = o.slice(); r[k] = t;
        T(r[0], r[1], r[2], v);
        const tv = k === 2 ? [0, 0, tick] : [0, -tick, 0];      // in screen coordinates
        segs.push(v.x, v.y, v.z, v.x + tv[0], v.y + tv[1], v.z + tv[2]);
        const sp = label(fmtTick(t, st), "#93a4ad", 13);
        sp.position.set(v.x + 3 * tv[0], v.y + 3 * tv[1], v.z + 3 * tv[2]);
        sp.center.set(k === 2 ? 0 : 0.5, k === 2 ? 0.5 : 1);
        axes.add(sp);
      }
      // the title past the end of x and y; beside the top of z (above it,
      // a tall model's title would sit under the header)
      const r = o.slice(); r[k] = k === 2 ? a1 : a1 + 0.04 * diag;
      T(r[0], r[1], r[2], v);
      const t = label(`${AX[k][0]} (${unit})`, "#" + AX[k][1].toString(16).padStart(6, "0"), 16);
      t.position.copy(v); if (k === 2) t.center.set(-0.25, 0.5); axes.add(t);
    }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.Float32BufferAttribute(segs, 3));
    axes.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x6b7d86 })));
  })();
  function fmtTick(t, st) {
    const d = st >= 1 ? 0 : Math.min(6, Math.ceil(-Math.log10(st)));
    return (Math.abs(t) < 1e-9 * st ? 0 : t).toFixed(d);
  }
  $("axes").addEventListener("change", e => { axes.visible = e.target.checked; tri.visible = e.target.checked; });
  // the orientation triad: its own little scene and camera, drawn last in a
  // corner viewport with the main camera's direction
  const triScene = new THREE.Scene(), tri = new THREE.Group(); triScene.add(tri);
  const triCam = new THREE.OrthographicCamera(-1.5, 1.5, 1.5, -1.5, 0.1, 20);
  [[1, 0, 0], [0, 0, -1], [0, 1, 0]].forEach((d, k) => {           // model x, y, z on screen
    const dv = new THREE.Vector3(...d);
    tri.add(new THREE.ArrowHelper(dv, new THREE.Vector3(), 1, AX[k][1], 0.26, 0.13));
    const sp = label(AX[k][0], "#" + AX[k][1].toString(16).padStart(6, "0"), 40);
    sp.material.sizeAttenuation = true; sp.scale.set(0.36 * sp.scale.x / sp.scale.y, 0.36, 1);
    sp.position.copy(dv.multiplyScalar(1.28)); tri.add(sp);
  });
  const triDir = new THREE.Vector3();
  function drawTriad() {
    if (!tri.visible) return;
    triDir.subVectors(camera.position, controls.target).normalize().multiplyScalar(5);
    triCam.position.copy(triDir); triCam.up.copy(camera.up); triCam.lookAt(0, 0, 0);
    const size = 110, foot = $("foot").getBoundingClientRect();
    const y = innerHeight - foot.top + 8;
    renderer.clearDepth();
    renderer.setScissorTest(true);
    renderer.setViewport(12, y, size, size); renderer.setScissor(12, y, size, size);
    renderer.render(triScene, triCam);
    renderer.setScissorTest(false);
    renderer.setViewport(0, 0, innerWidth, innerHeight);
  }

  // ---------------------------------------------------------- colours
  const TURBO = [0x30123b, 0x4662d7, 0x36aaf9, 0x1ae4b6, 0x72fe5e, 0xc8ef34, 0xfaba39, 0xf66b19, 0xca2a04].map(c => new THREE.Color(c));
  const col = new THREE.Color();
  function turbo(u, out) {
    u = Math.min(Math.max(u, 0), 1) * (TURBO.length - 1);
    const i = Math.min(Math.floor(u), TURBO.length - 2);
    return out.copy(TURBO[i]).lerp(TURBO[i + 1], u - i);
  }
  const cBase = new THREE.Color(0xb9c4cb), cGone = new THREE.Color(0x4c545a), cRebar = new THREE.Color(0xd08a3a);

  // ------------------------------------------------------------ parts
  const box = new THREE.BoxGeometry(1, 1, 1);
  // CONTINUOUS COLOURING along a member: every instance carries a second
  // colour (instanceColorB) for its j end, and the vertex shader blends the
  // two along the member's axis (local y runs -0.5 at end i to +0.5 at end j,
  // for the box and for an extruded section alike)
  function twoTone(mat) {
    mat.onBeforeCompile = sh => {
      sh.vertexShader = sh.vertexShader
        .replace("#include <common>", "#include <common>\nattribute vec3 instanceColorB;")
        .replace("#include <color_vertex>", "#include <color_vertex>\n#ifdef USE_INSTANCING_COLOR\n  vColor.xyz = mix(instanceColor.xyz, instanceColorB, clamp(position.y + 0.5, 0.0, 1.0));\n#endif");
    };
    return mat;
  }
  function instanced(geo, mat, n) {
    const g = geo.clone();                          // its own copy: the second colour is per mesh
    g.setAttribute("instanceColorB", new THREE.InstancedBufferAttribute(new Float32Array(3 * n), 3).setUsage(THREE.DynamicDrawUsage));
    const m = new THREE.InstancedMesh(g, twoTone(mat), n);
    m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    return m;
  }
  const setB = (m, slot, c) => { const a = m.geometry.attributes.instanceColorB.array; a[3 * slot] = c.r; a[3 * slot + 1] = c.g; a[3 * slot + 2] = c.b; };
  const m4 = new THREE.Matrix4(), a = new THREE.Vector3(), b = new THREE.Vector3(), mid = new THREE.Vector3(), dir = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0), ALONG = new THREE.Vector3(1, 0, 0), d1 = new THREE.Vector3(), d2 = new THREE.Vector3();
  // real section outlines (write(..., profiles=)): each one extruded over a
  // unit length along local y, the outline in local x (width) and z (depth)
  // in metres -- drawn unscaled across, stretched to the member's length
  const profOf = META.profile_of || null;
  // An outboard section gets a second, mirrored geometry: mirroring through
  // the instance matrix would turn its faces inside out (lit black)
  const extrude = (pts) => {
    const g = new THREE.ExtrudeGeometry(new THREE.Shape(pts.map(([y, z]) => new THREE.Vector2(y, z))),
                                        { depth: 1, bevelEnabled: false, steps: 1 });
    g.translate(0, 0, -0.5); g.rotateX(Math.PI / 2);          // extrusion along y, outline y -> z
    g.computeVertexNormals();
    return g;
  };
  const profGeo = (META.profiles || []).map(p => [extrude(p.outline)].concat(
    p.outboard ? [extrude(p.outline.map(([y, z]) => [-y, z]).reverse())] : []));
  const parts = META.parts.map(p => {
    if (p.kind === "line") {
      // two instance slots per member: itself, and -- once broken and its
      // ends parted -- a second stub so each end keeps half of it
      const mesh = instanced(box, new THREE.MeshStandardMaterial({ roughness: 0.6, metalness: 0.05 }), 2 * p.conn.length);
      scene.add(mesh);
      // one more instanced mesh per real section this part uses, every slot
      // empty until a member of that section is drawn into it
      const pm = profGeo.map((gs, k) => {
        if (!profOf || p.group !== "elements" || !p.owner.some(o => profOf[o] === k)) return null;
        return gs.map(g => {
          const m = instanced(g, new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.1 }), 2 * p.conn.length);
          m4.makeScale(0, 0, 0);
          for (let s = 0; s < 2 * p.conn.length; s++) m.setMatrixAt(s, m4);
          scene.add(m);
          return m;
        });
      });
      const L0 = p.conn.map(([i, j]) => Math.hypot(nodes[i][0] - nodes[j][0], nodes[i][1] - nodes[j][1], nodes[i][2] - nodes[j][2]));
      const adj = {};
      p.conn.forEach(([i, j], e) => { (adj[i] = adj[i] || []).push(e); (adj[j] = adj[j] || []).push(e); });
      // the KIND of each member for averaging an element field onto its
      // nodes: same drawn section, same kind (a girder's value is not
      // blended with the cable's that meets it at the node)
      const kinds = {}, mkind = new Int32Array(p.conn.length);
      p.owner.forEach((o, i) => {
        const key = p.group !== "elements" ? "0" : (profOf && profOf[o] >= 0 ? "p" + profOf[o] : (sizes ? sizes[o].join(",") : "0"));
        mkind[i] = kinds[key] === undefined ? (kinds[key] = Object.keys(kinds).length) : kinds[key];
      });
      return Object.assign({ mesh, pm, L0, adj, mkind, nk: Object.keys(kinds).length }, p);
    }
    // faces: non-indexed triangles so every element can carry its own flat colour
    const tris = [];
    for (let i = 0; i < p.conn.length; i++) {
      const c = p.conn[i];
      tris.push([c[0], c[1], c[2], i]);
      if (c.length === 4) tris.push([c[0], c[2], c[3], i]);
    }
    const geo = new THREE.BufferGeometry();
    const P = new Float32Array(tris.length * 9), C = new Float32Array(tris.length * 9);
    geo.setAttribute("position", new THREE.BufferAttribute(P, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute("color", new THREE.BufferAttribute(C, 3).setUsage(THREE.DynamicDrawUsage));
    const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ vertexColors: true, side: THREE.DoubleSide, roughness: 0.7, metalness: 0.05,
                                                                       polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 }));
    scene.add(mesh);
    const edges = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0x0d1418, transparent: true, opacity: 0.35 }));
    const E = new Float32Array(p.conn.reduce((s, c) => s + c.length * 6, 0));
    edges.geometry.setAttribute("position", new THREE.BufferAttribute(E, 3).setUsage(THREE.DynamicDrawUsage));
    scene.add(edges);
    return Object.assign({ mesh, tris, P, C, edges, E }, p);
  });
  let rebar = null;
  if (META.rebar) {
    const g = new THREE.BufferGeometry();
    const R = new Float32Array(META.rebar.conn.length * 6);
    g.setAttribute("position", new THREE.BufferAttribute(R, 3));
    rebar = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: cRebar }));
    scene.add(rebar);
    for (let i = 0; i < META.rebar.conn.length; i++) {
      const [na, nb] = META.rebar.conn[i];
      const A = META.rebar.nodes[na], B = META.rebar.nodes[nb];
      T(A[0], A[1], A[2], a); T(B[0], B[1], B[2], b);
      R.set([a.x, a.y, a.z, b.x, b.y, b.z], i * 6);
    }
  }

  // ------------------------------------------------------------ state
  let frame = 0, playing = true, amp = 1, speed = 1, field = null, hideGone = false, relative = A.nf > 1;
  $("rel").checked = relative;
  const AMPS = [1, 2, 5, 10, 30, 100, 300, 1000];
  const pick = u => u > 0 ? Math.max(1, Math.pow(10, Math.floor(Math.log10(0.05 * diag / u)))) : 1;
  let autoAmp = pick(A.umax0);
  for (const v of [["auto", `自动 ×${autoAmp}`], ...AMPS.map(v => [v, "×" + v])]) {
    const o = document.createElement("option"); o.value = v[0]; o.textContent = v[1]; $("amp").appendChild(o);
  }
  amp = autoAmp;
  function setAuto() {
    autoAmp = pick(relative ? A.umax0 : A.umax);
    $("amp").options[0].textContent = `自动 ×${autoAmp}`;
    if ($("amp").value === "auto") amp = autoAmp;
  }
  $("rel").addEventListener("change", e => { relative = e.target.checked; setAuto(); draw(); });
  $("amp").addEventListener("change", e => { amp = e.target.value === "auto" ? autoAmp : +e.target.value; draw(); });
  $("speed").addEventListener("change", e => { speed = +e.target.value; });
  $("hide").addEventListener("change", e => { hideGone = e.target.checked; draw(); });
  function fillFields() {
    const sel = $("field"), keep = field ? field.on + ":" + field.name : null;
    sel.innerHTML = "";
    const o = document.createElement("option"); o.value = ""; o.textContent = "不着色 none"; sel.appendChild(o);
    for (const f of A.fields) {
      const opt = document.createElement("option"); opt.value = f.on + ":" + f.name;
      opt.textContent = f.name + (f.on === "nodes" ? "" : f.on === "members" ? " (members)" : " (elem)");
      sel.appendChild(opt);
    }
    const pref = (keep && A.byName[keep]) || A.fields.find(f => f.name === "capacity_ratio") || A.fields.find(f => f.name === "|u|");
    field = pref || null;
    sel.value = field ? field.on + ":" + field.name : "";
  }
  fillFields();
  $("field").addEventListener("change", e => { field = A.byName[e.target.value] || null; legend(); draw(); });
  function useAct(k) {
    A = decode(acts[k]);
    // a single frame (a static result) can only be amplified from the mesh
    if (A.nf < 2) { relative = false; $("rel").checked = false; }
    document.querySelectorAll("#acts button").forEach((b, i) => b.setAttribute("aria-pressed", i === k ? "true" : "false"));
    frame = 0; subtitle(); setAuto(); fillFields(); fillMonitors(); legend(); note(); draw();
  }
  function fmt(x) { const ax = Math.abs(x); return ax >= 1000 ? x.toExponential(2) : ax >= 10 ? x.toFixed(1) : ax >= 0.01 || ax === 0 ? x.toFixed(3) : x.toExponential(2); }
  function legend() {
    $("ramp").style.visibility = field ? "visible" : "hidden";
    $("t-lo").textContent = field ? fmt(field.lo) : ""; $("t-mid").textContent = field ? fmt((field.lo + field.hi) / 2) : "";
    $("t-hi").textContent = field ? (field.clipped ? "≥ " : "") + fmt(field.hi) : "";
    if (field && field.clipped) $("t-lo").textContent = "≤ " + fmt(field.lo);
  }
  legend();

  function nodePos(f, n, out) {
    // the reference the amplification works from: the mesh, or the first frame
    const pos = A.pos, o = (f * nn + n) * 3, o0 = n * 3;
    const rx = relative ? pos[o0] : nodes[n][0], ry = relative ? pos[o0 + 1] : nodes[n][1], rz = relative ? pos[o0 + 2] : nodes[n][2];
    return T(rx + amp * (pos[o] - rx), ry + amp * (pos[o + 1] - ry), rz + amp * (pos[o + 2] - rz), out);
  }
  const hidden = new Set(META.hidden || []);
  // the two particles of line item i of a part at frame f: the frame's
  // connectivity where the act carries one (only the "elements" group,
  // whose ids are the model's), the mesh's otherwise
  function ends(f, part, i) {
    return (A.connAt && part.group === "elements") ? A.connAt(f, part.owner[i]) : part.conn[i];
  }
  function nodal(f, n) {
    // a nodal field's value at node n, in 0..1 of its range: a stored byte
    // (255 = no value), or a displacement component read off the positions
    if (field.derived === undefined) { const b = field.data[f * nn + n]; return b === 255 ? null : b / 254; }
    const o = (f * nn + n) * 3, p0 = nodes[n];
    const dx = A.pos[o] - p0[0], dy = A.pos[o + 1] - p0[1], dz = A.pos[o + 2] - p0[2];
    const v = field.derived === 0 ? dx : field.derived === 1 ? dy : field.derived === 2 ? dz : Math.hypot(dx, dy, dz);
    return (v - field.lo) / Math.max(field.hi - field.lo, 1e-30);
  }
  function value(f, part, i) {
    // the colour value of item i of a part: an element field through the
    // owner, a nodal field as the mean over the item's nodes
    if (!field) return null;
    if (field.on === "nodes") {
      const c = part.kind === "line" ? ends(f, part, i) : part.conn[i]; let s = 0;
      for (const n of c) { const v = nodal(f, n); if (v === null) return null; s += v; }
      return s / c.length;
    }
    if (field.on !== part.group) return null;
    const b = field.data[f * field.n + part.owner[i]];
    return b === 255 ? null : b / 254;
  }
  const profileAt = (part, slot) => {
    // `pm` is an array even when the page has no real sections (profGeo is
    // empty and so is pm): without the profOf test every page without
    // section outlines -- every page the command line made -- stopped here,
    // black, on its first frame
    if (!part.pm || !profOf) return null;
    const k = profOf[part.owner[slot % part.conn.length]];
    return k >= 0 ? part.pm[k] : null;
  };
  function setBox(part, slot, centre, d, L, sz, colour, colourB) {
    colourB = colourB || colour;
    // a box's depth stays as vertical as its axis allows, so a deck box
    // lies flat and a tower leg keeps its faces to the bridge
    d1.copy(Math.abs(d.y) > 0.95 ? ALONG : up).addScaledVector(d, -(Math.abs(d.y) > 0.95 ? d.x : d.y)).normalize();
    d2.crossVectors(d, d1);
    const pms = profileAt(part, slot);
    if (pms) {
      // a real section, at its real size; an outboard one drawn so that
      // its +y side faces away from the model's centre (the mirrored copy
      // where the member's own +y faces inwards)
      const k = pms.length > 1 && d2.x * centre.x + d2.z * centre.z < 0 ? 1 : 0;
      m4.makeBasis(d2, dir.copy(d).multiplyScalar(L), d1);
      m4.setPosition(centre);
      pms[k].setMatrixAt(slot, m4); pms[k].setColorAt(slot, colour); setB(pms[k], slot, colourB);
      m4.makeScale(0, 0, 0); part.mesh.setMatrixAt(slot, m4);
      if (pms.length > 1) pms[1 - k].setMatrixAt(slot, m4);
      return;
    }
    m4.makeBasis(d2.multiplyScalar(sz ? sz[1] : thickness), dir.copy(d).multiplyScalar(L), d1.multiplyScalar(sz ? sz[0] : thickness));
    m4.setPosition(centre);
    part.mesh.setMatrixAt(slot, m4);
    part.mesh.setColorAt(slot, colour); setB(part.mesh, slot, colourB);
  }
  const hideSlot = (part, slot) => {
    m4.makeScale(0, 0, 0); part.mesh.setMatrixAt(slot, m4);
    const pms = profileAt(part, slot); if (pms) for (const m of pms) m.setMatrixAt(slot, m4);
  };
  const sDir = new THREE.Vector3(), sMid = new THREE.Vector3(), nA = new THREE.Vector3(), nB = new THREE.Vector3();
  function stubAt(f, part, e, n, other, failed, out) {
    // the direction a broken member's stub keeps at node n: that of an
    // unbroken neighbour there (the piece it is still welded to), signed
    // towards where the other end used to be; an isolated node's fragment
    // lies along the model's x axis
    const p0 = nodes[n], q0 = nodes[other];
    for (const e2 of part.adj[n] || []) {
      if (e2 === e || (failed && failed.data[f * failed.n + part.owner[e2]] > 127)) continue;
      const c2 = ends(f, part, e2); nodePos(f, c2[0], nA); nodePos(f, c2[1], nB);
      out.subVectors(nB, nA);
      if (out.lengthSq() < 1e-18) continue;
      out.normalize();
      const s = out.x * (q0[0] - p0[0]) + out.z * -(q0[1] - p0[1]) + out.y * (q0[2] - p0[2]);
      if (s < 0) out.negate();
      return out;
    }
    return out.copy(ALONG).multiplyScalar(q0[0] >= p0[0] ? 1 : -1);
  }
  function colourOf(f, part, i, out) {
    const v = value(f, part, i);
    return v === null ? out.copy(cBase) : turbo(v, out);
  }
  let smooth = true;
  $("smooth").addEventListener("change", e => { smooth = e.target.checked; draw(); });
  // an element field averaged onto the nodes, per kind of member (lines) or
  // over the part (faces), from the frame's own connectivity -- so the two
  // sides of a break are not blended
  function nodeAverage(f, part, skip) {
    const K = part.mkind ? part.nk : 1;
    if (!part.S || part.S.length !== nn * K) { part.S = new Float32Array(nn * K); part.Sn = new Uint16Array(nn * K); }
    const S = part.S, Sn = part.Sn; S.fill(0); Sn.fill(0);
    for (let i = 0; i < part.conn.length; i++) {
      if (skip && skip(i)) continue;
      const v = value(f, part, i);
      if (v === null) continue;
      const k = part.mkind ? part.mkind[i] : 0;
      for (const n of (part.mkind ? ends(f, part, i) : part.conn[i])) { S[n * K + k] += v; Sn[n * K + k]++; }
    }
  }
  function endColour(f, part, i, n, out) {
    if (field.on === "nodes") { const v = nodal(f, n); return v === null ? out.copy(cBase) : turbo(v, out); }
    const K = part.mkind ? part.nk : 1, o = n * K + (part.mkind ? part.mkind[i] : 0);
    return part.Sn[o] ? turbo(part.S[o] / part.Sn[o], out) : colourOf(f, part, i, out);
  }
  const col2 = new THREE.Color();

  function draw() {
    const f = frame;
    let broken = 0;
    for (const part of parts) {
      const failed = failedOf(part.group);
      if (part.kind === "line") {
        const N = part.conn.length;
        const cont = smooth && field && (field.on === "nodes" || field.on === part.group);
        if (cont && field.on !== "nodes") nodeAverage(f, part, i =>
          (part.group === "elements" && hidden.has(part.owner[i])) || (failed && failed.data[f * failed.n + part.owner[i]] > 127));
        for (let i = 0; i < N; i++) {
          const j = N + i;
          const gone = failed && failed.data[f * failed.n + part.owner[i]] > 127;
          if (gone) broken++;
          const dropped = part.group === "elements" && A.dropped[part.owner[i]] !== undefined && A.dropped[part.owner[i]] <= A.times[f];
          if ((part.group === "elements" && hidden.has(part.owner[i])) || (gone && hideGone) || dropped) { hideSlot(part, i); hideSlot(part, j); continue; }
          const ci = ends(f, part, i);
          nodePos(f, ci[0], a); nodePos(f, ci[1], b);
          mid.addVectors(a, b).multiplyScalar(0.5); dir.subVectors(b, a);
          const L = dir.length();
          const sz = sizes && part.group === "elements" && sizes[part.owner[i]][0] > 0 ? sizes[part.owner[i]] : null;
          if (!gone || L <= 1.15 * part.L0[i]) {
            // whole, or broken with its ends still together: one box
            hideSlot(part, j);
            if (L < 1e-9) { hideSlot(part, i); continue; }
            if (gone) setBox(part, i, mid, dir.divideScalar(L), L, sz, cGone);
            else if (cont) setBox(part, i, mid, dir.divideScalar(L), L, sz, endColour(f, part, i, ci[0], col), endColour(f, part, i, ci[1], col2));
            else setBox(part, i, mid, dir.divideScalar(L), L, sz, colourOf(f, part, i, col));
          } else if (!sz || sz[2] > 0) {
            // parted: a snapped cable is gone, and so is a member whose
            // section the page does not know
            hideSlot(part, i); hideSlot(part, j);
          } else {
            // the ends have parted: half a box stays on each node
            const [na, nb] = ci;
            stubAt(f, part, i, na, nb, failed, sDir);
            sMid.copy(a).addScaledVector(sDir, part.L0[i] / 4);
            setBox(part, i, sMid, sDir, part.L0[i] / 2, sz, cGone);
            stubAt(f, part, i, nb, na, failed, sDir);
            sMid.copy(b).addScaledVector(sDir, part.L0[i] / 4);
            setBox(part, j, sMid, sDir, part.L0[i] / 2, sz, cGone);
          }
        }
        for (const m of [part.mesh].concat(...(part.pm || []).filter(Boolean))) {
          if (!m) continue;
          m.instanceMatrix.needsUpdate = true;
          if (m.instanceColor) m.instanceColor.needsUpdate = true;
          m.geometry.attributes.instanceColorB.needsUpdate = true;
        }
      } else {
        const P = part.P, C = part.C;
        const contF = smooth && field && field.on === part.group;
        if (contF) nodeAverage(f, part, i => failed && failed.data[f * failed.n + part.owner[i]] > 127);
        for (let t = 0; t < part.tris.length; t++) {
          const tri = part.tris[t], i = tri[3];
          const gone = failed && failed.data[f * failed.n + part.owner[i]] > 127;
          for (let k = 0; k < 3; k++) {
            if (gone && hideGone) { P.fill(0, t * 9 + k * 3, t * 9 + k * 3 + 3); continue; }
            nodePos(f, tri[k], a);
            P[t * 9 + k * 3] = a.x; P[t * 9 + k * 3 + 1] = a.y; P[t * 9 + k * 3 + 2] = a.z;
            let c;
            if (gone) c = cGone;
            else if (field && field.on === "nodes") { const v = nodal(f, tri[k]); c = v === null ? col.copy(cBase) : turbo(v, col); }
            else if (contF) c = endColour(f, part, i, tri[k], col);
            else c = colourOf(f, part, i, col);
            C[t * 9 + k * 3] = c.r; C[t * 9 + k * 3 + 1] = c.g; C[t * 9 + k * 3 + 2] = c.b;
          }
        }
        let e = 0;
        for (let i = 0; i < part.conn.length; i++) {
          const c = part.conn[i];
          for (let k = 0; k < c.length; k++) {
            nodePos(f, c[k], a); nodePos(f, c[(k + 1) % c.length], b);
            part.E.set([a.x, a.y, a.z, b.x, b.y, b.z], e); e += 6;
          }
        }
        part.mesh.geometry.attributes.position.needsUpdate = true;
        part.mesh.geometry.attributes.color.needsUpdate = true;
        part.mesh.geometry.computeVertexNormals();
        part.edges.geometry.attributes.position.needsUpdate = true;
      }
    }
    const t = A.times[f];
    $("r-t").textContent = fText(f, 4); $("time").textContent = fText(f, 3);
    $("r-f").textContent = `${f + 1} / ${A.nf}`;
    const nb = A.failures.filter(x => x[0] <= t && x[3] < 2).length;
    $("r-n").textContent = nb; $("r-n").classList.toggle("hot", nb > 0);
    $("scrub").value = Math.round(1000 * f / Math.max(A.nf - 1, 1));
    drawChart(t);
  }

  // ------------------------------------------------------------- chart
  let mon = null;
  function fillMonitors() {
    const sel = $("mon"), keep = mon; sel.innerHTML = "";
    const names = Object.keys(A.monitors).filter(k => k !== "time" && k !== "kinetic_energy");
    if (A.monitors.kinetic_energy) names.push("kinetic_energy");
    for (const m of names) { const o = document.createElement("option"); o.value = m; o.textContent = m; sel.appendChild(o); }
    mon = (keep && A.monitors[keep]) ? keep : (names[0] || null);
    if (mon) sel.value = mon;
  }
  fillMonitors();
  $("mon").addEventListener("change", e => { mon = e.target.value; drawChart(A.times[frame]); });
  const cx = $("chart").getContext("2d");
  function drawChart(tNow) {
    const cv = $("chart"), dpr = Math.min(devicePixelRatio, 2);
    const W = cv.clientWidth * dpr, H = cv.clientHeight * dpr;
    if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
    cx.clearRect(0, 0, W, H);
    const tm = A.monitors.time || A.times, tEnd = A.times[A.nf - 1] || 1;
    const X = t => 8 * dpr + (W - 16 * dpr) * t / tEnd;
    // on a load axis every curve runs along its own record index
    const nm = (mon && A.monitors[mon]) ? A.monitors[mon].length : A.nf;
    const XI = i => 8 * dpr + (W - 16 * dpr) * i / Math.max(nm - 1, 1);
    const xAt = i => isLoad() ? XI(i) : X(tm[i]);
    const firstAt = t => { for (let i = 0; i < tm.length; i++) if (tm[i] >= t - 1e-9) return i; return tm.length - 1; };
    if (mon && A.monitors[mon]) {
      const y = A.monitors[mon];
      let lo = Infinity, hi = -Infinity;
      for (const v of y) { if (isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
      if (hi <= lo) { hi = lo + 1; }
      const Y = v => H - 14 * dpr - (H - 28 * dpr) * (v - lo) / (hi - lo);
      cx.strokeStyle = "#5fd3c8"; cx.lineWidth = 1.5 * dpr; cx.beginPath();
      for (let i = 0; i < y.length; i++) { const px = xAt(i), py = Y(y[i]); i ? cx.lineTo(px, py) : cx.moveTo(px, py); }
      cx.stroke();
      cx.fillStyle = "#93a4ad"; cx.font = `${10 * dpr}px ui-monospace, monospace`;
      cx.fillText(`${mon}  ${fmt(hi)}`, 8 * dpr, 10 * dpr);
      cx.fillText(fmt(lo), 8 * dpr, H - 3 * dpr);
    }
    cx.strokeStyle = "#ff6a4d"; cx.lineWidth = 1 * dpr;
    for (const ev of A.failures) { if (ev[3] >= 2) continue; const px = isLoad() ? XI(firstAt(ev[0])) : X(ev[0]); cx.beginPath(); cx.moveTo(px, H - 10 * dpr); cx.lineTo(px, H - 2 * dpr); cx.stroke(); }
    const now = !isLoad() ? X(tNow) : A.frame_mon ? XI(A.frame_mon[frame]) : 8 * dpr + (W - 16 * dpr) * frame / Math.max(A.nf - 1, 1);
    cx.strokeStyle = "#ffb347"; cx.lineWidth = 1.5 * dpr; cx.beginPath(); cx.moveTo(now, 2 * dpr); cx.lineTo(now, H - 2 * dpr); cx.stroke();
  }

  // ---------------------------------------------------------- controls
  $("play").addEventListener("click", () => { playing = !playing; $("play").textContent = playing ? "暂停" : "播放"; });
  $("scrub").addEventListener("input", e => { frame = Math.round((A.nf - 1) * e.target.value / 1000); draw(); });
  addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); drawChart(A.times[frame]); });
  function note() {
    // cause 2: a massless connector removed because what it
    // connected has gone -- not a break, so neither counted nor ticked
    // a connector that broke and was later removed (its joint parked) is
    // one break, not also one removal
    const real = A.failures.filter(x => x[3] < 2).length;
    const broke = new Set(A.failures.filter(x => x[3] < 2).map(x => x[1] | 0));
    const dev = new Set(A.failures.filter(x => x[3] >= 2 && !broke.has(x[1] | 0)).map(x => x[1] | 0)).size;
    $("note").textContent = A.failures.length
      ? `共 ${real} 次断裂（刻度线）${dev ? `，另有 ${dev} 根无质量连接件随之移除` : ""}；灰色是已断构件。`
      : "拖动旋转，滚轮缩放，右键平移。";
  }
  note();
  let last = performance.now(), acc = 0;
  function loop(now) {
    requestAnimationFrame(loop);
    const dt = (now - last) / 1000; last = now;
    const seconds = Math.max(4, Math.min(30, isLoad() ? A.nf / 10 : (A.times[A.nf - 1] || 1)));   // one replay lasts 4..30 s of wall time at 1x
    if (playing && A.nf > 1) {
      acc += dt * speed * (A.nf - 1) / seconds;
      if (acc >= 1) { frame = (frame + Math.floor(acc)) % A.nf; acc -= Math.floor(acc); draw(); }
    }
    controls.update();
    renderer.clear();
    renderer.render(scene, camera);
    drawTriad();
  }
  draw();
  requestAnimationFrame(loop);
})();
</script>
</body>
</html>
'''
