"""The page `vfife view model.yaml` writes (post/model_view.py):
one HTML file, the model embedded as base64, three.js r128 from a CDN (or
written in with --scripts). Three placeholders: __TITLE__, __META__ (the
JSON model_view builds, with `index` naming each array in the blob) and
__BLOB__.

The members are boxes re-sized at each frame to a width in pixels (a WebGL
line is one pixel wide on most systems); the supports', connectors' and soil
springs' symbols are drawn on a 2D canvas over the scene, a fixed size in
pixels: one symbol each with a short code, the one clicked in full."""

PAGE = r'''<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · vfife model</title>
<style>
  :root { --bg: #0d1418; --panel: rgba(16, 24, 29, 0.88); --edge: #2a3a44; --ink: #e6ecef; --dim: #93a4ad;
          --accent: #ffb347; --new: #ff6a4d; --mono: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
          --body: system-ui, -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; }
  html, body { height: 100%; margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 var(--body); overflow: hidden; }
  canvas { display: block; }
  .panel { position: absolute; z-index: 2; background: var(--panel); border: 1px solid var(--edge); border-radius: 6px; backdrop-filter: blur(6px); }
  #head { top: 16px; left: 16px; padding: 12px 16px; width: min(390px, calc(100vw - 32px)); max-height: calc(100vh - 140px); overflow: auto; }
  #head h1 { font: 600 18px/1.25 var(--body); margin: 0 0 4px; }
  .sub { color: var(--dim); font: 12px var(--mono); }
  .row { display: flex; align-items: center; gap: 8px; margin-top: 8px; flex-wrap: wrap; }
  .row label { color: var(--dim); font-size: 12px; }
  select { font: 12px var(--mono); background: #101a20; color: var(--ink); border: 1px solid var(--edge); border-radius: 4px; padding: 3px 4px; max-width: 220px; }
  .btn { font: 500 13px var(--body); color: var(--ink); background: transparent; border: 1px solid var(--edge); border-radius: 4px; padding: 4px 10px; cursor: pointer; }
  .btn:hover { border-color: var(--accent); }
  #legend { margin-top: 10px; font: 12px var(--mono); display: grid; grid-template-columns: 14px 1fr; gap: 3px 8px; align-items: center; }
  #legend i { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
  #step { margin-top: 10px; font: 12px var(--mono); color: var(--ink); }
  #step .k { color: var(--dim); }
  #step ul { margin: 4px 0 0 16px; padding: 0; color: var(--ink); }
  #glyphs { position: absolute; inset: 0; pointer-events: none; z-index: 1; }
  #legend2 { margin-top: 8px; display: block; max-width: 100%; }
  #pick { top: 16px; right: 16px; padding: 10px 14px; width: min(360px, calc(100vw - 32px)); font: 12px var(--mono); display: none; }
  #pick .k { color: var(--dim); }
  #foot { left: 16px; right: 16px; bottom: 16px; padding: 8px 14px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  #scrub { flex: 1; min-width: 160px; accent-color: var(--accent); }
  #stepname { font: 13px var(--mono); min-width: 120px; }
  @media (max-width: 720px) { #pick { top: auto; bottom: 70px; } }
</style>
</head>
<body>
<div class="panel" id="head">
  <h1>__TITLE__</h1>
  <div class="sub" id="sub"></div>
  <div class="row"><label>着色 colour by</label>
    <select id="colour"><option value="kind">构件类型 kind</option><option value="section">截面 section</option>
      <option value="step">建造步 build step</option><option value="region">区域 region</option></select></div>
  <div class="row">
    <label><input type="checkbox" id="ghost" checked> 未建构件淡显 ghosts</label>
    <label><input type="checkbox" id="arms" checked> 刚臂 arms</label>
    <label><input type="checkbox" id="dots" checked> 节点 nodes</label>
  </div>
  <div class="row">
    <label><input type="checkbox" id="sup" checked> 支承 supports</label>
    <label><input type="checkbox" id="cn" checked> 连接器 connectors</label>
    <label><input type="checkbox" id="soil" checked> 土弹簧 soil</label>
    <label><input type="checkbox" id="codes" checked> 代码 codes</label>
  </div>
  <div class="row"><label>符号大小 symbol size</label><input id="size" type="range" min="0.6" max="2.5" step="0.1" value="1"></div>
  <div class="row"><label>区域 region</label><select id="region"><option value="-1">全部 all</option></select></div>
  <div class="row">
    <label><input type="checkbox" id="loads" checked> 荷载 loads</label>
    <label><input type="checkbox" id="axes" checked> 坐标轴 axes</label>
  </div>
  <div id="legend"></div>
  <canvas id="legend2" width="350" height="196"></canvas>
  <div id="step"></div>
</div>
<canvas id="glyphs"></canvas>
<div class="panel" id="pick"></div>
<div class="panel" id="foot">
  <button class="btn" id="prev">◀</button>
  <input id="scrub" type="range" min="0" max="0" value="0" step="1" aria-label="step">
  <button class="btn" id="next">▶</button>
  <span id="stepname">–</span>
  <button class="btn" id="view">换视角 view</button>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function main() {
  const M = __META__;
  const B64 = "__BLOB__";
  const $ = id => document.getElementById(id);
  const raw = Uint8Array.from(atob(B64), c => c.charCodeAt(0)).buffer;
  const TYPES = { float32: Float32Array, int32: Int32Array, int8: Int8Array, uint8: Uint8Array };
  const A = {};
  for (const [name, [dt, off, n]] of Object.entries(M.index)) A[name] = new TYPES[dt](raw, off, n);
  const nn = M.n_nodes, ne = M.n_members, ns = M.steps.length;
  // beam, cable/bar, tendon, rigid arm, connector
  const KIND_COLOUR = [0x9fc3dc, 0xf5d76e, 0xc98a3a, 0xff5a5a, 0x7f9a8a];
  const KIND_PX = [2, 1.5, 1, 1.5, 1];                 // line widths in pixels (--thickness: the beams')
  const PXS = (M.thickness || 2) / 2;
  const NEW = new THREE.Color(0xff4fd8);
  // the symbols' colours
  const C = { fix: "#ff5c5c", moving: "#4dd8ff", spring: "#5fd68a", curve: "#d6e04a", rot: "#ffa043",
              damper: "#4da3ff", contact: "#b48cff", soil: "#c8a46a", conn: "#8a9ba4", ink: "#e6ecef",
              rotfix: "#e070ff", dim: "#5d6d75" };

  // ---- scene
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  document.body.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0d1418);
  const D = M.diag || 1;
  const camera = new THREE.PerspectiveCamera(40, innerWidth / innerHeight, D * 1e-4, D * 50);
  camera.up.set(0, 0, 1);                       // (before the controls: they orbit about the `up` they find)
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  const views = [[0.6, -1.2, 0.7], [0, -1, 0.05], [1, 0, 0.05], [0, -0.001, 1]];
  let viewIndex = 0;
  const G = $("glyphs"), g = G.getContext("2d");
  function fit() {
    // the model to the right of the panel on a wide screen
    const shift = innerWidth > 900 ? Math.min(220, innerWidth * 0.17) : 0;
    camera.setViewOffset(innerWidth, innerHeight, -shift, 0, innerWidth, innerHeight);
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    const r = Math.min(devicePixelRatio, 2);
    G.width = innerWidth * r; G.height = innerHeight * r;
    G.style.width = innerWidth + "px"; G.style.height = innerHeight + "px";
    g.setTransform(r, 0, 0, r, 0, 0);
  }
  fit();
  function setView(k) {
    const v = views[k % views.length];
    camera.position.set(v[0] * D * 0.9, v[1] * D * 0.9, v[2] * D * 0.9);
    controls.target.set(0, 0, 0); controls.update();
  }
  setView(0);
  $("view").addEventListener("click", () => setView(++viewIndex));
  const axes = new THREE.AxesHelper(D * 0.08); scene.add(axes);
  $("axes").addEventListener("change", e => { axes.visible = e.target.checked; render(); });

  // ---- members: lines of a fixed width in pixels. A WebGL line is one pixel
  // wide on most systems, so each member is a box whose cross-section the
  // vertex shader sizes at each vertex to the pixels wanted at that depth: a
  // long member keeps its width from end to end, close up or far
  const U = { uWpp: { value: 1 }, uNear: { value: camera.near } };
  function widthInPixels(mat) {
    mat.onBeforeCompile = sh => {
      sh.uniforms.uWpp = U.uWpp; sh.uniforms.uNear = U.uNear;
      sh.vertexShader = "attribute float aPx;\nuniform float uWpp;\nuniform float uNear;\n" + sh.vertexShader.replace(
        "#include <project_vertex>",
        ["vec4 cl = instanceMatrix * vec4( transformed.x, 0.0, 0.0, 1.0 );",
         "vec4 side = instanceMatrix * vec4( 0.0, transformed.y, transformed.z, 0.0 );",
         "vec4 mvC = modelViewMatrix * cl;",
         "vec4 mvPosition = modelViewMatrix * ( cl + side * ( max( -mvC.z, uNear ) * uWpp * aPx ) );",
         "gl_Position = projectionMatrix * mvPosition;"].join("\n"));
    };
    return mat;
  }
  const pxOf = new Float32Array(ne), pxGhost = new Float32Array(ne).fill(1);
  const boxS = new THREE.BoxGeometry(1, 1, 1), boxG = boxS.clone();
  boxS.setAttribute("aPx", new THREE.InstancedBufferAttribute(pxOf, 1));
  boxG.setAttribute("aPx", new THREE.InstancedBufferAttribute(pxGhost, 1));
  const solid = new THREE.InstancedMesh(boxS, widthInPixels(new THREE.MeshBasicMaterial()), ne);
  const ghost = new THREE.InstancedMesh(boxG, widthInPixels(new THREE.MeshBasicMaterial({ color: 0x9aa7ad, transparent: true, opacity: 0.16, depthWrite: false })), ne);
  solid.frustumCulled = ghost.frustumCulled = false;
  scene.add(solid); scene.add(ghost);
  const m4 = new THREE.Matrix4(), s3 = new THREE.Vector3(), p3 = new THREE.Vector3();
  const X = new THREE.Vector3(1, 0, 0), dvec = new THREE.Vector3(), ZERO = new THREE.Matrix4().makeScale(0, 0, 0);
  const node = i => new THREE.Vector3(A.nodes[3 * i], A.nodes[3 * i + 1], A.nodes[3 * i + 2]);
  const builtAt = (e, k) => A.built[e * ns + k] === 1;
  const isNew = (e, k) => builtAt(e, k) && !(k ? builtAt(e, k - 1) : A.built0[e] === 1);
  const mids = new Float32Array(3 * ne), placed = [];
  for (let e = 0; e < ne; e++) {
    const a = node(A.conn[2 * e]), b = node(A.conn[2 * e + 1]);
    dvec.subVectors(b, a);
    const q = new THREE.Quaternion().setFromUnitVectors(X, dvec.clone().normalize());
    p3.addVectors(a, b).multiplyScalar(0.5);
    mids[3 * e] = p3.x; mids[3 * e + 1] = p3.y; mids[3 * e + 2] = p3.z;
    placed.push(new THREE.Matrix4().compose(p3.clone(), q, new THREE.Vector3(Math.max(dvec.length(), 1e-9), 1, 1)));
    pxOf[e] = KIND_PX[A.kind[e]] * PXS;
  }
  const shown = new Uint8Array(ne);                  // 1 built, 2 a ghost, 0 neither
  const camDir = new THREE.Vector3();
  let wpp0 = 1;                                      // world length of a pixel at unit depth
  const depthOf = (x, y, z) => Math.max((x - camera.position.x) * camDir.x + (y - camera.position.y) * camDir.y
                                         + (z - camera.position.z) * camDir.z, camera.near);
  function place() {
    camera.getWorldDirection(camDir);
    wpp0 = 2 * Math.tan(camera.fov * Math.PI / 360) / innerHeight;
    U.uWpp.value = wpp0;
  }

  // ---- nodes: dots of a fixed size, those of the beams, bars and connectors
  // standing in the step (a tendon's own nodes are not drawn)
  const dotTex = (() => { const c = document.createElement("canvas"); c.width = c.height = 32;
    const x = c.getContext("2d"); x.fillStyle = "#fff"; x.beginPath(); x.arc(16, 16, 12, 0, 2 * Math.PI); x.fill();
    return new THREE.CanvasTexture(c); })();
  const dotGeo = new THREE.BufferGeometry();
  dotGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(3 * Math.max(nn, 1)), 3));
  const dots = new THREE.Points(dotGeo, new THREE.PointsMaterial({ size: 4, sizeAttenuation: false, map: dotTex,
    transparent: true, alphaTest: 0.5, color: 0xffffff, depthTest: false }));
  scene.add(dots);
  const standing = new Uint8Array(nn), dotted = new Uint8Array(nn);

  // ---- colours
  function hsl(i, out) { return (out || new THREE.Color()).setHSL(((i * 0.618034) % 1), 0.55, 0.58); }
  const stepHue = k => 0.66 - 0.66 * ((k + 1) / Math.max(ns, 1));
  const GREY = new THREE.Color(0x666666), work = new THREE.Color();
  function colourOf(mode, e) {                       // (one Color reused: called for every member at each redraw)
    if (mode === "kind") return work.set(KIND_COLOUR[A.kind[e]]);
    if (mode === "section") return A.sec[e] < 0 ? GREY : hsl(A.sec[e], work);
    if (mode === "region") return A.reg[e] < 0 ? GREY : hsl(A.reg[e], work);
    return work.setHSL(stepHue(Math.min(A.born[e], ns - 1)), 0.7, 0.55);
  }
  let mode = "kind";

  // ---- the symbols, drawn on a 2D layer over the scene, a fixed size in pixels
  const CN = A.cn_ids.length, SN = A.soil_nodes.length;
  const cnOf = new Int32Array(ne).fill(-1); for (let c = 0; c < CN; c++) cnOf[A.cn_ids[c]] = c;
  const soilOf = new Map(); for (let k = 0; k < SN; k++) soilOf.set(A.soil_nodes[k], (soilOf.get(A.soil_nodes[k]) || 0) | A.soil_axes[k]);
  const supRow = new Map(); for (let k = 0; k < A.sup_nodes.length; k++) supRow.set(A.sup_nodes[k], k);
  const pv = new THREE.Vector3();
  function scr(x, y, z) {
    pv.set(x, y, z).project(camera);
    if (pv.z > 1 || pv.z < -1) return null;
    return [(pv.x + 1) / 2 * innerWidth, (1 - pv.y) / 2 * innerHeight];
  }
  const pt = (x, y, z) => { const s = scr(x, y, z); return s && { x, y, z, s, w: wpp0 * depthOf(x, y, z) }; };
  // the screen vector of a pixel's length along world direction a at P (foreshortened)
  function axv(P, a, sgn) {
    const t = P.w * (sgn || 1), q = scr(P.x + a[0] * t, P.y + a[1] * t, P.z + a[2] * t);
    return q ? [q[0] - P.s[0], q[1] - P.s[1]] : [0, 0];
  }
  const nrm = v => { const l = Math.hypot(v[0], v[1]) || 1; return [v[0] / l, v[1] / l]; };
  const at = (p, d, t) => [p[0] + d[0] * t, p[1] + d[1] * t];
  function seg(c, a, b, col, w, dash) { c.strokeStyle = col; c.lineWidth = w; c.setLineDash(dash || []);
    c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.stroke(); }
  function pl(c, pts, col, w, dash, close) { c.strokeStyle = col; c.lineWidth = w; c.setLineDash(dash || []);
    c.beginPath(); pts.forEach((p, i) => i ? c.lineTo(p[0], p[1]) : c.moveTo(p[0], p[1])); if (close) c.closePath(); c.stroke(); }
  function hatch(c, p, d, col) {
    const n = [-d[1], d[0]], a = at(p, n, -8), b = at(p, n, 8);
    seg(c, a, b, col, 1.6);
    for (let k = 0; k < 4; k++) { const q = at(a, [b[0] - a[0], b[1] - a[1]], (k + 0.5) / 4);
      seg(c, q, [q[0] + d[0] * 5 - n[0] * 4, q[1] + d[1] * 5 - n[1] * 4], col, 1); }
  }
  function zig(c, a, b, col, n, amp, w) {
    n = n || 4; amp = amp || 4.2;
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]), d = nrm([b[0] - a[0], b[1] - a[1]]), q = [-d[1], d[0]];
    const pts = [a, at(a, d, 4)];
    for (let k = 0; k < 2 * n; k++) { const t = 4 + (L - 8) * (k + 0.5) / (2 * n), s = k % 2 ? -1 : 1;
      pts.push([a[0] + d[0] * t + q[0] * amp * s, a[1] + d[1] * t + q[1] * amp * s]); }
    pts.push(at(a, d, L - 4), b); pl(c, pts, col, w || 1.7);
  }
  function pot(c, a, b, col) {
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]), d = nrm([b[0] - a[0], b[1] - a[1]]), q = [-d[1], d[0]];
    const m0 = at(a, d, L * 0.3), m1 = at(a, d, L * 0.68), p = at(a, d, L * 0.45);
    seg(c, a, p, col, 1.6); seg(c, at(p, q, -4), at(p, q, 4), col, 2);
    pl(c, [at(m0, q, -6), at(m1, q, -6), at(m1, q, 6), at(m0, q, 6)], col, 1.6); seg(c, m1, b, col, 1.6);
  }
  function gap(c, a, b, col) {                       // free travel, then a stop
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]), d = nrm([b[0] - a[0], b[1] - a[1]]), q = [-d[1], d[0]];
    const m0 = at(a, d, L * 0.42), m1 = at(a, d, L * 0.58);
    seg(c, a, m0, col, 1.6); seg(c, m1, b, col, 1.6);
    seg(c, at(m0, q, -6), at(m0, q, 6), col, 2.2); seg(c, at(m1, q, -6), at(m1, q, 6), col, 2.2);
  }
  function slide(c, a, b, col) {                     // a block on a track: friction, a pendulum
    const L = Math.hypot(b[0] - a[0], b[1] - a[1]), d = nrm([b[0] - a[0], b[1] - a[1]]), q = [-d[1], d[0]];
    const m = at(a, d, L / 2);
    seg(c, a, b, col, 1.2);
    pl(c, [at(at(m, d, -7), q, -2), at(at(m, d, 7), q, -2), at(at(m, d, 7), q, -9), at(at(m, d, -7), q, -9)], col, 1.8, null, true);
  }
  const ellipse = (cen, u, v, r, a0, a1, n, r1) => { const pts = [];
    for (let k = 0; k <= n; k++) { const t = a0 + (a1 - a0) * k / n, rr = r1 === undefined ? r : r + (r1 - r) * k / n;
      pts.push([cen[0] + rr * (u[0] * Math.cos(t) + v[0] * Math.sin(t)), cen[1] + rr * (u[1] * Math.cos(t) + v[1] * Math.sin(t))]); }
    return pts; };
  function ringFix(c, cen, u, v, col, r) {           // a rotation held: a ring about the axis, a lock across it
    pl(c, ellipse(cen, u, v, r, 0, 2 * Math.PI, 40), col, 2.4);
    const p = ellipse(cen, u, v, r, 0, 0, 1)[0], d = nrm([p[0] - cen[0], p[1] - cen[1]]);
    seg(c, at(p, d, -4), at(p, d, 6), col, 2.4);
  }
  function spiral(c, cen, u, v, col, r, dash) { pl(c, ellipse(cen, u, v, 2.5, 0, 2.4 * Math.PI, 56, r), col, 1.8, dash); }
  function ringPot(c, cen, u, v, col, r) {
    pl(c, ellipse(cen, u, v, r, 0.35, 2 * Math.PI - 0.35, 36), col, 1.6, [3, 2]);
    const p = ellipse(cen, u, v, r, 0, 0, 1)[0]; c.setLineDash([]); c.strokeStyle = col; c.lineWidth = 1.6;
    c.strokeRect(p[0] - 4, p[1] - 4, 8, 8);
  }
  function hollow(c, p, col) { c.setLineDash([]); c.fillStyle = "#0d1418"; c.strokeStyle = col; c.lineWidth = 1.4;
    c.beginPath(); c.arc(p[0], p[1], 3, 0, 2 * Math.PI); c.fill(); c.stroke(); }
  // one translational component between a and b: what carries it, a damper beside it
  function component(c, a, b, base, damp, offset) {
    const d = nrm([b[0] - a[0], b[1] - a[1]]), q = [-d[1], d[0]], sh = base && damp ? offset : 0;
    const a1 = at(a, q, sh), b1 = at(b, q, sh);
    if (base === 1) zig(c, a1, b1, C.spring, 4, 4.2);
    else if (base === 2) zig(c, a1, b1, C.curve, 4, 4.2);
    else if (base === 3) gap(c, a1, b1, C.contact);
    else if (base === 4) slide(c, a1, b1, C.contact);
    if (damp) pot(c, at(a, q, -sh), at(b, q, -sh), C.damper);
    if (base && damp) { seg(c, at(a, q, -sh), at(a, q, sh), C.spring, 1.6); seg(c, at(b, q, -sh), at(b, q, sh), C.spring, 1.6); }
  }
  function rotation(c, cen, u, v, base, damp) {
    if (base) spiral(c, cen, u, v, base === 2 ? C.curve : base === 1 ? C.rot : C.contact, 12, base > 2 ? [3, 2] : null);
    if (damp) ringPot(c, cen, u, v, C.damper, base ? 17 : 11);
  }
  const AX = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const SL = 46;
  // which way a symbol points: away from the middle of the model (z: down)
  const out = (P, a) => (a[2] > 0.9 ? -1 : ((P.x * a[0] + P.y * a[1] + P.z * a[2]) < 0 ? -1 : 1));
  const plane = (P, R, k) => [axv(P, R[(k + 1) % 3]), axv(P, R[(k + 2) % 3])];
  function drawNode(P, mask, moving, soil) {
    // each held dof: red, cyan where it moves (a prescribed motion)
    if (mask) {
      for (let a = 0; a < 3; a++) {
        const t = (mask >> a) & 1, r = (mask >> (a + 3)) & 1;
        if (!t && !r) continue;
        const ct = (moving >> a) & 1 ? C.moving : C.fix, cr = (moving >> (a + 3)) & 1 ? C.moving : C.fix;
        const d = nrm(axv(P, AX[a], out(P, AX[a]))), e = at(P.s, d, SL);
        seg(g, P.s, e, t ? ct : cr, t ? 2.6 : 1, t ? null : [3, 3]); hatch(g, e, d, t ? ct : cr);
        if (r) { const [u, v] = plane(P, AX, a); ringFix(g, at(P.s, d, SL * 0.55), u, v, cr, 12); }
      }
    }
    if (soil) {
      const ends = [];
      for (let a = 0; a < 3; a++) {
        if (!((soil >> a) & 1)) continue;
        // (the other way from a support on the same axis)
        const away = ((mask >> a) & 1) || ((mask >> (a + 3)) & 1) ? -1 : 1;
        const d = nrm(axv(P, AX[a], away * out(P, AX[a]))), e = at(P.s, d, 34);
        zig(g, P.s, e, C.soil, 3, 3.6, 1.5); hatch(g, e, d, C.soil); ends.push(Math.atan2(d[1], d[0]));
      }
      if (soil & 8 && ends.length === 2) {        // radial: one spring on the resultant of the two
        let da = ends[1] - ends[0]; da = (da + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
        const pts = []; for (let k = 0; k <= 16; k++) { const t = ends[0] + da * k / 16; pts.push([P.s[0] + 20 * Math.cos(t), P.s[1] + 20 * Math.sin(t)]); }
        pl(g, pts, C.soil, 1, [2, 2]);
      }
    }
  }
  function connAxes(c) { const r = A.cn_axes; return [0, 1, 2].map(k => [r[9 * c + 3 * k], r[9 * c + 3 * k + 1], r[9 * c + 3 * k + 2]]); }
  function drawConnector(c, Pm) {
    const e = A.cn_ids[c], i = A.conn[2 * e], j = A.conn[2 * e + 1];
    const sa = scr(A.nodes[3 * i], A.nodes[3 * i + 1], A.nodes[3 * i + 2]), sb = scr(A.nodes[3 * j], A.nodes[3 * j + 1], A.nodes[3 * j + 2]);
    if (!sa || !sb) return;
    const R = connAxes(c), ch = [A.nodes[3 * j] - A.nodes[3 * i], A.nodes[3 * j + 1] - A.nodes[3 * i + 1], A.nodes[3 * j + 2] - A.nodes[3 * i + 2]];
    const cl = Math.hypot(ch[0], ch[1], ch[2]) || 1;
    let along = -1, best = 0.99;
    for (let k = 0; k < 3; k++) { const d = Math.abs(R[k][0] * ch[0] + R[k][1] * ch[1] + R[k][2] * ch[2]) / cl; if (d > best) { best = d; along = k; } }
    if (Math.hypot(sb[0] - sa[0], sb[1] - sa[1]) < 24) along = -1;     // too short on the screen: a spoke too
    const ends = {};
    const code = k => A.cn_codes[6 * c + k];
    for (let k = 0; k < 3; k++) {
      const t = code(k), r = code(k + 3);
      if (!t && !r) continue;
      if (k === along) {
        if (t) component(g, sa, sb, t & 7, t & 8, 7);
        if (r) { const [u, v] = plane(Pm, R, k); rotation(g, Pm.s, u, v, r & 7, r & 8); }
        continue;
      }
      const d = nrm(axv(Pm, R[k], out(Pm, R[k]))), tip = at(Pm.s, d, SL);
      if (t) { component(g, Pm.s, tip, t & 7, t & 8, 6); hollow(g, tip, (t & 7) ? C.spring : C.damper); }
      if (t & 16) ends[k] = Math.atan2(d[1], d[0]);
      if (r) {
        const from = t ? tip : Pm.s, cen = at(from, d, t ? 22 : SL * 0.7);
        seg(g, from, cen, C.rot, 1, [2, 2]);
        const [u, v] = plane(Pm, R, k); rotation(g, cen, u, v, r & 7, r & 8);
      }
    }
    if (ends[1] !== undefined && ends[2] !== undefined) {   // one law in the y-z plane: the two tied
      let da = ends[2] - ends[1]; da = (da + 3 * Math.PI) % (2 * Math.PI) - Math.PI;
      const pts = []; for (let k = 0; k <= 16; k++) { const t = ends[1] + da * k / 16; pts.push([Pm.s[0] + 26 * Math.cos(t), Pm.s[1] + 26 * Math.sin(t)]); }
      pl(g, pts, C.contact, 1, [2, 2]);
    }
    seg(g, [Pm.s[0] - 1, Pm.s[1]], [Pm.s[0] + 1, Pm.s[1]], C.conn, 3);
  }
  // THE SYMBOLS, as the usual programs draw them: one symbol
  // for each support, each connector, each node with soil springs, a fixed
  // size on the screen times the size slider; what each dof or component is
  // in a short code beside it (a support's "111000": 1 held, 0 free, cyan
  // where it moves; a connector's "KK0000": K a spring, N a curve or a
  // bilinear law, G a gap, F friction or a pendulum, D a damper alone, blue
  // where a damper is beside it, 0 free), its axes as a small triad. A code
  // that would overlap one already drawn is left out (the symbol stays). The
  // one clicked is drawn in full: each held dof, each component on its axis.
  let anchors = [], counts = { symbols: 0, labels: 0, hidden: 0, detail: 0, codes: [] };
  let picked = null;                                 // {member: e} or {node: i}
  const SZ = () => +$("size").value;
  function glyphPass() {
    g.clearRect(0, 0, innerWidth, innerHeight);
    anchors = []; counts = { symbols: 0, labels: 0, hidden: 0, detail: 0, codes: [] };
    symbols();
    if (picked && picked.node !== undefined) {
      const s = scr(A.nodes[3 * picked.node], A.nodes[3 * picked.node + 1], A.nodes[3 * picked.node + 2]);
      if (s) { g.setLineDash([]); g.strokeStyle = C.ink; g.lineWidth = 1.5; g.beginPath(); g.arc(s[0], s[1], 8, 0, 2 * Math.PI); g.stroke(); }
    }
  }
  const maskOf = i => { const r = supRow.get(i); return r === undefined ? 0 : A.sup_masks[r * ns + step]; };
  const movingOf = i => { const r = supRow.get(i); return r === undefined ? 0 : A.sup_moving[r * ns + step]; };
  const CODE = ["0", "K", "N", "G", "F"];
  function supportCode(m, mv) {
    return [0, 1, 2, 3, 4, 5].map(k => [(m >> k) & 1 ? "1" : "0",
      (mv >> k) & 1 ? C.moving : (m >> k) & 1 ? (k < 3 ? C.fix : C.rotfix) : C.dim]);
  }
  // (lower case: one law on uy and uz together, in the y-z plane; a trailing *: a coupled 6x6 matrix)
  function connectorCode(c) {
    const out = [0, 1, 2, 3, 4, 5].map(k => { const t = A.cn_codes[6 * c + k], b = t & 7, d = t & 8;
      const ch = b ? CODE[b] : d ? "D" : "0";
      return [t & 16 ? ch.toLowerCase() : ch, !(t & 15) ? C.dim : d ? C.damper : b > 2 ? C.contact : b === 2 ? C.curve : C.spring]; });
    if ([0, 1, 2, 3, 4, 5].some(k => A.cn_codes[6 * c + k] & 32)) out.push(["*", C.ink]);
    return out;
  }
  // the codes: placed where they do not overlap one placed before
  let placedBoxes = null;
  function code(chars, x, y, z) {
    const w = 6.2 * z * chars.length, h = 11 * z, cell = 64;
    const box = [x, y - h, x + w, y + 2];
    const k0 = Math.floor(box[0] / cell), k1 = Math.floor(box[2] / cell), j0 = Math.floor(box[1] / cell), j1 = Math.floor(box[3] / cell);
    for (let a = k0; a <= k1; a++) for (let b = j0; b <= j1; b++)
      for (const o of placedBoxes.get(a + "," + b) || [])
        if (box[0] < o[2] && o[0] < box[2] && box[1] < o[3] && o[1] < box[3]) { counts.hidden++; return; }
    for (let a = k0; a <= k1; a++) for (let b = j0; b <= j1; b++) {
      const kk = a + "," + b; if (!placedBoxes.has(kk)) placedBoxes.set(kk, []); placedBoxes.get(kk).push(box); }
    g.font = `${Math.round(10 * z)}px ui-monospace, Menlo, Consolas, monospace`;
    chars.forEach(([ch, col], i) => { g.fillStyle = col; g.fillText(ch, x + i * 6.2 * z, y); });
    counts.labels++; counts.codes.push(chars.map(c => c[0]).join(""));
  }
  function supportSymbol(P, m, mv, z) {
    // a triangle under the node: filled where a translation is held, open
    // where only rotations are; a bar under it where a rotation is held
    const s = P.s, h = 8 * z, col = mv ? C.moving : (m & 7) ? C.fix : C.rotfix;
    g.setLineDash([]); g.strokeStyle = col; g.fillStyle = col; g.lineWidth = 1.4;
    g.beginPath(); g.moveTo(s[0], s[1]); g.lineTo(s[0] - h * 0.6, s[1] + h); g.lineTo(s[0] + h * 0.6, s[1] + h); g.closePath();
    if (m & 7) g.fill(); else g.stroke();
    if (m & 56) { g.strokeStyle = (mv & 56) ? C.moving : C.rotfix; g.lineWidth = 2 * z;
      g.beginPath(); g.moveTo(s[0] - h * 0.8, s[1] + h + 2.5 * z); g.lineTo(s[0] + h * 0.8, s[1] + h + 2.5 * z); g.stroke(); }
  }
  function soilSymbol(P, bits, z) {
    let a = 0; while (a < 3 && !((bits >> a) & 1)) a++;
    const d = nrm(axv(P, AX[a], out(P, AX[a]))), e = at(P.s, d, 16 * z);
    zig(g, P.s, e, C.soil, 2, 2.6 * z, 1.2); hatch(g, e, d, C.soil);
  }
  function connectorSymbol(c, Pm, z) {
    const e = A.cn_ids[c], i = A.conn[2 * e], j = A.conn[2 * e + 1];
    const sa = scr(A.nodes[3 * i], A.nodes[3 * i + 1], A.nodes[3 * i + 2]), sb = scr(A.nodes[3 * j], A.nodes[3 * j + 1], A.nodes[3 * j + 2]);
    const r = 3 * z;
    g.setLineDash([]); g.fillStyle = C.spring; g.strokeStyle = C.spring; g.lineWidth = 1.2;
    g.fillRect(sa[0] - r, sa[1] - r, 2 * r, 2 * r);                         // its first node: a square
    g.beginPath(); g.moveTo(sb[0], sb[1] - r * 1.2); g.lineTo(sb[0] - r, sb[1] + r); g.lineTo(sb[0] + r, sb[1] + r); g.closePath(); g.fill();  // its second: a triangle
    const R = connAxes(c);
    ["#ff6b6b", "#5fd68a", "#5aa9ff"].forEach((col, k) => {             // its axes: x red, y green, z blue
      const v = axv(Pm, R[k]); seg(g, Pm.s, [Pm.s[0] + v[0] * 14 * z, Pm.s[1] + v[1] * 14 * z], col, 1.4); });
  }
  function symbols() {
    const doSup = $("sup").checked, doCn = $("cn").checked, doSoil = $("soil").checked, doCode = $("codes").checked;
    const z = SZ(), mx = 40, inView = s => s[0] > -mx && s[1] > -mx && s[0] < innerWidth + mx && s[1] < innerHeight + mx;
    placedBoxes = new Map();
    const isPicked = a => picked && ((a.node !== undefined && picked.node === a.node) || (a.member !== undefined && picked.member === a.member));
    // soil springs, connectors, then supports on top; codes after all the symbols
    const later = [], first = [];               // (the supports' codes are placed first)
    if (doSoil) for (const [i, bits] of soilOf) {
      if (!standing[i]) continue;
      const P = pt(A.nodes[3 * i], A.nodes[3 * i + 1], A.nodes[3 * i + 2]);
      if (!P || !inView(P.s)) continue;
      const a = { node: i, P }; anchors.push(a);
      if (isPicked(a)) continue;
      soilSymbol(P, bits, z); counts.symbols++;
    }
    if (doCn) for (let c = 0; c < CN; c++) {
      const e = A.cn_ids[c]; if (shown[e] !== 1) continue;
      const i = A.conn[2 * e], j = A.conn[2 * e + 1];
      if (!scr(A.nodes[3 * i], A.nodes[3 * i + 1], A.nodes[3 * i + 2]) || !scr(A.nodes[3 * j], A.nodes[3 * j + 1], A.nodes[3 * j + 2])) continue;
      const P = pt(mids[3 * e], mids[3 * e + 1], mids[3 * e + 2]);
      if (!P || !inView(P.s)) continue;
      const a = { member: e, c, P }; anchors.push(a);
      if (isPicked(a)) { drawConnector(c, P); counts.detail++; continue; }
      connectorSymbol(c, P, z); counts.symbols++;
      if (doCode) later.push(() => code(connectorCode(c), P.s[0] + 6 * z, P.s[1] - 5 * z, z));
    }
    if (doSup) for (let k = 0; k < A.sup_nodes.length; k++) {
      const i = A.sup_nodes[k], m = A.sup_masks[k * ns + step];
      if (!m || !standing[i]) continue;
      const P = pt(A.nodes[3 * i], A.nodes[3 * i + 1], A.nodes[3 * i + 2]);
      if (!P || !inView(P.s)) continue;
      const a = { node: i, P }; anchors.push(a);
      const mv = A.sup_moving[k * ns + step];
      if (isPicked(a)) { drawNode(P, m, mv, doSoil ? soilOf.get(i) || 0 : 0); counts.detail++; continue; }
      supportSymbol(P, m, mv, z); counts.symbols++;
      if (doCode) first.push(() => code(supportCode(m, mv), P.s[0] + 7 * z, P.s[1] + 10 * z, z));
    }
    // a picked node with soil springs only
    if (picked && picked.node !== undefined && (!maskOf(picked.node) || !doSup) && soilOf.has(picked.node) && doSoil && standing[picked.node]) {
      const P = pt(A.nodes[3 * picked.node], A.nodes[3 * picked.node + 1], A.nodes[3 * picked.node + 2]);
      if (P) { drawNode(P, 0, 0, soilOf.get(picked.node)); counts.detail++; }
    }
    first.forEach(f => f()); later.forEach(f => f());
  }

  // ---- legends
  function legend() {
    const L = $("legend"); L.innerHTML = "";
    const add = (colour, text) => { const i = document.createElement("i"); i.style.background = "#" + colour.getHexString();
      const t = document.createElement("span"); t.textContent = text; L.append(i, t); };
    if (mode === "kind") {
      const count = new Array(M.kinds.length).fill(0); for (let e = 0; e < ne; e++) count[A.kind[e]]++;
      M.kinds.forEach((k, i) => { if (count[i]) add(new THREE.Color(KIND_COLOUR[i]), `${k} (${count[i]})`); });
    } else if (mode === "step") {
      const c0 = new THREE.Color(); c0.setHSL(stepHue(-1), 0.7, 0.55); add(c0, "from the start 开始即有");
      M.steps.forEach((s, k) => { const c = new THREE.Color(); c.setHSL(stepHue(k), 0.7, 0.55); if (s.built) add(c, `${k + 1}. ${s.name} (${s.built})`); });
    } else {
      const names = mode === "section" ? M.section_names : M.regions;
      const shownNames = names.slice(0, 24); shownNames.forEach((n, i) => add(hsl(i), n));
      if (names.length > 24) { const t = document.createElement("span"); t.textContent = `… ${names.length - 24} more (click a member)`; L.append(document.createElement("i"), t); }
    }
    if (mode !== "step" && M.steps[step].built) add(NEW, "built in this step 本步新建");
  }
  function symbolLegend() {
    const cv = $("legend2"), r = Math.min(devicePixelRatio, 2), W = 350, H = 196;
    cv.width = W * r; cv.height = H * r; cv.style.width = W + "px"; cv.style.height = H + "px";
    const c = cv.getContext("2d"); c.setTransform(r, 0, 0, r, 0, 0);
    const body = "12px system-ui, \"PingFang SC\", \"Microsoft YaHei\", sans-serif", mono = "11px ui-monospace, Menlo, Consolas, monospace";
    const text = (t, x, y) => { c.font = body; c.fillStyle = "#93a4ad"; c.fillText(t, x, y); };
    const codeAt = (chars, x, y) => { c.font = mono; chars.forEach(([ch, col], i) => { c.fillStyle = col; c.fillText(ch, x + i * 6.6, y); }); };
    const tri = (x, y, col, fill) => { c.setLineDash([]); c.strokeStyle = c.fillStyle = col; c.lineWidth = 1.4; c.beginPath();
      c.moveTo(x, y - 6); c.lineTo(x - 5, y + 3); c.lineTo(x + 5, y + 3); c.closePath(); fill ? c.fill() : c.stroke(); };
    let y = 14;
    tri(12, y, C.fix, true); text("支承：约束了平动", 44, y + 4);
    tri(190, y, C.rotfix, false); seg(c, [183, y + 6], [197, y + 6], C.rotfix, 2); text("只约束转动", 216, y + 4);
    y += 24; codeAt(supportCode(0b111011, 0b000001), 4, y + 4); text("1 约束 0 自由；红平动、紫转动、青给定运动", 52, y + 4);
    y += 24; c.fillStyle = C.spring; c.fillRect(4, y - 3, 6, 6); seg(c, [10, y], [30, y], C.conn, 1);
    c.beginPath(); c.moveTo(34, y - 4); c.lineTo(30, y + 3); c.lineTo(38, y + 3); c.closePath(); c.fill();
    seg(c, [20, y], [28, y - 7], "#ff6b6b", 1.4); seg(c, [20, y], [14, y - 8], "#5fd68a", 1.4); seg(c, [20, y], [20, y - 10], "#5aa9ff", 1.4);
    text("连接器：方块首节点，三角末节点，局部轴 x y z", 44, y + 4);
    y += 24; codeAt([["K", C.spring], ["K", C.damper], ["N", C.curve], ["G", C.contact], ["F", C.contact], ["D", C.damper]], 4, y + 4);
    text("K 弹簧 N 非线性 G 间隙 F 摩擦 D 阻尼 0 自由", 52, y + 4);
    y += 18; text("蓝：并有阻尼器；小写：uy uz 同一律；*：耦合", 52, y + 4);
    y += 24; zig(c, [4, y], [22, y], C.soil, 2, 3, 1.2); hatch(c, [22, y], [1, 0], C.soil); text("土弹簧", 44, y + 4);
    y += 24; text("点中支承或连接器：按每个自由度、每个分量画出", 4, y + 4);
    y += 18; text("代码会互相遮挡时不写，符号照画", 4, y + 4);
  }

  // ---- arrows: a shaft and a head per arrow, instanced
  const TA = D / 300;
  const AW = 9;                                    // [from step, to step, kind, x, y, z, vx, vy, vz]
  const na = A.arrows.length / AW;
  const shaft = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.5, 0.5, 1, 6), new THREE.MeshBasicMaterial(), Math.max(na, 1));
  const head = new THREE.InstancedMesh(new THREE.ConeGeometry(1, 1, 8), new THREE.MeshBasicMaterial(), Math.max(na, 1));
  scene.add(shaft); scene.add(head);
  const ARROW_COLOUR = [new THREE.Color(0xff5252), new THREE.Color(0xc77dff), new THREE.Color(0x4fc3f7)];
  const bigOf = [0, 0, 0];
  for (let i = 0; i < na; i++) { const o = AW * i, k = A.arrows[o + 2];
    bigOf[k] = Math.max(bigOf[k], Math.hypot(A.arrows[o + 6], A.arrows[o + 7], A.arrows[o + 8])); }
  const Y = new THREE.Vector3(0, 1, 0);
  // every instanced mesh gets its colour buffer before the first frame: one
  // made later is not in the compiled shader, and its colours never show
  const WHITE = new THREE.Color(0xffffff);
  for (const [mesh, n] of [[solid, ne], [shaft, Math.max(na, 1)], [head, Math.max(na, 1)]])
    for (let i = 0; i < n; i++) mesh.setColorAt(i, WHITE);

  let step = 0, cnStanding = 0;
  const inRegion = new Uint8Array(ne);
  function draw() {
    const showGhost = $("ghost").checked, showArms = $("arms").checked, region = +$("region").value;
    standing.fill(0); dotted.fill(0);
    inRegion.fill(region < 0 ? 1 : 0);
    if (region >= 0) for (let k = A.reg_offsets[region]; k < A.reg_offsets[region + 1]; k++) inRegion[A.reg_members[k]] = 1;
    for (let e = 0; e < ne; e++) {
      const built = builtAt(e, step);
      const hidden = !showArms && A.kind[e] === 3;
      const inside = inRegion[e] === 1;
      shown[e] = built && !hidden && inside ? 1 : ((!built || !inside) && showGhost && !hidden ? 2 : 0);
      solid.setMatrixAt(e, shown[e] === 1 ? placed[e] : ZERO); ghost.setMatrixAt(e, shown[e] === 2 ? placed[e] : ZERO);
      if (built && inside) {                         // (the arms switch hides arms only, not what stands on them)
        const a = A.conn[2 * e], b = A.conn[2 * e + 1], k = A.kind[e];
        standing[a] = standing[b] = 1;
        if (k === 0 || k === 1 || k === 4) dotted[a] = dotted[b] = 1;
      }
      solid.setColorAt(e, mode !== "step" && isNew(e, step) ? NEW : colourOf(mode, e));
    }
    if (solid.instanceColor) solid.instanceColor.needsUpdate = true;
    solid.instanceMatrix.needsUpdate = true; ghost.instanceMatrix.needsUpdate = true;
    cnStanding = 0; for (let c = 0; c < CN; c++) if (builtAt(A.cn_ids[c], step)) cnStanding++;
    // the nodes' dots
    const pos = dotGeo.attributes.position.array; let n = 0;
    if ($("dots").checked) for (let i = 0; i < nn; i++) if (dotted[i]) { pos[3 * n] = A.nodes[3 * i]; pos[3 * n + 1] = A.nodes[3 * i + 1]; pos[3 * n + 2] = A.nodes[3 * i + 2]; n++; }
    dotGeo.setDrawRange(0, n); dotGeo.attributes.position.needsUpdate = true; dotGeo.computeBoundingSphere();
    // arrows: the largest of each kind D/12 long, each as its magnitude (at least a fifth of that)
    const showLoads = $("loads").checked;
    const v = new THREE.Vector3(), dir = new THREE.Vector3(), tip = new THREE.Vector3(), start = new THREE.Vector3(),
          end = new THREE.Vector3(), mid = new THREE.Vector3(), rot = new THREE.Quaternion();
    for (let i = 0; i < na; i++) {
      const o = AW * i, k = A.arrows[o + 2];
      if (!showLoads || step < A.arrows[o] || step > A.arrows[o + 1]) { shaft.setMatrixAt(i, ZERO); head.setMatrixAt(i, ZERO); continue; }
      v.set(A.arrows[o + 6], A.arrows[o + 7], A.arrows[o + 8]);
      const mag = v.length(); if (!mag) { shaft.setMatrixAt(i, ZERO); head.setMatrixAt(i, ZERO); continue; }
      const len = D / 12 * Math.max(mag / (bigOf[k] || mag), 0.2);
      dir.copy(v).normalize(); tip.set(A.arrows[o + 3], A.arrows[o + 4], A.arrows[o + 5]);
      // a force points AT its node, a moment's axis starts from it
      if (k === 1) { start.copy(tip); end.copy(tip).addScaledVector(dir, len); }
      else { start.copy(tip).addScaledVector(dir, -len); end.copy(tip); }
      rot.setFromUnitVectors(Y, dir); const hl = len * 0.25;
      mid.copy(start).addScaledVector(dir, (len - hl) / 2);
      shaft.setMatrixAt(i, m4.compose(mid, rot, s3.set(TA * 0.5, len - hl, TA * 0.5)));
      head.setMatrixAt(i, m4.compose(end.addScaledVector(dir, -hl / 2), rot, s3.set(TA * 1.4, hl, TA * 1.4)));
      shaft.setColorAt(i, ARROW_COLOUR[k]); head.setColorAt(i, ARROW_COLOUR[k]);
    }
    shaft.instanceMatrix.needsUpdate = true; head.instanceMatrix.needsUpdate = true;
    if (shaft.instanceColor) { shaft.instanceColor.needsUpdate = true; head.instanceColor.needsUpdate = true; }
    stepText(); legend(); render();
    if (picked) showPick(picked);
  }
  function stepText() {
    const s = M.steps[step];
    $("stepname").textContent = `${step + 1}/${ns} ${s.name}`;
    const el = $("step"); el.innerHTML = "";
    const line = (k, v) => { const d = document.createElement("div"); const a = document.createElement("span"); a.className = "k"; a.textContent = k + " ";
      const b = document.createElement("span"); b.textContent = v; d.append(a, b); el.append(d); };
    line("步 step", `${step + 1}. ${s.name}` + (s.procedure ? ` (${s.procedure})` : ""));
    line("构件 members", `${s.standing} standing, ${s.built} built, ${s.removed} removed`);
    if (s.released.length) line("释放 released", s.released.join(", "));
    const held = Array.from({ length: A.sup_nodes.length }, (_, i) => A.sup_masks[i * ns + step]).filter(m => m).length;
    line("支承节点 supported nodes", String(held) + (s.prescribed ? `, ${s.prescribed} with a prescribed motion 给定位移` : ""));
    if (CN) line("连接器 connectors", String(cnStanding));
    if (M.soil.length) line("土弹簧 soil springs", M.soil.join("; "));
    if (s.loads.length) { line("荷载 loads", ""); const ul = document.createElement("ul");
      s.loads.slice(0, 30).forEach(t => { const li = document.createElement("li"); li.textContent = t; ul.append(li); });
      if (s.loads.length > 30) { const li = document.createElement("li"); li.textContent = `… ${s.loads.length - 30} more`; ul.append(li); }
      el.append(ul); }
  }
  // ---- picking: a symbol's anchor within 8 px, else the member nearest the
  // click on the screen, within 8 px (a member a few pixels wide is all but
  // impossible to hit with a ray)
  let down = null;
  const pa = new THREE.Vector3(), pb = new THREE.Vector3();
  function nearest(x, y) {
    let best = null, bestD = 8;
    for (const a of anchors) { const d = Math.hypot(a.P.s[0] - x, a.P.s[1] - y);
      if (d < bestD) { bestD = d; best = a.node !== undefined ? { node: a.node } : { member: a.member }; } }
    if (best) return best;
    for (let e = 0; e < ne; e++) {
      if (shown[e] !== 1) continue;
      pa.copy(node(A.conn[2 * e])).project(camera); pb.copy(node(A.conn[2 * e + 1])).project(camera);
      if (pa.z > 1 || pb.z > 1) continue;
      const ax = (pa.x + 1) / 2 * innerWidth, ay = (1 - pa.y) / 2 * innerHeight;
      const bx = (pb.x + 1) / 2 * innerWidth, by = (1 - pb.y) / 2 * innerHeight;
      const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
      const t = l2 ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / l2)) : 0;
      const d = Math.hypot(x - ax - t * dx, y - ay - t * dy);
      if (d < bestD) { bestD = d; best = { member: e }; }
    }
    return best;
  }
  renderer.domElement.addEventListener("pointerdown", e => { down = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", e => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const r = renderer.domElement.getBoundingClientRect();
    picked = nearest(e.clientX - r.left, e.clientY - r.top);
    showPick(picked); render();
  });
  function stepsWhere(test) {
    const out = []; for (let k = 0; k < ns; k++) if (test(k)) out.push(`step ${k + 1}: ${M.steps[k].name}`);
    return out.join("; ");
  }
  const DOF = ["ux", "uy", "uz", "rx", "ry", "rz"];
  const fmt = v => v.map(x => (Math.abs(x) < 5e-7 ? 0 : +x.toFixed(3))).join(", ");
  function showPick(p) {
    const P = $("pick");
    if (!p) { P.style.display = "none"; return; }
    let rows;
    if (p.node !== undefined) {
      const i = p.node, r = supRow.get(i), m = r === undefined ? 0 : A.sup_masks[r * ns + step];
      const xyz = [0, 1, 2].map(k => A.nodes[3 * i + k] + M.centre[k]);
      rows = [["节点 node", `${i} (${fmt(xyz)})`],
              ["约束 held", m ? DOF.filter((_, k) => (m >> k) & 1).join(" ") : "–"]];
      const mv = r === undefined ? 0 : A.sup_moving[r * ns + step];
      if (mv) rows.push(["给定运动 moving", DOF.filter((_, k) => (mv >> k) & 1).join(" ")]);
      const s = soilOf.get(i);
      if (s) rows.push(["土弹簧 soil", (s & 8 ? "radial in the plane of " : "along ") + ["x", "y", "z"].filter((_, k) => (s >> k) & 1).join(", ")]);
    } else {
      const e = p.member, a = A.conn[2 * e], b = A.conn[2 * e + 1];
      const L = node(a).distanceTo(node(b));
      const reg = A.reg[e] >= 0 ? M.regions[A.reg[e]] : "–";
      rows = [["构件 member", `${e} (nodes ${a}, ${b})`], ["类型 kind", M.kinds[A.kind[e]]], ["区域 region", reg],
        ["截面 section", A.sec[e] >= 0 ? M.sections[A.sec[e]] : "(none assigned)"], ["长度 length", L.toPrecision(5)],
        ["建造 built", A.built0[e] ? "from the start" : stepsWhere(k => isNew(e, k))],
        ["拆除 removed", stepsWhere(k => (k ? builtAt(e, k - 1) : A.built0[e] === 1) && !builtAt(e, k)) || "–"]];
      const t = A.reg[e] >= 0 ? M.tendon_of_region[A.reg[e]] : undefined;
      if (t !== undefined) rows.push(["钢束 tendon", t]);
      const c = cnOf[e];
      if (c >= 0) {
        const R = connAxes(c), w = M.connector_words[String(A.sec[e])] || [];
        rows.push(["轴 axes", `x (${fmt(R[0])}), y (${fmt(R[1])}), z (${fmt(R[2])})`]);
        DOF.forEach((d, k) => rows.push([d, w[k] || "free 自由"]));
      }
    }
    P.innerHTML = ""; P.style.display = "block";
    rows.forEach(([k, v]) => { const d = document.createElement("div"); const s1 = document.createElement("span"); s1.className = "k"; s1.textContent = k + " ";
      const s2 = document.createElement("span"); s2.textContent = v; d.append(s1, s2); P.append(d); });
  }
  // ---- controls
  $("scrub").max = String(ns - 1);
  $("scrub").addEventListener("input", e => { step = +e.target.value; draw(); });
  $("prev").addEventListener("click", () => { step = Math.max(0, step - 1); $("scrub").value = step; draw(); });
  $("next").addEventListener("click", () => { step = Math.min(ns - 1, step + 1); $("scrub").value = step; draw(); });
  $("colour").addEventListener("change", e => { mode = e.target.value; legend(); draw(); });
  ["ghost", "arms", "loads", "dots", "region"].forEach(id => $(id).addEventListener("change", draw));
  ["sup", "cn", "soil", "codes"].forEach(id => $(id).addEventListener("change", render));
  $("size").addEventListener("input", render);
  M.regions.forEach((r, k) => { const o = document.createElement("option"); o.value = String(k); o.textContent = r; $("region").append(o); });
  const u = M.units || {};
  $("sub").textContent = `${M.case} · ${nn} nodes · ${ne} members · ${ns} steps` + (M.tendons.length ? ` · ${M.tendons.length} tendons` : "")
    + (u.length ? ` · ${u.length}, ${u.force || "N"}` : "") + (M.gravity.length ? ` · gravity ${JSON.stringify(M.gravity[0][1])}` : "");
  function render() { place(); renderer.render(scene, camera); glyphPass(); }
  controls.addEventListener("change", render);
  addEventListener("resize", () => { fit(); renderer.setSize(innerWidth, innerHeight); symbolLegend(); render(); });
  symbolLegend(); legend(); draw();
  // for a test: the screen point of a member's middle and of a node; the
  // symbols and codes drawn; the camera brought to a node; a pick
  const toScreen = v => { v.project(camera); const r = renderer.domElement.getBoundingClientRect();
    return [r.left + (v.x + 1) / 2 * innerWidth, r.top + (1 - v.y) / 2 * innerHeight]; };
  const at2 = e => toScreen(node(A.conn[2 * e]).add(node(A.conn[2 * e + 1])).multiplyScalar(0.5));
  const nodeAt = i => toScreen(node(i));
  const look = (i, dist) => { const p = node(i); controls.target.copy(p);
    camera.position.set(p.x + 0.6 * dist, p.y - 1.2 * dist, p.z + 0.7 * dist); controls.update(); render(); };
  const pick = p => { picked = p; showPick(p); render(); };
  window.__vfife_model = { ready: true, steps: ns, members: ne, at: at2, node_at: nodeAt, look, pick, glyphs: () => counts };
})();
</script>
</body>
</html>
'''
