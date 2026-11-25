// A procedural induction motor, built from lathes and instanced parts, taken apart with
// explode.js.   ?explode=0..1  ?cut=0..1  ?xray=1  set the state for stills and checks;
// ?transparent=1 renders on a clear background with no UI, for compositing;
// ?record=1 plays a scripted take-apart for the frame-capture skill.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Exploder, Labels, Xray, Section, Picker } from '../explode.js';

const params = new URLSearchParams(location.search);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

const canvas = document.getElementById('gl');
const CLEAR = params.has('transparent');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: CLEAR, preserveDrawingBuffer: params.has('record') || CLEAR });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.localClippingEnabled = true;
renderer.setClearColor('#eeeae2', CLEAR ? 0 : 1);   // the page's paper colour: an opaque canvas is black by default
if (CLEAR) {
  for (const el of document.querySelectorAll('header, .panel, .labels')) el.style.display = 'none';
  document.documentElement.style.background = document.body.style.background = 'transparent';
}

const scene = new THREE.Scene();
scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.9;
const camera = new THREE.PerspectiveCamera(30, innerWidth / innerHeight, 0.01, 50);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true; controls.minDistance = 0.6; controls.maxDistance = 5; controls.maxPolarAngle = 1.5;

const key = new THREE.DirectionalLight('#fff6ea', 2.2);
key.position.set(1.2, 2.4, 1.6); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -1.2, right: 1.2, top: 1.2, bottom: -1.2, near: 0.5, far: 6 });
key.shadow.bias = -0.0004; key.shadow.radius = 4;
scene.add(key);
const floor = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShadowMaterial({ opacity: 0.16 }));
floor.rotation.x = -Math.PI / 2; floor.position.y = -0.163; floor.receiveShadow = true; scene.add(floor);

// ---------------------------------------------------------------- parts
// every part gets its own materials, so x-ray and hover highlight touch only that part
const mat = (o) => new THREE.MeshPhysicalMaterial({ roughness: 0.45, metalness: 0.3, ...o });
const PAINT = { color: '#35556e', roughness: 0.42, metalness: 0.25, clearcoat: 0.6, clearcoatRoughness: 0.3 };
const STEEL = { color: '#b9bdc2', roughness: 0.32, metalness: 0.9 };
const COPPER = { color: '#c26b32', roughness: 0.35, metalness: 0.95 };
const ALU = { color: '#d5d8db', roughness: 0.38, metalness: 0.85 };

// lathe around the X axis (the shaft): profile [[radius, x], ...]
function latheX(points, segs = 64) {
  const g = new THREE.LatheGeometry(points.map(([r, x]) => new THREE.Vector2(r, x)), segs);
  g.rotateZ(-Math.PI / 2);
  return g;
}
const mesh = (g, m) => { const o = new THREE.Mesh(g, m); o.castShadow = o.receiveShadow = true; return o; };
function ring(geometry, material, n, place) {
  const im = new THREE.InstancedMesh(geometry, material, n);
  const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), p = new THREE.Vector3(), s = new THREE.Vector3(1, 1, 1);
  for (let i = 0; i < n; i++) { place(i, p, q, s); im.setMatrixAt(i, m4.compose(p, q, s)); }
  im.castShadow = im.receiveShadow = true;
  return im;
}
const aroundX = (a) => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), a);

const motor = new THREE.Group();
scene.add(motor);
const part = (name, opts = {}) => { const g = new THREE.Group(); g.name = name; Object.assign(g.userData, opts); motor.add(g); return g; };

// housing: finned shell, feet and terminal box
const housing = part('housing', { shell: true, stage: 0, explode: [0, 0.32, 0] });
const hMat = mat(PAINT);
housing.add(mesh(latheX([[0.105, -0.15], [0.12, -0.15], [0.12, 0.15], [0.105, 0.15], [0.105, -0.15]]), hMat));
housing.add(ring(new THREE.BoxGeometry(0.27, 0.018, 0.005), hMat, 26, (i, p, q) => {
  const a = (i / 26) * Math.PI * 2; if (Math.abs(Math.sin(a)) > 0.93 && Math.cos(a) < 0) { p.set(0, 99, 0); return; }
  q.copy(aroundX(a)); p.set(0, Math.cos(a) * 0.128, Math.sin(a) * 0.128);
}));
for (const z of [-0.085, 0.085]) { const f = mesh(new THREE.BoxGeometry(0.24, 0.05, 0.05), hMat); f.position.set(0, -0.138, z); housing.add(f); }
const tbox = mesh(new THREE.BoxGeometry(0.09, 0.05, 0.08), hMat); tbox.position.set(0.02, 0.145, 0); housing.add(tbox);
const tlid = mesh(new THREE.BoxGeometry(0.096, 0.008, 0.086), mat(PAINT)); tlid.position.set(0.02, 0.173, 0); housing.add(tlid);

// stator: laminated core and copper windings
const stator = part('stator', { stage: 1, explode: [0, 0, 0] });
stator.add(mesh(latheX([[0.074, -0.1], [0.103, -0.1], [0.103, 0.1], [0.074, 0.1], [0.074, -0.1]]), mat({ color: '#6e737a', roughness: 0.55, metalness: 0.7 })));
const cu = mat(COPPER);
for (const x of [-0.108, 0.108]) stator.add(ring(new THREE.TorusGeometry(0.016, 0.0065, 10, 20), cu, 12, (i, p, q) => {
  const a = (i / 12) * Math.PI * 2; q.copy(aroundX(a)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI / 2));
  p.set(x, Math.cos(a) * 0.086, Math.sin(a) * 0.086);
}));

// rotor with skewed bars, on the shaft
const rotor = part('rotor', { stage: 2, explode: [0.42, 0, 0] });
rotor.add(mesh(latheX([[0.001, -0.095], [0.068, -0.095], [0.068, 0.095], [0.001, 0.095]]), mat({ color: '#80868d', roughness: 0.5, metalness: 0.8 })));
rotor.add(ring(new THREE.BoxGeometry(0.19, 0.006, 0.01), mat(ALU), 22, (i, p, q) => {
  const a = (i / 22) * Math.PI * 2; q.copy(aroundX(a)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.12));
  p.set(0, Math.cos(a) * 0.069, Math.sin(a) * 0.069);
}));
const shaft = mesh(latheX([[0.001, -0.24], [0.014, -0.24], [0.014, 0.3], [0.012, 0.31], [0.012, 0.34], [0.001, 0.34]], 32), mat(STEEL));
rotor.add(shaft);

// bearings: outer race, inner race, balls
function bearing(name, x, explodeX) {
  const b = part(name, { stage: 1, explode: [explodeX, 0, 0] });
  const s = mat(STEEL);
  b.add(mesh(latheX([[0.026, x - 0.008], [0.032, x - 0.008], [0.032, x + 0.008], [0.026, x + 0.008], [0.026, x - 0.008]]), s));
  b.add(mesh(latheX([[0.014, x - 0.008], [0.019, x - 0.008], [0.019, x + 0.008], [0.014, x + 0.008], [0.014, x - 0.008]]), s));
  b.add(ring(new THREE.SphereGeometry(0.0045, 12, 8), s, 9, (i, p) => { const a = (i / 9) * Math.PI * 2; p.set(x, Math.cos(a) * 0.0225, Math.sin(a) * 0.0225); }));
  return b;
}
const frontBearing = bearing('front bearing', 0.16, 0.6);
const rearBearing = bearing('rear bearing', -0.16, -0.22);

// end shields with bolts
function endShield(name, x, dir, explodeX) {
  const e = part(name, { shell: true, stage: 0, explode: [explodeX, 0, 0] });
  const m = mat({ ...PAINT, color: '#3d6079' });
  const prof = [[0.033, x], [0.12, x], [0.12, x + dir * 0.012], [0.06, x + dir * 0.022], [0.045, x + dir * 0.03], [0.033, x + dir * 0.03], [0.033, x]];
  e.add(mesh(latheX(dir > 0 ? prof : prof.reverse()), m));   // keep the winding (and normals) outward on both ends
  e.add(ring(new THREE.CylinderGeometry(0.006, 0.006, 0.012, 6).rotateZ(Math.PI / 2), mat(STEEL), 4, (i, p) => {
    const a = (i / 4) * Math.PI * 2 + Math.PI / 4; p.set(x + dir * 0.012, Math.cos(a) * 0.108, Math.sin(a) * 0.108);
  }));
  return e;
}
const frontShield = endShield('front shield', 0.15, 1, 0.84);
const rearShield = endShield('rear shield', -0.15, -1, -0.32);

// cooling fan and its cover
const fan = part('fan', { stage: 1, explode: [-0.46, 0, 0] });
const fm = mat({ color: '#2b2f33', roughness: 0.6, metalness: 0.1 });
fan.add(mesh(latheX([[0.016, -0.215], [0.03, -0.215], [0.03, -0.195], [0.016, -0.195]]), fm));
fan.add(ring(new THREE.BoxGeometry(0.004, 0.075, 0.03), fm, 10, (i, p, q) => {
  const a = (i / 10) * Math.PI * 2; q.copy(aroundX(a)).multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.35));
  p.set(-0.205, Math.cos(a) * 0.065, Math.sin(a) * 0.065);
}));
const cover = part('fan cover', { shell: true, stage: 0, explode: [-0.62, 0, 0] });
const cm = mat({ ...PAINT, color: '#2f4b60' });
cover.add(mesh(latheX([[0.118, -0.16], [0.12, -0.16], [0.12, -0.235], [0.1, -0.25], [0.05, -0.255], [0.048, -0.25], [0.098, -0.245], [0.116, -0.232], [0.118, -0.16]]), cm));

// ---------------------------------------------------------------- explode, x-ray, section, labels
const ex = new Exploder(motor);
const xray = new Xray(motor);
const section = new Section(motor, [0, 0, -1]);
const labels = new Labels(document.querySelector('.labels'), camera, canvas);
const L = [
  [housing, 'Housing', 'cast frame, cooling fins', [0, 0.13, 0.06]],
  [stator, 'Stator', 'fixed copper windings', [0.05, 0.1, 0.03]],
  [rotor, 'Rotor', 'turns in the magnetic field', [0.03, 0.07, 0]],
  [shaft, 'Shaft', 'carries the torque out', [0.3, 0.014, 0]],
  [frontBearing, 'Bearings', 'keep the rotor centred', [0.16, 0.032, 0]],
  [frontShield, 'End shields', 'hold the bearings', [0.16, 0.1, 0.04]],
  [fan, 'Fan', 'pulls air over the fins', [-0.205, 0.1, 0]],
  [cover, 'Fan cover', '', [-0.2, 0.12, 0]],
].map(([o, t, n, a]) => labels.add(o, t, { note: n, anchor: a }));

const parts = motor.children;
new Picker(canvas, camera, parts, {
  onHover: (p) => labels.items.forEach((it) => { let o = it.object; while (o && o.parent !== motor) o = o.parent; it.focus = !!p && o === p; }),
});

// ---------------------------------------------------------------- controls
const exSlider = document.getElementById('explode'), cutSlider = document.getElementById('cut'), xBtn = document.getElementById('xray');
let explodeTarget = +(params.get('explode') ?? 0), explodeNow = explodeTarget;
const setCut = (v) => section.set(v > 0.001 ? 0.14 * (1 - v) : null);
exSlider.value = explodeTarget;
exSlider.addEventListener('input', () => { explodeTarget = +exSlider.value; });
cutSlider.value = params.get('cut') ?? 0; setCut(+cutSlider.value);
cutSlider.addEventListener('input', () => setCut(+cutSlider.value));
const setX = (on) => { xray.set(on); xBtn.setAttribute('aria-pressed', String(on)); };
setX(params.get('xray') === '1');
xBtn.addEventListener('click', () => setX(!xray.on));

// fit the assembly to any screen shape: the camera pulls back as the motor comes apart,
// until the visitor zooms or orbits by hand
const fit = { near: 1, far: 2 }, viewDir = new THREE.Vector3(0.42, 0.34, 1).normalize();
let userMoved = false;
controls.addEventListener('start', () => { userMoved = true; });
function frameView() {
  camera.aspect = innerWidth / innerHeight;
  const t = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
  const d = (halfW, halfH) => Math.max(halfH / t, halfW / (t * camera.aspect)) * 1.2;
  fit.near = d(0.36, 0.24); fit.far = d(1.05, 0.45);
  controls.maxDistance = Math.max(5, fit.far * 1.4);     // or the controls clamp the fitted distance
  controls.target.set(0.1, 0.02, 0);
  camera.updateProjectionMatrix();
  userMoved = false;
}
function autoDistance(k) {
  if (userMoved) return;
  camera.position.copy(controls.target).addScaledVector(viewDir, THREE.MathUtils.lerp(fit.near, fit.far, k));
}
frameView();
autoDistance(explodeNow);
addEventListener('resize', () => { renderer.setSize(innerWidth, innerHeight, false); frameView(); });

window.__app = { renderer, scene, camera, ex, xray, section };

// ?record=1: a scripted take-apart for frame-capture, timed on the page clock
if (params.has('record')) {
  document.querySelector('.panel').style.display = 'none';
  const ease = (k) => { k = Math.min(1, Math.max(0, k)); return k * k * (3 - 2 * k); };
  window.__recordFrame = (t) => {
    explodeTarget = t < 6 ? ease((t - 0.8) / 2.6) : 1 - ease((t - 6) / 2.2);
    const want = t > 3.8 && t < 6;
    if (xray.on !== want) setX(want);
    const a = 0.4 + t * 0.035;                       // a slow drift round the motor
    viewDir.set(Math.sin(a), 0.34, Math.cos(a)).normalize();
  };
}
let frames = 0, last = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000); last = now;
  explodeNow += (explodeTarget - explodeNow) * (REDUCED ? 1 : 1 - Math.pow(0.001, dt));
  ex.set(explodeNow);
  autoDistance(explodeNow);
  if (!REDUCED) rotor.rotation.x += dt * (0.4 + 2.4 * (1 - explodeNow));    // spins while assembled
  controls.update();
  renderer.render(scene, camera);
  labels.update();
  if (++frames === 3) window.__ready = true;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
