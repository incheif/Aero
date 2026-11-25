// Exploded views for any three.js assembly: staged explode, x-ray, section cut, pinned labels
// with leader lines, and hover/click picking.
//
//   import { Exploder, Labels, Xray, Section, Picker } from "./explode.js";
//   const ex = new Exploder(motor, { axis: new THREE.Vector3(1, 0, 0) });   // parts = motor.children
//   ex.set(0.6);                                 // 0 = assembled, 1 = fully exploded
//   const labels = new Labels(document.querySelector(".labels"), camera, canvas);
//   labels.add(rotor, "Rotor", { anchor: [0, 0.4, 0] });
//   // every frame, after rendering: labels.update();
//
// Per-part options on part.userData:
//   explode: [x, y, z]  offset at full explosion (overrides the computed direction)
//   stage: n            parts with lower stages move first (outer shells before inner parts)
//   shell: true         an outer casing: x-ray makes it ghostly, the section cut opens it
import * as THREE from "three";

const smooth = (t) => t * t * (3 - 2 * t);
const _v = new THREE.Vector3(), _b = new THREE.Box3();

export class Exploder {
  constructor(root, { parts = null, axis = null, spread = 1, center = null } = {}) {
    this.root = root;
    this.parts = parts || [...root.children];
    root.updateMatrixWorld(true);
    const all = new THREE.Box3().setFromObject(root);
    const c = center ? new THREE.Vector3(...center) : all.getCenter(new THREE.Vector3());
    const size = all.getSize(new THREE.Vector3()).length();
    const stages = new Set();
    this.items = this.parts.map((p) => {
      const home = p.position.clone();
      let off;
      if (p.userData.explode) off = new THREE.Vector3(...p.userData.explode);
      else {
        // away from the assembly centre (along `axis` if given), scaled by distance from it
        const pc = _b.setFromObject(p).getCenter(_v).clone();
        const d = pc.sub(c);
        if (axis) d.copy(axis).multiplyScalar(d.dot(axis));
        off = d.lengthSq() < 1e-8 ? new THREE.Vector3(0, size * 0.25, 0) : d.multiplyScalar(spread * 1.2);
        if (p.parent) off.applyQuaternion(p.parent.getWorldQuaternion(new THREE.Quaternion()).invert());
      }
      const stage = p.userData.stage ?? 0;
      stages.add(stage);
      return { p, home, off, stage };
    });
    this.stages = [...stages].sort((a, b) => a - b);
    this.t = 0;
  }

  // staged: each stage gets its own window of t, overlapping by half, so outer parts lead
  set(t) {
    this.t = THREE.MathUtils.clamp(t, 0, 1);
    const n = this.stages.length, w = n > 1 ? 2 / (n + 1) : 1;
    for (const it of this.items) {
      const i = this.stages.indexOf(it.stage), start = n > 1 ? i * w / 2 : 0;
      const k = smooth(THREE.MathUtils.clamp((this.t - start) / w, 0, 1));
      it.p.position.copy(it.home).addScaledVector(it.off, k);
    }
  }
}

// HTML labels pinned to points on parts, laid out in two columns beside the model with
// leader lines, so they never cover the thing they name. Container must overlay the canvas.
export class Labels {
  constructor(container, camera, canvas, { margin = 24, gap = 6 } = {}) {
    Object.assign(this, { container, camera, canvas, margin, gap, items: [] });
    container.style.pointerEvents = "none";
    this.svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    Object.assign(this.svg.style, { position: "absolute", inset: "0", width: "100%", height: "100%", overflow: "visible" });
    container.appendChild(this.svg);
  }

  add(object, text, { anchor = [0, 0, 0], note = "" } = {}) {
    const el = document.createElement("div");
    el.className = "label";
    el.innerHTML = `<b></b>${note ? "<span></span>" : ""}`;
    el.querySelector("b").textContent = text;
    if (note) el.querySelector("span").textContent = note;
    Object.assign(el.style, { position: "absolute", left: "0", top: "0", whiteSpace: "nowrap" });
    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    line.setAttribute("class", "leader");
    this.container.appendChild(el);
    this.svg.appendChild(line);
    const item = { object, el, line, anchor: new THREE.Vector3(...anchor), focus: false };
    this.items.push(item);
    return item;
  }

  update() {
    const r = this.canvas.getBoundingClientRect();
    const mid = r.width / 2;
    const shown = [];
    for (const it of this.items) {
      _v.copy(it.anchor);
      it.object.localToWorld(_v);
      _v.project(this.camera);
      const visible = it.object.visible && _v.z < 1 && Math.abs(_v.x) < 1.05 && Math.abs(_v.y) < 1.05;
      it.el.style.display = it.line.style.display = visible ? "" : "none";
      if (!visible) continue;
      it.sx = (_v.x * 0.5 + 0.5) * r.width;
      it.sy = (-_v.y * 0.5 + 0.5) * r.height;
      it.h = it.el.offsetHeight || 20;
      it.w = it.el.offsetWidth || 80;
      shown.push(it);
    }
    // two columns; within each, keep the anchors' vertical order and push apart overlaps
    for (const left of [true, false]) {
      const col = shown.filter((it) => (it.sx < mid) === left).sort((a, b) => a.sy - b.sy);
      let y = this.margin;
      for (const it of col) { it.ly = Math.max(it.sy - it.h / 2, y); y = it.ly + it.h + this.gap; }
      const over = y - this.gap - (r.height - this.margin);
      if (over > 0) for (const it of col) it.ly -= over;   // shift the column up if it runs off
      for (const it of col) {
        const x = left ? this.margin : r.width - this.margin - it.w;
        it.el.style.transform = `translate(${x}px, ${it.ly}px)`;
        it.el.style.textAlign = left ? "left" : "right";
        const ex = left ? x + it.w + 6 : x - 6, ey = it.ly + it.h / 2;
        const bend = left ? ex + 18 : ex - 18;
        it.line.setAttribute("d", `M${ex},${ey} L${bend},${ey} L${it.sx},${it.sy}`);
        it.el.classList.toggle("focus", it.focus);
        it.line.classList.toggle("focus", it.focus);
      }
    }
  }
}

// X-ray: shells turn into faint, depth-sorted glass so the inside shows; other parts stay solid
export class Xray {
  constructor(root) {
    this.mats = [];
    root.traverse((o) => {
      if (!o.isMesh) return;
      let shell = false;
      for (let p = o; p; p = p.parent) if (p.userData.shell) shell = true;
      if (!shell) return;
      for (const m of [].concat(o.material)) if (!this.mats.find((e) => e.m === m)) this.mats.push({ m, opacity: m.opacity, transparent: m.transparent, depthWrite: m.depthWrite });
    });
    this.on = false;
  }
  set(on, opacity = 0.14) {
    this.on = on;
    for (const e of this.mats) {
      e.m.transparent = on || e.transparent;
      e.m.opacity = on ? opacity : e.opacity;
      e.m.depthWrite = on ? false : e.depthWrite;
      e.m.needsUpdate = true;
    }
  }
}

// Section cut: a clipping plane through the shells. renderer.localClippingEnabled must be true.
export class Section {
  constructor(root, normal = [0, 0, -1]) {
    this.plane = new THREE.Plane(new THREE.Vector3(...normal), 1e6);
    this.mats = new Set();
    root.traverse((o) => {
      if (!o.isMesh) return;
      let shell = false;
      for (let p = o; p; p = p.parent) if (p.userData.shell) shell = true;
      if (shell) for (const m of [].concat(o.material)) { m.clippingPlanes = [this.plane]; m.clipShadows = true; m.side = THREE.DoubleSide; this.mats.add(m); }
    });
  }
  // offset along the normal; Infinity (or null) switches the cut off
  set(offset) { this.plane.constant = offset == null || !isFinite(offset) ? 1e6 : offset; }
}

// Hover highlight and click selection of whole parts
export class Picker {
  constructor(canvas, camera, parts, { onPick = null, onHover = null, color = "#ff8a3d" } = {}) {
    Object.assign(this, { canvas, camera, parts, onPick, onHover });
    this.ray = new THREE.Raycaster();
    this.hot = null;
    this.glow = new THREE.Color(color);
    const pick = (e) => {
      const r = canvas.getBoundingClientRect();
      this.ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
      const hit = this.ray.intersectObjects(parts, true).find((h) => h.object.visible && !(h.object.material?.opacity < 0.3));
      if (!hit) return null;
      let o = hit.object;
      while (o && !parts.includes(o)) o = o.parent;
      return o;
    };
    canvas.addEventListener("pointermove", (e) => { const p = pick(e); if (p !== this.hot) { this.highlight(this.hot, false); this.hot = p; this.highlight(p, true); canvas.style.cursor = p ? "pointer" : ""; onHover && onHover(p); } });
    canvas.addEventListener("click", (e) => { const p = pick(e); onPick && onPick(p); });
  }
  highlight(part, on) {
    part?.traverse((o) => {
      if (!o.isMesh || !o.material.emissive) return;
      if (on) { o.userData._em = o.material.emissive.clone(); o.material.emissive.copy(this.glow).multiplyScalar(0.25); }
      else if (o.userData._em) o.material.emissive.copy(o.userData._em);
    });
  }
}
