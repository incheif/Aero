/**
 * telemetry.js
 * SVG Circular Gauges, Sparkline Chart, and Terminal Streamer for AERO.
 */

class TelemetryManager {
  constructor() {
    this.circ = 2 * Math.PI * 38; // r=38, circumference ≈ 238.76

    // Gauge elements
    this.goalDistEl = document.getElementById('val-goal-dist');
    this.goalDistCircle = document.getElementById('gauge-goal-dist');

    this.simClockEl = document.getElementById('val-sim-clock');
    this.simClockCircle = document.getElementById('gauge-sim-clock');

    this.clearanceEl = document.getElementById('val-clearance');
    this.clearanceCircle = document.getElementById('gauge-clearance');

    // Sparkline canvas
    this.sparkCanvas = document.getElementById('sparkline-canvas');
    this.sparkCtx = this.sparkCanvas ? this.sparkCanvas.getContext('2d') : null;
    this.sparkHistory = [];

    // Terminal container
    this.terminalEl = document.getElementById('terminal-logs');

    // Status pill
    this.statusPill = document.getElementById('status-pill');
    this.statusText = document.getElementById('status-text');

    // Iteration indicators
    this.iterationEl = document.getElementById('iteration-badge');
  }

  setGauge(circle, value, maxVal, color) {
    if (!circle) return;
    const progress = Math.max(0, Math.min(1, value / maxVal));
    const offset = this.circ * (1 - progress);
    circle.style.strokeDasharray = `${this.circ}`;
    circle.style.strokeDashoffset = `${offset}`;
    if (color) circle.style.stroke = color;
  }

  updateMetrics(frame) {
    if (!frame) return;

    // 1. Goal Distance (0 to 4.5m)
    if (frame.goal_distance !== undefined) {
      const dist = frame.goal_distance;
      this.goalDistEl.textContent = `${dist.toFixed(2)}m`;
      const color = dist <= 0.10 ? '#10b981' : '#00e5ff';
      this.setGauge(this.goalDistCircle, 4.5 - dist, 4.5, color);
    }

    // 2. Sim Clock (0 to 30s)
    if (frame.sim_time !== undefined) {
      const t = frame.sim_time;
      this.simClockEl.textContent = `${t.toFixed(1)}s`;
      const color = t > 25 ? '#f59e0b' : '#00e5ff';
      this.setGauge(this.simClockCircle, t, 30.0, color);
    }

    // 3. Obstacle Clearance
    if (frame.min_obstacle_distance !== undefined) {
      const c = frame.min_obstacle_distance;
      this.clearanceEl.textContent = c > 90 ? '>3m' : `${c.toFixed(2)}m`;
      let color = '#10b981';
      if (c < 0.3) color = '#f43f5e';
      else if (c < 0.6) color = '#f59e0b';
      this.setGauge(this.clearanceCircle, c, 2.5, color);

      // Record in sparkline
      this.addSparklinePoint(c > 90 ? 2.5 : c);
    }

    // 4. Status Pill
    if (frame.status) {
      this.setStatus(frame.status, frame.reason);
    }
  }

  setStatus(status, reason = '') {
    if (!this.statusPill) return;
    this.statusPill.className = `status-pill ${status.toLowerCase()}`;
    const reasonTxt = reason && reason !== 'IN_PROGRESS' ? ` (${reason})` : '';
    this.statusText.textContent = `${status}${reasonTxt}`;
  }

  setIteration(current, max) {
    if (this.iterationEl) {
      this.iterationEl.textContent = `Iteration ${current} / ${max}`;
    }
    // Update timeline stepper dots
    for (let i = 1; i <= 5; i++) {
      const dot = document.getElementById(`step-${i}`);
      if (dot) {
        if (i === current) dot.className = 'timeline-step active';
        else if (i < current) dot.className = 'timeline-step passed';
        else dot.className = 'timeline-step';
      }
    }
  }

  addSparklinePoint(val) {
    if (!this.sparkCtx) return;
    this.sparkHistory.push(val);
    if (this.sparkHistory.length > 50) this.sparkHistory.shift();

    const ctx = this.sparkCtx;
    const w = this.sparkCanvas.width;
    const h = this.sparkCanvas.height;

    ctx.clearRect(0, 0, w, h);
    if (this.sparkHistory.length < 2) return;

    ctx.beginPath();
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 1.5;

    const maxVal = 2.5;
    const step = w / (this.sparkHistory.length - 1);

    for (let i = 0; i < this.sparkHistory.length; i++) {
      const x = i * step;
      const y = h - (this.sparkHistory[i] / maxVal) * (h - 6) - 3;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Area fill
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0, 229, 255, 0.08)';
    ctx.fill();
  }

  log(message, level = 'info') {
    if (!this.terminalEl) return;
    const entry = document.createElement('div');
    entry.className = `log-entry log-${level}`;
    const timestamp = new Date().toLocaleTimeString();
    entry.textContent = `[${timestamp}] ${message}`;
    this.terminalEl.appendChild(entry);
    this.terminalEl.scrollTop = this.terminalEl.scrollHeight;

    // Prune old logs if exceeding 200 lines
    while (this.terminalEl.children.length > 200) {
      this.terminalEl.removeChild(this.terminalEl.firstChild);
    }
  }

  clearLogs() {
    if (this.terminalEl) this.terminalEl.innerHTML = '';
  }
}

window.TelemetryManager = TelemetryManager;
