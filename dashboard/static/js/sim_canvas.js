/**
 * sim_canvas.js
 * 60 FPS HTML5 Canvas Renderer for AERO Arena, Robot, LiDAR, and Trajectory.
 */

class SimCanvasRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');

    // Arena bounds (in simulation meters)
    this.xMin = -2.0;
    this.xMax = 6.0;
    this.yMin = -2.0;
    this.yMax = 5.0;

    // Viewport transform
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;

    // Simulation state
    this.robot = { x: 0, y: 0, yaw: 0, linear_vel: 0, angular_vel: 0 };
    this.target = { x: 3.0, y: 3.0, tolerance: 0.10 };
    this.obstacles = [
      { x: 1.5, y: 1.0, radius: 0.28 },
      { x: 1.0, y: 2.2, radius: 0.28 },
      { x: 2.4, y: 2.2, radius: 0.28 },
      { x: 3.8, y: 1.5, radius: 0.28 },
    ];
    this.lidarRanges = [];
    this.trajectory = [{ x: 0, y: 0 }];

    this.pulsePhase = 0;

    // Auto-resize
    window.addEventListener('resize', () => this.resize());
    this.resize();

    // Start render loop
    this.render = this.render.bind(this);
    requestAnimationFrame(this.render);
  }

  resize() {
    const rect = this.canvas.parentElement.getBoundingClientRect();
    this.canvas.width = rect.width;
    this.canvas.height = rect.height;
  }

  updateFrame(frame) {
    if (!frame) return;
    if (frame.robot) {
      this.robot = frame.robot;
      this.trajectory.push({ x: this.robot.x, y: this.robot.y });
      if (this.trajectory.length > 500) this.trajectory.shift();
    }
    if (frame.target) this.target = frame.target;
    if (frame.obstacles) this.obstacles = frame.obstacles;
    if (frame.lidar_ranges) this.lidarRanges = frame.lidar_ranges;
    if (frame.arena) {
      this.xMin = frame.arena.x_min;
      this.xMax = frame.arena.x_max;
      this.yMin = frame.arena.y_min;
      this.yMax = frame.arena.y_max;
    }
  }

  resetTrajectory() {
    this.trajectory = [{ x: this.robot.x, y: this.robot.y }];
  }

  worldToScreen(x, y) {
    const meterSpanX = this.xMax - this.xMin;
    const meterSpanY = this.yMax - this.yMin;

    const padding = 50;
    const availW = this.canvas.width - padding * 2;
    const availH = this.canvas.height - padding * 2;

    const scale = Math.min(availW / meterSpanX, availH / meterSpanY) * this.zoom;
    const originX = (this.canvas.width - meterSpanX * scale) / 2 + this.panX;
    const originY = (this.canvas.height + meterSpanY * scale) / 2 + this.panY;

    const sx = originX + (x - this.xMin) * scale;
    const sy = originY - (y - this.yMin) * scale; // invert Y for Cartesian
    return { x: sx, y: sy, scale };
  }

  render() {
    this.pulsePhase = (this.pulsePhase + 0.05) % (Math.PI * 2);
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // 1. Draw Grid
    this.drawGrid(ctx);

    // 2. Draw Arena Walls
    this.drawArenaWalls(ctx);

    // 3. Draw Target Waypoint
    this.drawTarget(ctx);

    // 4. Draw Obstacles
    this.drawObstacles(ctx);

    // 5. Draw Trajectory Trail
    this.drawTrajectory(ctx);

    // 6. Draw LiDAR Raycasts
    this.drawLiDAR(ctx);

    // 7. Draw Robot
    this.drawRobot(ctx);

    requestAnimationFrame(this.render);
  }

  drawGrid(ctx) {
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;

    for (let x = Math.ceil(this.xMin); x <= Math.floor(this.xMax); x += 1.0) {
      const p1 = this.worldToScreen(x, this.yMin);
      const p2 = this.worldToScreen(x, this.yMax);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();

      // Axis label
      ctx.fillStyle = 'rgba(255, 255, 255, 0.18)';
      ctx.font = '10px monospace';
      ctx.fillText(`${x}m`, p1.x + 4, p1.y - 6);
    }

    for (let y = Math.ceil(this.yMin); y <= Math.floor(this.yMax); y += 1.0) {
      const p1 = this.worldToScreen(this.xMin, y);
      const p2 = this.worldToScreen(this.xMax, y);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
  }

  drawArenaWalls(ctx) {
    const tl = this.worldToScreen(this.xMin, this.yMax);
    const br = this.worldToScreen(this.xMax, this.yMin);

    ctx.save();
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 3;
    ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);

    // Subtle cyan inner border
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.12)';
    ctx.lineWidth = 1;
    ctx.strokeRect(tl.x + 2, tl.y + 2, (br.x - tl.x) - 4, (br.y - tl.y) - 4);
    ctx.restore();
  }

  drawTarget(ctx) {
    const p = this.worldToScreen(this.target.x, this.target.y);
    const tolRadius = this.target.tolerance * p.scale;
    const pulse = Math.sin(this.pulsePhase) * 4;

    ctx.save();
    // Glowing outer circle
    const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 24 + pulse);
    grad.addColorStop(0, 'rgba(16, 185, 129, 0.45)');
    grad.addColorStop(1, 'rgba(16, 185, 129, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 24 + pulse, 0, Math.PI * 2);
    ctx.fill();

    // Target core
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
    ctx.fill();

    // Tolerance boundary
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.7)';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(p.x, p.y, Math.max(12, tolRadius), 0, Math.PI * 2);
    ctx.stroke();

    // Coordinate text
    ctx.setLineDash([]);
    ctx.fillStyle = '#34d399';
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillText(`Target (${this.target.x}, ${this.target.y})`, p.x + 12, p.y - 10);
    ctx.restore();
  }

  drawObstacles(ctx) {
    for (const obs of this.obstacles) {
      const p = this.worldToScreen(obs.x, obs.y);
      const r = obs.radius * p.scale;

      ctx.save();
      // Obstacle body
      ctx.fillStyle = '#ef4444';
      ctx.shadowColor = 'rgba(239, 68, 68, 0.4)';
      ctx.shadowBlur = 12;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();

      // Top bevel
      ctx.fillStyle = 'rgba(255, 255, 255, 0.15)';
      ctx.beginPath();
      ctx.arc(p.x - r * 0.2, p.y - r * 0.2, r * 0.45, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  drawTrajectory(ctx) {
    if (this.trajectory.length < 2) return;

    ctx.save();
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.4)';
    ctx.lineWidth = 2;
    ctx.beginPath();

    const start = this.worldToScreen(this.trajectory[0].x, this.trajectory[0].y);
    ctx.moveTo(start.x, start.y);

    for (let i = 1; i < this.trajectory.length; i++) {
      const pt = this.worldToScreen(this.trajectory[i].x, this.trajectory[i].y);
      ctx.lineTo(pt.x, pt.y);
    }
    ctx.stroke();

    // Breadcrumb dots
    ctx.fillStyle = '#00e5ff';
    for (let i = 0; i < this.trajectory.length; i += 6) {
      const pt = this.worldToScreen(this.trajectory[i].x, this.trajectory[i].y);
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  drawLiDAR(ctx) {
    if (!this.lidarRanges || this.lidarRanges.length === 0) return;

    const rp = this.worldToScreen(this.robot.x, this.robot.y);
    const n = this.lidarRanges.length;

    ctx.save();
    for (let i = 0; i < n; i++) {
      const range = this.lidarRanges[i];
      const angle = this.robot.yaw + (2 * Math.PI * i / n);
      const endX = this.robot.x + range * Math.cos(angle);
      const endY = this.robot.y + range * Math.sin(angle);
      const ep = this.worldToScreen(endX, endY);

      // Color based on proximity
      let strokeColor = 'rgba(0, 229, 255, 0.15)';
      if (range < 0.45) strokeColor = 'rgba(239, 68, 68, 0.45)';
      else if (range < 0.8) strokeColor = 'rgba(245, 158, 11, 0.3)';

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(rp.x, rp.y);
      ctx.lineTo(ep.x, ep.y);
      ctx.stroke();
    }
    ctx.restore();
  }

  drawRobot(ctx) {
    const p = this.worldToScreen(this.robot.x, this.robot.y);
    const r = 0.15 * p.scale; // 15cm radius

    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(-this.robot.yaw); // invert angle for Cartesian screen

    // Robot body
    ctx.fillStyle = '#0f172a';
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = 'rgba(0, 229, 255, 0.35)';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(0, 0, Math.max(10, r), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Direction arrow
    ctx.fillStyle = '#00e5ff';
    ctx.beginPath();
    ctx.moveTo(r * 0.85, 0);
    ctx.lineTo(-r * 0.35, -r * 0.4);
    ctx.lineTo(-r * 0.15, 0);
    ctx.lineTo(-r * 0.35, r * 0.4);
    ctx.closePath();
    ctx.fill();

    // Wheels
    ctx.fillStyle = '#475569';
    const wheelW = r * 0.6;
    const wheelH = r * 0.25;
    ctx.fillRect(-wheelW / 2, -r - wheelH / 2, wheelW, wheelH);
    ctx.fillRect(-wheelW / 2, r - wheelH / 2, wheelW, wheelH);

    ctx.restore();
  }
}

window.SimCanvasRenderer = SimCanvasRenderer;
