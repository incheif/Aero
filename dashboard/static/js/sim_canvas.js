/**
 * sim_canvas.js
 * 60 FPS HTML5 Canvas Renderer for AERO Multi-Room Cognitive Simulation.
 * Renders house walls, doorways, rooms, semantic 3D furniture landmarks,
 * robot camera FOV cone, LiDAR raycasts, and dynamic path trails.
 */

class SimCanvasRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');

    // House arena bounds (meters)
    this.xMin = -6.0;
    this.xMax = 6.0;
    this.yMin = -5.0;
    this.yMax = 5.0;

    // Viewport transform
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;

    // State
    this.robot = { x: 0, y: 0, yaw: 0, linear_vel: 0, angular_vel: 0, camera_fov: 1.13, camera_range: 3.5 };
    this.target = { name: 'Red Sofa', x: 3.2, y: 3.0, tolerance: 0.15 };
    this.interiorWalls = [];
    this.obstacles = [];
    this.semanticObjects = [];
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
      if (this.trajectory.length > 600) this.trajectory.shift();
    }
    if (frame.target) this.target = frame.target;
    if (frame.interior_walls) this.interiorWalls = frame.interior_walls;
    if (frame.obstacles) this.obstacles = frame.obstacles;
    if (frame.semantic_objects) this.semanticObjects = frame.semantic_objects;
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

    const padding = 45;
    const availW = this.canvas.width - padding * 2;
    const availH = this.canvas.height - padding * 2;

    const scale = Math.min(availW / meterSpanX, availH / meterSpanY) * this.zoom;
    const originX = (this.canvas.width - meterSpanX * scale) / 2 + this.panX;
    const originY = (this.canvas.height + meterSpanY * scale) / 2 + this.panY;

    const sx = originX + (x - this.xMin) * scale;
    const sy = originY - (y - this.yMin) * scale;
    return { x: sx, y: sy, scale };
  }

  render() {
    this.pulsePhase = (this.pulsePhase + 0.05) % (Math.PI * 2);
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // 1. Grid & Room Labels
    this.drawGridAndRooms(ctx);

    // 2. Interior Partition Walls & Doorways
    this.drawWalls(ctx);

    // 3. Semantic Physical Landmarks (Sofa, Table, Boxes, Dock)
    this.drawSemanticLandmarks(ctx);

    // 4. Target Waypoint
    this.drawTarget(ctx);

    // 5. Breadcrumb Trajectory Trail
    this.drawTrajectory(ctx);

    // 6. Camera Field-of-View Cone
    this.drawCameraCone(ctx);

    // 7. 360-degree LiDAR Raycasts
    this.drawLiDAR(ctx);

    // 8. Mobile Robot
    this.drawRobot(ctx);

    requestAnimationFrame(this.render);
  }

  drawGridAndRooms(ctx) {
    // Subtle background coordinate lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.03)';
    ctx.lineWidth = 1;

    for (let x = Math.ceil(this.xMin); x <= Math.floor(this.xMax); x += 2.0) {
      const p1 = this.worldToScreen(x, this.yMin);
      const p2 = this.worldToScreen(x, this.yMax);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }
    for (let y = Math.ceil(this.yMin); y <= Math.floor(this.yMax); y += 2.0) {
      const p1 = this.worldToScreen(this.xMin, y);
      const p2 = this.worldToScreen(this.xMax, y);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Room Titles on Canvas
    const rooms = [
      { name: 'KITCHEN', x: -3.0, y: 4.2 },
      { name: 'LIVING ROOM', x: 3.0, y: 4.2 },
      { name: 'DOCKING BAY', x: -3.0, y: -4.2 },
      { name: 'STORAGE BAY', x: 3.0, y: -4.2 },
    ];
    ctx.font = '700 12px Inter, sans-serif';
    ctx.fillStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.textAlign = 'center';
    for (const r of rooms) {
      const p = this.worldToScreen(r.x, r.y);
      ctx.fillText(r.name, p.x, p.y);
    }
  }

  drawWalls(ctx) {
    // Outer perimeter
    const tl = this.worldToScreen(this.xMin, this.yMax);
    const br = this.worldToScreen(this.xMax, this.yMin);

    ctx.save();
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 4;
    ctx.strokeRect(tl.x, tl.y, br.x - tl.x, br.y - tl.y);

    // Subtle neon cyan glow on boundary
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.strokeRect(tl.x + 2, tl.y + 2, (br.x - tl.x) - 4, (br.y - tl.y) - 4);

    // Interior walls (partition dividers with doors)
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 5;
    for (const wall of this.interiorWalls) {
      const p1 = this.worldToScreen(wall.x1, wall.y1);
      const p2 = this.worldToScreen(wall.x2, wall.y2);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.stroke();
    }

    // Doorway markers
    const door1 = this.worldToScreen(0.0, 0.25);
    ctx.fillStyle = 'rgba(0, 229, 255, 0.2)';
    ctx.font = '500 10px monospace';
    ctx.fillText('DOORWAY', door1.x + 4, door1.y);
    ctx.restore();
  }

  drawSemanticLandmarks(ctx) {
    for (const obj of this.semanticObjects) {
      const p = this.worldToScreen(obj.x, obj.y);
      const r = (obj.radius || 0.45) * p.scale;

      ctx.save();
      // Outer aura if discovered by robot
      if (obj.discovered) {
        const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 1.8);
        glow.addColorStop(0, 'rgba(0, 229, 255, 0.3)');
        glow.addColorStop(1, 'rgba(0, 229, 255, 0)');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r * 1.8, 0, Math.PI * 2);
        ctx.fill();
      }

      // Base circle
      ctx.fillStyle = obj.color || '#3b82f6';
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();

      // Border
      ctx.strokeStyle = obj.discovered ? '#00e5ff' : 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Icon & Name
      ctx.textAlign = 'center';
      ctx.font = `${Math.max(14, r * 0.9)}px sans-serif`;
      ctx.fillText(obj.icon || '📦', p.x, p.y + (r * 0.35));

      ctx.fillStyle = obj.discovered ? '#f8fafc' : '#94a3b8';
      ctx.font = '600 11px Inter, sans-serif';
      ctx.fillText(obj.name, p.x, p.y - r - 6);

      // Discovery status pill
      ctx.fillStyle = obj.discovered ? '#10b981' : '#64748b';
      ctx.font = '500 9px monospace';
      ctx.fillText(obj.discovered ? '● SIGHTED' : '○ UNSEEN', p.x, p.y + r + 14);

      ctx.restore();
    }
  }

  drawTarget(ctx) {
    const p = this.worldToScreen(this.target.x, this.target.y);
    const pulse = Math.sin(this.pulsePhase) * 5;

    ctx.save();
    // Glowing ring
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.8)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 20 + pulse, 0, Math.PI * 2);
    ctx.stroke();

    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
    ctx.fill();

    ctx.setLineDash([]);
    ctx.fillStyle = '#34d399';
    ctx.font = '700 11px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`Target: ${this.target.name || 'Goal'}`, p.x, p.y - 28);
    ctx.restore();
  }

  drawCameraCone(ctx) {
    const rp = this.worldToScreen(this.robot.x, this.robot.y);
    const fov = this.robot.camera_fov || 1.13;
    const rangePx = (this.robot.camera_range || 3.5) * rp.scale;

    const leftAngle = -this.robot.yaw - fov / 2;
    const rightAngle = -this.robot.yaw + fov / 2;

    ctx.save();
    const grad = ctx.createRadialGradient(rp.x, rp.y, 0, rp.x, rp.y, rangePx);
    grad.addColorStop(0, 'rgba(0, 229, 255, 0.15)');
    grad.addColorStop(1, 'rgba(0, 229, 255, 0.01)');
    ctx.fillStyle = grad;

    ctx.beginPath();
    ctx.moveTo(rp.x, rp.y);
    ctx.arc(rp.x, rp.y, rangePx, leftAngle, rightAngle);
    ctx.closePath();
    ctx.fill();

    // Cone edge lines
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.25)';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();
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

      let strokeColor = 'rgba(0, 229, 255, 0.12)';
      if (range < 0.45) strokeColor = 'rgba(239, 68, 68, 0.4)';
      else if (range < 0.8) strokeColor = 'rgba(245, 158, 11, 0.25)';

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
    const r = (this.robot.radius || 0.18) * p.scale;

    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.rotate(-this.robot.yaw);

    // Chassis
    ctx.fillStyle = '#0f172a';
    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = 'rgba(0, 229, 255, 0.4)';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(0, 0, Math.max(11, r), 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Heading direction
    ctx.fillStyle = '#00e5ff';
    ctx.beginPath();
    ctx.moveTo(r * 0.9, 0);
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
