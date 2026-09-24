/**
 * sim_canvas_3d.js
 * High-Fidelity 3D WebGL Simulation Engine for AERO.
 * Built with Three.js and OrbitControls (inspired by VSArena 3D studio viewport).
 * Renders a full 3D interactive multi-room apartment, 3D mobile robot,
 * rotating LiDAR puck, camera vision cone, and 3D furniture models.
 */

class SimCanvas3DRenderer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Container #${containerId} not found.`);
      return;
    }

    // Coordinate mapping:
    // Simulation (x, y) 2D plane -> Three.js (x, Y_UP=height, z = -y)
    this.cameraMode = 'orbit'; // 'orbit', 'top', 'follow'

    // Scene & Renderer
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x070b14);
    this.scene.fog = new THREE.FogExp2(0x070b14, 0.035);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;

    // Remove any previous canvas and append WebGL canvas
    const oldCanvas = this.container.querySelector('canvas');
    if (oldCanvas) oldCanvas.remove();
    this.container.appendChild(this.renderer.domElement);

    // Camera setup
    const aspect = this.container.clientWidth / this.container.clientHeight;
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 100);
    this.camera.position.set(0, 14, 15);

    // OrbitControls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05; // Don't dip below floor
    this.controls.minDistance = 3;
    this.controls.maxDistance = 35;
    this.controls.target.set(0, 0.5, 0);

    // Simulation State
    this.robotPose = { x: 0, y: 0, yaw: 0, linear_vel: 0, angular_vel: 0 };
    this.targetPose = { x: 3.2, y: 3.0, name: 'Red Sofa' };
    this.lidarRanges = [];
    this.trajectoryPoints = [new THREE.Vector3(0, 0.05, 0)];

    // Initialize 3D Environment
    this.setupLighting();
    this.setupFloorAndGrid();
    this.setupWalls();
    this.setupFurnitureModels();
    this.setupRobotModel();
    this.setupTargetMarker();
    this.setupTrajectoryRibbon();

    // Auto-resize
    window.addEventListener('resize', () => this.onWindowResize());

    // Animation Loop
    this.clock = new THREE.Clock();
    this.animate = this.animate.bind(this);
    requestAnimationFrame(this.animate);
  }

  setupLighting() {
    // Ambient soft fill
    const ambient = new THREE.AmbientLight(0xdde5ff, 0.55);
    this.scene.add(ambient);

    // Main studio directional light
    const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
    keyLight.position.set(8, 18, 10);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.width = 2048;
    keyLight.shadow.mapSize.height = 2048;
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 40;
    const d = 12;
    keyLight.shadow.camera.left = -d;
    keyLight.shadow.camera.right = d;
    keyLight.shadow.camera.top = d;
    keyLight.shadow.camera.bottom = -d;
    keyLight.shadow.bias = -0.0005;
    this.scene.add(keyLight);

    // Secondary fill light (cool cyan accent)
    const fillLight = new THREE.DirectionalLight(0x00e5ff, 0.35);
    fillLight.position.set(-10, 10, -10);
    this.scene.add(fillLight);

    // Subtle blue under-glow point light
    const pointLight = new THREE.PointLight(0x00e5ff, 0.8, 15);
    pointLight.position.set(0, 1, 0);
    this.scene.add(pointLight);
  }

  setupFloorAndGrid() {
    // Dark metallic floor
    const floorGeo = new THREE.PlaneGeometry(16, 14);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x090f1d,
      roughness: 0.65,
      metalness: 0.25,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    // Grid Helper
    const grid = new THREE.GridHelper(16, 16, 0x00e5ff, 0x1e293b);
    grid.position.y = 0.01;
    this.scene.add(grid);
  }

  setupWalls() {
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.8,
      metalness: 0.1,
    });

    const createWall = (x, z, w, d, h = 1.4) => {
      const geo = new THREE.BoxGeometry(w, h, d);
      const mesh = new THREE.Mesh(geo, wallMat);
      mesh.position.set(x, h / 2, z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.scene.add(mesh);
      return mesh;
    };

    // Outer boundary walls: 12m x 10m
    createWall(0, -5, 12, 0.2); // North (z = -5)
    createWall(0, 5, 12, 0.2);  // South (z = 5)
    createWall(-6, 0, 0.2, 10); // West (x = -6)
    createWall(6, 0, 0.2, 10);  // East (x = 6)

    // Interior divider 1: Living Room / Kitchen (doorway in middle)
    createWall(0, -3.2, 0.2, 3.6);
    // Interior divider 2: Storage Bay / Dock
    createWall(0, 3.2, 0.2, 3.6);
  }

  setupFurnitureModels() {
    // 1. 🛋️ Modern Red Sofa at (3.2, 3.0) -> (3.2, z = -3.0)
    const sofaGroup = new THREE.Group();
    const sofaMat = new THREE.MeshStandardMaterial({ color: 0xdc2626, roughness: 0.5 }); // Velvet red
    const frameMat = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.7 });

    // Base
    const base = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.25, 0.85), sofaMat);
    base.position.y = 0.2;
    base.castShadow = true;
    sofaGroup.add(base);

    // Backrest
    const back = new THREE.Mesh(new THREE.BoxGeometry(1.8, 0.55, 0.22), sofaMat);
    back.position.set(0, 0.55, -0.32);
    back.castShadow = true;
    sofaGroup.add(back);

    // Armrests
    const armL = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.45, 0.85), sofaMat);
    armL.position.set(-0.9, 0.42, 0);
    sofaGroup.add(armL);
    const armR = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.45, 0.85), sofaMat);
    armR.position.set(0.9, 0.42, 0);
    sofaGroup.add(armR);

    // Cushions
    const c1 = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.12, 0.65), sofaMat);
    c1.position.set(-0.42, 0.38, 0.05);
    sofaGroup.add(c1);
    const c2 = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.12, 0.65), sofaMat);
    c2.position.set(0.42, 0.38, 0.05);
    sofaGroup.add(c2);

    sofaGroup.position.set(3.2, 0, -3.0);
    this.scene.add(sofaGroup);

    // 2. 🪑 Kitchen Dining Table at (-2.2, 2.5) -> (-2.2, z = -2.5)
    const tableGroup = new THREE.Group();
    const woodMat = new THREE.MeshStandardMaterial({ color: 0x3b82f6, roughness: 0.4 }); // Sleek blue top
    const legMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.5 });

    const top = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.08, 1.1), woodMat);
    top.position.y = 0.75;
    top.castShadow = true;
    tableGroup.add(top);

    const legGeo = new THREE.CylinderGeometry(0.04, 0.04, 0.75, 12);
    const legCoords = [[-0.65, -0.42], [0.65, -0.42], [-0.65, 0.42], [0.65, 0.42]];
    legCoords.forEach(([lx, lz]) => {
      const leg = new THREE.Mesh(legGeo, legMat);
      leg.position.set(lx, 0.375, lz);
      leg.castShadow = true;
      tableGroup.add(leg);
    });
    tableGroup.position.set(-2.2, 0, -2.5);
    this.scene.add(tableGroup);

    // 3. 📦 Storage Boxes at (2.8, -1.8) -> (2.8, z = 1.8)
    const boxGroup = new THREE.Group();
    const crateMat = new THREE.MeshStandardMaterial({ color: 0xd97706, roughness: 0.8 }); // Amber cardboard

    const b1 = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.6, 0.7), crateMat);
    b1.position.set(0, 0.3, 0);
    b1.castShadow = true;
    boxGroup.add(b1);

    const b2 = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.45, 0.55), crateMat);
    b2.position.set(0.05, 0.825, 0.05);
    b2.rotation.y = 0.2;
    b2.castShadow = true;
    boxGroup.add(b2);

    boxGroup.position.set(2.8, 0, 1.8);
    this.scene.add(boxGroup);

    // 4. ⚡ Charging Dock at (0.0, -3.2) -> (0.0, z = 3.2)
    const dockGroup = new THREE.Group();
    const dockBaseMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.7 });
    const dockBase = new THREE.Mesh(new THREE.CylinderGeometry(0.55, 0.6, 0.08, 24), dockBaseMat);
    dockBase.position.y = 0.04;
    dockGroup.add(dockBase);

    // Glowing energy ring
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.35, 0.45, 24), ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.085;
    dockGroup.add(ring);

    dockGroup.position.set(0.0, 0, 3.2);
    this.scene.add(dockGroup);
  }

  setupRobotModel() {
    this.robotGroup = new THREE.Group();

    // 1. Chassis
    const chassisMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      metalness: 0.85,
      roughness: 0.25,
    });
    const chassis = new THREE.Mesh(new THREE.CylinderGeometry(0.24, 0.26, 0.12, 32), chassisMat);
    chassis.position.y = 0.12;
    chassis.castShadow = true;
    this.robotGroup.add(chassis);

    // Accent ring
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x00e5ff });
    const ring = new THREE.Mesh(new THREE.TorusGeometry(0.25, 0.015, 12, 32), ringMat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = 0.14;
    this.robotGroup.add(ring);

    // 2. Wheels
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.9 });
    const wheelGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.05, 16);
    wheelGeo.rotateZ(Math.PI / 2);

    this.wheelL = new THREE.Mesh(wheelGeo, wheelMat);
    this.wheelL.position.set(0, 0.09, 0.22);
    this.wheelL.castShadow = true;
    this.robotGroup.add(this.wheelL);

    this.wheelR = new THREE.Mesh(wheelGeo, wheelMat);
    this.wheelR.position.set(0, 0.09, -0.22);
    this.wheelR.castShadow = true;
    this.robotGroup.add(this.wheelR);

    // 3. Top Mounted LiDAR Puck
    const lidarMat = new THREE.MeshStandardMaterial({ color: 0x020617, metalness: 0.9 });
    this.lidarHead = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.07, 24), lidarMat);
    this.lidarHead.position.set(0, 0.21, 0);
    this.robotGroup.add(this.lidarHead);

    // 4. Camera Perception Volumetric Cone (Translucent Cyan)
    const coneGeo = new THREE.ConeGeometry(2.2, 3.5, 24, 1, true);
    // Rotate cone forward (pointing towards +X)
    coneGeo.rotateZ(-Math.PI / 2);
    coneGeo.translate(1.75, 0, 0);
    const coneMat = new THREE.MeshBasicMaterial({
      color: 0x00e5ff,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    this.cameraCone = new THREE.Mesh(coneGeo, coneMat);
    this.cameraCone.position.set(0.1, 0.18, 0);
    this.robotGroup.add(this.cameraCone);

    // 5. Dynamic 3D LiDAR Rays
    this.lidarLineGeo = new THREE.BufferGeometry();
    const lidarLineMat = new THREE.LineBasicMaterial({
      color: 0x00e5ff,
      transparent: true,
      opacity: 0.45,
    });
    this.lidarLines = new THREE.LineSegments(this.lidarLineGeo, lidarLineMat);
    this.scene.add(this.lidarLines);

    this.scene.add(this.robotGroup);
  }

  setupTargetMarker() {
    this.targetGroup = new THREE.Group();

    // Floating Target Beacon
    const beaconMat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
    const beacon = new THREE.Mesh(new THREE.OctahedronGeometry(0.2, 0), beaconMat);
    beacon.position.y = 1.1;
    this.targetGroup.add(beacon);

    // Pulsing floor ring
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.6 });
    const ring = new THREE.Mesh(new THREE.RingGeometry(0.4, 0.5, 32), ringMat);
    ring.rotation.x = -Math.PI / 2;
    ring.position.y = 0.02;
    this.targetGroup.add(ring);

    this.targetBeacon = beacon;
    this.targetRing = ring;

    this.targetGroup.position.set(3.2, 0, -3.0);
    this.scene.add(this.targetGroup);
  }

  setupTrajectoryRibbon() {
    this.trajGeo = new THREE.BufferGeometry().setFromPoints(this.trajectoryPoints);
    const trajMat = new THREE.LineBasicMaterial({
      color: 0x00e5ff,
      linewidth: 2,
      transparent: true,
      opacity: 0.7,
    });
    this.trajLine = new THREE.Line(this.trajGeo, trajMat);
    this.scene.add(this.trajLine);
  }

  updateFrame(frame) {
    if (!frame) return;

    // Update Robot
    if (frame.robot) {
      this.robotPose = frame.robot;
      const x = frame.robot.x;
      const z = -frame.robot.y; // Invert for Three.js
      this.robotGroup.position.set(x, 0, z);
      this.robotGroup.rotation.y = frame.robot.yaw;

      // Wheel spin
      const spin = (frame.robot.linear_vel || 0) * 0.4;
      this.wheelL.rotation.x += spin;
      this.wheelR.rotation.x += spin;

      // Trajectory update
      this.trajectoryPoints.push(new THREE.Vector3(x, 0.04, z));
      if (this.trajectoryPoints.length > 500) this.trajectoryPoints.shift();
      this.trajGeo.setFromPoints(this.trajectoryPoints);

      // Camera Follow Mode
      if (this.cameraMode === 'follow') {
        const offset = new THREE.Vector3(-2.8 * Math.cos(frame.robot.yaw), 1.6, 2.8 * Math.sin(frame.robot.yaw));
        this.camera.position.lerp(new THREE.Vector3(x, 0, z).add(offset), 0.1);
        this.controls.target.set(x, 0.3, z);
      }
    }

    // Update Target
    if (frame.target) {
      this.targetPose = frame.target;
      this.targetGroup.position.set(frame.target.x, 0, -frame.target.y);
    }

    // Update 3D LiDAR Rays
    if (frame.lidar_ranges && frame.lidar_ranges.length > 0) {
      this.lidarRanges = frame.lidar_ranges;
      const positions = [];
      const rx = this.robotPose.x;
      const rz = -this.robotPose.y;
      const ry = 0.22;
      const n = this.lidarRanges.length;

      for (let i = 0; i < n; i += 2) { // Render half rays for optimal 60 FPS
        const r = this.lidarRanges[i];
        const angle = this.robotPose.yaw + (2 * Math.PI * i / n);
        const ex = rx + r * Math.cos(angle);
        const ez = rz - r * Math.sin(angle);

        positions.push(rx, ry, rz);
        positions.push(ex, ry, ez);
      }
      this.lidarLineGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    }
  }

  setCameraView(mode) {
    this.cameraMode = mode;
    if (mode === 'top') {
      this.camera.position.set(0, 20, 0.01);
      this.controls.target.set(0, 0, 0);
    } else if (mode === 'orbit') {
      this.camera.position.set(0, 14, 15);
      this.controls.target.set(0, 0.5, 0);
    }
    this.controls.update();
  }

  onWindowResize() {
    if (!this.container) return;
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  animate() {
    requestAnimationFrame(this.animate);

    const elapsed = this.clock.getElapsedTime();

    // Rotate LiDAR puck head
    if (this.lidarHead) {
      this.lidarHead.rotation.y = elapsed * 10.0;
    }

    // Bob target beacon
    if (this.targetBeacon) {
      this.targetBeacon.position.y = 1.0 + Math.sin(elapsed * 3.0) * 0.15;
      this.targetBeacon.rotation.y = elapsed * 2.0;
    }
    if (this.targetRing) {
      const s = 1.0 + Math.sin(elapsed * 4.0) * 0.15;
      this.targetRing.scale.set(s, s, s);
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

window.SimCanvas3DRenderer = SimCanvas3DRenderer;
