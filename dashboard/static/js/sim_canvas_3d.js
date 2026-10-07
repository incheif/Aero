/**
 * Three.js 3D Viewport & Panda-style Cobot Kinematic Renderer
 * Directly ports and enhances the system from new_project_1 (RobotArm.tsx & ArenaCanvas.tsx).
 */

class SimCanvas3D {
  constructor(canvasElement) {
    this.canvas = canvasElement;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;

    // Kinematic Robot Arm Groups
    this.arm = {
      base: null,
      yaw: null,
      shoulder: null,
      elbow: null,
      wrist: null,
      palm: null,
      jawLeft: null,
      jawRight: null,
      tcpMarker: null,
    };

    // Scene Meshes
    this.cubes = {};
    this.targetPad = null;
    this.tcpTrailLine = null;
    this.trailPoints = [];

    // Camera Rigs
    this.activeCameraView = 'orbit'; // 'orbit', 'top', 'wrist', 'table'
    this.cameraRigs = {
      orbit: { pos: [2.35, 1.65, 2.10], target: [0.08, 0.78, 0.0] },
      top: { pos: [0.08, 3.45, 0.02], target: [0.08, 0.72, 0.0] },
      table: { pos: [1.35, 1.25, 1.05], target: [0.48, 0.75, 0.22] },
    };

    // Display Toggles
    this.showTrails = true;
    this.showSemanticLabels = true;
    this.showColliders = false;

    // Colors matching monochromatic noir palette
    this.PALETTE = {
      WHITE: 0xf4f4f5,
      BLACK: 0x18181b,
      METAL: 0xa1a1aa,
      RUBBER: 0x27272a,
      GRAPHITE: 0x3f3f46,
      TRAIL: 0xffffff,
    };

    this.init();
  }

  init() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;

    // 1. Three.js Scene & Fog (Pure Obsidian)
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x09090b);
    this.scene.fog = new THREE.Fog(0x09090b, 8, 22);

    // 2. Camera & Controls (40 deg FOV for well-proportioned framing)
    this.camera = new THREE.PerspectiveCamera(40, width / height, 0.04, 25);
    const initialRig = this.cameraRigs.orbit;
    this.camera.position.set(...initialRig.pos);

    // 3. WebGL Renderer with High-Performance Tone Mapping
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.35;

    // Orbit Controls
    if (typeof THREE.OrbitControls !== 'undefined') {
      this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
      this.controls.enableDamping = true;
      this.controls.dampingFactor = 0.08;
      this.controls.target.set(...initialRig.target);
      this.controls.minDistance = 0.6;
      this.controls.maxDistance = 5.0;
      this.controls.maxPolarAngle = Math.PI / 2 - 0.04;
    }

    // 4. Studio Lighting (StudioEnv from new_project_1)
    this.setupLighting();

    // 5. Studio Environment: Floor, Table, Target Zone
    this.setupEnvironment();

    // 6. Articulated Panda Robot Arm (RobotArm.tsx from new_project_1)
    this.buildRobotArm();

    // 7. TCP Trajectory Trail
    this.setupTrailLine();

    // Window Resize Listener
    window.addEventListener('resize', () => this.onWindowResize());

    // Animation Loop
    this.animate();
  }

  setupLighting() {
    const hemiLight = new THREE.HemisphereLight(0xd4d4d8, 0x18181b, 0.7);
    this.scene.add(hemiLight);

    const ambLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambLight);

    // Key Spotlight casting sharp/soft shadows
    const keySpot = new THREE.SpotLight(0xffffff, 4.2);
    keySpot.position.set(1.8, 3.4, 1.7);
    keySpot.angle = 0.65;
    keySpot.penumbra = 0.55;
    keySpot.castShadow = true;
    keySpot.shadow.mapSize.width = 2048;
    keySpot.shadow.mapSize.height = 2048;
    keySpot.shadow.bias = -0.0006;
    this.scene.add(keySpot);

    // Fill Spotlight
    const fillSpot = new THREE.SpotLight(0xd4d4d8, 1.8);
    fillSpot.position.set(-2.0, 2.8, 1.2);
    fillSpot.angle = 0.75;
    fillSpot.penumbra = 0.8;
    this.scene.add(fillSpot);

    // Pure white rim light
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.85);
    rimLight.position.set(-1.6, 2.2, -2.0);
    this.scene.add(rimLight);
  }

  setupEnvironment() {
    // 1. Grid Floor (Monochrome Zinc Grid)
    const gridHelper = new THREE.GridHelper(16, 32, 0x3f3f46, 0x18181b);
    gridHelper.position.y = 0.001;
    this.scene.add(gridHelper);

    // Floor Contact Shadow Plane (Pure Noir)
    const floorGeo = new THREE.PlaneGeometry(16, 16);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x09090b,
      roughness: 0.95,
      metalness: 0.05,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    // 2. Workcell Table (dimensions: 1.4m x 0.08m x 0.9m at Y=0.72m top)
    const tableTopGeo = new THREE.BoxGeometry(1.4, 0.08, 0.9);
    const tableMat = new THREE.MeshStandardMaterial({
      color: 0x18181b,
      roughness: 0.4,
      metalness: 0.3,
    });
    const tableTop = new THREE.Mesh(tableTopGeo, tableMat);
    tableTop.position.set(0.0, 0.68, 0.0);
    tableTop.receiveShadow = true;
    tableTop.castShadow = true;
    this.scene.add(tableTop);

    // Table Rim Bevel Trim (Zinc Metallic Accent)
    const trimGeo = new THREE.BoxGeometry(1.42, 0.02, 0.92);
    const trimMat = new THREE.MeshStandardMaterial({
      color: 0x27272a,
      roughness: 0.25,
      metalness: 0.7,
    });
    const trim = new THREE.Mesh(trimGeo, trimMat);
    trim.position.set(0.0, 0.71, 0.0);
    this.scene.add(trim);

    // 4 Brushed Steel Table Legs
    const legGeo = new THREE.CylinderGeometry(0.032, 0.032, 0.64, 16);
    const legMat = new THREE.MeshStandardMaterial({ color: 0x52525b, metalness: 0.85, roughness: 0.3 });
    const legOffsets = [
      [0.64, 0.32, 0.39],
      [-0.64, 0.32, 0.39],
      [0.64, 0.32, -0.39],
      [-0.64, 0.32, -0.39],
    ];
    legOffsets.forEach(([lx, ly, lz]) => {
      const leg = new THREE.Mesh(legGeo, legMat);
      leg.position.set(lx, ly, lz);
      leg.castShadow = true;
      this.scene.add(leg);
    });

    // 3. Target Zone Landing Pad (Charcoal disc on table with pure white luminous ring)
    const padGroup = new THREE.Group();
    padGroup.position.set(0.48, 0.721, 0.22);

    const padGeo = new THREE.CylinderGeometry(0.048, 0.048, 0.003, 32);
    const padMat = new THREE.MeshStandardMaterial({
      color: 0x27272a,
      roughness: 0.6,
      metalness: 0.2,
    });
    const padMesh = new THREE.Mesh(padGeo, padMat);
    padMesh.receiveShadow = true;
    padGroup.add(padMesh);

    // Glowing Concentric Ring (Pure White Beacon)
    const ringGeo = new THREE.RingGeometry(0.042, 0.047, 32);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.8,
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = -Math.PI / 2;
    ringMesh.position.y = 0.002;
    padGroup.add(ringMesh);

    this.scene.add(padGroup);
    this.targetPad = padGroup;
  }

  buildRobotArm() {
    // Mount position: [-0.28, 0.72, 0]
    const root = new THREE.Group();
    root.position.set(-0.28, 0.72, 0.0);
    this.arm.base = root;

    // Pedestal Base (black flanged cylinder with bolts)
    const baseMat = new THREE.MeshStandardMaterial({ color: this.PALETTE.BLACK, roughness: 0.6 });
    const metalMat = new THREE.MeshStandardMaterial({ color: this.PALETTE.METAL, metalness: 0.85, roughness: 0.25 });
    const whiteMat = new THREE.MeshStandardMaterial({ color: this.PALETTE.WHITE, roughness: 0.35 });
    const rubberMat = new THREE.MeshStandardMaterial({ color: this.PALETTE.RUBBER, roughness: 0.9 });

    const pedestalBase = new THREE.Mesh(new THREE.CylinderGeometry(0.096, 0.096, 0.036, 32), baseMat);
    pedestalBase.position.y = 0.018;
    pedestalBase.castShadow = true;
    pedestalBase.receiveShadow = true;
    root.add(pedestalBase);

    // Pedestal Taper
    const pedestalTaper = new THREE.Mesh(new THREE.CylinderGeometry(0.044, 0.068, 0.066, 32), baseMat);
    pedestalTaper.position.y = 0.069;
    pedestalTaper.castShadow = true;
    root.add(pedestalTaper);

    // Turntable Flange
    const flange = new THREE.Mesh(new THREE.CylinderGeometry(0.046, 0.046, 0.008, 32), metalMat);
    flange.position.y = 0.106;
    root.add(flange);

    // 1. Base Yaw Turntable Group (at Pedestal Height 0.11m)
    const yawGroup = new THREE.Group();
    yawGroup.position.set(0.0, 0.11, 0.0);
    root.add(yawGroup);
    this.arm.yaw = yawGroup;

    // Shoulder Joint Can
    const shoulderCan = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.038, 0.072, 28), metalMat);
    shoulderCan.rotation.x = Math.PI / 2;
    yawGroup.add(shoulderCan);

    // 2. Shoulder Group (Pitches in Z)
    const shoulderGroup = new THREE.Group();
    yawGroup.add(shoulderGroup);
    this.arm.shoulder = shoulderGroup;

    // Upper Arm Link (length 0.38m along local X)
    const upperLink = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.052, 0.052), whiteMat);
    upperLink.position.set(0.38 * 0.5, 0.0, 0.0);
    upperLink.castShadow = true;
    shoulderGroup.add(upperLink);

    // Conduit Pipe along upper arm
    const conduitUpper = new THREE.Mesh(new THREE.CylinderGeometry(0.004, 0.004, 0.32, 12), metalMat);
    conduitUpper.rotation.z = Math.PI / 2;
    conduitUpper.position.set(0.19, 0.032, 0.0);
    shoulderGroup.add(conduitUpper);

    // 3. Elbow Group (at X=0.38m of upper arm)
    const elbowGroup = new THREE.Group();
    elbowGroup.position.set(0.38, 0.0, 0.0);
    shoulderGroup.add(elbowGroup);
    this.arm.elbow = elbowGroup;

    // Elbow Joint Can
    const elbowCan = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, 0.066, 28), metalMat);
    elbowCan.rotation.x = Math.PI / 2;
    elbowGroup.add(elbowCan);

    // Forearm Link (length 0.32m along local X)
    const forearmLink = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.044, 0.044), whiteMat);
    forearmLink.position.set(0.32 * 0.5, 0.0, 0.0);
    forearmLink.castShadow = true;
    elbowGroup.add(forearmLink);

    // 4. Wrist Group (at X=0.32m of forearm)
    const wristGroup = new THREE.Group();
    wristGroup.position.set(0.32, 0.0, 0.0);
    elbowGroup.add(wristGroup);
    this.arm.wrist = wristGroup;

    // Wrist Joint Can
    const wristCan = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.054, 24), metalMat);
    wristCan.rotation.x = Math.PI / 2;
    wristGroup.add(wristCan);

    // Palm / Gripper Body (length 0.13m to pinch point)
    const palm = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.042, 0.065), baseMat);
    palm.position.set(0.035, 0.0, 0.0);
    palm.castShadow = true;
    wristGroup.add(palm);
    this.arm.palm = palm;

    // 5. Parallel Gripper Sliding Jaws
    const jawGeo = new THREE.BoxGeometry(0.08, 0.024, 0.012);

    // Left Jaw
    const jawLeft = new THREE.Mesh(jawGeo, whiteMat);
    jawLeft.position.set(0.075, 0.0, 0.045);
    jawLeft.castShadow = true;
    wristGroup.add(jawLeft);
    this.arm.jawLeft = jawLeft;

    // Left Jaw Rubber Gripper Pad
    const padLeft = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.02, 0.004), rubberMat);
    padLeft.position.set(0.02, 0.0, -0.008);
    jawLeft.add(padLeft);

    // Right Jaw
    const jawRight = new THREE.Mesh(jawGeo, whiteMat);
    jawRight.position.set(0.075, 0.0, -0.045);
    jawRight.castShadow = true;
    wristGroup.add(jawRight);
    this.arm.jawRight = jawRight;

    // Right Jaw Rubber Gripper Pad
    const padRight = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.02, 0.004), rubberMat);
    padRight.position.set(0.02, 0.0, 0.008);
    jawRight.add(padRight);

    // 6. TCP Glowing Indicator (mid-finger pinch point)
    const tcpGroup = new THREE.Group();
    tcpGroup.position.set(0.18, 0.0, 0.0);

    const tcpSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.006, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    tcpGroup.add(tcpSphere);
    wristGroup.add(tcpGroup);
    this.arm.tcpMarker = tcpGroup;

    this.scene.add(root);
  }

  setupTrailLine() {
    const maxPoints = 120;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(maxPoints * 3);
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.LineBasicMaterial({
      color: this.PALETTE.TRAIL,
      transparent: true,
      opacity: 0.8,
      linewidth: 2,
    });

    this.tcpTrailLine = new THREE.Line(geometry, material);
    this.scene.add(this.tcpTrailLine);
  }

  updateFromTelemetry(data) {
    if (!data) return;

    // 1. Update Arm Joints from FK
    const joints = data.joints;
    if (joints && this.arm.yaw) {
      this.arm.yaw.rotation.y = joints.baseYaw || 0;
      this.arm.shoulder.rotation.z = joints.shoulderPitch || 0;
      this.arm.elbow.rotation.z = joints.elbowPitch || 0;
      this.arm.wrist.rotation.z = joints.wristPitch || 0;

      // Gripper Jaw Separation
      const grip = joints.gripper || 0.0;
      const minSep = 0.069 / 2;
      const maxSep = 0.125 / 2;
      const sep = minSep + (maxSep - minSep) * (1.0 - grip);

      if (this.arm.jawLeft) this.arm.jawLeft.position.z = sep;
      if (this.arm.jawRight) this.arm.jawRight.position.z = -sep;
    }

    // 2. Update Dynamic Cubes
    const blocks = data.blocks || [];
    blocks.forEach((b) => {
      let mesh = this.cubes[b.id];
      if (!mesh) {
        // Create Cube with bevel
        const geo = new THREE.BoxGeometry(0.055, 0.055, 0.055);
        const hex = parseInt(b.color.replace('#', '0x'), 16);
        const mat = new THREE.MeshStandardMaterial({
          color: hex,
          roughness: 0.25,
          metalness: 0.15,
        });
        mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        this.scene.add(mesh);
        this.cubes[b.id] = mesh;
      }

      // Smooth position update
      mesh.position.set(b.position[0], b.position[1], b.position[2]);
      if (b.rotation) {
        mesh.quaternion.set(b.rotation[0], b.rotation[1], b.rotation[2], b.rotation[3]);
      }
    });

    // 3. Update TCP Trail Line
    if (this.showTrails && data.tcp_trail && data.tcp_trail.length > 1) {
      const pts = data.tcp_trail;
      const positions = this.tcpTrailLine.geometry.attributes.position.array;
      for (let i = 0; i < pts.length; i++) {
        positions[i * 3] = pts[i][0];
        positions[i * 3 + 1] = pts[i][1];
        positions[i * 3 + 2] = pts[i][2];
      }
      this.tcpTrailLine.geometry.setDrawRange(0, pts.length);
      this.tcpTrailLine.geometry.attributes.position.needsUpdate = true;
    }

    // 4. Update Wrist Camera if active
    if (this.activeCameraView === 'wrist' && data.tcp) {
      const tcpPos = data.tcp.position;
      this.camera.position.set(tcpPos[0] - 0.22, tcpPos[1] + 0.18, tcpPos[2]);
      this.camera.lookAt(tcpPos[0], tcpPos[1], tcpPos[2]);
    }
  }

  setCameraView(viewName) {
    this.activeCameraView = viewName;
    if (viewName === 'wrist') {
      if (this.controls) this.controls.enabled = false;
      return;
    }

    if (this.controls) this.controls.enabled = true;
    const rig = this.cameraRigs[viewName] || this.cameraRigs.orbit;

    // Smooth transition
    this.camera.position.set(...rig.pos);
    if (this.controls) {
      this.controls.target.set(...rig.target);
      this.controls.update();
    }
  }

  zoomIn(factor = 1.2) {
    if (!this.controls) return;
    this.controls.dollyIn(factor);
    this.controls.update();
  }

  zoomOut(factor = 1.2) {
    if (!this.controls) return;
    this.controls.dollyOut(factor);
    this.controls.update();
  }

  resetCamera() {
    this.setCameraView(this.activeCameraView || 'orbit');
  }

  toggleTrails() {
    this.showTrails = !this.showTrails;
    if (this.tcpTrailLine) this.tcpTrailLine.visible = this.showTrails;
    return this.showTrails;
  }

  onWindowResize() {
    if (!this.canvas || !this.renderer || !this.camera) return;
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    if (this.controls && this.controls.enabled) {
      this.controls.update();
    }

    // Subtle pulsing of target pad ring
    if (this.targetPad) {
      const ring = this.targetPad.children[1];
      if (ring && ring.material) {
        ring.material.opacity = 0.5 + Math.sin(Date.now() * 0.003) * 0.25;
      }
    }

    this.renderer.render(this.scene, this.camera);
  }
}

window.SimCanvas3D = SimCanvas3D;
