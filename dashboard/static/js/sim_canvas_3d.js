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
    this.activeCameraView = 'orbit'; // 'orbit', 'play', 'top', 'wrist', 'table'
    this.cameraRigs = {
      orbit: { pos: [2.35, 1.65, 2.10], target: [0.08, 0.78, 0.0] },
      play: { pos: [1.68, 1.38, 1.52], target: [0.20, 0.76, 0.10] },
      top: { pos: [0.08, 3.45, 0.02], target: [0.08, 0.72, 0.0] },
      table: { pos: [1.35, 1.25, 1.05], target: [0.48, 0.75, 0.22] },
    };

    // Display Toggles
    this.showTrails = true;
    this.showSemanticLabels = true;
    this.showColliders = false;
    this.showPerceptionBoxes = true;
    this.showFrustums = true;
    this.perceptionMeshes = {};
    this.sensorFrustumGroup = null;
    this.wristSensorBeam = null;
    this.plannedPathLine = null;
    this.ghostTargetMesh = null;

    // Multi-View Screen Management State (1 = single view, 2+ = split screen)
    this.mainScreens = ['studio'];
    this.paneProvider = null;

    // Live WebGL PiP Sensor Feed State
    this.hudFeedMode = 'rgb';
    this.pipFrameCounter = 0;
    this.pipSize = 160;
    this.pipCanvas = null;
    this.pipCtx = null;
    this.pipRenderTarget = null;
    this.pipReadBuffer = null;
    this.pipImageData = null;
    this.overheadSensorCam = null;
    this.wristSensorCam = null;
    this.depthMaterial = null;

    // Performance Mode ('smooth' 60 FPS optimized for laptops, or 'hd' for high-end GPUs)
    this.perfMode = localStorage.getItem('AERO_PERF_MODE') || 'smooth';

    // Kinematic joint interpolation for judder-free 60 FPS motion
    this.targetJoints = null;
    this.currentJoints = null;
    this.cubeTargets = {};

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
    const container = this.canvas.parentElement || this.canvas;
    const width = container.clientWidth || window.innerWidth || 960;
    const height = container.clientHeight || window.innerHeight || 540;

    // 1. Three.js Scene & Fog (Pure Obsidian)
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x09090b);
    this.scene.fog = new THREE.Fog(0x09090b, 8, 22);

    // 2. Camera & Controls (40 deg FOV for well-proportioned framing)
    this.camera = new THREE.PerspectiveCamera(40, width / height, 0.04, 25);
    const initialRig = this.cameraRigs.orbit;
    this.camera.position.set(...initialRig.pos);

    // 3. WebGL Renderer with Laptop-Optimized Fillrate and Tone Mapping
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    this.renderer.setSize(width, height, false);

    // Smooth mode limits DPR to 1.2 to eliminate GPU fillrate bottlenecks on laptops
    const dprCap = this.perfMode === 'smooth' ? 1.2 : 1.75;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, dprCap));

    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = this.perfMode === 'smooth' ? THREE.PCFShadowMap : THREE.PCFSoftShadowMap;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;

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

    // 7. TCP Trajectory Trail & Planned Preview
    this.setupTrailLine();
    this.setupPlannedPreview();

    // 8. 3D Sensor Visualizations & Live Sensor Cameras
    this.setupSensorVisualizations();
    this.setupSensorCameras();

    // Dynamic Viewport Resize Observer for seamless full-space filling
    if (window.ResizeObserver && this.canvas.parentElement) {
      this.resizeObserver = new ResizeObserver(() => {
        this.onWindowResize();
      });
      this.resizeObserver.observe(this.canvas.parentElement);
    }
    window.addEventListener('resize', () => this.onWindowResize());

    // Trigger initial adjustment once DOM is fully rendered
    setTimeout(() => this.onWindowResize(), 60);

    // Animation Loop
    this.animate();
  }

  setupLighting() {
    const hemiLight = new THREE.HemisphereLight(0xd4d4d8, 0x18181b, 0.7);
    this.scene.add(hemiLight);

    const ambLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambLight);

    // Key Spotlight casting sharp shadows (balanced intensity to preserve vibrant saturation)
    const keySpot = new THREE.SpotLight(0xffffff, 2.2);
    keySpot.position.set(1.8, 3.4, 1.7);
    keySpot.angle = 0.65;
    keySpot.penumbra = 0.55;
    keySpot.castShadow = true;
    const shadowSize = this.perfMode === 'smooth' ? 1024 : 2048;
    keySpot.shadow.mapSize.width = shadowSize;
    keySpot.shadow.mapSize.height = shadowSize;
    keySpot.shadow.bias = -0.0008;
    keySpot.shadow.camera.near = 0.5;
    keySpot.shadow.camera.far = 6.0;
    this.scene.add(keySpot);
    this.keySpot = keySpot;

    // Fill Spotlight
    const fillSpot = new THREE.SpotLight(0xd4d4d8, 1.0);
    fillSpot.position.set(-2.0, 2.8, 1.2);
    fillSpot.angle = 0.75;
    fillSpot.penumbra = 0.8;
    this.scene.add(fillSpot);

    // Pure white rim light
    const rimLight = new THREE.DirectionalLight(0xffffff, 0.55);
    rimLight.position.set(-1.6, 2.2, -2.0);
    this.scene.add(rimLight);
  }

  setupEnvironment() {
    // 1. Grid Floor (Monochrome Zinc Grid)
    const gridHelper = new THREE.GridHelper(16, 32, 0x3f3f46, 0x18181b);
    gridHelper.position.y = 0.001;
    gridHelper.matrixAutoUpdate = false;
    gridHelper.updateMatrix();
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
    floor.matrixAutoUpdate = false;
    floor.updateMatrix();
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
    tableTop.matrixAutoUpdate = false;
    tableTop.updateMatrix();
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
    trim.matrixAutoUpdate = false;
    trim.updateMatrix();
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
      leg.matrixAutoUpdate = false;
      leg.updateMatrix();
      this.scene.add(leg);
    });

    // 3. ESD Workbench Lab Mat (5 cm calibration grid across active workcell)
    const matTexCanvas = document.createElement('canvas');
    matTexCanvas.width = 512;
    matTexCanvas.height = 384;
    const mctx = matTexCanvas.getContext('2d');
    mctx.fillStyle = '#14171f';
    mctx.fillRect(0, 0, 512, 384);
    mctx.strokeStyle = 'rgba(56, 189, 248, 0.14)';
    mctx.lineWidth = 1;
    const stepX = 512 / 18;
    const stepY = 384 / 14;
    for (let i = 1; i < 18; i++) {
      mctx.beginPath();
      mctx.moveTo(i * stepX, 12);
      mctx.lineTo(i * stepX, 372);
      mctx.stroke();
    }
    for (let j = 1; j < 14; j++) {
      mctx.beginPath();
      mctx.moveTo(12, j * stepY);
      mctx.lineTo(500, j * stepY);
      mctx.stroke();
    }
    mctx.strokeStyle = 'rgba(56, 189, 248, 0.35)';
    mctx.lineWidth = 2;
    mctx.strokeRect(12, 12, 488, 360);
    // Subtle center workcell reticle
    mctx.beginPath();
    mctx.moveTo(246, 192);
    mctx.lineTo(266, 192);
    mctx.moveTo(256, 182);
    mctx.lineTo(256, 202);
    mctx.stroke();

    const matTex = new THREE.CanvasTexture(matTexCanvas);
    matTex.anisotropy = 4;
    const labMat = new THREE.Mesh(
      new THREE.PlaneGeometry(0.92, 0.70),
      new THREE.MeshStandardMaterial({
        map: matTex,
        roughness: 0.72,
        metalness: 0.08,
      })
    );
    labMat.rotation.x = -Math.PI / 2;
    labMat.position.set(0.04, 0.7204, 0.0);
    labMat.receiveShadow = true;
    labMat.matrixAutoUpdate = false;
    labMat.updateMatrix();
    this.scene.add(labMat);

    // 4. Overhead Stereo Camera Bar & Vertical Stanchion Rig at (0.08, 1.85, 0.0)
    const camRigGroup = new THREE.Group();
    const rigMetalMat = new THREE.MeshStandardMaterial({ color: 0x27272a, metalness: 0.75, roughness: 0.3 });
    const lensGlassMat = new THREE.MeshStandardMaterial({ color: 0x090d16, metalness: 0.9, roughness: 0.1 });

    // Vertical rear mast from back table edge up to camera bar
    const rearPost = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.014, 1.16, 14), rigMetalMat);
    rearPost.position.set(0.08, 1.30, -0.43);
    camRigGroup.add(rearPost);
    // Horizontal cantilever arm to (0.08, 1.86, 0.0)
    const boomArm = new THREE.Mesh(new THREE.BoxGeometry(0.024, 0.022, 0.45), rigMetalMat);
    boomArm.position.set(0.08, 1.87, -0.215);
    camRigGroup.add(boomArm);
    // Stereo camera bar housing (sits at Y=1.855, above optical center Y=1.835)
    const camBar = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.024, 0.036), rigMetalMat);
    camBar.position.set(0.08, 1.855, 0.0);
    camRigGroup.add(camBar);
    [-0.042, 0.042].forEach((lx) => {
      const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.009, 0.009, 0.012, 16), lensGlassMat);
      lens.position.set(0.08 + lx, 1.842, 0.0);
      camRigGroup.add(lens);
    });
    const camStatusDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.004, 10, 10),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
    );
    camStatusDot.position.set(0.08, 1.855, 0.020);
    camRigGroup.add(camStatusDot);
    this.scene.add(camRigGroup);

    // 5. Clean, unmarked tabletop (all target spot decals removed)
    this.targetPads = [];
    this.targetPad = null;
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

    // Wrist Link Extension (L_WRIST = 0.13m from wrist joint axis to palm)
    const wristLink = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.018, 0.115, 24), metalMat);
    wristLink.rotation.z = Math.PI / 2;
    wristLink.position.set(0.065, 0.0, 0.0);
    wristLink.castShadow = true;
    wristGroup.add(wristLink);

    // Palm / Gripper Base Plate (at X = 0.13m)
    const palm = new THREE.Mesh(new THREE.BoxGeometry(0.024, 0.046, 0.072), baseMat);
    palm.position.set(0.13, 0.0, 0.0);
    palm.castShadow = true;
    wristGroup.add(palm);
    this.arm.palm = palm;

    // Eye-in-Hand Wrist Camera Pod + Status LED Ring mounted on top of palm
    const wristPod = new THREE.Mesh(
      new THREE.BoxGeometry(0.032, 0.018, 0.034),
      new THREE.MeshStandardMaterial({ color: 0x18181b, metalness: 0.7, roughness: 0.3 })
    );
    wristPod.position.set(0.122, 0.032, 0.0);
    wristGroup.add(wristPod);

    const wristLens = new THREE.Mesh(
      new THREE.CylinderGeometry(0.0065, 0.0065, 0.008, 14),
      new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9, roughness: 0.1 })
    );
    wristLens.rotation.z = Math.PI / 2;
    wristLens.position.set(0.139, 0.032, 0.0);
    wristGroup.add(wristLens);

    const statusLedMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x38bdf8,
      emissiveIntensity: 1.4,
      roughness: 0.2,
    });
    const statusLed = new THREE.Mesh(new THREE.SphereGeometry(0.0045, 12, 12), statusLedMat);
    statusLed.position.set(0.124, 0.042, 0.012);
    wristGroup.add(statusLed);
    this.arm.statusLed = statusLed;

    // Mounted Eye-in-Hand PerspectiveCamera looking forward/slightly down between jaws
    const wristCam = new THREE.PerspectiveCamera(58, 1.0, 0.032, 4.0);
    wristCam.position.set(0.142, 0.042, 0.0);
    const wristLookMat = new THREE.Matrix4().lookAt(
      new THREE.Vector3(0.142, 0.042, 0.0),
      new THREE.Vector3(0.265, -0.022, 0.0),
      new THREE.Vector3(0, 1, 0)
    );
    wristCam.quaternion.setFromRotationMatrix(wristLookMat);
    wristGroup.add(wristCam);
    this.wristSensorCam = wristCam;

    // 5. Parallel Gripper Sliding Jaws (JAW_LENGTH = 0.095m, centered at X = 0.18m matching TCP)
    const jawGeo = new THREE.BoxGeometry(0.095, 0.036, 0.012);

    // Left Jaw
    const jawLeft = new THREE.Mesh(jawGeo, whiteMat);
    jawLeft.position.set(0.18, 0.0, 0.045);
    jawLeft.castShadow = true;
    wristGroup.add(jawLeft);
    this.arm.jawLeft = jawLeft;

    // Left Jaw Rubber Gripper Pad (centered on inner face of left jaw at X = 0.18m)
    const padLeft = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.028, 0.004), rubberMat);
    padLeft.position.set(0.0, 0.0, -0.008);
    jawLeft.add(padLeft);

    // Right Jaw
    const jawRight = new THREE.Mesh(jawGeo, whiteMat);
    jawRight.position.set(0.18, 0.0, -0.045);
    jawRight.castShadow = true;
    wristGroup.add(jawRight);
    this.arm.jawRight = jawRight;

    // Right Jaw Rubber Gripper Pad (centered on inner face of right jaw at X = 0.18m)
    const padRight = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.028, 0.004), rubberMat);
    padRight.position.set(0.0, 0.0, 0.008);
    jawRight.add(padRight);

    // 6. TCP Glowing Indicator (mid-finger pinch point: at X = 0.18m, exactly at pad center)
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

  setupPlannedPreview() {
    // Remaining future waypoint trajectory line (emerald dashed/solid cue)
    const maxPts = 64;
    const planGeo = new THREE.BufferGeometry();
    planGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(maxPts * 3), 3));
    const planMat = new THREE.LineBasicMaterial({
      color: 0x10b981,
      transparent: true,
      opacity: 0.72,
    });
    this.plannedPathLine = new THREE.Line(planGeo, planMat);
    this.plannedPathLine.geometry.setDrawRange(0, 0);
    this.scene.add(this.plannedPathLine);

    // Translucent target destination ghost cube
    const ghostGeo = new THREE.BoxGeometry(0.056, 0.056, 0.056);
    const ghostMat = new THREE.MeshBasicMaterial({
      color: 0x10b981,
      transparent: true,
      opacity: 0.16,
      depthWrite: false,
    });
    const ghostMesh = new THREE.Mesh(ghostGeo, ghostMat);
    const ghostEdges = new THREE.LineSegments(
      new THREE.EdgesGeometry(ghostGeo),
      new THREE.LineBasicMaterial({ color: 0x34d399, transparent: true, opacity: 0.75 })
    );
    ghostMesh.add(ghostEdges);
    ghostMesh.visible = false;
    this.scene.add(ghostMesh);
    this.ghostTargetMesh = ghostMesh;
  }

  setupSensorCameras() {
    this.overheadSensorCam = new THREE.PerspectiveCamera(38, 1.0, 0.12, 4.0);
    this.overheadSensorCam.position.set(0.08, 1.835, 0.0);
    this.overheadSensorCam.up.set(0, 0, -1);
    this.overheadSensorCam.lookAt(0.08, 0.72, 0.0);

    this.pipCanvas = document.getElementById('vla-pip-canvas');
    this.pipCtx = this.pipCanvas ? this.pipCanvas.getContext('2d') : null;
    this.pipRenderTarget = new THREE.WebGLRenderTarget(this.pipSize, this.pipSize, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
    });
    this.pipReadBuffer = new Uint8Array(this.pipSize * this.pipSize * 4);
    if (this.pipCtx) {
      this.pipImageData = this.pipCtx.createImageData(this.pipSize, this.pipSize);
    }

    this.depthMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uNear: { value: 0.65 },
        uFar: { value: 1.35 },
      },
      vertexShader: `
        varying float vViewDist;
        void main() {
          vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
          vViewDist = -mvPos.z;
          gl_Position = projectionMatrix * mvPos;
        }
      `,
      fragmentShader: `
        uniform float uNear;
        uniform float uFar;
        varying float vViewDist;
        void main() {
          float t = clamp((uFar - vViewDist) / (uFar - uNear), 0.0, 1.0);
          vec3 col = mix(vec3(0.03, 0.06, 0.14), vec3(0.02, 0.72, 0.84), smoothstep(0.0, 0.45, t));
          col = mix(col, vec3(0.96, 0.72, 0.18), smoothstep(0.45, 0.82, t));
          col = mix(col, vec3(1.0, 0.96, 0.88), smoothstep(0.82, 1.0, t));
          gl_FragColor = vec4(col, 1.0);
        }
      `,
    });
  }

  setupSensorVisualizations() {
    this.sensorFrustumGroup = new THREE.Group();

    // 1. Overhead Camera Frustum (from 0.08, 1.85, 0.0 to table bounds [-0.35, 0.35])
    const camPos = new THREE.Vector3(0.08, 1.85, 0.0);
    const corners = [
      new THREE.Vector3(-0.35, 0.722, -0.35),
      new THREE.Vector3(0.35, 0.722, -0.35),
      new THREE.Vector3(0.35, 0.722, 0.35),
      new THREE.Vector3(-0.35, 0.722, 0.35),
    ];

    const frustumMat = new THREE.LineBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.35,
    });

    const fPoints = [
      camPos, corners[0],
      camPos, corners[1],
      camPos, corners[2],
      camPos, corners[3],
      corners[0], corners[1],
      corners[1], corners[2],
      corners[2], corners[3],
      corners[3], corners[0]
    ];
    const fGeo = new THREE.BufferGeometry().setFromPoints(fPoints);
    const fLines = new THREE.LineSegments(fGeo, frustumMat);
    this.sensorFrustumGroup.add(fLines);

    // Subtle overhead camera icon marker at (0.08, 1.85, 0.0)
    const camIcon = new THREE.Mesh(
      new THREE.BoxGeometry(0.04, 0.02, 0.04),
      new THREE.MeshBasicMaterial({ color: 0x38bdf8 })
    );
    camIcon.position.copy(camPos);
    this.sensorFrustumGroup.add(camIcon);

    // 2. Wrist Camera / Laser Sensor Beam (originates from TCP and shoots down)
    const beamGeo = new THREE.CylinderGeometry(0.003, 0.038, 0.20, 16, 1, true);
    const beamMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.28,
      side: THREE.DoubleSide
    });
    this.wristSensorBeam = new THREE.Mesh(beamGeo, beamMat);
    this.wristSensorBeam.visible = true;
    this.sensorFrustumGroup.add(this.wristSensorBeam);

    this.scene.add(this.sensorFrustumGroup);
  }

  updateFromTelemetry(data) {
    if (!data) return;

    // 1. Update Target Arm Joints for Smooth Frame Interpolation
    const joints = data.joints;
    if (joints) {
      if (!this.currentJoints) {
        this.currentJoints = {
          baseYaw: joints.baseYaw || 0,
          shoulderPitch: joints.shoulderPitch || 0,
          elbowPitch: joints.elbowPitch || 0,
          wristPitch: joints.wristPitch || 0,
          gripper: joints.gripper || 0,
        };
      }
      this.targetJoints = {
        baseYaw: joints.baseYaw || 0,
        shoulderPitch: joints.shoulderPitch || 0,
        elbowPitch: joints.elbowPitch || 0,
        wristPitch: joints.wristPitch || 0,
        gripper: joints.gripper || 0,
      };
    }

    // 2. Update Dynamic Cubes with vivid, high-contrast recognizable colors
    const blocks = data.blocks || [];
    blocks.forEach((b) => {
      let mesh = this.cubes[b.id];
      const hex = parseInt(b.color.replace('#', '0x'), 16);
      const colorObj = new THREE.Color(hex);

      if (!mesh) {
        // Create Cube with high-saturation material and sharp edge highlights
        const geo = new THREE.BoxGeometry(0.055, 0.055, 0.055);
        const mat = new THREE.MeshStandardMaterial({
          color: colorObj,
          roughness: 0.22,
          metalness: 0.0,
          emissive: colorObj,
          emissiveIntensity: 0.38, // ensures rich vivid saturation and zero washed out bleach
        });
        mesh = new THREE.Mesh(geo, mat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.position.set(b.position[0], b.position[1], b.position[2]);

        // Clean white outline edge lines for sharp physical clarity & distinct visibility
        const edges = new THREE.EdgesGeometry(geo);
        const edgeLine = new THREE.LineSegments(
          edges,
          new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.6, linewidth: 1.5 })
        );
        mesh.add(edgeLine);

        this.scene.add(mesh);
        this.cubes[b.id] = mesh;
      } else {
        // Dynamically update material color if changed or re-spawned
        if (mesh.material && mesh.material.color && mesh.material.color.getHex() !== hex) {
          mesh.material.color.setHex(hex);
          if (mesh.material.emissive) {
            mesh.material.emissive.setHex(hex);
            mesh.material.emissiveIntensity = 0.38;
          }
        }
      }

      if (!this.cubeTargets) this.cubeTargets = {};
      this.cubeTargets[b.id] = {
        pos: b.position,
        rot: b.rotation,
      };
    });

    // 3. Update 3D Perception Bounding Volumes (Estimated Track Positions)
    if (this.showPerceptionBoxes && data.perception && data.perception.tracks) {
      const activeTrackIds = new Set();
      data.perception.tracks.forEach((track) => {
        activeTrackIds.add(track.track_id);
        let pMesh = this.perceptionMeshes[track.track_id];
        if (!pMesh) {
          const pGeo = new THREE.BoxGeometry(0.062, 0.062, 0.062);
          const pEdges = new THREE.EdgesGeometry(pGeo);
          const pMat = new THREE.LineBasicMaterial({
            color: 0x38bdf8,
            linewidth: 2,
            transparent: true,
            opacity: 0.9
          });
          pMesh = new THREE.LineSegments(pEdges, pMat);
          this.scene.add(pMesh);
          this.perceptionMeshes[track.track_id] = pMesh;
        }
        if (track.position) {
          pMesh.visible = true;
          pMesh.position.set(track.position[0], track.position[1], track.position[2]);
        }
      });
      // Hide inactive
      Object.keys(this.perceptionMeshes).forEach((tid) => {
        if (!activeTrackIds.has(tid)) {
          this.perceptionMeshes[tid].visible = false;
        }
      });
    } else {
      Object.values(this.perceptionMeshes).forEach((m) => { m.visible = false; });
    }

    // 4. Update TCP Trail Line
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
    } else if (this.tcpTrailLine) {
      this.tcpTrailLine.geometry.setDrawRange(0, 0);
    }

    // 4b. Update Planned Path Preview & Target Ghost Cube
    if (this.plannedPathLine && data.planned_path && data.planned_path.length > 1) {
      const pPts = data.planned_path;
      const count = Math.min(64, pPts.length);
      const posArr = this.plannedPathLine.geometry.attributes.position.array;
      for (let i = 0; i < count; i++) {
        posArr[i * 3] = pPts[i][0];
        posArr[i * 3 + 1] = pPts[i][1];
        posArr[i * 3 + 2] = pPts[i][2];
      }
      this.plannedPathLine.geometry.setDrawRange(0, count);
      this.plannedPathLine.geometry.attributes.position.needsUpdate = true;
      this.plannedPathLine.visible = true;
    } else if (this.plannedPathLine) {
      this.plannedPathLine.geometry.setDrawRange(0, 0);
      this.plannedPathLine.visible = false;
    }

    if (this.ghostTargetMesh) {
      if (data.planned_target && data.planned_target.length >= 3) {
        const gt = data.planned_target;
        this.ghostTargetMesh.position.set(gt[0], Math.max(0.7475, gt[1]), gt[2]);
        this.ghostTargetMesh.visible = true;
      } else {
        this.ghostTargetMesh.visible = false;
      }
    }

    // 4c. Update Wrist Status LED Color (amber grasp / green active / red error / cyan idle)
    if (this.arm.statusLed && this.arm.statusLed.material) {
      const vlaStatus = (data.vla && data.vla.status) ? String(data.vla.status).toUpperCase() : 'IDLE';
      let ledHex = 0x38bdf8;
      if (data.grasped_block_id) ledHex = 0xf59e0b;
      else if (vlaStatus === 'EXECUTING' || vlaStatus === 'PLANNING') ledHex = 0x10b981;
      else if (vlaStatus === 'ERROR') ledHex = 0xef4444;
      this.arm.statusLed.material.color.setHex(ledHex);
      this.arm.statusLed.material.emissive.setHex(ledHex);
    }

    // 4. Update Wrist Camera if active
    if (this.activeCameraView === 'wrist' && data.tcp) {
      const tcpPos = data.tcp.position;
      this.camera.position.set(tcpPos[0] - 0.22, tcpPos[1] + 0.18, tcpPos[2]);
      this.camera.lookAt(tcpPos[0], tcpPos[1], tcpPos[2]);
    }

    // 5. Update Sensor Frustums and Wrist Laser Beam
    if (this.sensorFrustumGroup) {
      this.sensorFrustumGroup.visible = this.showFrustums;
    }
    if (this.wristSensorBeam && data.tcp && data.tcp.position) {
      const tcp = data.tcp.position;
      const alt = Math.max(0.02, tcp[1] - 0.72);
      this.wristSensorBeam.position.set(tcp[0], 0.72 + alt * 0.5, tcp[2]);
      this.wristSensorBeam.scale.set(1.0, alt / 0.20, 1.0);
      let nearBlock = false;
      if (data.blocks) {
        for (let b of data.blocks) {
          const bp = b.position;
          if (Math.hypot(bp[0] - tcp[0], bp[2] - tcp[2]) < 0.06 && Math.abs(bp[1] - tcp[1]) < 0.12) {
            nearBlock = true;
            break;
          }
        }
      }
      this.wristSensorBeam.material.color.setHex(nearBlock ? 0xf59e0b : 0x38bdf8);
      this.wristSensorBeam.material.opacity = nearBlock ? 0.45 : 0.25;
    }
  }

  setCameraView(viewName) {
    this.activeCameraView = viewName;
    const allowOrbit = this.mainScreens.includes('studio') && viewName !== 'wrist';
    if (this.controls) this.controls.enabled = allowOrbit;
    if (viewName === 'wrist') {
      return;
    }

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

  togglePerfMode() {
    this.perfMode = this.perfMode === 'smooth' ? 'hd' : 'smooth';
    localStorage.setItem('AERO_PERF_MODE', this.perfMode);
    const dprCap = this.perfMode === 'smooth' ? 1.2 : 1.75;
    if (this.renderer) {
      this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, dprCap));
      this.renderer.shadowMap.type = this.perfMode === 'smooth' ? THREE.PCFShadowMap : THREE.PCFSoftShadowMap;
      if (this.keySpot) {
        const size = this.perfMode === 'smooth' ? 1024 : 2048;
        this.keySpot.shadow.mapSize.width = size;
        this.keySpot.shadow.mapSize.height = size;
        if (this.keySpot.shadow.map) {
          this.keySpot.shadow.map.dispose();
          this.keySpot.shadow.map = null;
        }
      }
      this.onWindowResize();
    }
    return this.perfMode;
  }

  toggleTrails() {
    this.showTrails = !this.showTrails;
    if (this.tcpTrailLine) this.tcpTrailLine.visible = this.showTrails;
    return this.showTrails;
  }

  onWindowResize() {
    if (!this.canvas || !this.renderer || !this.camera) return;
    const container = this.canvas.parentElement || this.canvas;
    const width = container.clientWidth || this.canvas.clientWidth || window.innerWidth;
    const height = container.clientHeight || this.canvas.clientHeight || window.innerHeight;
    if (width <= 0 || height <= 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    if (this.controls && this.controls.enabled) {
      this.controls.update();
    }

    // Kinematic smoothing: interpolate joint angles for judder-free 60 FPS motion
    if (this.targetJoints && this.currentJoints && this.arm.yaw) {
      const j = this.targetJoints;
      const cur = this.currentJoints;
      const alpha = 0.55;
      cur.baseYaw += (j.baseYaw - cur.baseYaw) * alpha;
      cur.shoulderPitch += (j.shoulderPitch - cur.shoulderPitch) * alpha;
      cur.elbowPitch += (j.elbowPitch - cur.elbowPitch) * alpha;
      cur.wristPitch += (j.wristPitch - cur.wristPitch) * alpha;
      cur.gripper += (j.gripper - cur.gripper) * alpha;

      this.arm.yaw.rotation.y = cur.baseYaw;
      this.arm.shoulder.rotation.z = cur.shoulderPitch;
      this.arm.elbow.rotation.z = cur.elbowPitch;
      this.arm.wrist.rotation.z = cur.wristPitch;

      const minSep = 0.069 / 2;
      const maxSep = 0.125 / 2;
      const sep = minSep + (maxSep - minSep) * (1.0 - cur.gripper);

      if (this.arm.jawLeft) this.arm.jawLeft.position.z = sep;
      if (this.arm.jawRight) this.arm.jawRight.position.z = -sep;
    }

    // Kinematic smoothing: interpolate dynamic cubes
    if (this.cubeTargets) {
      for (const [id, target] of Object.entries(this.cubeTargets)) {
        const mesh = this.cubes[id];
        if (mesh && target.pos) {
          mesh.position.x += (target.pos[0] - mesh.position.x) * 0.55;
          mesh.position.y += (target.pos[1] - mesh.position.y) * 0.55;
          mesh.position.z += (target.pos[2] - mesh.position.z) * 0.55;
          mesh.position.y = Math.max(0.7475, mesh.position.y);
          if (target.rot) {
            mesh.quaternion.slerp(new THREE.Quaternion(target.rot[0], target.rot[1], target.rot[2], target.rot[3]), 0.55);
          }
        }
      }
    }

    // Subtle pulsing of target pad rings
    if (this.targetPads && this.targetPads.length) {
      const op = 0.55 + Math.sin(Date.now() * 0.003) * 0.22;
      this.targetPads.forEach((pad) => {
        const ring = pad.children[1];
        if (ring && ring.material) {
          ring.material.opacity = op;
        }
      });
    }

    // Subtle pulsing of target ghost cube
    if (this.ghostTargetMesh && this.ghostTargetMesh.visible) {
      this.ghostTargetMesh.material.opacity = 0.14 + Math.sin(Date.now() * 0.006) * 0.07;
    }

    // Throttled 30 FPS real 3D WebGL Vision HUD rendering for OVERHEAD / WRIST / DEPTH
    this.renderPipHud();

    // Render 1, 2, or 3+ selected views across the main viewport
    this.renderMainScreens();
  }

  setMainScreens(screens, paneProvider = null) {
    this.mainScreens = (Array.isArray(screens) && screens.length > 0) ? [...screens] : ['studio'];
    this.paneProvider = paneProvider;
    if (this.controls) {
      this.controls.enabled = this.mainScreens.includes('studio') && this.activeCameraView !== 'wrist';
    }
  }

  _pushHideOverlays() {
    const saved = {
      frustum: this.sensorFrustumGroup ? this.sensorFrustumGroup.visible : false,
      plan: this.plannedPathLine ? this.plannedPathLine.visible : false,
      ghost: this.ghostTargetMesh ? this.ghostTargetMesh.visible : false,
      trail: this.tcpTrailLine ? this.tcpTrailLine.visible : false,
      tcp: this.arm.tcpMarker ? this.arm.tcpMarker.visible : false,
      boxes: [],
    };
    if (this.sensorFrustumGroup) this.sensorFrustumGroup.visible = false;
    if (this.plannedPathLine) this.plannedPathLine.visible = false;
    if (this.ghostTargetMesh) this.ghostTargetMesh.visible = false;
    if (this.tcpTrailLine) this.tcpTrailLine.visible = false;
    if (this.arm.tcpMarker) this.arm.tcpMarker.visible = false;
    Object.values(this.perceptionMeshes).forEach((m) => {
      saved.boxes.push([m, m.visible]);
      m.visible = false;
    });
    return saved;
  }

  _popOverlays(saved) {
    if (!saved) return;
    if (this.sensorFrustumGroup) this.sensorFrustumGroup.visible = saved.frustum;
    if (this.plannedPathLine) this.plannedPathLine.visible = saved.plan;
    if (this.ghostTargetMesh) this.ghostTargetMesh.visible = saved.ghost;
    if (this.tcpTrailLine) this.tcpTrailLine.visible = saved.trail;
    if (this.arm.tcpMarker) this.arm.tcpMarker.visible = saved.tcp;
    saved.boxes.forEach(([m, v]) => { m.visible = v; });
  }

  renderMainScreens() {
    if (!this.renderer || !this.scene || !this.camera) return;

    const isDefaultSingle = this.mainScreens.length === 1 && this.mainScreens[0] === 'studio';
    const cw = this.canvas.clientWidth || 1;
    const ch = this.canvas.clientHeight || 1;

    if (isDefaultSingle) {
      this.renderer.setScissorTest(false);
      if (Math.abs(this.camera.aspect - (cw / ch)) > 0.002) {
        this.camera.aspect = cw / ch;
        this.camera.updateProjectionMatrix();
      }
      this.renderer.setViewport(0, 0, cw, ch);
      this.renderer.render(this.scene, this.camera);
      return;
    }

    const panes = this.paneProvider ? this.paneProvider() : [];
    const canvasRect = this.canvas.getBoundingClientRect();
    if (!panes || panes.length === 0 || canvasRect.width <= 0 || canvasRect.height <= 0) {
      this.renderer.setScissorTest(false);
      this.renderer.setViewport(0, 0, cw, ch);
      this.renderer.render(this.scene, this.camera);
      return;
    }

    // Clear whole canvas first so split grid dividers are clean black
    this.renderer.setScissorTest(false);
    this.renderer.setViewport(0, 0, canvasRect.width, canvasRect.height);
    this.renderer.clear();
    this.renderer.setScissorTest(true);

    const prevOverride = this.scene.overrideMaterial;
    for (let i = 0; i < panes.length; i++) {
      const pane = panes[i];
      const screenType = pane.dataset ? pane.dataset.screen : null;
      if (!screenType || screenType === 'map') continue;

      const pr = pane.getBoundingClientRect();
      const w = Math.max(1, Math.round(pr.width));
      const h = Math.max(1, Math.round(pr.height));
      const x = Math.round(pr.left - canvasRect.left);
      const y = Math.round(canvasRect.height - (pr.bottom - canvasRect.top));

      this.renderer.setViewport(x, y, w, h);
      this.renderer.setScissor(x, y, w, h);

      if (screenType === 'studio') {
        if (Math.abs(this.camera.aspect - (w / h)) > 0.002) {
          this.camera.aspect = w / h;
          this.camera.updateProjectionMatrix();
        }
        this.scene.overrideMaterial = prevOverride;
        this.renderer.render(this.scene, this.camera);
      } else {
        const cam = (screenType === 'wrist' && this.wristSensorCam)
          ? this.wristSensorCam
          : this.overheadSensorCam;
        if (!cam) continue;
        cam.aspect = w / h;
        cam.updateProjectionMatrix();
        const saved = this._pushHideOverlays();
        this.scene.overrideMaterial = (screenType === 'depth' && this.depthMaterial)
          ? this.depthMaterial
          : prevOverride;
        this.renderer.render(this.scene, cam);
        this.scene.overrideMaterial = prevOverride;
        this._popOverlays(saved);
      }
    }

    this.scene.overrideMaterial = prevOverride;
    this.renderer.setScissorTest(false);
  }

  renderPipHud() {
    if (!this.renderer || !this.pipRenderTarget || !this.pipCtx || !this.pipCanvas) return;
    if (this.hudFeedMode === 'map' || this.pipCanvas.offsetParent === null) return;
    if ((this.pipFrameCounter++ % 2) !== 0) return;

    const activeCam = (this.hudFeedMode === 'wrist' && this.wristSensorCam)
      ? this.wristSensorCam
      : this.overheadSensorCam;
    if (!activeCam) return;

    // Ensure square aspect & no lingering scissor clipping on the offscreen PiP target
    this.renderer.setScissorTest(false);
    if (Math.abs(activeCam.aspect - 1.0) > 0.002) {
      activeCam.aspect = 1.0;
      activeCam.updateProjectionMatrix();
    }

    const saved = this._pushHideOverlays();
    const prevOverride = this.scene.overrideMaterial;
    if (this.hudFeedMode === 'depth' && this.depthMaterial) {
      this.scene.overrideMaterial = this.depthMaterial;
    }

    const prevTarget = this.renderer.getRenderTarget();
    this.renderer.setRenderTarget(this.pipRenderTarget);
    this.renderer.clear();
    this.renderer.render(this.scene, activeCam);
    this.renderer.readRenderTargetPixels(
      this.pipRenderTarget, 0, 0, this.pipSize, this.pipSize, this.pipReadBuffer
    );
    this.renderer.setRenderTarget(prevTarget);
    this.scene.overrideMaterial = prevOverride;

    this._popOverlays(saved);

    // Flip Y rows from WebGL bottom-up origin into top-down 2D canvas
    const w = this.pipSize;
    const h = this.pipSize;
    const rowBytes = w * 4;
    const dst = this.pipImageData.data;
    const src = this.pipReadBuffer;
    for (let y = 0; y < h; y++) {
      const srcOffset = (h - 1 - y) * rowBytes;
      const dstOffset = y * rowBytes;
      dst.set(src.subarray(srcOffset, srcOffset + rowBytes), dstOffset);
    }
    this.pipCtx.putImageData(this.pipImageData, 0, 0);
  }
}

window.SimCanvas3D = SimCanvas3D;
