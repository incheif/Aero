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
      wristRoll: null,
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
    this.showAxes = false;
    this.showGrid = true;
    this.axesHelper = null;
    this.tableAxesHelper = null;
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
    this.setupAxesHelper();

    // 6. Articulated Panda Robot Arm (RobotArm.tsx from new_project_1)
    this.buildRobotArm();

    // 7. TCP Trajectory Trail & Planned Preview
    this.setupTrailLine();
    this.setupPlannedPreview();

    // 8. 3D Sensor Visualizations & Live Sensor Cameras
    this.setupSensorVisualizations();
    this.setupSensorCameras();

    // 9. Initial Default Blocks (Guarantees blocks render instantly on scene load without network wait)
    this.initDefaultBlocks();

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

    // Initialize Theme (Graphite, Midnight, Light Engineering, or High Contrast)
    const activeTheme = (typeof document !== 'undefined' && document.documentElement.getAttribute('data-theme')) || 'graphite';
    this.setTheme(activeTheme);

    // Animation Loop
    this.animate();
  }

  // Dynamic Block Management
  updateBlocks(blocks) {
    if (!blocks || !Array.isArray(blocks)) return;
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
        rot: b.rotation || [0, 0, 0, 1],
      };
    });
  }

  // Pre-instantiate canonical blocks so tabletop is populated the moment DOM mounts
  initDefaultBlocks() {
    const DEFAULT_BLOCKS = [
      { id: 'block_magenta', name: 'Red Cube', color: '#FF1E27', position: [0.26, 0.7535, 0.12], rotation: [0, 0, 0, 1] },
      { id: 'block_cyan', name: 'Blue Cube', color: '#1E60FF', position: [0.24, 0.7535, -0.18], rotation: [0, 0, 0, 1] },
      { id: 'block_yellow', name: 'Yellow Cube', color: '#FFC300', position: [0.21, 0.7535, -0.05], rotation: [0, 0, 0, 1] },
      { id: 'block_emerald', name: 'Green Cube', color: '#00B341', position: [0.22, 0.7535, 0.24], rotation: [0, 0, 0, 1] },
      { id: 'block_orange', name: 'Orange Cube', color: '#FF5500', position: [0.34, 0.7535, -0.06], rotation: [0, 0, 0, 1] },
      { id: 'block_purple', name: 'Purple Cube', color: '#9333EA', position: [0.20, 0.7535, -0.27], rotation: [0, 0, 0, 1] },
    ];
    this.updateBlocks(DEFAULT_BLOCKS);
  }

  setupAxesHelper() {
    // 1. Arm Base Mount Coordinate Axes (RGB = XYZ: Red=X, Green=Y, Blue=Z)
    this.axesHelper = new THREE.AxesHelper(0.22);
    this.axesHelper.position.set(-0.28, 0.725, 0.0);
    this.axesHelper.visible = !!this.showAxes;
    this.scene.add(this.axesHelper);

    // 2. Table Center Origin Coordinate Axes
    this.tableAxesHelper = new THREE.AxesHelper(0.14);
    this.tableAxesHelper.position.set(0.0, 0.721, 0.0);
    this.tableAxesHelper.visible = !!this.showAxes;
    this.scene.add(this.tableAxesHelper);
  }

  toggleAxes(show) {
    this.showAxes = (show !== undefined ? !!show : !this.showAxes);
    if (this.axesHelper) this.axesHelper.visible = this.showAxes;
    if (this.tableAxesHelper) this.tableAxesHelper.visible = this.showAxes;
    return this.showAxes;
  }

  toggleTrails(show) {
    this.showTrails = (show !== undefined ? !!show : !this.showTrails);
    if (this.tcpTrailLine) this.tcpTrailLine.visible = this.showTrails;
    return this.showTrails;
  }

  toggleFrustums(show) {
    this.showFrustums = (show !== undefined ? !!show : !this.showFrustums);
    if (this.sensorFrustumGroup) this.sensorFrustumGroup.visible = this.showFrustums;
    if (this.wristSensorBeam) this.wristSensorBeam.visible = this.showFrustums;
    return this.showFrustums;
  }

  togglePerceptionBoxes(show) {
    this.showPerceptionBoxes = (show !== undefined ? !!show : !this.showPerceptionBoxes);
    Object.values(this.perceptionMeshes).forEach((m) => {
      m.visible = this.showPerceptionBoxes;
    });
    return this.showPerceptionBoxes;
  }

  toggleGrid(show) {
    this.showGrid = (show !== undefined ? !!show : !this.showGrid);
    if (this.gridHelper) this.gridHelper.visible = this.showGrid;
    return this.showGrid;
  }

  setTheme(themeName) {
    this.currentTheme = themeName || 'graphite';
    let bgColor = 0x0e1117;
    let floorColor = 0x090c12;
    let tableColor = 0x161b24;
    let trimColor = 0x222b38;
    let grid1 = 0x384454;
    let grid2 = 0x161b22;
    let ambIntensity = 0.45;
    let hemiSky = 0xcfd8dc;
    let hemiGround = 0x161b22;
    let hemiIntensity = 0.75;

    if (themeName === 'graphite' || themeName === 'monochrome' || themeName === 'cyberpunk') {
      bgColor = 0x0e1117;
      floorColor = 0x090c12;
      tableColor = 0x161b24;
      trimColor = 0x222b38;
      grid1 = 0x384454;
      grid2 = 0x161b22;
      ambIntensity = 0.45;
      hemiSky = 0xcfd8dc;
      hemiGround = 0x161b22;
      hemiIntensity = 0.75;
    } else if (themeName === 'midnight' || themeName === 'high-contrast') {
      bgColor = 0x080c14;
      floorColor = 0x05080e;
      tableColor = 0x0f172a;
      trimColor = 0x1e293b;
      grid1 = 0x1e3a5f;
      grid2 = 0x0f172a;
      ambIntensity = 0.45;
      hemiSky = 0x93c5fd;
      hemiGround = 0x0f172a;
      hemiIntensity = 0.75;
    } else if (themeName === 'light') {
      bgColor = 0xf1f5f9;
      floorColor = 0xe2e8f0;
      tableColor = 0xffffff;
      trimColor = 0xcbd5e1;
      grid1 = 0x94a3b8;
      grid2 = 0xe2e8f0;
      ambIntensity = 0.65;
      hemiSky = 0xffffff;
      hemiGround = 0xcbd5e1;
      hemiIntensity = 0.85;
    } else if (themeName === 'solar-light' || themeName === 'solar-amber') {
      bgColor = 0xfaf8f5;
      floorColor = 0xf5f0e8;
      tableColor = 0xffffff;
      trimColor = 0xd97706;
      grid1 = 0xd97706;
      grid2 = 0xe5dccd;
      ambIntensity = 0.65;
      hemiSky = 0xfffbeb;
      hemiGround = 0xebd9c6;
      hemiIntensity = 0.85;
    }

    if (this.scene) {
      if (this.scene.background) this.scene.background.setHex(bgColor);
      if (this.scene.fog) {
        this.scene.fog.color.setHex(bgColor);
        this.scene.fog.near = 7;
        this.scene.fog.far = 24;
      }
    }

    if (this.floorMat) this.floorMat.color.setHex(floorColor);
    if (this.tableMat) this.tableMat.color.setHex(tableColor);
    if (this.trimMat) this.trimMat.color.setHex(trimColor);

    if (this.gridHelper && this.scene) {
      this.scene.remove(this.gridHelper);
      if (this.gridHelper.geometry) this.gridHelper.geometry.dispose();
      if (this.gridHelper.material) this.gridHelper.material.dispose();
      this.gridHelper = new THREE.GridHelper(16, 32, grid1, grid2);
      this.gridHelper.position.y = 0.001;
      this.gridHelper.matrixAutoUpdate = false;
      this.gridHelper.updateMatrix();
      this.gridHelper.visible = (this.showGrid !== false);
      this.scene.add(this.gridHelper);
    }

    if (this.ambLight) this.ambLight.intensity = ambIntensity;
    if (this.hemiLight) {
      this.hemiLight.color.setHex(hemiSky);
      this.hemiLight.groundColor.setHex(hemiGround);
      this.hemiLight.intensity = hemiIntensity;
    }
  }

  setupLighting() {
    const hemiLight = new THREE.HemisphereLight(0xd4d4d8, 0x18181b, 0.7);
    this.scene.add(hemiLight);
    this.hemiLight = hemiLight;

    const ambLight = new THREE.AmbientLight(0xffffff, 0.4);
    this.scene.add(ambLight);
    this.ambLight = ambLight;

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
    this.gridHelper = gridHelper;

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
    this.floorMat = floorMat;

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
    this.tableMat = tableMat;

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
    this.trimMat = trimMat;

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

  _createFeetechSTS3215(hasHorn = true, hornColor = 0x38bdf8) {
    const group = new THREE.Group();

    // 1. Dark matte charcoal composite body casing (45.2mm x 24.7mm x 35.0mm)
    const bodyMat = new THREE.MeshStandardMaterial({
      color: 0x14161a,
      roughness: 0.72,
      metalness: 0.15,
    });
    const body = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.025, 0.035), bodyMat);
    body.castShadow = true;
    body.receiveShadow = true;
    group.add(body);

    // 2. Anodized aluminum heatsink center band with cooling ridges
    const heatSinkMat = new THREE.MeshStandardMaterial({
      color: 0x383e4a,
      roughness: 0.28,
      metalness: 0.85,
    });
    const heatSink = new THREE.Mesh(new THREE.BoxGeometry(0.0456, 0.0256, 0.012), heatSinkMat);
    group.add(heatSink);

    // Subtle cooling fin grooves on heatsink
    const finMat = new THREE.MeshStandardMaterial({ color: 0x111317, roughness: 0.5, metalness: 0.8 });
    for (let f = -1; f <= 1; f++) {
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.046, 0.0258, 0.0018), finMat);
      fin.position.y = f * 0.0035;
      group.add(fin);
    }

    // 3. Output Spline Gear Collar
    const collarMat = new THREE.MeshStandardMaterial({ color: 0x1f242d, roughness: 0.4, metalness: 0.8 });
    const collar = new THREE.Mesh(new THREE.CylinderGeometry(0.010, 0.010, 0.006, 16), collarMat);
    collar.position.set(0.012, 0.0135, 0.0);
    group.add(collar);

    // 4. Metallic Output Horn Flange (with radial fastener sockets)
    if (hasHorn) {
      const hornMat = new THREE.MeshStandardMaterial({
        color: hornColor,
        roughness: 0.22,
        metalness: 0.88,
      });
      const horn = new THREE.Mesh(new THREE.CylinderGeometry(0.0135, 0.0135, 0.0035, 20), hornMat);
      horn.position.set(0.012, 0.017, 0.0);
      group.add(horn);

      // Center lock screw
      const screwMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.2, metalness: 0.95 });
      const centerScrew = new THREE.Mesh(new THREE.CylinderGeometry(0.003, 0.003, 0.004, 12), screwMat);
      centerScrew.position.set(0.012, 0.018, 0.0);
      group.add(centerScrew);
    }

    // 5. Dual 3-Pin JST Daisy-Chain Wire Connector on motor rear
    const connMat = new THREE.MeshStandardMaterial({ color: 0x27272a, roughness: 0.5 });
    const socket = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.006, 0.016), connMat);
    socket.position.set(-0.021, -0.008, 0.0);
    group.add(socket);

    // Mini wiring harness leads (red/black/white)
    const wireMatR = new THREE.MeshBasicMaterial({ color: 0xef4444 });
    const wireMatB = new THREE.MeshBasicMaterial({ color: 0x09090b });
    const wireMatW = new THREE.MeshBasicMaterial({ color: 0xf8fafc });

    const wireR = new THREE.Mesh(new THREE.CylinderGeometry(0.0009, 0.0009, 0.018, 6), wireMatR);
    wireR.position.set(-0.024, -0.008, 0.004);
    wireR.rotation.z = Math.PI / 2;
    group.add(wireR);

    const wireB = new THREE.Mesh(new THREE.CylinderGeometry(0.0009, 0.0009, 0.018, 6), wireMatB);
    wireB.position.set(-0.024, -0.008, 0.0);
    wireB.rotation.z = Math.PI / 2;
    group.add(wireB);

    const wireW = new THREE.Mesh(new THREE.CylinderGeometry(0.0009, 0.0009, 0.018, 6), wireMatW);
    wireW.position.set(-0.024, -0.008, -0.004);
    wireW.rotation.z = Math.PI / 2;
    group.add(wireW);

    return group;
  }

  buildRobotArm() {
    // Mount position: [-0.28, 0.72, 0]
    const root = new THREE.Group();
    root.position.set(-0.28, 0.72, 0.0);
    this.arm.base = root;

    // Materials Palette matching authentic SO-ARM100 / SO-101 3D printed PETG & Hardware
    const petgDarkMat = new THREE.MeshStandardMaterial({ color: 0x16181d, roughness: 0.7, metalness: 0.12 });
    const petgAccentMat = new THREE.MeshStandardMaterial({ color: 0x0284c7, roughness: 0.48, metalness: 0.22 });
    const metalMat = new THREE.MeshStandardMaterial({ color: 0x94a3b8, metalness: 0.88, roughness: 0.22 });
    const fastenerMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.95, roughness: 0.15 });
    const rubberMat = new THREE.MeshStandardMaterial({ color: 0x10b981, roughness: 0.92 }); // High-contrast emerald grip pads
    const hornColor = 0x38bdf8;

    // 1. Desk Mount Plate (octagonal with 4 stainless M4 corner bolts)
    const basePlate = new THREE.Mesh(new THREE.CylinderGeometry(0.092, 0.096, 0.016, 8), petgDarkMat);
    basePlate.position.y = 0.008;
    basePlate.castShadow = true;
    basePlate.receiveShadow = true;
    root.add(basePlate);

    // 4 Corner Mounting Screws on tabletop flange
    for (let i = 0; i < 4; i++) {
      const angle = (i * Math.PI) / 2 + Math.PI / 4;
      const bx = Math.cos(angle) * 0.076;
      const bz = Math.sin(angle) * 0.076;
      const bolt = new THREE.Mesh(new THREE.CylinderGeometry(0.0035, 0.0035, 0.004, 8), fastenerMat);
      bolt.position.set(bx, 0.017, bz);
      root.add(bolt);
    }

    // Side-mounted Waveshare bus adapter PCB plate
    const pcbMount = new THREE.Mesh(new THREE.BoxGeometry(0.044, 0.005, 0.068), new THREE.MeshStandardMaterial({ color: 0x064e3b, roughness: 0.4 }));
    pcbMount.position.set(-0.068, 0.012, 0.0);
    root.add(pcbMount);

    const pcbLed = new THREE.Mesh(new THREE.SphereGeometry(0.0025, 8, 8), new THREE.MeshBasicMaterial({ color: 0x10b981 }));
    pcbLed.position.set(-0.078, 0.016, 0.022);
    root.add(pcbLed);

    // Upright Pedestal Tower with weight-saving ribbing
    const pedestalTower = new THREE.Mesh(new THREE.CylinderGeometry(0.046, 0.068, 0.076, 24), petgDarkMat);
    pedestalTower.position.y = 0.054;
    pedestalTower.castShadow = true;
    root.add(pedestalTower);

    // Embedded Feetech STS3215 #1 (Base Yaw Motor)
    const servo1 = this._createFeetechSTS3215(true, hornColor);
    servo1.position.set(0.0, 0.088, 0.0);
    servo1.rotation.y = Math.PI / 2;
    servo1.rotation.z = Math.PI;
    root.add(servo1);

    // Base Turntable Bearing Collar (at Pedestal Height 0.11m)
    const flange = new THREE.Mesh(new THREE.CylinderGeometry(0.048, 0.048, 0.008, 32), metalMat);
    flange.position.y = 0.106;
    root.add(flange);

    // 2. Base Yaw Turntable Group (at Pedestal Height 0.11m)
    const yawGroup = new THREE.Group();
    yawGroup.position.set(0.0, 0.11, 0.0);
    root.add(yawGroup);
    this.arm.yaw = yawGroup;

    // Dual-clevis Shoulder Bracket (SO-ARM100 U-fork with side lightening cutouts)
    const shoulderForkBase = new THREE.Mesh(new THREE.CylinderGeometry(0.046, 0.046, 0.014, 24), petgDarkMat);
    shoulderForkBase.position.y = 0.007;
    yawGroup.add(shoulderForkBase);

    // Dual upright clevis ears
    const leftEar = new THREE.Mesh(new THREE.BoxGeometry(0.016, 0.064, 0.044), petgDarkMat);
    leftEar.position.set(0.0, 0.038, 0.036);
    yawGroup.add(leftEar);

    const rightEar = new THREE.Mesh(new THREE.BoxGeometry(0.016, 0.064, 0.044), petgDarkMat);
    rightEar.position.set(0.0, 0.038, -0.036);
    yawGroup.add(rightEar);

    // Feetech STS3215 #2 (Shoulder Lift Motor) housed in left clevis
    const servo2 = this._createFeetechSTS3215(true, hornColor);
    servo2.position.set(0.0, 0.038, 0.036);
    servo2.rotation.x = Math.PI / 2;
    yawGroup.add(servo2);

    // Passive Pivot Bearing Cap on right clevis
    const bearingCap = new THREE.Mesh(new THREE.CylinderGeometry(0.016, 0.016, 0.008, 16), metalMat);
    bearingCap.rotation.x = Math.PI / 2;
    bearingCap.position.set(0.0, 0.038, -0.045);
    yawGroup.add(bearingCap);

    // 3. Shoulder Group (Pitches in Z at Y=0.038m above yaw turntable)
    const shoulderGroup = new THREE.Group();
    shoulderGroup.position.set(0.0, 0.038, 0.0);
    yawGroup.add(shoulderGroup);
    this.arm.shoulder = shoulderGroup;

    // Upper Arm Link (length 0.38m along local X)
    // Realistic SO-ARM100 Skeletal Twin-Rail Truss Arm
    const upperLinkMat = petgDarkMat;
    const topRail = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.014, 0.042), upperLinkMat);
    topRail.position.set(0.19, 0.022, 0.0);
    topRail.castShadow = true;
    shoulderGroup.add(topRail);

    const bottomRail = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.014, 0.042), upperLinkMat);
    bottomRail.position.set(0.19, -0.022, 0.0);
    bottomRail.castShadow = true;
    shoulderGroup.add(bottomRail);

    // Diagonal truss reinforcement cross-ribs
    for (let r = 0; r < 5; r++) {
      const rx = 0.06 + r * 0.065;
      const rib = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.032, 0.040), petgAccentMat);
      rib.position.set(rx, 0.0, 0.0);
      shoulderGroup.add(rib);
    }

    // Flexible cable harness clip routing along upper arm
    const cableHarness = new THREE.Mesh(new THREE.CylinderGeometry(0.0035, 0.0035, 0.32, 10), new THREE.MeshBasicMaterial({ color: 0x1e293b }));
    cableHarness.rotation.z = Math.PI / 2;
    cableHarness.position.set(0.19, 0.030, 0.0);
    shoulderGroup.add(cableHarness);

    // Feetech STS3215 #3 (Elbow Flex Motor) at X=0.38m of upper arm
    const servo3 = this._createFeetechSTS3215(true, hornColor);
    servo3.position.set(0.38, 0.0, 0.034);
    servo3.rotation.x = Math.PI / 2;
    shoulderGroup.add(servo3);

    // 4. Elbow Group (at X=0.38m of upper arm)
    const elbowGroup = new THREE.Group();
    elbowGroup.position.set(0.38, 0.0, 0.0);
    shoulderGroup.add(elbowGroup);
    this.arm.elbow = elbowGroup;

    // Forearm Link (length 0.32m along local X)
    // Sculpted tapered forearm beam
    const forearmLink = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.038, 0.036), petgDarkMat);
    forearmLink.position.set(0.16, 0.0, 0.0);
    forearmLink.castShadow = true;
    elbowGroup.add(forearmLink);

    // Contrast accent spine on forearm
    const forearmSpine = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.006, 0.038), petgAccentMat);
    forearmSpine.position.set(0.16, 0.021, 0.0);
    elbowGroup.add(forearmSpine);

    // Feetech STS3215 #4 (Wrist Pitch Motor) at X=0.32m of forearm
    const servo4 = this._createFeetechSTS3215(true, hornColor);
    servo4.position.set(0.32, 0.0, 0.028);
    servo4.rotation.x = Math.PI / 2;
    elbowGroup.add(servo4);

    // 5. Wrist Group (at X=0.32m of forearm, pitches in Z)
    const wristGroup = new THREE.Group();
    wristGroup.position.set(0.32, 0.0, 0.0);
    elbowGroup.add(wristGroup);
    this.arm.wrist = wristGroup;

    // Wrist pitch clevis
    const wristClevis = new THREE.Mesh(new THREE.BoxGeometry(0.036, 0.036, 0.048), petgDarkMat);
    wristClevis.position.set(0.016, 0.0, 0.0);
    wristGroup.add(wristClevis);

    // In-line cylindrical housing for Feetech STS3215 #5 (Wrist Roll Motor)
    const wristRollHousing = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 0.075, 24), petgDarkMat);
    wristRollHousing.rotation.z = Math.PI / 2;
    wristRollHousing.position.set(0.065, 0.0, 0.0);
    wristRollHousing.castShadow = true;
    wristGroup.add(wristRollHousing);

    // 6. Wrist Roll Group (Rotates around local X: approach axis)
    const wristRollGroup = new THREE.Group();
    wristRollGroup.position.set(0.0, 0.0, 0.0);
    wristGroup.add(wristRollGroup);
    this.arm.wristRoll = wristRollGroup;

    // Feetech STS3215 #5 (Wrist Roll) Output Horn rotating at X=0.105m
    const rollHorn = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.018, 0.006, 24), new THREE.MeshStandardMaterial({ color: hornColor, metalness: 0.85, roughness: 0.25 }));
    rollHorn.rotation.z = Math.PI / 2;
    rollHorn.position.set(0.105, 0.0, 0.0);
    wristRollGroup.add(rollHorn);

    // Palm / Gripper Base Plate (at X = 0.13m)
    const palm = new THREE.Mesh(new THREE.BoxGeometry(0.026, 0.048, 0.074), petgDarkMat);
    palm.position.set(0.13, 0.0, 0.0);
    palm.castShadow = true;
    wristRollGroup.add(palm);
    this.arm.palm = palm;

    // Feetech STS3215 #6 (Gripper Drive Motor) mounted on palm
    const servo6 = this._createFeetechSTS3215(true, hornColor);
    servo6.position.set(0.128, -0.018, 0.0);
    servo6.scale.set(0.75, 0.75, 0.75);
    wristRollGroup.add(servo6);

    // Eye-in-Hand Wrist Camera Pod + Status LED Ring mounted on top of palm
    const wristPod = new THREE.Mesh(
      new THREE.BoxGeometry(0.032, 0.018, 0.034),
      new THREE.MeshStandardMaterial({ color: 0x18181b, metalness: 0.7, roughness: 0.3 })
    );
    wristPod.position.set(0.122, 0.032, 0.0);
    wristRollGroup.add(wristPod);

    const wristLens = new THREE.Mesh(
      new THREE.CylinderGeometry(0.0065, 0.0065, 0.008, 14),
      new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9, roughness: 0.1 })
    );
    wristLens.rotation.z = Math.PI / 2;
    wristLens.position.set(0.139, 0.032, 0.0);
    wristRollGroup.add(wristLens);

    const statusLedMat = new THREE.MeshStandardMaterial({
      color: 0x38bdf8,
      emissive: 0x38bdf8,
      emissiveIntensity: 1.4,
      roughness: 0.2,
    });
    const statusLed = new THREE.Mesh(new THREE.SphereGeometry(0.0045, 12, 12), statusLedMat);
    statusLed.position.set(0.124, 0.042, 0.012);
    wristRollGroup.add(statusLed);
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
    wristRollGroup.add(wristCam);
    this.wristSensorCam = wristCam;

    // 7. Parallel Gripper Sliding Jaws (JAW_LENGTH = 0.095m, centered at X = 0.18m matching TCP)
    const jawGeo = new THREE.BoxGeometry(0.095, 0.036, 0.012);

    // Left Jaw
    const jawLeft = new THREE.Mesh(jawGeo, petgDarkMat);
    jawLeft.position.set(0.18, 0.0, 0.045);
    jawLeft.castShadow = true;
    wristRollGroup.add(jawLeft);
    this.arm.jawLeft = jawLeft;

    // Left Jaw Rubber Gripper Pad
    const padLeft = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.028, 0.004), rubberMat);
    padLeft.position.set(0.0, 0.0, -0.008);
    jawLeft.add(padLeft);

    // Right Jaw
    const jawRight = new THREE.Mesh(jawGeo, petgDarkMat);
    jawRight.position.set(0.18, 0.0, -0.045);
    jawRight.castShadow = true;
    wristRollGroup.add(jawRight);
    this.arm.jawRight = jawRight;

    // Right Jaw Rubber Gripper Pad
    const padRight = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.028, 0.004), rubberMat);
    padRight.position.set(0.0, 0.0, 0.008);
    jawRight.add(padRight);

    // 8. TCP Glowing Indicator (mid-finger pinch point: at X = 0.18m, exactly at pad center)
    const tcpGroup = new THREE.Group();
    tcpGroup.position.set(0.18, 0.0, 0.0);

    const tcpSphere = new THREE.Mesh(
      new THREE.SphereGeometry(0.006, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    tcpGroup.add(tcpSphere);
    wristRollGroup.add(tcpGroup);
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
    // Disabled: green ghost destination cube and planned trajectory lines removed per user request
    this.plannedPathLine = null;
    this.ghostTargetMesh = null;
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
          wristRoll: joints.wristRoll || 0,
          gripper: joints.gripper || 0,
        };
      }
      this.targetJoints = {
        baseYaw: joints.baseYaw || 0,
        shoulderPitch: joints.shoulderPitch || 0,
        elbowPitch: joints.elbowPitch || 0,
        wristPitch: joints.wristPitch || 0,
        wristRoll: joints.wristRoll || 0,
        gripper: joints.gripper || 0,
      };
    }

    // 2. Update Dynamic Cubes with vivid, high-contrast recognizable colors
    this.updateBlocks(data.blocks);

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

    // 4b. Planned Path Preview & Target Ghost Cube (disabled per user request)
    if (this.plannedPathLine) {
      this.plannedPathLine.geometry.setDrawRange(0, 0);
      this.plannedPathLine.visible = false;
    }
    if (this.ghostTargetMesh) {
      this.ghostTargetMesh.visible = false;
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
      cur.wristRoll = (cur.wristRoll || 0) + ((j.wristRoll || 0) - (cur.wristRoll || 0)) * alpha;
      cur.gripper += (j.gripper - cur.gripper) * alpha;

      this.arm.yaw.rotation.y = cur.baseYaw;
      this.arm.shoulder.rotation.z = cur.shoulderPitch;
      this.arm.elbow.rotation.z = cur.elbowPitch;
      this.arm.wrist.rotation.z = cur.wristPitch;
      if (this.arm.wristRoll) this.arm.wristRoll.rotation.x = cur.wristRoll;

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
