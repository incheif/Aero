/**
 * AERO | Scroll-Driven Frame-by-Frame Robot Arm Storytelling Engine
 * Controls 3D Three.js Panda-style cobot assembling a 3-layer tower in real time,
 * completely bound to scroll progress with synchronous inverse kinematics and zero teleportation.
 * Sequence: Pick Blue -> Place on Red -> Pick Green -> Place on Blue (on Red).
 */

(function () {
  'use strict';

  function initScrollRobotStory() {
    const storyContainer = document.getElementById('scroll-story');
    const canvas = document.getElementById('scrolly-robot-canvas');
    if (!storyContainer || !canvas || typeof THREE === 'undefined') {
      return;
    }

    // Story Slide Elements & Interactive Instruction Terminal
    const storySticky = document.getElementById('scroll-story-sticky');
    const suspenseOverlay = document.getElementById('suspense-overlay');
    const slideSuspense = document.getElementById('slide-suspense');
    const suspenseLine1 = document.getElementById('suspense-line-1');
    const suspenseIndicator = document.getElementById('suspense-indicator');
    const slideAero = document.getElementById('slide-aero');
    const giantAeroTitle = document.getElementById('giant-aero-title');
    const slideTagline = document.getElementById('slide-tagline');
    const slidePrompt = document.getElementById('slide-prompt');
    const instructionBox = document.getElementById('instruction-box');
    const instructionTypedText = document.getElementById('instruction-typed-text');
    const instructionCursor = document.getElementById('instruction-cursor');
    const instructionStatusPill = document.getElementById('instruction-status-pill');
    const instructionPillText = document.getElementById('instruction-pill-text');
    const slideRobotHud = document.getElementById('slide-robot-hud');
    const hudFrameCounter = document.getElementById('hud-frame-counter');
    const slideDesc = document.getElementById('slide-desc');
    const slideStats = document.getElementById('slide-stats');
    const slideCtas = document.getElementById('slide-ctas');
    const storyCtaHeroBlock = document.getElementById('story-cta-hero-block');
    const storyCommandsWrap = document.getElementById('commands');

    // 1. Setup Three.js Scene
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x09090b);
    scene.fog = new THREE.Fog(0x09090b, 7, 20);

    const width = window.innerWidth;
    const height = window.innerHeight;
    const camera = new THREE.PerspectiveCamera(36, width / height, 0.05, 30);
    camera.position.set(2.25, 1.55, 1.95);
    camera.lookAt(0.12, 0.78, 0.0);

    const renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    });
    renderer.setSize(width, height, false);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    // 2. Lighting Setup
    const hemiLight = new THREE.HemisphereLight(0xe4e4e7, 0x18181b, 0.85);
    scene.add(hemiLight);

    const ambLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambLight);

    const keySpot = new THREE.SpotLight(0xffffff, 2.4);
    keySpot.position.set(1.9, 3.5, 1.8);
    keySpot.angle = 0.65;
    keySpot.penumbra = 0.55;
    keySpot.castShadow = true;
    keySpot.shadow.mapSize.width = 1024;
    keySpot.shadow.mapSize.height = 1024;
    keySpot.shadow.bias = -0.0006;
    scene.add(keySpot);

    const fillSpot = new THREE.SpotLight(0x93c5fd, 1.1);
    fillSpot.position.set(-2.0, 2.8, 1.2);
    fillSpot.angle = 0.75;
    fillSpot.penumbra = 0.8;
    scene.add(fillSpot);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.6);
    rimLight.position.set(-1.6, 2.2, -2.0);
    scene.add(rimLight);

    // 3. Environment: Floor & Table
    const gridHelper = new THREE.GridHelper(16, 32, 0x3f3f46, 0x18181b);
    gridHelper.position.y = 0.001;
    scene.add(gridHelper);

    const floorGeo = new THREE.PlaneGeometry(16, 16);
    const floorMat = new THREE.MeshStandardMaterial({ color: 0x09090b, roughness: 0.95, metalness: 0.05 });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Workcell Table (1.4m x 0.08m x 0.9m at Y=0.68m, tabletop at Y=0.72m)
    const tableTop = new THREE.Mesh(
      new THREE.BoxGeometry(1.4, 0.08, 0.9),
      new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.35, metalness: 0.35 })
    );
    tableTop.position.set(0.0, 0.68, 0.0);
    tableTop.receiveShadow = true;
    tableTop.castShadow = true;
    scene.add(tableTop);

    // Table Bevel Trim
    const trim = new THREE.Mesh(
      new THREE.BoxGeometry(1.42, 0.02, 0.92),
      new THREE.MeshStandardMaterial({ color: 0x27272a, roughness: 0.25, metalness: 0.7 })
    );
    trim.position.set(0.0, 0.71, 0.0);
    scene.add(trim);

    // Table Legs
    const legGeo = new THREE.CylinderGeometry(0.032, 0.032, 0.64, 16);
    const legMat = new THREE.MeshStandardMaterial({ color: 0x52525b, metalness: 0.85, roughness: 0.3 });
    [
      [0.64, 0.32, 0.39],
      [-0.64, 0.32, 0.39],
      [0.64, 0.32, -0.39],
      [-0.64, 0.32, -0.39],
    ].forEach(([lx, ly, lz]) => {
      const leg = new THREE.Mesh(legGeo, legMat);
      leg.position.set(lx, ly, lz);
      leg.castShadow = true;
      scene.add(leg);
    });

    // 4. Build Articulated Panda Robot Arm
    const PALETTE = {
      WHITE: 0xf4f4f5,
      BLACK: 0x18181b,
      METAL: 0xa1a1aa,
      RUBBER: 0x27272a,
    };

    const baseMat = new THREE.MeshStandardMaterial({ color: PALETTE.BLACK, roughness: 0.6 });
    const metalMat = new THREE.MeshStandardMaterial({ color: PALETTE.METAL, metalness: 0.85, roughness: 0.25 });
    const whiteMat = new THREE.MeshStandardMaterial({ color: PALETTE.WHITE, roughness: 0.35 });
    const rubberMat = new THREE.MeshStandardMaterial({ color: PALETTE.RUBBER, roughness: 0.9 });

    const root = new THREE.Group();
    root.position.set(-0.28, 0.72, 0.0);
    scene.add(root);

    // Pedestal Base
    const pedestalBase = new THREE.Mesh(new THREE.CylinderGeometry(0.096, 0.096, 0.036, 32), baseMat);
    pedestalBase.position.y = 0.018;
    pedestalBase.castShadow = true;
    pedestalBase.receiveShadow = true;
    root.add(pedestalBase);

    const pedestalTaper = new THREE.Mesh(new THREE.CylinderGeometry(0.044, 0.068, 0.066, 32), baseMat);
    pedestalTaper.position.y = 0.069;
    pedestalTaper.castShadow = true;
    root.add(pedestalTaper);

    const flange = new THREE.Mesh(new THREE.CylinderGeometry(0.046, 0.046, 0.008, 32), metalMat);
    flange.position.y = 0.106;
    root.add(flange);

    // Turntable Group (Yaw)
    const yawGroup = new THREE.Group();
    yawGroup.position.set(0.0, 0.11, 0.0);
    root.add(yawGroup);

    const shoulderCan = new THREE.Mesh(new THREE.CylinderGeometry(0.038, 0.038, 0.072, 28), metalMat);
    shoulderCan.rotation.x = Math.PI / 2;
    yawGroup.add(shoulderCan);

    // Shoulder Pitch Group
    const shoulderGroup = new THREE.Group();
    yawGroup.add(shoulderGroup);

    const upperLink = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.052, 0.052), whiteMat);
    upperLink.position.set(0.19, 0.0, 0.0);
    upperLink.castShadow = true;
    shoulderGroup.add(upperLink);

    // Elbow Pitch Group
    const elbowGroup = new THREE.Group();
    elbowGroup.position.set(0.38, 0.0, 0.0);
    shoulderGroup.add(elbowGroup);

    const elbowCan = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, 0.066, 28), metalMat);
    elbowCan.rotation.x = Math.PI / 2;
    elbowGroup.add(elbowCan);

    const forearmLink = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.044, 0.044), whiteMat);
    forearmLink.position.set(0.16, 0.0, 0.0);
    forearmLink.castShadow = true;
    elbowGroup.add(forearmLink);

    // Wrist Pitch Group
    const wristGroup = new THREE.Group();
    wristGroup.position.set(0.32, 0.0, 0.0);
    elbowGroup.add(wristGroup);

    const wristCan = new THREE.Mesh(new THREE.CylinderGeometry(0.028, 0.028, 0.054, 24), metalMat);
    wristCan.rotation.x = Math.PI / 2;
    wristGroup.add(wristCan);

    const palm = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.042, 0.065), baseMat);
    palm.position.set(0.035, 0.0, 0.0);
    palm.castShadow = true;
    wristGroup.add(palm);

    // Parallel Gripper Sliding Jaws (centered at local x = 0.085)
    const jawGeo = new THREE.BoxGeometry(0.06, 0.024, 0.012);
    const jawLeft = new THREE.Mesh(jawGeo, whiteMat);
    jawLeft.position.set(0.085, 0.0, 0.05);
    jawLeft.castShadow = true;
    wristGroup.add(jawLeft);

    const jawRight = new THREE.Mesh(jawGeo, whiteMat);
    jawRight.position.set(0.085, 0.0, -0.05);
    jawRight.castShadow = true;
    wristGroup.add(jawRight);

    // Inner rubber pads for the jaws
    const padGeo = new THREE.BoxGeometry(0.05, 0.02, 0.004);
    const padLeft = new THREE.Mesh(padGeo, rubberMat);
    padLeft.position.set(0.0, 0.0, -0.006);
    jawLeft.add(padLeft);

    const padRight = new THREE.Mesh(padGeo, rubberMat);
    padRight.position.set(0.0, 0.0, 0.006);
    jawRight.add(padRight);

    // TCP Marker for World Positioning at jaw center (L3 = 0.085m)
    const tcpMarker = new THREE.Object3D();
    tcpMarker.position.set(0.085, 0.0, 0.0);
    wristGroup.add(tcpMarker);

    // 5. Cubes Setup (Base Red, Mid Blue, Apex Green)
    function createCube(colorHex) {
      const geo = new THREE.BoxGeometry(0.055, 0.055, 0.055);
      const mat = new THREE.MeshStandardMaterial({
        color: colorHex,
        roughness: 0.22,
        metalness: 0.05,
        emissive: colorHex,
        emissiveIntensity: 0.35,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;

      const edges = new THREE.EdgesGeometry(geo);
      const line = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.65, linewidth: 1.5 })
      );
      mesh.add(line);
      scene.add(mesh);
      return mesh;
    }

    const cubeRed = createCube(0xef4444);   // Base Red
    const cubeBlue = createCube(0x38bdf8);  // Mid Blue
    const cubeGreen = createCube(0x10b981); // Apex Green
    const cubeYellow = createCube(0xfacc15); // Tabletop Yellow (4th workcell cube)
    const cubeOrange = createCube(0xf97316); // Tabletop Orange (5th workcell cube)

    // Physical World Constants
    const TOWER_X = 0.18;
    const TOWER_Z = 0.0;
    const TABLE_Y = 0.72;
    const CUBE_HALF = 0.0275;
    const CUBE_SIZE = 0.055;

    // Foundation Base Red Cube Position
    const redBasePos = [TOWER_X, TABLE_Y + CUBE_HALF, TOWER_Z];
    cubeRed.position.set(...redBasePos);

    // Start positions on table
    // Blue is on the left (Z = -0.22), Green is on the right (Z = +0.22)
    const blueStart = [0.12, TABLE_Y + CUBE_HALF, -0.22];
    const greenStart = [0.12, TABLE_Y + CUBE_HALF, 0.22];

    cubeBlue.position.set(...blueStart);
    cubeGreen.position.set(...greenStart);

    // 5-block workcell tabletop positions for Yellow & Orange cubes
    const yellowStart = [0.30, TABLE_Y + CUBE_HALF, -0.11];
    const orangeStart = [0.28, TABLE_Y + CUBE_HALF, 0.13];
    cubeYellow.position.set(...yellowStart);
    cubeYellow.rotation.y = 0.22;
    cubeOrange.position.set(...orangeStart);
    cubeOrange.rotation.y = -0.28;

    // Target Stacked Positions in 3-layer tower: Red -> Blue -> Green
    const blueStacked = [TOWER_X, TABLE_Y + CUBE_HALF + CUBE_SIZE, TOWER_Z];
    const greenStacked = [TOWER_X, TABLE_Y + CUBE_HALF + CUBE_SIZE * 2, TOWER_Z];

    // Math utilities
    function smoothstep(min, max, val) {
      const x = Math.max(0, Math.min(1, (val - min) / (max - min)));
      return x * x * (3 - 2 * x);
    }

    function lerp(a, b, t) {
      return a + (b - a) * t;
    }

    function vec3Lerp(a, b, t) {
      return [
        lerp(a[0], b[0], t),
        lerp(a[1], b[1], t),
        lerp(a[2], b[2], t),
      ];
    }

    // 6. Closed-Form Analytical Inverse Kinematics (5-DOF Cobot with Top-Down Pitch)
    const BASE_X = -0.28;
    const BASE_Y = 0.83; // 0.72 table + 0.11 pedestal
    const BASE_Z = 0.0;
    const L1 = 0.38;
    const L2 = 0.32;
    const L3 = 0.085;
    const PHI = -Math.PI / 2; // -90 deg perpendicular to tabletop

    function solveIK(tx, ty, tz) {
      const dx = tx - BASE_X;
      const dz = tz - BASE_Z;
      // In Three.js, positive rotation around +Y turns towards -Z, so yaw = -atan2(dz, dx)
      const yaw = -Math.atan2(dz, dx);
      const target_r = Math.sqrt(dx * dx + dz * dz);

      // Wrist pivot position in 2D vertical plane (r, y) with top-down gripper
      const wr = target_r;
      const wy = (ty - BASE_Y) + L3;

      const D = Math.sqrt(wr * wr + wy * wy);
      const maxReach = (L1 + L2) * 0.999;
      const minReach = Math.abs(L1 - L2) * 1.001;
      const clampedD = Math.max(minReach, Math.min(maxReach, D));

      const cosElbow = Math.max(-1, Math.min(1, (clampedD * clampedD - L1 * L1 - L2 * L2) / (2 * L1 * L2)));
      const elbow = -Math.acos(cosElbow);

      const alpha = Math.atan2(wy, wr);
      const beta = Math.atan2(L2 * Math.sin(-elbow), L1 + L2 * Math.cos(-elbow));
      const shoulder = alpha + beta;
      const wrist = PHI - (shoulder + elbow);

      return { yaw, shoulder, elbow, wrist };
    }

    // Standby angles
    const standbyIK = {
      yaw: 0.0,
      shoulder: 0.72,
      elbow: -1.18,
      wrist: -0.32,
    };

    const tempTcpWorld = new THREE.Vector3();

    // 7. Kinematic Frame-by-Frame Solver: Pick Blue -> Place on Red -> Pick Green -> Place on Blue
    function updateKinematics(p) {
      let ik = { ...standbyIK };
      let gripper = 1.0; // 0.0 = clamped, 1.0 = open
      let camPos = [2.25, 1.55, 1.95];
      const camTarget = [0.12, 0.78, 0.0];

      let carryingBlue = false;
      let carryingGreen = false;

      // Segment 0.0 - 0.43: Suspense Darkness, AERO Reveal, Tagline & Instruction Typing
      if (p <= 0.43) {
        const sub = smoothstep(0.38, 0.43, p);
        ik.yaw = lerp(0.0, -0.08, sub);
        ik.shoulder = lerp(0.72, 0.65, sub);
        ik.elbow = lerp(-1.18, -1.06, sub);
        ik.wrist = lerp(-0.32, -0.28, sub);
        gripper = 1.0;

        cubeBlue.position.set(...blueStart);
        cubeGreen.position.set(...greenStart);

        camPos = [lerp(2.25, 2.10, sub), lerp(1.55, 1.48, sub), lerp(1.95, 1.85, sub)];
      }
      // Segment 0.43 - 0.57: STEP 1: Go to BLUE, pick BLUE, place on RED
      else if (p > 0.43 && p <= 0.57) {
        const t = (p - 0.43) / 0.14; // normalized 0.0 to 1.0

        // Exact Waypoints for Blue Pick & Place
        const wp_hoverBlue = [blueStart[0], blueStart[1] + 0.14, blueStart[2]];
        const wp_graspBlue = [blueStart[0], blueStart[1], blueStart[2]];
        const wp_liftBlue = [blueStart[0], 0.98, blueStart[2]];
        const wp_transitBlue = [blueStacked[0], 0.98, blueStacked[2]];
        const wp_placeBlue = [blueStacked[0], blueStacked[1], blueStacked[2]];
        const wp_clearBlue = [blueStacked[0], 0.96, blueStacked[2]];

        let currentTcp = wp_hoverBlue;

        if (t < 0.18) {
          // Approach hover directly above Blue Cube
          const k = smoothstep(0.0, 0.18, t);
          currentTcp = wp_hoverBlue;
          const targetIK = solveIK(wp_hoverBlue[0], wp_hoverBlue[1], wp_hoverBlue[2]);
          ik.yaw = lerp(-0.08, targetIK.yaw, k);
          ik.shoulder = lerp(0.65, targetIK.shoulder, k);
          ik.elbow = lerp(-1.06, targetIK.elbow, k);
          ik.wrist = lerp(-0.28, targetIK.wrist, k);
          gripper = 1.0;
          cubeBlue.position.set(...blueStart);
        } else if (t < 0.28) {
          // Pure vertical descent to grasp Blue Cube
          const k = smoothstep(0.18, 0.28, t);
          currentTcp = vec3Lerp(wp_hoverBlue, wp_graspBlue, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 1.0;
          cubeBlue.position.set(...blueStart);
        } else if (t < 0.35) {
          // Stationary dwell: Clamp gripper jaws firmly around Blue Cube
          const k = smoothstep(0.28, 0.35, t);
          currentTcp = wp_graspBlue;
          ik = solveIK(wp_graspBlue[0], wp_graspBlue[1], wp_graspBlue[2]);
          gripper = lerp(1.0, 0.0, k);
          cubeBlue.position.set(...blueStart);
        } else if (t < 0.48) {
          // Pure vertical lift with Blue Cube securely in jaws
          const k = smoothstep(0.35, 0.48, t);
          currentTcp = vec3Lerp(wp_graspBlue, wp_liftBlue, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 0.0;
          carryingBlue = true;
        } else if (t < 0.68) {
          // Horizontal transit swing across table directly over Red Base Cube
          const k = smoothstep(0.48, 0.68, t);
          currentTcp = vec3Lerp(wp_liftBlue, wp_transitBlue, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 0.0;
          carryingBlue = true;
        } else if (t < 0.80) {
          // Pure vertical descent to Mid stack height (on Red)
          const k = smoothstep(0.68, 0.80, t);
          currentTcp = vec3Lerp(wp_transitBlue, wp_placeBlue, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 0.0;
          carryingBlue = true;
        } else if (t < 0.88) {
          // Stationary dwell: Release gripper jaws on top of Red Cube
          const k = smoothstep(0.80, 0.88, t);
          currentTcp = wp_placeBlue;
          ik = solveIK(wp_placeBlue[0], wp_placeBlue[1], wp_placeBlue[2]);
          gripper = lerp(0.0, 1.0, k);
          cubeBlue.position.set(...blueStacked);
        } else {
          // Pure vertical ascent away from Mid layer
          const k = smoothstep(0.88, 1.0, t);
          currentTcp = vec3Lerp(wp_placeBlue, wp_clearBlue, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 1.0;
          cubeBlue.position.set(...blueStacked);
        }

        cubeGreen.position.set(...greenStart);
      }
      // Segment 0.57 - 0.71: STEP 2: Go to GREEN, pick GREEN, place on BLUE, arm rises up
      else if (p > 0.57 && p <= 0.71) {
        cubeBlue.position.set(...blueStacked);
        const t = (p - 0.57) / 0.14; // normalized 0.0 to 1.0

        // Exact Waypoints for Green Pick & Place
        const wp_clearBlue = [blueStacked[0], 0.96, blueStacked[2]];
        const wp_hoverGreen = [greenStart[0], greenStart[1] + 0.14, greenStart[2]];
        const wp_graspGreen = [greenStart[0], greenStart[1], greenStart[2]];
        const wp_liftGreen = [greenStart[0], 1.02, greenStart[2]];
        const wp_transitGreen = [greenStacked[0], 1.02, greenStacked[2]];
        const wp_placeGreen = [greenStacked[0], greenStacked[1], greenStacked[2]];
        const wp_clearGreen = [greenStacked[0], 1.05, greenStacked[2]];

        let currentTcp = wp_hoverGreen;

        if (t < 0.16) {
          // Swing from tower clear over to Green Cube hover
          const k = smoothstep(0.0, 0.16, t);
          currentTcp = vec3Lerp(wp_clearBlue, wp_hoverGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 1.0;
          cubeGreen.position.set(...greenStart);
        } else if (t < 0.26) {
          // Pure vertical descent to grasp Green Cube
          const k = smoothstep(0.16, 0.26, t);
          currentTcp = vec3Lerp(wp_hoverGreen, wp_graspGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 1.0;
          cubeGreen.position.set(...greenStart);
        } else if (t < 0.34) {
          // Stationary dwell: Clamp gripper jaws firmly around Green Cube
          const k = smoothstep(0.26, 0.34, t);
          currentTcp = wp_graspGreen;
          ik = solveIK(wp_graspGreen[0], wp_graspGreen[1], wp_graspGreen[2]);
          gripper = lerp(1.0, 0.0, k);
          cubeGreen.position.set(...greenStart);
        } else if (t < 0.48) {
          // Pure vertical lift with Green Cube securely in jaws
          const k = smoothstep(0.34, 0.48, t);
          currentTcp = vec3Lerp(wp_graspGreen, wp_liftGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 0.0;
          carryingGreen = true;
        } else if (t < 0.66) {
          // Horizontal transit swing across table directly over Apex of tower (over Blue)
          const k = smoothstep(0.48, 0.66, t);
          currentTcp = vec3Lerp(wp_liftGreen, wp_transitGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 0.0;
          carryingGreen = true;
        } else if (t < 0.78) {
          // Pure vertical descent to Apex stack height (on Blue)
          const k = smoothstep(0.66, 0.78, t);
          currentTcp = vec3Lerp(wp_transitGreen, wp_placeGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 0.0;
          carryingGreen = true;
        } else if (t < 0.86) {
          // Stationary dwell: Release gripper jaws on top of Blue Cube (Green placed!)
          const k = smoothstep(0.78, 0.86, t);
          currentTcp = wp_placeGreen;
          ik = solveIK(wp_placeGreen[0], wp_placeGreen[1], wp_placeGreen[2]);
          gripper = lerp(0.0, 1.0, k);
          cubeGreen.position.set(...greenStacked);
        } else {
          // Pure vertical ascent away from completed 3-layer tower: Arm rises up!
          const k = smoothstep(0.86, 1.0, t);
          currentTcp = vec3Lerp(wp_placeGreen, wp_clearGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 1.0;
          cubeGreen.position.set(...greenStacked);
        }
      }
      // Segment 0.71 - 1.00: 3 Layer Tower Complete, Arm has risen and is completely STOPPED
      else {
        cubeBlue.position.set(...blueStacked);
        cubeGreen.position.set(...greenStacked);

        // Completely stationary in elevated clear posture directly above completed tower
        ik = solveIK(greenStacked[0], 1.05, greenStacked[2]);
        gripper = 1.0;

        camPos = [2.15, 1.50, 1.85];
      }

      // Apply joint rotations directly to arm hierarchy
      yawGroup.rotation.y = ik.yaw;
      shoulderGroup.rotation.z = ik.shoulder;
      elbowGroup.rotation.z = ik.elbow;
      wristGroup.rotation.z = ik.wrist;

      // Jaws clamping: closed = 0.0335m (rubber pads touch cube faces), open = 0.052m
      const sep = 0.0335 + 0.0185 * gripper;
      jawLeft.position.z = sep;
      jawRight.position.z = -sep;

      // Update world matrix to calculate exact gripper jaw center position
      root.updateMatrixWorld(true);
      tcpMarker.getWorldPosition(tempTcpWorld);

      // Carry cube synchronously with the gripper jaws
      if (carryingBlue) {
        cubeBlue.position.copy(tempTcpWorld);
      }
      if (carryingGreen) {
        cubeGreen.position.copy(tempTcpWorld);
      }

      camera.position.set(...camPos);
      camera.lookAt(...camTarget);
    }

    // 8. Synchronize HTML Slide Visibility & Animation
    function updateSlides(p) {
      // Suspense Thriller Vignette Overlay
      if (suspenseOverlay) {
        if (p < 0.10) {
          suspenseOverlay.style.opacity = '1.0';
        } else if (p <= 0.18) {
          suspenseOverlay.style.opacity = String(lerp(1.0, 0.0, smoothstep(0.10, 0.18, p)));
        } else {
          suspenseOverlay.style.opacity = '0.0';
        }
      }

      // Dynamic 3D Workcell Lighting: Dark Suspense -> Smooth Rise to Studio Brilliance
      if (p < 0.08) {
        keySpot.intensity = 0.0;
        fillSpot.intensity = 0.06;
        ambLight.intensity = 0.02;
        hemiLight.intensity = 0.05;
        rimLight.intensity = 0.25;
        scene.background.setHex(0x020204);
        scene.fog.color.setHex(0x020204);
      } else if (p >= 0.08 && p <= 0.18) {
        const lit = smoothstep(0.08, 0.18, p);
        keySpot.intensity = lerp(0.0, 2.4, lit);
        fillSpot.intensity = lerp(0.06, 1.1, lit);
        ambLight.intensity = lerp(0.02, 0.4, lit);
        hemiLight.intensity = lerp(0.05, 0.85, lit);
        rimLight.intensity = lerp(0.25, 0.6, lit);
        scene.background.setHex(0x09090b);
        scene.fog.color.setHex(0x09090b);
      } else {
        keySpot.intensity = 2.4;
        fillSpot.intensity = 1.1;
        ambLight.intensity = 0.4;
        hemiLight.intensity = 0.85;
        rimLight.intensity = 0.6;
        scene.background.setHex(0x09090b);
        scene.fog.color.setHex(0x09090b);
      }

      // Slide 0: Suspense Thriller Hook (Active: 0.00 - 0.08)
      if (slideSuspense) {
        if (p <= 0.08) {
          slideSuspense.style.opacity = '1';
          slideSuspense.style.pointerEvents = 'auto';

          if (suspenseLine1) {
            let op1 = 1.0;
            if (p > 0.045) op1 = smoothstep(0.075, 0.045, p);
            const y1 = lerp(0, -25, smoothstep(0.04, 0.075, p));
            suspenseLine1.style.opacity = op1;
            suspenseLine1.style.transform = `translateY(${y1}px)`;
          }

          if (suspenseIndicator) {
            suspenseIndicator.style.opacity = p < 0.035 ? 1.0 : smoothstep(0.055, 0.035, p);
          }
        } else {
          slideSuspense.style.opacity = '0';
          slideSuspense.style.pointerEvents = 'none';
        }
      }

      // Slide 1: Word AERO (Active: 0.07 - 0.17, Full Plateau: 0.10 - 0.14)
      if (slideAero) {
        let opAero = 0;
        if (p >= 0.07 && p <= 0.10) {
          opAero = smoothstep(0.07, 0.10, p);
        } else if (p > 0.10 && p <= 0.14) {
          opAero = 1.0;
        } else if (p > 0.14 && p <= 0.17) {
          opAero = smoothstep(0.17, 0.14, p);
        }

        const scale = lerp(0.96, 1.0, smoothstep(0.07, 0.12, p));
        if (giantAeroTitle) {
          giantAeroTitle.style.transform = `scale(${scale})`;
        }
        slideAero.style.opacity = opAero;
        slideAero.style.pointerEvents = opAero > 0.1 ? 'auto' : 'none';
      }

      // Sticky transform reset
      if (storySticky) {
        storySticky.style.transform = '';
      }

      // Slide 2: Agentic Execution Tagline (Active: 0.16 - 0.27, Full Plateau: 0.20 - 0.24)
      if (slideTagline) {
        let op = 0;
        let yTagline = 20;
        if (p >= 0.16 && p <= 0.20) {
          op = smoothstep(0.16, 0.20, p);
          yTagline = lerp(20, 0, smoothstep(0.16, 0.20, p));
        } else if (p > 0.20 && p <= 0.24) {
          op = 1.0;
          yTagline = 0;
        } else if (p > 0.24 && p <= 0.27) {
          op = smoothstep(0.27, 0.24, p);
          yTagline = lerp(0, -20, smoothstep(0.24, 0.27, p));
        }
        slideTagline.style.opacity = op;
        slideTagline.style.transform = `translateY(${yTagline}px)`;
        slideTagline.style.pointerEvents = op > 0.1 ? 'auto' : 'none';
      }

      // Slide 2.5: Interactive Natural Language Instruction Typing Box (Active: 0.26 - 0.43)
      if (slidePrompt) {
        let op = 0;
        let yPrompt = 25;
        if (p >= 0.26 && p <= 0.29) {
          op = smoothstep(0.26, 0.29, p);
          yPrompt = lerp(25, 0, smoothstep(0.26, 0.29, p));
        } else if (p > 0.29 && p <= 0.39) {
          op = 1.0;
          yPrompt = 0;
        } else if (p > 0.39 && p <= 0.43) {
          op = smoothstep(0.43, 0.39, p);
          yPrompt = lerp(0, -25, smoothstep(0.39, 0.43, p));
        }
        slidePrompt.style.opacity = op;
        slidePrompt.style.transform = `translateY(${yPrompt}px)`;
        slidePrompt.style.pointerEvents = op > 0.1 ? 'auto' : 'none';

        // Typing Effect: types the command between p = 0.29 and 0.37
        const fullInstruction = "Build a 3 layer tower: stack blue on red, then green on apex.";
        const typeT = Math.max(0, Math.min(1, (p - 0.29) / 0.08));
        const count = Math.floor(typeT * fullInstruction.length);

        if (instructionTypedText && op > 0.02) {
          instructionTypedText.textContent = fullInstruction.slice(0, count);
        }

        if (typeT >= 1.0) {
          if (instructionBox) instructionBox.classList.add('active-executing');
          if (instructionStatusPill) instructionStatusPill.classList.add('status-dispatched');
          if (instructionPillText) instructionPillText.textContent = "✓ Compiled & Dispatched to Workcell";
        } else {
          if (instructionBox) instructionBox.classList.remove('active-executing');
          if (instructionStatusPill) instructionStatusPill.classList.remove('status-dispatched');
          if (instructionPillText) instructionPillText.textContent = typeT > 0 ? "Typing instruction..." : "Awaiting command...";
        }
      }

      // Slide 3: Robot HUD overlay (Active during arm action: 0.42 - 0.72)
      if (slideRobotHud) {
        let op = 0;
        if (p >= 0.42 && p <= 0.45) op = smoothstep(0.42, 0.45, p);
        else if (p > 0.45 && p <= 0.69) op = 1.0;
        else if (p > 0.69 && p <= 0.72) op = smoothstep(0.72, 0.69, p);
        slideRobotHud.style.opacity = op;

        if (hudFrameCounter && op > 0.05) {
          const frameProgress = Math.min(100, Math.round(smoothstep(0.43, 0.70, p) * 100));
          hudFrameCounter.innerText = `FRAME ${frameProgress < 10 ? '0' + frameProgress : frameProgress} / 100`;
        }
      }

      // Slide 4: Description (Active: 0.72 - 0.81, Plateau: 0.75 - 0.78), Revealed strictly after arm stops
      if (slideDesc) {
        let op = 0;
        if (p >= 0.72 && p <= 0.75) op = smoothstep(0.72, 0.75, p);
        else if (p > 0.75 && p <= 0.78) op = 1.0;
        else if (p > 0.78 && p <= 0.81) op = smoothstep(0.81, 0.78, p);
        slideDesc.style.opacity = op;
        slideDesc.style.transform = `translateY(${lerp(30, -20, smoothstep(0.72, 0.81, p))}px)`;
        slideDesc.style.pointerEvents = op > 0.1 ? 'auto' : 'none';
      }

      // Slide 5: Stats (Active: 0.80 - 0.89, Plateau: 0.83 - 0.86)
      if (slideStats) {
        let op = 0;
        if (p >= 0.80 && p <= 0.83) op = smoothstep(0.80, 0.83, p);
        else if (p > 0.83 && p <= 0.86) op = 1.0;
        else if (p > 0.86 && p <= 0.89) op = smoothstep(0.89, 0.86, p);
        slideStats.style.opacity = op;
        slideStats.style.transform = `translateY(${lerp(30, -20, smoothstep(0.80, 0.89, p))}px)`;
        slideStats.style.pointerEvents = op > 0.1 ? 'auto' : 'none';
      }

      // Slide 6: Try it yourself CTAs & Verified Manipulation Commands (Active: 0.88 - 1.00)
      // Staggered reveal: Launch button arrives first, suggestion / commands arrive afterwards
      if (slideCtas) {
        let opCtas = 0;
        if (p >= 0.88) opCtas = smoothstep(0.88, 0.91, p);
        slideCtas.style.opacity = opCtas;
        slideCtas.style.pointerEvents = opCtas > 0.1 ? 'auto' : 'none';

        // Stage 1: "Try it yourself" + "Launch Interactive 3D Studio" button arrives first
        if (storyCtaHeroBlock) {
          let opBtn = 0;
          let yBtn = 25;
          if (p >= 0.88 && p <= 0.92) {
            const k = smoothstep(0.88, 0.92, p);
            opBtn = k;
            yBtn = lerp(25, 0, k);
          } else if (p > 0.92) {
            opBtn = 1.0;
            yBtn = 0;
          }
          storyCtaHeroBlock.style.opacity = opBtn;
          storyCtaHeroBlock.style.transform = `translateY(${yBtn}px)`;
          storyCtaHeroBlock.style.pointerEvents = opBtn > 0.1 ? 'auto' : 'none';
        }

        // Stage 2: Commands & Suggestions arrive strictly afterwards
        if (storyCommandsWrap) {
          let opCmds = 0;
          let yCmds = 30;
          if (p >= 0.925 && p <= 0.965) {
            const k = smoothstep(0.925, 0.965, p);
            opCmds = k;
            yCmds = lerp(30, 0, k);
          } else if (p > 0.965) {
            opCmds = 1.0;
            yCmds = 0;
          }
          storyCommandsWrap.style.opacity = opCmds;
          storyCommandsWrap.style.transform = `translateY(${yCmds}px)`;
          storyCommandsWrap.style.pointerEvents = opCmds > 0.1 ? 'auto' : 'none';
        }
      }
    }

    // 9. Synchronous Scroll Rendering (1:1 with Scroll Events)
    function renderCurrent(p) {
      updateKinematics(p);
      updateSlides(p);
      renderer.render(scene, camera);
    }

    let ticking = false;
    let lastProgress = 0.0;

    function onScroll() {
      const rect = storyContainer.getBoundingClientRect();
      const totalScrollable = storyContainer.offsetHeight - window.innerHeight;
      if (totalScrollable <= 0) return;

      const scrolled = -rect.top;
      const p = Math.max(0, Math.min(1, scrolled / totalScrollable));
      lastProgress = p;

      if (!ticking) {
        requestAnimationFrame(() => {
          renderCurrent(lastProgress);
          ticking = false;
        });
        ticking = true;
      }
    }

    // Window Resize Handler
    function onResize() {
      const w = window.innerWidth;
      const h = window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h, false);
      renderCurrent(lastProgress);
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onResize);

    // Initial render
    onScroll();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initScrollRobotStory);
  } else {
    initScrollRobotStory();
  }
})();
