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

    // Story Slide Elements
    const slideAero = document.getElementById('slide-aero');
    const slideTagline = document.getElementById('slide-tagline');
    const slideRobotHud = document.getElementById('slide-robot-hud');
    const hudFrameCounter = document.getElementById('hud-frame-counter');
    const slideDesc = document.getElementById('slide-desc');
    const slideStats = document.getElementById('slide-stats');
    const slideCtas = document.getElementById('slide-ctas');

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

      // Segment 0.0 - 0.26: Standby & Initial Wakeup
      if (p <= 0.26) {
        const sub = smoothstep(0.12, 0.26, p);
        ik.yaw = lerp(0.0, -0.08, sub);
        ik.shoulder = lerp(0.72, 0.65, sub);
        ik.elbow = lerp(-1.18, -1.06, sub);
        ik.wrist = lerp(-0.32, -0.28, sub);
        gripper = 1.0;

        cubeBlue.position.set(...blueStart);
        cubeGreen.position.set(...greenStart);

        camPos = [lerp(2.25, 2.10, sub), lerp(1.55, 1.48, sub), lerp(1.95, 1.85, sub)];
      }
      // Segment 0.26 - 0.44: STEP 1: Go to BLUE, pick BLUE, place on RED
      else if (p > 0.26 && p <= 0.44) {
        const t = (p - 0.26) / 0.18; // normalized 0.0 to 1.0

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
      // Segment 0.44 - 0.62: STEP 2: Go to GREEN, pick GREEN, place on BLUE
      else if (p > 0.44 && p <= 0.62) {
        cubeBlue.position.set(...blueStacked);
        const t = (p - 0.44) / 0.18; // normalized 0.0 to 1.0

        // Exact Waypoints for Green Pick & Place
        const wp_clearBlue = [blueStacked[0], 0.96, blueStacked[2]];
        const wp_hoverGreen = [greenStart[0], greenStart[1] + 0.14, greenStart[2]];
        const wp_graspGreen = [greenStart[0], greenStart[1], greenStart[2]];
        const wp_liftGreen = [greenStart[0], 1.02, greenStart[2]];
        const wp_transitGreen = [greenStacked[0], 1.02, greenStacked[2]];
        const wp_placeGreen = [greenStacked[0], greenStacked[1], greenStacked[2]];
        const wp_clearGreen = [greenStacked[0], 1.02, greenStacked[2]];

        let currentTcp = wp_hoverGreen;

        if (t < 0.18) {
          // Swing from tower clear over to Green Cube hover
          const k = smoothstep(0.0, 0.18, t);
          currentTcp = vec3Lerp(wp_clearBlue, wp_hoverGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 1.0;
          cubeGreen.position.set(...greenStart);
        } else if (t < 0.28) {
          // Pure vertical descent to grasp Green Cube
          const k = smoothstep(0.18, 0.28, t);
          currentTcp = vec3Lerp(wp_hoverGreen, wp_graspGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 1.0;
          cubeGreen.position.set(...greenStart);
        } else if (t < 0.35) {
          // Stationary dwell: Clamp gripper jaws firmly around Green Cube
          const k = smoothstep(0.28, 0.35, t);
          currentTcp = wp_graspGreen;
          ik = solveIK(wp_graspGreen[0], wp_graspGreen[1], wp_graspGreen[2]);
          gripper = lerp(1.0, 0.0, k);
          cubeGreen.position.set(...greenStart);
        } else if (t < 0.48) {
          // Pure vertical lift with Green Cube securely in jaws
          const k = smoothstep(0.35, 0.48, t);
          currentTcp = vec3Lerp(wp_graspGreen, wp_liftGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 0.0;
          carryingGreen = true;
        } else if (t < 0.68) {
          // Horizontal transit swing across table directly over Apex of tower (over Blue)
          const k = smoothstep(0.48, 0.68, t);
          currentTcp = vec3Lerp(wp_liftGreen, wp_transitGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 0.0;
          carryingGreen = true;
        } else if (t < 0.80) {
          // Pure vertical descent to Apex stack height (on Blue)
          const k = smoothstep(0.68, 0.80, t);
          currentTcp = vec3Lerp(wp_transitGreen, wp_placeGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 0.0;
          carryingGreen = true;
        } else if (t < 0.88) {
          // Stationary dwell: Release gripper jaws on top of Blue Cube
          const k = smoothstep(0.80, 0.88, t);
          currentTcp = wp_placeGreen;
          ik = solveIK(wp_placeGreen[0], wp_placeGreen[1], wp_placeGreen[2]);
          gripper = lerp(0.0, 1.0, k);
          cubeGreen.position.set(...greenStacked);
        } else {
          // Pure vertical ascent away from completed 3-layer tower
          const k = smoothstep(0.88, 1.0, t);
          currentTcp = vec3Lerp(wp_placeGreen, wp_clearGreen, k);
          ik = solveIK(currentTcp[0], currentTcp[1], currentTcp[2]);
          gripper = 1.0;
          cubeGreen.position.set(...greenStacked);
        }
      }
      // Segment 0.62 - 1.0: Tower Complete — Arm Ascends to High Standby
      else {
        cubeBlue.position.set(...blueStacked);
        cubeGreen.position.set(...greenStacked);

        const sub = smoothstep(0.62, 0.76, p);
        const clearPose = solveIK(greenStacked[0], 1.02, greenStacked[2]);

        ik.yaw = lerp(clearPose.yaw, -0.15, sub);
        ik.shoulder = lerp(clearPose.shoulder, 0.76, sub);
        ik.elbow = lerp(clearPose.elbow, -1.24, sub);
        ik.wrist = lerp(clearPose.wrist, -0.32, sub);
        gripper = 1.0;

        camPos = [lerp(2.10, 1.95, sub), lerp(1.48, 1.38, sub), lerp(1.85, 1.72, sub)];
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
      // Slide 1: Big AERO (Active: 0.00 - 0.15)
      if (slideAero) {
        const op = p < 0.10 ? 1.0 : smoothstep(0.15, 0.10, p);
        const scale = lerp(1.0, 0.88, smoothstep(0.0, 0.15, p));
        slideAero.style.opacity = op;
        slideAero.style.transform = `scale(${scale}) translateY(${lerp(0, -30, smoothstep(0.0, 0.15, p))}px)`;
        slideAero.style.pointerEvents = op > 0.1 ? 'auto' : 'none';
      }

      // Slide 2: Agentic Execution (Active: 0.14 - 0.28)
      if (slideTagline) {
        let op = 0;
        if (p >= 0.12 && p <= 0.20) {
          op = smoothstep(0.12, 0.18, p);
        } else if (p > 0.20 && p <= 0.29) {
          op = smoothstep(0.29, 0.23, p);
        }
        slideTagline.style.opacity = op;
        slideTagline.style.transform = `translateY(${lerp(25, -20, smoothstep(0.12, 0.29, p))}px)`;
        slideTagline.style.pointerEvents = op > 0.1 ? 'auto' : 'none';
      }

      // Slide 3: Robot HUD overlay (Active during arm action: 0.26 - 0.64)
      if (slideRobotHud) {
        let op = 0;
        if (p >= 0.25 && p <= 0.32) op = smoothstep(0.25, 0.32, p);
        else if (p > 0.32 && p <= 0.62) op = 1.0;
        else if (p > 0.62 && p <= 0.67) op = smoothstep(0.67, 0.62, p);
        slideRobotHud.style.opacity = op;

        if (hudFrameCounter && op > 0.05) {
          const frameProgress = Math.round(smoothstep(0.26, 0.62, p) * 100);
          hudFrameCounter.innerText = `FRAME ${frameProgress < 10 ? '0' + frameProgress : frameProgress} / 100`;
        }
      }

      // Slide 4: Description (Active: 0.64 - 0.77)
      if (slideDesc) {
        let op = 0;
        if (p >= 0.63 && p <= 0.70) op = smoothstep(0.63, 0.69, p);
        else if (p > 0.70 && p <= 0.77) op = smoothstep(0.77, 0.72, p);
        slideDesc.style.opacity = op;
        slideDesc.style.transform = `translateY(${lerp(30, -20, smoothstep(0.63, 0.77, p))}px)`;
        slideDesc.style.pointerEvents = op > 0.1 ? 'auto' : 'none';
      }

      // Slide 5: Stats (Active: 0.76 - 0.88)
      if (slideStats) {
        let op = 0;
        if (p >= 0.75 && p <= 0.81) op = smoothstep(0.75, 0.81, p);
        else if (p > 0.81 && p <= 0.88) op = smoothstep(0.88, 0.83, p);
        slideStats.style.opacity = op;
        slideStats.style.transform = `translateY(${lerp(30, -20, smoothstep(0.75, 0.88, p))}px)`;
        slideStats.style.pointerEvents = op > 0.1 ? 'auto' : 'none';
      }

      // Slide 6: Try it yourself CTAs (Active: 0.88 - 1.00)
      if (slideCtas) {
        let op = 0;
        if (p >= 0.87) op = smoothstep(0.87, 0.94, p);
        slideCtas.style.opacity = op;
        slideCtas.style.transform = `translateY(${lerp(30, 0, smoothstep(0.87, 0.96, p))}px)`;
        slideCtas.style.pointerEvents = op > 0.1 ? 'auto' : 'none';
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
