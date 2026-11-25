// AERO SO-100 Manipulator Interactive Exploded View & Internal Anatomy Engine
// Powered by Three.js & exploded-view skill (Exploder, Labels, Xray, Section, Picker)

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Exploder, Labels, Xray, Section, Picker } from './explode.js';

// Query parameters & setup
const params = new URLSearchParams(location.search);
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;

const canvas = document.getElementById('gl');
const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  alpha: false,
  powerPreference: 'high-performance'
});

renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight, false);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.localClippingEnabled = true;

const scene = new THREE.Scene();
const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.85;

const camera = new THREE.PerspectiveCamera(36, innerWidth / innerHeight, 0.05, 50);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 0.4;
controls.maxDistance = 5.0;
controls.maxPolarAngle = Math.PI / 2 + 0.05;

// Studio Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambientLight);

const keyLight = new THREE.DirectionalLight(0xfff5ea, 2.4);
keyLight.position.set(2.0, 3.5, 2.5);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.bias = -0.0003;
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 8;
keyLight.shadow.camera.left = -1.5;
keyLight.shadow.camera.right = 1.5;
keyLight.shadow.camera.top = 1.5;
keyLight.shadow.camera.bottom = -1.5;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x38bdf8, 1.2);
fillLight.position.set(-2.5, 2.0, -1.8);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xa78bfa, 0.9);
rimLight.position.set(0, -2.0, -2.5);
scene.add(rimLight);

// Ground Shadow Floor
const floorGeo = new THREE.PlaneGeometry(10, 10);
const floorMat = new THREE.ShadowMaterial({ opacity: 0.28 });
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.01;
floor.receiveShadow = true;
scene.add(floor);

// Ground Grid Helper
const grid = new THREE.GridHelper(4, 20, 0x38bdf8, 0x1e293b);
grid.position.y = 0.0;
scene.add(grid);

// ---------------------------------------------------------------- Materials
const mat = (opts) => new THREE.MeshPhysicalMaterial({ roughness: 0.45, metalness: 0.25, ...opts });

const SHELL_MAT_CF = mat({ color: '#1e2530', roughness: 0.35, metalness: 0.2, clearcoat: 0.3 });
const SHELL_MAT_ACCENT = mat({ color: '#0284c7', roughness: 0.32, metalness: 0.3, clearcoat: 0.4 });
const SERVO_CASE = mat({ color: '#0f172a', roughness: 0.4, metalness: 0.4 });
const HEATSINK_ALU = mat({ color: '#64748b', roughness: 0.25, metalness: 0.9 });
const HORN_ALU = mat({ color: '#38bdf8', roughness: 0.28, metalness: 0.85 });
const STEEL_GEAR = mat({ color: '#cbd5e1', roughness: 0.28, metalness: 0.95 });
const BRASS_PINION = mat({ color: '#f59e0b', roughness: 0.35, metalness: 0.85 });
const COPPER_COIL = mat({ color: '#ea580c', roughness: 0.32, metalness: 0.92 });
const ENCODER_PCB = mat({ color: '#047857', roughness: 0.45, metalness: 0.3 });
const WIRE_RED = mat({ color: '#ef4444', roughness: 0.4, metalness: 0.1 });
const WIRE_BLK = mat({ color: '#1e293b', roughness: 0.5, metalness: 0.05 });
const WIRE_SIG = mat({ color: '#06b6d4', roughness: 0.3, metalness: 0.1 });
const CONNECTOR_WHT = mat({ color: '#f8fafc', roughness: 0.5, metalness: 0.1 });
const GRIP_RUBBER = mat({ color: '#090d16', roughness: 0.85, metalness: 0.05 });

// ---------------------------------------------------------------- Robot Arm Root & Parts
const armAssembly = new THREE.Group();
scene.add(armAssembly);

function createPart(name, opts = {}) {
  const g = new THREE.Group();
  g.name = name;
  Object.assign(g.userData, opts);
  armAssembly.add(g);
  return g;
}

const addMesh = (parent, geo, m) => {
  const o = new THREE.Mesh(geo, m);
  o.castShadow = true;
  o.receiveShadow = true;
  parent.add(o);
  return o;
};

// ----------------------------------------------------------------
// 1. BASE TURNTABLE & YAW ACTUATOR (Feetech STS3215 ID: 1)
// ----------------------------------------------------------------
// 1A. Base Mount Outer Shell & Skirt
const baseShell = createPart('Base Mounting Flange & Skirt', {
  shell: true,
  stage: 0,
  explode: [0, -0.22, 0],
  category: 'Structural Chassis',
  specs: {
    material: 'PETG-CF / Carbon Fiber Infill',
    mounting: '4x M4 Counter-sunk fasteners to aluminum extrusion table',
    weight: '145 g',
    description: 'Rigid base foundation providing counter-torque mounting and wire breakout passage.'
  }
});
const basePlinth = addMesh(baseShell, new THREE.CylinderGeometry(0.11, 0.125, 0.022, 36), SHELL_MAT_CF);
basePlinth.position.y = 0.011;
const baseRim = addMesh(baseShell, new THREE.TorusGeometry(0.118, 0.008, 16, 40), SHELL_MAT_ACCENT);
baseRim.rotation.x = Math.PI / 2;
baseRim.position.y = 0.022;

// 1B. Main Base Controller PCB (RP2040 Bus Transceiver & 12V Step-Down)
const basePCB = createPart('AERO Master Bus Controller Board', {
  stage: 2,
  explode: [0, -0.11, 0],
  category: 'Electronics & Control',
  specs: {
    controller: 'Dual Cortex-M0+ RP2040 @ 133 MHz',
    busInterface: 'Half-Duplex TTL Serial Bus (1.0 Mbps Baud Rate)',
    powerInput: '12.0 V DC @ 8.0 A Max (Mean Well Power Supply)',
    firmware: 'AERO Real-Time Kinematics & High-Speed Telemetry Bridge'
  }
});
const pcbPlate = addMesh(basePCB, new THREE.BoxGeometry(0.08, 0.004, 0.08), ENCODER_PCB);
pcbPlate.position.y = 0.026;
// Terminal block
const termBlock = addMesh(basePCB, new THREE.BoxGeometry(0.025, 0.014, 0.02), mat({ color: '#15803d' }));
termBlock.position.set(-0.024, 0.034, -0.024);
// Microcontroller IC
const mcuChip = addMesh(basePCB, new THREE.BoxGeometry(0.012, 0.003, 0.012), mat({ color: '#090d16', metalness: 0.8 }));
mcuChip.position.set(0.005, 0.029, 0.005);

// 1C. Base 6806ZZ High-Precision Deep-Groove Ball Bearing
const baseBearing = createPart('Base Turntable Ball Bearing (6806ZZ)', {
  stage: 1,
  explode: [0, -0.05, 0],
  category: 'Mechanical Transmission',
  specs: {
    type: 'Deep-groove Radial Ball Bearing',
    dimensions: 'ID 30mm x OD 42mm x Width 7mm',
    material: 'High-Carbon GCr15 Chrome Bearing Steel',
    dynamicLoad: '4.5 kN axial/radial dynamic thrust rating'
  }
});
const outerRace = addMesh(baseBearing, new THREE.CylinderGeometry(0.048, 0.048, 0.012, 32, 1, true), STEEL_GEAR);
outerRace.position.y = 0.035;
const innerRace = addMesh(baseBearing, new THREE.CylinderGeometry(0.032, 0.032, 0.012, 32, 1, true), STEEL_GEAR);
innerRace.position.y = 0.035;
// Ball instances
for (let i = 0; i < 12; i++) {
  const angle = (i / 12) * Math.PI * 2;
  const ball = addMesh(baseBearing, new THREE.SphereGeometry(0.0042, 12, 8), STEEL_GEAR);
  ball.position.set(Math.cos(angle) * 0.040, 0.035, Math.sin(angle) * 0.040);
}

// 1D. Base Yaw Actuator STS3215 (ID 1) - Servo Assembly
const baseYawServo = createPart('Base Yaw Actuator STS3215 (ID 1)', {
  stage: 1,
  explode: [-0.14, 0.04, -0.14],
  category: 'Smart Serial Bus Servo',
  specs: {
    model: 'Feetech STS3215 High-Torque Bus Servo',
    busId: '1 (Base Yaw Axis)',
    stallTorque: '30.0 kg·cm @ 12V (2.94 N·m)',
    speed: '0.14 sec/60° (70 RPM)',
    encoder: '12-bit Contactless Magnetic (4096 Steps / 0.088° Resolution)',
    operatingVoltage: '7.4V - 12.0V DC (Nominal 12.0V)'
  }
});
// Servo Main Housing
const servoBody1 = addMesh(baseYawServo, new THREE.BoxGeometry(0.042, 0.045, 0.024), SERVO_CASE);
servoBody1.position.set(0, 0.052, 0);
// CNC Aluminum Heatsink Mid-Plate
const heatsink1 = addMesh(baseYawServo, new THREE.BoxGeometry(0.043, 0.014, 0.025), HEATSINK_ALU);
heatsink1.position.set(0, 0.052, 0);
// Aluminum Spline Output Horn
const horn1 = addMesh(baseYawServo, new THREE.CylinderGeometry(0.013, 0.013, 0.007, 24), HORN_ALU);
horn1.position.set(0.01, 0.078, 0);

// 1E. Internal Stator & Metal Gearbox of Base Servo (Exploded to reveal inside the motor!)
const baseInternalMotor = createPart('Base Actuator Core: Stator, Gears & Encoder', {
  stage: 2,
  explode: [-0.28, 0.04, -0.28],
  category: 'Motor Internals',
  specs: {
    gearTrain: '4-Stage All-Metal Hardened Steel Spur Reduction (1:345)',
    stator: 'High-density 9-Pole Copper Coil Armature',
    rotor: 'Sintered NdFeB Neodymium Permanent Magnet',
    encoderSensor: 'AS5600 12-Bit On-Axis Magnetic Angle IC'
  }
});
// Stator core with copper windings
const statorCore1 = addMesh(baseInternalMotor, new THREE.CylinderGeometry(0.011, 0.011, 0.022, 16), mat({ color: '#475569', metalness: 0.8 }));
statorCore1.position.set(-0.01, 0.052, 0);
for (let c = 0; c < 6; c++) {
  const ca = (c / 6) * Math.PI * 2;
  const coil = addMesh(baseInternalMotor, new THREE.BoxGeometry(0.005, 0.018, 0.006), COPPER_COIL);
  coil.position.set(-0.01 + Math.cos(ca) * 0.010, 0.052, Math.sin(ca) * 0.010);
}
// Reduction Spur Gears
const gearA = addMesh(baseInternalMotor, new THREE.CylinderGeometry(0.010, 0.010, 0.004, 20), STEEL_GEAR);
gearA.position.set(0.01, 0.060, 0);
const gearB = addMesh(baseInternalMotor, new THREE.CylinderGeometry(0.006, 0.006, 0.004, 16), BRASS_PINION);
gearB.position.set(0.01, 0.066, 0);
// Magnetic Encoder Board
const encBoard1 = addMesh(baseInternalMotor, new THREE.BoxGeometry(0.016, 0.002, 0.016), ENCODER_PCB);
encBoard1.position.set(-0.01, 0.035, 0);

// ----------------------------------------------------------------
// 2. SHOULDER PITCH JOINT & BICEP ASSEMBLY (STS3215 ID: 2)
// ----------------------------------------------------------------
// 2A. Turntable Upper Hub (Connects Base Horn to Shoulder Yoke)
const shoulderHub = createPart('Shoulder Base Turntable Hub', {
  stage: 0,
  explode: [0, 0.08, 0],
  category: 'Structural Joint',
  specs: {
    material: 'CNC Anodized 6061-T6 Aluminum',
    splineInterface: '25T Standard Servo Spline with M3 center lock screw',
    weight: '48 g'
  }
});
const hubDisk = addMesh(shoulderHub, new THREE.CylinderGeometry(0.038, 0.038, 0.018, 32), SHELL_MAT_CF);
hubDisk.position.y = 0.085;

// 2B. Shoulder Outer Enclosure Clamshell Left (shell: true)
const shoulderShellLeft = createPart('Shoulder Casing (Left Half)', {
  shell: true,
  stage: 0,
  explode: [-0.22, 0.18, 0],
  category: 'Protective Enclosure',
  specs: {
    material: 'PETG Carbon Fiber Shell (2.4mm wall thickness)',
    features: 'Wire routing channel, bearing seat for secondary pivot support',
    weight: '62 g'
  }
});
const shLeftBox = addMesh(shoulderShellLeft, new THREE.BoxGeometry(0.024, 0.12, 0.06), SHELL_MAT_CF);
shLeftBox.position.set(-0.032, 0.155, 0);
const shAccent1 = addMesh(shoulderShellLeft, new THREE.BoxGeometry(0.004, 0.08, 0.04), SHELL_MAT_ACCENT);
shAccent1.position.set(-0.045, 0.155, 0);

// 2C. Shoulder Outer Enclosure Clamshell Right (shell: true)
const shoulderShellRight = createPart('Shoulder Casing (Right Half)', {
  shell: true,
  stage: 0,
  explode: [0.22, 0.18, 0],
  category: 'Protective Enclosure',
  specs: {
    material: 'PETG Carbon Fiber Shell',
    features: 'M3 Brass threaded heat-set inserts for rigid clamping',
    weight: '58 g'
  }
});
const shRightBox = addMesh(shoulderShellRight, new THREE.BoxGeometry(0.024, 0.12, 0.06), SHELL_MAT_CF);
shRightBox.position.set(0.032, 0.155, 0);

// 2D. Shoulder Pitch Actuator STS3215 (ID 2)
const shoulderServo = createPart('Shoulder Pitch Actuator STS3215 (ID 2)', {
  stage: 1,
  explode: [0, 0.22, -0.18],
  category: 'Smart Serial Bus Servo',
  specs: {
    model: 'Feetech STS3215 High-Torque Bus Servo',
    busId: '2 (Shoulder Pitch Axis)',
    stallTorque: '30.0 kg·cm @ 12V (2.94 N·m)',
    dualBearing: 'Supported by counter-bearing on output shaft opposite horn',
    peakPower: '32.4 Watts'
  }
});
const shServoBody = addMesh(shoulderServo, new THREE.BoxGeometry(0.024, 0.045, 0.042), SERVO_CASE);
shServoBody.position.set(0, 0.155, 0);
const shHeatsink = addMesh(shoulderServo, new THREE.BoxGeometry(0.025, 0.014, 0.043), HEATSINK_ALU);
shHeatsink.position.set(0, 0.155, 0);
const shHorn = addMesh(shoulderServo, new THREE.CylinderGeometry(0.013, 0.013, 0.008, 24), HORN_ALU);
shHorn.rotation.z = Math.PI / 2;
shHorn.position.set(0.018, 0.165, 0.01);

// 2E. Shoulder Internal Motor Core & Gears
const shoulderInternals = createPart('Shoulder Actuator Internal Gearset & Stator', {
  stage: 2,
  explode: [0, 0.32, -0.32],
  category: 'Motor Internals',
  specs: {
    gearbox: 'Multi-stage Planetary/Spur steel gear transmission (1:345)',
    encoder: 'Magnetic sensing disk positioned directly opposite the output shaft',
    backlash: '< 0.4° mechanical play under rated load'
  }
});
const shStator = addMesh(shoulderInternals, new THREE.CylinderGeometry(0.010, 0.010, 0.020, 16), COPPER_COIL);
shStator.rotation.x = Math.PI / 2;
shStator.position.set(0, 0.155, -0.01);
const shGear1 = addMesh(shoulderInternals, new THREE.CylinderGeometry(0.012, 0.012, 0.005, 20), STEEL_GEAR);
shGear1.rotation.z = Math.PI / 2;
shGear1.position.set(0.008, 0.165, 0.01);

// 2F. Bicep Structure Link Tube & Channel
const bicepLink = createPart('Bicep Spar & Cable Trunk Channel', {
  stage: 1,
  explode: [0, 0.28, 0.18],
  category: 'Structural Link',
  specs: {
    material: 'Reinforced Carbon Fiber Tube Spar',
    length: '180 mm effective joint center-to-center',
    cableGuide: 'Internal conduit isolating 12V bus wires from joint rotation'
  }
});
const bicepTube = addMesh(bicepLink, new THREE.CylinderGeometry(0.018, 0.020, 0.16, 24), SHELL_MAT_CF);
bicepTube.position.set(0, 0.28, 0);
const bicepStripe = addMesh(bicepLink, new THREE.CylinderGeometry(0.0185, 0.0185, 0.02, 24), SHELL_MAT_ACCENT);
bicepStripe.position.set(0, 0.28, 0);

// ----------------------------------------------------------------
// 3. ELBOW PITCH JOINT ASSEMBLY (STS3215 ID: 3)
// ----------------------------------------------------------------
// 3A. Elbow Outer Protective Casing (shell: true)
const elbowShell = createPart('Elbow Joint Clamshell Shell', {
  shell: true,
  stage: 0,
  explode: [0, 0.48, -0.22],
  category: 'Protective Enclosure',
  specs: {
    material: 'PETG-CF Lightweight Shell',
    protection: 'Seals actuator gearbox from dust & debris in tabletop environments',
    weight: '44 g'
  }
});
const elbowCasing = addMesh(elbowShell, new THREE.BoxGeometry(0.05, 0.065, 0.065), SHELL_MAT_CF);
elbowCasing.position.set(0, 0.42, 0);

// 3B. Elbow Pitch Actuator STS3215 (ID 3)
const elbowServo = createPart('Elbow Pitch Actuator STS3215 (ID 3)', {
  stage: 1,
  explode: [-0.18, 0.42, 0],
  category: 'Smart Serial Bus Servo',
  specs: {
    model: 'Feetech STS3215 High-Torque Bus Servo',
    busId: '3 (Elbow Pitch Axis)',
    stallTorque: '30.0 kg·cm @ 12V (2.94 N·m)',
    range: '360° Continuous Turn / -180° to +180° Position Mode',
    currentSense: 'Integrated real-time current telemetry (A)'
  }
});
const elServoBody = addMesh(elbowServo, new THREE.BoxGeometry(0.024, 0.045, 0.042), SERVO_CASE);
elServoBody.position.set(0, 0.42, 0);
const elHeatsink = addMesh(elbowServo, new THREE.BoxGeometry(0.025, 0.014, 0.043), HEATSINK_ALU);
elHeatsink.position.set(0, 0.42, 0);
const elHorn = addMesh(elbowServo, new THREE.CylinderGeometry(0.013, 0.013, 0.008, 24), HORN_ALU);
elHorn.rotation.z = Math.PI / 2;
elHorn.position.set(0.018, 0.42, 0.01);

// 3C. Elbow Internal Motor Stator & Gears
const elbowInternals = createPart('Elbow Motor Copper Stator & Gear Reduction', {
  stage: 2,
  explode: [-0.34, 0.42, 0],
  category: 'Motor Internals',
  specs: {
    windings: 'High-temperature enamelled copper wire (Class H 180°C)',
    reduction: 'Case-hardened carbon steel spur gear train',
    thermalProtection: 'Internal NTC thermistor reporting real-time servo temperature'
  }
});
const elStatorMesh = addMesh(elbowInternals, new THREE.CylinderGeometry(0.009, 0.009, 0.018, 16), COPPER_COIL);
elStatorMesh.position.set(0, 0.42, -0.01);
const elGearsMesh = addMesh(elbowInternals, new THREE.CylinderGeometry(0.011, 0.011, 0.005, 20), STEEL_GEAR);
elGearsMesh.position.set(0, 0.42, 0.01);

// 3D. Forearm Structural Tube
const forearmLink = createPart('Forearm Tube Chassis', {
  stage: 1,
  explode: [0, 0.54, 0.15],
  category: 'Structural Link',
  specs: {
    material: 'Carbon Fiber Weave Rod',
    length: '150 mm effective span',
    weight: '36 g'
  }
});
const faTube = addMesh(forearmLink, new THREE.CylinderGeometry(0.015, 0.016, 0.13, 24), SHELL_MAT_CF);
faTube.position.set(0, 0.54, 0);

// ----------------------------------------------------------------
// 4. WRIST PITCH & ROLL GIMBAL ASSEMBLY (STS3215 ID: 4 & ID: 5)
// ----------------------------------------------------------------
// 4A. Wrist Casing / Gimbal Yoke (shell: true)
const wristYokeShell = createPart('Wrist Gimbal Yoke Casing', {
  shell: true,
  stage: 0,
  explode: [0.20, 0.68, 0],
  category: 'Protective Enclosure',
  specs: {
    material: 'PETG-CF 3D Print',
    jointConfiguration: 'Dual-axis intersecting Pitch/Roll Gimbal',
    weight: '38 g'
  }
});
const yokeBox = addMesh(wristYokeShell, new THREE.BoxGeometry(0.044, 0.06, 0.044), SHELL_MAT_CF);
yokeBox.position.set(0, 0.66, 0);

// 4B. Wrist Pitch Actuator STS3215 (ID 4)
const wristPitchServo = createPart('Wrist Pitch Actuator STS3215 (ID 4)', {
  stage: 1,
  explode: [-0.16, 0.66, 0.14],
  category: 'Smart Serial Bus Servo',
  specs: {
    model: 'Feetech STS3215 High-Torque Bus Servo',
    busId: '4 (Wrist Pitch Axis)',
    stallTorque: '30.0 kg·cm @ 12V (2.94 N·m)',
    positionFeedback: 'Magnetic encoder with ±0.088° precision'
  }
});
const wpBody = addMesh(wristPitchServo, new THREE.BoxGeometry(0.022, 0.038, 0.038), SERVO_CASE);
wpBody.position.set(0, 0.65, 0);
const wpHorn = addMesh(wristPitchServo, new THREE.CylinderGeometry(0.011, 0.011, 0.006, 20), HORN_ALU);
wpHorn.rotation.z = Math.PI / 2;
wpHorn.position.set(0.016, 0.65, 0);

// 4C. Wrist Roll Actuator STS3215 (ID 5)
const wristRollServo = createPart('Wrist Roll Actuator STS3215 (ID 5)', {
  stage: 1,
  explode: [0, 0.78, -0.16],
  category: 'Smart Serial Bus Servo',
  specs: {
    model: 'Feetech STS3215 High-Torque Bus Servo',
    busId: '5 (Wrist Roll / Tool Flange Axis)',
    stallTorque: '30.0 kg·cm @ 12V (2.94 N·m)',
    function: 'Axial tool orientation for precise grasp pose alignment'
  }
});
const wrBody = addMesh(wristRollServo, new THREE.BoxGeometry(0.022, 0.038, 0.038), SERVO_CASE);
wrBody.position.set(0, 0.72, 0);
const wrHorn = addMesh(wristRollServo, new THREE.CylinderGeometry(0.012, 0.012, 0.007, 24), HORN_ALU);
wrHorn.position.set(0, 0.745, 0);

// 4D. Wrist Internal Stator & Gears
const wristInternals = createPart('Wrist Actuators Internal Components', {
  stage: 2,
  explode: [-0.28, 0.66, 0.24],
  category: 'Motor Internals',
  specs: {
    drive: 'Miniature precision reduction gearbox',
    encoderBoard: 'Dual magnetic sensing pickup boards for pitch/roll angles'
  }
});
const wrCoil = addMesh(wristInternals, new THREE.CylinderGeometry(0.007, 0.007, 0.015, 14), COPPER_COIL);
wrCoil.position.set(0, 0.65, 0);
const wrGear = addMesh(wristInternals, new THREE.CylinderGeometry(0.009, 0.009, 0.004, 16), BRASS_PINION);
wrGear.position.set(0, 0.65, 0.01);

// ----------------------------------------------------------------
// 5. PARALLEL GRIPPER MECHANISM (STS3215 / SCS0009 ID: 6)
// ----------------------------------------------------------------
// 5A. Gripper Main Base Housing (shell: true)
const gripperShell = createPart('Gripper Housing & Gearbox Enclosure', {
  shell: true,
  stage: 0,
  explode: [0, 0.90, 0.18],
  category: 'End Effector Tool',
  specs: {
    material: 'PETG-CF Lightweight Housing',
    mechanism: 'Synchronized Dual-Rack and Pinion Parallel Clamping',
    weight: '72 g (complete gripper module)'
  }
});
const grBox = addMesh(gripperShell, new THREE.BoxGeometry(0.046, 0.040, 0.046), SHELL_MAT_CF);
grBox.position.set(0, 0.81, 0);

// 5B. Gripper Drive Servo (ID 6)
const gripperServo = createPart('Gripper Drive Actuator (ID 6)', {
  stage: 1,
  explode: [0, 0.81, -0.15],
  category: 'Smart Serial Bus Servo',
  specs: {
    model: 'Feetech Serial Bus Servo (Grasp Drive)',
    busId: '6 (Parallel Jaw Pinch Axis)',
    currentLimitedGrip: 'Software torque-limited clamp prevents crushing target blocks',
    stroke: '52mm fully open to 0mm closed'
  }
});
const grServoBody = addMesh(gripperServo, new THREE.BoxGeometry(0.020, 0.032, 0.032), SERVO_CASE);
grServoBody.position.set(0, 0.81, 0);

// 5C. Dual Rack and Pinion Metal Transmission Gears
const gripperGears = createPart('Dual Rack & Pinion Mechanism', {
  stage: 2,
  explode: [0, 0.88, 0],
  category: 'Mechanical Transmission',
  specs: {
    type: 'Dual synchronized rack slides with central brass driving pinion',
    material: 'CNC Brass Pinion & Steel Linear Guide Racks',
    gripForce: 'Up to 14.5 N sustained pinch force'
  }
});
const centralPinion = addMesh(gripperGears, new THREE.CylinderGeometry(0.008, 0.008, 0.008, 20), BRASS_PINION);
centralPinion.position.set(0, 0.835, 0);
const rackL = addMesh(gripperGears, new THREE.BoxGeometry(0.006, 0.006, 0.036), STEEL_GEAR);
rackL.position.set(-0.014, 0.835, 0);
const rackR = addMesh(gripperGears, new THREE.BoxGeometry(0.006, 0.006, 0.036), STEEL_GEAR);
rackR.position.set(0.014, 0.835, 0);

// 5D. Left Gripper Jaw Finger (shell: true)
const jawLeft = createPart('Left Gripper Jaw & High-Friction Silicone Pad', {
  shell: true,
  stage: 0,
  explode: [-0.18, 0.94, 0],
  category: 'End Effector Tool',
  specs: {
    material: 'PETG-CF Structural Finger with Shore 40A Silicone Rubber Pad',
    frictionCoeff: 'μ = 1.15 against acrylic/wooden manipulation cubes',
    weight: '14 g'
  }
});
const jLMesh = addMesh(jawLeft, new THREE.BoxGeometry(0.008, 0.055, 0.018), SHELL_MAT_CF);
jLMesh.position.set(-0.022, 0.87, 0);
const padL = addMesh(jawLeft, new THREE.BoxGeometry(0.003, 0.040, 0.016), GRIP_RUBBER);
padL.position.set(-0.017, 0.87, 0);

// 5E. Right Gripper Jaw Finger (shell: true)
const jawRight = createPart('Right Gripper Jaw & High-Friction Silicone Pad', {
  shell: true,
  stage: 0,
  explode: [0.18, 0.94, 0],
  category: 'End Effector Tool',
  specs: {
    material: 'PETG-CF Structural Finger with Shore 40A Silicone Rubber Pad',
    frictionCoeff: 'μ = 1.15 against acrylic/wooden manipulation cubes',
    weight: '14 g'
  }
});
const jRMesh = addMesh(jawRight, new THREE.BoxGeometry(0.008, 0.055, 0.018), SHELL_MAT_CF);
jRMesh.position.set(0.022, 0.87, 0);
const padR = addMesh(jawRight, new THREE.BoxGeometry(0.003, 0.040, 0.016), GRIP_RUBBER);
padR.position.set(0.017, 0.87, 0);

// ----------------------------------------------------------------
// 6. DAISY-CHAIN 3-PIN SERIAL BUS WIRING HARNESS & CONNECTORS (Connections!)
// ----------------------------------------------------------------
// Build realistic color-coded 3-wire ribbon runs between servos:
// Red (+12V VCC), Black (GND), Cyan/Yellow (Half-Duplex TTL UART Serial 1Mbps)

function createWireRibbon(parent, curvePoints, radius = 0.0016) {
  const curve = new THREE.CatmullRomCurve3(curvePoints);
  // Red wire (+12V)
  const redPts = curvePoints.map(p => new THREE.Vector3(p.x - 0.002, p.y, p.z));
  const redGeo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(redPts), 24, radius, 8, false);
  addMesh(parent, redGeo, WIRE_RED);

  // Black wire (GND)
  const blkPts = curvePoints.map(p => new THREE.Vector3(p.x, p.y, p.z));
  const blkGeo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(blkPts), 24, radius, 8, false);
  addMesh(parent, blkGeo, WIRE_BLK);

  // Cyan wire (UART TTL 1Mbps)
  const sigPts = curvePoints.map(p => new THREE.Vector3(p.x + 0.002, p.y, p.z));
  const sigGeo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(sigPts), 24, radius, 8, false);
  addMesh(parent, sigGeo, WIRE_SIG);
}

function createJstConnector(parent, pos, rot = [0, 0, 0]) {
  const conn = addMesh(parent, new THREE.BoxGeometry(0.008, 0.006, 0.012), CONNECTOR_WHT);
  conn.position.copy(pos);
  conn.rotation.set(...rot);
  return conn;
}

// 6A. Base-to-Shoulder Daisy Chain Harness
const harnessBaseToShoulder = createPart('Daisy-Chain Bus: Base to Shoulder (ID 1 -> ID 2)', {
  stage: 2,
  explode: [0.14, 0.12, 0.14],
  category: 'Electrical Wiring Harness',
  specs: {
    protocol: 'Half-Duplex Asynchronous TTL UART (1,000,000 Baud)',
    wires: '3-Pin: Red (+12V Power Rail), Black (Power & Signal GND), Cyan (UART Data TX/RX)',
    connector: 'JST-XH 2.54mm Pitch 3-Pin Locking Female Housing',
    resistance: '< 0.04 Ω loop resistance for stable high-current torque delivery'
  }
});
createWireRibbon(harnessBaseToShoulder, [
  new THREE.Vector3(-0.01, 0.035, -0.01),
  new THREE.Vector3(-0.02, 0.06, 0.01),
  new THREE.Vector3(0.01, 0.10, 0.02),
  new THREE.Vector3(0.0, 0.14, 0.01)
]);
createJstConnector(harnessBaseToShoulder, new THREE.Vector3(-0.01, 0.035, -0.01));
createJstConnector(harnessBaseToShoulder, new THREE.Vector3(0.0, 0.14, 0.01));

// 6B. Shoulder-to-Elbow Daisy Chain Harness
const harnessShoulderToElbow = createPart('Daisy-Chain Bus: Shoulder to Elbow (ID 2 -> ID 3)', {
  stage: 2,
  explode: [0.16, 0.32, 0.16],
  category: 'Electrical Wiring Harness',
  specs: {
    protocol: 'Half-Duplex TTL Serial Bus (Daisy-Chained Passthrough)',
    flexRating: 'High-flex silicone stranded wire (22 AWG Power / 28 AWG Signal)',
    bendCycles: '> 1,000,000 continuous articulation flex cycles'
  }
});
createWireRibbon(harnessShoulderToElbow, [
  new THREE.Vector3(0.0, 0.17, 0.01),
  new THREE.Vector3(0.01, 0.24, 0.02),
  new THREE.Vector3(-0.01, 0.33, 0.02),
  new THREE.Vector3(0.0, 0.40, 0.01)
]);
createJstConnector(harnessShoulderToElbow, new THREE.Vector3(0.0, 0.17, 0.01));
createJstConnector(harnessShoulderToElbow, new THREE.Vector3(0.0, 0.40, 0.01));

// 6C. Elbow-to-Wrist Daisy Chain Harness
const harnessElbowToWrist = createPart('Daisy-Chain Bus: Elbow to Wrist (ID 3 -> ID 4/5)', {
  stage: 2,
  explode: [0.18, 0.56, 0.18],
  category: 'Electrical Wiring Harness',
  specs: {
    protocol: 'Half-Duplex TTL Serial Bus (ID 4 Pitch & ID 5 Roll Multicast)',
    routing: 'Internal channel through carbon fiber forearm tube'
  }
});
createWireRibbon(harnessElbowToWrist, [
  new THREE.Vector3(0.0, 0.44, 0.01),
  new THREE.Vector3(0.01, 0.50, 0.015),
  new THREE.Vector3(-0.01, 0.58, 0.015),
  new THREE.Vector3(0.0, 0.63, 0.01)
]);
createJstConnector(harnessElbowToWrist, new THREE.Vector3(0.0, 0.44, 0.01));
createJstConnector(harnessElbowToWrist, new THREE.Vector3(0.0, 0.63, 0.01));

// 6D. Wrist-to-Gripper Terminal Bus Harness
const harnessWristToGripper = createPart('Daisy-Chain Bus: Wrist to Gripper (ID 5 -> ID 6)', {
  stage: 2,
  explode: [0.16, 0.76, 0.16],
  category: 'Electrical Wiring Harness',
  specs: {
    protocol: 'Terminal Bus Drop to Gripper Actuator',
    termination: 'Built-in 10 kΩ pull-up on TTL UART bus transceiver'
  }
});
createWireRibbon(harnessWristToGripper, [
  new THREE.Vector3(0.0, 0.72, 0.01),
  new THREE.Vector3(0.01, 0.76, 0.015),
  new THREE.Vector3(0.0, 0.80, 0.01)
]);
createJstConnector(harnessWristToGripper, new THREE.Vector3(0.0, 0.72, 0.01));
createJstConnector(harnessWristToGripper, new THREE.Vector3(0.0, 0.80, 0.01));

// ----------------------------------------------------------------
// 7. EXPLODER, LABELS, X-RAY, SECTION & PICKER INTEGRATION
// ----------------------------------------------------------------
const ex = new Exploder(armAssembly);
const xray = new Xray(armAssembly);
const section = new Section(armAssembly, [0, 0, -1]);

const labelsContainer = document.querySelector('.labels');
const labels = new Labels(labelsContainer, camera, canvas);

// Add descriptive leader-line labels to critical components
labels.add(baseShell, 'Base Mount Flange', { anchor: [0, 0.01, 0.11], note: 'Rigid PETG-CF base plinth' });
labels.add(baseYawServo, 'Base Yaw Servo (ID 1)', { anchor: [0.02, 0.05, 0], note: 'Feetech STS3215 · 30 kg·cm' });
labels.add(baseBearing, '6806ZZ Turntable Bearing', { anchor: [0.04, 0.035, 0], note: 'Chrome steel thrust balls' });
labels.add(harnessBaseToShoulder, '3-Pin Daisy-Chain Bus', { anchor: [0.01, 0.10, 0.02], note: '12V VCC · GND · TTL UART 1Mbps' });
labels.add(shoulderServo, 'Shoulder Actuator (ID 2)', { anchor: [0.02, 0.16, 0], note: 'Dual-bearing high torque pitch' });
labels.add(shoulderShellLeft, 'Shoulder PETG Shell', { anchor: [-0.03, 0.16, 0], note: 'Lightweight clamshell casing' });
labels.add(elbowServo, 'Elbow Actuator (ID 3)', { anchor: [0.02, 0.42, 0], note: 'STS3215 1:345 metal reduction' });
labels.add(elbowInternals, 'Steel Gearbox & Stator', { anchor: [0, 0.42, -0.01], note: 'Hardened spur gears & coils' });
labels.add(wristPitchServo, 'Wrist Gimbal (ID 4/5)', { anchor: [0.02, 0.65, 0], note: 'Pitch & roll tool orientation' });
labels.add(gripperServo, 'Gripper Drive (ID 6)', { anchor: [0, 0.81, -0.02], note: 'Parallel jaw rack & pinion' });
labels.add(jawLeft, 'Clamping Jaws', { anchor: [-0.02, 0.88, 0], note: 'Silicone high-friction grip' });

// Interactive Part Picker & Inspector Link
const parts = armAssembly.children;
const inspectorCard = document.getElementById('inspector-card');
const inspectorTitle = document.getElementById('inspector-title');
const inspectorCategory = document.getElementById('inspector-category');
const inspectorSpecs = document.getElementById('inspector-specs');
const inspectorDesc = document.getElementById('inspector-desc');

function showPartDetails(part) {
  if (!part) return;
  const d = part.userData;
  inspectorTitle.textContent = part.name || 'Component';
  inspectorCategory.textContent = d.category || 'Manipulator Subassembly';
  inspectorDesc.textContent = d.specs?.description || 'Engineered modular component of the SO-100 open-source 5-DOF robotic manipulator.';

  inspectorSpecs.innerHTML = '';
  if (d.specs) {
    for (const [key, val] of Object.entries(d.specs)) {
      if (key === 'description') continue;
      const box = document.createElement('div');
      box.className = 'spec-box';
      const k = document.createElement('div');
      k.className = 'spec-key';
      k.textContent = key.replace(/([A-Z])/g, ' $1').toLowerCase();
      const v = document.createElement('div');
      v.className = 'spec-val';
      v.textContent = String(val);
      box.appendChild(k);
      box.appendChild(v);
      inspectorSpecs.appendChild(box);
    }
  }

  inspectorCard.classList.remove('collapsed');
}

new Picker(canvas, camera, parts, {
  onHover: (p) => {
    labels.items.forEach((it) => {
      let o = it.object;
      while (o && o.parent !== armAssembly) o = o.parent;
      it.focus = !!p && o === p;
    });
  },
  onPick: (p) => {
    if (p) showPartDetails(p);
  }
});

// Close inspector button
document.getElementById('btn-close-inspector')?.addEventListener('click', () => {
  inspectorCard.classList.add('collapsed');
});

// ---------------------------------------------------------------- Controls & Sliders
const exSlider = document.getElementById('explode');
const exVal = document.getElementById('explode-val');
const cutSlider = document.getElementById('cut');
const cutVal = document.getElementById('cut-val');
const xBtn = document.getElementById('xray');
const autoRotBtn = document.getElementById('autorotate');
const resetCamBtn = document.getElementById('reset-camera');

let explodeTarget = +(params.get('explode') ?? 0);
let explodeNow = explodeTarget;
let autoRotate = true;

exSlider.value = explodeTarget;
if (exVal) exVal.textContent = `${Math.round(explodeTarget * 100)}%`;

exSlider.addEventListener('input', () => {
  explodeTarget = +exSlider.value;
  if (exVal) exVal.textContent = `${Math.round(explodeTarget * 100)}%`;
});

const setCut = (v) => {
  section.set(v > 0.001 ? 0.45 * (1 - v) : null);
  if (cutVal) cutVal.textContent = `${Math.round(v * 100)}%`;
};
cutSlider.value = params.get('cut') ?? 0;
setCut(+cutSlider.value);
cutSlider.addEventListener('input', () => setCut(+cutSlider.value));

const setX = (on) => {
  xray.set(on, 0.16);
  xBtn.setAttribute('aria-pressed', String(on));
  xBtn.classList.toggle('active', on);
};
setX(params.get('xray') === '1');
xBtn.addEventListener('click', () => setX(!xray.on));

autoRotBtn?.addEventListener('click', () => {
  autoRotate = !autoRotate;
  autoRotBtn.classList.toggle('active', autoRotate);
});

// Camera View Presets
const viewDir = new THREE.Vector3(0.55, 0.42, 0.9).normalize();
const centerTarget = new THREE.Vector3(0, 0.42, 0);

function resetCameraView(preset = 'iso') {
  controls.target.copy(centerTarget);
  if (preset === 'iso') {
    camera.position.set(0.9, 0.85, 1.2);
  } else if (preset === 'base') {
    controls.target.set(0, 0.08, 0);
    camera.position.set(0.35, 0.22, 0.45);
  } else if (preset === 'shoulder') {
    controls.target.set(0, 0.25, 0);
    camera.position.set(0.45, 0.35, 0.55);
  } else if (preset === 'wrist') {
    controls.target.set(0, 0.65, 0);
    camera.position.set(0.35, 0.72, 0.42);
  } else if (preset === 'gripper') {
    controls.target.set(0, 0.85, 0);
    camera.position.set(0.30, 0.92, 0.35);
  }
  controls.update();
}

resetCamBtn?.addEventListener('click', () => resetCameraView('iso'));

document.querySelectorAll('.preset-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.preset-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    resetCameraView(btn.dataset.preset);
  });
});

resetCameraView('iso');

addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight, false);
});

// Render Loop
let lastTime = performance.now();

function animate(now) {
  const dt = Math.min(0.05, (now - lastTime) / 1000);
  lastTime = now;

  // Smooth explode interpolation
  explodeNow += (explodeTarget - explodeNow) * (REDUCED ? 1 : 1 - Math.pow(0.001, dt));
  ex.set(explodeNow);

  // Auto rotation if user is idle
  if (autoRotate && !controls.state == -1) {
    armAssembly.rotation.y += dt * 0.18 * (1.0 - explodeNow * 0.5);
  }

  controls.update();
  renderer.render(scene, camera);
  labels.update();

  requestAnimationFrame(animate);
}

requestAnimationFrame(animate);
