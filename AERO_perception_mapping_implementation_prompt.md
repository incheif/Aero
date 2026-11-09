# AERO: Perception, Mapping, Localisation & Autonomous Manipulation Foundation

## Implementation prompt for Claude Code / Gemini

Repository: https://github.com/incheif/Aero

Implement the next engineering phase of AERO: **camera/sensor-driven perception, tabletop mapping, robot localisation, collision-aware motion planning, and closed-loop manipulation**.

**Do not implement a world model, JEPA, V-JEPA, latent dynamics, learned action prediction, or model-predictive control in this phase.** Those will be designed and added later on top of the interfaces created here.

The goal is to make AERO operate from observations and a maintained estimate of the workcell, rather than reading exact object coordinates from the physics engine and passing hard-coded metric targets directly into the controller.

---

# 1. First: audit the actual repository

Before changing code:

1. Inspect the complete repository and identify the current active branch and working tree state.
2. Read the existing implementation and tests, especially:
   - `simulation/camera.py`
   - `simulation/physics_engine.py`
   - `simulation/mujoco_engine.py` (if present and active)
   - `simulation/kinematics.py`
   - `vla/semantic_mapper.py`
   - `vla/frontier_explorer.py`
   - `vla/trajectory_controller.py`
   - `vla/google_vla_agent.py`
   - `dashboard/server.py`
   - `dashboard/static/`
   - `ros2_ws/`
   - existing tests and dependency files
3. Trace the complete runtime data path:
   - camera frame generation
   - object detection/localisation
   - semantic map updates
   - instruction grounding
   - target selection
   - IK and trajectory generation
   - physics execution
   - completion verification
4. Identify every location where runtime logic reads simulator ground truth, including exact block positions, velocities, grasp state, support relationships, and collision state.
5. Identify every hard-coded object/target coordinate used by task planning or action generation.
6. Determine which simulator backend is actually active by default. Preserve compatibility with the existing backend(s).

Do not assume the README accurately describes every implementation detail. Verify behavior in code.

Before implementation, write a concise audit to `docs/perception_mapping_audit.md` covering:
- current architecture and data flow
- ground-truth dependencies
- existing reusable modules
- concrete gaps
- proposed minimal implementation plan

Then implement the work in small, testable stages. Do not stop after writing the audit.

---

# 2. Main engineering objective

Build this runtime pipeline:

```text
Camera / Sensor Interface
        |
        v
Image Acquisition + Calibration
        |
        v
Object Detection / Segmentation
        |
        v
3D Object Localisation + Tracking
        |
        v
Robot State Estimation + Coordinate Transforms
        |
        v
Persistent Tabletop Scene Map
        |
        v
Semantic Target Grounding
        |
        v
Collision-Aware Motion Planning
        |
        v
Existing IK + Trajectory Controller
        |
        v
Simulation / Robot Execution
        |
        v
Camera Re-observation + Verification
        |
        +------> Update map, detect errors, recover/replan
```

The perception and mapping system must be the source of object state used by runtime planning. The simulator may still use its internal state to advance physics and render the scene, but the planner must not query that hidden state to obtain exact object poses.

**Important distinction:** this phase creates a reliable state-estimation and execution foundation. It does not create a learned or predictive world model.

---

# 3. Non-negotiable architecture rules

## 3.1 Separate ground truth from estimated state

Create a strict boundary between:

- **Simulator ground truth:** internal physics state used by the physics engine and test evaluation.
- **Sensor observations:** RGB/depth images and robot joint/gripper readings exposed through sensor interfaces.
- **Estimated state:** object poses, robot pose, map geometry, confidence, and semantic relations inferred from observations.
- **Planner state:** only the estimated state and known static workcell configuration.

Do not pass the physics engine's full snapshot into the perception or planning stack.

The runtime planner must not access:
- exact block positions from `get_snapshot()`
- hidden block IDs or semantic labels unavailable to the sensor
- exact grasp state from the simulator
- exact support relationships from physics internals
- hidden collision contacts or future simulator state

Ground truth may be used in tests, simulation rendering, synthetic sensor generation, and offline evaluation. Keep those paths explicit and separate.

Add tests that fail if runtime perception/planning code directly reads hidden simulator object state.

## 3.2 Preserve the existing project

- Keep the current Three.js studio and dashboard functional.
- Preserve the current arm model, IK/FK, physics behavior, task types, execution logs, and deterministic offline mode.
- Reuse and refactor existing modules where sensible.
- Do not create a second planner, mapper, or controller that duplicates existing functionality.
- Avoid broad rewrites and unrelated UI redesigns.
- Do not change working physics/collision behavior unless a test demonstrates a specific defect.
- Do not silently remove existing commands or task capabilities.

## 3.3 Keep the system lightweight

Do not introduce ROS 2, Isaac Sim, Isaac Lab, MuJoCo, Gazebo, OpenCV, PyTorch, a cloud vision API, or a large planning framework as a mandatory dependency just to complete this phase.

First inspect installed dependencies and current project conventions. Prefer the existing Python, NumPy, and Pillow stack for the initial simulation implementation.

If an optional integration is useful, isolate it behind an adapter and keep the default installation runnable without it.

## 3.4 Do not fake sensor-based autonomy

A synthetic camera is acceptable for simulation, but it must behave like a sensor boundary:
- the perception module receives image/depth data, not simulator object dictionaries
- detections are derived from the image/depth data
- localisation is derived from camera calibration and sensor measurements
- uncertainty and missed detections are represented
- the runtime planner uses estimated poses

Do not render a frame from a simulator snapshot and then pass the same snapshot's exact object coordinates to the mapper while claiming camera-based perception.

---

# 4. Workstream A: Camera and sensor abstraction

The current `simulation/camera.py` creates a top-down RGB image using the simulator snapshot and provides pixel/world conversion helpers. Refactor this into a sensor interface without breaking current callers unnecessarily.

Implement:

1. A camera observation data structure containing:
   - timestamp or simulation tick
   - RGB image
   - optional depth image
   - camera intrinsics
   - camera-to-robot-base transform
   - image dimensions
   - sensor validity/status
2. A camera interface/protocol with methods equivalent to:
   - `capture()`
   - `get_calibration()`
   - `is_available()`
3. A simulation camera adapter that produces RGB and, where practical, synthetic depth from the rendered scene.
4. A clear adapter boundary so a physical RGB-D camera can be added later without changing perception or planning code.
5. A robot-state sensor interface for joint positions and gripper state. In simulation this may read the robot's exposed joint telemetry, but must not expose object ground truth.
6. Camera calibration utilities:
   - intrinsics
   - extrinsics
   - camera-to-base and base-to-camera transforms
   - pixel/depth to 3D point conversion
   - consistent metres/radians conventions

For the fixed tabletop setup, a calibrated overhead camera is sufficient for the first implementation. Do not build a general multi-camera SLAM system now.

If the existing renderer is orthographic, implement and document the correct orthographic projection model. Do not apply pinhole-camera equations to an orthographic renderer. Keep the calibration interface general enough to support a perspective RGB-D camera later.

Add calibration validation tests using known simulated landmarks.

---

# 5. Workstream B: Image-based object perception

Implement a perception module that accepts only camera observations.

Initial simulation scope:
- detect the coloured tabletop blocks already used by AERO
- estimate each visible block's image-space bounding box or mask
- estimate its colour/class where distinguishable
- estimate its 3D position from depth or the calibrated tabletop plane
- estimate visible dimensions where feasible
- return a confidence score and observation timestamp
- handle occlusion, partial visibility, and missing detections

Do not use block IDs or exact coordinates from the physics snapshot as detection inputs.

A simple colour segmentation approach is acceptable for the first version if it is robust under the current renderer. Keep the detector modular so a learned detector can replace it later.

Use explicit data structures, for example:

```python
@dataclass
class ObjectDetection:
    class_name: str
    color: str | None
    bbox_xyxy: tuple[int, int, int, int]
    centroid_uv: tuple[float, float]
    position_camera: tuple[float, float, float] | None
    position_base: tuple[float, float, float] | None
    dimensions: tuple[float, float, float] | None
    confidence: float
    timestamp: float
```

Adapt the schema to the codebase rather than copying it blindly.

Important:
- A 2D RGB image alone does not provide reliable metric height for arbitrary objects. Use depth or a known tabletop plane and clearly mark assumptions.
- If depth is unavailable, represent uncertain height and use the known tabletop surface only when the object is classified as resting on it.
- Do not invent a precise 3D pose when the sensor data does not support one.

---

# 6. Workstream C: 3D localisation and coordinate frames

Create a small, explicit coordinate-frame system for the tabletop arm.

At minimum define:
- `camera_frame`
- `robot_base_frame`
- `table_frame`
- `tcp_frame`

Requirements:

1. Use homogeneous transforms or an equally clear, tested representation.
2. Centralise frame conversion logic. Do not scatter coordinate conversion formulas across the detector, mapper, and controller.
3. Convert camera observations into robot-base coordinates using calibrated extrinsics.
4. Estimate robot TCP pose from joint-state observations and existing forward kinematics.
5. Keep units consistent: metres and radians internally.
6. Add tests for:
   - identity transform
   - translation
   - rotation
   - camera-to-base conversion
   - pixel/depth-to-3D conversion
   - FK-derived TCP pose
7. Reject or flag observations outside calibrated workspace bounds rather than silently clamping them into plausible positions.

The current robot is fixed-base and the camera/workcell can initially be treated as static. This is **tabletop localisation**, not full mobile-robot SLAM. Do not add loop closure, pose-graph optimisation, IMU fusion, or mobile-base localisation in this phase.

---

# 7. Workstream D: Persistent tabletop map and object tracking

Refactor `vla/semantic_mapper.py` so the map is updated from perception estimates, not directly from simulator snapshots.

Keep and improve the useful existing concepts:
- semantic entities
- bounding volumes
- object state
- support relationships
- grasp affordance
- target zones
- spatial relations
- confidence

Add:

1. Stable tracked object IDs generated by the perception/tracking layer. These IDs must not be copied from simulator IDs.
2. Association of detections across frames using position, appearance/colour, and bounding-box overlap as appropriate.
3. Track state:
   - current estimated pose
   - previous pose
   - estimated velocity if enough observations exist
   - last-seen timestamp/tick
   - confidence
   - visible/occluded/lost status
4. A policy for missed detections and stale tracks. Do not instantly delete an object after one missed frame.
5. Map update methods that accept detections and robot-state estimates, not physics snapshots.
6. Spatial relations computed from estimated geometry and confidence.
7. Support/stacking relations inferred from geometry and observations. Do not copy `supported_by` or `stacked_layer` from the physics engine.
8. A representation of unknown/unobserved areas. Do not mark the whole table as explored merely because a scan waypoint was visited.
9. Static landmarks for the table and target pad, with calibration/configuration as the source of their locations.

For this fixed tabletop arm, maintain a persistent workcell scene map. A full voxel SLAM implementation is not required. Use a simple 2.5D or 3D object/obstacle representation appropriate to the current task.

Keep a clear distinction between:
- observed
- inferred
- configured/static
- unknown

Every estimated entity should expose confidence and observation freshness.

---

# 8. Workstream E: Real exploration and coverage

Refactor `vla/frontier_explorer.py`.

The current explorer uses a predefined list of scan waypoints and increases coverage when a waypoint is advanced. Replace this bookkeeping with map-driven exploration.

Requirements:

1. Use the map's observed/unknown/occluded regions to identify where more observations are useful.
2. Generate safe camera viewpoints or arm-mounted-camera viewpoints within the current workcell limits.
3. Prefer viewpoints that reveal currently occluded objects or unknown table regions.
4. Check each viewpoint for reachability and collision safety before execution.
5. Update coverage only after a valid observation has been captured and integrated.
6. Track exploration progress based on actual observed area or cells, not waypoint count.
7. Keep the current four-quadrant exploration behavior as a fallback/demo mode if useful, but do not report a quadrant as explored simply because the arm passed through its waypoint.

If the camera is fixed overhead in the initial implementation, exploration may be limited to image-based visibility and object occlusion analysis. Do not pretend a fixed camera can reveal hidden surfaces by moving the arm.

---

# 9. Workstream F: Semantic target grounding

The instruction interpreter/VLA layer should express *what object or region is needed*, not supply hard-coded object coordinates.

Refactor the interface between `GoogleVLAAgent`, `SemanticSpatialMapper`, and the motion controller.

Examples:
- "Pick up the red block" -> resolve a tracked object whose estimated colour/class is red.
- "Place it on the target pad" -> resolve a configured target landmark and placement affordance.
- "Move the blue block next to the red block" -> resolve both tracked objects and derive a target relation from their estimated poses.
- "Build a pyramid" -> create a task goal/assembly specification; resolve actual object poses and placement targets from the current map at execution time.

Requirements:
1. The planner may use configured workcell geometry and named static landmarks.
2. Dynamic object coordinates must come from the current estimated map.
3. Do not use hidden simulator IDs to resolve objects.
4. If two objects match a description, ask for clarification or apply a documented deterministic disambiguation policy.
5. If an object is not visible or confidence is too low, request another observation or explore before attempting a grasp.
6. Separate task intent from geometric target resolution.
7. Keep existing natural-language commands working where possible.

Remove or isolate hard-coded coordinates that represent dynamic object positions. Static robot home poses, calibrated target landmarks, and safe transit parameters may remain configured values, but centralise them and document why they are static.

---

# 10. Workstream G: Collision-aware motion planning

AERO already has analytical IK/FK and a trajectory controller. Preserve these and add a planning layer between semantic target resolution and trajectory execution.

The intended flow is:

```text
Semantic target
    -> candidate end-effector pose
    -> IK feasibility check
    -> collision-aware path generation
    -> trajectory validation
    -> existing trajectory controller
```

Requirements:

1. Use the estimated scene map to construct collision geometry for blocks, table, and static workcell obstacles.
2. Use the current robot configuration and FK to establish the start state.
3. Use existing IK to test candidate target poses and joint limits.
4. Plan safe approach, grasp, lift, transit, place, and retreat segments.
5. Avoid straight-line Cartesian paths when the map indicates an obstacle or collision risk.
6. Include clearance margins around objects and the arm/gripper.
7. Validate the full path before execution.
8. If no safe path exists, fail safely and report why. Do not silently fall back to an unsafe direct path.
9. Preserve the current controller as the low-level execution component.
10. Keep planning modular so a sampling-based planner such as RRT-Connect/OMPL can be integrated later if required.

For the first version, a lightweight collision-aware planner using the existing robot model is acceptable. Do not add OMPL as a mandatory dependency unless the current code cannot reasonably support the required planning.

Important: the scene map is an estimate and can be incomplete. Use conservative margins around uncertain obstacles. The planner must not interpret unknown space as guaranteed free space.

---

# 11. Workstream H: Closed-loop manipulation and verification

Convert manipulation from a one-shot plan execution into an observe-act-verify cycle.

For each manipulation step:

1. Capture a fresh observation.
2. Update object tracks and the scene map.
3. Resolve the target from the latest map.
4. Check confidence, reachability, and collision safety.
5. Execute only the next safe segment or manipulation primitive.
6. Capture another observation.
7. Verify whether the expected state change occurred.
8. Continue, retry within a bounded limit, request another view, or replan.

Examples of conditions to verify:
- target object moved to the expected grasp region
- object is being carried, based on visual evidence and gripper feedback
- object was released
- object is within placement tolerance
- stack remains stable after release
- unexpected object displacement or collision occurred

Do not infer a successful grasp solely because the simulator's internal `grasped_block_id` is set. Use visual evidence and available gripper/joint feedback.

Add bounded recovery behavior:
- failed detection -> request another observation
- low confidence -> inspect from another viewpoint if possible
- unreachable target -> report and select another feasible target if task semantics allow
- failed grasp -> retreat safely, reacquire, and retry a limited number of times
- failed placement -> inspect the scene, update the map, and replan
- unexpected movement -> stop, observe, and rebuild the local plan

Never retry indefinitely. Make retry counts and tolerances configurable.

---

# 12. Workstream I: Dashboard and observability

Update the existing dashboard and telemetry without redesigning the UI.

Expose enough information to understand whether the system is actually perception-driven:

- latest camera frame
- optional depth visualisation
- detected objects and confidence
- tracked object IDs
- estimated object positions
- coordinate frame/calibration status
- robot joint state and estimated TCP pose
- map entities and spatial relations
- observed/unknown/occluded regions
- selected semantic target
- planned path and collision checks
- execution phase
- verification result
- recovery/replanning reason

Clearly label estimated values as estimates. If ground truth is shown for debugging, label it as simulator ground truth and keep it out of the planner-facing state.

Do not claim sensor accuracy or SLAM capability based only on synthetic data.

---

# 13. Workstream J: Tests and evaluation

Add focused tests as each workstream is implemented.

At minimum cover:

## Sensor and perception
- camera capture returns valid observation data
- image detector identifies the current block colours
- detector does not receive simulator object coordinates
- occluded/missing objects reduce visibility/confidence
- depth/plane localisation returns expected metric coordinates

## Coordinate frames
- transforms and inverse transforms are correct
- calibration maps known landmarks within tolerance
- invalid/out-of-bounds points are rejected or flagged

## Mapping and tracking
- new detections create tracks
- repeated detections preserve track identity
- small motion updates the same track
- one missed frame does not delete an object
- stale tracks eventually become lost
- support relations are inferred from geometry, not simulator metadata
- unknown regions remain unknown until observed

## Planning
- reachable free-space target generates a valid path
- obstacle between start and target causes a detour or safe rejection
- unreachable IK target is rejected
- path respects workspace and joint limits
- unknown/uncertain obstacles trigger conservative behavior

## Closed-loop execution
- a successful grasp is verified from observations/feedback
- a failed grasp does not report success
- a failed placement triggers bounded recovery
- map is updated after an object moves
- a new command does not silently reset the entire workcell unless reset was explicitly requested

## Regression
- existing pyramid, tower, train, gesture, and visual-servoing tests continue to pass where applicable
- existing dashboard starts
- existing deterministic offline mode remains functional
- both supported simulation backends remain compatible if they are still supported

Add an integration scenario that randomises the starting positions of the coloured blocks within safe workcell bounds. AERO must detect and localise them from camera observations, build/update its map, and execute a simple manipulation task without reading their true positions from the simulator.

Use simulator ground truth only after the episode to calculate evaluation error, never as runtime planner input.

Report measurable metrics:
- object detection precision/recall where ground truth is available for evaluation
- localisation error in metres
- tracking continuity / identity switches
- grasp success rate
- placement error
- collision count
- task completion rate
- recovery attempts
- average time per manipulation

---

# 14. Suggested module boundaries

Do not follow this layout blindly. First inspect the repository and fit the design to its conventions. Prefer extending existing modules where practical.

A possible clean separation:

```text
simulation/
    camera.py                 # renderer and simulation sensor adapter
    sensor_interfaces.py      # camera and robot-state contracts
    coordinate_frames.py      # calibration and transforms

perception/
    detector.py               # image-based object detection
    localisation.py            # 2D/depth/plane to 3D estimates
    tracker.py                 # stable IDs and temporal tracking
    models.py                  # observation and detection data types

vla/
    semantic_mapper.py         # persistent estimated scene map
    frontier_explorer.py      # map-driven observation planning
    target_grounding.py       # semantic descriptions to tracked entities
    motion_planner.py         # collision-aware path generation
    trajectory_controller.py  # retain/refine existing low-level controller
    execution_monitor.py      # observe-act-verify and bounded recovery
```

Keep the data contracts explicit. Avoid circular imports and avoid making the dashboard responsible for robotics logic.

---

# 15. Implementation sequence

Implement in this order. Keep each stage runnable and tested before moving on.

### Stage 0 — Audit and baseline
- Complete `docs/perception_mapping_audit.md`.
- Run the existing test suite and record the baseline.
- Document the active simulator backend and current runtime data flow.
- Identify current failures separately from regressions.

### Stage 1 — Sensor boundary and calibration
- Introduce camera observation and robot-state sensor interfaces.
- Refactor the simulation camera to provide sensor observations.
- Add calibration and coordinate-frame utilities.
- Preserve current dashboard image output.

### Stage 2 — Image-based perception and localisation
- Detect coloured blocks from RGB/depth observations.
- Estimate their positions in robot-base coordinates.
- Add confidence and uncertainty.
- Add perception and calibration tests.

### Stage 3 — Estimated scene map and tracking
- Refactor `SemanticSpatialMapper` to consume detections.
- Add stable track IDs, stale-track handling, and estimated relations.
- Remove simulator-ground-truth dependencies from planner-facing mapping.
- Add map and tracking tests.

### Stage 4 — Exploration and semantic target grounding
- Make exploration depend on actual map coverage/unknown regions.
- Resolve object descriptions to tracked objects.
- Replace dynamic hard-coded object coordinates with map-derived targets.
- Add target ambiguity and low-confidence handling.

### Stage 5 — Collision-aware planning
- Add a planning layer before the existing trajectory controller.
- Use map-derived obstacles, IK feasibility, and conservative clearance.
- Validate paths and add obstacle tests.

### Stage 6 — Closed-loop execution
- Add observe-act-verify behavior.
- Add bounded recovery for grasp and placement failures.
- Ensure state persists across commands unless reset is requested.

### Stage 7 — Dashboard and integration evaluation
- Expose perception, map, planning, and verification telemetry.
- Run randomized object-position scenarios.
- Run regression tests and document results.

At the end of each stage:
- run relevant tests
- report files changed
- report what works
- report known limitations
- do not claim completion for untested behavior

---

# 16. Explicitly out of scope for this phase

Do not implement any of the following yet:

- JEPA, V-JEPA, VL-JEPA, or any other learned latent world model
- action-conditioned latent dynamics
- future-state prediction networks
- learned physics or particle-voxel dynamics
- MPC or learned predictive control
- training datasets for a world model
- world-model training or inference pipelines
- VLA architecture replacement
- multi-agent orchestration
- full mobile robot SLAM
- autonomous navigation for drones/cars
- mandatory physical robot hardware integration

However, keep clean interfaces so a future world model can consume:

```text
estimated scene state
robot state
task/action representation
observed state transitions
execution outcomes
```

Do not build placeholder neural networks or speculative abstractions for that future work.

---

# 17. Acceptance criteria

This phase is complete only when all of the following are true:

1. AERO can obtain camera observations through a sensor interface.
2. The perception stack detects blocks from image/depth data.
3. 3D positions are estimated using documented calibration and coordinate transforms.
4. The semantic map is built and updated from estimated observations.
5. Dynamic object IDs and positions are not copied from simulator ground truth.
6. The robot TCP estimate is derived from joint-state observations and FK.
7. Natural-language object references resolve to tracked map entities.
8. Dynamic object targets are derived from the current map, not hard-coded coordinates.
9. The motion planner checks IK, workspace limits, and collisions before execution.
10. Manipulation is verified through fresh observations and feedback.
11. Failed grasps/placements trigger bounded recovery rather than false success.
12. Randomised block positions can be handled without exposing ground-truth positions to the planner.
13. Existing AERO functionality and dashboard remain operational.
14. Tests demonstrate the ground-truth/perception boundary.
15. No world model or JEPA implementation has been added.

---

# 18. How to work

Work as a senior robotics/software engineer making a controlled change to an existing project.

- Read before editing.
- Reuse before replacing.
- Make small changes with clear ownership.
- Keep interfaces typed and documented.
- Do not create large abstractions without an immediate use.
- Do not add dependencies without justification.
- Do not rewrite the whole application.
- Do not change physics parameters to hide perception or planning bugs.
- Do not hard-code current block positions as a substitute for perception.
- Do not mark tests as passed unless they were actually run.
- If a requested feature cannot be implemented reliably with the current renderer/sensor model, explain the limitation and implement the honest subset.
- If a design choice is uncertain, inspect the code and choose the simplest compatible solution.

**Start with Stage 0, then implement the stages in sequence. Do not stop after producing a plan.** At the end, provide a concise implementation report with architecture changes, files changed, tests run and results, remaining limitations, and the exact commands needed to run the updated AERO system.
