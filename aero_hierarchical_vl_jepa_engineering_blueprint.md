# AERO: Two-Level Hierarchical VL-JEPA Engineering Design & Implementation Blueprint

**Repository:** `https://github.com/incheif/Aero`  
**Target Agent / Executor:** Antigravity / AI Coding Agent  
**Objective:** Evolve AERO (Agentic Execution for Robotic Operations) from a single-level deterministic instruction-matching robot arm stack into a closed-loop, Two-Level Hierarchical Vision-Language JEPA (H-VL-JEPA) predictive world-model system capable of multi-step manipulation (such as building a 5-block tower or a 3-block pyramid), consequence prediction, stability verification, and autonomous replanning.

---

## 1. Executive Summary and Recommended Approach

**AERO** currently operates as a modular, deterministic robotic manipulation platform with a 5-DOF articulated arm (`simulation/kinematics.py`), a 60 Hz Python/Rapier-aligned tabletop physics engine (`simulation/physics_engine.py`), a FastAPI/WebSocket telemetry dashboard (`dashboard/server.py`), and a ROS 2 node graph (`ros2_ws/src/vla_robot_arm/`). Despite file names like `vla/google_vla_agent.py`, AERO does **not** currently execute an end-to-end neural Vision-Language-Action (VLA) policy or call cloud LLM APIs. Instead, it uses an offline, zero-dependency char-3-gram + color-boosted embedding matcher (`vla/instruction_interpreter.py`) coupled with a deterministic state machine (`vla/google_vla_agent.py`) and an analytical geometric Inverse Kinematics (IK) solver (`simulation/kinematics.py`).

While this deterministic architecture solved the severe latency, false-positive completion hallucinations, and out-of-distribution (OOD) stalls seen with end-to-end generative VLAs, it has three core limitations for complex, multi-step tasks like building a 5-block tower or a 3-block pyramid:
1. **No Consequence Prediction:** The agent cannot simulate whether placing a block at an offset will cause a multi-block stack to topple before releasing the gripper.
2. **Hardcoded Phase Sequences:** `GoogleVLAAgent` assumes every task is a single-object cycle (`IDLE` $\rightarrow$ `EXPLORE` $\rightarrow$ `PICK` $\rightarrow$ `MOVE` $\rightarrow$ `DROP` $\rightarrow$ `COMPLETE`) or relies on regex text splitting (`then`, `and then`) in `InstructionInterpreter.parse_sequence()`.
3. **Ground-Truth Cheat in Simulation Perception:** `simulation/camera.py` projects ground-truth 3D physics object coordinates through a pinhole camera matrix to synthesize 2D bounding boxes rather than extracting dense visual embeddings from rendered RGB frames.

### Recommended Architectural Evolution
Do **not** replace AERO's fast, deterministic 5-DOF analytical IK solver (`simulation/kinematics.py`) or its 60 Hz P-controller (`vla/trajectory_controller.py`) with a black-box neural network. As Yann LeCun and the **H-JEPA** (*End-to-End Learning of Hierarchical World Models for Visual Planning*, Zhang et al., AMI Labs/NYU/INRIA, 2026) and **V-JEPA 2-AC** (Meta FAIR, 2025) papers demonstrate, low-level joint kinematics for a rigid robot arm are already governed by exact closed-form equations.

Instead, integrate a **Two-Level Hierarchical Vision-Language JEPA (H-VL-JEPA)** architecture above AERO's existing kinematic and physics execution layer:
* **Level 2 (High-Level Semantic & Task Subgoal Planner):** Uses a local structured VLM/LLM (`Qwen2.5-VL-3B-Instruct` or offline macro-decomposer) for multi-step task decomposition (e.g., breaking *"build a tower of five blocks"* into 5 ordered placement subgoals) combined with a **VL-JEPA / Level-2 Abstract World Model** (`H-JEPA L2` / `V-JEPA 2` encoder) that predicts coarse subgoal state embeddings $z^{(2)}_k$ at the **subtask boundary timescale** (1 step = 1 completed pick-and-place phase) and checks structural feasibility.
* **Level 1 (Low-Level Spatial & Action-Conditioned World Model):** Uses a frozen visual representation backbone (`facebook/dinov3-vits16plus` or `vjepa2_vit_large`) paired with a lightweight **Action-Conditioned Latent Predictor** (`LeWorldModel` / `V-JEPA 2-AC` style) operating over 3 to 5 macro-waypoint steps. Given the current workspace observation embedding $z^{(1)}_t$ and candidate 3D pick/place waypoint deltas $a_{t:t+H}$ sampled via the **Cross-Entropy Method (CEM)**, Level 1 predicts the resulting latent state $\hat{z}^{(1)}_{t+H}$, evaluates distance to the active Level 2 subgoal, and hands the winning collision-free 3D waypoint to AERO's existing `TrajectoryController` and analytical IK solver.

---

## 2. Audit of AERO's Existing Architecture

An inspection of the `incheif/Aero` repository reveals the exact implementation across `simulation/`, `vla/`, `dashboard/`, and `ros2_ws/`:

### 2.1 How Natural-Language Instructions Are Currently Interpreted
* **`vla/instruction_interpreter.py` (`InstructionInterpreter`):**
  * **Task Decomposition:** `parse_sequence(prompt)` splits multi-step instructions strictly using regex delimiters (`\bthen\b`, `\band then\b`, `\bafter that\b`, `;`) and clause patterns where a second action verb follows a color token. A prompt like *"build a tower of five blocks"* has only one verb and no `"then"` delimiter, so `parse_sequence` treats it as a single atomic instruction rather than decomposing it into 5 sequential placements.
  * **Offline Embedding Matcher:** Computes a 212-dimensional vector (`_DIM = 212`) consisting of 192 hashed character 3-gram bins (`_NGRAM_DIM = 192`) plus 20 explicit boost dimensions (`_BOOST_DIM = 20`) for 6 colors (`red`, `green`, `blue`, `yellow`, `purple`, `cyan`), 5 actions (`pick`, `place`, `move`, `push`, `stack`), and spatial prepositions. It computes cosine similarity against a pre-built `_CATALOGUE` of canonical phrases and returns a `ParsedSubtask(action, target_color, dest_color, spatial_rel, raw_text, confidence)`.

### 2.2 Perception, Spatial Representation, Kinematics, Motion Planning, and Simulation
* **`simulation/camera.py` (`OverheadCamera`, `GripperCamera`):**
  * Builds pinhole intrinsic ($K$) and extrinsic ($R, t$) matrices (`IMAGE_WIDTH = 640`, `IMAGE_HEIGHT = 480`, `CAMERA_FOV_DEG = 60.0`).
  * **Critical Finding:** `capture(objects, end_effector_pos)` does not render pixel arrays or run a neural vision model. It projects ground-truth 3D positions from `PhysicsEngine.objects` into 2D pixel coordinates `(u, v)`, adds Gaussian noise (`PIXEL_NOISE_STD = 1.5`), checks line-of-sight occlusion spheres, and returns synthetic `Detection` dataclasses (`label`, `color`, `bbox`, `confidence`, `depth`).
* **`vla/semantic_mapper.py` (`SemanticSpatialMapper`):**
  * Consumes `CameraFrame` detections, back-projects `(u, v, depth)` into 3D world coordinates using `OverheadCamera.pixel_to_world()`, and maintains an Exponential Moving Average (`_EMA_ALPHA = 0.35`) dictionary of `WorldObject` entries (`x, y, z, confidence, last_seen_step`).
* **`simulation/kinematics.py` (`RobotKinematics`):**
  * Implements exact 5-DOF analytical geometric Forward Kinematics (`forward_kinematics`) and Inverse Kinematics (`inverse_kinematics`) for joint limits (`JOINT_LIMITS_DEG`), link lengths (`L_BASE=1.4`, `L_SHOULDER=1.05`, `L1=2.6`, `L2=2.2`, `L_GRIPPER=0.85`), and max reach (`MAX_REACH=4.85`).
  * Includes a multi-branch analytical solver (`_solve_ik_branch`, wrist pitch sweeps from $-90^\circ$ to $0^\circ$) followed by a 60-iteration damped least-squares numerical Jacobian refinement loop if analytical error exceeds `IK_POS_TOLERANCE = 0.12`.
* **`vla/trajectory_controller.py` (`TrajectoryController`):**
  * Takes a sequence of `Waypoint(x, y, z, gripper_open, label)` targets, solves IK for each waypoint once via `RobotKinematics.inverse_kinematics()`, and steps joint angles using a proportional controller (`P_GAIN = 4.5`, `MAX_JOINT_VEL_RAD = 2.8`, `WAYPOINT_CONVERGENCE_RAD = 0.035`, `GRIPPER_SETTLE_STEPS = 12`).
* **`simulation/physics_engine.py` (`PhysicsEngine`):**
  * Runs at 60 Hz (`SIM_DT = 1/60`). Simulates gravity (`GRAVITY = -9.81`), table collisions (`TABLE_SURFACE_Y = 0.0`), object-to-object AABB stacking (`_find_support_surface` checks $XZ$ overlap $\ge 35\%$ of half-width), gripper attachment (`GRASP_PROXIMITY = 1.15`), and toppling instability (`STACK_OFFSET_INSTABILITY = 0.70` applies lateral push velocity when an upper block's center of mass overhangs the support block).
* **`vla/google_vla_agent.py` (`GoogleVLAAgent`):**
  * Runs a finite state machine (`IDLE` $\rightarrow$ `EXPLORE` $\rightarrow$ `COLOR_SEEK` $\rightarrow$ `PICK` $\rightarrow$ `MOVE` $\rightarrow$ `DROP` $\rightarrow$ `COMPLETE` / `FAILED`). It builds fixed 4-waypoint pick trajectories (`TrajectoryController.build_pick_waypoints`) and 3-waypoint place trajectories (`build_place_waypoints`), advancing multi-step queues only via `self._subtask_queue`.

### 2.3 Component Audit: Reuse vs. Refactor vs. Extend

| Existing Module / File | Current Status | Action Plan | Role in New H-VL-JEPA Architecture |
| :--- | :--- | :--- | :--- |
| `simulation/kinematics.py` | Production-ready 5-DOF analytical + Jacobian IK | **Keep Unchanged** | Executes fast, deterministic joint-space conversion for Level 1 waypoints and safety reachability checks. |
| `simulation/physics_engine.py` | 60 Hz tabletop rigid-body & stacking simulator | **Extend Slightly** | Add 5-block spawning and support for RGB frame rasterization so vision encoders receive real RGB tensors alongside synthetic detections. |
| `vla/trajectory_controller.py` | P-controller over joint waypoints | **Keep & Wrap** | Remains the innermost motor execution loop (60 Hz) driven by Level 1 action-conditioned waypoints. |
| `vla/semantic_mapper.py` | 3D EMA coordinate tracker | **Extend** | Upgrade `WorldObject` to store patch-level DINOv3/V-JEPA latent vectors $z_i \in \mathbb{R}^d$ and 3D bounding geometry alongside $(x, y, z)$. |
| `vla/instruction_interpreter.py` | Char-3-gram offline embedding matcher | **Keep as Fast-Path Fallback; Extend with Level 2 Planner** | Retain for $<1\text{ ms}$ simple commands (`"pick red"`); add compositional decomposition (`"build a 5-block tower"`) and route complex commands to the Level 2 VL-JEPA / VLM Task Planner. |
| `vla/google_vla_agent.py` | Hardcoded FSM orchestrator | **Refactor into `HierarchicalJEPAAgent`** | Replace fixed open-loop `PICK`/`MOVE`/`DROP` transitions with a two-level closed-loop planning, prediction, and verification supervisor. |
| `dashboard/server.py` | FastAPI + WebSocket 20 Hz streamer | **Extend** | Add WebSocket telemetry fields for Level 2 subgoal queues, Level 1 CEM candidate rollouts, and predicted vs. actual latent energy errors. |
| `ros2_ws/src/vla_robot_arm/` | ROS 2 wrappers (`vla_cognitive_node.py`, etc.) | **Extend** | Add a `/jepa/world_model_predictions` publisher and update `vla_cognitive_node.py` to call `HierarchicalJEPAAgent`. |

---

## 3. Complete System Architecture Diagram

```text
+---------------------------------------------------------------------------------------------------+
|                                   USER INSTRUCTION INPUT                                          |
|              "Build a stable tower of five blocks: red, blue, green, yellow, purple"              |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                        INPUT ROUTER & INSTRUCTION INTERPRETER (vla/instruction_interpreter.py)    |
|  - Fast Path (Simple 1-step command, conf >= 0.85) ---> Canonical Subtask                         |
|  - Complex / Compositional Path ("tower", "pyramid", N-blocks, constraints)                       |
+-------------------------------------------------+-------------------------------------------------+
                                                  |
                                                  v
=====================================================================================================
  LEVEL 2: HIGH-LEVEL SEMANTIC & TASK PLANNING (Subtask Horizon: k = 1 .. K macro steps)
=====================================================================================================
|                                                                                                   |
|  +-------------------------------------------+      +------------------------------------------+  |
|  | Symbolic Task Decomposer (VLM / LLM)      |      | VL-JEPA / Level-2 Abstract World Model   |  |
|  | (Qwen2.5-VL-3B-Instruct / Gemini API)     |----->| - Context Encoder E^(2)(S_k)             |  |
|  | - Parses goal constraints & block order   |      | - Text/Goal Encoder E_txt(subgoal_k)     |  |
|  | - Emits candidate Subgoal Queue [G_1..G_K]|      | - Macro Predictor F^(2)(z^(2)_k, u_k)    |  |
|  +-------------------------------------------+      |   -> Predicts abstract state z^(2)_{k+1} |  |
|                                                     |   -> Checks global tower stability       |  |
|                                                     +--------------------+---------------------+  |
|                                                                          |                        |
|  Active Subgoal Specification G_k + Target Latent Embedding z^(2)_goal,k |                        |
|  ^                                                                       |                        |
===|=======================================================================|=========================
   | (Failure / Divergence Escalation:                                     v
   |  e_L2 > tau_replan or >2 L1 retries)        (Downward Conditioning: Target z^(2)_goal,k +
   |                                              Spatial Priors [target_id, dest_xyz, tolerances])
===|=======================================================================|=========================
  LEVEL 1: LOW-LEVEL SPATIAL & ACTION-CONDITIONED WORLD MODEL (Horizon: H = 3..5 action steps)
=====================================================================================================
|  |                                                                       |                        |
|  |  +--------------------------------------------------------------------+---------------------+  |
|  |  | Visual & Spatial Context Encoder E^(1)                                                   |  |
|  |  | - Frozen Backbone: DINOv3-ViT-S/16+ (or V-JEPA 2 ViT-L) on Overhead + Gripper RGB        |  |
|  |  | - Spatial Fusion: Concatenates 3D Object Map (SemanticSpatialMapper) + 5-DOF Joint State |  |
|  |  | -> Outputs Current Latent State z^(1)_t                                                  |  |
|  |  +-------------------------------------------------+----------------------------------------+  |
|  |                                                    |                                           |
|  |                                                    v                                           |
|  |  +------------------------------------------------------------------------------------------+  |
|  |  | Action-Conditioned Latent Predictor F^(1)(z^(1)_t, a_{t:t+H}) + CEM Trajectory Planner   |  |
|  |  | 1. Samples N=128 candidate 3D grasp/place action sequences a^(i)_{t:t+H}                 |  |
|  |  | 2. Filters out IK-unreachable or colliding candidates via RobotKinematics                |  |
|  |  | 3. Rolls out F^(1) for H=4 steps in latent space -> \hat{z}^(1)_{t+H}                    |  |
|  |  | 4. Scores cost: J(a) = || P_{1->2}(\hat{z}^(1)_{t+H}) - z^(2)_goal,k ||_2^2 + C_stab     |  |
|  |  | 5. Refits Gaussian over top M=16 elite sequences (3 CEM iterations) -> Optimal a*_t      |  |
|  |  +-------------------------------------------------+----------------------------------------+  |
|  |                                                    |                                           |
|  |  +-------------------------------------------------+----------------------------------------+  |
|  |  | Outcome Verifier & Surprise Detector                                                     |  |
|  |  | - Compares predicted \hat{z}^(1)_{t+1} vs. actual observed z^(1)_{t+1}                   |  |
|  |  | - Prediction Error: e_pred = || \hat{z}^(1)_{t+1} - z^(1)_{t+1} ||_2                     |  |
|  |  | - Goal Energy:      e_goal = || P_{1->2}(z^(1)_{t+1}) - z^(2)_goal,k ||_2                |  |
|  +--+-------------------------------------------------+----------------------------------------+  |
========================================================|============================================
                                                        | Winning Feasible Waypoints [W_1 .. W_m]
                                                        v
=====================================================================================================
  DETERMINISTIC SAFETY, KINEMATICS & EXECUTION LAYER (60 Hz Control Loop)
=====================================================================================================
|  - Analytical 5-DOF IK & Jacobian Refinement (`simulation/kinematics.py`)                         |
|  - Joint Limit & Workspace Boundary Guard (`MAX_REACH = 4.85`, `JOINT_LIMITS_DEG`)                |
|  - Proportional Joint Trajectory Controller (`vla/trajectory_controller.py`)                      |
|  - 60 Hz Physics Engine (`simulation/physics_engine.py`) OR ROS 2 Hardware Interface              |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Detailed Explanation of the Two Hierarchical Levels and Their Interaction

### 4.1 Distinction Between Architectural Roles
To avoid conflating components, each module in the hierarchy has a strict mathematical and functional boundary:
1. **VLM / LLM Task Decomposer (`Qwen2.5-VL-3B-Instruct` or Compositional Rule Decomposer):** Performs **symbolic & linguistic reasoning**. Given a natural language prompt (`"Build a 5-block tower"`) and the current object inventory from `SemanticSpatialMapper`, it outputs a structured JSON graph of ordered subgoals $G_1, \dots, G_K$. A JEPA predictor cannot tokenize arbitrary English syntax or invent combinatorial task graphs out of thin air; using a lightweight local VLM/LLM or structured grammar for symbolic task decomposition is both practical and standard.
2. **Genuine JEPA Predictive World Models (Level 2 $F^{(2)}$ and Level 1 $F^{(1)}$):** Perform **action-conditioned representation prediction in continuous latent space**. Neither $F^{(2)}$ nor $F^{(1)}$ generates text tokens or reconstructs RGB pixels. Instead:
   * Context encoders $E^{(1)}, E^{(2)}$ map observations (visual patch features + 3D object/proprioceptive states) into latent embeddings $z^{(1)}_t, z^{(2)}_k$.
   * Target encoders $\bar{E}^{(1)}, \bar{E}^{(2)}$ (updated via Exponential Moving Average of context encoder weights during training) encode target future observations to prevent representation collapse.
   * Predictors $F^{(1)}, F^{(2)}$ predict the future target embedding $\hat{z}_{t+\Delta t} = F(z_t, a_t)$ conditioned on candidate actions.
3. **Deterministic Robotic Controller (`RobotKinematics` + `TrajectoryController`):** Performs **closed-form geometry and 60 Hz servoing**. It converts Level 1's selected 3D end-effector pose $(x, y, z, \text{pitch}, \text{gripper})$ into 5 joint angles $(\theta_0, \dots, \theta_4)$ and steps motor velocities.

### 4.2 Level 2: High-Level Semantic and Task Planning
* **Timescale & Horizon:** Operates at the **subtask boundary rate** ($\Delta t_2 \approx 3\text{ to }8\text{ seconds}$ per step, where 1 macro-step = 1 complete object manipulation such as picking up a block and placing it on the stack). It plans over $K = 3\text{ to }6$ macro-steps (e.g., placing 5 blocks sequentially).
* **Representations:**
  * **Input State $S_k$:** Coarse scene representation comprising the global overhead visual embedding from the frozen vision backbone plus an object-centric graph $V_k = \{(\text{color}_i, \hat{p}_i \in \mathbb{R}^3, \text{support\_id}_i)\}_{i=1}^M$ from `SemanticSpatialMapper`.
  * **High-Level Latent Embedding $z^{(2)}_k = E^{(2)}(S_k) \in \mathbb{R}^{512}$:** Captures the structural arrangement of objects on the table (ignoring transient robot arm joint angles or velocities).
  * **Macro-Action $u_k$:** A discrete-continuous tuple $(\text{obj\_id}, \Delta x_{\text{place}}, \Delta y_{\text{place}}, \Delta z_{\text{place}})$ specifying which object is moved and its target placement pose.
* **Prediction & Feasibility Verification:**
  Before dispatching Subgoal $G_k$ to Level 1, the Level 2 predictor $F^{(2)}$ rolls out the entire $K$-step sequence:
  $$\hat{z}^{(2)}_{k+1} = F^{(2)}\big(\hat{z}^{(2)}_k, u_k\big), \quad k = 0, \dots, K-1$$
  A lightweight stability probe $h_{\text{stab}}(\hat{z}^{(2)}_{k+1}) \in [0, 1]$ trained on AERO's `PhysicsEngine` (`STACK_OFFSET_INSTABILITY = 0.70`) predicts the probability that the tower remains standing after step $k+1$. If $h_{\text{stab}} < 0.80$ (for example, if a lower block placement is off-center or occluded), Level 2 adjusts the target placement coordinates before the robot arm ever moves.

### 4.3 Level 1: Low-Level Spatial and Action Planning
* **Timescale & Horizon:** Operates at the **motion-phase / waypoint rate** ($\Delta t_1 \approx 0.4\text{ to }1.0\text{ second}$ per step, or every 25–60 simulation ticks), planning over a short horizon of $H = 4$ steps (1: Pre-grasp hover $\rightarrow$ 2: Descend & Grasp $\rightarrow$ 3: Lift & Transport $\rightarrow$ 4: Align & Release). Because $H = 4$ is well within the 3–5 step calibration limit of flat JEPA world models, rollout drift never accumulates.
* **Representations:**
  * **Input Observation $O_t$:** Overhead + Gripper RGB frames ($256 \times 256$), current 5-DOF joint angles $q_t \in \mathbb{R}^5$, end-effector 3D pose $x_{\text{ee}, t} \in \mathbb{R}^4$, gripper state $g_t \in \{0, 1\}$, and local 3D poses of the target and destination objects.
  * **Low-Level Latent Embedding $z^{(1)}_t = E^{(1)}(O_t) \in \mathbb{R}^{384}$:** Preserves fine-grained spatial geometry (gripper-to-block alignment, contact offset, clearance above existing stack).
  * **Action $a_t \in \mathbb{R}^5$:** Continuous end-effector waypoint command $(x, y, z, \Delta\text{pitch}, g_{\text{cmd}})$.
* **Short-Horizon Planning via Cross-Entropy Method (CEM):**
  Given the active Level 2 subgoal embedding $z^{(2)}_{\text{goal}, k}$ and spatial prior $p_{\text{target}}$, Level 1 initializes a Gaussian distribution $\mathcal{N}(\mu^{(0)}_{1:H}, \Sigma^{(0)}_{1:H})$ centered on the nominal pick-and-place waypoints (`build_pick_waypoints` / `build_place_waypoints`).
  For CEM iteration $j = 1, \dots, 3$:
  1. Sample $N = 128$ action sequences $\{a^{(i)}_{1:H}\}_{i=1}^{128}$.
  2. **Deterministic Hard Filter:** Reject any sequence where `RobotKinematics.inverse_kinematics()` returns `reachable == False` or where the arm swept volume intersects an existing stacked block.
  3. **Latent Rollout:** Roll out the Level 1 predictor $F^{(1)}$ for $H = 4$ steps:
     $$\hat{z}^{(1)}_{t+h} = F^{(1)}\big(\hat{z}^{(1)}_{t+h-1}, a^{(i)}_h\big), \quad h = 1, \dots, H$$
  4. **Cross-Level Cost Evaluation:** Project the terminal Level 1 state into Level 2's embedding space via a learned projection head $P_{1\to 2}: \mathbb{R}^{384} \to \mathbb{R}^{512}$ and evaluate:
     $$\mathcal{J}\big(a^{(i)}_{1:H}\big) = \Big\| P_{1\to 2}\big(\hat{z}^{(1)}_{t+H}\big) - z^{(2)}_{\text{goal}, k} \Big\|_2^2 + \lambda_{\text{align}} \, d_{\text{com}}\big(\hat{z}^{(1)}_{t+H}\big) + \lambda_{\text{ctrl}} \sum_{h=1}^H \big\|a^{(i)}_h - a^{(i)}_{h-1}\big\|_2^2$$
  5. Select the top $M = 16$ elite trajectories and update $(\mu^{(j)}, \Sigma^{(j)})$.

### 4.4 Cross-Level Communication, Error Propagation, and Replanning
1. **Downward Conditioning (Level 2 $\rightarrow$ Level 1):** Level 2 passes a `SubgoalSpecification` containing both the target latent vector $z^{(2)}_{\text{goal}, k} \in \mathbb{R}^{512}$ and explicit spatial parameters (`source_object_id`, `target_xyz`, `required_support_id`).
2. **Upward Verification & Error Escalation (Level 1 $\rightarrow$ Level 2):**
   After executing each phase $h$, Level 1 encodes the real post-action observation $z^{(1)}_{t+1} = E^{(1)}(O_{t+1})$ and computes two metrics:
   * **Latent Surprise (Prediction Error):** $e_{\text{pred}} = \| \hat{z}^{(1)}_{t+1} - z^{(1)}_{t+1} \|_2$.
   * **Subgoal Distance (Goal Energy):** $e_{\text{goal}} = \| P_{1\to 2}(z^{(1)}_{t+1}) - z^{(2)}_{\text{goal}, k} \|_2$.
   * **Case A (Nominal Progress, $e_{\text{pred}} \le \tau_{\text{pred}}$ and $e_{\text{goal}} \le \tau_{\text{goal}}$):** Subgoal $k$ is verified complete. Level 1 notifies Level 2, which pops Subgoal $k$ and activates Subgoal $k+1$. This eliminates the "false positive at 10%" bug because completion requires both latent goal convergence and physical support verification (`PhysicsEngine` / `SemanticSpatialMapper` height check).
   * **Case B (Local Perturbation, $e_{\text{pred}} > \tau_{\text{pred}}$ but tower intact):** For example, the gripper slipped during grasp (`grasped_id is None`). Level 1 automatically re-runs CEM from the new current state $z^{(1)}_{t+1}$ (up to `MAX_L1_RETRIES = 2`) without invoking the slower Level 2 planner.
   * **Case C (Structural Failure / Tower Collapse, $e_{\text{pred}} \gg \tau_{\text{pred}}$ or support blocks displaced):** Level 1 raises a `SubgoalExecutionFailure` event to Level 2 with the updated `SemanticSpatialMapper` world state. Level 2 recomputes the remaining subgoal queue from the actual fallen block positions.

---

## 5. Model Selection Table and Recommended Inference Stack

| Architectural Component | Recommended Primary Model (Local) | Source Repository / Hugging Face ID | License & Commercial Use | VRAM (FP16 / Quantized) | Approx. Latency (RTX 3060 / 4070) | Offline / API | Training / Fine-Tuning Required? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Visual Patch Encoder (Primary Local)** | **DINOv3 ViT-S/16+** (`dinov3-vits16plus`) | `facebook/dinov3-vits16plus-pretrain-lvd1689m` (`github.com/facebookresearch/dinov3`) | DINOv3 License (Apache-2.0 alternative: `google/siglip2-base-patch16-256`) | **~0.4 GB** (FP16) | **~6 ms** per $256\times 256$ frame | 100% Offline | **Frozen** pretrained weights; no backbone training needed. |
| **Video / Spatio-Temporal JEPA Backbone** | **V-JEPA 2 ViT-Large** (`vjepa2_vit_large`) or **V-JEPA 2.1 ViT-Base** | `facebookresearch/vjepa2` (`vjepa2_1_vit_base_384` / `vjepa2_vit_large`) | **MIT License** (Commercial use permitted) | **~0.8 GB** (ViT-B) / **~1.4 GB** (ViT-L, FP16) | **~18 ms** (ViT-B) / **~32 ms** (ViT-L) | 100% Offline | **Frozen** backbone; use to extract spatio-temporal video patch embeddings. |
| **Pretrained Action-Conditioned World Model (Reference)** | **V-JEPA 2-AC ViT-Giant** (`vjepa2_ac_vit_giant`) | `facebookresearch/vjepa2` (`torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')`) | **MIT License** (Commercial use permitted) | **~4.8 GB** (FP16) | **~65 ms** per rollout step | 100% Offline | Pretrained on Franka Panda 7-DOF trajectories; requires a lightweight 5-DOF action adapter for AERO. |
| **Level 1 & Level 2 Hierarchical Predictor Architecture** | **H-JEPA / LeWorldModel Predictor MLP/Transformer** | `github.com/kevinghst/H-JEPA` & `quentinll/lewm-cube` (`jepa-world-models/h-jepa`) | **MIT License** (Commercial use permitted) | **~0.25 GB** (L1 + L2 predictors) | **~4 ms** for $N=128, H=4$ batched CEM rollout | 100% Offline | **Train on AERO simulation trajectories** (10k–25k transitions collected automatically from `PhysicsEngine`). |
| **Level 2 Task Decomposer & VL Grounding (Local)** | **Qwen2.5-VL-3B-Instruct** (AWQ / GGUF Q4_K_M) | `Qwen/Qwen2.5-VL-3B-Instruct` | **Apache-2.0** (Commercial use permitted) | **~2.6 GB** (4-bit quantized) | **~280 ms** per task decomposition (called once per prompt/replan) | 100% Offline (via `vLLM`, `llama.cpp`, or `Ollama`) | Zero-shot structured JSON output; no training required. |
| **Level 2 Task Decomposer (Hybrid API Option)** | **Gemini 2.5 Flash** | Google GenAI SDK (`gemini-2.5-flash`) | Commercial API ToS | **0 GB Local VRAM** | **~350–600 ms** network RTT | Cloud API (with local fallback) | Zero-shot structured JSON schema output. |
| **Text Embedding Alignment (VL-JEPA Text Encoder)** | **BAAI/bge-small-en-v1.5** | `BAAI/bge-small-en-v1.5` (Hugging Face / `onnxruntime`) | **MIT License** (Commercial use permitted) | **~0.13 GB** (FP16 / ONNX) | **~2 ms** per instruction | 100% Offline | Frozen encoder + trained linear/MLP projection head into Level 2 latent space. |
| **Monocular Depth Estimation (For Real Camera Sim-to-Real)** | **Depth Anything V2 Small** | `depth-anything/Depth-Anything-V2-Small-hf` | **Apache-2.0** (Commercial use permitted) | **~0.15 GB** (FP16) | **~8 ms** per frame | 100% Offline | Frozen weights; replaces synthetic `OverheadCamera` depth when moving to a real USB webcam. |

---

## 6. Offline Models, APIs, and Infrastructure (Configurations A & B)

### Configuration A: Fully Local (Zero Cloud Dependency)
* **Runtime Stack:**
  * Vision & H-JEPA Predictors: PyTorch (`torch >= 2.4`) with `torch.compile(mode="reduce-overhead")` or ONNX Runtime (`onnxruntime-gpu`).
  * Local VLM Decomposer: `llama-cpp-python` or `Ollama` serving `Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf` on `localhost:11434` with grammar-constrained JSON output.
* **Consumer Hardware Constraints & CPU Fallback Strategy:**
  * **GPU Mode ($\ge 6\text{ GB}$ VRAM):** Full pipeline runs in $\sim 10\text{ ms}$ per Level 1 CEM planning cycle and $\sim 280\text{ ms}$ per Level 2 task decomposition.
  * **Low-VRAM / CPU-Only Fallback (e.g., Render.com Free Tier or Laptop CPU):**
    1. If `torch.cuda.is_available()` is `False`, disable `Qwen2.5-VL-3B` and use an expanded rule + embedding grammar inside `vla/instruction_interpreter.py` (which adds macro-templates for `tower(N, colors)` and `pyramid(base_colors, top_color)` using zero GPU memory).
    2. Swap `DINOv3-ViT-S/16+` for the **State-Plus-Patch Lightweight Encoder** (processes the 3D object state graph + downsampled $64\times 64$ ROI crops via a 2-layer ConvNeXt-Tiny / MLP encoder in $<3\text{ ms}$ on CPU).

### Configuration B: Hybrid (Local Real-Time Control + Cloud API High-Level Reasoning)
* **Workload Split:**
  * **100% Local (Latency-Critical):** Camera capture, `SemanticSpatialMapper`, `DINOv3` feature extraction, Level 1 H-JEPA predictor + CEM trajectory optimization, `RobotKinematics`, and `TrajectoryController` (60 Hz).
  * **Cloud API (Called Only at Task Start or Level 2 Replan):** `gemini-2.5-flash` (or `claude-sonnet-4-5`) for decomposing ambiguous, complex natural-language instructions into the Level 2 `TaskPlan` JSON schema.
* **Resilience, Security, Caching, and Cost Control:**
  * **Credentials:** Loaded strictly via `os.environ.get("AERO_VLM_API_KEY")` (added as an optional entry in `.env.example`, never committed).
  * **Timeouts & Circuit Breaker:** Hard request timeout of `2.5s` (`httpx.Timeout(2.5)`), max 1 retry with exponential jitter (`0.4s`), and a circuit breaker that trips to local offline mode for 5 minutes after 2 consecutive timeouts or `HTTP 429` responses.
  * **Semantic Prompt Cache:** An SQLite/LRU cache keyed by `(normalized_prompt, sorted_workspace_colors)` stores decomposed `TaskPlan` schemas so repeated benchmark runs cost `$0.00` and `0 ms`.
  * **Safety Guarantee:** If the API drops mid-execution, the robot arm **never** freezes in an unsafe state: Level 1 and `TrajectoryController` run locally at 60 Hz and complete the active subgoal before transitioning to `IDLE` or falling back to the local interpreter.

---

## 7. Module-by-Module Integration Plan with AERO's Codebase

### Suggested Directory Structure
Preserve all existing paths so `python run_dashboard.py`, Render deployments (`render.yaml`), and ROS 2 nodes continue to work out of the box, while adding a modular `vla/jepa/` subpackage:

```text
Aero/
├── simulation/
│   ├── constants.py               # [EXTEND] Add H-JEPA horizon, CEM, and stability constants + 5-block defaults
│   ├── camera.py                  # [EXTEND] Add render_rgb_frame() software/offscreen rasterizer
│   ├── kinematics.py              # [KEEP UNCHANGED] 5-DOF analytical + Jacobian IK
│   ├── physics_engine.py          # [KEEP UNCHANGED] 60 Hz rigid-body & stacking physics
│   └── scoring.py                 # [EXTEND] Add multi-block tower & prediction error metrics
├── vla/
│   ├── instruction_interpreter.py # [EXTEND] Add macro task decomposer ("tower", "pyramid") + VLM router
│   ├── semantic_mapper.py         # [EXTEND] Attach latent feature vector & support_id to WorldObject
│   ├── trajectory_controller.py   # [KEEP UNCHANGED] 60 Hz P-controller over waypoints
│   ├── google_vla_agent.py        # [REFACTOR] Delegate to HierarchicalJEPAAgent when mode="hjepa"
│   └── jepa/                      # [NEW PACKAGE]
│       ├── __init__.py
│       ├── schemas.py             # Typed dataclasses / Pydantic schemas for L1/L2 contracts
│       ├── encoders.py            # Frozen DINOv3 / V-JEPA 2 + 3D spatial state fusion encoders
│       ├── level2_world_model.py  # Level 2 VL-JEPA subgoal planner & tower stability predictor
│       ├── level1_world_model.py  # Level 1 action-conditioned predictor & CEM trajectory planner
│       ├── verifier.py            # Latent surprise detector, physical stability check & replan trigger
│       ├── data_collector.py      # Automated simulation rollout collector for H-JEPA training
│       └── train_hjepa.py         # Self-supervised VICReg + multi-step latent prediction trainer
├── dashboard/
│   └── server.py                  # [EXTEND] Stream L2 subgoals, L1 CEM rollouts, & surprise metrics
└── tests/                         # [NEW DIRECTORY]
    ├── test_tower_decomposition.py
    ├── test_hjepa_predictors.py
    └── benchmark_aero_vs_hjepa.py
```

---

## 8. Data Schemas and Key Interfaces (`vla/jepa/schemas.py`)

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple
import numpy as np


@dataclass
class ObjectLatentState:
    """Grounded 3D object state fused with visual patch embedding."""
    label: str
    color: str
    position_xyz: Tuple[float, float, float]
    half_extents: Tuple[float, float, float]
    confidence: float
    support_object_label: Optional[str]       # None if resting on table (y == 0)
    visual_embedding: np.ndarray              # Shape: (D_vis,) e.g. (384,) from DINOv3


@dataclass
class WorkspaceObservation:
    """Complete multi-modal observation passed to Level 1 and Level 2 encoders."""
    step_id: int
    sim_time: float
    overhead_rgb: np.ndarray                  # Shape: (256, 256, 3), uint8
    gripper_rgb: np.ndarray                   # Shape: (256, 256, 3), uint8
    joint_angles_rad: Tuple[float, float, float, float, float]
    end_effector_xyz: Tuple[float, float, float]
    gripper_open: bool
    grasped_object_label: Optional[str]
    objects: Dict[str, ObjectLatentState]


@dataclass
class SubgoalSpecification:
    """Contract emitted by Level 2 and consumed by Level 1."""
    subgoal_id: int
    action_type: Literal["pick_and_place", "stack", "push", "clear_obstacle"]
    source_label: str
    source_color: str
    target_xyz: Tuple[float, float, float]
    support_label: Optional[str]              # Expected object directly underneath after placement
    target_latent_l2: np.ndarray              # z^(2)_{goal, k} in R^{512}
    predicted_stability_prob: float           # h_stab(z^(2)_{goal, k}) in [0, 1]
    position_tolerance_m: float = 0.22
    description: str = ""


@dataclass
class CEMPlanResult:
    """Output of Level 1 Action-Conditioned World Model + CEM Planner."""
    subgoal_id: int
    optimal_actions: np.ndarray               # Shape: (H, 5) -> [x, y, z, pitch_deg, gripper_open]
    predicted_latents_l1: np.ndarray          # Shape: (H, 384) -> \hat{z}^(1)_{t+1:t+H}
    terminal_goal_energy: float               # || P_{1->2}(\hat{z}^(1)_{t+H}) - z^(2)_goal ||_2
    predicted_stability: float                # Predicted post-release stability in [0, 1]
    candidate_trajectories_xyz: List[List[Tuple[float, float, float]]] = field(default_factory=list)
    ik_feasible: bool = True


@dataclass
class VerificationReport:
    """Closed-loop outcome check emitted by OutcomeVerifier after action execution."""
    subgoal_id: int
    phase_index: int
    latent_prediction_error: float            # e_pred = || \hat{z}^(1)_{t+1} - z^(1)_{t+1} ||_2
    subgoal_energy_distance: float            # e_goal = || P_{1->2}(z^(1)_{t+1}) - z^(2)_goal ||_2
    physical_contact_verified: bool           # True if block is grasped or resting stably on target
    tower_intact: bool                        # False if any previously placed block toppled
    status: Literal["IN_PROGRESS", "SUBGOAL_COMPLETED", "RETRY_L1", "ESCALATE_L2_REPLAN"]
    failure_reason: Optional[str] = None
```

---

## 9. Training and Fine-Tuning Strategy for the JEPA Components

Because no off-the-shelf checkpoint is pre-trained on AERO's specific 5-DOF arm geometry and tabletop coordinate frame, we use **frozen foundation encoders (`DINOv3` + `bge-small-en-v1.5`)** and train **lightweight Level 1 and Level 2 predictor networks (`~3.5M` parameters total)** directly on simulation trajectories generated by AERO's `PhysicsEngine`.

### 9.1 Observations and Encoder Architecture
1. **Level 1 Context Encoder $E^{(1)}_\theta$:**
   * Extracts patch tokens from `overhead_rgb` and `gripper_rgb` using frozen `DINOv3-ViT-S/16+`, applying attentive pooling over the object bounding regions to obtain $v_t \in \mathbb{R}^{384}$.
   * Encodes the proprioceptive & local object state $s^{(1)}_t = [q_t, x_{\text{ee}, t}, g_t, p_{\text{src}, t} - x_{\text{ee}, t}, p_{\text{dst}, t} - p_{\text{src}, t}] \in \mathbb{R}^{18}$ via a 2-layer MLP with LayerNorm and SiLU into $e_{\text{state}} \in \mathbb{R}^{128}$.
   * Fuses $[v_t; e_{\text{state}}]$ through a trainable linear projection to produce $z^{(1)}_t \in \mathbb{R}^{384}$.
2. **Level 2 Context Encoder $E^{(2)}_\phi$:**
   * Operates as an **Object-Centric Transformer Encoder** over the set of workspace objects $\{(\text{color\_emb}_i, p_{i,t}, \text{extents}_i, v_{i,t})\}_{i=1}^M$ pooled with the global scene feature, outputting $z^{(2)}_k \in \mathbb{R}^{512}$.
   * **VL-JEPA Language Alignment Head $E_{\text{txt}}$:** Maps the subgoal text description + target object attributes into the same $\mathbb{R}^{512}$ space so Level 2 can specify goals directly from language embeddings without requiring a rendered picture of the finished 5-block tower.

### 9.2 Automated Data Collection in AERO (`vla/jepa/data_collector.py`)
Collect $20,000$ multi-block episodes headless at $2,000\text{ FPS}$ ($\sim 15\text{ minutes}$ wall-clock time on CPU) using `PhysicsEngine`:
* **50% Expert Demonstrations:** Scripted multi-block tower and pyramid placements with randomized initial block coordinates.
* **35% Noisy / Perturbed Trajectories:** Add Gaussian offsets $\epsilon \sim \mathcal{N}(0, 0.35\text{ m})$ to pick and place waypoints so the world model observes missed grasps, edge collisions, and **tower toppling events** when overhang exceeds `STACK_OFFSET_INSTABILITY = 0.70`.
* **15% Random Exploration Rollouts:** Random reachable waypoints for broad workspace coverage.

### 9.3 Training Objectives and Collapse Prevention (VICReg + Multi-Step Latent Prediction)
Following **H-JEPA** (`kevinghst/H-JEPA`) and **VICReg** (Bardes, Ponce, LeCun 2022), train the encoders and predictors with a four-part loss that prevents representation collapse without needing negative contrastive batches:

1. **Level 1 Multi-Step Action-Conditioned Prediction Loss:**
   $$\mathcal{L}_{\text{pred}}^{(1)} = \frac{1}{H} \sum_{h=1}^H \Big\| \hat{z}^{(1)}_{t+h} - \text{sg}\big(\bar{E}^{(1)}(O_{t+h})\big) \Big\|_2^2, \quad \text{where } \hat{z}^{(1)}_{t+h} = F^{(1)}\big(\hat{z}^{(1)}_{t+h-1}, a_{t+h-1}\big)$$
   Here $\text{sg}(\cdot)$ is stop-gradient and $\bar{E}^{(1)}$ is the EMA target encoder ($\tau_{\text{EMA}} = 0.996$).
2. **Level 2 Subtask-Boundary Prediction & VL-JEPA Alignment Loss:**
   $$\mathcal{L}_{\text{pred}}^{(2)} = \Big\| F^{(2)}\big(z^{(2)}_k, u_k\big) - \text{sg}\big(\bar{E}^{(2)}(S_{k+1})\big) \Big\|_2^2 + \lambda_{\text{vl}} \Big\| E_{\text{txt}}(\text{subgoal}_k) - \text{sg}\big(\bar{E}^{(2)}(S_{k+1})\big) \Big\|_2^2$$
3. **Cross-Level Consistency Loss:** Aligns the terminal state of a completed Level 1 rollout with the Level 2 state space:
   $$\mathcal{L}_{\text{hier}} = \Big\| P_{1\to 2}\big(z^{(1)}_{t+H}\big) - \text{sg}\big(z^{(2)}_{k+1}\big) \Big\|_2^2$$
4. **VICReg Variance-Covariance Regularization (Collapse Prevention):**
   Applied to the batch of embeddings $Z \in \mathbb{R}^{B \times D}$:
   $$\mathcal{L}_{\text{VIC}}(Z) = \frac{1}{D} \sum_{j=1}^D \max\big(0, 1 - \sqrt{\text{Var}(Z_{:, j}) + \epsilon}\big) + \frac{1}{D} \sum_{i \neq j} \big[\text{Cov}(Z)\big]_{i,j}^2$$
   plus binary cross-entropy $\mathcal{L}_{\text{stab}}$ on the auxiliary stability probe $h_{\text{stab}}$ using ground-truth topple flags from `PhysicsEngine`.

---

## 10. Phased Implementation Roadmap

Walkthrough benchmark task across all phases: **`"Build a tower of five blocks"`** (requiring 5 sequential block localizations, grasp-transport-stack cycles, center-of-mass alignment checks to avoid `STACK_OFFSET_INSTABILITY = 0.70`, and automatic recovery if a block slips or falls).

### Phase 1: Compositional Task Decomposition & Closed-Loop Verification Foundation (Week 1 — ~20 hrs)
* **Objective:** Upgrade AERO from single-object/regex-split execution to multi-block compositional task planning (`5-block tower`, `3-block pyramid`) with physical + spatial subgoal verification, prior to neural predictor training.
* **Concrete Tasks:**
  1. Create `vla/jepa/schemas.py` and `vla/jepa/verifier.py`.
  2. Extend `simulation/physics_engine.py` and `simulation/constants.py` to support spawning 5 distinct blocks (`red`, `blue`, `green`, `yellow`, `purple`) with collision-free initial placement.
  3. Upgrade `vla/instruction_interpreter.py` to decompose *"build a tower of N blocks"* and *"build a pyramid"* into an ordered `List[SubgoalSpecification]` with computed stack heights $y_k = (k - 0.5) \cdot \text{OBJECT\_SIZE}$.
  4. Add closed-loop verification in `verifier.py`: verify that block $k$'s actual 3D position after `DROP` settles at $(x_{\text{target}}, y_k, z_{\text{target}})$ within `0.25m` and velocity $\|v\| < 0.05\text{ m/s}$ before advancing to block $k+1$. If the block fell to $y \approx 0.25$, re-queue the subgoal automatically.
* **Deliverables & Acceptance Criteria:**
  * Running `"build a tower of five blocks"` in the dashboard decomposes into 5 sequential stack subtasks and completes a stable 5-block vertical tower with $\ge 85\%$ success under zero noise.

### Phase 2: RGB Rasterizer, Frozen Vision Encoder (`DINOv3`), and Trajectory Data Collector (Week 2 — ~25 hrs)
* **Objective:** Replace pure coordinate-cheat perception with real RGB frame rendering and frozen `DINOv3` / `bge-small` latent feature extraction, and collect 20,000 simulation trajectories.
* **Concrete Tasks:**
  1. Implement `OverheadCamera.render_rgb()` and `GripperCamera.render_rgb()` in `simulation/camera.py`.
  2. Implement `vla/jepa/encoders.py` wrapping frozen `facebook/dinov3-vits16plus-pretrain-lvd1689m` (with an automatic lightweight MLP state+crop encoder fallback when running without PyTorch/CUDA).
  3. Implement `vla/jepa/data_collector.py` and generate `data/aero_hjepa_20k.npz` containing $(O_t, a_t, O_{t+1}, \text{toppled}_t)$ across 1-to-5 block stacking episodes.
* **Deliverables & Acceptance Criteria:**
  * `encoders.py` encodes $(O_t) \to z^{(1)}_t \in \mathbb{R}^{384}$ in $<10\text{ ms}$ on GPU ($<3\text{ ms}$ in lightweight mode on CPU).
  * Dataset of 20,000 multi-step stacking episodes saved and validated.

### Phase 3: Training Level 1 & Level 2 H-JEPA World Models + CEM Planner (Weeks 3–4 — ~35 hrs)
* **Objective:** Train the Level 1 (`level1_world_model.py`) and Level 2 (`level2_world_model.py`) predictors and integrate Cross-Entropy Method (CEM) trajectory optimization.
* **Concrete Tasks:**
  1. Implement `vla/jepa/train_hjepa.py` with the joint $\mathcal{L}_{\text{pred}}^{(1)} + \mathcal{L}_{\text{pred}}^{(2)} + \mathcal{L}_{\text{hier}} + \mathcal{L}_{\text{VIC}} + \mathcal{L}_{\text{stab}}$ objective.
  2. Implement `CEMPlanner` inside `vla/jepa/level1_world_model.py` ($N=128$ candidates, $H=4$ steps, 3 iterations, IK reachability filter via `RobotKinematics.inverse_kinematics`).
  3. Implement Level 2 stability lookahead in `level2_world_model.py`: before executing a placement on top of a slightly off-center lower block, Level 2 shifts the target $(x, z)$ placement toward the lower block's actual center of mass to maximize $h_{\text{stab}}(\hat{z}^{(2)}_{k+1})$.
* **Deliverables & Acceptance Criteria:**
  * Validation latent cosine similarity $\ge 0.92$ at $h=4$ steps; topple prediction AUROC $\ge 0.94$.
  * Level 1 CEM planning completes in $<15\text{ ms}$ per subgoal.

### Phase 4: Full Closed-Loop Hierarchical Agent, Hybrid VLM Router & Dashboard Telemetry (Week 5 — ~20 hrs)
* **Objective:** Wire Level 2, Level 1, and `OutcomeVerifier` into `GoogleVLAAgent` (`mode="hjepa"`), expose real-time world-model telemetry in the 3D Three.js dashboard (`dashboard/`), and update ROS 2 nodes.
* **Concrete Tasks:**
  1. Refactor `vla/google_vla_agent.py` to orchestrate the Level 2 $\leftrightarrow$ Level 1 loop, handling L1 CEM retries on grasp slips and L2 replanning on external block perturbations.
  2. Update `dashboard/server.py` and `dashboard/static/js/sim_canvas_3d.js` to render the top CEM candidate trajectories as translucent 3D arcs and display live `e_pred` (surprise) and `h_stab` (predicted stability) gauges.
  3. Update `ros2_ws/src/vla_robot_arm/vla_robot_arm/vla_cognitive_node.py` to publish `/vla/world_model_telemetry`.
* **Deliverables & Acceptance Criteria:**
  * Live 3D web dashboard visualizes CEM rollouts, subgoal queue progression, and autonomous recovery when a block is mid-task displaced via `/api/reset` or external perturbation.

---

## 11. Evaluation and Benchmarking Strategy

Create `tests/benchmark_aero_vs_hjepa.py` to run 100 seeded episodes across 4 benchmark suites, comparing three architectures head-to-head:
1. **Baseline 1 (Current AERO):** `InstructionInterpreter` + open-loop `GoogleVLAAgent` FSM.
2. **Baseline 2 (VLM Planner + Deterministic IK, No JEPA):** `Qwen2.5-VL` / `InstructionInterpreter` subgoal decomposition + direct nominal waypoints (`build_pick_waypoints` / `build_place_waypoints`) without Level 1/2 JEPA prediction or CEM.
3. **Proposed (2-Level H-VL-JEPA + CEM):** Full Level 2 + Level 1 predictive world model with CEM center-of-mass optimization and latent outcome verification.

| Benchmark Suite | Description & Perturbation Conditions | Metric | Baseline 1: Current AERO | Baseline 2: VLM + Nominal IK (No JEPA) | Proposed: 2-Level H-VL-JEPA + CEM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Suite 1: Compositional 5-Block Tower** | Prompt: *"Build a tower of five blocks"* with standard camera noise (`PIXEL_NOISE_STD = 1.5` px) | Task Completion Rate (%) | **0%** (parses as 1 step; stops after 1st block) | **52%** (cumulative EMA localization drift causes block 4 or 5 to overhang $>70\%$ and topple) | **$\ge 88\%$** (CEM adjusts placement $(x, z)$ using predicted stability $h_{\text{stab}}$ and verifies each layer) |
| **Suite 2: High Perception Noise & Occlusion** | 3-block stack with elevated camera noise (`PIXEL_NOISE_STD = 4.5` px) and partial occlusion | Grasp & Stack Success Rate (%) | **38%** | **46%** | **$\ge 82\%$** (Multi-frame latent state + L1 retry recovers from missed grasps) |
| **Suite 3: Mid-Task External Perturbation (Recovery)** | At step $t = 180$, block 2 is displaced by $0.8\text{ m}$ or knocked off the stack while arm is in transit | Recovery Success Rate (%) | **0%** (drops block in mid-air at old coordinates and marks `COMPLETE`) | **15%** (no pre-release state comparison) | **$\ge 85\%$** (`OutcomeVerifier` detects $e_{\text{pred}} > \tau_{\text{pred}}$, triggers L2 replan, re-stacks fallen block) |
| **Suite 4: End-to-End Latency & Compute** | Wall-clock planning time per manipulation step | Mean Planning Latency (ms) | **0.4 ms** | **280 ms** (L2 once) + **0.5 ms**/step | **280 ms** (L2 once) + **8.5 ms**/subgoal (L1 CEM) |

---

## 12. Key Technical Risks, Limitations, and Fallback Plans

| Risk / Limitation | Severity | Root Cause | Concrete Mitigation & Fallback Plan |
| :--- | :--- | :--- | :--- |
| **1. Latent Representation Collapse During H-JEPA Training** | High | Predictor outputs constant vectors if target encoder updates too quickly or variance regularization is under-weighted. | Use frozen `DINOv3` visual features as anchor targets alongside EMA target encoders ($\tau = 0.996$) and enforce the VICReg variance hinge loss ($\gamma = 1.0$, weight $= 25.0$). Monitor batch standard deviation $\sigma(Z) > 0.5$ during training. |
| **2. Compounding Rollout Drift Beyond 4–5 Steps** | Medium | Autoregressive rollout of $F^{(1)}$ accumulates approximation error over long horizons. | Strictly cap Level 1 rollout horizon to $H = 4$ macro-waypoints (Pre-grasp, Grasp, Lift/Transport, Release) and delegate multi-block sequencing to Level 2 ($K \le 6$ subgoals). |
| **3. Heavy Dependencies Breaking Render.com Cloud Demo (`render.yaml`)** | Medium | Full PyTorch + `transformers` + `Qwen2.5-VL` exceeds typical free-tier cloud container RAM limits (512 MB – 2 GB). | Export trained `H-JEPA` Level 1/Level 2 predictors and the lightweight State+Crop encoder to **NumPy / ONNX** (`< 25 MB` total) for cloud CPU deployments, while supporting full PyTorch GPU backbones locally via environment flag `AERO_BACKEND=onnx\|torch`. |
| **4. Sim-to-Real Gap When Moving to Physical 5-DOF Hardware** | Medium | Synthetic polygon rasterizer in `camera.py` lacks real lighting, shadows, and sensor depth noise. | Because `DINOv3` is pre-trained on 1.7B real-world web images, its patch embeddings transfer directly to real cameras; pair it with `Depth-Anything-V2-Small` in `ros2_ws/src/vla_robot_arm/semantic_spatial_mapper_node.py` when connecting a physical depth/RGB camera. |

---

## 13. Initial Implementation Tasks and Suggested Coding Sequence (Start Here)

Execute the following 5 steps in sequence to implement Phase 1 and the core H-JEPA scaffolding:

1. **Step 1 — Add 5-Block Support to Simulation Constants (`simulation/constants.py` & `simulation/physics_engine.py`):**
   * In `simulation/constants.py`, add `"yellow"` and `"purple"` blocks to `DEFAULT_OBJECTS` (or add a `spawn_five_blocks()` helper in `PhysicsEngine`) so the tabletop workspace supports 5-block tower and pyramid tasks.
2. **Step 2 — Create the Typed Contracts (`vla/jepa/schemas.py`):**
   * Implement the `ObjectLatentState`, `WorkspaceObservation`, `SubgoalSpecification`, `CEMPlanResult`, and `VerificationReport` dataclasses from Section 8.
3. **Step 3 — Upgrade `InstructionInterpreter.parse_sequence()` (`vla/instruction_interpreter.py`):**
   * Add a compositional pattern matcher before the single-step fallback in `parse_sequence(prompt)` that detects tower/pyramid/multi-block instructions (e.g., `r"\b(tower|stack|pyramid)\b.*?\b(\d+|three|four|five)\s+blocks?\b"`) and emits an ordered sequence of `ParsedSubtask` / `SubgoalSpecification` items with progressive stack heights.
4. **Step 4 — Implement the Closed-Loop `OutcomeVerifier` & `Level1WorldModel` (`vla/jepa/verifier.py` & `vla/jepa/level1_world_model.py`):**
   * Implement the CEM planner over `RobotKinematics.inverse_kinematics()` and the post-drop physical/latent verifier so the agent checks whether each block actually settled on top of the previous block before advancing `self._subtask_queue` in `vla/google_vla_agent.py`.
5. **Step 5 — Run the Baseline vs. Hierarchical Benchmark (`tests/benchmark_aero_vs_hjepa.py`):**
   * Run 50 episodes of `"build a tower of five blocks"` in headless mode (`PhysicsEngine` + `GoogleVLAAgent`) to verify that task completion jumps from 0% (single-step stop) to $\ge 85\%$ with automatic retry on slip or topple.