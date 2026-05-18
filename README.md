# Multi-Robot Warehouse Simulation

2D simulation of an autonomous multi-robot warehouse system with computer vision, path planning, and force control.

## Overview

This project implements a complete multi-robot coordination system for warehouse logistics using vision-based perception (no LiDAR/GPS). The system demonstrates:

- Vision-based navigation using RGB cameras
- A* path planning with dynamic replanning
- Multi-robot task allocation
- Force-controlled manipulation
- Real-time metrics tracking

## System Architecture

### Robots
- **Husky A200**: Skid-steer mobile robot for corridor clearing
- **ANYmal**: Quadruped robot for PuzzleBot transport
- **PuzzleBots (x3)**: Mobile manipulators with 3-DOF planar arms

### Computer Vision
- Color-based object detection
- Contour detection and analysis
- ArUco landmark localization
- Distance estimation from image size

### Planning & Control
- A* pathfinding on discretized grid
- Greedy task allocation
- Collision avoidance with visual perception
- Dynamic replanning on obstacle detection

## Project Structure

```
├── sim.py                    # Main 2D simulator with vision integration
├── coordinator.py            # Phase orchestration and state machine
├── husky_pusher.py          # Husky A200 with visual perception
├── anymal_gait.py           # ANYmal quadruped kinematics
├── puzzlebot_arm.py         # 3-DOF arm with force control
├── vision_camera.py         # Synthetic RGB camera
├── vision_perception.py     # Multi-technique visual perception
├── pathfinding.py           # A* planner with replanning
├── task_allocator.py        # Multi-robot task assignment
├── metrics_tracker.py       # Performance metrics
├── vision_config.py         # Vision parameters
├── torque_logger.py         # Force control logging
├── robot_ml.py              # ML models integration
└── requirements.txt         # Dependencies
```

## Key Features

**Phase 1 - Corridor Clearing**
- Skid-steer locomotion with slip compensation
- Visual obstacle detection (replaces LiDAR)
- Non-blocking state machine for smooth animation

**Phase 2 - Transport**
- Quadruped trot gait with diagonal pairs
- Per-leg FK/IK with singularity monitoring
- Payload transport (3 PuzzleBots)

**Phase 3 - Collaborative Stacking**
- 3-DOF planar arm manipulation
- Force control: τ = J^T × f
- Event-based synchronization (C → B → A)
- Collision avoidance with exclusion zones

**Computer Vision**
- Color-based detection
- Contour analysis
- ArUco landmarks
- Distance estimation from pixel size

**Planning & Metrics**
- A* pathfinding on 2D grid
- Dynamic replanning on obstacle detection
- Makespan, collision avoidance, replan tracking

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python sim.py
```

The simulation will:
1. Display live animation with aerial view and camera view
2. Execute all three phases sequentially
3. Generate performance metrics
4. Save output images to `results/`

## Output Files

- `results/sim_output.png` - Key frames from simulation
- `results/metrics.png` - Performance metrics dashboard
- `results/torque_report.json` - Force control data
- `results/torque_analysis.png` - Torque analysis plots

## Technical Details

**Vision System**
- Camera resolution: 320x240 pixels
- FOV: 90 degrees horizontal
- Range: 10 meters
- Detection techniques: Color, contour, ArUco, distance estimation

**Path Planning**
- Grid resolution: 0.15 meters
- Heuristic: Euclidean distance
- Movement: 8-directional

**Task Allocation**
- Strategy: Greedy (nearest robot)
- Alternative: Round-robin

**Performance Metrics**
- Makespan: Total mission time
- Collisions avoided: Safety events
- Replanning events: Dynamic adaptations
- Fleet efficiency: Tasks per robot-time

## Theoretical Background

### Vision-Based Perception

**Color Detection**
Objects are classified by dominant RGB channel. For each detection, the system compares R, G, B values to determine object type (obstacles, robots, landmarks).

**Contour Analysis**
Bounding boxes are extracted from detections and filtered by area (50-10000 px²). Object size is estimated from contour dimensions using perspective projection.

**Distance Estimation**
Distance d is computed from pixel size using pinhole camera model:
```
d = (real_size × focal_length) / pixel_size
```
where focal_length is calibrated to 200 pixels.

**ArUco Landmarks**
Fiducial markers provide absolute position references. The system detects 4 ArUco markers at known positions for visual localization.

### Path Planning

**A\* Algorithm**
The planner uses A\* search on a discretized 2D grid (0.15m resolution). Cost function:
```
f(n) = g(n) + h(n)
```
where g(n) is path cost from start and h(n) is Euclidean heuristic to goal.

**Dynamic Replanning**
When visual perception detects new obstacles, the system:
1. Adds obstacle to occupancy grid
2. Invalidates current path if blocked
3. Triggers A\* replanning from current position
4. Logs replanning event for metrics

### Force Control

**Jacobian Transpose Method**
For 3-DOF planar arm, joint torques τ are computed from desired Cartesian force f:
```
τ = J^T × f
```
where J is the 3×3 analytical Jacobian matrix.

**Singularity Detection**
Singularities are monitored via Jacobian determinant:
```
|det(J)| < 10^-3  →  singularity warning
```

### Multi-Robot Coordination

**Task Allocation**
Greedy strategy assigns each task to nearest available robot. Distance metric is Euclidean norm in workspace.

**Event-Based Synchronization**
PuzzleBots stack in order C → B → A using event flags. Each robot waits for predecessor's completion event before starting.

**Collision Avoidance**
Exclusion zones (0.5m radius) prevent inter-robot collisions. Visual perception triggers avoidance when obstacles detected within threshold.

## Dependencies

- Python 3.10+
- numpy >= 2.0.0
- matplotlib >= 3.8.0

## License

Academic project - TE3002B Mobile Terrestrial Robots
