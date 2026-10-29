# Edge-Optimized Sensor Fusion (EOS-DRL): A Unified SLAM-DRL Framework for Autonomous Service Robots

A high-performance, edge-optimized robotic navigation framework combining classical global path planning (A*) with deep reinforcement learning (PPO) for local reactive obstacle avoidance, designed for resource-constrained edge computing platforms (e.g., NVIDIA Jetson Nano) and differential-drive mobile robots.

## Project Architecture

```
prithvi_ai_model/
├── eos_drl_robotics/
│   ├── drl_navigator.py             # Asymmetric Actor-Critic PPO architecture
│   ├── drl_navigator_node.py        # ROS2 inference node for real-time local planning
│   ├── kinematics_controller.py     # Inverse kinematics & deadband-compensated motor mapping
│   ├── perception_node.py           # MobileNetV3 semantic occupancy grid encoder & ROS2 node
│   ├── reward_environment.py        # Custom Gymnasium environment with dynamic obstacles & A*
│   ├── simulate_pipeline.py         # End-to-end closed-loop HIL simulation runner
│   ├── train_ppo.py                 # PPO training pipeline with GAE
│   ├── firmware/
│   │   └── arduino_motor_controller.ino # Low-level Arduino PID/PWM motor controller firmware
│   ├── simulator/
│   │   └── index.html               # Interactive browser-based 2D simulation canvas
│   ├── resource/
│   ├── package.xml                  # ROS2 package manifest (ament_python)
│   └── setup.py                     # ROS2 / Python package setup script
├── .gitignore
└── README.md
```

## Core Features

- **Hybrid A* + PPO Local Reactive Navigator**: Combines global topological guidance with local evasion of dense, dynamic obstacles.
- **Lightweight Perception Pipeline**: Strided MobileNetV3 backbone compresses Bird's Eye View (BEV) semantic occupancy grids for real-time edge execution.
- **Differential Drive Kinematics**: Translates `(v, ω)` command velocities into 8-bit PWM signals with non-linear deadband compensation.
- **Embedded Firmware**: Arduino motor driver code supporting dual H-Bridge control with interrupt-based optical encoder feedback.
- **Interactive Simulator**: Zero-dependency HTML5/Canvas visualization for real-time testing of perception, navigation, and obstacle evasion.

## Quick Start

### 1. Requirements

- Python 3.8+
- PyTorch
- Gymnasium
- NumPy, OpenCV
- ROS2 (Humble / Iron / Rolling) - *optional for standalone simulation*

### 2. Standalone Simulation

Run the end-to-end pipeline simulation:

```bash
cd eos_drl_robotics
python simulate_pipeline.py
```

### 3. PPO Policy Training

Train the local reactive navigation policy:

```bash
cd eos_drl_robotics
python train_ppo.py
```

### 4. Interactive Visualizer

Open `eos_drl_robotics/simulator/index.html` in any modern web browser to view the interactive simulation canvas.

### 5. ROS2 Package Build

In a ROS2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cp -r eos_drl_robotics ~/ros2_ws/src/
cd ~/ros2_ws
colcon build --packages-select eos_drl_robotics
source install/setup.bash
```

Launch the nodes:

```bash
ros2 run eos_drl_robotics perception_node
ros2 run eos_drl_robotics drl_navigator_node
ros2 run eos_drl_robotics kinematics_controller
```

## License

Apache-2.0
