#!/usr/bin/env python3
"""
================================================================================
Project: Edge-Optimized Sensor Fusion (EOS-DRL)
Module: DRL Navigator ROS 2 Node (drl_navigator_node.py)
Target Hardware: NVIDIA Jetson Nano
================================================================================
ROS 2 Humble node that:
1. Subscribes to /perception/occupancy_grid_tensor (from perception_node.py)
2. Subscribes to /planning/waypoint_distance (or tracks A* waypoint distance)
3. Feeds them through the PyTorch Actor-Critic network using CUDA inference
4. Publishes continuous geometry_msgs/Twist commands to /drl/cmd_vel
================================================================================
"""

import os
import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32MultiArray, Float32
from geometry_msgs.msg import Twist

from drl_navigator import ActorCritic


class DRLNavigatorNode(Node):
    def __init__(self):
        super().__init__('drl_navigator_node')

        self.declare_parameter('model_weights_path', 'ppo_eos_checkpoint.pt')
        self.declare_parameter('max_linear_velocity', 0.5)
        self.declare_parameter('max_angular_velocity', 1.0)
        self.declare_parameter('control_rate_hz', 20.0)

        self.weights_path = self.get_parameter('model_weights_path').value
        self.v_max = self.get_parameter('max_linear_velocity').value
        self.omega_max = self.get_parameter('max_angular_velocity').value
        self.rate_hz = self.get_parameter('control_rate_hz').value

        # Compute device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"[DRL_NAV_INIT] Compute Backend: {self.device}")

        # Instantiate Actor-Critic Policy
        self.policy = ActorCritic(
            grid_channels=3,
            guidance_dim=1,
            max_linear_vel=self.v_max,
            max_angular_vel=self.omega_max
        ).to(self.device)

        if os.path.exists(self.weights_path):
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            self.policy.load_state_dict(checkpoint['model_state_dict'])
            self.get_logger().info(f"[DRL_NAV_INIT] Loaded model weights from {self.weights_path}")
        else:
            self.get_logger().warn(f"[DRL_NAV_INIT] No checkpoint found at {self.weights_path}. Running with initialized weights.")

        self.policy.eval()

        # State buffers
        self.latest_grid = np.zeros((3, 64, 64), dtype=np.float32)
        self.latest_grid[0, :, :] = 1.0  # Default free space
        self.latest_wp_distance = 1.0     # Default 1.0m to waypoint
        self.received_perception = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        self.grid_sub = self.create_subscription(
            Float32MultiArray,
            '/perception/occupancy_grid_tensor',
            self.grid_callback,
            qos
        )
        self.dist_sub = self.create_subscription(
            Float32,
            '/planning/waypoint_distance',
            self.dist_callback,
            10
        )

        # Velocity Publisher
        self.cmd_pub = self.create_publisher(Twist, '/drl/cmd_vel', 10)

        # Control Loop Timer
        self.timer = self.create_timer(1.0 / self.rate_hz, self.control_loop)
        self.get_logger().info("[DRL_NAV_INIT] DRL Navigator Node Running at 20 Hz.")

    def grid_callback(self, msg: Float32MultiArray):
        # Reshape flattened 12,288 float array back to [3, 64, 64]
        try:
            arr = np.array(msg.data, dtype=np.float32).reshape((3, 64, 64))
            self.latest_grid = arr
            self.received_perception = True
        except Exception as e:
            self.get_logger().error(f"Error unpacking grid tensor: {e}")

    def dist_callback(self, msg: Float32):
        self.latest_wp_distance = float(msg.data)

    def control_loop(self):
        # Neural Inference
        grid_t = torch.from_numpy(self.latest_grid).unsqueeze(0).to(self.device)
        dist_t = torch.tensor([[self.latest_wp_distance]], dtype=torch.float32, device=self.device)

        action, log_prob, val = self.policy.act(grid_t, dist_t, deterministic=True)
        v = float(action[0])
        omega = float(action[1])

        # Publish Twist command to kinematics controller
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = omega
        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = DRLNavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[DRL_NAV_SHUTDOWN] Terminating.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
