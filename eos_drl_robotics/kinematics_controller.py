#!/usr/bin/env python3
"""
================================================================================
Project: Edge-Optimized Sensor Fusion: A Unified SLAM-DRL Framework for Autonomous Service Robots
Module: Kinematics & Actuation Controller (kinematics_controller.py)
Target Hardware: NVIDIA Jetson Nano (ROS 2 Humble) -> USB Serial / ROS 2 -> Arduino Mega 2560
Actuation: Differential Drive Platform, Optical Wheel Encoders, Dual H-Bridge Motor Drivers

ACADEMIC RESEARCH GAP & METHODOLOGICAL RATIONALE:
--------------------------------------------------------------------------------
In deep reinforcement learning robotics pipelines, a persistent "Sim-to-Real Reality Gap"
occurs at the cyber-physical actuation boundary:
    1. Unconstrained Continuous Velocity Inference: Neural network actors output continuous
       velocities (v, omega) assuming ideal holonomic acceleration or instantaneous velocity
       tracking. In reality, differential drive mobile robots are governed by non-holonomic
       kinematic constraints (Pfaffian constraint: dy*cos(theta) - dx*sin(theta) = 0, no lateral slip).
    2. Actuator Non-Linearities & Low-Level Electrical Transients: Direct conversion from
       desired wheel velocity to voltage/PWM exhibits deadbands (static Coulomb friction stiction),
       inductive current spikes during rapid acceleration (which brown out 5V edge logic buses),
       and wheel slippage (violating dead-reckoning odometry).

OUR SOLUTION (The EOS Inverse Kinematics & Actuator Bridge):
--------------------------------------------------------------------------------
This ROS 2 Humble node closes the cyber-physical loop between the high-level PPO DRL
navigator and the low-level Arduino Mega embedded motor microcontroller:
    1. Non-Holonomic Differential Drive Inverse Kinematics:
       Converts Cartesian twist commands (v, omega) into individual left and right wheel
       angular velocities (omega_L, omega_R) parameterized by wheelbase L and wheel radius R:
           omega_R = (v + (omega * L) / 2.0) / R
           omega_L = (v - (omega * L) / 2.0) / R
    2. Actuator Non-Linearity Compensation (Deadband & 8-Bit PWM Mapping):
       Maps theoretical wheel speeds into signed 8-bit pulse-width modulation integers
       [-255, 255] for the Arduino's hardware timers, incorporating empirical static
       friction deadband compensation:
           PWM = sgn(omega) * [PWM_min + (|omega| / omega_max) * (255 - PWM_min)]
    3. Slew-Rate Limiting & Safety Watchdog:
       Enforces hardware-safe acceleration ramps (slew-rate limiting) to prevent wheel
       slip and back-EMF resets, backed by a 500ms heartbeat watchdog that halts the
       motors if the DRL neural pipeline suffers inference stalls.
================================================================================
"""

import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from std_msgs.msg import Int16MultiArray, MultiArrayDimension, MultiArrayLayout
from std_msgs.msg import Float32MultiArray


class DifferentialDriveKinematicsController(Node):
    """
    ROS 2 Humble Node translating DRL Twist commands into 8-bit Arduino PWM signals
    via differential drive inverse kinematics.
    """
    def __init__(self):
        super().__init__('kinematics_controller')

        # ----------------------------------------------------------------------
        # Robot Physical Parameters & Actuator Limits
        # ----------------------------------------------------------------------
        self.declare_parameter('wheelbase_meters', 0.20)         # L: Track width between wheel centers
        self.declare_parameter('wheel_radius_meters', 0.033)      # R: Wheel radius (33mm standard)
        self.declare_parameter('max_linear_velocity', 0.5)        # v_max: 0.5 m/s
        self.declare_parameter('max_angular_velocity', 1.0)       # omega_max: 1.0 rad/s
        self.declare_parameter('pwm_deadband_min', 45)            # Minimum PWM required to overcome stiction
        self.declare_parameter('max_pwm_value', 255)              # Maximum 8-bit PWM limit (Timer counter)
        self.declare_parameter('max_acceleration_pwm_per_sec', 500.0)  # Slew-rate acceleration cap
        self.declare_parameter('watchdog_timeout_sec', 0.5)       # Failsafe timeout
        self.declare_parameter('cmd_vel_topic', '/drl/cmd_vel')
        self.declare_parameter('motor_pwm_topic', '/arduino/motor_pwm')
        self.declare_parameter('wheel_speeds_topic', '/kinematics/wheel_speeds_rad_s')

        # Read resolved parameter values
        self.L = self.get_parameter('wheelbase_meters').value
        self.R = self.get_parameter('wheel_radius_meters').value
        self.v_max = self.get_parameter('max_linear_velocity').value
        self.omega_max = self.get_parameter('max_angular_velocity').value
        self.pwm_min = self.get_parameter('pwm_deadband_min').value
        self.pwm_max = self.get_parameter('max_pwm_value').value
        self.max_pwm_rate = self.get_parameter('max_acceleration_pwm_per_sec').value
        self.watchdog_timeout = self.get_parameter('watchdog_timeout_sec').value
        self.cmd_topic = self.get_parameter('cmd_vel_topic').value
        self.pwm_topic = self.get_parameter('motor_pwm_topic').value
        self.speeds_topic = self.get_parameter('wheel_speeds_topic').value

        # Calculate maximum physical wheel angular velocity (rad/s)
        self.max_wheel_omega = self.v_max / self.R

        # State tracking for slew-rate limiter
        self.current_left_pwm = 0.0
        self.current_right_pwm = 0.0
        self.last_control_time = self.get_clock().now()
        self.last_cmd_vel_time = self.get_clock().now()

        # ----------------------------------------------------------------------
        # ROS 2 Communication Interfaces
        # ----------------------------------------------------------------------
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber to DRL velocity commands
        self.cmd_sub = self.create_subscription(
            Twist,
            self.cmd_topic,
            self.cmd_vel_callback,
            cmd_qos
        )

        # Publisher for signed 8-bit PWM to Arduino Mega [left_pwm, right_pwm]
        self.pwm_pub = self.create_publisher(Int16MultiArray, self.pwm_topic, 10)

        # Telemetry publisher for wheel angular velocities [omega_left, omega_right] in rad/s
        self.wheel_speed_pub = self.create_publisher(Float32MultiArray, self.speeds_topic, 10)

        # Control & Watchdog Timer: 50 Hz control loop (20 ms period)
        self.control_timer = self.create_timer(0.02, self.control_loop_callback)

        self.get_logger().info(
            f"[KINEMATICS_INIT] Controller started with Wheelbase L={self.L}m, "
            f"Radius R={self.R}m, Max Wheel Speed={self.max_wheel_omega:.2f} rad/s."
        )

    def cmd_vel_callback(self, msg: Twist):
        """
        Ingests continuous velocity commands generated by the PPO DRL Navigator.
        Updates internal target velocities and resets the watchdog heartbeat timer.
        """
        self.last_cmd_vel_time = self.get_clock().now()

        # Clamp command velocities to robot dynamic safety boundaries
        v = float(np.clip(msg.linear.x, 0.0, self.v_max))
        omega = float(np.clip(msg.angular.z, -self.omega_max, self.omega_max))

        # ----------------------------------------------------------------------
        # INVERSE KINEMATICS FORMULATION
        # ----------------------------------------------------------------------
        # omega_R = (v + (omega * L) / 2) / R
        # omega_L = (v - (omega * L) / 2) / R
        omega_r = (v + (omega * self.L) / 2.0) / self.R
        omega_l = (v - (omega * self.L) / 2.0) / self.R

        # Compute raw target PWM values
        self.target_left_pwm = self._omega_to_pwm(omega_l)
        self.target_right_pwm = self._omega_to_pwm(omega_r)

        # Publish telemetry: wheel speeds in rad/s
        speeds_msg = Float32MultiArray()
        speeds_msg.data = [float(omega_l), float(omega_r)]
        self.wheel_speed_pub.publish(speeds_msg)

    def _omega_to_pwm(self, omega_wheel: float) -> float:
        """
        Transforms wheel angular velocity (rad/s) to signed 8-bit PWM [-255, 255].
        Incorporates deadband compensation to overcome static friction stiction.
        
        Mathematical Formulation:
            PWM = sgn(omega) * [PWM_min + (|omega| / omega_max) * (255 - PWM_min)]
        """
        if abs(omega_wheel) < 1e-4:
            return 0.0

        sign = 1.0 if omega_wheel > 0 else -1.0
        abs_omega = abs(omega_wheel)

        # Normalize relative to maximum possible wheel speed
        ratio = min(abs_omega / self.max_wheel_omega, 1.0)

        # Interpolate between minimum stiction overcoming PWM and 255
        pwm = sign * (self.pwm_min + ratio * (self.pwm_max - self.pwm_min))
        return float(np.clip(pwm, -self.pwm_max, self.pwm_max))

    def control_loop_callback(self):
        """
        High-frequency periodic control loop (50 Hz).
        Applies watchdog safety validation and slew-rate acceleration limits.
        """
        now = self.get_clock().now()
        dt = (now - self.last_control_time).nanoseconds * 1e-9
        self.last_control_time = now

        # ----------------------------------------------------------------------
        # WATCHDOG FAILSAFE VERIFICATION
        # ----------------------------------------------------------------------
        # If DRL inference stalls or ROS node crashes, command emergency stop.
        time_since_last_cmd = (now - self.last_cmd_vel_time).nanoseconds * 1e-9
        if time_since_last_cmd > self.watchdog_timeout:
            target_l = 0.0
            target_r = 0.0
        else:
            target_l = getattr(self, 'target_left_pwm', 0.0)
            target_r = getattr(self, 'target_right_pwm', 0.0)

        # ----------------------------------------------------------------------
        # SLEW-RATE ACCELERATION LIMITING
        # ----------------------------------------------------------------------
        # delta_PWM_max = max_pwm_rate * dt
        # Prevents high inductive back-EMF spikes and eliminates wheel slippage.
        max_delta = self.max_pwm_rate * dt

        delta_l = np.clip(target_l - self.current_left_pwm, -max_delta, max_delta)
        delta_r = np.clip(target_r - self.current_right_pwm, -max_delta, max_delta)

        self.current_left_pwm += delta_l
        self.current_right_pwm += delta_r

        # Integer round for 8-bit microcontroller registers
        pwm_l_int = int(round(self.current_left_pwm))
        pwm_r_int = int(round(self.current_right_pwm))

        # Clamp strictly to signed 8-bit integer interval [-255, 255]
        pwm_l_int = max(-self.pwm_max, min(self.pwm_max, pwm_l_int))
        pwm_r_int = max(-self.pwm_max, min(self.pwm_max, pwm_r_int))

        # ----------------------------------------------------------------------
        # PUBLISH ACTUATION TOPIC TO ARDUINO MEGA
        # ----------------------------------------------------------------------
        pwm_msg = Int16MultiArray()
        pwm_msg.layout.dim = [
            MultiArrayDimension(label='motors', size=2, stride=2)
        ]
        # Data format: [Left_PWM, Right_PWM]
        # Positive PWM = Forward rotation, Negative PWM = Reverse rotation
        pwm_msg.data = [pwm_l_int, pwm_r_int]
        self.pwm_pub.publish(pwm_msg)


# ==============================================================================
# ENTRYPOINT
# ==============================================================================

def main(args=None):
    rclpy.init(args=args)
    node = DifferentialDriveKinematicsController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("[KINEMATICS_SHUTDOWN] Controller shutting down safely.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
