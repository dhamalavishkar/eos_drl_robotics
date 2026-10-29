#!/usr/bin/env python3
"""
================================================================================
Project: Edge-Optimized Sensor Fusion (EOS-DRL)
Module: End-to-End Hardware-in-the-Loop Simulation Runner (simulate_pipeline.py)
================================================================================
This script integrates and executes the entire closed-loop pipeline:
1. Environment generates local 64x64 BEV occupancy grids and A* waypoint guidance.
2. DRL Actor-Critic network predicts continuous velocities (v, omega).
3. Differential Drive Inverse Kinematics maps velocities to left/right 8-bit PWM signals.
4. Simulates a complete navigation episode with obstacle evasion and waypoint tracking.
"""

import time
import numpy as np
import torch

from drl_navigator import ActorCritic
from reward_environment import DynamicNavigationEnv

def run_simulation(num_steps: int = 25):
    print("=" * 80)
    print("EOS-DRL PIPELINE: CLOSED-LOOP SIMULATION RUN")
    print("Target: NVIDIA Jetson Nano GPU / Differential Drive Service Robot")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[SYSTEM] Neural Compute Backend: {device.upper()}")

    # 1. Initialize Gymnasium Environment
    env = DynamicNavigationEnv(grid_dim=64, dt=0.1)
    obs, info = env.reset()
    print(f"[PLANNER] Global A* Path generated {info['waypoints_total']} waypoints.")

    # 2. Initialize DRL Actor-Critic Network
    navigator = ActorCritic(
        grid_channels=3,
        guidance_dim=1,
        max_linear_vel=0.5,
        max_angular_vel=1.0
    ).to(device)
    navigator.eval()

    # 3. Kinematic Parameters (Matching kinematics_controller.py)
    L = 0.20        # Wheelbase: 20 cm
    R = 0.033       # Wheel radius: 3.3 cm
    max_wheel_omega = 0.5 / R
    pwm_min = 45
    pwm_max = 255

    total_reward = 0.0
    print("-" * 80)
    print(f"{'Step':<5} | {'Dist to WP':<11} | {'Twist (v, w)':<18} | {'PWM [L, R]':<14} | {'Reward':<8} | {'Status'}")
    print("-" * 80)

    start_sim_time = time.perf_counter()

    for step in range(1, num_steps + 1):
        grid_tensor = torch.from_numpy(obs['grid']).unsqueeze(0).to(device)
        dist_scalar = torch.from_numpy(obs['distance_to_waypoint']).to(device)

        # A) Neural DRL Inference
        action, log_prob, val = navigator.act(grid_tensor, dist_scalar, deterministic=True)
        v, omega = float(action[0]), float(action[1])

        # B) Differential Drive Inverse Kinematics
        omega_r = (v + (omega * L) / 2.0) / R
        omega_l = (v - (omega * L) / 2.0) / R

        # C) PWM Mapping with Deadband Compensation
        def omega_to_pwm(w):
            if abs(w) < 1e-4: return 0
            sign = 1 if w > 0 else -1
            ratio = min(abs(w) / max_wheel_omega, 1.0)
            return int(round(sign * (pwm_min + ratio * (pwm_max - pwm_min))))

        pwm_l = np.clip(omega_to_pwm(omega_l), -pwm_max, pwm_max)
        pwm_r = np.clip(omega_to_pwm(omega_r), -pwm_max, pwm_max)

        # D) Environment Step
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward

        dist_str = f"{obs['distance_to_waypoint'][0]:.2f}m"
        twist_str = f"({v:.2f}m/s, {omega:+.2f}rad/s)"
        pwm_str = f"[{pwm_l:+4d}, {pwm_r:+4d}]"
        status_str = "COLLISION" if step_info['is_collision'] else ("GOAL" if terminated else "RUNNING")

        print(f"{step:<5} | {dist_str:<11} | {twist_str:<18} | {pwm_str:<14} | {reward:+.2f}    | {status_str}")

        if terminated or truncated:
            break

    elapsed_ms = (time.perf_counter() - start_sim_time) * 1000.0
    print("-" * 80)
    print(f"[SUMMARY] Total Reward: {total_reward:+.2f} | Steps Completed: {step} | Total Elapsed Time: {elapsed_ms:.1f} ms")
    print(f"[SUMMARY] Average Pipeline Rate: {step / (elapsed_ms / 1000.0):.1f} Hz (Real-time target >= 10 Hz)")
    print("=" * 80)

if __name__ == '__main__':
    run_simulation()
