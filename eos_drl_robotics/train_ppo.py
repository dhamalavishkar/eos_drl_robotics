#!/usr/bin/env python3
"""
================================================================================
Project: Edge-Optimized Sensor Fusion (EOS-DRL)
Module: PPO Training Pipeline (train_ppo.py)
================================================================================
Trains the ActorCritic agent on the DynamicNavigationEnv using:
- Generalized Advantage Estimation (GAE)
- Clipped surrogate objective
- Potential-based A* waypoint reward shaping (+0.1 closer, -1.0 collision, +1.0 goal)
- Model checkpointing for edge deployment on NVIDIA Jetson Nano
================================================================================
"""

import os
import time
import numpy as np
import torch

from drl_navigator import ActorCritic, PPONavigatorAgent
from reward_environment import DynamicNavigationEnv

def train(
    total_timesteps: int = 20000,
    rollout_steps: int = 512,
    epochs_per_update: int = 4,
    batch_size: int = 64,
    save_path: str = "ppo_eos_checkpoint.pt"
):
    print("=" * 80)
    print("STARTING EOS-DRL PPO TRAINING PIPELINE")
    print(f"Total Timesteps: {total_timesteps} | Rollout Buffer: {rollout_steps} steps")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[TRAIN] Active PyTorch Device: {device.upper()}")

    # 1. Instantiate Environment & Model
    env = DynamicNavigationEnv(grid_dim=64, dt=0.1, max_episode_steps=200)
    actor_critic = ActorCritic(
        grid_channels=3,
        guidance_dim=1,
        max_linear_vel=0.5,
        max_angular_vel=1.0
    ).to(device)

    agent = PPONavigatorAgent(
        actor_critic=actor_critic,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        device=device
    )

    obs, info = env.reset()
    episode_reward = 0.0
    episode_count = 0
    episodes_success = 0
    timestep = 0

    while timestep < total_timesteps:
        # Rollout Storage Buffers
        buf_grids = []
        buf_guidance = []
        buf_actions = []
        buf_log_probs = []
        buf_rewards = []
        buf_values = []
        buf_dones = []

        # 2. Collect Rollout Trajectory
        for step in range(rollout_steps):
            grid_np = obs['grid']
            dist_np = obs['distance_to_waypoint']

            grid_t = torch.from_numpy(grid_np).unsqueeze(0).to(device)
            dist_t = torch.from_numpy(dist_np).to(device)

            action, log_prob, value = actor_critic.act(grid_t, dist_t, deterministic=False)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            buf_grids.append(torch.from_numpy(grid_np))
            buf_guidance.append(torch.from_numpy(dist_np))
            buf_actions.append(torch.from_numpy(action))
            buf_log_probs.append(log_prob)
            buf_rewards.append(reward)
            buf_values.append(value)
            buf_dones.append(float(done))

            episode_reward += reward
            timestep += 1
            obs = next_obs

            if done:
                episode_count += 1
                if reward >= 1.0:
                    episodes_success += 1
                obs, info = env.reset()
                episode_reward = 0.0

            if timestep >= total_timesteps:
                break

        # 3. Bootstrap Value for GAE
        with torch.inference_mode():
            last_grid = torch.from_numpy(obs['grid']).unsqueeze(0).to(device)
            last_dist = torch.from_numpy(obs['distance_to_waypoint']).to(device)
            _, _, last_val = actor_critic.act(last_grid, last_dist, deterministic=True)
            next_val_t = torch.tensor(last_val, device=device)

        # 4. Prepare Batch Tensors
        t_grids = torch.stack(buf_grids)
        t_guidance = torch.stack(buf_guidance)
        t_actions = torch.stack(buf_actions)
        t_old_log_probs = torch.tensor(buf_log_probs, dtype=torch.float32, device=device)
        t_rewards = torch.tensor(buf_rewards, dtype=torch.float32, device=device)
        t_values = torch.tensor(buf_values, dtype=torch.float32, device=device)
        t_dones = torch.tensor(buf_dones, dtype=torch.float32, device=device)

        # 5. Compute Generalized Advantage Estimation
        advantages, returns = agent.compute_gae(t_rewards, t_values, t_dones, next_val_t)

        # 6. Mini-batch PPO Policy Update
        loss_dict = agent.update_policy(
            b_grids=t_grids,
            b_guidance=t_guidance,
            b_actions=t_actions,
            b_old_log_probs=t_old_log_probs,
            b_advantages=advantages,
            b_returns=returns,
            epochs=epochs_per_update,
            batch_size=batch_size
        )

        success_rate = (episodes_success / max(1, episode_count)) * 100.0
        print(
            f"[TIMESTEP {timestep:>6}/{total_timesteps}] | "
            f"Policy Loss: {loss_dict['policy_loss']:+.4f} | "
            f"Value Loss: {loss_dict['value_loss']:.4f} | "
            f"Entropy: {loss_dict['entropy']:.3f} | "
            f"Episodes: {episode_count} | Success Rate: {success_rate:.1f}%"
        )

    # 7. Save Checkpoint for Jetson Edge Deployment
    checkpoint = {
        'model_state_dict': actor_critic.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'hyperparameters': {
            'grid_channels': 3,
            'guidance_dim': 1,
            'max_linear_vel': 0.5,
            'max_angular_vel': 1.0
        }
    }
    torch.save(checkpoint, save_path)
    print("=" * 80)
    print(f"[SUCCESS] Trained policy checkpoint saved to: {os.path.abspath(save_path)}")
    print("=" * 80)

if __name__ == '__main__':
    train(total_timesteps=1500, rollout_steps=256)
