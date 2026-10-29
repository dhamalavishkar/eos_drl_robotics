#!/usr/bin/env python3
"""
================================================================================
Project: Edge-Optimized Sensor Fusion: A Unified SLAM-DRL Framework for Autonomous Service Robots
Module: Deep Reinforcement Learning Local Navigator (drl_navigator.py)
Target Hardware: NVIDIA Jetson Nano (Edge GPU) / Real-Time Controller
Architecture: Asymmetric Actor-Critic with Multimodal Sensor Fusion (PPO)

ACADEMIC RESEARCH GAP & METHODOLOGICAL RATIONALE:
--------------------------------------------------------------------------------
Standard robotic navigation paradigms suffer from a dichotomy:
    1. Classical Hierarchical Planners (e.g., Global A* / Dijkstra + Local DWA / TEB):
       While globally optimal and computationally predictable on static maps, local
       trajectory planners rely on hand-tuned geometric costmaps. They struggle with
       dense, stochastic dynamic obstacles (e.g., moving pedestrians), frequently
       getting trapped in oscillatory freeze-robot conditions or local minima.
    2. Pure End-to-End Deep Reinforcement Learning (DRL):
       Directly mapping raw sensory inputs (camera/LiDAR) to continuous motor actions
       suffers from extreme sample inefficiency and policy degeneration due to sparse
       rewards over long geometric horizons. Without global guidance, DRL agents exhibit
       myopic behavior, wandering aimlessly in complex multi-room indoor environments.

OUR SOLUTION (The Hybrid A* + PPO Local Reactive Navigator):
--------------------------------------------------------------------------------
This module implements the PPO Actor-Critic neural network architecture within our
hybrid framework:
    - Global Path Grounding: A static A* planner computes topologically valid global
      waypoints. The network is conditioned on the scalar Euclidean distance to the
      immediate next A* waypoint (and optional target heading vector).
    - Local Dynamic Obstacle Evasion: The network simultaneously digests the 64x64
      2D semantic occupancy grid produced by our MobileNetV3 perception node.
    - Asymmetric Multimodal Sensor Fusion: Spatial grid features are compressed via
      lightweight strided convolutional filters and fused with the waypoint vector.
    - Continuous Stochastic Policy: The Actor outputs parameters for continuous
      Gaussian distributions over linear velocity v in [0, v_max] and angular velocity
      omega in [-omega_max, omega_max].
    - Scaled Critic Head: Predicts state value V(s) for Generalized Advantage
      Estimation (GAE) to stabilize gradient updates on resource-constrained edge hardware.
================================================================================
"""

import math
from typing import Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# ==============================================================================
# SECTION 1: MULTIMODAL FEATURE EXTRACTOR BACKBONE
# ==============================================================================

class SpatialGridEncoder(nn.Module):
    """
    Lightweight 2D CNN feature extractor for 64x64 Semantic Occupancy Grids.
    Designed with shallow depth and low channel counts to ensure sub-5ms forward
    passes on Jetson Nano edge GPUs.
    
    Input: [Batch, In_Channels, 64, 64]
           In_Channels = 3 (Channel 0: Free, Channel 1: Static, Channel 2: Dynamic)
    Output: [Batch, Latent_Dim=128]
    """
    def __init__(self, in_channels: int = 3, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            # Stage 1: 64x64 -> 32x32
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),

            # Stage 2: 32x32 -> 16x16
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),

            # Stage 3: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),

            # Stage 4: Global Adaptive Pooling -> [Batch, 64, 1, 1]
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.encoder(grid)


class WaypointGuidanceMLP(nn.Module):
    """
    Multi-Layer Perceptron embedding scalar distance (and optional heading error)
    to the immediate A* waypoint into a high-dimensional conditioning vector.
    
    Input: [Batch, Guidance_Dim] (typically 1 for scalar distance, or 2 for [dist, heading])
    Output: [Batch, Latent_Dim=32]
    """
    def __init__(self, guidance_dim: int = 1, latent_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(guidance_dim, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, guidance: torch.Tensor) -> torch.Tensor:
        return self.mlp(guidance)


# ==============================================================================
# SECTION 2: ACTOR-CRITIC MULTIMODAL POLICY NETWORK
# ==============================================================================

class ActorCritic(nn.Module):
    """
    Unified Actor-Critic Architecture for PPO (Proximal Policy Optimization).
    
    State Representation:
        s = { Occupancy Grid G in R^(3 x 64 x 64), Waypoint Distance d in R^1 }
    Action Space:
        a = [v, omega] in Continuous R^2
        v in [0.0, v_max] (Forward linear velocity, non-holonomic constraint)
        omega in [-omega_max, omega_max] (Angular yaw velocity)
    """
    def __init__(
        self,
        grid_channels: int = 3,
        guidance_dim: int = 1,
        grid_latent_dim: int = 128,
        guidance_latent_dim: int = 32,
        fusion_dim: int = 128,
        action_dim: int = 2,
        max_linear_vel: float = 0.5,
        max_angular_vel: float = 1.0,
        init_log_std: float = -0.5
    ):
        super().__init__()
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.action_dim = action_dim

        # 1. Feature Extractors
        self.grid_encoder = SpatialGridEncoder(in_channels=grid_channels, latent_dim=grid_latent_dim)
        self.waypoint_mlp = WaypointGuidanceMLP(guidance_dim=guidance_dim, latent_dim=guidance_latent_dim)

        # 2. Multimodal Early Fusion Layer
        total_latent = grid_latent_dim + guidance_latent_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(total_latent, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        # 3. Policy Head (Actor): Outputs Mean [mu_v, mu_omega]
        self.actor_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, action_dim)
        )

        # Learnable log standard deviation for continuous stochastic exploration
        # Log-std is state-independent to guarantee training stability under PPO
        self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * init_log_std)

        # 4. Value Head (Critic): Outputs Scalar State Value V(s)
        self.critic_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 1)
        )

        # Orthogonal weight initialization for faster reinforcement learning convergence
        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization stabilizes gradient flow in deep RL pipelines."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2.0))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        # Scale down output layers for initial actions near zero mean
        nn.init.orthogonal_(self.actor_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head[-1].weight, gain=1.0)

    def extract_fused_features(self, grid: torch.Tensor, guidance: torch.Tensor) -> torch.Tensor:
        """
        Extracts and concatenates spatial BEV features with waypoint scalar guidance.
        """
        # Ensure dimensions match [Batch, ...]
        if grid.dim() == 3:
            grid = grid.unsqueeze(0)
        if guidance.dim() == 1:
            guidance = guidance.unsqueeze(1)
        elif guidance.dim() == 0:
            guidance = guidance.view(1, 1)

        z_grid = self.grid_encoder(grid)
        z_wp = self.waypoint_mlp(guidance)
        z_concat = torch.cat([z_grid, z_wp], dim=-1)
        return self.fusion_layer(z_concat)

    def get_action_distribution(self, fused_features: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Computes Gaussian distribution over normalized action space.
        Applies bounding squashing functions:
            mu_linear in [0.0, max_linear_vel] via Sigmoid
            mu_angular in [-max_angular_vel, max_angular_vel] via Tanh
        """
        raw_mu = self.actor_head(fused_features)

        # Scaled mean activations satisfying non-holonomic differential drive bounds
        mu_linear = torch.sigmoid(raw_mu[:, 0:1]) * self.max_linear_vel
        mu_angular = torch.tanh(raw_mu[:, 1:2]) * self.max_angular_vel
        action_mean = torch.cat([mu_linear, mu_angular], dim=-1)

        # Clamp standard deviation to prevent numerical instability or policy explosion
        log_std = torch.clamp(self.actor_log_std, min=-5.0, max=1.0)
        action_std = log_std.exp().expand_as(action_mean)

        dist = Normal(action_mean, action_std)
        return dist, action_mean

    def forward(self, grid: torch.Tensor, guidance: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Forward pass returning action distribution and state-value V(s).
        """
        fused = self.extract_fused_features(grid, guidance)
        dist, _ = self.get_action_distribution(fused)
        value = self.critic_head(fused)
        return dist, value

    def act(
        self,
        grid: torch.Tensor,
        guidance: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, float, float]:
        """
        Inference execution for real-time edge robot navigation.
        
        Args:
            grid: PyTorch tensor or numpy array [3, 64, 64]
            guidance: Scalar float distance to A* waypoint
            deterministic: If True, uses mean (evaluation mode); if False, samples.
            
        Returns:
            action: NumPy array [v, omega]
            log_prob: Float scalar log probability of selected action
            value: Float scalar value estimation V(s)
        """
        with torch.inference_mode():
            if isinstance(grid, np.ndarray):
                grid = torch.from_numpy(grid).float()
            if isinstance(guidance, (float, int, np.ndarray)):
                guidance = torch.as_tensor(guidance).float()

            device = next(self.parameters()).device
            grid = grid.to(device)
            guidance = guidance.to(device)

            fused = self.extract_fused_features(grid, guidance)
            dist, action_mean = self.get_action_distribution(fused)
            value = self.critic_head(fused)

            if deterministic:
                action = action_mean
            else:
                action = dist.sample()

            # Bound final sampled continuous velocities
            action[:, 0] = torch.clamp(action[:, 0], 0.0, self.max_linear_vel)
            action[:, 1] = torch.clamp(action[:, 1], -self.max_angular_vel, self.max_angular_vel)

            # Sum log probabilities across action dimensions
            log_prob = dist.log_prob(action).sum(dim=-1)

            return (
                action.squeeze(0).cpu().numpy(),
                log_prob.item(),
                value.squeeze(0).item()
            )

    def evaluate_actions(
        self,
        grids: torch.Tensor,
        guidance: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluates batch transitions during PPO policy optimization steps.
        
        Returns:
            log_probs: [Batch] Action log probabilities under current policy
            values: [Batch] Value function predictions
            entropy: [Batch] Policy distribution entropy (exploration bonus)
        """
        fused = self.extract_fused_features(grids, guidance)
        dist, _ = self.get_action_distribution(fused)
        values = self.critic_head(fused).squeeze(-1)

        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, values, entropy


# ==============================================================================
# SECTION 3: PPO AGENT OPTIMIZATION WRAPPER
# ==============================================================================

class PPONavigatorAgent:
    """
    PPO Algorithm Manager implementing Clipped Surrogate Objective, Generalized
    Advantage Estimation (GAE), and Edge Inference Optimization.
    """
    def __init__(
        self,
        actor_critic: ActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = torch.device(device)
        self.policy = actor_critic.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes Generalized Advantage Estimations (Schulman et al., 2015).
        Reduces variance while preserving low bias in policy gradient updates.
        
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
        """
        num_steps = len(rewards)
        advantages = torch.zeros(num_steps, device=self.device)
        last_gae = 0.0

        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update_policy(
        self,
        b_grids: torch.Tensor,
        b_guidance: torch.Tensor,
        b_actions: torch.Tensor,
        b_old_log_probs: torch.Tensor,
        b_advantages: torch.Tensor,
        b_returns: torch.Tensor,
        epochs: int = 4,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Performs mini-batch clipped PPO gradient descent.
        
        L^{CLIP}(theta) = E [ min( r_t(theta) * A_t, clip(r_t(theta), 1-eps, 1+eps) * A_t ) ]
        L^{TOTAL}(theta) = - L^{CLIP}(theta) + c_1 * L^{VF}(theta) - c_2 * S[pi_theta](s_t)
        """
        dataset_size = b_grids.size(0)
        # Normalize advantages across batch to stabilize gradient step size
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        update_steps = 0

        for _ in range(epochs):
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                mb_grids = b_grids[batch_idx].to(self.device)
                mb_guidance = b_guidance[batch_idx].to(self.device)
                mb_actions = b_actions[batch_idx].to(self.device)
                mb_old_log_probs = b_old_log_probs[batch_idx].to(self.device)
                mb_advantages = b_advantages[batch_idx].to(self.device)
                mb_returns = b_returns[batch_idx].to(self.device)

                log_probs, values, entropy = self.policy.evaluate_actions(
                    mb_grids, mb_guidance, mb_actions
                )

                # Importance Sampling Ratio: r_t(theta) = exp(log_prob - old_log_prob)
                ratios = torch.exp(log_probs - mb_old_log_probs)

                # Clipped Surrogate Objective
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Function Loss (Mean Squared Error)
                value_loss = 0.5 * F.mse_loss(values, mb_returns)

                # Total Loss with Entropy Regularization
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()

                # Optimization Step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                update_steps += 1

        return {
            'policy_loss': total_policy_loss / max(1, update_steps),
            'value_loss': total_value_loss / max(1, update_steps),
            'entropy': total_entropy / max(1, update_steps)
        }


# ==============================================================================
# SECTION 4: UNIT VERIFICATION (EDGE BENCHMARKING)
# ==============================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("Testing DRL Navigator Network Architecture on Edge Benchmark")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Initialize network
    actor_critic = ActorCritic(
        grid_channels=3,
        guidance_dim=1,
        max_linear_vel=0.5,
        max_angular_vel=1.0
    ).to(device)

    # Synthetic observation matching perception_node output: [Batch=1, Channels=3, H=64, W=64]
    dummy_grid = torch.randn(1, 3, 64, 64, device=device)
    dummy_dist = torch.tensor([[1.45]], device=device)  # 1.45 meters to immediate A* waypoint

    # Forward act benchmark
    action, log_prob, val = actor_critic.act(dummy_grid, dummy_dist, deterministic=False)
    print(f"Sampled Action: Linear Velocity v = {action[0]:.3f} m/s, Angular Velocity omega = {action[1]:.3f} rad/s")
    print(f"Action Log-Probability: {log_prob:.4f} | State Value Estimation V(s): {val:.4f}")

    # Latency test
    import time
    iterations = 100
    start = time.time()
    for _ in range(iterations):
        _ = actor_critic.act(dummy_grid, dummy_dist, deterministic=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    latency_ms = ((time.time() - start) / iterations) * 1000.0
    print(f"Average Forward Inference Latency: {latency_ms:.2f} ms ({1000.0/latency_ms:.1f} FPS)")
    print("=" * 70)
