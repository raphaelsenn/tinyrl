from abc import ABC, abstractmethod
from typing import Dict, Any

import os
import time
import random
import copy

import torch
import numpy as np
import gymnasium as gym

from tinyrl.networks.base import Actor, Critic
from tinyrl.stats.evaluation_stats import EvluationStats
from tinyrl.buffers.replay_buffer import ReplayBuffer
from tinyrl.buffers.rollout_buffer import RolloutBuffer


DEFAULT_AC_KWARGS = {
    "h1_dim" : 256,
    "h2_dim" : 256,
}


class RLAlgorithm(ABC):
    """Base actor-critic algorithm class.""" 
    def __init__(
        self,
        env: gym.Env,
        *,
        ac_kwargs: Dict[str, Any] = DEFAULT_AC_KWARGS,
        gamma: float,
        tau_polyak: float,
        save_every: int, 
        eval_every: int, 
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        ...
        super().__init__()

        self.env = env
        self.val_env = copy.deepcopy(env)

        # Computing device
        assert device in {"cpu", "cuda", "mps"}, (
            f"Unkdown device, expected device in [`cpu`, `cuda`, `mps`], got: {device}."
        )
        self.device = torch.device(device)

        # Actor/critic models
        self.ac_kwargs = ac_kwargs
        self.actor: Actor = ...
        self.critic: Critic = ...

        # Environment
        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.obs_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = int(np.prod(self.action_shape))

        self.action_low_np = np.asarray(self.env.action_space.low, dtype=np.float32)
        self.action_high_np = np.asanyarray(self.env.action_space.high, dtype=np.float32)
        self.action_scale_np = (self.action_high_np - self.action_low_np) / 2.0
        self.action_bias_np = (self.action_high_np + self.action_low_np) / 2.0
        
        self.action_low_torch = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=self.device)
        self.action_high_torch = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=self.device)
        self.action_scale_torch = torch.as_tensor(self.action_scale_np, dtype=torch.float32, device=self.device)
        self.action_bias_torch = torch.as_tensor(self.action_bias_np, dtype=torch.float32, device=self.device)

        # Hyperparameters and save/eval settings
        self.gamma = gamma
        self.tau_polyak = tau_polyak
        self.save_every = save_every
        self.eval_every = eval_every 
        self.verbose = verbose
        
        # Counters
        self.episode_counter = 0
        self.grad_step_counter = 0
        self.global_step_counter = 0

        # Evaluation
        self.eval_stats = EvluationStats()

        # Reproducability
        self.seed = seed 
        self.torch_rng = torch.Generator(device=self.device)
        self.set_seeds()

    @abstractmethod
    def act(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def collect_rollouts(self) -> tuple[np.ndarray, Dict[str, Any]]:
        raise NotImplementedError

    @torch.no_grad()
    def act_numpy(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)   # [obs_dim]
        obs_t = obs_t.unsqueeze(0)                                              # [1, obs_dim]
        action = self.act(obs_t, deterministic)                                 # [1, action_dim]
        action = action.flatten().cpu().numpy()                                 # [action_dim]
        return action

    @torch.no_grad()
    def evaluate(self, n_eval_episodes: int = 10) -> None:
        env = self.val_env
        episode_rewards = []

        for episode in range(n_eval_episodes):
            eval_seed = self.seed + episode + 1
            obs, _ = env.reset(seed=eval_seed)

            done = False
            episode_reward = 0.0

            while not done:
                action = self.act_numpy(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)

                episode_reward += float(reward)
                done = terminated or truncated

            episode_rewards.append(episode_reward)

        self.eval_stats.update(rewards=np.asarray(episode_rewards, dtype=np.float32))

    def set_seeds(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.torch_rng.manual_seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def print_stats(self) -> None:
        report = (
            f"Timestep: {self.global_step_counter:>9d}  "
            f"Episode: {self.episode_counter:>6d}  "
            f"Average Return: {self.eval_stats.last_average_reward:>12.4f}  "
            f"Std Return: {self.eval_stats.last_std_reward:>12.4f}  "
        )

        print(report)

    def update(self) -> None:
        if self.global_step_counter % self.eval_every == 0:
            self.evaluate()

            if self.verbose is True:
                self.print_stats()

        if self.global_step_counter % self.save_every == 0:
            self.save()

    def save(self) -> None:
        algo = self.__class__.__name__.lower()
        env = self.env.spec.id.lower()
        run = f"{env}-{algo}-lr_pi{self.lr_actor}-lr_q{self.lr_critic}-seed{self.seed}"
        path = os.path.join("checkpoints", algo, run)
        os.makedirs(path, exist_ok=True)

        torch.save(self.actor.state_dict(), os.path.join(path, f"actor-{self.global_step_counter}.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, f"critic-{self.global_step_counter}.pt"))
        self.eval_stats.to_csv(os.path.join(path, "evaluation.csv"))


class OffPolicyAlgorithm(RLAlgorithm, ABC):
    def __init__(
        self,
        env: gym.Env,
        *,
        ac_kwargs: Dict[str, Any],
        gamma: float,
        tau_polyak: float,
        buffer_size: int,
        buffer_start_size: int,
        batch_size: int,
        gradient_steps: int,
        save_every: int,
        eval_every: int,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            env=env,
            ac_kwargs=ac_kwargs,
            gamma=gamma,
            tau_polyak=tau_polyak,
            save_every=save_every,
            eval_every=eval_every,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        self.buffer_size = buffer_size
        self.buffer_start_size = buffer_start_size
        self.batch_size = batch_size
        self.gradient_steps = gradient_steps

        self.replay_buffer = ReplayBuffer(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            capacity=buffer_size,
            batch_size=batch_size,
            device=device,
        )

    def collect_rollouts(self) -> tuple[np.ndarray, Dict[str, Any]]:
        env = self.env  # NOTE: Training environment

        env.action_space.seed(self.seed)
        obs, info = env.reset(seed=self.seed)
        
        for _ in range(self.buffer_start_size):
            # Sample random action and step environment.
            action = env.action_space.sample()
            obs_next, reward, terminated, truncated, _ = env.step(action)
            
            # Save in replay memory.
            self.replay_buffer.push(obs, action, reward, obs_next, terminated)

            obs = obs_next
            if terminated or truncated:
                obs, info = env.reset()

        return obs, info


class OnPolicyAlgorithm(RLAlgorithm, ABC):
    def __init__(
        self,
        env: gym.Env,
        *,
        ac_kwargs: Dict[str, Any],
        gamma: float,
        tau_polyak: float,
        horizon: int,
        batch_size: int,
        epochs: int,
        save_every: int,
        eval_every: int,
        verbose: bool = True,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            env=env,
            ac_kwargs=ac_kwargs,
            gamma=gamma,
            tau_polyak=tau_polyak,
            save_every=save_every,
            eval_every=eval_every,
            verbose=verbose,
            seed=seed,
            device=device,
        )

        self.horizon = horizon
        self.batch_size = batch_size
        self.epochs = epochs

        self.rollout_buffer = RolloutBuffer(
            obs_shape=self.obs_shape,
            action_shape=self.action_shape,
            horizon=horizon,
            batch_size=batch_size,
        )
