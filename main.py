from argparse import Namespace, ArgumentParser
import random

import torch
import numpy as np
import gymnasium as gym

from tinyrl.algorithms.base import RLAlgorithm
from tinyrl import DDPG, TD3, SAC


DDPG_CONFIG = {
    "lr_actor": 0.0003,
    "lr_critic": 0.0003,
    "buffer_size": 1_000_000,
    "buffer_start_size": 25_000,
    "batch_size": 256,
    "exploration_noise" : 0.1,
    "policy_delay": 2,
    "tau_polyak": 0.005,
    "save_every": 10_000,
    "eval_every": 10_000,
}


TD3_CONFIG = {
    "lr_actor": 0.0003,
    "lr_critic": 0.0003,
    "buffer_size": 1_000_000,
    "buffer_start_size": 25_000,
    "batch_size": 256,
    "tau_polyak": 0.005,
    "save_every": 10_000,
    "eval_every": 10_000,
}


SAC_CONFIG = {
    "lr_actor": 0.0003,
    "lr_critic": 0.0003,
    "lr_alpha": 0.0003,
    "buffer_size": 1_000_000,
    "buffer_start_size": 25_000,
    "batch_size": 256,
    "tau_polyak": 0.005,
    "save_every": 10_000,
    "eval_every": 10_000,
}


def algorithm_factory(args: Namespace, env: gym.Env) -> RLAlgorithm:
    if args.algorithm.lower() == "ddpg":
        return DDPG(
            env=env, 
            seed=args.seed, 
            device=args.device, 
            verbose=args.verbose, 
            **DDPG_CONFIG
        )
    elif args.algorithm.lower() == "td3":
        return TD3(
            env=env, 
            seed=args.seed, 
            device=args.device, 
            verbose=args.verbose, 
            **TD3_CONFIG
        )
    elif args.algorithm.lower() == "sac":
        return SAC(
            env=env,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose, 
            **SAC_CONFIG
        )
    else:
        raise ValueError("Unkown algorithm.")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Tiny Deep RL Training")

    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--algorithm", type=str, default="sac")
    parser.add_argument("--n_timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", default=True)

    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    env = gym.make(args.env, render_mode=None)
    alg = algorithm_factory(args, env)
    alg.train(args.n_timesteps)


if __name__ == "__main__":
    main()
