#!/bin/bash

ENVS=(
    "Ant-v5"
    "HalfCheetah-v5"
    "Hopper-v5"
    "Humanoid-v5"
    "Walker2d-v5",
    "Pusher-v5"
)

ALGORITHMS=("ddpg" "td3" "sac")

N_TIMESTEPS=1000000
N_SEEDS=3

for env in "${ENVS[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
        for ((seed=0; seed<N_SEEDS; seed++)); do
            echo "Running $algorithm on $env with seed $seed"

            python3 main.py \
                --env="$env" \
                --algorithm="$algorithm" \
                --n_timesteps="$N_TIMESTEPS" \
                --seed="$seed"
        done
    done
done