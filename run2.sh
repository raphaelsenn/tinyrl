#!/bin/bash

ENVS=(
    "Pusher-v5"
)

ALGORITHMS=("ddpg" "td3" "sac")

N_TIMESTEPS=3000000
N_SEEDS=2

for env in "${ENVS[@]}"; do
    for algorithm in "${ALGORITHMS[@]}"; do
        for ((seed=0; seed<N_SEEDS; seed++)); do
            echo "Running $algorithm on $env with seed $seed"

            python main.py \
                --env="$env" \
                --algorithm="$algorithm" \
                --n_timesteps="$N_TIMESTEPS" \
                --seed="$seed"
        done
    done
done