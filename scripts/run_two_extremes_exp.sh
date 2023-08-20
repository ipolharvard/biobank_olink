#!/bin/bash
#SBATCH --job-name=olink
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120

NUM_GPUS=10

for panel in "all" "cardiometabolic" "inflammation"; do
    two-extremes-exp \
        --interactions 101 \
        --exp_type cls \
        --not_optimize \
        --model lr \
        --panel $panel \
        --target PP \
        --threshold 0.35 \
        --nan_th 0.3 \
        --corr_th 0.9 \
        --outer_splits 5 \
        --inner_splits 5 \
        --n_trials 200 \
        --optuna_n_workers $NUM_GPUS \
        --num_gpus $NUM_GPUS \
        --seed 42 &
done
wait
