#!/usr/bin/env bash
#SBATCH --job-name=olink
#SBATCH --time=12:00:00
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --cpus-per-task=120

NUM_GPUS="8"

biobank_olink two-extremes \
    --target sbp \
    --model xgb \
    --panel all \
    --threshold 0.35 \
    --nan_th 0.6 \
    --corr_th 0.9 \
    --outer_splits 5 \
    --inner_splits 5 \
    --n_trials 200 \
    --optuna_n_workers ${NUM_GPUS} \
    --num_gpus ${NUM_GPUS} \
    --seed 42
