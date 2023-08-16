#!/bin/bash

python scripts/two_extremes_exp.py \
    --threshold 0.35 \
    --target sbp \
    --nan_handling remove \
    --corr_handling ignore \
    --outer_splits 5 \
    --inner_splits 5 \
    --n_trials 1000 \
    --optuna_n_workers 2 \
    --seed 42