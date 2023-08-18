#!/bin/bash

two-extremes-exp \
    --exp_type cls \
    --threshold 0.35 \
    --target PP2 \
    --nan_handling remove \
    --corr_handling drop \
    --outer_splits 5 \
    --inner_splits 5 \
    --n_trials 1000 \
    --optuna_n_workers 8 \
    --num_gpus 8 \
    --seed 42