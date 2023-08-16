#!/bin/bash

two-extremes-exp \
    --threshold 0.35 \
    --target SBP \
    --nan_handling remove \
    --corr_handling drop \
    --outer_splits 5 \
    --inner_splits 5 \
    --n_trials 200 \
    --optuna_n_workers 2 \
    --seed 42