clear

biobank_olink two-extremes \
    --target sbp \
    --model tfr \
    --panel all \
    --threshold 0.35 \
    --nan_th 0 \
    --corr_th 0.9 \
    --n_trials 300 \
    --outer_splits 5 \
    --inner_splits 2 \
    --optuna_n_workers 1 \
    --num_gpus "${SLURM_GPUS_ON_NODE}"
