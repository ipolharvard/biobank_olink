clear

case $1 in
1)
    biobank_olink two-extremes \
        --target dbp \
        --model xgb \
        --panel cardiometabolic \
        --threshold 0.35 \
        --nan_th 0 \
        --corr_th 0.9 \
        --n_trials 300 \
        --outer_splits 5 \
        --inner_splits 5 \
        --optuna_n_workers 8 \
        --num_gpus "${SLURM_GPUS_ON_NODE}"
    ;;
2)
    biobank_olink pred-diagnosis \
        --model xgb \
        --panel all \
        --years 10 \
        --lifestyle \
        --bp \
        --olink \
        --nan_th 0 \
        --corr_th 0.9 \
        --n_trials 300 \
        --outer_splits 5 \
        --inner_splits 5 \
        --optuna_n_workers 8 \
        --num_gpus "${SLURM_GPUS_ON_NODE}"
    ;;
3)
    biobank_olink transformer \
        --panel all \
        --nan_th 0 \
        --corr_th 0.9 \
        --n_trials 100 \
        --outer_splits 5 \
        --inner_splits 2 \
        --optuna_n_workers 1 \
        --num_gpus "${SLURM_GPUS_ON_NODE}"
    ;;
*)
    echo "Wrong number $1"
    exit 1
    ;;
esac
