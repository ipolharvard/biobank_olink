#!/bin/bash -l
#SBATCH --job-name=ukb_olink
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=128
#SBATCH --partition=defq
#SBATCH --output=olink2.log

module load singularity

script_text="
cd /olink/biobank_olink
python setup.py develop --user --no-deps
export PATH=\$HOME/.local/bin:\$PATH
clear

case $1 in
1)
    biobank_olink cross-sectional-adj \
        --target sbp \
        --model xgb \
        --panel immune_infla2 \
        --threshold 0.35 \
        --nan_th 0 \
        --corr_th 0.9 \
        --n_trials 300 \
        --outer_splits 5 \
        --inner_splits 5 \
        --optuna_n_workers 8 \
        --num_gpus \"\${SLURM_GPUS_ON_NODE}\"
    ;;
2)
    biobank_olink cross-sectional \
        --model xgb \
        --panel renal \
        --lifestyle \
        --ext \
        --olink \
        --nan_th 0 \
        --corr_th 0.9 \
        --n_trials 150 \
        --outer_splits 5 \
        --inner_splits 5 \
        --optuna_n_workers 8 \
        --num_gpus \"\${SLURM_GPUS_ON_NODE}\"
    ;;
3)
    biobank_olink prospective \
        --model xgb \
        --years 10 \
        --panel all \
        --lifestyle \
        --ext \
        --olink \
        --nan_th 0 \
        --corr_th 0.9 \
        --n_trials 100 \
        --outer_splits 5 \
        --inner_splits 5 \
        --optuna_n_workers 8 \
        --num_gpus \"\${SLURM_GPUS_ON_NODE}\"
    ;;
4)
    biobank_olink cross-on-prospective \
        --model xgb \
        --panel all \
        --nan_th 0 \
        --corr_th 0.9 \
        --n_trials 8 \
        --outer_splits 5 \
        --inner_splits 2 \
        --optuna_n_workers 8
    ;;
5)
    biobank_olink transformer \
        --model tfr \
        --panel all \
        --nan_th 0 \
        --corr_th 0.9 \
        --n_trials 100 \
        --outer_splits 5 \
        --inner_splits 2 \
        --optuna_n_workers 1 \
        --num_gpus \"\${SLURM_GPUS_ON_NODE}\"
    ;;
6)
    biobank_olink interactions \
        --n_jobs 128 \
        --fold 0
    ;;
*)
    echo \"Wrong number $1\"
    exit 1
    ;;
esac
"

clear
singularity exec \
    --contain \
    --nv \
    --writable-tmpfs \
    --bind "$SCRATCH":/olink \
    olink_latest.sif \
    bash -c "$script_text"
