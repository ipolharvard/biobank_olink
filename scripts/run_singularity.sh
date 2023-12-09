#!/bin/bash -l
#SBATCH --job-name=ukb_olink
#SBATCH --time=7-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:8
#SBATCH --output=olink2.log
#SBATCH --error=olink2.log

module load singularity

export exp_number=4

script_text="
cd /olink/biobank_olink
python setup.py develop --user --no-deps
export PATH=\$HOME/.local/bin:\$PATH

clear

case \$exp_number in
1)
    biobank_olink two-extremes \
        --target pp \
        --model xgb \
        --panel neurology \
        --no_opt \
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
    biobank_olink pred-diagnosis \
        --model xgb \
        --panel all \
        --years 10 \
        --lifestyle \
        --ext \
        --nan_th 0 \
        --corr_th 0.9 \
        --n_trials 300 \
        --outer_splits 5 \
        --inner_splits 5 \
        --optuna_n_workers 8 \
        --num_gpus \"\${SLURM_GPUS_ON_NODE}\"
    ;;
3)
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
4)
    biobank_olink pred-diag2 \
        --model xgb \
        --panel cardiometabolic \
        --nan_th 0 \
        --corr_th 0.9 \
        --n_trials 8 \
        --outer_splits 5 \
        --inner_splits 2 \
        --optuna_n_workers 8 \
        --num_gpus \"\${SLURM_GPUS_ON_NODE}\"
    ;;
*)
    echo \"Wrong number $exp_number\"
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
