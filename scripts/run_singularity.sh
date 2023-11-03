#!/bin/bash -l
#SBATCH --job-name=ukb_olink
#SBATCH --time=7-00:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:5
#SBATCH --output=olink.log

module load singularity

clear
singularity exec \
    --contain \
    --nv \
    --writable-tmpfs \
    --bind "$SCRATCH":/olink \
    olink_latest.sif \
    bash -c "cd /olink/biobank_olink &&
    python setup.py develop --user --no-deps &&
    export PATH=\$HOME/.local/bin:\$PATH &&
    bash scripts/run_two_extremes_exp.sh"
