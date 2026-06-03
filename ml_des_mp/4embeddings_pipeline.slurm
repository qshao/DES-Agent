#!/bin/bash
 
#SBATCH --time=0-48:00:00                       #Time for the job to run
#SBATCH --nodes=1                               #Number of nodes required
#SBATCH --mem=16g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                    #Number of cores needed for the job
#SBATCH --partition=V4V32_CAS40M192_L  #Name of the queue
##SBATCH --partition=V4V32_SKY32M192_L
##SBATCH --partition=H8V141_SAP112M2000_L 
#SBATCH --gres=gpu:1                            #Number of GPU's
#SBATCH --account=gol_qsh226_uksr               #Name of account to run under
##SBATCH --mail-type ALL                         #Send email on start/end
##SBATCH --mail-user 8653438959@tmomail.net               #Where to send email


#source "/project/qsh226_uksr/kzh259/miniconda3/etc/profile.d/conda.sh"
#conda activate /project/qsh226_uksr/kzh259/env/ml

module load ccs/container/R/4.4.2

export APPTAINER_CACHEDIR=/project/qsh226_uksr/kzh259/.apptainer/cache
export APPTAINER_TMPDIR=/project/qsh226_uksr/kzh259/.apptainer/tmp
export SINGULARITY_CACHEDIR=$APPTAINER_CACHEDIR
export SINGULARITY_TMPDIR=$APPTAINER_TMPDIR
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

SIF=/project/qsh226_uksr/kzh259/pytorch_2.2.2-cu121.sif
VENV=/project/qsh226_uksr/kzh259/venv/ml

# HF / Transformers / Torch caches
export HF_HOME=/project/qsh226_uksr/kzh259/hf
export HF_HUB_CACHE=/project/qsh226_uksr/kzh259/hf/hub
export TRANSFORMERS_CACHE=/project/qsh226_uksr/kzh259/hf/transformers
export TORCH_HOME=/project/qsh226_uksr/kzh259/hf/torch
export PIP_CACHE_DIR=/project/qsh226_uksr/kzh259/hf/pip-cache

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$TORCH_HOME" "$PIP_CACHE_DIR"

# model training
singularity exec --nv "$SIF" "$VENV/bin/python" \
main_train.py --config config.yaml

#External prediction
#singularity exec --nv "$SIF" "$VENV/bin/python" \
#predict.py --config config.yaml --ckpt runs/chemberta_strict_pair_fold05of05_best.pt --csv #validate2.csv --out_csv predict.csv


#singularity exec --nv "$SIF" "$VENV/bin/python" \
#interpret_updated.py --config config.yaml --ckpt runs/chemberta_strict_pair_fold05of05_best.pt 


