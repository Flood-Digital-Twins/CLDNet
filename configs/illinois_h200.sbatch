#!/bin/bash
#SBATCH -J hydro_surrogate_H200
#SBATCH -A <your-account>
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h200:8
#SBATCH -N 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
test -f splits/illinois_split.json

unset PYTHONPATH PYTHONHOME
module load anaconda3
conda activate pytorch
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=8

model="${1:-ldnet}"
case "$model" in
  ldnet) model_args=() ;;
  cldnet) model_args=(--use-static-features --static-feature-dim 3) ;;
  *) echo "Use ldnet or cldnet as the first argument" >&2; exit 2 ;;
esac

torchrun --nproc_per_node=8 code/ldnet/ldnet_chicago_efficient.py \
  --ddp --num-trajectories 110 --num-train 90 --num-valid 3 \
  --split-file splits/illinois_split.json --sample-indices 100000 \
  --all-vars --num-latent-states 200 --fourier-mapping-size 32 \
  --no-resume-latest "${model_args[@]}" \
  --model-path "outputs/training/${model}_illinois_h200"
