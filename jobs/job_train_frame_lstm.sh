#!/bin/bash
#SBATCH --job-name=ivy_frame_lstm
#SBATCH --output=frame_lstm_%j.out
#SBATCH --error=frame_lstm_%j.err
#SBATCH --time=40:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

module load miniforge/24.7.1
conda activate ivyfake

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /users/sc222an/mastersprojtest
mkdir -p runs

python train_baseline.py \
  --mode frames \
  --backbone frame_lstm \
  --batch_size 4 \
  --num_workers 8 \
  --epochs 20 \
  --lr 3e-4 \
  --out runs/frame_lstm.pt
