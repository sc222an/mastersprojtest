#!/bin/bash
#SBATCH --job-name=ivy_test_all
#SBATCH --output=test_all_%j.out
#SBATCH --error=test_all_%j.err
#SBATCH --time=04:00:00
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

echo "=============================="
echo "Testing all frame_lstm models"
echo "=============================="

for checkpoint in runs/*.pt; do
    echo ""
    echo "--- Testing: $checkpoint ---"
    python test.py \
        --checkpoint "$checkpoint" \
        --mode frames \
        --backbone frame_lstm \
        --batch_size 4 \
        --num_workers 8
    echo "--- Done: $checkpoint ---"
done

echo ""
echo "=============================="
echo "All models tested."
echo "=============================="