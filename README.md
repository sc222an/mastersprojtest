# Mastersproj26 Model Training

## Project Structure
IVY-Fake/  
│  
├── src/  
│   ├── data/              # Dataset + loaders  
│   ├── models/            # ADD MODELS HERE  
│  
├── configs/               # ADD YAML CONFIGS HERE  
│  
├── train_baseline.py      # Main training entrypoint  
│  
├── jobs/  
│   ├── job_train_*.sh     # Slurm job scripts  
│  
└── runs/                  # Saved checkpoints  

## Adding a New Model
### 1. Create Your Model

Add a file in:

src/models/

Example:

src/models/my_model.py

Requirements:

Inherit from torch.nn.Module

Accept correct input shape:

Mode	Input Shape
frames	(B, T, C, H, W)
clip	(B, C, T, H, W)

### 2. Register Model in train_baseline.py

Add import:

from src.models.my_model import MyModel

Then update model selection:

if args.backbone == "my_model":
    model = MyModel(...)

## Writing a Slurm Job File

Create:

job_train_my_model.sh

Template:

#!/bin/bash  
#SBATCH --job-name=ivy_my_model  
#SBATCH --output=my_model_%j.out  
#SBATCH --error=my_model_%j.err  
#SBATCH --time=12:00:00  
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

cd /users/{user}
mkdir -p runs

python train_baseline.py \
  --mode frames \
  --backbone my_model \
  --batch_size 8 \
  --num_workers 4 \
  --out runs/my_model.pt

Submit job:

sbatch job_train_my_model.sh

Monitor:

squeue --me
tail -f my_model_<JOBID>.out
## Model Outputs

All checkpoints are saved to:

/users/{user}/IVY-Fake/runs/
