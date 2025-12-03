#!/bin/bash
#SBATCH --job-name=knee_eval_all
#SBATCH --output=knee_eval_all-%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mail-user=akatragadda@scu.edu
#SBATCH --mail-type=END

module load Python
module load CUDA/12.2.1

cd /WAVE/projects2/CSEN-240-Fall25/akatragadda
pip install "tensorflow[and-cuda]==2.15.1" "numpy<2" seaborn matplotlib scikit-learn opencv-python-headless --user --quiet

export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false

srun python3 eval_knee_all_splits.py