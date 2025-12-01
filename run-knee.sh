#!/bin/bash
#SBATCH --job-name=knee_it8
#SBATCH --output=knee_it8-%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-user=akatragadda@scu.edu
#SBATCH --mail-type=END

module load Python
# We try to load CUDA, but we will let pip handle the heavy lifting
module load CUDA/12.2.1 

cd /WAVE/projects2/CSEN-240-Fall25/akatragadda

# --- THE STABILITY FIX ---
# Instead of installing the newest TF, we pin it to 2.15.1
# This version is much friendlier to university clusters.
pip install "tensorflow[and-cuda]==2.15.1" "numpy<2" seaborn matplotlib scikit-learn opencv-python-headless --user --quiet

# Fix the internal crash (just in case)
export XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false

# Run the Iteration 8 script
srun python3 knee-osteo-it8.py