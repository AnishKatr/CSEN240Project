#!/bin/bash
#SBATCH --job-name=knee_base
#SBATCH --output=knee_base-%j.out
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mail-user=akatragadda@scu.edu
#SBATCH --mail-type=END

# Load software for Python
module load Python

# Move to project folder
cd /WAVE/projects2/CSEN-240-Fall25/akatragadda

# Run the program
srun python3 knee-osteo.py