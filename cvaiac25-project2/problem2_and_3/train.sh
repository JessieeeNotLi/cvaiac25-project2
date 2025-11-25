#!/bin/bash
#SBATCH --job-name=train_job
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%j.out

# exit when any command fails
set -e

# load conda environment
source /usr/bmicnas03/data-biwi-01/$USER/data/conda/etc/profile.d/conda.sh
conda activate py39

# navigate to project directory
cd /usr/bmicnas03/data-biwi-01/$USER/data/cvaiac25-project2/problem2_and_3/

# add parent directory to PYTHONPATH - necessary for imports from problem1/
export PYTHONPATH=..:.

# run training script
python train.py
