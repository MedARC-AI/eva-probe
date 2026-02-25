#!/usr/bin/env bash
#SBATCH --job-name=OM_eva
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --partition=main
#SBATCH --account=training
#SBATCH --output=~/OpenMidnight/slurms/%j.out
#SBATCH --error=~/OpenMidnight/slurms/%j.err

set -euo pipefail

source ~/OpenMidnight/.venv/bin/activate
cd ~/OpenMidnight/eva-probe

./run_evals.sh /data/OpenMidnight_ckpts/openmidnight_checkpoint.pth ~/openmidnight_eval_results