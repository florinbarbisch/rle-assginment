#!/bin/sh
#SBATCH --time=6:00:00 # never more than 6 hours
#SBATCH --cpus-per-task=1  # must match num_env, never more than 16
#SBATCH --partition=performance # do not change
#SBATCH --output=out/random-baseline-%A_%a.out
#SBATCH --error=out/random-baseline-%A_%a.err

.venv/bin/python random_baseline.py --track