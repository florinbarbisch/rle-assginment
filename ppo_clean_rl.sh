#!/bin/sh
#SBATCH --time=6:00:00 # never more than 6 hours
#SBATCH --cpus-per-task=16  # must match num_env, never more than 16
#SBATCH --gres=gpu:0  # never more than 0
#SBATCH --partition=performance # do not change
#SBATCH --output=out/ppo-clean-rl-%A_%a.out

.venv/bin/python ppo_clean_rl.py --track