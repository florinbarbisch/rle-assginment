#!/bin/sh
#SBATCH --time=6:00:00 # never more than 6 hours
#SBATCH --cpus-per-task=16  # must match num_env, never more than 16
#SBATCH --gres=gpu:0  # never more than 0
#SBATCH --partition=performance # do not change
#SBATCH --output=out/initial-run-%A_%a.out
#SBATCH --error=out/initial-run-%A_%a.err

.venv/bin/python ppo_clean_rl.py --track --exp_name initial_run
