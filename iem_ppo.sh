#!/bin/sh
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16  # must match num_env, never more than 16
#SBATCH --gres=gpu:0  # never more than 0
#SBATCH --partition=performance # do not change
#SBATCH --output=out/iem-ppo-%A_%a.out
#SBATCH --error=out/iem-ppo-%A_%a.err

.venv/bin/python iem_ppo.py --track --exp_name "iem_ppo_coef_$1" --uncertainty_coef $1