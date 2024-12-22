#!/bin/sh
#SBATCH --time=6:00:00 # never more than 6 hours
#SBATCH --cpus-per-task=16  # must match num_env, never more than 16
#SBATCH --gres=gpu:0  # never more than 0
#SBATCH --partition=performance # do not change
#SBATCH --output=out/epsilon-tuning-%A_%a.out
#SBATCH --error=out/epsilon-tuning-%A_%a.err

.venv/bin/python ppo_clean_rl.py --track --exp_name epsilon_tuning_0.1 --clip_coef 0.1
.venv/bin/python ppo_clean_rl.py --track --exp_name epsilon_tuning_0.2 --clip_coef 0.2
.venv/bin/python ppo_clean_rl.py --track --exp_name epsilon_tuning_0.3 --clip_coef 0.3
.venv/bin/python ppo_clean_rl.py --track --exp_name epsilon_tuning_0.4 --clip_coef 0.4