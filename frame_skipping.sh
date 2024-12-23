#!/bin/sh
#SBATCH --time=12:00:00 # never more than 6 hours
#SBATCH --cpus-per-task=16  # must match num_env, never more than 16
#SBATCH --gres=gpu:0  # never more than 0
#SBATCH --partition=performance # do not change
#SBATCH --output=out/frame-skipping-%A_%a.out
#SBATCH --error=out/frame-skipping-%A_%a.err


.venv/bin/python ppo_clean_rl.py --track --exp_name skip_frames_2 --skip_frames 2
.venv/bin/python ppo_clean_rl.py --track --exp_name skip_frames_4 --skip_frames 4
.venv/bin/python ppo_clean_rl.py --track --exp_name skip_frames_6 --skip_frames 6
.venv/bin/python ppo_clean_rl.py --track --exp_name skip_frames_8 --skip_frames 8
.venv/bin/python ppo_clean_rl.py --track --exp_name skip_frames_16 --skip_frames 16
.venv/bin/python ppo_clean_rl.py --track --exp_name skip_frames_32 --skip_frames 32
