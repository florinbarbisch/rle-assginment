# RLE Mini-Challenge

Ziel dieser Mini-Challenge ist es einen Deep Reinforcemen Learning Agenten zu trainieren, der einen möglichst hohen Score im Atari Spiel "Space Invaders" erreicht.

In diesem Repository ist der Code für meine Implemenierung zu finden. Sowie der [Berich](Bericht.md).

## Atari Space Invaders Environment

![](https://www.gymlibrary.ml/_images/space_invaders.gif)

Spiel Beschreibung: [https://atariage.com/manual_html_page.php?SoftwareLabelID=460](https://atariage.com/manual_html_page.php?SoftwareLabelID=460)

Gym GitHub: [https://github.com/openai/gym](https://github.com/openai/gym)

Gym Dokumentation: [https://www.gymlibrary.ml](https://www.gymlibrary.ml)

Gym Space Invaders Dokumentation: [https://www.gymlibrary.ml/environments/atari/space_invaders/](https://www.gymlibrary.ml/environments/atari/space_invaders/)


## Installation

Zuerst folgendes pip install:
```
pip install -r requirements.xt
```

Ausserdem muss CUDA PyTorch installiert werden:

[https://pytorch.org/get-started](https://pytorch.org/get-started)


## Run Experiments on SLURM
Evaluate Baseline (Random Agent):
```bash
sbatch eval_random_baseline.sh
```

Initial Setup:
```bash
sbatch initial_run.sh
```

Epsilon Hyperparameter Tuning:
```bash
sbatch epsilon_tuning.sh
```

Intrinsic Exploration Module (IEM):
```bash
sbatch iem_ppo.sh
```

ResNet Agent:
```bash
sbatch ppo_resnet.sh
```

Frame Skipping:
```bash
sbatch frame_skipping.sh
```

## Copy SLURM Run Files to Local Machine

```bash
scp -r f.barbisch@slurmlogin.cs.technik.fhnw.ch:~/rle-assginment/runs ./slurm/
scp -r f.barbisch@slurmlogin.cs.technik.fhnw.ch:~/rle-assginment/out ./slurm/
```

## Inspect Logs wih Tensorboard

```bash
tensorboard --logdir slurm/runs/ALE/ 
```
