
# RLE Mini-Challenge

Ziel dieser Mini-Challenge ist es einen Deep Reinforcemen Learning Agenten zu trainieren, der einen möglichst hohen Score im Atari Spiel "Space Invaders" erreicht.

In diesem Repository findet ihr ein Template, auf dem ihr eure Lösung implementieren könnt, sowie eine Beispiel-Implementation eines einfachen DQN Agenten.

## Atari Space Invaders Environment

![](https://www.gymlibrary.ml/_images/space_invaders.gif)

Spiel Beschreibung: [https://atariage.com/manual_html_page.php?SoftwareLabelID=460](https://atariage.com/manual_html_page.php?SoftwareLabelID=460)

Gym GitHub: [https://github.com/openai/gym](https://github.com/openai/gym)

Gym Dokumentation: [https://www.gymlibrary.ml](https://www.gymlibrary.ml)

Gym Space Invaders Dokumentation: [https://www.gymlibrary.ml/environments/atari/space_invaders/](https://www.gymlibrary.ml/environments/atari/space_invaders/)


## Installation

**Empfehlung:** Verwenden sie ein separates virtual environment.

Zuerst folgendes pip install:
```
pip install cleanrl gymnasium[atari,accept-rom-license] ale-py pygame tensorboard opencv-python absl-py tensorboardX stable-baselines3 tyro moviepy
```

Ausserdem muss PyTorch installiert werden:

[https://pytorch.org/get-started](https://pytorch.org/get-started)

## Tensorboard
Einzelne Experimente können in Tensorboard gelogged werden.
So können diese visualisiert werden:
```
tensorboard --logdir runs
```

## Training starten:

PPO Clean RL
```python ppo_clean_rl.py```

DQN Clean RL
```python dqn_clean_rl.py```

DQN Yanick
```python dqn_example.py```

 ## Evaluation

PPO Clean RL:
 ```python ppo_clean_rl.py --eval-checkpoint "runs/{run_name}/{args.exp_name}.cleanrl_model"```

DQN Clean RL:
  ```python dqn_clean_rl.py --eval-checkpoint "runs/{run_name}/{args.exp_name}.cleanrl_model"NT```

DQN Yanick
```python dqn_example.py --eval_checkpoint PATH_TO_CHECKPOINT```

## Inhalt

### run.py

Template für das Implementieren der Lösung.

Es steht euch jedoch offen, ob ihr dieses Template verwendet oder einen eigenen Ansatz verfolgt.

### run.sh

Script um `run.py` auf dem SLURM cluster auszuführen.

### dqn_example.py

Beispiel-Implementation eines einfachen DQN agent. Kann für die mini-challenge verwendet und erweitert werden.

### dqn_clen_rl.py

Beispiel-Implementation DQN von [CleanRL](https://docs.cleanrl.dev/). Kann für die mini-challenge verwendet und erweitert werden.

### ppo_clen_rl.py

Beispiel-Implementation PPO von [CleanRL](https://docs.cleanrl.dev/). Kann für die mini-challenge verwendet und erweitert werden.


### rle_assignment/env.py

Beinhaltet die `make_env` Funktion, zum erstellen einer Environment-Instanz.

### rle_assignment/utils.py

Beinhaltet nützliche Funktionen und Klassen.

## SLURM Cluster
Zugriff erhalten:

Public key an Yanick senden, damit er euch einen Account erstellt.

### Daten auf den Cluster kopieren

``` scp -r <local_path> <username>@slurmlogin.cs.technik.fhnw.ch:~/<remote_path> ```

### SLURM Befehle
Job starten:
``` sbatch run.sh ```
Job stoppen:
``` scancel <job_id> ```
Job Status:
``` squeue -u <username> ```
Job Output:
``` cat out/rle-mini-challenge-<job_id>.out ```


## Copy run files to local machine

```bash
scp -r f.barbisch@slurmlogin.cs.technik.fhnw.ch:~/rle-assginment/runs ./slurm/runs
scp -r f.barbisch@slurmlogin.cs.technik.fhnw.ch:~/rle-assginment/out ./slurm/out
```
