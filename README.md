# cs285-final-project

## Installation

```
conda create -n cs285 python=3.7

source activate cs285

# Get tensorflow 1.14 and CUDA 10.2
conda install cudatoolkit=10.2
pip install tensorflow-gpu=1.14

# Include 'baselines' folder
pip install -e baselines

# Get retro contest depedendency, might have to install docker,rest
git clone --recursive https://github.com/openai/retro-contest.git
pip install -e "retro-contest/support[docker,rest]"

# download game ROMS
python -m retro.import.sega_classics

## Joint Reptile training

Edit `ppo2_reptile.py` with the desired hyperparameters/checkpoints. Run with

```
python ppo2_reptile.py
```

TODO: Test evaluation (Andrew)

## Important Links

[Gym Retro Setup](https://contest.openai.com/2018-1/details/)

[Retro Docs](https://retro.readthedocs.io/en/latest/getting_started.html)