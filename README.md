# Importance Weighted Supervised Fine Tuning (iw-SFT)

Author's Pytorch implementation of **I**mportance **W**eighted **S**upervised **F**ine **T**uning (iw-SFT). Iw-SFT uses importance weights to adaptively upweight or downweight points during training; we show this provides a much tighter bound to the RL training objective in comparison to SFT alone.


## Overview of the Code
The code consists of 7 Python scripts and the file `main.py` contains various parameter settings which are interpreted and described in our paper.
### Requirements
- `torch                         1.12.0`
- `mujoco                        2.2.1`
- `mujoco-py                     2.1.2.14`
- `d4rl                          1.1`

### Running the code
- `python main.py`: trains the network, storing checkpoints along the way.
- `Example`: 
```bash
python3 stratefied_filtered_bc.py
```
