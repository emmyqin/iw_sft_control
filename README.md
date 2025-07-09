# Importance Weighted Supervised Fine Tuning (iw-SFT)

Author's Pytorch implementation of **I**mportance **W**eighted **S**upervised **F**ine **T**uning (iw-SFT). Iw-SFT uses importance weights to adaptively upweight or downweight points during training; we show this provides a much tighter bound to the RL training objective in comparison to SFT alone.


## Overview of the Code
The code consists of 3 Python scripts and the file `main.py`. To install the correct packages, run the following in the command line. Please note first you need to have Mujoco downloaded in the folder /home/ubuntu/.mujoco/mujoco.
~~~
pip install -r requirements.txt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

sudo apt-get update
sudo apt-get install libx11-dev patchelf libglew-dev libxcb1
~~~


### Running the code
- `python3 main.py`: trains the network, make sure to enter your wandb details.
