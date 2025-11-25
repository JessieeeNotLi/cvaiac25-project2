# CVAIAC Project 2

### Environment Setup

If not already done, please follow the [First Tutorial](https://www.trace.ethz.ch/teaching/CVAIAC2024/exercises/Exercise_03-intro_to_slurm.pdf) to setup your environment. 
We can keep using the environment from project 1.
To extend the environment for this exercise you should run (in an interactive session with a GPU):
```shell script
# Start a 5h interactive session with 1 gpu for proper setup
srun --time 300 --gres=gpu:1 --pty bash -i

# Activate the conda environment from project 1
conda activate py39

# Set up pip cache and tmp dir 
# (important so you don't run out of space during installation)
mkdir /usr/bmicnas03/data-biwi-01/$USER/data/tmp
mkdir /usr/bmicnas03/data-biwi-01/$USER/data/pip_cache
export TMPDIR=/usr/bmicnas03/data-biwi-01/$USER/data/tmp
export PIP_CACHE_DIR=/usr/bmicnas03/data-biwi-01/$USER/data/pip_cache

# Assuming that you already installed project 1 requirements.txt
pip install -r requirements.txt
```

Some detailed notes about the installation:
- This upgrades `numpy` from project 1's version `1.19.5` to `1.23.5` in order to be compatible with `vispy`
- activating the GPU before installing is necessary for the compilation of the `pointnet` package (to ensure it being usable with the same GPU later on)
- The installation presented here is meant to work on the cluster with its Titan GPUs. If you want to run it on your local machine (e.g. for debugging), possibly with a different GPU, some dependencies will likely have to be adjusted.

### Interactive Debugging (Problem 2.1-2.5)

Problem 2.1-2.5 do not require a GPU, so you can request a interactive session without a GPU.
We recomand using CPU only jobs for these tasks, as you can be kicked out of GPU sessions, if you do not utilize the GPU.
You can launch an interactive session with:

```shell script
srun --nodes=1 --time=2:00:00 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=4 --mem-per-cpu=4G --pty bash -i
```

To keep it alive you can use tmux before launching the interactive session. If you want to close the connection
but keep the script running, detach from tmux using Ctrl+B and D. After that, you can exit the ssh connection, while
tmux and the training keep running. You can enter the scroll mode using Ctrl+B and [ and exit it with Q. 
In the scroll mode, you can scroll using the arrow keys or page up and down. Tmux has also some other nice features
such as multiple windows or panels (https://www.hamvocke.com/blog/a-quick-and-easy-guide-to-tmux/). Please note
that there is a **timeout** of 24 hours to the instance. If you find that not sufficient, please adjust 
`TIMEOUT = 24  # in hours`

### Interactive Debugging (Problem 2.6 and Problem 3)

You can launch an interactive session with a GPU:

```shell script
srun --nodes=1 --time=8:00:00 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=4 --mem-per-cpu=4G --gpus-per-node=1 --pty bash -i
```

### Problem 2.1-2.5

When testing the speed of your code for task 1 & 2, do not use a debugger (i.e. VSCode) but run it directly in the console. 
The debugger can slow down the computation.

To test your source code for task 1, 2, 4, and 5, you can run following script:

```shell script
python tests/test.py --task X  # , where X is the task number.
```


### Weights and Biases Monitoring

You can monitor the training via the wandb web interface https://wandb.ai/home. 
In the workspace panel, we recommend switching the x-axis to epoch (x icon in the top right corner) for
visualization.
The logged histograms, you can only view if you click on a single run.

Before submitting a job, run:
```shell script
wandb login
```

We provide and exemplary trainings log [here](https://api.wandb.ai/links/timbr/sh17eabu). 
Your log should look similar, but not identical due to the randomness in the training process.

### Training (Problem 2.6 & 3)

In the [config.yaml](problem2_and_3/config.yaml) file, change the `group_id` to your group id and the `name` to the name of the experiment you are running.

You can start and debug the training in an interactive session using:

```shell script
python train.py
```

You can launch a training on a cluster GPU using:

```shell script
sbatch train.sh
```
The logs will be saved to the [logs](logs) folder.

You can change the training hyperparameters in [config.yaml](problem2_and_3/config.yaml). 

### Report Problem 3:
You can use the `CVAIAC_2024_Report_Template.zip` latex template for your report on the problem 3. 
For this, you can import it as a zip file into an additor (i.e. `https://www.overleaf.com`).
