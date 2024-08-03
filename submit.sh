#!/bin/bash
#SBATCH --job-name=momentum
#SBATCH --output=job_output_%j_%a.txt
#SBATCH --mem-per-gpu=78G
#SBATCH --gpus=1
#SBATCH --array=5

cd /nfs/scratch-1/benjami/NeuralTangentEnsemble

export MAMBA_EXE='/nfs/scratch-1/benjami/bin/micromamba';
export MAMBA_ROOT_PREFIX='/nfs/scratch-1/benjami/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  
fi
unset __mamba_setup

# Load the Python environment
micromamba activate neural-tangent-ensemble

# Task incremental learning test. No wandb.
python train-continual-learning.py model=simple-cnn optimizer=sgd ++data.batchsize=64 ++training.nepochs=10 data=cifar100 ntasks=10 task_style=task-incremental ++model.nclasses=100

## Old commands
# python train-continual-learning.py --multirun model=resnet18,simple-cnn,convnext optimizer=sgd ++optimizer.learning_rate=0.01 ++optimizer.momentum=0,0.1,0.2,0.3,.4,.5,.6,.7,.8,.85,.9,.91,.93,.95,.97,.99,.995 ++data.batchsize=512 ++training.nepochs=100 logger=wandb-logger data=cifar100 ntasks=10 task_style=task-incremental ++model.nclasses=100 ++logger.tags=momentum_sweep_cifar100-task ++seed=${SLURM_ARRAY_TASK_ID}

# python train-continual-learning.py --multirun model=simple-cnn,convnext optimizer=adam ++optimizer.inv_temperature=0.005 model.width_multiplier=1,2,4,8,16,32,64,128 ++data.batchsize=128 ++training.nepochs=100 logger=wandb-logger data=cifar100 ntasks=10 task_style=task-incremental ++model.nclasses=100 ++logger.tags=width-convnext-cifar100-taskInc ++seed=${SLURM_ARRAY_TASK_ID}

# python train-continual-learning.py --multirun model=convnext_big optimizer=ntk-ensemble optimizer.inv_temperature=0.001,0.002,0.005 ++data.batchsize=512 ++training.nepochs=100 logger=wandb-logger data=cifar100 ntasks=10 task_style=class_subset ++model.nclasses=10 ++logger.tags=LR-convnext-cifar100 ++seed=${SLURM_ARRAY_TASK_ID}


# python train-continual-learning.py model=convnext_big optimizer=sgd ++data.batchsize=64 ++training.nepochs=100 data=cifar100 ntasks=10 task_style=class_subset ++model.nclasses=10 logger=wandb-logger ++logger.tags=big-convnext-cifar100 ++seed=${SLURM_ARRAY_TASK_ID}
