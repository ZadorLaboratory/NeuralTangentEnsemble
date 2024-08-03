# Continual learning with the Neural Tangent Ensemble

Motivated by the problem of continual learning, and also by the biological observations of Dale's Law and pruning, we make the following observations:
1. Ensembles are natural continual learners
2. Neural networks can be interpreted as large ensembles over the space of input/output functions. 
  a. In the NTK expansion (i.e. just Taylor expand your network around the init), each parameter contributes its own component function given by its *Jacobian* (or "neural tangent").
  b. The size of the change in parameter is the weight of that component function in the ensemble.
  c. If you expand around the origin, the weights themselves (rather than the change) become the weights in the ensemble.
2. This interpretation gives you a learning rule which is, arguably, more biological. First, it is multiplicative, so positive weights stay positive and negative weights stay negative. Second, it gives you a way of thinking about pruning.

See the manuscript document for the Algorithm.

## Current demonstrations

### Jax
#### Installation
```bash
# or replace micromamba with conda...but you should really use micromamba
micromamba create -f environment.yaml

# active environment
micromamba activate neural-tangent-ensemble

# install deps
poetry install --no-root
```
#### Running
The Shuffled MNIST experiment can be run with:
```bash
# for backprop
python train-continual-learning.py

# for nte
python train-continual-learning.py optimizer=ntk-ensemble
```

Check out the configs for more options about hyperparameters, tasks, etc. All configurations are managed with Hydra.
### Pytorch demo notebooks

This implementation uses Pytorch `functools` to obtain per-example gradients via `vmap`. Using Shuffled MNIST as a testbed, there are demonstration notebooks for:
 - `pytorch_demo/demo_bayes.ipynb`: Demonstrates the NTK ensemble idea.
 - `pytorch_demo/demo_pruning.ipynb`: Plays around with learning by ONLY pruning. 
 - `pytorch_demo/gradient alignment is the problem.ipynb`: Compares the gradients of the NTK ensemble with the gradients at initialization. The two can diverge rapidly.
