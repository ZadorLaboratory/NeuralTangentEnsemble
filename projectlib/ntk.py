import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jrng
import optax
from flax.core import FrozenDict
from typing import NamedTuple
from functools import partial
from optax._src import base
from typing import Callable, Any

from projectlib.models import Array

def perturb_params(rng, params, noise_fn = jrng.normal):
    leaves, structure = jtu.tree_flatten(params)
    keys = jrng.split(rng, len(leaves))
    leaves = (p + noise_fn(k, p.shape, p.dtype) for p, k in zip(leaves, keys))

    return jtu.tree_unflatten(structure, leaves)

def log_ntk_likelihood(sign_delta, grad, inv_temperature, log_max_norm, eta):
    log_likelihood = eta * jnp.log1p( - inv_temperature * sign_delta * grad)
    return log_likelihood

class NTKEnsembleState(NamedTuple):
    sign_deltas: FrozenDict[str, Array]
    log_deltas: FrozenDict[str, Array]
    log_max_norm: float
    deltas: FrozenDict[str, Array] # yes this is redundant. will fix later.

class AdditiveNTEState(NamedTuple):
    deltas: FrozenDict[str, Array]
    Z: float

def gradient_scale_products(grad, delta, learning_rate):
    gprod = jnp.prod(1 - learning_rate * jnp.sign(delta) * grad, axis=0)
    return gprod

def scaled_true_sgd(learning_rate, seed, renormalize=False, noise_scale=1e-3 ):
    r"""An additive version of the NTE update rule
    i.e. true stochastic Gradient Descent (SGD) optimizer in which the updates are scaled by the magnitude of
    updates so far - and where the batch size is always 1.
    Through the magic of vmap, we can run several examples at a time.
    Note this implies the same network is used to get the gradient of k examples.

    Assumes that the first dimension of grads is vmapped over examples.

    We implement the following update rule which 
    scales the learning rate by the magnitude of the change in the weights since init.
    Defining ∆W_t = w_t - w_0, 
                            ∆w_t = ∆w_{t-1} - eta |∆w_{t-1}|) grad_i
                        =>  ∆w_t = ∆w_{t-1}(1 - eta sign(∆W) grad_i)

    NOTE: this does not project the gradients to ensure the L1 norm of the ∆W stays fixed.

    If noise_scale==0, we set w_0=0 and thus ∆w=w

    """
    def init_fn(params):
        leaves, structure = jtu.tree_flatten(params)
        rng = jrng.PRNGKey(seed)
        keys = jrng.split(rng, len(leaves))
        if noise_scale > 0:
            leaves = [noise_scale * jrng.normal(k, p.shape, p.dtype)
                      for p, k in zip(leaves, keys)]
        Z = sum(jnp.sum(jnp.abs(d)) for d in leaves)
        # num_params = sum(jnp.prod(jnp.array(p.shape)) for p in jax.tree.leaves(params))
        # desired_Z = jnp.sqrt(num_params * learning_rate)
        # leaves = leaves * desired_Z / Z
        deltas = jtu.tree_unflatten(structure, leaves) 
               
        return AdditiveNTEState(deltas, Z)

    def update_fn_normalized(grads, state: AdditiveNTEState, _=None):

        prod_fn = partial(gradient_scale_products,
                          learning_rate=learning_rate)
        grads_product = jtu.tree_map(prod_fn, grads, state.deltas)

        deltas = jtu.tree_map(lambda dp, prod_: dp*prod_, state.deltas, grads_product)

        new_Z = sum(jnp.sum(jnp.abs(d)) for d in jtu.tree_leaves(deltas))
        deltas = jtu.tree_map(lambda dp: dp * state.Z / new_Z, deltas)
        
        return deltas, AdditiveNTEState(deltas, state.Z)

    def update_fn_unnormalized(grads, state: AdditiveNTEState, _=None):

        prod_fn = partial(gradient_scale_products,
                          learning_rate=learning_rate)
        grads_product = jtu.tree_map(prod_fn, grads, state.deltas)

        deltas = jtu.tree_map(lambda dp, prod_: dp*prod_, state.deltas, grads_product)
        
        return deltas, AdditiveNTEState(deltas, state.Z)

    if renormalize:
        return optax.GradientTransformation(init_fn, update_fn_normalized)
    else:
        return optax.GradientTransformation(init_fn, update_fn_unnormalized)

def ntk_ensemble(inv_temperature, seed, max_delta = None, noise_scale = 1e-3, eta = 1):
    def init_fn(params):
        leaves, structure = jtu.tree_flatten(params)
        rng = jrng.PRNGKey(seed)
        keys = jrng.split(rng, len(leaves))
        if noise_scale > 0:
            leaves = [noise_scale * jrng.normal(k, p.shape, p.dtype)
                      for p, k in zip(leaves, keys)]
        log_max_norm = jnp.log(sum(jnp.sum(jnp.abs(d)) for d in leaves))
        deltas = jtu.tree_unflatten(structure, leaves)
        sign_deltas = jtu.tree_map(lambda d: jnp.asarray(jnp.sign(d), dtype=jax.numpy.int8), deltas) # int8 but in theory could be a single bit per param, e.g. signbit plus later logic
        log_deltas = jtu.tree_map(jnp.log, jtu.tree_map(jnp.abs, deltas))
        
        return NTKEnsembleState(sign_deltas, log_deltas, log_max_norm, deltas)

    def update_fn(grads, state: NTKEnsembleState, _ = None):
        # compute ntk likelihood
        log_likelihood_fn = partial(log_ntk_likelihood,
                                inv_temperature=inv_temperature,
                                log_max_norm=state.log_max_norm,
                                eta=eta)
        log_likelihood = jtu.tree_map(log_likelihood_fn, state.sign_deltas, grads)
        # sum over batch
        log_likelihood = jtu.tree_map(lambda l: jnp.sum(l, axis=0), log_likelihood)   
        # compute update 
        log_deltas = jtu.tree_map(lambda ld, ll: ld + ll,
                                  state.log_deltas, log_likelihood)
        # renormalize and clip
        log_current_norm = jax.nn.logsumexp(jnp.array([
            jax.nn.logsumexp(ll)
            for ll in jtu.tree_leaves(log_deltas)
        ]))
        log_max_delta = jnp.log(max_delta)
        log_deltas = jtu.tree_map(
            lambda ld: jnp.clip(ld + state.log_max_norm - log_current_norm,
                                           a_max=log_max_delta),
            log_deltas,
        )                 
        deltas = jtu.tree_map(lambda ld, s: jnp.exp(ld) * s,
                                  log_deltas, state.sign_deltas)

        return deltas, NTKEnsembleState(state.sign_deltas, log_deltas, state.log_max_norm, deltas)

    return optax.GradientTransformation(init_fn, update_fn)
