import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax.random as jrng
import optax
from flax.core import FrozenDict
from typing import NamedTuple
from functools import partial

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
        
        return NTKEnsembleState(sign_deltas, log_deltas, log_max_norm)

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

        return deltas, NTKEnsembleState(state.sign_deltas, log_deltas, state.log_max_norm)

    return optax.GradientTransformation(init_fn, update_fn)
