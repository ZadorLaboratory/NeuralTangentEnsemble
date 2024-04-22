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

def ntk_likelihood(delta, grad, inv_temperature, max_norm):
    # why is this not multiplied by max_norm?
    likelihood = 1 - inv_temperature * jnp.sign(delta) * grad

    return likelihood

class NTKEnsembleState(NamedTuple):
    deltas: FrozenDict[str, Array]
    max_norm: float

def ntk_ensemble(inv_temperature, seed, max_delta = None, noise_scale = 1e-3):
    def init_fn(params):
        leaves, structure = jtu.tree_flatten(params)
        rng = jrng.PRNGKey(seed)
        keys = jrng.split(rng, len(leaves))
        if noise_scale > 0:
            leaves = [noise_scale * jrng.normal(k, p.shape, p.dtype)
                      for p, k in zip(leaves, keys)]
        max_norm = sum(jnp.sum(jnp.abs(d)) for d in leaves)
        deltas = jtu.tree_unflatten(structure, leaves)

        return NTKEnsembleState(deltas, max_norm)

    def update_fn(grads, state: NTKEnsembleState, _ = None):
        # compute ntk likelihood
        likelihood_fn = partial(ntk_likelihood,
                                inv_temperature=inv_temperature,
                                max_norm=state.max_norm)
        likelihood_fn = jax.vmap(likelihood_fn, in_axes=(None, 0))
        likelihood = jtu.tree_map(likelihood_fn, state.deltas, grads)
        # lift per example likelihoods into log space and sum
        log_likelihood = jtu.tree_map(lambda l: jnp.sum(jnp.log(l), axis=0),
                                      likelihood)
        # compute update and unlift from log space
        log_deltas = jtu.tree_map(lambda d, ll: jnp.log(jnp.abs(d)) + ll,
                                  state.deltas, log_likelihood)
        # renormalize and clip
        log_max_norm = jnp.log(state.max_norm)
        log_current_norm = jax.nn.logsumexp(jnp.array([
            jax.nn.logsumexp(ll)
            for ll in jtu.tree_leaves(log_deltas)
        ]))
        log_max_delta = jnp.log(max_delta)
        deltas = jtu.tree_map(
            lambda ld, d: jnp.exp(jnp.clip(ld + log_max_norm - log_current_norm,
                                           a_max=log_max_delta)) * jnp.sign(d),
            log_deltas, state.deltas
        )

        return deltas, NTKEnsembleState(deltas, state.max_norm)

    return optax.GradientTransformation(init_fn, update_fn)
